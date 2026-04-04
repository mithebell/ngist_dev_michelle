import logging
import os
import time

import h5py
import numpy as np
from astropy.io import ascii, fits
from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.lineStrengths import lsindex_spec_updated as lsindex
from ngistPipeline.lineStrengths import ssppop_fitting as ssppop
from joblib import Parallel, delayed
from ppxf.ppxf_util import gaussian_filter1d
from printStatus import printStatus
from tqdm import tqdm

cvel = 299792.458

"""
PURPOSE:
  This module executes the measurement of line strength indices in the pipeline
  using an integral-based EW method (Michelle Ding, Jan 2026). It acts as an
  interface between the pipeline and the integral line strength measurement
  routine (lsindex_spec_updated), and their conversion to single stellar
  population equivalent population properties with the updated MCMC algorithm
  of Martin-Navaroo et al. 2018 (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M).
"""

def calculate_minimisation_diagnostics(ls_indices, names, model_indices, params, config, index_names):
    target_indices = ['Hbeta_o', 'Fe5270', 'Mgb']
    obs_indices_positions = []
    found_indices = []
    for target in target_indices:
        idx_pos = np.where(names == target)[0]
        if len(idx_pos) > 0:
            obs_indices_positions.append(idx_pos[0])
            found_indices.append(target)

    if len(found_indices) != 3:
        logging.warning(f"Not all minimisation indices found. Found: {found_indices}")
        printStatus.warning(f"Expected Hbeta_o, Fe5270, Mgb — found: {found_indices}")
        nbins = ls_indices.shape[0]
        return np.full((nbins, params.shape[1]), np.nan), np.full(nbins, np.nan), np.full(nbins, -1, dtype=int)

    obs_indices_positions = np.array(obs_indices_positions)

    model_column_order = []
    for target in target_indices:
        if target not in index_names:
            logging.warning(f"{target} not in index_names — cannot map to model column")
            nbins = ls_indices.shape[0]
            return np.full((nbins, params.shape[1]), np.nan), np.full(nbins, np.nan), np.full(nbins, -1, dtype=int)
        model_column_order.append(index_names.index(target))

    model_indices_3 = model_indices[:, model_column_order]

    nbins = ls_indices.shape[0]
    nmodels = model_indices_3.shape[0]
    ndim_params   = params.shape[1]
    min_params    = np.zeros((nbins, ndim_params))
    min_residuals = np.zeros(nbins)
    min_indices   = np.zeros(nbins, dtype=int)

    printStatus.running("Computing minimisation diagnostics (Hbeta_o, Fe5270, Mgb)")

    for i in range(nbins):
        obs = ls_indices[i, obs_indices_positions]
        if np.any(np.isnan(obs)):
            min_params[i, :] = np.nan
            min_residuals[i] = np.nan
            min_indices[i]   = -1
            continue

        hbeta_res = np.abs((model_indices_3[:, 0] - obs[0]) / np.ptp(model_indices_3[:, 0]))
        fe_res    = np.abs((model_indices_3[:, 1] - obs[1]) / np.ptp(model_indices_3[:, 1]))
        mgb_res   = np.abs((model_indices_3[:, 2] - obs[2]) / np.ptp(model_indices_3[:, 2]))
        quad_residual = np.sqrt(hbeta_res**2 + fe_res**2 + mgb_res**2)

        min_idx = np.argmin(quad_residual)
        min_params[i, :]  = params[min_idx, :]
        min_residuals[i]  = quad_residual[min_idx]
        min_indices[i]    = min_idx

    printStatus.updateDone("Computing minimisation diagnostics (Hbeta_o, Fe5270, Mgb)", progressbar=False)

    return min_params, min_residuals, min_indices

def run_ls(
    wave,
    spec,
    espec,
    redshift,
    config,
    lickfile,
    names,
    index_names,
    model_indices,
    params,
    tri,
    labels,
    nbins,
    i,
    MCMC,
):
    """
    Calls the integral-based line strength measurement routine
    (lsindex_spec_updated, Michelle Ding Jan 2026), and if required, the
    updated MCMC algorithm from Martin-Navaroo et al. 2018
    (ui.adsabs.harvard.edu/#abs/2018MNRAS.475.3700M) to determine SSP
    properties. Supports EW and MCMC corner plots, and uses a minimisation
    solution as the MCMC walker initialisation point.

    Args:
    wave (array): Wavelength data
    spec (array): Spectral data
    espec (array): Error spectral data
    redshift (array): Redshift data
    config (dict): Configuration data
    lickfile (str): Lick index file
    names (array): Index names
    index_names (array): Names of the indices in consideration
    model_indices (array): Model indices
    params (array): Parameters
    tri (array): Triangulation data
    labels (array): Labels data
    nbins (int): Number of bins
    i (int): Iteration number
    MCMC (bool): Flag for using MCMC algorithm

    Returns:
    tuple: (indices, errors, vals, percentiles, mc_chains) if MCMC, else (indices, errors, mc_chains)
    """
    nindex = len(index_names)

    try:
        plot_flag = 0
        plot_corner = False

        resolution = config["LS"].get("_CURRENT_RESOLUTION", "ORIGINAL")
        if resolution == "ADAPTED":
            plot_bins = config["LS"].get("PLOT", False)
            if plot_bins is True:
                plot_flag = 1
                plot_corner = True
            elif isinstance(plot_bins, int) and i < plot_bins:
                plot_flag = 1
                plot_corner = True
            elif isinstance(plot_bins, list) and i in plot_bins:
                plot_flag = 1
                plot_corner = True

        plot_dir = os.path.join(config["GENERAL"]["OUTPUT"], "Fig_LS")
        corner_dir = config["GENERAL"]["OUTPUT"]

        if plot_flag == 1:
            logging.info(f"Plotting enabled for bin {i}")
        names, indices, errors, mc_chains = lsindex.lsindex(
            wave,
            spec,
            espec,
            redshift[0],
            lickfile,
            sims=config["LS"]["MC_LS"],
            z_err=redshift[1],
            plot=plot_flag,
            plot_dir=plot_dir,
            bin_id=i,
            run_id=config["GENERAL"]["RUN_ID"],
        )

        data = np.zeros(nindex)
        error = np.zeros(nindex)
        for o in range(nindex):
            idx = np.where(names == index_names[o])[0]
            data[o] = indices[idx]
            error[o] = errors[idx]

        if MCMC == True:
            # Use minimisation solution (Hbeta_o, Fe5270, Mgb) as p0_centre if possible
            target_indices = ['Hbeta_o', 'Fe5270', 'Mgb']
            min_positions = [np.where(names == t)[0] for t in target_indices]

            if all(len(pos) > 0 for pos in min_positions):
                obs_3 = np.array([indices[pos[0]] for pos in min_positions])
                model_col_order = [index_names.index(t) for t in target_indices if t in index_names]

                if len(model_col_order) == len(target_indices) and not np.any(np.isnan(obs_3)):
                    model_3 = model_indices[:, model_col_order]
                    hbeta_res = np.abs((model_3[:, 0] - obs_3[0]) / np.ptp(model_3[:, 0]))
                    fe_res    = np.abs((model_3[:, 1] - obs_3[1]) / np.ptp(model_3[:, 1]))
                    mgb_res   = np.abs((model_3[:, 2] - obs_3[2]) / np.ptp(model_3[:, 2]))
                    quad_res  = np.sqrt(hbeta_res**2 + fe_res**2 + mgb_res**2)
                    p0_centre = params[np.argmin(quad_res), :]
                else:
                    p0_centre = None
            else:
                p0_centre = None

            vals, chains = ssppop.ssppop_fitting(
                data,
                error,
                model_indices,
                params,
                tri,
                labels,
                config["LS"]["NWALKER"],
                config["LS"]["NCHAIN"],
                plot_corner,
                0,
                i,
                nbins,
                corner_dir,
                p0_centre=p0_centre,
            )

            percentiles = np.percentile(chains, np.arange(101), axis=0)

            return (indices, errors, vals, percentiles, mc_chains)

        elif MCMC == False:
            return (indices, errors, mc_chains)

    except Exception as e:
        import traceback
        logging.warning(f"run_ls failed for bin {i}: {e}")
        logging.warning(traceback.format_exc())
        if MCMC == True:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        elif MCMC == False:
            return (np.nan, np.nan, np.nan)


def save_ls(
    names,
    ls_indices,
    ls_errors,
    index_names,
    labels,
    RESOLUTION,
    MCMC,
    totalFWHM_flag,
    config,
    mc_chains=None, 
    vals=None,
    percentile=None,
    min_params=None, 
    min_residuals=None, 
    min_indices=None,
):
    """Saves all results to disk."""
    # Save results
    if RESOLUTION == "ORIGINAL":
        outfits = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_ls_orig_res.fits"
        )
        printStatus.running(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_orig_res.fits"
        )
    if RESOLUTION == "ADAPTED":
        outfits = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_ls_adap_res.fits"
        )
        printStatus.running(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_adap_res.fits"
        )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with results
    cols = []
    if MCMC == True:
        nparam = len(labels)
        for i in range(nparam):
            cols.append(
                fits.Column(name=labels[i], format="D", array=percentile[:, 50, i])
            )
        cols.append(fits.Column(name="lnP", format="D", array=vals[:, -2]))
        cols.append(fits.Column(name="Flag", format="D", array=vals[:, -1]))

    if min_params is not None:
        param_col_names = ["MIN_AGE", "MIN_METAL", "MIN_ALPHA"]
        for k in range(min_params.shape[1]):
            cols.append(fits.Column(name=param_col_names[k], format="D", array=min_params[:, k]))
        cols.append(fits.Column(name="MIN_RESIDUAL",  format="D", array=min_residuals))

    ndim = len(names)
    for i in range(ndim):
        if not np.all(np.isnan(ls_indices[:, i])):
            cols.append(fits.Column(name=names[i], format="D", array=ls_indices[:, i]))
        if not np.all(np.isnan(ls_errors[:, i])):
            cols.append(
                fits.Column(name="ERR_" + names[i], format="D", array=ls_errors[:, i])
            )
    cols.append(fits.Column(name="FWHM_FLAG", format="I", array=totalFWHM_flag[:]))

    lsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    lsHDU.name = "LS_DATA"

    # Extension 2: Table HDU with percentiles
    percentilesHDU = None
    if MCMC == True:
        cols = []
        nparam = len(labels)
        for i in range(nparam):
            cols.append(
                fits.Column(
                    name=labels[i] + "_PERCENTILES",
                    format="101D",
                    array=percentile[:, :, i],
                )
            )
        percentilesHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        percentilesHDU.name = "PERCENTILES"

    # Extension 3: Table HDU with MC chains
    mcChainsHDU = None 
    if mc_chains is not None:
        cols_mc = []
        ndim = len(names)
        for i in range(ndim):
            if not np.all(np.isnan(mc_chains[:, i, :])):
                cols_mc.append(
                    fits.Column(
                        name=names[i] + "_MC",
                        format=str(mc_chains.shape[2]) + "D",
                        array=mc_chains[:, i, :]
                    )
                )
        if len(cols_mc) > 0:
            mcChainsHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_mc))
            mcChainsHDU.name = "MC_CHAINS"
    
    if MCMC == False:
        if mcChainsHDU is not None:
            HDUList = fits.HDUList([priHDU, lsHDU, mcChainsHDU])
        else:
            HDUList = fits.HDUList([priHDU, lsHDU])
    elif MCMC == True:
        if mcChainsHDU is not None:
            HDUList = fits.HDUList([priHDU, lsHDU, percentilesHDU, mcChainsHDU])
        else:
            HDUList = fits.HDUList([priHDU, lsHDU, percentilesHDU])
    
    HDUList.writeto(outfits, overwrite=True)

    if RESOLUTION == "ORIGINAL":
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_orig_res.fits"
        )
    if RESOLUTION == "ADAPTED":
        printStatus.updateDone(
            "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_adap_res.fits"
        )
    logging.info("Wrote: " + outfits)



def saveCleanedLinearSpectra(spec, espec, wave, npix, config):
    """Save emission-subtracted, linearly binned spectra to disk."""
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_ls_cleaned_linear.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_cleaned_linear.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with cleaned, linear spectra
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=spec))
    cols.append(fits.Column(name="ESPEC", format=str(npix) + "D", array=espec))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "CLEANED_SPECTRA"

    # Extension 2: Table HDU with wave
    cols = []
    cols.append(fits.Column(name="LAM", format="D", array=wave))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LAM"

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_cleaned_linear.fits"
    )
    logging.info("Wrote: " + outfits)


def saveConvolvedLinearSpectra(spec, espec, wave, npix, config):
    """Save emission-subtracted, linearly binned, convolved spectra to disk."""
    outfits = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_ls_convolved_linear.fits"
    )
    printStatus.running(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_convolved_linear.fits"
    )

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with cleaned, linear spectra
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=spec))
    cols.append(fits.Column(name="ESPEC", format=str(npix) + "D", array=espec))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "CONVOLVED_SPECTRA"

    # Extension 2: Table HDU with wave
    cols = []
    cols.append(fits.Column(name="LAM", format="D", array=wave))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LAM"

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_ls_convolved_linear.fits"
    )
    logging.info("Wrote: " + outfits)


def log_unbinning(lamRange, spec, oversample=1, flux=True):
    """
    This function transforms logarithmically binned spectra back to linear
    binning. It is a Python translation of Michele Cappellari's
    "log_rebin_invert" function. Thanks to Michele Cappellari for his permission
    to include this function in the pipeline.
    """
    # Length of arrays
    n = len(spec)
    m = n * oversample

    # Log space
    dLam = (lamRange[1] - lamRange[0]) / (n - 1)  # Step in log-space
    lim = lamRange + np.array([-0.5, 0.5]) * dLam  # Min and max wavelength in log-space
    borders = np.linspace(lim[0], lim[1], n + 1)  # OLD logLam in log-space

    # Wavelength domain
    logLim = np.exp(lim)  # Min and max wavelength in Angst.
    lamNew = np.linspace(logLim[0], logLim[1], m + 1)  # new logLam in Angstroem
    newBorders = np.log(lamNew)  # new logLam in log-space

    # Translate indices of arrays so that newBorders[j] corresponds to borders[k[j]]
    k = np.floor((newBorders - lim[0]) / dLam).astype("int")

    # Construct new spectrum
    specNew = np.zeros(m)
    for j in range(0, m - 1):
        a = (newBorders[j] - borders[k[j]]) / dLam
        b = (borders[k[j + 1]] - newBorders[j + 1]) / dLam

        specNew[j] = np.sum(spec[k[j] : k[j + 1]]) - a * spec[k[j]] - b * spec[k[j + 1]]


    # Rescale flux
    if flux == True:
        specNew = (
            specNew
            / (newBorders[1:] - newBorders[:-1])
            * np.mean(newBorders[1:] - newBorders[:-1])
            * oversample
        )

    # Shift back the wavelength arrays
    lamNew = lamNew[:-1] + 0.5 * (lamNew[1] - lamNew[0])

    return (specNew, lamNew)


def measureLineStrengths(config, RESOLUTION="ORIGINAL"):
    """
    Starts the integral-based line strength analysis. Data is read in,
    emission-subtracted spectra are rebinned from logarithmic to linear scale,
    and the spectra convolved to meet the LIS measurement resolution, which is
    saved to disk. Line strength indices are measured using the integral EW
    method. If required, SSP properties are estimated via MCMC with minimisation-
    based walker initialisation. Supports DEBUG_BIN mode and saves MC chains
    to the output FITS file.

    Args:
        config (dict): Configuration parameters for the line strength analysis.
        RESOLUTION (str, optional): Resolution type. Defaults to "ORIGINAL".

    Returns:
        None
    """
    # Run MCMC only on the indices measured from convoluted spectra
    if config["LS"]["TYPE"] == "SPP" and RESOLUTION == "ADAPTED":
        MCMC = True
    else:
        MCMC = False

    config["LS"]["_CURRENT_RESOLUTION"] = RESOLUTION

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "LS")

    # Read the log-rebinned spectra and log-unbin them
    if (
        os.path.isfile(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_ls_cleaned_linear.fits"
        )
        == False
    ) or (config["GENERAL"]["OW_OUTPUT"] == True):
        # Read spectra
        if (
            os.path.isfile(
                os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                + "_gas_cleaned_bin.fits"
            )
            == True
        ):
            logging.info(
                "Using emission-subtracted spectra at "
                + os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                + "_gas_cleaned_bin.fits"
            )
            printStatus.done("Using emission-subtracted spectra")
            hdu_spec = fits.open(
                os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                + "_gas_cleaned_bin.fits"
            )
            binned_spec_data = hdu_spec[1].data["SPEC"]
            binned_loglam_data = hdu_spec[2].data["LOGLAM"]
        else:
            logging.info(
                "Using regular spectra without any emission-correction at "
                + os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                + "_bin_spectra.hdf5"
            )
            printStatus.done("Using regular spectra without any emission-correction")
            with h5py.File(
                os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                + "_bin_spectra.hdf5",
                "r",
            ) as f:
                binned_spec_data = f["SPEC"][:].T
                binned_loglam_data = f["LOGLAM"][:]
        with h5py.File(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_bin_spectra.hdf5",
            "r",
        ) as errorf:
            binned_espec_data = errorf["ESPEC"][:].T
            binned_eloglam_data = errorf["LOGLAM"][:].T
                
        idx_lamMin = np.where(binned_loglam_data[0] == binned_eloglam_data)[0]
        idx_lamMax = np.where(binned_loglam_data[-1] == binned_eloglam_data)[0]
        idx_lam = np.arange(idx_lamMin, idx_lamMax + 1)
        oldspec = np.array(binned_spec_data)
        oldespec = np.sqrt(np.array(binned_espec_data)[:, idx_lam]) * 0.001
        wave = np.array(binned_loglam_data)

        nbins = oldspec.shape[0]
        npix = oldspec.shape[1]
        lamRange = np.array([wave[0], wave[-1]])
        spec = np.zeros(oldspec.shape)
        espec = np.zeros(oldespec.shape)

        # Rebin the cleaned spectra from log to lin
        printStatus.running("Rebinning the spectra from log to lin")
        for i in range(nbins):
            spec[i, :], wave = log_unbinning(lamRange, oldspec[i, :])
        printStatus.updateDone(
            "Rebinning the spectra from log to lin", progressbar=False
        )

        # Rebin the error spectra from log to lin
        printStatus.running("Rebinning the error spectra from log to lin")
        for i in range(nbins):
            espec[i, :], _ = log_unbinning(lamRange, oldespec[i, :])
        printStatus.updateDone(
            "Rebinning the error spectra from log to lin", progressbar=False
        )

        # Save cleaned, linear spectra
        saveCleanedLinearSpectra(spec, espec, wave, npix, config)

    # Read the linearly-binned, cleaned spectra provided by previous LS-run
    else:
        logging.info(
            "Reading "
            + os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_ls_cleaned_linear.fits"
        )
        hdu = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_ls_cleaned_linear.fits"
        )
        spec = np.array(hdu[1].data.SPEC)
        espec = np.array(hdu[1].data.ESPEC)
        wave = np.array(hdu[2].data.LAM)
        nbins = spec.shape[0]
        npix = spec.shape[1]

    # Read PPXF results
    ppxf_data = fits.open(
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_kin.fits", 
        mem_map=True
    )[1].data
    redshift = np.zeros((nbins, 2))
    redshift[:, 0] = np.array(ppxf_data.V[:]) / cvel
    redshift[:, 1] = np.array(ppxf_data.FORM_ERR_V[:]) / cvel
    veldisp_kin = np.array(ppxf_data.SIGMA[:])

    # Read file defining the LS bands
    lickfile = os.path.join(config["GENERAL"]["CONFIG_DIR"], config["LS"]["LS_FILE"])
    tab = ascii.read(lickfile, comment="\s*#")
    names = tab["names"]

    # Flag spectra for which the total intrinsic dispersion is larger than the LIS measurement resolution
    totalFWHM_flag = np.zeros(nbins)

    # Broaden spectra to LIS resolution taking into account the measured velocity dispersion
    if RESOLUTION == "ADAPTED":
        printStatus.running("Broadening the spectra to LIS resolution")
        with h5py.File(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) 
            + "_bin_spectra.hdf5", 
            'r',
        ) as f:
            velscale = f.attrs["VELSCALE"]
        for i in range(0, nbins):
            veldisp_kin_Angst = veldisp_kin[i] * wave / cvel * 2.355
            total_dispersion = np.sqrt(LSF_Data(wave) ** 2 + veldisp_kin_Angst**2)
            FWHM_dif = np.sqrt(config["LS"]["CONV_COR"] ** 2 - total_dispersion**2)
            sigma = (FWHM_dif / wave) * cvel / 2.355 / velscale
            idx = np.where(np.isnan(sigma) == True)[0]
            if len(idx) > 0:
                sigma[idx] = 0.0
                totalFWHM_flag[i] = 1
            spec[i, :] = gaussian_filter1d(spec[i, :], sigma)
            espec[i, :] = gaussian_filter1d(espec[i, :], sigma)
        printStatus.updateDone(
            "Broadening the spectra to LIS resolution", progressbar=False
        )
        saveConvolvedLinearSpectra(spec, espec, wave, npix, config)

    # Get indices that are considered in SSP-conversion
    idx = np.where(tab["spp"] == 1)[0]
    index_names = tab["names"][idx].tolist()

    # Loading model predictions
    if MCMC == True:
        modelfile = os.path.join(
            config["GENERAL"]["TEMPLATE_DIR"], config["LS"]["SPP_FILE"]
        )
        model_indices, params, tri, labels = ssppop.load_models(modelfile, index_names)
        logging.info("Loading LS model file at " + modelfile)
    elif MCMC == False:
        model_indices, params, tri, labels = "dummy", "dummy", "dummy", "dummy"

    # Arrays to store results
    ls_indices = np.zeros((nbins, len(names)))
    ls_errors = np.zeros((nbins, len(names)))
    mc_chains_all = np.zeros((nbins, len(names), config["LS"]["MC_LS"]))
    vals = None
    percentile = None
    if MCMC == True:
        vals = np.zeros((nbins, len(labels) * 3 + 2))
        percentile = np.zeros((nbins, 101, len(labels)))

    # Run LS Measurements
    start_time = time.time()
    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running lineStrengths in parallel mode")
        logging.info("Running lineStrengths in parallel mode")

        def worker(chunk):
            results = []
            for i in chunk:
                result = run_ls(
                    wave,
                    spec[i, :],
                    espec[i, :],
                    redshift[i, :],
                    config,
                    lickfile,
                    names,
                    index_names,
                    model_indices,
                    params,
                    tri,
                    labels,
                    nbins,
                    i,
                    MCMC,
                )
                results.append(result)
            return results

        memmap_folder = (
            "/scratch" 
            if os.access("/scratch", os.W_OK) 
            else config["GENERAL"]["OUTPUT"]
        )
        max_nbytes = None
        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"] * 10))
        chunks = [
            range(i, min(i + chunk_size, nbins)) for i in range(0, nbins, chunk_size)
        ]
        parallel_configs = {
            "n_jobs": config["GENERAL"]["NCPU"],
            "max_nbytes": max_nbytes,
            "temp_folder": memmap_folder,
            "mmap_mode": "c",
            "return_as": "generator",
        }
        ppxf_tmp = list(
            tqdm(
                Parallel(**parallel_configs)(
                    delayed(worker)(chunk) for chunk in chunks
                ),
                total=len(chunks),
                desc="Processing chunks",
                ascii=" #",
                unit="chunk",
            )
        )
        ppxf_tmp = [result for chunk_results in ppxf_tmp for result in chunk_results]

        for i in range(0, nbins):
            if MCMC == True:
                ls_indices[i, :], ls_errors[i, :], vals[i, :], percentile[i, :, :], mc_chains_all[i, :, :] = ppxf_tmp[i]
            else:
                ls_indices[i, :], ls_errors[i, :], mc_chains_all[i, :, :] = ppxf_tmp[i]

        printStatus.updateDone(
            "Running lineStrengths in parallel mode", progressbar=False
        )

    if config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running lineStrengths in serial mode")
        logging.info("Running lineStrengths in serial mode")

        if 'DEBUG_BIN' in config["LS"] and config["LS"]["DEBUG_BIN"] is not False:
            runbin = config["LS"]["DEBUG_BIN"]
            printStatus.running("Running lineStrengths in debug mode on bins: " + str(runbin))
            logging.info("Running lineStrengths in debug mode on bins: " + str(runbin))
        else:
            runbin = np.arange(0, nbins)

        if MCMC == True:
            for i in runbin:
                (
                    ls_indices[i, :],
                    ls_errors[i, :],
                    vals[i, :],
                    percentile[i, :, :],
                    mc_chains_all[i, :, :],
                ) = run_ls(
                    wave,
                    spec[i, :],
                    espec[i, :],
                    redshift[i, :],
                    config,
                    lickfile,
                    names,
                    index_names,
                    model_indices,
                    params,
                    tri,
                    labels,
                    nbins,
                    i,
                    MCMC,
                )
        elif MCMC == False:
            for i in runbin:
                ls_indices[i, :], ls_errors[i, :], mc_chains_all[i, :, :] = run_ls(
                    wave,
                    spec[i, :],
                    espec[i, :],
                    redshift[i, :],
                    config,
                    lickfile,
                    names,
                    index_names,
                    model_indices,
                    params,
                    tri,
                    labels,
                    nbins,
                    i,
                    MCMC,
                )

        printStatus.updateDone(
            "Running lineStrengths in serial mode", progressbar=False
        )

    print(
        "             Running lineStrengths on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )
    logging.info(
        "Running lineStrengths on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check for exceptions which occurred during the analysis
    idx_error = np.where(np.all(np.isnan(ls_indices[:, :]), axis=1) == True)[0]
    if len(idx_error) != 0:
        printStatus.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
        )
        print("             " + str(idx_error))
        logging.warning(
            "There was a problem in the analysis of the spectra with the following BINID's: "
            + str(idx_error)
        )
    else:
        print("             " + "There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    min_params, min_residuals = None, None
    if config["LS"]["TYPE"] == "SPP":
        if isinstance(model_indices, str) and model_indices == "dummy":
            modelfile = os.path.join(
                config["GENERAL"]["TEMPLATE_DIR"], config["LS"]["SPP_FILE"]
            )
            model_indices, params, tri, labels = ssppop.load_models(
                modelfile, index_names
            )
            logging.info(
                "Loading LIS templates for minimisation diagnostics: " + modelfile
            )

        min_params, min_residuals, _ = calculate_minimisation_diagnostics(ls_indices, names, model_indices, params, config, index_names)

    save_ls(
        names,
        ls_indices,
        ls_errors,
        index_names,
        labels,
        RESOLUTION,
        MCMC,
        totalFWHM_flag,
        config,
        mc_chains=mc_chains_all,
        vals=vals if MCMC else None,
        percentile=percentile if MCMC else None,
        min_params=min_params,
        min_residuals=min_residuals,
    )

    # Repeat analysis with adapted spectral resolution
    if RESOLUTION == "ORIGINAL":
        measureLineStrengths(config, RESOLUTION="ADAPTED")
        # Convert DEBUG_BIN to string for FITS header after both passes complete
        if 'DEBUG_BIN' in config["LS"] and config["LS"]["DEBUG_BIN"] is not False:
            config["LS"]["DEBUG_BIN"] = str(config["LS"]["DEBUG_BIN"])

    return None

    # Return
    return None