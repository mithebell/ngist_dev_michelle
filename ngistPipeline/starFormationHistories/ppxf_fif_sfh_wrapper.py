import logging
import os
import time

import h5py
import numpy as np
import emcee

from astropy.io import ascii, fits
from joblib import Parallel, delayed, dump, load
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import gaussian_filter1d
from printStatus import printStatus
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.prepareTemplates import _prepareTemplates

import warnings
warnings.filterwarnings("ignore")

C = 299792.458  # speed of light in km/s
LAM_PAD = 100.0  # padding (Ang) when cropping templates to the Mgb window


"""
PURPOSE:
  This module extracts [alpha/Fe] by Full-Index Fitting (FIF) of the Mgb
  feature, following Martin-Navarro et al. 2019. It acts as an interface
  between the pipeline and pPXF (Cappellari & Emsellem 2004) for the
  age/metallicity step, and the EMCEE sampler (Foreman-Mackey et al. 2013)
  for the FIF step. It is a METHOD for the SFH module and requires the
  following additional MasterConfig keys:

      SFH.ALPHA_FIX    -- [alpha/Fe] value at which the template grid is
                          fixed for the full-range age/metallicity fit.
                          Snapped to the nearest available grid value.
      SFH.FIF_NWALKERS -- number of EMCEE walkers (default 32).
      SFH.FIF_NCHAIN   -- number of EMCEE iterations (default 500).

  Algorithm per bin:
    Steps 0-3: pPXF at fixed alpha/Fe (ALPHA_FIX) for age and metallicity,
               following the same EBV prefit, noise rescaling, and 3-sigma
               clipping sequence as ppxf_ebvmpolyfix_sfh_wrapper.
    Step 4:    Identify non-zero weight (age, metallicity) grid points.
    Step 5:    Rebuild template set at those (age, met) points for all alpha.
    Step 6:    Convolve to the lowest velocity dispersion in the dataset
               (sigma_min), following Martin-Navarro et al. 2019.
    Step 7:    Normalise the Mgb window (b1-b6) by a straight-line
               pseudo-continuum through the b1/b2 and b5/b6 sidebands.
               Extract central bandpass pixels (b3-b4).
    Step 8:    For each alpha grid value, build a weighted model FIF vector
               (age/met weights from step 0-3). Run 1-D EMCEE to recover
               the [alpha/Fe] posterior from the normalised pixels.
"""


def plot_ppxf_sfh(pp, x, i, outfig_ppxf, snrCubevar=-99, snrResid=-99,
                   goodpixelsPre=[], norm=False, mean_results=''):
    # routine to plot first and final pPXF fit
    fig = plt.figure(i, figsize=(13, 3.0))
    ax2 = plt.subplot(111)

    if norm == True:
        median_norm = np.nanmedian(pp.galaxy[pp.goodpixels])
    else:
        median_norm = 1

    stars_bestfit = pp.bestfit
    bestfit_shown = pp.bestfit
    galaxy = pp.galaxy
    resid = galaxy - stars_bestfit
    goodpixels = pp.goodpixels

    ll, rr = np.min(x), np.max(x)

    sig3 = np.percentile(abs(resid[goodpixels]), 99.73)
    bestfit_shown = bestfit_shown[goodpixels[0]: goodpixels[-1] + 1]
    mx = 2.49
    mn = -0.49
    plt.plot(x, galaxy, 'black', linewidth=0.5)
    plt.plot(x[goodpixels], resid[goodpixels], 'd',
              color='LimeGreen', mec='LimeGreen', ms=1)

    if len(goodpixelsPre) > 0:
        w = np.flatnonzero(np.diff(goodpixels) > 1)
        for wj in w:
            a, b = goodpixels[wj: wj + 2]
            plt.axvspan(x[a], x[b], facecolor='lightpink')
            plt.plot(x[a: b + 1], resid[a: b + 1], 'green', linewidth=0.5, alpha=0.5)
        for k in goodpixels[[0, -1]]:
            plt.plot(x[[k, k]], [mn, stars_bestfit[k]], 'lightpink', linewidth=0.5)

        # repeat square lines with pp_step1
        w = np.flatnonzero(np.diff(goodpixelsPre) > 1)
        for wj in w:
            a, b = goodpixelsPre[wj: wj + 2]
            plt.axvspan(x[a], x[b], facecolor='lightgray')
        for k in goodpixelsPre[[0, -1]]:
            plt.plot(x[[k, k]], [mn, stars_bestfit[k]], 'lightgray', linewidth=0.5)
    else:
        w = np.flatnonzero(np.diff(goodpixels) > 1)
        for wj in w:
            a, b = goodpixels[wj: wj + 2]
            plt.axvspan(x[a], x[b], facecolor='lightgray')
            plt.plot(x[a: b + 1], resid[a: b + 1], 'green', linewidth=0.5, alpha=0.5)
        for k in goodpixels[[0, -1]]:
            plt.plot(x[[k, k]], [mn, stars_bestfit[k]], 'lightgray', linewidth=0.5)

    plt.plot(x[goodpixels], goodpixels * 0, '.k', ms=1)
    plt.plot(x, stars_bestfit, 'red', linewidth=0.5)
    ax2.set(xlabel='wavelength [Ang]', ylabel='Flux [normalised]')
    ax2.set(ylim=(mn, mx))
    ax2.tick_params(direction='in', which='both')
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    nmom = np.max(pp.moments)

    if nmom == 2:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}") + \
                   (f", S/N Residual = {snrResid:.1f}")
    if nmom == 4:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}") + \
                   (f", S/N Residual = {snrResid:.1f}")
    if nmom == 6:
        plotText = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}, ") + \
                   (f"h5 = {pp.sol[4]:.3f}, h6 = {pp.sol[5]:.3f}") + \
                   (f", S/N Residual = {snrResid:.1f}")

    if len(mean_results) > 0:
        plotText += f", Age [Gyr] = {mean_results[0][0]:.2f}, [M/H] = {mean_results[0][1]:.2f}"
        if mean_results.shape[1] > 2:
            plotText += f", [alpha/Fe] = {mean_results[0][2]:.2f}"

    plt.text(0.01, 0.95, plotText, fontsize=10, ha='left', va='top',
              transform=ax2.transAxes, backgroundcolor='white')
    plt.savefig(outfig_ppxf, bbox_inches='tight', pad_inches=0.3)
    plt.close()


def plot_fif(wave_b1b6, gal_norm_b1b6, wave_fif, model_fif_best,
              b1, b2, b3, b4, b5, b6,
              alpha_fif, alpha_fif_lo, alpha_fif_hi,
              age, met, i, outfig):
    # FIF Mgb plot in pseudo-continuum normalised space: sidebands sit at ~1.0
    # by construction, continuum is a flat line at 1.0, model plots directly.
    # Mirrors the LS plot format (Martin-Navarro et al. 2019, Fig. 1).
    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(wave_b1b6, gal_norm_b1b6, 'black', linewidth=0.8)

    # pseudo-continuum is 1.0 by definition after normalisation
    ax.plot([b1, b6], [1.0, 1.0], color='red', linewidth=1.5, label='Continuum')

    # best-fit model (already normalised)
    ax.plot(wave_fif, model_fif_best, color='red', linewidth=1.5, label='Model')

    # band shading
    ax.axvspan(b1, b2, color='skyblue', alpha=0.5, label='Pseudo-continua')
    ax.axvspan(b3, b4, color='grey',    alpha=0.2, label='Central bandpass')
    ax.axvspan(b5, b6, color='skyblue', alpha=0.5)

    ax.set(xlabel='wavelength [Ang]', ylabel='Normalised flux')
    ax.tick_params(direction='in', which='both')
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    plotText = (f"nGIST - Bin {i:10.0f}: Age = {age:.2f} Gyr, [M/H] = {met:.2f}, "
                f"[alpha/Fe] = {alpha_fif:.3f} ({alpha_fif_lo:+.3f} / {alpha_fif_hi:+.3f})")
    ax.text(0.01, 0.95, plotText, fontsize=10, ha='left', va='top',
             transform=ax.transAxes, backgroundcolor='white')

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=9)

    plt.savefig(outfig, bbox_inches='tight', pad_inches=0.3)
    plt.close()


def clip_outliers(galaxy, bestfit, mask):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma in relative
    error until the bad bins don't change any more. This function uses eq.(34)
    of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    while True:
        scale = galaxy[mask] @ bestfit[mask] / np.sum(bestfit[mask]**2)
        resid = scale * bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < 3 * err
        if np.array_equal(mask, ok_old):
            break
    return mask


def robust_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417
    """
    np.seterr(all='ignore')
    y = np.ravel(y)
    d = y if zero else y - np.median(y)
    mad = np.median(np.abs(d))
    u2 = (d / (9.0 * mad))**2
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good] * u1**2)**2).sum()
    den = (u1 * (1.0 - 5.0 * u2[good])).sum()
    return np.sqrt(num / (den * (den - 1.0)))


def find_nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def resolution_sigma_pix(wave, native_fwhm, target_fwhm, velscale):
    # per-pixel Gaussian sigma (pixels, log-lambda grid) to broaden from
    # native_fwhm to target_fwhm (both in Ang; may be arrays).
    # Returns (sigma_pix, exceeded) where exceeded flags native >= target.
    diff_sq = target_fwhm**2 - native_fwhm**2
    exceeded = diff_sq < 0
    diff_sq = np.clip(diff_sq, 0.0, None)
    sigma_pix = (np.sqrt(diff_sq) / wave) * C / 2.355 / velscale
    sigma_pix[exceeded] = 0.0
    return sigma_pix, exceeded


# FIF pseudo-continuum helpers (following Martin-Navarro et al. 2019 and
# the integral EW method of lsindex_spec_updated, Michelle Ding Jan 2026)

def flux_density_integral(wave, flux, wave1, wave2):
    # mean flux density over [wave1, wave2] via trapz with interpolated boundaries
    mask = (wave >= wave1 - 10) & (wave <= wave2 + 10)
    if not np.any(mask):
        return 0.0
    w = wave[mask]
    f = flux[mask]
    f1 = np.interp(wave1, w, f)
    f2 = np.interp(wave2, w, f)
    w_int = np.concatenate([[wave1], w[(w > wave1) & (w < wave2)], [wave2]])
    f_int = np.concatenate([[f1],   f[(w > wave1) & (w < wave2)], [f2]])
    return np.trapz(f_int, w_int) / (wave2 - wave1)


def normalize_pseudocont(wave, flux, b1, b2, b5, b6, noise=None):
    # divide spectrum by a straight-line continuum through the b1/b2 and b5/b6
    # pseudo-continuum sidebands, as in the Lick/FIF convention
    cont_blue = flux_density_integral(wave, flux, b1, b2)
    cont_red  = flux_density_integral(wave, flux, b5, b6)
    cont_blue_mid = 0.5 * (b1 + b2)
    cont_red_mid  = 0.5 * (b5 + b6)
    slope = (cont_red - cont_blue) / (cont_red_mid - cont_blue_mid)
    continuum = cont_blue + slope * (wave - cont_blue_mid)
    norm_flux = flux / continuum
    if noise is not None:
        return norm_flux, noise / continuum, continuum
    return norm_flux, continuum


def get_mgb_band(config):
    # read Mgb bandpass (b1..b6, Ang) from the LS line-list file
    lickfile = os.path.join(config["GENERAL"]["CONFIG_DIR"], config["SFH"]["LS_FILE"])
    tab = ascii.read(lickfile, comment=r"\s*#")
    idx = np.where(np.array(tab["names"]) == "Mgb")[0]
    if len(idx) == 0:
        raise KeyError(f"'Mgb' not found in {lickfile}")
    row = tab[idx[0]]
    return (float(row["b1"]), float(row["b2"]), float(row["b3"]),
            float(row["b4"]), float(row["b5"]), float(row["b6"]))


def run_ppxf_firsttime(templates, log_bin_data, log_bin_error, velscale, start,
                        goodPixels, nmoments, offset, degree, mdeg, regul,
                        velscale_ratio, ncomb):
    printStatus.running("Running pPXF for the first time")
    median_log_bin_data = np.nanmedian(log_bin_data)
    log_bin_error = log_bin_error / median_log_bin_data
    log_bin_data = log_bin_data / median_log_bin_data
    pp = ppxf(templates, log_bin_data, log_bin_error, velscale, start,
               goodpixels=goodPixels, plot=False, quiet=True, moments=nmoments,
               degree=-1, vsyst=offset, mdegree=mdeg, regul=regul,
               velscale_ratio=velscale_ratio)
    reshaped_templates = templates.reshape((templates.shape[0], ncomb))
    normalized_weights = pp.weights / np.sum(pp.weights)
    nonzero_weights = np.shape(np.where(normalized_weights > 0)[0])[0]
    optimal_template = np.zeros((templates.shape[0], 1))
    optimal_template_set = np.zeros([templates.shape[0], nonzero_weights])
    printStatus.running('Number of Templates with non-zero weights ' + str(nonzero_weights))
    count_nonzero = 0
    for j in range(0, reshaped_templates.shape[1]):
        optimal_template[:, 0] = optimal_template[:, 0] + reshaped_templates[:, j] * normalized_weights[j]
        if normalized_weights[j] > 0:
            optimal_template_set[:, count_nonzero] = reshaped_templates[:, j]
            count_nonzero += 1
    return optimal_template, optimal_template_set


def run_fif_emcee_1d(data, error, model_fif, alpha_values, alpha_fix, nwalkers, nchain):
    """
    1-D EMCEE fit for [alpha/Fe] from FIF pixel data.
    model_fif has shape (nAlpha, n_pix); interpolation is linear via np.interp.
    Returns (median_alpha, err_lo, err_hi) where err_lo/hi are 16th/84th percentile
    offsets from the median (same sign convention as ssppop_fitting).
    """
    alpha_min = alpha_values[0]
    alpha_max = alpha_values[-1]

    def lnprob(par):
        alpha = par[0]
        if not (alpha_min <= alpha <= alpha_max):
            return -np.inf
        # interpolate model FIF vector at this alpha (vectorised over pixels)
        idx = np.searchsorted(alpha_values, alpha)
        idx = np.clip(idx, 1, len(alpha_values) - 1)
        t = (alpha - alpha_values[idx - 1]) / (alpha_values[idx] - alpha_values[idx - 1])
        model_at_alpha = (1.0 - t) * model_fif[idx - 1, :] + t * model_fif[idx, :]
        good = (error > 0) & np.isfinite(error) & np.isfinite(data)
        if np.sum(good) == 0:
            return -np.inf
        inv_sigma2 = 1.0 / error[good]**2
        lnlike = -0.5 * np.sum((data[good] - model_at_alpha[good])**2 * inv_sigma2
                                - np.log(inv_sigma2))
        return lnlike if np.isfinite(lnlike) else -np.inf

    # initialise walkers in a small ball around alpha_fix
    p0 = [[alpha_fix + 0.02 * np.random.randn()] for _ in range(nwalkers)]
    p0 = [[np.clip(p[0], alpha_min, alpha_max)] for p in p0]

    sampler = emcee.EnsembleSampler(nwalkers, 1, lnprob)
    sampler.run_mcmc(p0, nchain, progress=False)

    try:
        tau    = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin   = max(1, int(0.5 * np.min(tau)))
    except emcee.autocorr.AutocorrError:
        burnin = int(0.3 * nchain)
        thin   = 1

    if burnin >= nchain:
        burnin = int(0.3 * nchain)

    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)[:, 0]

    median   = np.percentile(flat, 50)
    err_lo   = np.percentile(flat, 16) - median
    err_hi   = np.percentile(flat, 84) - median

    return median, err_lo, err_hi


def run_ppxf(
    templates_alpha,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels_step0,
    goodPixels,
    nmoments,
    offset,
    degree,
    mdeg,
    regul,
    doclean,
    fixed,
    velscale_ratio,
    npix,
    ncomb_alpha,
    ncomb,
    nAges,
    nMetal,
    nAlpha,
    nbins,
    i,
    optimal_template_in,
    EBV_init,
    logLam,
    logAge_grid,
    metal_grid,
    alpha_grid,
    config,
    doplot,
    templates_mgb_lib,
    wave_temp_mgb,
    idx_gal_mgb,
    idx_gal_b3b4,
    lsf_data_full,
    sigma_max,           # maximum velocity dispersion across all bins (km/s)
    mgb_bands,
    nwalkers_fif,
    nchain_fif,
    alpha_values,
):
    """
    Calls pPXF for the age/metallicity fit at fixed alpha (steps 0-3),
    then runs FIF EMCEE on the Mgb feature for [alpha/Fe] (steps 4-8).
    See module docstring for the full algorithm.
    """

    b1, b2, b3, b4, b5, b6 = mgb_bands

    try:
        if len(optimal_template_in) > 1:

            # Normalise galaxy spectra and noise
            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error = log_bin_error / median_log_bin_data
            log_bin_data = log_bin_data / median_log_bin_data

            # Calculate SNR before the fit from flux and flux_err
            snr_prefit = np.nanmedian(log_bin_data / log_bin_error)

            # Step 0: EBV prefit -- no polynomials, dust only
            component_step0 = [0] * np.prod(optimal_template_in.shape[1:])
            component_true_step0 = np.array(component_step0) == 0
            dust = [{"start": [EBV_init], "bounds": [[0, 8]], "component": component_true_step0}]

            pp_step0 = ppxf(optimal_template_in, log_bin_data, log_bin_error, velscale,
                             lam=np.exp(logLam), goodpixels=goodPixels_step0, degree=-1,
                             mdegree=-1, vsyst=offset, velscale_ratio=velscale_ratio,
                             moments=nmoments, start=start, plot=False, dust=dust,
                             component=component_step0, regul=0, quiet=True)

            # check which optimal template method is preferred
            if config["SFH"]["OPT_TEMP"] == "default":
                reshaped_templates = templates_alpha.reshape((templates_alpha.shape[0], ncomb_alpha))
                normalized_weights_step0 = pp_step0.weights / np.sum(pp_step0.weights)
                wNonzero_weights_step0 = np.where(normalized_weights_step0 > 0)[0]
                nNonzero_weights_step0 = np.shape(wNonzero_weights_step0)[0]
                optimal_template_set_step0 = np.zeros([reshaped_templates.shape[0], nNonzero_weights_step0])
                for j in range(0, nNonzero_weights_step0):
                    optimal_template_set_step0[:, j] = reshaped_templates[:, wNonzero_weights_step0[j]]
                optimal_template_in = optimal_template_set_step0

            # Save dust values
            Rv = 4.05
            Av = pp_step0.dust[0]["sol"][0]
            EBV = Av / Rv

            component_step12 = [0] * (np.shape(optimal_template_in)[1])
            component_true_step12 = np.array(component_step12) == 0
            component_step6 = [0] * ncomb_alpha
            component_true_step6 = np.array(component_step6) == 0

            if config["SFH"]["DUST_CORR"] == True:
                dust_step12 = [{"start": [Av], "bounds": [[0, 8]],
                                 "component": component_true_step12, "fixed": [True]}]
                dust_step6  = [{"start": [Av], "bounds": [[0, 8]],
                                 "component": component_true_step6, "fixed": [True]}]
            else:
                dust_step12 = None
                dust_step6  = None

            # Step 1: noise rescaling -- use fake noise for first iteration
            fake_noise = np.full_like(log_bin_data, 1.0)
            pp_step1 = ppxf(optimal_template_in, log_bin_data, fake_noise, velscale, start,
                             goodpixels=goodPixels_step0, plot=False, quiet=True,
                             moments=nmoments, degree=-1, vsyst=offset, mdegree=mdeg,
                             fixed=fixed, lam=np.exp(logLam), velscale_ratio=velscale_ratio,
                             component=component_step12, dust=dust_step12)
            goodPixels_preclip = goodPixels
            noise_orig = np.mean(log_bin_error[goodPixels_step0])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels_step0] - pp_step1.bestfit[goodPixels_step0])
            snr_Resid1 = np.nanmedian(pp_step1.galaxy[goodPixels_step0] / noise_est)
            noise_new = log_bin_error * (noise_est / noise_orig)
            noise_new_std = robust_sigma(noise_new)
            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            # Step 2: 3-sigma clip
            mask0 = logLam > 0
            mask0[:] = False
            mask0[goodPixels] = True
            mask = mask0.copy()
            if doclean == True:
                mask = clip_outliers(log_bin_data, pp_step1.bestfit, mask)
                mask &= mask0

            # Step 3: science fit for age and metallicity at fixed alpha
            pp = ppxf(templates_alpha, log_bin_data, noise_new, velscale, start,
                       mask=mask, plot=False, quiet=True, moments=nmoments, degree=-1,
                       vsyst=offset, mdegree=mdeg, regul=regul, fixed=fixed,
                       lam=np.exp(logLam), velscale_ratio=velscale_ratio,
                       component=component_step6, dust=dust_step6)

        # Step 4: identify surviving (age, metallicity) grid points
        weights_alpha = pp.weights.reshape(templates_alpha.shape[1:]) / pp.weights.sum()
        survive_age_idx, survive_met_idx = np.where(weights_alpha > 0)
        n_survive = len(survive_age_idx)

        # Step 5: rebuild multi-alpha templates at surviving (age, met) points
        # (templates_mgb_lib is already convolved to sigma_min and cropped to the
        # Mgb window -- see extractStarFormationHistories)

        # Step 6: convolve galaxy from (LSF_Data + sigma_kin) to sigma_max so all
        # bins are compared at the same resolution with consistent Mgb bandpass.
        # Bins where sigma_kin > sigma_max are flagged (MGB_RES_FLAG=1).
        wave_gal_full = np.exp(logLam)
        sigma_kin_fwhm = pp.sol[1] * wave_gal_full / C * 2.355
        native_fwhm_gal = np.sqrt(lsf_data_full**2 + sigma_kin_fwhm**2)
        target_fwhm_gal = np.sqrt(lsf_data_full**2 +
                                   (sigma_max * wave_gal_full / C * 2.355)**2)
        sigma_pix_gal, flag_gal = resolution_sigma_pix(
            wave_gal_full, native_fwhm_gal, target_fwhm_gal, velscale)
        galaxy_conv = gaussian_filter1d(log_bin_data, sigma_pix_gal)
        noise_conv  = gaussian_filter1d(noise_new,    sigma_pix_gal)
        mgb_res_flag = int(np.any(flag_gal))

        # Step 7: pseudo-continuum normalisation over b1-b6, extract b3-b4 pixels.
        # Deredshift the galaxy wavelength array per bin so that rest-frame band
        # boundaries b1..b6 align correctly with the galaxy features.
        z_bin = pp.sol[0] / C  # V from _kin.fits is the galaxy recession velocity directly
        wave_gal_rest = wave_gal_full / (1.0 + z_bin)
        log_bin_data_b1b6 = galaxy_conv[idx_gal_mgb]
        noise_b1b6        = noise_conv[idx_gal_mgb]
        wave_b1b6         = wave_gal_rest[idx_gal_mgb]  # rest-frame wavelengths

        gal_norm, noise_norm, cont_gal = normalize_pseudocont(
            wave_b1b6, log_bin_data_b1b6, b1, b2, b5, b6, noise=noise_b1b6)

        # Use the fixed common wavelength grid for b3-b4 (computed from mean
        # redshift in extractStarFormationHistories) to ensure consistent
        # array sizes across bins despite per-bin deredshift variation.
        wave_fif  = wave_gal_rest[idx_gal_b3b4]   # fixed grid, npix_b3b4 pixels
        n_pix_fif = len(wave_fif)
        gal_fif   = np.interp(wave_fif, wave_b1b6, gal_norm)
        noise_fif = np.abs(np.interp(wave_fif, wave_b1b6, noise_norm))

        # Step 8: build per-alpha model FIF vectors and run EMCEE
        # For each alpha, sum templates weighted by the step-3 age/met weights,
        # normalise by pseudo-continuum, interpolate to the galaxy pixel grid.
        idx_b3b4_temp = (wave_temp_mgb >= b3) & (wave_temp_mgb <= b4)
        wave_b3b4_temp = wave_temp_mgb[idx_b3b4_temp]

        model_fif = np.zeros((nAlpha, n_pix_fif))
        for a_idx in range(nAlpha):
            model_spec_a = np.zeros(len(wave_temp_mgb))
            for k in range(n_survive):
                w = weights_alpha[survive_age_idx[k], survive_met_idx[k]]
                model_spec_a += w * templates_mgb_lib[:, survive_age_idx[k],
                                                         survive_met_idx[k], a_idx]
            model_norm_a, _ = normalize_pseudocont(wave_temp_mgb, model_spec_a, b1, b2, b5, b6)
            model_fif[a_idx, :] = np.interp(
                wave_fif, wave_b3b4_temp, model_norm_a[idx_b3b4_temp])

        # Templates are pre-convolved to LSF_Data + sigma_max in
        # extractStarFormationHistories. No per-bin model convolution needed.

        # 1-D EMCEE over alpha; each b3-b4 pixel is an independent observable
        # (Martin-Navarro et al. 2019, Eq. 3). Model is linearly interpolated
        # across the alpha grid using np.interp (avoids Delaunay for 1-D case).
        alpha_fif, alpha_fif_lo, alpha_fif_hi = run_fif_emcee_1d(
            gal_fif, noise_fif, model_fif, alpha_values,
            config["SFH"]["ALPHA_FIX"], nwalkers_fif, nchain_fif)

        # populate w_row: step-3 age/met weights at the nearest alpha grid point
        alpha_idx_fif = find_nearest_index(alpha_values, alpha_fif)
        w_full = np.zeros((nAges, nMetal, nAlpha))
        for k in range(n_survive):
            w_full[survive_age_idx[k], survive_met_idx[k], alpha_idx_fif] = \
                weights_alpha[survive_age_idx[k], survive_met_idx[k]]
        w_row = np.array([np.reshape(w_full, ncomb)])

        best_model_fif  = model_fif[alpha_idx_fif, :]
        spectral_mask_fif = np.ones(n_pix_fif)

        # full-range (step 3) diagnostics
        goodPixels_full = pp.goodpixels
        noise_est_full  = robust_sigma(pp.galaxy[goodPixels_full] - pp.bestfit[goodPixels_full])
        snr_postfit     = np.nanmean(pp.galaxy[goodPixels_full] / noise_est_full)
        formal_error    = pp.error * np.sqrt(pp.chi2)

        if doplot:
            outfigDir = os.path.join(config["GENERAL"]["OUTPUT"], "FigFit_SFH")
            if not os.path.exists(outfigDir):
                os.mkdir(outfigDir)

            plot_ppxf_sfh(pp_step1, np.exp(logLam), i,
                           os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                         + "_sfh_bin_" + str(i) + "_step1.pdf"),
                           snrCubevar=snr_prefit, snrResid=snr_Resid1)

            # weighted mean age and metallicity from step-3 weights
            alpha_fix_idx = find_nearest_index(alpha_values, config["SFH"]["ALPHA_FIX"])
            mean_age_step3 = np.sum(weights_alpha * 10**logAge_grid[:, :, alpha_fix_idx]) / np.sum(weights_alpha)
            mean_met_step3 = np.sum(weights_alpha * metal_grid[:, :, alpha_fix_idx]) / np.sum(weights_alpha)
            mean_results_step3 = np.array([[mean_age_step3, mean_met_step3]])

            if fixed is not None:
                pp.sol[0:nmoments] = start
            plot_ppxf_sfh(pp, np.exp(logLam), i,
                           os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                         + "_sfh_bin_" + str(i) + "_step3.pdf"),
                           snrCubevar=snr_prefit, snrResid=snr_postfit,
                           goodpixelsPre=goodPixels_preclip,
                           mean_results=mean_results_step3)

            plot_fif(wave_b1b6, gal_norm, wave_fif, best_model_fif,
                      b1, b2, b3, b4, b5, b6,
                      alpha_fif, alpha_fif_lo, alpha_fif_hi,
                      mean_age_step3, mean_met_step3, i,
                      os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                    + "_sfh_bin_" + str(i) + "_fif_mgb.pdf"))

        return (
            pp.sol[:],
            w_row,
            best_model_fif,
            formal_error,
            spectral_mask_fif,
            snr_postfit,
            pp.chi2,
            EBV,
            alpha_fif,
            alpha_fif_lo,
            alpha_fif_hi,
            n_survive,
            mgb_res_flag,
            gal_fif,
        )

    except Exception as e:
        import traceback
        logging.warning(f"run_ppxf failed for bin {i}: {e}\n{traceback.format_exc()}")
        print(f"ERROR in run_ppxf bin {i}: {e}", flush=True)
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan)


def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin.
    """
    mean = np.zeros((nbins, 3))
    mean[:, :] = np.nan
    for i in range(nbins):
        mean[i, 0] = np.sum(w_row[i] * ageGrid.ravel()) / np.sum(w_row[i])
        mean[i, 1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
        mean[i, 2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])
    return mean


def save_sfh(
    config,
    ppxf_result,
    formal_error,
    bestfit_fif,
    logLam_b3b4,
    logLam_template_mgb,
    npix_b3b4,
    spectral_mask_fif,
    bin_data_fif,
    snr_postfit,
    red_chi2,
    EBV,
    alpha_fif_arr,
    alpha_fif_lo_arr,
    alpha_fif_hi_arr,
    n_survivors,
    mgb_res_flag,
    mean_result,
    w_row,
    logAge_grid,
    metal_grid,
    alpha_grid,
    velscale,
    ncomb,
    nAges,
    nMetal,
    nAlpha,
):
    """ Save all results to disk. """

    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    columns = [
        fits.Column(name="AGE",          format="D", array=mean_result[:, 0]),
        fits.Column(name="METAL",         format="D", array=mean_result[:, 1]),
        # ALPHA is the nearest grid value to the EMCEE median (for gdu/gpu compatibility)
        fits.Column(name="ALPHA",         format="D", array=mean_result[:, 2]),
        # ALPHA_FIF is the continuous EMCEE median with 16th/84th percentile errors
        fits.Column(name="ALPHA_FIF",     format="D", array=alpha_fif_arr),
        fits.Column(name="DALPHA_FIF_LO", format="D", array=alpha_fif_lo_arr),
        fits.Column(name="DALPHA_FIF_HI", format="D", array=alpha_fif_hi_arr),
    ]

    if config["SFH"]["FIXED"] == False:
        kinematic_columns = [
            fits.Column(name=name, format="D", array=ppxf_result[:, i])
            for i, name in enumerate(["V", "SIGMA"])
        ]
        columns.extend(kinematic_columns)
        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(ppxf_result[:, i + 2]) != 0:
                columns.append(fits.Column(name=name, format="D", array=ppxf_result[:, i + 2]))
        error_columns = [
            fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i])
            for i, name in enumerate(["V", "SIGMA"])
        ]
        columns.extend(error_columns)
        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(formal_error[:, i + 2]) != 0:
                columns.append(fits.Column(name=f"FORM_ERR_{name}", format="D",
                                            array=formal_error[:, i + 2]))

    columns.append(fits.Column(name="SNR_POSTFIT",  format="D", array=snr_postfit[:]))
    columns.append(fits.Column(name="RED_CHI2",     format="D", array=red_chi2[:]))
    columns.append(fits.Column(name="N_SURVIVORS",  format="J", array=n_survivors[:]))
    # MGB_RES_FLAG=1 if sigma_kin > sigma_max (bin already broader than target)
    columns.append(fits.Column(name="MGB_RES_FLAG", format="J", array=mgb_res_flag[:]))
    columns.append(fits.Column(name="EBV",          format="D", array=EBV[:]))

    priHDU  = fits.PrimaryHDU()
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="SFH")
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    fits.HDUList([priHDU, dataHDU]).writeto(outfits_sfh, overwrite=True)
    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")
    logging.info("Wrote: " + outfits_sfh)

    # weights file -- same structure as other SFH wrappers
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh_weights.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights.fits")
    priHDU  = fits.PrimaryHDU()
    dataHDU = fits.BinTableHDU.from_columns(
        fits.ColDefs([fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)]),
        name="WEIGHTS")
    logAge_row, metal_row, alpha_row = [np.reshape(g, ncomb)
                                         for g in [logAge_grid, metal_grid, alpha_grid]]
    gridHDU = fits.BinTableHDU.from_columns(
        fits.ColDefs([fits.Column(name=n, format="D", array=a)
                      for n, a in zip(["LOGAGE", "METAL", "ALPHA"],
                                       [logAge_row, metal_row, alpha_row])]),
        name="GRID")
    hdul = fits.HDUList([_auxiliary.saveConfigToHeader(h, config["SFH"])
                          for h in [priHDU, dataHDU, gridHDU]])
    hdul.writeto(outfits_sfh, overwrite=True)
    for n, v in zip(["NAGES", "NMETAL", "NALPHA"], [nAges, nMetal, nAlpha]):
        fits.setval(outfits_sfh, n, value=v)
    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights.fits")
    logging.info("Wrote: " + outfits_sfh)

    # bestfit file -- BESTFIT/SPEC/LOGLAM describe the FIF b3-b4 central bandpass
    # (pseudo-continuum normalised), not the full SFH.LMIN/LMAX range
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh_bestfit.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")

    priHDU = fits.PrimaryHDU()
    priHDU.header["FITSHIST"] = "FIF b3-b4 central bandpass, pseudo-cont normalised"

    cols = [fits.Column(name='BESTFIT', format=str(npix_b3b4) + 'D', array=bestfit_fif)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)); dataHDU.name = "BESTFIT"

    cols = [fits.Column(name='LOGLAM', format='D', array=logLam_b3b4)]
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)); logLamHDU.name = "LOGLAM"

    cols = [fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template_mgb)]
    logLamTempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)); logLamTempHDU.name = "LOGLAM_TEMPLATE"

    cols = [fits.Column(name="SPEC", format=str(npix_b3b4) + "D", array=bin_data_fif)]
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)); specHDU.name = "SPEC"

    cols = [fits.Column(name="GOODPIX", format="J", array=np.arange(npix_b3b4))]
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)); goodpixHDU.name = "GOODPIX"

    cols = [fits.Column(name="GOODPIX_CLN", format=str(spectral_mask_fif.shape[1]) + "D",
                         array=spectral_mask_fif)]
    goodpixClnHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols)); goodpixClnHDU.name = "GOODPIX_CLN"

    for hdu in [priHDU, dataHDU, logLamHDU, logLamTempHDU, specHDU, goodpixHDU, goodpixClnHDU]:
        _auxiliary.saveConfigToHeader(hdu, config["SFH"])

    fits.HDUList([priHDU, dataHDU, logLamHDU, logLamTempHDU,
                   specHDU, goodpixHDU, goodpixClnHDU]).writeto(outfits_sfh, overwrite=True)
    fits.setval(outfits_sfh, "VELSCALE", value=velscale)
    fits.setval(outfits_sfh, "CRPIX1",   value=1.0)
    fits.setval(outfits_sfh, "CRVAL1",   value=logLam_b3b4[0])
    fits.setval(outfits_sfh, "CDELT1",   value=logLam_b3b4[1] - logLam_b3b4[0])
    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")
    logging.info("Wrote: " + outfits_sfh)


def extractStarFormationHistories(config):
    """
    Starts the computation of stellar population properties via FIF.
    A template grid at fixed alpha/Fe is used for the pPXF age/metallicity
    fit. The [alpha/Fe] is then determined by FIF of the Mgb feature using
    the EMCEE sampler. See module docstring for the full algorithm.
    Args:
    - config: dictionary containing configuration parameters
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"],
                                 config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5", 'r') as f:
        velscale = f.attrs["VELSCALE"]

    velscale_ratio = 2

    # Import all templates
    (
        templates_full, lamRange_temp, logLam_template, ntemplates,
        logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha,
    ) = _prepareTemplates.prepareTemplates_Module(
        config, config["SFH"]["LMIN"], config["SFH"]["LMAX"],
        velscale / velscale_ratio, LSF_Data, LSF_Templates, 'SFH', sortInGrid=True)

    # check that template wavelength range is larger than the fitting range
    if (lamRange_temp[0] >= config["SFH"]["LMIN"]) or (lamRange_temp[1] <= config["SFH"]["LMAX"]):
        logging.info("Template wavelength range needs to be larger than fitting range, exiting")
        printStatus.warning("Template wavelength range needs to be larger than fitting range, exiting")
        return

    # Limit to templates at the nearest grid value to ALPHA_FIX
    alpha_values = alpha_grid[0, 0, :]
    alpha_idx = find_nearest_index(alpha_values, config["SFH"]["ALPHA_FIX"])
    printStatus.running(
        f"SFH.ALPHA_FIX = {config['SFH']['ALPHA_FIX']}; "
        f"nearest grid value = {alpha_values[alpha_idx]:.3f}")
    logging.info(
        f"SFH.ALPHA_FIX requested = {config['SFH']['ALPHA_FIX']}, "
        f"nearest grid value = {alpha_values[alpha_idx]:.3f} (index {alpha_idx})")

    templates_alpha = templates_full[:, :, :, alpha_idx]
    ncomb_alpha = nAges * nMetal

    # Mgb band definition from the LS line-list file
    b1, b2, b3, b4, b5, b6 = get_mgb_band(config)
    mgb_bands = (b1, b2, b3, b4, b5, b6)

    # Define file paths
    gas_cleaned_file = (os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                        + '_gas_cleaned_' + config["GAS"]["LEVEL"].lower() + '.fits')
    bin_spectra_file = (os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
                        + "_bin_spectra.hdf5")

    if (config["SFH"]["SPEC_EMICLEAN"] == True) and os.path.isfile(gas_cleaned_file):
        logging.info(f"Using emission-subtracted spectra at {gas_cleaned_file}")
        printStatus.done("Using emission-subtracted spectra")
        with fits.open(gas_cleaned_file, mem_map=True) as hdul:
            logLam  = hdul[2].data["LOGLAM"]
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config["SFH"]["LMIN"],
                                               np.exp(logLam) < config["SFH"]["LMAX"]))[0]
            bin_data = hdul[1].data["SPEC"].T[idx_lam, :]
            bin_err  = hdul[1].data["ESPEC"].T[idx_lam, :]
            logLam   = logLam[idx_lam]
    else:
        logging.info(f"Using regular spectra without any emission-correction at {bin_spectra_file}")
        printStatus.done("Using regular spectra without any emission-correction")
        with h5py.File(bin_spectra_file, 'r') as f:
            logLam  = f["LOGLAM"][:]
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config["SFH"]["LMIN"],
                                               np.exp(logLam) < config["SFH"]["LMAX"]))[0]
            bin_data = f["SPEC"][idx_lam, :]
            bin_err  = f["ESPEC"][idx_lam, :]
            logLam   = logLam[idx_lam]

    nbins  = bin_data.shape[1]
    npix   = bin_data.shape[0]

    wave_gal_full = np.exp(logLam)
    lsf_data_full = LSF_Data(wave_gal_full)
    offset = (logLam_template[0] - logLam[0]) * C

    # galaxy wavelength indices for the Mgb window and b3-b4 central bandpass.
    # Use the mean systemic velocity to deredshift before comparing to rest-frame
    # band boundaries -- per-bin deredshift is then applied inside run_ppxf.
    if config["SFH"]["FIXED"] == True:
        V_all = np.array(fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True)[1].data.V[:])
        valid_V = V_all[np.array(fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True)[1].data.SIGMA[:]) > 0]
        mean_V = float(np.nanmean(valid_V)) if len(valid_V) > 0 else 0.0
    else:
        mean_V = 0.0
    z_mean = mean_V / C  # V from _kin.fits is the galaxy recession velocity directly
    wave_gal_rest_mean = wave_gal_full / (1.0 + z_mean)
    idx_gal_mgb  = np.where((wave_gal_rest_mean >= b1) & (wave_gal_rest_mean <= b6))[0]
    idx_gal_b3b4 = np.where((wave_gal_rest_mean >= b3) & (wave_gal_rest_mean <= b4))[0]
    npix_b3b4    = len(idx_gal_b3b4)
    logLam_b3b4  = logLam[idx_gal_b3b4]

    if config["SFH"]["NOISE"] == 'variance':
        noise = bin_err
    elif config["SFH"]["NOISE"] == 'constant':
        noise = np.ones((npix, nbins))
        med_bin_err = np.nanmedian(bin_err, axis=0)
        noise *= med_bin_err

    if config["SFH"]["MC_PPXF"] > 0:
        logging.warning("SFH.MC_PPXF > 0 is not implemented for ppxf_sfh_wrapper_fif; ignoring.")

    # Implementation of switch FIXED
    if config["SFH"]["FIXED"] == True:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        if config["SFH"]["MOM"] != config["KIN"]["MOM"]:
            printStatus.running("Moments not the same in KIN and SFH module")
            printStatus.running("Ignoring SFH MOMENTS, using KIN MOMENTS")
        fixed = [True] * config["KIN"]["MOM"]

        ppxf_data = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True)[1].data
        start = np.zeros((nbins, config["KIN"]["MOM"]))
        for i in range(nbins):
            start[i, :] = np.array(ppxf_data[i][: config["KIN"]["MOM"]])

        # sigma_max: maximum measured velocity dispersion -- common convolution
        # target so all bins see the same effective Mgb bandpass.
        sigma_kin_all = np.array(ppxf_data.SIGMA[:])
        valid_sigma = sigma_kin_all[sigma_kin_all > 0]
        if len(valid_sigma) > 0:
            sigma_max = float(np.nanmax(valid_sigma))
        else:
            sigma_max = float(config["KIN"]["SIGMA"])
            logging.warning(f"No valid sigma in _kin.fits; falling back to KIN.SIGMA = {sigma_max:.1f} km/s")
        logging.info(f"FIF sigma_max = {sigma_max:.1f} km/s")
        printStatus.running(f"FIF sigma_max = {sigma_max:.1f} km/s")

    elif config["SFH"]["FIXED"] == False:
        logging.info("Stellar kinematics are NOT FIXED.")
        fixed = None
        start = np.zeros((nbins, config["SFH"]["MOM"]))
        for i in range(nbins):
            if config["SFH"]["MOM"] == 2:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"]])
            elif config["SFH"]["MOM"] == 4:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"], 0.0, 0.0])
            elif config["SFH"]["MOM"] == 6:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"], 0.0, 0.0, 0.0, 0.0])
        sigma_max = float(config["KIN"]["SIGMA"])
        logging.warning(f"SFH.FIXED=False: sigma_max set to KIN.SIGMA = {sigma_max:.1f} km/s.")

    # Convolve the full template grid to the sigma_min-equivalent FWHM:
    # sqrt(LSF_Data^2 + (sigma_min * wave / C * 2.355)^2), evaluated at
    # template wavelengths. This matches the resolution of the minimum-sigma
    # data bin, following Martin-Navarro et al. 2019.
    wave_temp_full   = np.exp(logLam_template)
    lsf_data_at_temp = LSF_Data(wave_temp_full)
    # Pre-convolve templates from native LSF_Templates to LSF_Data only.
    # Kinematic broadening is applied per-bin inside run_ppxf.
    native_fwhm_temp = LSF_Templates(wave_temp_full)
    target_fwhm_temp = np.sqrt(lsf_data_at_temp**2 +
                                (sigma_max * wave_temp_full / C * 2.355)**2)
    sigma_pix_temp, flag_temp = resolution_sigma_pix(
        wave_temp_full, native_fwhm_temp, target_fwhm_temp, velscale / velscale_ratio)

    if np.any(flag_temp):
        logging.warning("Template native resolution already exceeds data LSF at some wavelengths.")

    printStatus.running(f"Convolving {ncomb} templates to sigma_max = {sigma_max:.1f} km/s...")
    templates_full_2d      = templates_full.reshape(templates_full.shape[0], ncomb)
    templates_full_conv_2d = np.empty_like(templates_full_2d)
    for k in range(ncomb):
        templates_full_conv_2d[:, k] = gaussian_filter1d(templates_full_2d[:, k], sigma_pix_temp)
    templates_full_conv = templates_full_conv_2d.reshape(templates_full.shape)
    printStatus.updateDone(f"Convolving {ncomb} templates to sigma_max = {sigma_max:.1f} km/s",
                            progressbar=False)

    # crop to the Mgb window (padded to cover pseudo-continuum sidebands)
    idx_temp_mgb      = np.where((wave_temp_full >= b1 - LAM_PAD) &
                                  (wave_temp_full <= b6 + LAM_PAD))[0]
    templates_mgb_lib = templates_full_conv[idx_temp_mgb, :, :, :]
    wave_temp_mgb     = wave_temp_full[idx_temp_mgb]
    logLam_template_mgb = logLam_template[idx_temp_mgb]

    if 'SPEC_PREMASK' in config["SFH"]:
        goodPixels_step0_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_PREMASK"], logLam)
    else:
        goodPixels_step0_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    goodPixels_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    doplot = config["SFH"].get("PLOT", False)

    sfh_cfg = config["SFH"]
    if "REGUL" in sfh_cfg:
        regul = sfh_cfg["REGUL"]
    elif "REGUL_ERR" in sfh_cfg:
        regul_err = sfh_cfg["REGUL_ERR"]
        regul = 0.0 if regul_err == 0 else 1.0 / regul_err
    else:
        raise KeyError("Either SFH.REGUL or SFH.REGUL_ERR must be set")

    nwalkers_fif = config["SFH"].get("FIF_NWALKERS", 32)
    nchain_fif   = config["SFH"].get("FIF_NCHAIN", 500)

    # Output arrays
    ppxf_result       = np.zeros((nbins, 6))
    w_row             = np.zeros((nbins, ncomb))
    bestfit_fif_all   = np.zeros((nbins, npix_b3b4))
    bin_data_fif_all  = np.zeros((nbins, npix_b3b4))
    formal_error      = np.zeros((nbins, 6))
    spectral_mask_all = np.zeros((nbins, npix_b3b4))
    snr_postfit       = np.zeros(nbins)
    red_chi2          = np.zeros(nbins)
    EBV               = np.zeros(nbins)
    alpha_fif_arr     = np.zeros(nbins)
    alpha_fif_lo_arr  = np.zeros(nbins)
    alpha_fif_hi_arr  = np.zeros(nbins)
    n_survivors       = np.zeros(nbins, dtype=int)
    mgb_res_flag      = np.zeros(nbins, dtype=int)

    if (config["SFH"]["OPT_TEMP"] == "galaxy_single") or (config["SFH"]["OPT_TEMP"] == "galaxy_set"):
        comb_spec  = np.nanmean(bin_data[:, :], axis=1)
        comb_espec = np.nanmean(bin_err[:, :], axis=1)
        optimal_template_out, optimal_template_set = run_ppxf_firsttime(
            templates_alpha, comb_spec, comb_espec, velscale, start[0, :],
            goodPixels_step0_sfh, config["SFH"]["MOM"], offset, -1,
            config["SFH"]["MDEG"], regul, velscale_ratio, ncomb_alpha)
        if config["SFH"]["OPT_TEMP"] == 'galaxy_single':
            optimal_template_comb = optimal_template_out
        if config["SFH"]["OPT_TEMP"] == 'galaxy_set':
            optimal_template_comb = optimal_template_set
    else:
        optimal_template_comb = templates_alpha

    EBV_init   = 0.1
    start_time = time.time()

    def _call(ii):
        return run_ppxf(
            templates_alpha, bin_data[:, ii], noise[:, ii],
            velscale, start[ii, :], goodPixels_step0_sfh, goodPixels_sfh,
            config["SFH"]["MOM"], offset, -1, config["SFH"]["MDEG"],
            regul, config["SFH"]["DOCLEAN"], fixed, velscale_ratio,
            npix, ncomb_alpha, ncomb, nAges, nMetal, nAlpha, nbins, ii,
            optimal_template_comb, EBV_init, logLam,
            logAge_grid, metal_grid, alpha_grid, config, doplot,
            templates_mgb_lib, wave_temp_mgb, idx_gal_mgb, idx_gal_b3b4, lsf_data_full,
            sigma_max, mgb_bands, nwalkers_fif, nchain_fif, alpha_values)

    def _unpack(ii, res):
        ppxf_result[ii, :config["SFH"]["MOM"]] = res[0]
        w_row[ii, :]             = res[1]
        bestfit_fif_all[ii, :]   = res[2]
        formal_error[ii, :config["SFH"]["MOM"]] = res[3]
        spectral_mask_all[ii, :] = res[4]
        snr_postfit[ii]          = res[5]
        red_chi2[ii]             = res[6]
        EBV[ii]                  = res[7]
        alpha_fif_arr[ii]        = res[8]
        alpha_fif_lo_arr[ii]     = res[9]
        alpha_fif_hi_arr[ii]     = res[10]
        n_survivors[ii]          = res[11]
        mgb_res_flag[ii]         = res[12]
        bin_data_fif_all[ii, :]  = res[13]

    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running pPXF+FIF in parallel mode")
        logging.info("Running pPXF+FIF in parallel mode")

        memmap_folder = "/scratch" if os.access("/scratch", os.W_OK) else config["GENERAL"]["OUTPUT"]

        ta_mm = memmap_folder + "/templates_alpha_memmap.tmp"
        dump(templates_alpha, ta_mm); templates_alpha = load(ta_mm, mmap_mode='r')
        tm_mm = memmap_folder + "/templates_mgb_lib_memmap.tmp"
        dump(templates_mgb_lib, tm_mm); templates_mgb_lib = load(tm_mm, mmap_mode='r')
        bd_mm = memmap_folder + "/bin_data_memmap.tmp"
        dump(bin_data, bd_mm); bin_data = load(bd_mm, mmap_mode='r')
        no_mm = memmap_folder + "/noise_memmap.tmp"
        dump(noise, no_mm); noise = load(no_mm, mmap_mode='r')

        def worker(chunk):
            return [_call(ii) for ii in chunk]

        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"] * 10))
        chunks = [range(ii, min(ii + chunk_size, nbins)) for ii in range(0, nbins, chunk_size)]
        parallel_configs = {"n_jobs": config["GENERAL"]["NCPU"], "max_nbytes": "1M",
                             "temp_folder": memmap_folder, "mmap_mode": "c",
                             "return_as": "generator"}
        ppxf_tmp = list(tqdm(
            Parallel(**parallel_configs)(delayed(worker)(ch) for ch in chunks),
            total=len(chunks), desc="Processing chunks", ascii=" #", unit="chunk"))
        ppxf_tmp = [r for ch in ppxf_tmp for r in ch]

        for ii in range(nbins):
            _unpack(ii, ppxf_tmp[ii])

        for f in [ta_mm, tm_mm, bd_mm, no_mm]:
            os.remove(f)
        printStatus.updateDone("Running pPXF+FIF in parallel mode", progressbar=False)

    if config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running pPXF+FIF in serial mode")
        logging.info("Running pPXF+FIF in serial mode")

        if 'DEBUG_BIN' in config["SFH"]:
            runbin = config["SFH"]["DEBUG_BIN"]
            printStatus.running("Running in Debug mode on bins: " + str(runbin))
        else:
            runbin = np.arange(0, nbins)

        for ii in runbin:
            _unpack(ii, _call(ii))
        printStatus.updateDone("Running pPXF+FIF in serial mode", progressbar=False)

    print("             Running pPXF+FIF on %s spectra took %.2fs using %i cores"
          % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"]))
    logging.info("Running pPXF+FIF on %s spectra took %.2fs using %i cores"
                 % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"]))

    idx_error = np.where(np.isnan(ppxf_result[:, 0]) == True)[0]
    if len(idx_error) != 0:
        printStatus.warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             " + str(idx_error))
        logging.warning("Problem bins: " + str(idx_error))
    else:
        print("             " + "There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    mean_results = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins)

    if 'DEBUG_BIN' in config["SFH"]:
        config["SFH"]["DEBUG_BIN"] = str(config["SFH"]["DEBUG_BIN"])

    save_sfh(
        config, ppxf_result, formal_error,
        bestfit_fif_all, logLam_b3b4, logLam_template_mgb, npix_b3b4,
        spectral_mask_all, bin_data_fif_all,
        snr_postfit, red_chi2, EBV,
        alpha_fif_arr, alpha_fif_lo_arr, alpha_fif_hi_arr,
        n_survivors, mgb_res_flag,
        mean_results, w_row,
        logAge_grid, metal_grid, alpha_grid,
        velscale, ncomb, nAges, nMetal, nAlpha,
    )

    return None