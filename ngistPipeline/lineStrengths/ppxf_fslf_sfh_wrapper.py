import logging
import os
import time

import h5py
import numpy as np

from astropy.io import fits
from joblib import Parallel, delayed, dump, load
from ppxf.ppxf import ppxf
from printStatus import printStatus
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.prepareTemplates import _prepareTemplates

import warnings
warnings.filterwarnings("ignore")

# Physical constants
C = 299792.458  # speed of light in km/s

# SFH science window -- covers Mgb, Fe5270, and Fe5335
SFH_LMIN = 5100.0
SFH_LMAX = 5400.0

# Fixed alpha/Fe value used for the Steps 0-3 age/metallicity fit
ALPHA_FIX = 0.20

# Hardcoded testing toggles
USE_STEP3_PRIOR = True # multiplies the templates by its weight

SFH_METHOD_HISTORY = "SFH derived via alpha=0.20 and mgb focused fitting"

"""
PURPOSE:
  This module extracts stellar population properties (age, [M/H], [alpha/Fe])
  by a direct linear pPXF regression over a reduced template basis restricted
  to the Mgb/Fe5270/Fe5335 region, with age/metallicity determined at a
  single fixed alpha/Fe over the full LMIN/LMAX fitting range and then
  expanded back out to the full alpha grid for the final SFH-window fit.

  Algorithm per bin:
    Steps 0-3: pPXF at fixed alpha/Fe (ALPHA_FIX = 0.20, snapped to nearest
               grid value) for age and metallicity, over the LMIN/LMAX
               fitting range, with an EBV prefit (Step 0), noise rescaling
               (Step 1), and 3-sigma clipping (Step 2). Step 3 is fit with
               regularisation (fixed SFH.REGUL / SFH.REGUL_ERR from config)
               to select the surviving (age, met) grid points.
    Step 6:    Identify the literal non-zero weight (age, metallicity) grid
               points from the Step-3 fit (no bounding box -- may be a
               scattered/non-rectangular subset); rebuild a reduced template
               array at exactly those (age, met) survivor points, crossed
               with the FULL native alpha grid (shape [npix_temp, n_survive,
               nAlpha]) -- if USE_STEP3_PRIOR is True, each survivor's
               template stack is pre-scaled by its (normalized) Step-3
               weight, biasing the fit toward the Step-3 proportions; and
               fit this reduced (n_survive x nAlpha) template basis to the
               galaxy data in the fixed SFH window (5100-5400 Ang, covering
               Mgb + Fe5270 + Fe5335, always -- not configurable), masked by
               the originally-imported spectral mask (SFH.SPEC_MASK,
               cropped to the SFH window -- not the Step 1-2 3-sigma clip
               mask), with dust/EBV fixed to the Step-0 value, mdegree=-1,
               regul=0 (never regularised here), linear=True. Kinematics may
               be fixed or free, controlled by SFH.FIXED as in the other
               wrappers.
    Step 7:    Scatter the reduced-grid weights back onto the full native
               (nAges, nMetal, nAlpha) grid (zero outside the survivor set)
               for output/compatibility with the other wrappers' FITS
               structure and mean_agemetalalpha().
"""


def plot_ppxf_sfh(pp, x, i, outfig_ppxf, snrCubevar=-99, snrResid=-99,
                   goodpixelsPre=[], norm=False, mean_results='', figsize=(13, 3.0)):
    # routine to plot first and final pPXF fit
    fig = plt.figure(i, figsize=figsize)
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

    # Shade the pixels entirely outside pp.goodpixels (before the first good
    # pixel and after the last) in grey -- these are excluded from the fit
    # (by the input mask and/or pPXF's own edge trimming) but were previously
    # only marked with a thin vertical line, not shaded as excluded.
    if goodpixels[0] > 0:
        plt.axvspan(x[0], x[goodpixels[0]], facecolor='lightgray')
    if goodpixels[-1] < len(x) - 1:
        plt.axvspan(x[goodpixels[-1]], x[-1], facecolor='lightgray')

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
        plotText_line1 = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}") + \
                   (f", S/N Residual = {snrResid:.1f}")
    if nmom == 4:
        plotText_line1 = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}") + \
                   (f", S/N Residual = {snrResid:.1f}")
    if nmom == 6:
        plotText_line1 = (f"nGIST - Bin {i:10.0f}: Vel = {pp.sol[0]:.0f}, Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, h4 = {pp.sol[3]:.3f}, ") + \
                   (f"h5 = {pp.sol[4]:.3f}, h6 = {pp.sol[5]:.3f}") + \
                   (f", S/N Residual = {snrResid:.1f}")

    plotText = plotText_line1
    if len(mean_results) > 0:
        plotText_line2 = f"Age [Gyr] = {mean_results[0][0]:.2f}, [M/H] = {mean_results[0][1]:.2f}"
        if mean_results.shape[1] > 2:
            plotText_line2 += f", [alpha/Fe] = {mean_results[0][2]:.2f}"
        if figsize[0] < 10:
            plotText = plotText_line1 + "\n" + plotText_line2
        else:
            plotText = plotText_line1 + ", " + plotText_line2

    plt.text(0.01, 0.95, plotText, fontsize=10, ha='left', va='top',
              transform=ax2.transAxes, backgroundcolor='white')
    plt.savefig(outfig_ppxf, bbox_inches='tight', pad_inches=0.3)
    plt.close()


def clip_outliers(galaxy, bestfit, mask):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma in relative
    error until the bad bins don't change any more. This function uses eq.(34)
    of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    while True:
        scale = galaxy[mask] @ bestfit[mask] / np.sum(bestfit[mask] ** 2)
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
    Added for sigma-clipping method
    """
    np.seterr(all='ignore')  # to avoid getting a lot of warnings in zerodivide

    y = np.ravel(y)
    d = y if zero else y - np.median(y)

    mad = np.median(np.abs(d))
    u2 = (d / (9.0 * mad)) ** 2  # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good] * u1 ** 2) ** 2).sum()
    den = (u1 * (1.0 - 5.0 * u2[good])).sum()
    sigma = np.sqrt(num / (den * (den - 1.0)))  # see note in above reference

    return sigma


def find_nearest_index(values, target):
    return int(np.argmin(np.abs(np.asarray(values) - target)))


def run_ppxf_firsttime(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    offset,
    degree,
    mdeg,
    regul,
    velscale_ratio,
    ncomb,
):
    """
    Call PPXF for first time to get optimal template. `templates` here is
    expected to already be at fixed alpha (i.e. templates_alpha), so `ncomb`
    is nAges*nMetal, not the full nAges*nMetal*nAlpha.
    """

    printStatus.running("Running pPXF for the first time")
    median_log_bin_data = np.nanmedian(log_bin_data)
    log_bin_error = log_bin_error / median_log_bin_data
    log_bin_data = log_bin_data / median_log_bin_data
    pp = ppxf(
        templates,
        log_bin_data,
        log_bin_error,
        velscale,
        start,
        goodpixels=goodPixels,
        plot=False,
        quiet=True,
        moments=nmoments,
        degree=-1,
        vsyst=offset,
        mdegree=mdeg,
        regul=regul,
        velscale_ratio=velscale_ratio,
    )

    reshaped_templates = templates.reshape((templates.shape[0], ncomb))
    normalized_weights = pp.weights / np.sum(pp.weights)

    optimal_template = np.zeros((templates.shape[0], 1))
    nonzero_weights = np.shape(np.where(normalized_weights > 0)[0])[0]
    optimal_template_set = np.zeros([templates.shape[0], nonzero_weights])
    printStatus.running('Number of Templates with non-zero weights ' + str(nonzero_weights))

    count_nonzero = 0
    for j in range(0, reshaped_templates.shape[1]):
        optimal_template[:, 0] = optimal_template[:, 0] + reshaped_templates[:, j] * normalized_weights[j]
        if normalized_weights[j] > 0:
            optimal_template_set[:, count_nonzero] = reshaped_templates[:, j]
            count_nonzero += 1

    return optimal_template, optimal_template_set


def run_ppxf(
    templates_alpha,
    templates_full,
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
    ncomb_full,
    nAges,
    nMetal,
    nAlpha,
    nbins,
    i,
    optimal_template_in,
    EBV_init,
    logLam,
    idx_lam_sfh,
    goodPixels_sfh_cropped,
    logLam_full,
    logLam_template,
    logAge_grid,
    metal_grid,
    alpha_grid,
    config,
    doplot,
):
    """
    Steps 0-2 run at fixed alpha/Fe (templates_alpha instead of the full
    multi-alpha grid) over the LMIN/LMAX fitting range: an EBV prefit
    (Step 0), a fake-noise fit to rescale the noise vector (Step 1), and
    3-sigma clipping (Step 2). Step 3 re-fits at fixed alpha/Fe with
    regularisation (`regul`, fixed from SFH.REGUL/SFH.REGUL_ERR) to select
    the literal non-zero-weight (age, met) survivors. Step 6 rebuilds a reduced (n_survive x nAlpha)
    template basis from templates_full at those survivor points (optionally
    pre-scaled by the Step-3 survivor weights as a prior, see
    USE_STEP3_PRIOR). Step 6 fits that reduced basis, unregularised, to the
    data cropped to the fixed Mgb window (5150-5200 Ang), with EBV fixed
    from Step 0.
    """

    try:
        if len(optimal_template_in) > 1:

            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error = log_bin_error / median_log_bin_data
            log_bin_data = log_bin_data / median_log_bin_data

            snr_prefit = np.nanmedian(log_bin_data / log_bin_error)

            ################ 0 ##################
            # Step 0: estimate dust E(B-V) over full range, no polynomials
            component_step0 = [0] * np.prod(optimal_template_in.shape[1:])
            component_true_step0 = np.array(component_step0) == 0
            dust = [{"start": [EBV_init], "bounds": [[0, 8]], "component": component_true_step0}]

            pp_step0 = ppxf(optimal_template_in, log_bin_data, log_bin_error, velscale, lam=np.exp(logLam),
                            goodpixels=goodPixels_step0, degree=-1, mdegree=-1, vsyst=offset,
                            velscale_ratio=velscale_ratio, moments=nmoments, start=start, plot=False,
                            dust=dust, component=component_step0, regul=0, quiet=True)

            if config["SFH"]["OPT_TEMP"] == "default":

                reshaped_templates_alpha = templates_alpha.reshape((templates_alpha.shape[0], ncomb_alpha))

                normalized_weights_step0 = pp_step0.weights / np.sum(pp_step0.weights)
                wNonzero_weights_step0 = np.where(normalized_weights_step0 > 0)[0]
                nNonzero_weights_step0 = np.shape(wNonzero_weights_step0)[0]

                optimal_template_set_step0 = np.zeros([reshaped_templates_alpha.shape[0], nNonzero_weights_step0])
                for j in range(0, nNonzero_weights_step0):
                    optimal_template_set_step0[:, j] = reshaped_templates_alpha[:, wNonzero_weights_step0[j]]

                optimal_template_in = optimal_template_set_step0

            Rv = 4.05
            Av = pp_step0.dust[0]["sol"][0]
            EBV = Av / Rv

            component_step12 = [0] * (np.shape(optimal_template_in)[1])
            component_true_step12 = np.array(component_step12) == 0
            component_step3 = [0] * ncomb_alpha
            component_true_step3 = np.array(component_step3) == 0

            if config["SFH"]["DUST_CORR"] == True:
                dust_step12 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step12,
                                          "fixed": [True]}]
                dust_step3 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step3,
                         "fixed": [True]}]
            else:
                dust_step12 = None
                dust_step3 = None

            ################ 1 ##################
            # Step 1: fake noise fit over full range to estimate noise
            fake_noise = np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels_step0,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                vsyst=offset,
                mdegree=mdeg,
                fixed=fixed,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                component=component_step12,
                dust=dust_step12,
            )

            goodPixels_preclip = goodPixels
            noise_orig = np.mean(log_bin_error[goodPixels_step0])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels_step0] - pp_step1.bestfit[goodPixels_step0])

            snr_Resid1 = np.nanmedian(pp_step1.galaxy[goodPixels_step0] / noise_est)
            noise_new = log_bin_error * (noise_est / noise_orig)
            noise_new_std = robust_sigma(noise_new)

            noise_new[np.where(noise_new <= noise_est - noise_new_std)] = noise_est

            ################ 2 ##################
            # Step 2: clip outliers over full range
            mask0 = logLam > 0
            mask0[:] = False
            mask0[goodPixels] = True
            mask = mask0.copy()

            if doclean == True:
                mask = clip_outliers(log_bin_data, pp_step1.bestfit, mask)
                mask &= mask0

            ################ 3 ##################
            # Step 3: LMIN/LMAX-range fit at fixed alpha/Fe, regularised
            # (fixed SFH.REGUL / SFH.REGUL_ERR from config), used to select
            # the surviving (age, met) grid points.
            pp_step3 = ppxf(
                templates_alpha,
                log_bin_data,
                noise_new,
                velscale,
                start,
                mask=mask,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                vsyst=offset,
                mdegree=mdeg,
                regul=regul,
                fixed=fixed,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                component=component_step3,
                dust=dust_step3,
            )

            ################ 6 ##################
            # Step 6: identify literal non-zero-weight (age, met) survivors
            # from the Step-3 fit; rebuild a reduced (n_survive x nAlpha)
            # template basis from templates_full at those survivor points
            # (if USE_STEP3_PRIOR, pre-scale each survivor's template stack
            # by its normalized Step-3 weight, biasing this fit toward the
            # Step-3 proportions; otherwise use the reduced basis
            # unweighted); then crop to the fixed SFH window (5100-5400 Ang,
            # Mgb + Fe5270 + Fe5335) and fit the reduced basis, masked by the
            # originally-imported spectral mask (goodPixels_sfh_cropped,
            # from SFH.SPEC_MASK, not the Step 1-2 3-sigma clip mask), with
            # EBV fixed from Step 0, no polynomial (mdegree=-1), never
            # regularised (regul=0).
            weights_alpha = pp_step3.weights.reshape(nAges, nMetal) / pp_step3.weights.sum()
            survive_age_idx, survive_met_idx = np.where(weights_alpha > 0)
            n_survive = len(survive_age_idx)
            survivor_weights = weights_alpha[survive_age_idx, survive_met_idx]

            # Mean age/metallicity from the Step-3 (fixed alpha/Fe = ALPHA_FIX)
            # fit -- age/met grids are identical across the alpha axis, so any
            # alpha slice gives the same (age, met) values.
            age_grid_2d = logAge_grid[:, :, 0]
            metal_grid_2d = metal_grid[:, :, 0]
            age_initial = np.sum(weights_alpha * (10 ** age_grid_2d))
            metal_initial = np.sum(weights_alpha * metal_grid_2d)

            noise_est_step3 = robust_sigma(
                pp_step3.galaxy[pp_step3.goodpixels] - pp_step3.bestfit[pp_step3.goodpixels])
            snr_Resid3 = np.nanmedian(pp_step3.galaxy[pp_step3.goodpixels] / noise_est_step3)

            templates_reduced = templates_full[:, survive_age_idx, survive_met_idx, :]
            # shape: (npix_temp, n_survive, nAlpha)

            if USE_STEP3_PRIOR:
                templates_reduced = templates_reduced * survivor_weights[None, :, None]

            templates_reduced_2d = templates_reduced.reshape(
                templates_reduced.shape[0], n_survive * nAlpha)

            log_bin_data_sfh = log_bin_data[idx_lam_sfh]
            noise_new_sfh = noise_new[idx_lam_sfh]
            logLam_sfh = logLam_full[idx_lam_sfh]

            component_reduced = [0] * (n_survive * nAlpha)
            component_true_reduced = np.array(component_reduced) == 0
            if config["SFH"]["DUST_CORR"] == True:
                dust_reduced = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_reduced,
                                 "fixed": [True]}]
            else:
                dust_reduced = None

            pp = ppxf(
                templates_reduced_2d,
                log_bin_data_sfh,
                noise_new_sfh,
                velscale,
                start,
                goodpixels=goodPixels_sfh_cropped,
                plot=False,
                quiet=True,
                moments=nmoments,
                degree=-1,
                mdegree=-1,
                regul=0,
                fixed=fixed,
                linear=True,
                lam=np.exp(logLam_sfh),
                lam_temp=np.exp(logLam_template),
                velscale_ratio=velscale_ratio,
                component=component_reduced,
                dust=dust_reduced,
            )

        goodPixels = pp.goodpixels

        spectral_mask = np.full_like(log_bin_data_sfh, 0.0)
        spectral_mask[goodPixels] = 1.0

        noise_est = robust_sigma(pp.galaxy[goodPixels] - pp.bestfit[goodPixels])
        snr_postfit = np.nanmean(pp.galaxy[goodPixels] / noise_est)

        formal_error = pp.error * np.sqrt(pp.chi2)

        ################ 7 ##################
        # Step 7: scatter reduced-grid weights back onto the full native
        # (nAges, nMetal, nAlpha) grid, zero outside the survivor set.
        w_reduced = pp.weights.reshape(n_survive, nAlpha) / pp.weights.sum()
        weights_full = np.zeros((nAges, nMetal, nAlpha))
        weights_full[survive_age_idx, survive_met_idx, :] = w_reduced
        w_row = np.array([np.reshape(weights_full, ncomb_full)])

        if doplot == True:

            outfigDir = os.path.join(config["GENERAL"]["OUTPUT"], "FigFit_SFH")
            if os.path.exists(outfigDir) == False:
                printStatus.running("Creating directory for pPXF figures:" + outfigDir)
                os.mkdir(outfigDir)

            outfigFile_step1 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_" + str(i) + "_step1.pdf"))
            outfigFile_step3 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_" + str(i) + "_step3.pdf"))
            outfigFile_step6 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_" + str(i) + "_step6.pdf"))

            # Step-3 mean results: age and metal only (2 columns) -- alpha is
            # fixed at ALPHA_FIX, not fitted, so plot_ppxf_sfh's alpha-text
            # branch (triggered only when shape[1] > 2) is skipped.
            mean_results_step3 = np.array([[age_initial, metal_initial]])

            mean_results_step6 = mean_agemetalalpha(w_row, 10 ** logAge_grid, metal_grid, alpha_grid, 1)

            if fixed != None:
                pp.sol[0:nmoments] = start

            tmp_plot1 = plot_ppxf_sfh(pp_step1, np.exp(logLam), i, outfigFile_step1,
                                      snrCubevar=snr_prefit, snrResid=snr_Resid1)

            tmp_plot3 = plot_ppxf_sfh(pp_step3, np.exp(logLam), i, outfigFile_step3,
                                      snrCubevar=snr_prefit, snrResid=snr_Resid3,
                                      goodpixelsPre=goodPixels_preclip,
                                      mean_results=mean_results_step3)

            tmp_plot6 = plot_ppxf_sfh(pp, np.exp(logLam_sfh), i, outfigFile_step6,
                                      snrCubevar=snr_prefit, snrResid=snr_postfit,
                                      mean_results=mean_results_step6,
                                      figsize=(8, 5.0))

        pp.bestfit = pp.bestfit * median_log_bin_data
        bin_data_sfh_out = log_bin_data_sfh * median_log_bin_data

        # No MPOLY is fit or baked in (mdegree=-1, nothing multiplied into
        # the templates) -- output ones for FITS-structure compatibility
        # with the other wrappers.
        mpoly = np.ones_like(log_bin_data_sfh)

        return (
            pp.sol[:],
            w_row,
            pp.bestfit,
            formal_error,
            spectral_mask,
            snr_postfit,
            pp.chi2,
            EBV,
            mpoly,
            bin_data_sfh_out,
            age_initial,
            metal_initial,
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logging.warning(f"run_ppxf failed for bin {i}: {e}\n{tb}")
        print(f"ERROR in run_ppxf bin {i}: {e}\n{tb}", flush=True)
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


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
    ppxf_bestfit,
    logLam,
    goodPixels,
    logLam_template,
    npix,
    spectral_mask,
    bin_data,
    snr_postfit,
    red_chi2,
    EBV,
    mpoly,
    mean_result,
    age_initial,
    metal_initial,
    w_row,
    logAge_grid,
    metal_grid,
    alpha_grid,
    velscale,
    logLam1,
    ncomb,
    nAges,
    nMetal,
    nAlpha,
):
    """ Save all results to disk. """

    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    # AGE/METAL/ALPHA are the Step-6 (Mgb-focused, free alpha/Fe) results --
    # the primary reported SFH. AGE_INITIAL/METAL_INITIAL/ALPHA_INITIAL are
    # the Step-3 (LMIN/LMAX range, fixed alpha/Fe = ALPHA_FIX) results used
    # to select the survivor set.
    columns = [
        fits.Column(name="AGE", format="D", array=mean_result[:, 0]),
        fits.Column(name="METAL", format="D", array=mean_result[:, 1]),
        fits.Column(name="ALPHA", format="D", array=mean_result[:, 2]),
        fits.Column(name="AGE_INITIAL", format="D", array=age_initial),
        fits.Column(name="METAL_INITIAL", format="D", array=metal_initial),
        fits.Column(name="ALPHA_INITIAL", format="D", array=np.full_like(age_initial, ALPHA_FIX)),
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
                columns.append(fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i + 2]))

    columns.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))
    columns.append(fits.Column(name="RED_CHI2", format="D", array=red_chi2[:]))
    columns.append(fits.Column(name="EBV", format="D", array=EBV[:]))

    priHDU = fits.PrimaryHDU()
    priHDU.header['HISTORY'] = SFH_METHOD_HISTORY
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="SFH")

    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE WEIGHTS AND GRID
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh_weights.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights.fits")

    priHDU = fits.PrimaryHDU()
    priHDU.header['HISTORY'] = SFH_METHOD_HISTORY

    cols_weights = [fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_weights), name="WEIGHTS")

    logAge_row, metal_row, alpha_row = map(np.reshape, [logAge_grid, metal_grid, alpha_grid], [ncomb] * 3)

    cols_grid = [fits.Column(name=name, format="D", array=array)
                 for name, array in zip(["LOGAGE", "METAL", "ALPHA"], [logAge_row, metal_row, alpha_row])]
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_grid), name="GRID")

    HDUList = fits.HDUList([_auxiliary.saveConfigToHeader(hdu, config["SFH"]) for hdu in [priHDU, dataHDU, gridHDU]])
    HDUList.writeto(outfits_sfh, overwrite=True)

    for name, value in zip(["NAGES", "NMETAL", "NALPHA"], [nAges, nMetal, nAlpha]):
        fits.setval(outfits_sfh, name, value=value)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights.fits")
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE BESTFIT
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh_bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")

    priHDU = fits.PrimaryHDU()
    priHDU.header['HISTORY'] = SFH_METHOD_HISTORY

    cols = []
    cols.append(fits.Column(name='BESTFIT', format=str(npix) + 'D', array=ppxf_bestfit))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=logLam))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    cols = []
    cols.append(fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template))
    logLamTempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamTempHDU.name = "LOGLAM_TEMPLATE"

    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data))
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = "SPEC"

    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    cols = []
    cols.append(fits.Column(name="GOODPIX_CLN", format=str(spectral_mask.shape[1]) + "D", array=spectral_mask))
    goodpixClnHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixClnHDU.name = "GOODPIX_CLN"

    cols = []
    cols.append(fits.Column(name="MPOLY", format=str(mpoly.shape[1]) + "D", array=mpoly))
    mpolyHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    mpolyHDU.name = "MPOLY"

    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["SFH"])
    logLamTempHDU = _auxiliary.saveConfigToHeader(logLamTempHDU, config["SFH"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["SFH"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["SFH"])
    goodpixClnHDU = _auxiliary.saveConfigToHeader(goodpixClnHDU, config["SFH"])
    mpolyHDU = _auxiliary.saveConfigToHeader(mpolyHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, logLamTempHDU, specHDU, goodpixHDU, goodpixClnHDU, mpolyHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh, "VELSCALE", value=velscale)
    fits.setval(outfits_sfh, "CRPIX1", value=1.0)
    fits.setval(outfits_sfh, "CRVAL1", value=logLam1[0])
    fits.setval(outfits_sfh, "CDELT1", value=logLam1[1] - logLam1[0])

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")
    logging.info("Wrote: " + outfits_sfh)


def extractStarFormationHistories(config):
    """
    Starts the computation of stellar population properties via a reduced
    fixed-alpha-selected multi-alpha template basis, restricted to the Mgb
    region. See module docstring for the full algorithm.

    Templates are prepared over the SFH.LMIN/SFH.LMAX fitting range. Steps
    0-3 run over that same range at fixed alpha/Fe (ALPHA_FIX = 0.20) to
    obtain EBV and the surviving (age, met) grid points. Steps 4-6 rebuild a
    reduced multi-alpha template basis at those survivor points and re-fit
    over the fixed Mgb window (5150-5200 Ang) with EBV fixed.

    Args:
    - config: dictionary containing configuration parameters
    """

    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5", 'r') as f:
        velscale = f.attrs["VELSCALE"]

    velscale_ratio = 2

    # Prepare templates over the SFH.LMIN/SFH.LMAX fitting range
    (
        templates_full,
        lamRange_temp,
        logLam_template,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb_full,
        nAges,
        nMetal,
        nAlpha,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        config["SFH"]["LMIN"],
        config["SFH"]["LMAX"],
        velscale / velscale_ratio,
        LSF_Data,
        LSF_Templates,
        'SFH',
        sortInGrid=True,
    )

    # Template wavelength range must be larger than the fitting range
    if (lamRange_temp[0] >= config["SFH"]["LMIN"]) or (lamRange_temp[1] <= config["SFH"]["LMAX"]):
        logging.info("Template wavelength range needs to be larger than fitting range, exiting")
        printStatus.warning("Template wavelength range needs to be larger than fitting range, exiting")
        return

    lmin_eff = config["SFH"]["LMIN"]
    lmax_eff = config["SFH"]["LMAX"]

    # The fixed SFH science window must also be covered by the fitting range.
    if lmin_eff > SFH_LMIN or lmax_eff < SFH_LMAX:
        logging.info("SFH.LMIN/SFH.LMAX does not cover the SFH science window, exiting")
        printStatus.warning("SFH.LMIN/SFH.LMAX does not cover the SFH science window, exiting")
        return

    # Limit to templates at the nearest grid value to ALPHA_FIX (always 0.20)
    alpha_values = alpha_grid[0, 0, :]
    alpha_idx = find_nearest_index(alpha_values, ALPHA_FIX)
    printStatus.running(
        f"ALPHA_FIX = {ALPHA_FIX}")
    logging.info(
        f"ALPHA_FIX = {ALPHA_FIX}")

    templates_alpha = templates_full[:, :, :, alpha_idx]
    ncomb_alpha = nAges * nMetal

    # Define file paths
    gas_cleaned_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + '_gas_cleaned_' + config["GAS"]["LEVEL"].lower() + '.fits'
    bin_spectra_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5"

    if (config["SFH"]["SPEC_EMICLEAN"] == True) and os.path.isfile(gas_cleaned_file):
        logging.info(f"Using emission-subtracted spectra at {gas_cleaned_file}")
        printStatus.done("Using emission-subtracted spectra")
        with fits.open(gas_cleaned_file, mem_map=True) as hdul:
            logLam = hdul[2].data["LOGLAM"]
            idx_lam = np.where(np.logical_and(
                np.exp(logLam) > lmin_eff,
                np.exp(logLam) < lmax_eff))[0]
            bin_data = hdul[1].data["SPEC"].T[idx_lam, :]
            bin_err = hdul[1].data["ESPEC"].T[idx_lam, :]
            logLam = logLam[idx_lam]
    else:
        logging.info(f"Using regular spectra without any emission-correction at {bin_spectra_file}")
        printStatus.done("Using regular spectra without any emission-correction")
        with h5py.File(bin_spectra_file, 'r') as f:
            logLam = f["LOGLAM"][:]
            idx_lam = np.where(np.logical_and(
                np.exp(logLam) > lmin_eff,
                np.exp(logLam) < lmax_eff))[0]
            bin_data = f["SPEC"][idx_lam, :]
            bin_err = f["ESPEC"][idx_lam, :]
            logLam = logLam[idx_lam]

    logLam_full = logLam.copy()

    # SFH window crop indices -- always 5100-5400 Ang (Mgb + Fe5270 + Fe5335),
    # extended to include the nearest grid point at or below SFH_LMIN and at
    # or above SFH_LMAX, so the window always fully brackets 5100-5400 Ang
    # rather than falling just inside it.
    wave_full = np.exp(logLam_full)
    idx_left_candidates = np.where(wave_full <= SFH_LMIN)[0]
    idx_right_candidates = np.where(wave_full >= SFH_LMAX)[0]
    i_left = idx_left_candidates[-1] if len(idx_left_candidates) > 0 else 0
    i_right = idx_right_candidates[0] if len(idx_right_candidates) > 0 else len(wave_full) - 1
    idx_lam_sfh = np.arange(i_left, i_right + 1)
    npix_sfh = len(idx_lam_sfh)

    nbins = bin_data.shape[1]
    npix = bin_data.shape[0]
    dv = (np.log(lamRange_temp[0]) - logLam[0]) * C

    offset = (logLam_template[0] - logLam[0]) * C

    if config["SFH"]["NOISE"] == 'variance':
        noise = bin_err
    elif config["SFH"]["NOISE"] == 'constant':
        noise = np.ones((npix, nbins))
        med_bin_err = np.nanmedian(bin_err, axis=0)
        noise *= med_bin_err

    # Implementation of switch FIXED
    if config["SFH"]["FIXED"] == True:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        if config["SFH"]["MOM"] != config["KIN"]["MOM"]:
            printStatus.running("Moments not the same in KIN and SFH module")
            printStatus.running("Ignoring SFH MOMENTS, using KIN MOMENTS")
        fixed = [True] * config["KIN"]["MOM"]

        ppxf_data = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True
        )[1].data
        start = np.zeros((nbins, config["KIN"]["MOM"]))
        for i in range(nbins):
            start[i, :] = np.array(ppxf_data[i][: config["KIN"]["MOM"]])

    elif config["SFH"]["FIXED"] == False:
        logging.info(
            "Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties."
        )
        fixed = None
        start = np.zeros((nbins, config["SFH"]["MOM"]))
        for i in range(nbins):
            if config["SFH"]["MOM"] == 2:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"]])
            elif config["SFH"]["MOM"] == 4:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"], 0.0, 0.0])
            elif config["SFH"]["MOM"] == 6:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"], 0.0, 0.0, 0.0, 0.0])

    if 'SPEC_PREMASK' in config["SFH"]:
        goodPixels_step0_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_PREMASK"], logLam)
    else:
        goodPixels_step0_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    goodPixels_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    goodPixels_sfh_cropped = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam_full[idx_lam_sfh])

    doplot = config["SFH"].get("PLOT", False)

    # Stacked (mean) spectrum -- used for OPT_TEMP (galaxy_single/galaxy_set)
    comb_spec = np.nanmean(bin_data[:, :], axis=1)
    comb_espec = np.nanmean(bin_err[:, :], axis=1)

    # Determine the regularisation value from the fixed config setting. Used
    # by the first fixed-alpha fit (run_ppxf_firsttime, optimal template)
    # and by the per-bin Step 3 fit in run_ppxf, which uses regularisation
    # to select the surviving templates.
    sfh_cfg = config["SFH"]
    if "REGUL" in sfh_cfg:
        regul = sfh_cfg["REGUL"]
    elif "REGUL_ERR" in sfh_cfg:
        regul_err = sfh_cfg["REGUL_ERR"]
        regul = 0.0 if regul_err == 0 else 1.0 / regul_err
    else:
        regul = 0.0

    printStatus.running(f"Using fixed regularisation from config: regul = {regul:.4g}")
    logging.info(f"Using fixed config regul = {regul:.4g}")

    # Output arrays
    ppxf_result = np.zeros((nbins, 6))
    w_row = np.zeros((nbins, ncomb_full))
    ppxf_bestfit = np.zeros((nbins, npix_sfh))
    formal_error = np.zeros((nbins, 6))
    spectral_mask = np.zeros((nbins, npix_sfh))
    snr_postfit = np.zeros(nbins)
    red_chi2 = np.zeros(nbins)
    EBV = np.zeros(nbins)
    mpoly = np.zeros((nbins, npix_sfh))
    bin_data_sfh_out = np.zeros((nbins, npix_sfh))
    age_initial = np.zeros(nbins)
    metal_initial = np.zeros(nbins)

    # OPT_TEMP: run once on combined spectrum if requested (uses templates_alpha)
    if (config["SFH"]["OPT_TEMP"] == "galaxy_single") or (config["SFH"]["OPT_TEMP"] == "galaxy_set"):
        optimal_template_out, optimal_template_set = run_ppxf_firsttime(
            templates_alpha,
            comb_spec,
            comb_espec,
            velscale,
            start[0, :],
            goodPixels_step0_sfh,
            config["SFH"]["MOM"],
            offset,
            -1,
            config["SFH"]["MDEG"],
            regul,
            velscale_ratio,
            ncomb_alpha,
        )

        if config["SFH"]["OPT_TEMP"] == 'galaxy_single':
            optimal_template_comb = optimal_template_out
        if config["SFH"]["OPT_TEMP"] == 'galaxy_set':
            optimal_template_comb = optimal_template_set
    else:
        optimal_template_comb = templates_alpha

    EBV_init = 0.1
    start_time = time.time()

    def _call(ii):
        return run_ppxf(
            templates_alpha,
            templates_full,
            bin_data[:, ii],
            noise[:, ii],
            velscale,
            start[ii, :],
            goodPixels_step0_sfh,
            goodPixels_sfh,
            config["SFH"]["MOM"],
            offset,
            -1,
            config["SFH"]["MDEG"],
            regul,
            config["SFH"]["DOCLEAN"],
            fixed,
            velscale_ratio,
            npix,
            ncomb_alpha,
            ncomb_full,
            nAges,
            nMetal,
            nAlpha,
            nbins,
            ii,
            optimal_template_comb,
            EBV_init,
            logLam,
            idx_lam_sfh,
            goodPixels_sfh_cropped,
            logLam_full,
            logLam_template,
            logAge_grid,
            metal_grid,
            alpha_grid,
            config,
            doplot,
        )

    def _unpack(ii, res):
        ppxf_result[ii, :config["SFH"]["MOM"]] = res[0]
        w_row[ii, :] = res[1]
        ppxf_bestfit[ii, :] = res[2]
        formal_error[ii, :config["SFH"]["MOM"]] = res[3]
        spectral_mask[ii, :] = res[4]
        snr_postfit[ii] = res[5]
        red_chi2[ii] = res[6]
        EBV[ii] = res[7]
        mpoly[ii, :] = res[8]
        bin_data_sfh_out[ii, :] = res[9]
        age_initial[ii] = res[10]
        metal_initial[ii] = res[11]

    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running pPXF in parallel mode")
        logging.info("Running pPXF in parallel mode")

        memmap_folder = "/scratch" if os.access("/scratch", os.W_OK) else config["GENERAL"]["OUTPUT"]

        ta_mm = memmap_folder + "/templates_alpha_memmap.tmp"
        dump(templates_alpha, ta_mm); templates_alpha_mm = load(ta_mm, mmap_mode='r')
        tf_mm = memmap_folder + "/templates_full_memmap.tmp"
        dump(templates_full, tf_mm); templates_full_mm = load(tf_mm, mmap_mode='r')
        bd_mm = memmap_folder + "/bin_data_memmap.tmp"
        dump(bin_data, bd_mm); bin_data_mm = load(bd_mm, mmap_mode='r')
        no_mm = memmap_folder + "/noise_memmap.tmp"
        dump(noise, no_mm); noise_mm = load(no_mm, mmap_mode='r')

        def _call_mm(ii):
            return run_ppxf(
                templates_alpha_mm,
                templates_full_mm,
                bin_data_mm[:, ii],
                noise_mm[:, ii],
                velscale,
                start[ii, :],
                goodPixels_step0_sfh,
                goodPixels_sfh,
                config["SFH"]["MOM"],
                offset,
                -1,
                config["SFH"]["MDEG"],
                regul,
                config["SFH"]["DOCLEAN"],
                fixed,
                velscale_ratio,
                npix,
                ncomb_alpha,
                ncomb_full,
                nAges,
                nMetal,
                nAlpha,
                nbins,
                ii,
                optimal_template_comb,
                EBV_init,
                logLam,
                idx_lam_sfh,
                goodPixels_sfh_cropped,
                logLam_full,
                logLam_template,
                logAge_grid,
                metal_grid,
                alpha_grid,
                config,
                doplot,
            )

        def worker(chunk):
            return [_call_mm(ii) for ii in chunk]

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

        for f in [ta_mm, tf_mm, bd_mm, no_mm]:
            os.remove(f)
        printStatus.updateDone("Running pPXF in parallel mode", progressbar=False)

    if config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running pPXF in serial mode")
        logging.info("Running pPXF in serial mode")

        if 'DEBUG_BIN' in config["SFH"]:
            runbin = config["SFH"]["DEBUG_BIN"]
            printStatus.running("Running in Debug mode on bins: " + str(runbin))
        else:
            runbin = np.arange(0, nbins)

        for ii in runbin:
            _unpack(ii, _call(ii))
        printStatus.updateDone("Running pPXF in serial mode", progressbar=False)

    print("             Running pPXF on %s spectra took %.2fs using %i cores"
          % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"]))
    logging.info("Running pPXF on %s spectra took %.2fs using %i cores"
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

    mean_results = mean_agemetalalpha(w_row, 10 ** logAge_grid, metal_grid, alpha_grid, nbins)

    if 'DEBUG_BIN' in config["SFH"]:
        config["SFH"]["DEBUG_BIN"] = str(config["SFH"]["DEBUG_BIN"])

    save_sfh(
        config,
        ppxf_result,
        formal_error,
        ppxf_bestfit,
        logLam_full[idx_lam_sfh],
        goodPixels_sfh_cropped,
        logLam_template,
        npix_sfh,
        spectral_mask,
        bin_data_sfh_out,
        snr_postfit,
        red_chi2,
        EBV,
        mpoly,
        mean_results,
        age_initial,
        metal_initial,
        w_row,
        logAge_grid,
        metal_grid,
        alpha_grid,
        velscale,
        logLam_full[idx_lam_sfh],
        ncomb_full,
        nAges,
        nMetal,
        nAlpha,
    )

    return None