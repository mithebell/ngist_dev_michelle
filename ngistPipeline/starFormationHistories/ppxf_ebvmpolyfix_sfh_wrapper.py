import logging
import os
import shutil
import tempfile
import time
import traceback

import extinction
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from astropy.io import fits
from joblib import Parallel, delayed, dump, load
from numpy.polynomial import legendre
from ppxf.ppxf import ppxf
from printStatus import printStatus
from tqdm import tqdm

from ngistPipeline.auxiliary import _auxiliary
from ngistPipeline.prepareTemplates import _prepareTemplates

import warnings
warnings.filterwarnings("ignore")

# Physical constants
C = 299792.458  # speed of light in km/s


"""
PURPOSE:
  This module performs the extraction of non-parametric star-formation histories
  by full-spectral fitting. It follows the step structure of ppxf_sfh_wrapper
  (Step 1 noise estimate, Step 2 clipping, Step 3 dust-only fit, Step 4 full
  fit). Templates are prepared, and Steps 1-4 are run, over the wavelength
  range LMIN_DUSTMPOLY/LMAX_DUSTMPOLY (falling back to LMIN/LMAX if not
  given) to obtain EBV and MPOLY. A final Step 5 then re-fits the weights over
  the science range (LMIN/LMAX) with EBV and MPOLY fixed from that fit.

  Steps:
    1: fake-noise fit (optimal template, MPOLY, no dust)  [DUSTMPOLY range]
    2: 3-sigma outlier clipping                            [DUSTMPOLY range]
    3: dust-only fit (no polynomials) -> EBV               [DUSTMPOLY range, only if DUST_CORR]
    4: full-template fit, EBV fixed, MPOLY free -> MPOLY   [DUSTMPOLY range]
    5: templates multiplied by the fixed MPOLY, EBV fixed  [LMIN/LMAX]

  All fits pass lam and lam_temp with vsyst=0, so pPXF works out the
  template/data offset itself for each wavelength range.
"""
def plot_ppxf_sfh(pp, x, i, outfig_ppxf, snrCubevar=-99, snrResid=-99,
                  goodpixelsPre=[], norm=False, mean_results='', EBV=None,
                  poly_type='mpoly', mpoly_fixed=None):

    mpl.rcParams['path.simplify'] = False
    mpl.rcParams['path.simplify_threshold'] = 0.0

    fig, (ax2, axpoly, axdust) = plt.subplots(
        3, 1, figsize=(13, 4.8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1],
                     'hspace': 0})

    if norm == True:
        median_norm = np.nanmedian(pp.galaxy[pp.goodpixels])
    else:
        median_norm = 1

    stars_bestfit = pp.bestfit
    galaxy = pp.galaxy
    resid = galaxy - stars_bestfit
    goodpixels = pp.goodpixels

    ll, rr = np.min(x), np.max(x)

    sig3 = np.percentile(abs(resid[goodpixels]), 99.73)

    if np.nanmax(stars_bestfit) > 2.59:
        mx = 3.49
    else:
        mx = 2.49
    mn = -0.49

    # Main spectrum panel
    ax2.axhline(0.0, color='lightgray', linewidth=0.8,
                antialiased=False, zorder=0)

    # Residuals on the full wavelength grid, excluding masked pixels
    resid_plot = np.full(len(resid), np.nan, dtype=float)
    resid_plot[goodpixels] = resid[goodpixels]

    # Stepped spectrum, residual, and best-fit model
    ax2.plot(x, galaxy, color='black', linewidth=0.1,
             drawstyle='steps-mid', antialiased=False,
             solid_joinstyle='miter', solid_capstyle='butt',
             zorder=2)

    ax2.plot(x, resid_plot, color='LimeGreen', linewidth=0.1,
             drawstyle='steps-mid', antialiased=False,
             solid_joinstyle='miter', solid_capstyle='butt',
             zorder=2)

    ax2.plot(x, stars_bestfit, color='red', linewidth=0.1,
             drawstyle='steps-mid', antialiased=False,
             solid_joinstyle='miter', solid_capstyle='butt',
             zorder=3)

    if len(goodpixelsPre) > 0:

        # Current masked regions: pink, with green residuals
        padded = np.r_[-1, goodpixels, len(x)]
        w = np.flatnonzero(np.diff(padded) > 1)

        for wj in w:
            start = padded[wj] + 1
            end = padded[wj + 1] - 1

            left = max(start - 1, 0)
            right = min(end + 1, len(x) - 1)

            ax2.axvspan(x[left], x[right], facecolor='lightpink')

            ax2.plot(x[left:right + 1], resid[left:right + 1],
                     color='green', linewidth=0.1,
                     drawstyle='steps-mid', alpha=0.5,
                     antialiased=False,
                     solid_joinstyle='miter',
                     solid_capstyle='butt', zorder=2)

        for k in goodpixels[[0, -1]]:
            ax2.plot(x[[k, k]], [mn, stars_bestfit[k]],
                     color='lightpink', linewidth=0.5,
                     antialiased=False)

        # Previous masked regions: grey only
        padded = np.r_[-1, goodpixelsPre, len(x)]
        w = np.flatnonzero(np.diff(padded) > 1)

        for wj in w:
            start = padded[wj] + 1
            end = padded[wj + 1] - 1

            left = max(start - 1, 0)
            right = min(end + 1, len(x) - 1)

            ax2.axvspan(x[left], x[right], facecolor='lightgray')

        for k in goodpixelsPre[[0, -1]]:
            ax2.plot(x[[k, k]], [mn, stars_bestfit[k]],
                     color='lightgray', linewidth=0.5,
                     antialiased=False)

    else:

        # Current masked regions: grey, with green residuals
        padded = np.r_[-1, goodpixels, len(x)]
        w = np.flatnonzero(np.diff(padded) > 1)

        for wj in w:
            start = padded[wj] + 1
            end = padded[wj + 1] - 1

            left = max(start - 1, 0)
            right = min(end + 1, len(x) - 1)

            ax2.axvspan(x[left], x[right], facecolor='lightgray')

            ax2.plot(x[left:right + 1], resid[left:right + 1],
                     color='green', linewidth=0.1,
                     drawstyle='steps-mid', alpha=0.5,
                     antialiased=False,
                     solid_joinstyle='miter',
                     solid_capstyle='butt', zorder=2)

        for k in goodpixels[[0, -1]]:
            ax2.plot(x[[k, k]], [mn, stars_bestfit[k]],
                     color='lightgray', linewidth=0.5,
                     antialiased=False)

    ax2.set_ylabel('Flux [normalised]')
    ax2.set_ylim(mn, mx)
    ax2.tick_params(direction='in', which='both')
    ax2.minorticks_on()
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    # Polynomial panel
    if poly_type == 'mpoly':
        polynomial = pp.mpoly
        poly_min = 0.7
        poly_max = 1.3
        reference_value = 1.0
        poly_label = 'm-poly'

    elif poly_type == 'apoly':
        polynomial = pp.apoly
        poly_min = -0.55
        poly_max = 0.55
        reference_value = 0.0
        poly_label = 'a-poly'

    else:
        raise ValueError("poly_type must be either 'mpoly' or 'apoly'")

    # In Step 5 the MPOLY is not fitted by pPXF but fixed beforehand and
    # multiplied into the templates, so show that fixed polynomial instead
    if mpoly_fixed is not None:
        polynomial = mpoly_fixed
        poly_label = 'm-poly (fixed)'

    # If no polynomial was fitted, use a constant reference curve
    if polynomial is None or np.size(polynomial) == 0:
        polynomial = np.full(len(x), reference_value, dtype=float)

    axpoly.plot(x, polynomial, color='orchid', linewidth=0.8,
                antialiased=False)

    axpoly.axhline(reference_value, color='black', linestyle='--',
                   linewidth=0.7, antialiased=False, zorder=0)

    axpoly.set_ylabel(poly_label)
    axpoly.set_ylim(poly_min, poly_max)
    axpoly.tick_params(direction='in', which='both')
    axpoly.minorticks_on()
    axpoly.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    # Dust attenuation panel
    if EBV is not None and np.isfinite(EBV) and EBV >= 0:

        rv = 4.05
        av = rv * float(EBV)

        # extinction.calzetti00 returns A_lambda in magnitudes
        a_lambda = extinction.calzetti00(x, av, rv)

        # Multiplicative factor applied to the model flux
        dust_factor = extinction.apply(
            a_lambda, np.ones_like(x, dtype=float))

        axdust.axhline(1.0, color='black', linestyle='--',
                       linewidth=0.7, antialiased=False)

        axdust.fill_between(x, dust_factor, 1.0,
                            color='mistyrose', alpha=0.35)

        axdust.plot(x, dust_factor, color='firebrick',
                    linewidth=0.8, antialiased=False,
                    label=r'Calzetti dust factor')

        axdust.set_ylim(0., 1.15)
    else:
        axdust.set_ylim(0.95, 1.05)
        axdust.set_yticks([1.0])  # one tick, so it does not overlap the m-poly panel's labels

    axdust.set_ylabel('dust factor')
    axdust.set_xlabel('rest-frame wavelength [Ang]')
    axdust.tick_params(direction='in', which='both')
    axdust.minorticks_on()
    axdust.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    if EBV is not None:
        axdust.legend(loc='best', fontsize=8, frameon=False)

    # add print statements
    nmom = np.max(pp.moments)

    if nmom == 2:
        plotText = (f"nGIST - Bin {i:5.0f}: Vel = {pp.sol[0]:.0f}, "
                    f"Sig = {pp.sol[1]:.0f}") + \
                   (f", S/N_res = {snrResid:.1f}")

    if nmom == 4:
        plotText = (f"nGIST - Bin {i:5.0f}: Vel = {pp.sol[0]:.0f}, "
                    f"Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, "
                    f"h4 = {pp.sol[3]:.3f}") + \
                   (f", S/N_res = {snrResid:.1f}")

    if nmom == 6:
        plotText = (f"nGIST - Bin {i:5.0f}: Vel = {pp.sol[0]:.0f}, "
                    f"Sig = {pp.sol[1]:.0f}, h3 = {pp.sol[2]:.3f}, "
                    f"h4 = {pp.sol[3]:.3f}, h5 = {pp.sol[4]:.3f}, "
                    f"h6 = {pp.sol[5]:.3f}") + \
                   (f", S/N_resid = {snrResid:.1f}")

    if len(mean_results) > 0:
        plotText += (f", Age [Gyr] = {mean_results[0][0]:.2f}, "
                     f"[M/H] = {mean_results[0][1]:.2f}") + \
                    (f", [alpha/Fe] = {mean_results[0][2]:.2f}")
        if EBV is not None:
            plotText += f", E(B-V) = {EBV:.2f}"
    else:
        if EBV is not None:
            plotText += f", E(B-V) = {EBV:.2f}"

    ax2.text(0.01, 0.95, plotText, fontsize=9, ha='left', va='top',
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
        scale = galaxy[mask] @ bestfit[mask]/np.sum(bestfit[mask]**2)
        resid = scale*bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < 3*err
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
     np.seterr(all='ignore') # to avoid getting a lot of warnings in zerodivide

     y = np.ravel(y)
     d = y if zero else y - np.median(y)

     mad = np.median(np.abs(d))
     u2 = (d/(9.0*mad))**2  # c = 9
     good = u2 < 1.0
     u1 = 1.0 - u2[good]
     num = y.size * ((d[good]*u1**2)**2).sum()
     den = (u1*(1.0 - 5.0*u2[good])).sum()
     sigma = np.sqrt(num/(den*(den - 1.0)))  # see note in above reference

     return sigma

def run_ppxf_firsttime(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels,
    nmoments,
    mdeg,
    regul,
    velscale_ratio,
    ncomb,
    logLam,
    logLam_template,
):
    """
    Call PPXF for first time to get optimal template.
    Runs over the LMIN_DUSTMPOLY/LMAX_DUSTMPOLY range on the combined spectrum.
    """

    printStatus.running("Running pPXF for the first time")
    # normalise galaxy spectra and noise
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
        vsyst=0,
        mdegree=mdeg,
        regul = regul,
        velscale_ratio=velscale_ratio,
        lam=np.exp(logLam),
        lam_temp=np.exp(logLam_template),
    )

    # Templates shape is currently [Wavelength, nAge, nMet, nAlpha]. Reshape to [Wavelength, ncomb] to create optimal template
    reshaped_templates = templates.reshape((templates.shape[0], ncomb))
    normalized_weights = pp.weights / np.sum( pp.weights )

    optimal_template   = np.zeros((templates.shape[0],1))
    nonzero_weights = np.shape(np.where(normalized_weights > 0)[0])[0]
    optimal_template_set = np.zeros( [templates.shape[0], nonzero_weights])
    printStatus.running('Number of Templates with non-zero weights ' +str(nonzero_weights))

    count_nonzero = 0
    for j in range(0, reshaped_templates.shape[1]):
        optimal_template[:,0] = optimal_template[:,0] + reshaped_templates[:,j]*normalized_weights[j]
        if normalized_weights[j] > 0:
            optimal_template_set[:,count_nonzero] = reshaped_templates[:,j]
            count_nonzero += 1


    return optimal_template, optimal_template_set

def run_ppxf(
    templates,
    log_bin_data,
    log_bin_error,
    velscale,
    start,
    goodPixels_premask,
    goodPixels_dust,
    goodPixels,
    nmoments,
    mdeg,
    regul,
    doclean,
    fixed,
    velscale_ratio,
    ncomb,
    i,
    optimal_template_in,
    EBV_init,
    logLam,
    nsims,
    idx_lam_sfh,
    logLam_template,
    logAge_grid,
    metal_grid,
    alpha_grid,
    config,
    doplot,
):

    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories.

    log_bin_data, log_bin_error, logLam and all goodPixels arrays cover the
    LMIN_DUSTMPOLY/LMAX_DUSTMPOLY range. Steps 1-4 use these arrays directly.
    Step 5 crops them to the science range (idx_lam_sfh, LMIN/LMAX) and
    re-fits the weights with EBV and MPOLY fixed from Steps 3 and 4.

    No additive polynomials are used anywhere (degree=-1).
    """

    try:
        if len(optimal_template_in) > 1:

            # Normalise galaxy spectra and noise
            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error = log_bin_error / median_log_bin_data
            log_bin_data = log_bin_data / median_log_bin_data

            # Calculate SNR before the fit from flux and flux_err
            snr_prefit = np.nanmedian(log_bin_data/log_bin_error)

            # With fixed kinematics and no multiplicative polynomial, pPXF has
            # no free non-linear parameters left in Steps 1 and 4 (dust is not
            # fitted there) and crashes inside capfit. In that case only the
            # linear fit for the weights is done.
            if fixed is not None and mdeg <= 0:
                linear_only = True
            else:
                linear_only = False

            ################ 1 ##################
            # First step: fit with fake noise to estimate the noise
            fake_noise = np.full_like(log_bin_data, 1.0)

            pp_step1 = ppxf(
                optimal_template_in,
                log_bin_data,
                fake_noise,
                velscale,
                start,
                goodpixels=goodPixels_premask,
                plot=False,
                quiet=True,
                moments=nmoments,
                vsyst=0,
                degree=-1,
                mdegree=mdeg,
                regul=0,
                fixed=fixed,
                linear=linear_only,
                lam=np.exp(logLam),
                lam_temp=np.exp(logLam_template),
                velscale_ratio=velscale_ratio,
            )

            goodPixels_preclip = goodPixels
            # Find a proper estimate of the noise
            noise_orig = np.mean(log_bin_error[goodPixels_premask])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels_premask]-pp_step1.bestfit[goodPixels_premask])

            # Calculate SNR postfit
            snr_Resid1 = np.nanmedian(pp_step1.galaxy[goodPixels_premask]/noise_est)

            # Calculate the new noise, and the sigma of the distribution
            noise_new = log_bin_error*(noise_est/noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est-noise_new_std)] = noise_est

            ################ 2 ##################
            # Second step: clip outliers
            mask0 = np.zeros(len(logLam), dtype=bool)
            mask0[goodPixels] = True
            mask = mask0.copy()

            if doclean == True:
                mask = clip_outliers(log_bin_data, pp_step1.bestfit, mask)
                # Add clipped pixels to the original masked emission lines regions
                mask &= mask0

            ################ 3 ##################
            # Third step: only fit dust, no polynomials
            EBV = np.nan
            Av = np.nan
            Rv = 4.05

            if config["SFH"]["DUST_CORR"]:

                # Create a mask for dust specifically
                mask_dust = np.zeros(len(logLam), dtype=bool)
                mask_dust[goodPixels_dust] = True
                mask_dust &= mask # Keep only pixels good in both masks

                # Create the dust model
                Av_init = Rv * EBV_init
                component_step3 = [0] * np.prod(optimal_template_in.shape[1:])
                component_true_step3 = np.array(component_step3) == 0
                dust = [{"start": [Av_init], "bounds": [[0, 8]], "component": component_true_step3}]

                pp_step3 = ppxf(
                    optimal_template_in,
                    log_bin_data,
                    noise_new,
                    velscale,
                    start,
                    mask=mask_dust,
                    plot=False,
                    quiet=True,
                    moments=nmoments,
                    vsyst=0,
                    degree=-1,
                    mdegree=-1,
                    regul=0,
                    fixed=fixed,
                    lam=np.exp(logLam),
                    lam_temp=np.exp(logLam_template),
                    velscale_ratio=velscale_ratio,
                    component=component_step3,
                    dust=dust,
                )

                # Save dust values
                Av = pp_step3.dust[0]["sol"][0]
                EBV = Av/Rv

            # Components and fixed dust for Steps 4 and 5 (all templates)
            component_step4 = [0]*ncomb
            component_true_step4 = np.array(component_step4) == 0

            if config["SFH"]["DUST_CORR"] == True:
                dust_step4 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step4,
                               "fixed": [True]}]
            else:
                dust_step4 = None

            ################ 4 ##################
            # Fourth step: fit with all templates, EBV fixed from Step 3, MPOLY free
            pp_step4 = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                mask=mask,
                plot=False,
                quiet=True,
                moments=nmoments,
                vsyst=0,
                degree=-1,
                mdegree=mdeg,
                regul=regul,
                fixed=fixed,
                linear=linear_only,
                lam=np.exp(logLam),
                lam_temp=np.exp(logLam_template),
                velscale_ratio=velscale_ratio,
                component=component_step4,
                dust=dust_step4,
            )

            ################ 5 ##################
            # Fifth step: crop to the science range (LMIN/LMAX)
            log_bin_data_sfh = log_bin_data[idx_lam_sfh]
            noise_new_sfh    = noise_new[idx_lam_sfh]
            logLam_sfh       = logLam[idx_lam_sfh]
            mask_sfh         = mask[idx_lam_sfh]

            # Rebuild MPOLY from the Legendre coefficients of Step 4.
            # pPXF stores the coefficients from degree 1 in mpolyweights:
            # MPOLY = 1 + sum(c_k * L_k(x)) for k=1..mdeg
            coeffs = np.zeros(mdeg + 1)
            coeffs[0] = 1.0
            coeffs[1:] = pp_step4.mpolyweights

            # Map wavelengths to [-1, 1] over the DUSTMPOLY range — the same
            # normalisation pPXF used internally in Step 4
            lam_first = logLam[0]
            lam_last = logLam[-1]

            # MPOLY on the template pixel grid (applied to the templates).
            # Template pixels outside the DUSTMPOLY data range are clipped to
            # the end values, so the polynomial is not extrapolated there.
            x_temp = 2*(logLam_template - lam_first)/(lam_last - lam_first) - 1
            x_temp = np.clip(x_temp, -1, 1)
            mpoly_on_temp_grid = legendre.legval(x_temp, coeffs)

            # MPOLY at the science-range data wavelengths (saved for output)
            x_sfh = 2*(logLam_sfh - lam_first)/(lam_last - lam_first) - 1
            mpoly_sfh = legendre.legval(x_sfh, coeffs)

            # Multiply every template by the MPOLY, then restore the grid shape
            templates_2d = templates.reshape(templates.shape[0], ncomb)
            templates_mpoly = templates_2d * mpoly_on_temp_grid[:, None]
            templates_mpoly = templates_mpoly.reshape(templates.shape)

            # Step 5 setup: MPOLY-multiplied templates, no free polynomial
            templates_step5 = templates_mpoly
            mdeg_step5 = -1

            # With fixed kinematics, fixed dust and no polynomials, pPXF has
            # no free non-linear parameters left and only the linear fit for
            # the weights can be done. When the kinematics are free, they are
            # fitted again over the science range.
            if fixed is not None:
                linear_step5 = True
            else:
                linear_step5 = False

            pp = ppxf(
                templates_step5,
                log_bin_data_sfh,
                noise_new_sfh,
                velscale,
                start,
                mask=mask_sfh,
                plot=False,
                quiet=True,
                moments=nmoments,
                vsyst=0,
                degree=-1,
                mdegree=mdeg_step5,
                regul=regul,
                fixed=fixed,
                linear=linear_step5,
                lam=np.exp(logLam_sfh),
                lam_temp=np.exp(logLam_template),
                velscale_ratio=velscale_ratio,
                component=component_step4,
                dust=dust_step4,
            )

        # Update goodpixels from Step 5
        goodPixels_final = pp.goodpixels

        # Make spectral mask (science range)
        spectral_mask = np.full_like(log_bin_data_sfh, 0.0)
        spectral_mask[goodPixels_final] = 1.0

        # Define goodPixels over SNR range for final SNR
        goodPixels_SNR_range = goodPixels_final[
            (np.exp(logLam_sfh[goodPixels_final]) >= config["READ_DATA"]["LMIN_SNR"])
            & (np.exp(logLam_sfh[goodPixels_final]) <= config["READ_DATA"]["LMAX_SNR"])]

        # Calculate the true S/N from the residual over the SNR MIN MAX range
        noise_est = robust_sigma(pp.galaxy[goodPixels_SNR_range] - pp.bestfit[goodPixels_SNR_range])
        snr_postfit = np.nanmedian(pp.galaxy[goodPixels_SNR_range]/noise_est)

        # Make the unconvolved optimal stellar template (no MPOLY, no dust)
        reshaped_templates = templates.reshape((templates.shape[0], ncomb))
        normalized_weights = pp.weights / np.sum( pp.weights )
        optimal_template   = np.zeros( reshaped_templates.shape[0] )

        for j in range(0, reshaped_templates.shape[1]):
            optimal_template = optimal_template + reshaped_templates[:,j]*normalized_weights[j]

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)
        weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum() # Take from 1D list to nD array (nAges, nMet, nAlpha)
        w_row   = np.array([np.reshape(weights, ncomb)])

        # Plotting output
        if doplot == True:

            outfigDir = os.path.join(config["GENERAL"]["OUTPUT"],"FigFit_SFH")
            os.makedirs(outfigDir, exist_ok=True)

            outfigFile_step1 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_"+str(i)+"_step1.pdf"))
            outfigFile_step3 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_"+str(i)+"_step3.pdf"))
            outfigFile_step4 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_"+str(i)+"_step4.pdf"))
            outfigFile_step5 = (
                os.path.join(outfigDir, config["GENERAL"]["RUN_ID"]
                                + "_sfh_bin_"+str(i)+"_step5.pdf"))

            # Calculate mean age, metallicity, and alpha of the final fit
            mean_results_step5 = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, 1)

            EBV_plot = None
            if config["SFH"]["DUST_CORR"]:
                EBV_plot = EBV

            # Steps 1, 3 and 4 are plotted over the DUSTMPOLY range
            tmp_plot1 = plot_ppxf_sfh(pp_step1, np.exp(logLam), i, outfigFile_step1,
                                      snrCubevar=snr_prefit, snrResid=snr_Resid1, poly_type='mpoly')

            if config["SFH"]["DUST_CORR"]:
                tmp_plot3 = plot_ppxf_sfh(pp_step3, np.exp(logLam), i, outfigFile_step3,
                                          snrCubevar=snr_prefit, snrResid=snr_Resid1,
                                          EBV=EBV, poly_type='mpoly')

            tmp_plot4 = plot_ppxf_sfh(pp_step4, np.exp(logLam), i, outfigFile_step4,
                                      snrCubevar=snr_prefit, snrResid=snr_Resid1,
                                      goodpixelsPre=goodPixels_preclip, EBV=EBV_plot,
                                      poly_type='mpoly')

            # Step 5 is plotted over the science range. Pre-clip goodpixels
            # re-indexed to the science-range array (idx_lam_sfh is
            # contiguous, so subtracting its first index works).
            goodPixels_preclip_sfh = np.intersect1d(goodPixels_preclip, idx_lam_sfh) - idx_lam_sfh[0]

            tmp_plot5 = plot_ppxf_sfh(pp, np.exp(logLam_sfh), i, outfigFile_step5,
                                      snrCubevar=snr_prefit, snrResid=snr_postfit,
                                      goodpixelsPre=goodPixels_preclip_sfh,
                                      mean_results=mean_results_step5, EBV=EBV_plot,
                                      poly_type='mpoly', mpoly_fixed=mpoly_sfh)

        # Currently only apply MC described by Pessa et al. 2023 (https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.147P/abstract)
        if nsims > 0:

            w_row_MC = np.zeros((nsims, ncomb))

            for o in range(0, nsims):
                # MC iterated spectrum: gaussian random sampling with mean
                # "log_bin_data_sfh" and sigma "noise_new_sfh"; no regularisation
                log_bin_data_iter = np.random.normal(loc=log_bin_data_sfh, scale=noise_new_sfh)
                mc_iter = ppxf(
                    templates_step5,
                    log_bin_data_iter,
                    noise_new_sfh,
                    velscale,
                    start,
                    mask=mask_sfh,
                    plot=False,
                    quiet=True,
                    moments=nmoments,
                    vsyst=0,
                    degree=-1,
                    mdegree=mdeg_step5,
                    regul=0,
                    fixed=fixed,
                    linear=linear_step5,
                    lam=np.exp(logLam_sfh),
                    lam_temp=np.exp(logLam_template),
                    velscale_ratio=velscale_ratio,
                    component=component_step4,
                    dust=dust_step4,
                )
                weights_mc_iter   = mc_iter.weights.reshape(templates.shape[1:])/mc_iter.weights.sum()
                w_row_MC[o, :]    = np.reshape(weights_mc_iter, ncomb)

            # Calculate mean and error of weights and weighted properties from MC realizations
            w_row_MC_mean         = np.nanmean(w_row_MC, axis=0)
            w_row_MC_err          = (np.nanpercentile(w_row_MC, q=84, axis=0) - np.nanpercentile(w_row_MC, q=16, axis=0))/2
            mean_results_MC_array = mean_agemetalalpha(w_row_MC, 10**logAge_grid, metal_grid, alpha_grid, nsims)
            mean_results_MC_mean  = np.nanmean(mean_results_MC_array, axis=0)
            mean_results_MC_err   = (np.nanpercentile(mean_results_MC_array, q=84, axis=0) - np.nanpercentile(mean_results_MC_array, q=16, axis=0))/2

            mc_results = {
                "w_row_MC_iter": w_row_MC,
                "w_row_MC_mean": w_row_MC_mean,
                "w_row_MC_err": w_row_MC_err,
                "mean_results_MC_iter": mean_results_MC_array,
                "mean_results_MC_mean":  mean_results_MC_mean,
                "mean_results_MC_err":  mean_results_MC_err
            }

        else:
            mc_results = {
                "w_row_MC_iter": np.nan,
                "w_row_MC_mean": np.nan,
                "w_row_MC_err": np.nan,
                "mean_results_MC_iter": np.nan,
                "mean_results_MC_mean":  np.nan,
                "mean_results_MC_err":  np.nan
            }

        # Restore the median normalisation
        pp.bestfit = pp.bestfit * median_log_bin_data

        # MPOLY saved is the Step 4 Legendre poly at the science-range wavelengths
        mpoly = mpoly_sfh

        return(
            pp.sol[:],
            w_row,
            pp.bestfit,
            optimal_template,
            mc_results,
            formal_error,
            spectral_mask,
            snr_postfit,
            pp.chi2,
            EBV,
            mpoly,
        )

    except Exception as e:
        tb = traceback.format_exc()
        printStatus.warning(f"An error occurred in bin {i}: {e}")
        logging.warning(f"run_ppxf failed for bin {i}: {e}\n{tb}")
        mc_results_nan = {
            "w_row_MC_iter": np.nan,
            "w_row_MC_mean": np.nan,
            "w_row_MC_err": np.nan,
            "mean_results_MC_iter": np.nan,
            "mean_results_MC_mean":  np.nan,
            "mean_results_MC_err":  np.nan
            }
        return( np.nan, np.nan, np.nan, np.nan, mc_results_nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin.
    """
    mean = np.zeros( (nbins,3) ); mean[:,:] = np.nan

    for i in range( nbins ):
        mean[i,0] = np.sum(w_row[i] * ageGrid.ravel())   / np.sum(w_row[i])
        mean[i,1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
        mean[i,2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])

    return(mean)


def save_sfh(
    config,
    ppxf_result,
    formal_error,
    ppxf_bestfit,
    logLam,
    goodPixels,
    optimal_template,
    logLam_template,
    npix,
    spectral_mask,
    bin_data,
    snr_postfit,
    red_chi2,
    EBV,
    mpoly,
    mean_result,
    mean_result_MC_mean,
    mean_result_MC_err,
    w_row,
    w_row_MC_iter,
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

    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    # Table HDU with stellar kinematics
    columns = [
        fits.Column(name="AGE", format="D", array=mean_result[:, 0]),
        fits.Column(name="METAL", format="D", array=mean_result[:, 1]),
        fits.Column(name="ALPHA", format="D", array=mean_result[:, 2])
    ]

    if config["SFH"]["MC_PPXF"] > 0:
        mc_columns = [
            fits.Column(name=f"{name}_MC", format="D", array=mean_result_MC_mean[:,i])
            for i, name in enumerate(["AGE", "METAL", "ALPHA"])
        ]
        err_columns = [
            fits.Column(name=f"ERR_{name}_MC", format="D", array=mean_result_MC_err[:,i])
            for i, name in enumerate(["AGE", "METAL", "ALPHA"])
        ]
        columns.extend(mc_columns + err_columns)

    if config["SFH"]["FIXED"] == False:
        kinematic_columns = [
            fits.Column(name=name, format="D", array=ppxf_result[:, i])
            for i, name in enumerate(["V", "SIGMA"])
        ]
        columns.extend(kinematic_columns)

        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(ppxf_result[:, i+2]) != 0:
                columns.append(fits.Column(name=name, format="D", array=ppxf_result[:, i+2]))

        error_columns = [
            fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i])
            for i, name in enumerate(["V", "SIGMA"])
        ]
        columns.extend(error_columns)

        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(formal_error[:, i+2]) != 0:
                columns.append(fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i+2]))

    columns.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))
    columns.append(fits.Column(name="RED_CHI2", format="D", array=red_chi2[:]))
    # E(B-V) from the Step 3 dust-only fit (NaN if DUST_CORR is False)
    columns.append(fits.Column(name="EBV", format="D", array=EBV[:]))

    priHDU = fits.PrimaryHDU()
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

    cols_weights = [fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_weights), name="WEIGHTS")

    logAge_row, metal_row, alpha_row = map(np.reshape, [logAge_grid, metal_grid, alpha_grid], [ncomb]*3)

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
    # SAVE MC RESULTS OF WEIGHTS AND GRID
    if config["SFH"]["MC_PPXF"] > 0:

        outfits_sfh = (
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_sfh-weights_mc.fits"
        )
        printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights_mc.fits")

        priHDU = fits.PrimaryHDU()

        dataHDU_list = []
        for iter_i in range(config["SFH"]["MC_PPXF"]):
            cols = [fits.Column(name="WEIGHTS_MC", format=str(w_row_MC_iter.shape[2]) + "D", array=w_row_MC_iter[:, iter_i, :])]
            dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
            dataHDU.name = "MC_ITER_%s" % iter_i
            dataHDU_list.append(dataHDU)

        priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
        HDUList = fits.HDUList([priHDU] + dataHDU_list)
        HDUList.writeto(outfits_sfh, overwrite=True)

        fits.setval(outfits_sfh, "NAGES", value=nAges)
        fits.setval(outfits_sfh, "NMETAL", value=nMetal)
        fits.setval(outfits_sfh, "NALPHA", value=nAlpha)

        printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights_mc.fits")
        logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE BESTFIT
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh_bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")

    priHDU = fits.PrimaryHDU()

    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=ppxf_bestfit ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    cols = []
    cols.append(fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template))
    logLamTempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamTempHDU.name = "LOGLAM_TEMPLATE"

    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data.T))
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

    # Table HDU with per-bin optimal templates (on template pixel grid, use LOGLAM_TEMPLATE)
    cols = []
    cols.append(fits.Column(name="OPTIMAL_TEMPLATES", format=str(optimal_template.shape[1]) + "D", array=optimal_template))
    optHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    optHDU.name = "OPTIMAL_TEMPLATES"

    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["SFH"])
    logLamTempHDU = _auxiliary.saveConfigToHeader(logLamTempHDU, config["SFH"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["SFH"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["SFH"])
    goodpixClnHDU = _auxiliary.saveConfigToHeader(goodpixClnHDU, config["SFH"])
    mpolyHDU = _auxiliary.saveConfigToHeader(mpolyHDU, config["SFH"])
    optHDU = _auxiliary.saveConfigToHeader(optHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, logLamTempHDU, specHDU, goodpixHDU,
                            goodpixClnHDU, mpolyHDU, optHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh, "VELSCALE", value=velscale)
    fits.setval(outfits_sfh, "CRPIX1", value=1.0)
    fits.setval(outfits_sfh, "CRVAL1", value=logLam1[0])
    fits.setval(outfits_sfh, "CDELT1", value=logLam1[1] - logLam1[0])

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")
    logging.info("Wrote: " + outfits_sfh)




def extractStarFormationHistories(config):
    """
    Starts the computation of non-parametric star-formation histories with pPXF.

    Templates and data are prepared/loaded over LMIN_DUSTMPOLY/LMAX_DUSTMPOLY
    (config["SFH"], defaulting to LMIN/LMAX). Steps 1-4 run over that range to
    obtain EBV and MPOLY. Step 5 then crops to the science range (LMIN/LMAX)
    and re-fits the weights with EBV and MPOLY fixed.

    Args:
    - config: dictionary containing configuration parameters
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    # Open the HDF5 file to get VELSCALE
    with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5", 'r') as f:
        velscale = f.attrs["VELSCALE"]

    velscale_ratio = 2

    # This wrapper fixes the multiplicative polynomial from Step 4, so it needs one
    mdeg = config["SFH"]["MDEG"]
    if mdeg < 1:
        raise ValueError(
            "ppxf_ebvmpolyfix_sfh_wrapper fixes MPOLY from Step 4, so SFH.MDEG "
            f"must be >= 1 (got SFH.MDEG={mdeg})."
        )

    # Additive polynomials are not used in this wrapper
    if config["SFH"].get("ADEG", -1) != -1:
        printStatus.warning("SFH.ADEG is ignored by this wrapper: no additive polynomials are used (degree=-1)")
        logging.warning("SFH.ADEG is ignored by this wrapper: no additive polynomials are used (degree=-1)")

    # Wavelength range for the templates and Steps 1-4. Falls back to LMIN/LMAX
    # when LMIN_DUSTMPOLY/LMAX_DUSTMPOLY are not given.
    lmin_dustmpoly = config["SFH"].get("LMIN_DUSTMPOLY", config["SFH"]["LMIN"])
    lmax_dustmpoly = config["SFH"].get("LMAX_DUSTMPOLY", config["SFH"]["LMAX"])

    # The DUSTMPOLY range must cover the science range, since Step 5 crops
    # the data, mask and MPOLY from it down to LMIN/LMAX
    if lmin_dustmpoly > config["SFH"]["LMIN"] or lmax_dustmpoly < config["SFH"]["LMAX"]:
        logging.info("LMIN_DUSTMPOLY/LMAX_DUSTMPOLY does not cover science range LMIN/LMAX, exiting")
        printStatus.warning("LMIN_DUSTMPOLY/LMAX_DUSTMPOLY does not cover science range LMIN/LMAX, exiting")
        return

    # Prepare templates over the DUSTMPOLY range
    (
        templates,
        lamRange_temp,
        logLam_template,
        ntemplates,
        logAge_grid,
        metal_grid,
        alpha_grid,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    ) = _prepareTemplates.prepareTemplates_Module(
        config,
        lmin_dustmpoly,
        lmax_dustmpoly,
        velscale/velscale_ratio,
        LSF_Data,
        LSF_Templates,
        'SFH',
        sortInGrid=True,
    )

    # Do Checks for template sizes
    if templates.shape[1:] != logAge_grid.shape:
        raise ValueError("Template and log-age grid shapes do not match")
    if templates.shape[1:] != metal_grid.shape:
        raise ValueError("Template and metallicity grid shapes do not match")
    if templates.shape[1:] != alpha_grid.shape:
        raise ValueError("Template and alpha grid shapes do not match")
    if ncomb != np.prod(templates.shape[1:]):
        raise ValueError("ncomb does not match the template-grid dimensions")

    # Effective DUSTMPOLY range — whichever is most restrictive: the requested
    # LMIN_DUSTMPOLY/LMAX_DUSTMPOLY, the template coverage, or the data
    # coverage (LMIN_TOT/LMAX_TOT)
    lmin_eff = max(lmin_dustmpoly, lamRange_temp[0], config["READ_DATA"]["LMIN_TOT"])
    lmax_eff = min(lmax_dustmpoly, lamRange_temp[1], config["READ_DATA"]["LMAX_TOT"])

    if lmin_eff >= lmax_eff:
        logging.info("Template wavelength range does not overlap with data range, exiting")
        printStatus.warning("Template wavelength range does not overlap with data range, exiting")
        return

    if lmin_eff > lmin_dustmpoly or lmax_eff < lmax_dustmpoly:
        printStatus.warning(
            f"Templates or data do not cover the full LMIN_DUSTMPOLY/LMAX_DUSTMPOLY range. "
            f"Using effective range {lmin_eff:.1f}-{lmax_eff:.1f} A for Steps 1-4."
        )
        logging.warning(
            f"Effective DUSTMPOLY range cropped to template/data coverage: {lmin_eff:.1f}-{lmax_eff:.1f} A"
        )

    # Check science range is still covered
    if lmin_eff > config["SFH"]["LMIN"] or lmax_eff < config["SFH"]["LMAX"]:
        logging.info("Template/data wavelength range does not cover science range LMIN/LMAX, exiting")
        printStatus.warning("Template/data wavelength range does not cover science range LMIN/LMAX, exiting")
        return

    # Define file paths
    gas_cleaned_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + '_gas_cleaned_'+config["GAS"]["LEVEL"].lower()+'.fits'
    bin_spectra_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5"

    # Load spectra over the effective DUSTMPOLY range
    if (config["SFH"]["SPEC_EMICLEAN"] == True) and os.path.isfile(gas_cleaned_file):
        logging.info(f"Using emission-subtracted spectra at {gas_cleaned_file}")
        printStatus.done("Using emission-subtracted spectra")
        with fits.open(gas_cleaned_file, mem_map=True) as hdul:
            logLam = hdul[2].data["LOGLAM"]
            idx_lam = np.where(np.logical_and(
                np.exp(logLam) > lmin_eff,
                np.exp(logLam) < lmax_eff))[0]
            bin_data = hdul[1].data["SPEC"].T[idx_lam, :]
            bin_err  = hdul[1].data["ESPEC"].T[idx_lam, :]
            logLam   = logLam[idx_lam]
    else:
        logging.info(f"Using regular spectra without any emission-correction at {bin_spectra_file}")
        printStatus.done("Using regular spectra without any emission-correction")
        with h5py.File(bin_spectra_file, 'r') as f:
            logLam   = f["LOGLAM"][:]
            idx_lam  = np.where(np.logical_and(
                np.exp(logLam) > lmin_eff,
                np.exp(logLam) < lmax_eff))[0]
            bin_data = f["SPEC"][idx_lam, :]
            bin_err  = f["ESPEC"][idx_lam, :]
            logLam   = logLam[idx_lam]

    # Science-range indices within the DUSTMPOLY-range array
    idx_lam_sfh = np.where(np.logical_and(
        np.exp(logLam) > config["SFH"]["LMIN"],
        np.exp(logLam) < config["SFH"]["LMAX"]
    ))[0]
    npix_sfh = len(idx_lam_sfh)

    # Define additional variables
    nbins = bin_data.shape[1]
    npix  = bin_data.shape[0]

    # Check what type of noise should be passed on
    if config["SFH"]["NOISE"] == 'variance': # use noise from cube
        noise = bin_err  # already converted to noise, i.e. sqrt(variance)
    elif config["SFH"]["NOISE"] == 'constant': # use constant noise
        noise = np.ones((npix, nbins))
        # while constant, the noise does need to be scaled to match the bin_err
        med_bin_err = np.nanmedian(bin_err, axis=0)
        noise *= med_bin_err

    nsims = config["SFH"]["MC_PPXF"]

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config["SFH"]["FIXED"]:
        logging.info("Stellar kinematics are FIXED to the results obtained before.")
        if config["SFH"]["MOM"] != config["KIN"]["MOM"]:
            raise ValueError(
                "With SFH.FIXED=True, SFH.MOM must equal KIN.MOM "
                f"(got SFH.MOM={config['SFH']['MOM']} and "
                f"KIN.MOM={config['KIN']['MOM']})."
            )
        fixed = [True] * config["KIN"]["MOM"]

        # Read PPXF results
        ppxf_data = fits.open(
            os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
            + "_kin.fits", mem_map=True
        )[1].data
        start = np.zeros((nbins, config["KIN"]["MOM"]))
        for i in range(nbins):
            start[i, :] = np.array(ppxf_data[i][: config["KIN"]["MOM"]])

    # Do *NOT* fix kinematics to those obtained previously
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

    # Define goodpixels over the DUSTMPOLY range
    goodPixels_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    # Check if a premask for the first step has been defined
    if 'SPEC_PREMASK' in config["SFH"]:
        goodPixels_premask_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_PREMASK"], logLam)
    else:
        goodPixels_premask_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    # Check if a separate mask for the dust step has been defined
    if 'SPEC_DUSTMASK' in config["SFH"]:
        goodPixels_dust_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_DUSTMASK"], logLam)
    else:
        goodPixels_dust_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    # Goodpixels for saving — indexed into the science range only
    goodPixels_sfh_cropped = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam[idx_lam_sfh])

    # Check if plot keyword is set
    doplot = config["SFH"].get("PLOT", False)

    # Define regularisation value
    sfh_cfg = config["SFH"]
    if "REGUL" in sfh_cfg:
        regul = sfh_cfg["REGUL"]
    elif "REGUL_ERR" in sfh_cfg:
        regul_err = sfh_cfg["REGUL_ERR"]
        regul = 0.0 if regul_err == 0 else 1.0 / regul_err
    else:
        raise KeyError("Either SFH.REGUL or SFH.REGUL_ERR must be set")

    # Define output arrays (sized to the science range, except optimal_template)
    ppxf_result   = np.zeros((nbins, 6))
    w_row         = np.zeros((nbins, ncomb))
    ppxf_bestfit  = np.zeros((nbins, npix_sfh))
    formal_error  = np.zeros((nbins, 6))
    spectral_mask = np.zeros((nbins, npix_sfh))
    snr_postfit   = np.zeros(nbins)
    red_chi2      = np.zeros(nbins)
    EBV           = np.zeros(nbins)
    mpoly         = np.zeros((nbins, npix_sfh))
    # optimal_template is on the template pixel grid (DUSTMPOLY range)
    optimal_template = np.zeros((nbins, templates.shape[0]))

    # Define output arrays of MC realizations
    if nsims > 0:
        logging.info('MC realizations will be applied with %s iterations to estimate weights uncertainties.' % nsims)
    w_row_MC_iter        = np.zeros((nbins, nsims, ncomb))
    w_row_MC_mean        = np.zeros((nbins, ncomb))
    w_row_MC_err         = np.zeros((nbins, ncomb))
    mean_results_MC_iter = np.zeros((nbins, nsims, 3))
    mean_results_MC_mean = np.zeros((nbins, 3))
    mean_results_MC_err  = np.zeros((nbins, 3))

    # ====================
    # If OPT_TEMP keyword set to 'galaxy_single' or 'galaxy_set' then
    # run PPXF once on combined mean spectrum to get a single or optimal template set
    if (config["SFH"]["OPT_TEMP"] == "galaxy_single") or (config["SFH"]["OPT_TEMP"] == "galaxy_set"):
        comb_spec  = np.nanmean(bin_data[:, :], axis=1)
        comb_espec = np.nanmean(bin_err[:, :], axis=1)

        optimal_template_out, optimal_template_set = run_ppxf_firsttime(
            templates,
            comb_spec,
            comb_espec,
            velscale,
            start[0, :],
            goodPixels_premask_sfh,
            config["SFH"]["MOM"],
            mdeg,
            regul,
            velscale_ratio,
            ncomb,
            logLam,
            logLam_template,
        )

        if config["SFH"]["OPT_TEMP"] == 'galaxy_single':
            optimal_template_comb = optimal_template_out # single template
        if config["SFH"]["OPT_TEMP"] == 'galaxy_set':
            optimal_template_comb = optimal_template_set # selected set from total galaxy fit
    else:
        optimal_template_comb = templates # all templates

    # ====================
    EBV_init = 0.1  # PHANGS value initial guess

    # ====================
    # Run PPXF
    start_time = time.time()

    if config["GENERAL"]["PARALLEL"] == True:
        printStatus.running("Running pPXF in parallel mode")
        logging.info("Running pPXF in parallel mode")

        # Create a unique temporary directory for this run's memmaps
        memmap_parent = ("/scratch"
            if os.access("/scratch", os.W_OK)
            else config["GENERAL"]["OUTPUT"])

        memmap_folder = tempfile.mkdtemp(
            prefix=f"{config['GENERAL']['RUN_ID']}_sfh_",
            dir=memmap_parent)

        # Dump the arrays and reload them as read-only memmaps
        templates_filename_memmap = os.path.join(memmap_folder, "templates_memmap.tmp")
        dump(templates, templates_filename_memmap)
        templates = load(templates_filename_memmap, mmap_mode="r")

        if config["SFH"]["OPT_TEMP"] == "default":
            optimal_template_comb = templates
        else:
            opt_temp_file = os.path.join(memmap_folder, "optimal_template_memmap.tmp")
            dump(optimal_template_comb, opt_temp_file)
            optimal_template_comb = load(opt_temp_file, mmap_mode="r")

        bin_data_filename_memmap = os.path.join(memmap_folder, "bin_data_memmap.tmp")
        dump(bin_data, bin_data_filename_memmap)
        bin_data = load(bin_data_filename_memmap, mmap_mode="r")

        noise_filename_memmap = os.path.join(memmap_folder, "noise_memmap.tmp")
        dump(noise, noise_filename_memmap)
        noise = load(noise_filename_memmap, mmap_mode="r")

        # Define a function to encapsulate the work done in the loop
        def worker(chunk, templates):
            results = []
            for i in chunk:
                result = run_ppxf(
                    templates,
                    bin_data[:, i],
                    noise[:, i],
                    velscale,
                    start[i, :],
                    goodPixels_premask_sfh,
                    goodPixels_dust_sfh,
                    goodPixels_sfh,
                    config["SFH"]["MOM"],
                    mdeg,
                    regul,
                    config["SFH"]["DOCLEAN"],
                    fixed,
                    velscale_ratio,
                    ncomb,
                    i,
                    optimal_template_comb,
                    EBV_init,
                    logLam,
                    nsims,
                    idx_lam_sfh,
                    logLam_template,
                    logAge_grid,
                    metal_grid,
                    alpha_grid,
                    config,
                    doplot,
                )
                results.append(result)
            return results

        # Use joblib to parallelize the work
        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"] * 10))
        chunks = [range(i, min(i + chunk_size, nbins)) for i in range(0, nbins, chunk_size)]

        with Parallel(n_jobs=config["GENERAL"]["NCPU"], max_nbytes=None, return_as="generator") as parallel:
            ppxf_tmp = list(tqdm(
                parallel(delayed(worker)(chunk, templates) for chunk in chunks),
                total=len(chunks), desc="Processing chunks",
                ascii=" #", unit="chunk"
            ))

        # Flatten the results
        ppxf_tmp = [result for chunk_results in ppxf_tmp for result in chunk_results]

        # Unpack results
        for i in range(0, nbins):
            ppxf_result[i, :config["SFH"]["MOM"]] = ppxf_tmp[i][0]
            w_row[i, :]                            = ppxf_tmp[i][1]
            ppxf_bestfit[i, :]                     = ppxf_tmp[i][2]
            optimal_template[i, :]                 = ppxf_tmp[i][3]
            w_row_MC_iter[i, :, :]                 = ppxf_tmp[i][4]["w_row_MC_iter"]
            w_row_MC_mean[i, :]                    = ppxf_tmp[i][4]["w_row_MC_mean"]
            w_row_MC_err[i, :]                     = ppxf_tmp[i][4]["w_row_MC_err"]
            mean_results_MC_iter[i, :, :]          = ppxf_tmp[i][4]["mean_results_MC_iter"]
            mean_results_MC_mean[i, :]             = ppxf_tmp[i][4]["mean_results_MC_mean"]
            mean_results_MC_err[i, :]              = ppxf_tmp[i][4]["mean_results_MC_err"]
            formal_error[i, :config["SFH"]["MOM"]] = ppxf_tmp[i][5]
            spectral_mask[i, :]                    = ppxf_tmp[i][6]
            snr_postfit[i]                         = ppxf_tmp[i][7]
            red_chi2[i]                            = ppxf_tmp[i][8]
            EBV[i]                                 = ppxf_tmp[i][9]
            mpoly[i, :]                            = ppxf_tmp[i][10]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=False)

    if config["GENERAL"]["PARALLEL"] == False:
        printStatus.running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")

        # Check if we need to run all bins or only a subset
        debug_bins = config["SFH"].get("DEBUG_BIN") or []
        if debug_bins:
            runbin = debug_bins
            printStatus.running("Running PPXF in Debug mode on bins: " + str(runbin))
        else:
            runbin = np.arange(0, nbins)

        for i in runbin:
            (
                ppxf_result[i, :config["SFH"]["MOM"]],
                w_row[i, :],
                ppxf_bestfit[i, :],
                optimal_template[i, :],
                mc_results_i,
                formal_error[i, :config["SFH"]["MOM"]],
                spectral_mask[i, :],
                snr_postfit[i],
                red_chi2[i],
                EBV[i],
                mpoly[i, :],
            ) = run_ppxf(
                templates,
                bin_data[:, i],
                noise[:, i],
                velscale,
                start[i, :],
                goodPixels_premask_sfh,
                goodPixels_dust_sfh,
                goodPixels_sfh,
                config["SFH"]["MOM"],
                mdeg,
                regul,
                config["SFH"]["DOCLEAN"],
                fixed,
                velscale_ratio,
                ncomb,
                i,
                optimal_template_comb,
                EBV_init,
                logLam,
                nsims,
                idx_lam_sfh,
                logLam_template,
                logAge_grid,
                metal_grid,
                alpha_grid,
                config,
                doplot,
            )
            w_row_MC_iter[i, :, :]        = mc_results_i["w_row_MC_iter"]
            w_row_MC_mean[i, :]           = mc_results_i["w_row_MC_mean"]
            w_row_MC_err[i, :]            = mc_results_i["w_row_MC_err"]
            mean_results_MC_iter[i, :, :] = mc_results_i["mean_results_MC_iter"]
            mean_results_MC_mean[i, :]    = mc_results_i["mean_results_MC_mean"]
            mean_results_MC_err[i, :]     = mc_results_i["mean_results_MC_err"]
        printStatus.updateDone("Running PPXF in serial mode", progressbar=False)

    print(
        "             Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )
    logging.info(
        "Running PPXF on %s spectra took %.2fs using %i cores"
        % (nbins, time.time() - start_time, config["GENERAL"]["NCPU"])
    )

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( ppxf_result[:, 0] ) == True )[0]

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

    # Calculate mean age, metallicity and alpha
    mean_results = mean_agemetalalpha(
        w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins
    )

    # Save to file
    if 'DEBUG_BIN' in config["SFH"]:
        # replace config keyword with string to save it in header later
        config["SFH"]["DEBUG_BIN"] = str(config["SFH"]["DEBUG_BIN"])

    save_sfh(
        config,
        ppxf_result,
        formal_error,
        ppxf_bestfit,
        logLam[idx_lam_sfh],
        goodPixels_sfh_cropped,
        optimal_template,
        logLam_template,
        npix_sfh,
        spectral_mask,
        bin_data[idx_lam_sfh, :],
        snr_postfit,
        red_chi2,
        EBV,
        mpoly,
        mean_results,
        mean_results_MC_mean,
        mean_results_MC_err,
        w_row,
        w_row_MC_iter,
        logAge_grid,
        metal_grid,
        alpha_grid,
        velscale,
        logLam[idx_lam_sfh],
        ncomb,
        nAges,
        nMetal,
        nAlpha,
    )

    # Clean up the memmaps (after save_sfh, which still reads bin_data)
    if config["GENERAL"]["PARALLEL"] == True:
        templates._mmap.close()
        bin_data._mmap.close()
        noise._mmap.close()

        if config["SFH"]["OPT_TEMP"] != "default":
            optimal_template_comb._mmap.close()

        shutil.rmtree(memmap_folder)

    return None
