import logging
import os
import time

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import extinction
import shutil
import tempfile

from astropy.io import fits
from astropy.stats import biweight_location
from joblib import Parallel, delayed, dump, load
from packaging import version
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
  This module extracts non-parametric star-formation histories by full-spectral
  fitting.  Basically, it acts as an interface between pipeline and the pPXF
  routine from Cappellari & Emsellem 2004
  (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).

  On top of the full-grid fit of ppxf_sfh_wrapper.py it adds a feature-focused
  ("short") fit that constrains [alpha/Fe] from the SFH.SPEC_FSLF window, with
  the mean age and mean metallicity of the full-range ("long") fit held fixed.

  METHOD: fix <Age> and <[M/H]>   (ppxf_fslf_fixmeanagemet_sfh_wrapper.py)

  Steps 1-4 are identical to ppxf_sfh_wrapper.py (no Monte Carlo):
    Step 1: fake-noise fit to rescale the noise vector.
    Step 2: 3-sigma clipping of outliers (SFH.DOCLEAN).
    Step 3: dust-only (E(B-V)) fit, no polynomials.
    Step 4: regularised fit of the FULL (nAges x nMetal x nAlpha) template grid
            over the SFH.LMIN/SFH.LMAX range, EBV fixed from Step 3 when
            SFH.DUST_CORR is set.  This is the "long" fit: it provides the
            kinematics, RED_CHI2, SNR_POSTFIT, the WEIGHTS cube, and the mean
            age / [M/H] / [alpha/Fe] (AGE, METAL, ALPHA_LONG).

  Step 5 (feature fit, the "short" fit):
    - <Age> and <[M/H]> are the weighted means of the Step-4 weights (same
      definition as the saved AGE / METAL: linear age in Gyr, arithmetic
      [M/H]), each snapped to the nearest grid value.
    - Only the nAlpha templates at that single (age, [M/H]) grid cell are fit,
      to the galaxy cropped to the SFH.SPEC_FSLF window (bounding extent of its
      good pixels; SFH.SPEC_MASK if SPEC_FSLF is not set).
    - Pixel mask: the SFH.SPEC_MASK good pixels cropped to that window (not the
      Step 2 clip mask).
    - EBV is fixed to the Step-3 value and the kinematics are fixed to the
      Step-4 solution (so linear=True is safe), mdegree=-1, no additive
      polynomial, regul=0.
    - ALPHA is the weighted mean alpha of the fitted alpha weights (NaN if they
      are all zero).

  Outputs:
    _sfh.fits          AGE, METAL (Step 4), ALPHA (Step 5), ALPHA_LONG (Step 4),
                       kinematics / SNR_POSTFIT / RED_CHI2 / EBV (Step 4),
                       RED_CHI2_SHORT (Step 5).
    _sfh_weights.fits  WEIGHTS (Step-4 cube), GRID, WEIGHTS_SHORT (Step-5 cube,
                       last extension; only the fitted (age, [M/H]) cell is
                       populated and it sums to 1).
    _sfh_bestfit.fits  All base extensions (Step 4, full range) plus
                       BESTFIT_SHORT and LOGLAM_SHORT (Step 5, the SPEC_FSLF
                       window); header keys IPIXLO / IPIXHI on BESTFIT_SHORT
                       give the first / last pixel (0-based, inclusive) of that
                       window in LOGLAM (LOGLAM_SHORT == LOGLAM[IPIXLO:IPIXHI+1]).
"""


def plot_ppxf_sfh(pp, x, i, outfig_ppxf, snrCubevar=-99, snrResid=-99,
                  goodpixelsPre=[], norm=False, mean_results='', EBV=None,
                  poly_type='mpoly', figsize=(13, 4.8)):

    mpl.rcParams['path.simplify'] = False
    mpl.rcParams['path.simplify_threshold'] = 0.0

    fig, (ax2, axpoly, axdust) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
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

    # Residuals on the full wavelength grid, excluding masked pixels
    resid_plot = np.full(len(resid), np.nan, dtype=float)
    resid_plot[goodpixels] = resid[goodpixels]

    # Main spectrum panel
    ax2.axhline(0.0, color='lightgray', linewidth=0.8,
                antialiased=False, zorder=0)

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

    # If no polynomial was fitted, use a constant reference curve
    if polynomial is None or np.size(polynomial) == 0:
        polynomial = np.full(len(x), reference_value, dtype=float)

    if EBV == None:
        poly_min = np.nanmin(polynomial)
        poly_max = np.nanmax(polynomial)
        padding = 0.25 * max(poly_max - poly_min, 1.0)
        poly_min -= padding
        poly_max += padding
        EBVstring = 0.0
    else:
        EBVstring = 0.0

    axpoly.plot(x, polynomial, color='orchid', linewidth=0.8,
                antialiased=False)

    axpoly.axhline(reference_value, color='black', linestyle='--',
                   linewidth=0.7, antialiased=False, zorder=0)

    axpoly.set_ylabel(poly_label)
    axpoly.set_ylim(poly_min, poly_max)
    axpoly.set_xlabel('wavelength [Ang]')
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

        dust_min = np.nanmin(dust_factor)

        #axdust.set_ylim(max(0.0, dust_min - 0.03), 1.03)
        axdust.set_ylim(0., 1.15)
    else:
        axdust.set_ylim(0.95, 1.05)

    
    axdust.set_ylabel('dust factor')
    axdust.set_xlabel('wavelength [Ang]')
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
    offset,
    degree,
    mdeg,
    regul,
    velscale_ratio,
    ncomb,
):
    """
    Call PPXF for first time to get optimal template
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
        vsyst=offset,
        degree=degree,
        mdegree=mdeg,
        regul = regul,
        velscale_ratio=velscale_ratio,
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
    offset,
    degree,
    mdeg,
    regul,
    doclean,
    fixed,
    velscale_ratio,
    npix,
    ncomb,
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
    i_left,
    i_right,
    goodPixels_short,
    logLam_template,
    nAges,
    nMetal,
    nAlpha,
):

    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).

    Steps 1-4 are identical to ppxf_sfh_wrapper.py (long fit, full template
    grid, LMIN/LMAX range).  Step 5 fixes <Age> and <[M/H]> from Step 4
    (snapped to the nearest grid cell) and fits only the nAlpha templates at
    that cell to the SPEC_FSLF window (short fit), with EBV and the kinematics
    fixed to the Step 3 / Step 4 values.
    """
    # printStatus.progressBar(i, nbins, barLength=50)

    try:
        if len(optimal_template_in) > 1:

            # Normalise galaxy spectra and noise
            median_log_bin_data = np.nanmedian(log_bin_data)
            log_bin_error = log_bin_error / median_log_bin_data
            log_bin_data = log_bin_data / median_log_bin_data

            # Calculate SNR before the fit from flux and flux_err
            snr_prefit = np.nanmedian(log_bin_data/log_bin_error)

            # First Call PPXF - do fit and estimate noise
            # use fake noise for first iteration
            fake_noise=np.full_like(log_bin_data, 1.0)

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
                vsyst=offset,
                degree=degree,
                mdegree=mdeg,
                regul=0,
                fixed=fixed,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,  
            )
            
            goodPixels_preclip = goodPixels
            # Find a proper estimate of the noise
            noise_orig = np.mean(log_bin_error[goodPixels_premask])
            noise_est = robust_sigma(
                pp_step1.galaxy[goodPixels_premask]-pp_step1.bestfit[goodPixels_premask])

            # Calculate SNR postfit
            snr_Resid1 = np.nanmedian(pp_step1.galaxy[goodPixels_premask]/noise_est)
            # Calculate the new noise, and the sigma of the distribution.
            noise_new = log_bin_error*(noise_est/noise_orig)
            noise_new_std = robust_sigma(noise_new)

            # A temporary fix for the noise issue where a single high S/N spaxel causes clipping of the entire spectrum
            noise_new[np.where(noise_new <= noise_est-noise_new_std)] = noise_est

            ################ 2 ##################
            # Second step (formely done with pPXF CLEAN)
            # switch to mask instead of goodpixels
            mask0 = np.zeros(len(logLam), dtype=bool)
            mask0[goodPixels] = True
            mask = mask0.copy()

            if doclean == True:
                # Now use new function to clip outliers
                mask = clip_outliers(log_bin_data, pp_step1.bestfit, mask)
                # Add clipped pixels to the original masked emission lines regions and repeat the fit
                mask &= mask0

            ################ 3 ##################
            # Third step - Only fit dust, no polynomials allowed
            #create a mask for dust specifically
            mask_dust = np.zeros_like(mask, dtype=bool)
            mask_dust[goodPixels_dust] = True
            mask_dust &= mask # Keep only pixels good in both masks

            # create the dust model
            Rv = 4.05
            Av_init = 4.05 * EBV_init            
            component_step3 = [0] *  np.prod(optimal_template_in.shape[1:])
            component_true_step3 = np.array(component_step3) == 0
            dust = [{"start": [Av_init], "bounds": [[0, 8]], "component": component_true_step3}]

            # fit only for dust
            pp_step3 = ppxf(
                optimal_template_in, 
                log_bin_data, 
                noise_new, 
                velscale, 
                lam=np.exp(logLam), 
                mask=mask_dust,
                degree=-1, 
                mdegree=-1,
                regul=0,
                fixed=fixed,
                vsyst=offset, 
                velscale_ratio=velscale_ratio,
                moments=nmoments, 
                start=start, 
                plot=False, 
                dust = dust, 
                component = component_step3, 
                quiet=True,
            )

            # Save dust values
            Av = pp_step3.dust[0]["sol"][0]
            EBV = Av/Rv
            component_step4 = [0]*ncomb
            component_true_step4 = np.array(component_step4) == 0

            # apply the dust correction if keyword is set:
            if config["SFH"]["DUST_CORR"] == True:
                dust_step4 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step4, 
                         "fixed":[True]}]
            else:
                dust_step4 = None
            
            ################ 4 ##################
            # Fourth step: Last Call PPXF - use all templates, get best-fit
            # (long fit: full template grid over the LMIN/LMAX range)
            pp = ppxf(
                templates,
                log_bin_data,
                noise_new,
                velscale,
                start,
                mask=mask,
                plot=False,
                quiet=True,
                moments=nmoments,
                vsyst=offset,
                degree=degree,
                mdegree=mdeg,
                regul = regul,
                fixed=fixed,
                lam=np.exp(logLam),
                velscale_ratio=velscale_ratio,
                component=component_step4,
                dust=dust_step4,                
            )

        #update goodpixels again
        goodPixels = pp.goodpixels

        #make spectral mask
        spectral_mask = np.full_like(log_bin_data, 0.0)
        spectral_mask[goodPixels] = 1.0

        # define goodPixels over SNR range for final SNR
        goodPixels_SNR_range = goodPixels[
            (np.exp(logLam[goodPixels]) >= config["READ_DATA"]["LMIN_SNR"])
            & (np.exp(logLam[goodPixels]) <= config["READ_DATA"]["LMAX_SNR"])]

        # Calculate the true S/N from the residual over the SNR MIN MAX range
        noise_est = robust_sigma(pp.galaxy[goodPixels_SNR_range] - pp.bestfit[goodPixels_SNR_range])
        snr_postfit = np.nanmedian(pp.galaxy[goodPixels_SNR_range]/noise_est)
        
        # Make the unconvolved optimal stellar template
        reshaped_templates = templates.reshape((templates.shape[0], ncomb)) #
        normalized_weights = pp.weights / np.sum( pp.weights ) #
        optimal_template   = np.zeros( reshaped_templates.shape[0] )

        for j in range(0, reshaped_templates.shape[1]):
            optimal_template = optimal_template + reshaped_templates[:,j]*normalized_weights[j]

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)
        weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum() # Take from 1D list to nD array (nAges, nMet, nAlpha)
        w_row   = np.array([np.reshape(weights, ncomb)])

        ################ 5 ##################
        # Fifth step: feature ("short") fit with <Age> and <[M/H]> fixed.
        # <Age> and <[M/H]> are the Step-4 weighted means (same definition as
        # the saved AGE/METAL), each snapped to the nearest grid value. Only the
        # nAlpha templates at that (age, [M/H]) cell are fit, to the galaxy
        # cropped to the SPEC_FSLF window, with the SPEC_MASK good pixels
        # cropped to the same window (not the Step-2 clip mask). EBV is fixed to
        # the Step-3 value and the kinematics to the Step-4 solution, so a
        # purely linear fit (linear=True) is safe. mdegree=-1, no additive
        # polynomial, never regularised.
        mean_long = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, 1)[0]

        age_axis = 10**logAge_grid[:, 0, 0]
        metal_axis = metal_grid[0, :, 0]
        a_idx = int(np.argmin(np.abs(age_axis - mean_long[0])))
        m_idx = int(np.argmin(np.abs(metal_axis - mean_long[1])))

        templates_short = np.ascontiguousarray(templates[:, a_idx, m_idx, :])  # (npix_temp, nAlpha)

        log_bin_data_short = log_bin_data[i_left:i_right + 1]
        noise_short = noise_new[i_left:i_right + 1]
        logLam_short = logLam[i_left:i_right + 1]

        # Kinematics fixed to the Step-4 solution
        start_short = np.array(pp.sol[:nmoments], dtype=float)
        fixed_short = [True] * nmoments

        component_step5 = [0] * nAlpha
        component_true_step5 = np.array(component_step5) == 0
        if config["SFH"]["DUST_CORR"] == True:
            dust_step5 = [{"start": [Av], "bounds": [[0, 8]], "component": component_true_step5,
                           "fixed": [True]}]
        else:
            dust_step5 = None

        mask_short = np.zeros(len(logLam_short), dtype=bool)
        mask_short[goodPixels_short] = True

        # Note: lam and lam_temp are both given, so vsyst must NOT be passed
        # (pPXF computes it, and asserts vsyst == 0 in that case).
        pp_step5 = ppxf(
            templates_short,
            log_bin_data_short,
            noise_short,
            velscale,
            start_short,
            mask=mask_short,
            plot=False,
            quiet=True,
            moments=nmoments,
            degree=-1,
            mdegree=-1,
            regul=0,
            fixed=fixed_short,
            linear=True,
            lam=np.exp(logLam_short),
            lam_temp=np.exp(logLam_template),
            velscale_ratio=velscale_ratio,
            component=component_step5,
            dust=dust_step5,
        )

        # Short-fit weights: only the fitted (age, [M/H]) cell is populated
        w_alpha = np.asarray(pp_step5.weights, dtype=float)
        w_alpha_sum = w_alpha.sum()
        if w_alpha_sum > 0:
            weights_short = np.zeros((nAges, nMetal, nAlpha))
            weights_short[a_idx, m_idx, :] = w_alpha / w_alpha_sum
            w_row_short = np.array([np.reshape(weights_short, ncomb)])
        else:
            # No alpha weight at all: flag as NaN rather than inventing a shape
            w_row_short = np.full((1, ncomb), np.nan)

        #plotting output
        if doplot == True:

            # check if figure  folder exists, otherwise
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

            #calculate mean age, metallicity, and alpha for the long (step 4) and short (step 5) fits
            mean_results_step4 = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, 1)
            mean_results_step5 = mean_agemetalalpha(w_row_short, 10**logAge_grid, metal_grid, alpha_grid, 1)

            # S/N of the residuals in the short window
            gp5 = pp_step5.goodpixels
            noise_est_short = robust_sigma(pp_step5.galaxy[gp5] - pp_step5.bestfit[gp5])
            snr_postfit_short = np.nanmedian(pp_step5.galaxy[gp5]/noise_est_short)
            
            #produce plots
            tmp_plot1 = plot_ppxf_sfh(pp_step1,np.exp(logLam),i,outfigFile_step1,snrCubevar=snr_prefit,
                                      snrResid=snr_Resid1,poly_type='mpoly')
            tmp_plot3 = plot_ppxf_sfh(pp_step3,np.exp(logLam),i,outfigFile_step3,snrCubevar=snr_prefit,
                                      snrResid=snr_Resid1,EBV=EBV,poly_type='mpoly')
            tmp_plot4 = plot_ppxf_sfh(pp,np.exp(logLam),i,outfigFile_step4,snrCubevar=snr_prefit,snrResid=snr_postfit,\
                             goodpixelsPre=goodPixels_preclip,mean_results=mean_results_step4,EBV=EBV,poly_type='mpoly')
            tmp_plot5 = plot_ppxf_sfh(pp_step5,np.exp(logLam_short),i,outfigFile_step5,snrCubevar=snr_prefit,
                                      snrResid=snr_postfit_short,mean_results=mean_results_step5,EBV=EBV,
                                      poly_type='mpoly',figsize=(8, 4.8))

        # add normalisation factor back in main results
        pp.bestfit = pp.bestfit * median_log_bin_data
        bestfit_short = pp_step5.bestfit * median_log_bin_data

        # Save additive & multiplicative Legendre polynomials (pp.a/mpoly is None if a/mdeg=-1)
        apoly = pp.apoly if pp.apoly is not None else np.ones(len(log_bin_data))
        mpoly = pp.mpoly if pp.mpoly is not None else np.ones(len(log_bin_data))

        return(
            pp.sol[:],
            w_row,
            pp.bestfit,
            optimal_template,
            formal_error,
            spectral_mask,
            snr_postfit,
            pp.chi2,
            EBV,
            mpoly,
            apoly,
            w_row_short,
            bestfit_short,
            pp_step5.chi2,
        )

    except Exception as e:
        # Handle any other type of exception
        import traceback
        tb = traceback.format_exc()
        logging.warning(f"run_ppxf failed for bin {i}: {e}\n{tb}")
        printStatus.warning(f"An error occurred in bin {i}: {e}")
        return( np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


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
    ppxf_bestfit_short,
    logLam,
    goodPixels,
    optimal_template,
    logLam_template,
    npix,
    spectral_mask,
    bin_data,
    snr_postfit,
    red_chi2,
    red_chi2_short,
    EBV,
    mpoly,
    apoly,
    mean_result_long,
    mean_result_short,
    w_row,
    w_row_short,
    logAge_grid,
    metal_grid,
    alpha_grid,
    velscale,
    logLam1,
    ncomb,
    nAges,
    nMetal,
    nAlpha,
    i_left,
    i_right,
):
    """ Save all results to disk. """

    history = ("SFH: Steps 1-4 full-grid long fit (AGE, METAL, ALPHA_LONG); "
               "Step 5 feature fit with fixed mean age and [M/H] (ALPHA)")

    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")

    # Table HDU with stellar kinematics
    # AGE/METAL: mean age and [M/H] of the Step-4 (long, full-grid) fit, which
    # are held fixed in the Step-5 feature fit. ALPHA: mean [alpha/Fe] of the
    # Step-5 (short, SPEC_FSLF window) fit. ALPHA_LONG: mean [alpha/Fe] of the
    # Step-4 fit, kept for comparing full-spectrum and feature fitting.
    columns = [
        fits.Column(name="AGE", format="D", array=mean_result_long[:, 0]),
        fits.Column(name="METAL", format="D", array=mean_result_long[:, 1]),
        fits.Column(name="ALPHA", format="D", array=mean_result_short[:, 2]),
        fits.Column(name="ALPHA_LONG", format="D", array=mean_result_long[:, 2]),
    ]

    # If FIXED is False, add kinematic columns (from the Step-4 long fit)
    if config["SFH"]["FIXED"] == False:
        # Define kinematic columns (V and SIGMA)
        kinematic_columns = [
            fits.Column(name=name, format="D", array=ppxf_result[:, i])
            for i, name in enumerate(["V", "SIGMA"])  # Loop over V and SIGMA
        ]
        # Add kinematic columns to the main list
        columns.extend(kinematic_columns)

        # Add higher-order kinematic columns (H3, H4, H5, H6) if they exist
        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(ppxf_result[:, i+2]) != 0:  # Check if the column exists
                columns.append(fits.Column(name=name, format="D", array=ppxf_result[:, i+2]))

        # Define formal error columns for kinematic parameters
        error_columns = [
            fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i])
            for i, name in enumerate(["V", "SIGMA"])  # Loop over V and SIGMA
        ]
        # Add formal error columns to the main list
        columns.extend(error_columns)

        # Add formal error columns for higher-order kinematic parameters
        for i, name in enumerate(["H3", "H4", "H5", "H6"]):
            if np.any(formal_error[:, i+2]) != 0:  # Check if the column exists
                columns.append(fits.Column(name=f"FORM_ERR_{name}", format="D", array=formal_error[:, i+2]))

    # Add SNR_POSTFIT column to the main list (Step 4)
    columns.append(fits.Column(name="SNR_POSTFIT", format="D", array=snr_postfit[:]))

    # Add Chi2 columns to the main list (Step 4 long fit and Step 5 short fit)
    columns.append(fits.Column(name="RED_CHI2", format="D", array=red_chi2[:]))
    columns.append(fits.Column(name="RED_CHI2_SHORT", format="D", array=red_chi2_short[:]))

    # Add E(B-V) derived from pPXF 0th step with reddening but no polynomials
    columns.append(fits.Column(name="EBV", format="D", array=EBV[:]))
    
    # Create the HDUs
    priHDU = fits.PrimaryHDU()
    priHDU.header['HISTORY'] = history
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="SFH")

    # Save the configuration to the headers
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh.fits")
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE WEIGHTS AND GRID
    # WEIGHTS: Step-4 (long, full-grid) cube. GRID: template grid.
    # WEIGHTS_SHORT (last extension): Step-5 (short) cube, same flattened
    # (nAges, nMetal, nAlpha) format, only the fitted (age, [M/H]) cell is
    # populated. Its column is called WEIGHTS so the same reader code works.
    # Define the output file
    outfits_sfh = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_sfh_weights.fits"
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    priHDU.header['HISTORY'] = history

    # Table HDU with weights
    cols_weights = [fits.Column(name="WEIGHTS", format=str(w_row.shape[1]) + "D", array=w_row)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_weights), name="WEIGHTS")

    # Reshape the grids
    logAge_row, metal_row, alpha_row = map(np.reshape, [logAge_grid, metal_grid, alpha_grid], [ncomb]*3)

    # Table HDU with grids
    cols_grid = [fits.Column(name=name, format="D", array=array) 
                 for name, array in zip(["LOGAGE", "METAL", "ALPHA"], [logAge_row, metal_row, alpha_row])]
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_grid), name="GRID")

    # Table HDU with the short-fit weights (last extension)
    cols_weights_short = [fits.Column(name="WEIGHTS", format=str(w_row_short.shape[1]) + "D", array=w_row_short)]
    shortHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols_weights_short), name="WEIGHTS_SHORT")

    # Create HDU list and write to file
    HDUList = fits.HDUList([_auxiliary.saveConfigToHeader(hdu, config["SFH"]) for hdu in [priHDU, dataHDU, gridHDU, shortHDU]])
    HDUList.writeto(outfits_sfh, overwrite=True)

    # Set additional header values
    for name, value in zip(["NAGES", "NMETAL", "NALPHA"], [nAges, nMetal, nAlpha]):
        fits.setval(outfits_sfh, name, value=value)

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_weights.fits"
    )
    logging.info("Wrote: " + outfits_sfh)

    # ========================
    # SAVE BESTFIT
    outfits_sfh = (
        os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"])
        + "_sfh_bestfit.fits"
    )
    printStatus.running("Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits")

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    priHDU.header['HISTORY'] = history

    # Table HDU with SFH bestfit (Step 4, full range)
    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=ppxf_bestfit ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = "BESTFIT"

    # Table HDU with SFH logLam
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = "LOGLAM"

    # Table HDU with template wavelength grid
    cols = []
    cols.append(fits.Column(name="LOGLAM_TEMPLATE", format="D", array=logLam_template))
    logLamTempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamTempHDU.name = "LOGLAM_TEMPLATE"

    # Table HDU with observed spectra
    cols = []
    cols.append(fits.Column(name="SPEC", format=str(npix) + "D", array=bin_data.T))
    specHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    specHDU.name = "SPEC"

    # Table HDU with SFH goodpixels
    cols = []
    cols.append(fits.Column(name="GOODPIX", format="J", array=goodPixels))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = "GOODPIX"

    # Table HDU with 3 sigma clipped regions
    cols = []
    cols.append(fits.Column(name="GOODPIX_CLN", format=str(spectral_mask.shape[1]) + "D", array=spectral_mask))
    goodpixClnHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixClnHDU.name = "GOODPIX_CLN"

    # Table HDU with multiplicative Legendre polynomials
    cols = []
    cols.append(fits.Column(name="MPOLY", format=str(mpoly.shape[1]) + "D", array=mpoly))
    mpolyHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    mpolyHDU.name = "MPOLY"

    # Table HDU with additive Legendre polynomials
    cols = []
    cols.append(fits.Column(name="APOLY", format=str(apoly.shape[1]) + "D", array=apoly))
    apolyHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    apolyHDU.name = "APOLY"

    # Table HDU with per-bin optimal templates
    cols = []
    cols.append(fits.Column(name="OPTIMAL_TEMPLATES", format=str(optimal_template.shape[1]) + "D", array=optimal_template))
    optHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    optHDU.name = "OPTIMAL_TEMPLATES"

    # Table HDU with the Step-5 (short window) bestfit. The window is a
    # contiguous crop of LOGLAM: IPIXLO / IPIXHI are its first / last pixel
    # (0-based, inclusive).
    npix_short = ppxf_bestfit_short.shape[1]
    cols = []
    cols.append(fits.Column(name="BESTFIT", format=str(npix_short) + "D", array=ppxf_bestfit_short))
    bestfitShortHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    bestfitShortHDU.name = "BESTFIT_SHORT"

    # Table HDU with the Step-5 (short window) logLam: logLam[i_left:i_right+1],
    # i.e. the same crop as BESTFIT_SHORT / IPIXLO / IPIXHI, saved directly for
    # convenience.
    cols = []
    cols.append(fits.Column(name="LOGLAM_SHORT", format="D", array=logLam[i_left:i_right + 1]))
    logLamShortHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamShortHDU.name = "LOGLAM_SHORT"

    # Create HDU list and write to file
    priHDU = _auxiliary.saveConfigToHeader(priHDU, config["SFH"])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config["SFH"])
    logLamHDU = _auxiliary.saveConfigToHeader(logLamHDU, config["SFH"])
    logLamTempHDU = _auxiliary.saveConfigToHeader(logLamTempHDU, config["SFH"])
    specHDU = _auxiliary.saveConfigToHeader(specHDU, config["SFH"])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config["SFH"])
    goodpixClnHDU = _auxiliary.saveConfigToHeader(goodpixClnHDU, config["SFH"])
    mpolyHDU = _auxiliary.saveConfigToHeader(mpolyHDU, config["SFH"])
    optHDU = _auxiliary.saveConfigToHeader(optHDU, config["SFH"])
    bestfitShortHDU = _auxiliary.saveConfigToHeader(bestfitShortHDU, config["SFH"])
    bestfitShortHDU.header["IPIXLO"] = (int(i_left), "first pixel of window in LOGLAM (0-based)")
    bestfitShortHDU.header["IPIXHI"] = (int(i_right), "last pixel of window in LOGLAM (0-based, inclusive)")
    logLamShortHDU = _auxiliary.saveConfigToHeader(logLamShortHDU, config["SFH"])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, logLamTempHDU, specHDU, goodpixHDU, goodpixClnHDU, mpolyHDU, optHDU, bestfitShortHDU, logLamShortHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh, "VELSCALE", value=velscale)
    fits.setval(outfits_sfh, "CRPIX1", value=1.0)
    fits.setval(outfits_sfh, "CRVAL1", value=logLam1[0])
    fits.setval(outfits_sfh, "CDELT1", value=logLam1[1] - logLam1[0])

    printStatus.updateDone(
        "Writing: " + config["GENERAL"]["RUN_ID"] + "_sfh_bestfit.fits"
    )
    logging.info("Wrote: " + outfits_sfh)



def extractStarFormationHistories(config):
    """
    Starts the computation of non-parametric star-formation histories with
    pPXF.  A spectral template library sorted in a three-dimensional grid of
    age, metallicity, and alpha-enhancement is loaded.  Emission-subtracted
    spectra are used for the fit. An according emission-line mask is
    constructed. The stellar kinematics can or cannot be fixed to those obtained
    with a run of unregularized pPXF and the analysis started.  Results are
    saved to disk and the plotting routines called.

    This wrapper (fix <Age> and <[M/H]>) additionally runs a feature-focused
    fit of the nAlpha templates at the cell nearest the Step-4 mean age and
    [M/H] over the SFH.SPEC_FSLF window. See the module docstring.
    Args:
    - config: dictionary containing configuration parameters
    """

    # Read LSF information
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config, "SFH")

    # Prepare template library
    # Open the HDF5 file
    with h5py.File(os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5", 'r') as f:
        # Read the VELSCALE attribute from the file
        velscale = f.attrs["VELSCALE"]
        
    velscale_ratio = 2

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
        config["SFH"]["LMIN"],
        config["SFH"]["LMAX"],
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

    # check that template wavelength is larger than requested fit range otherwise stop
    if (lamRange_temp[0] >= config["SFH"]["LMIN"]) or (lamRange_temp[1] <= config["SFH"]["LMAX"]):
        logging.info("Template wavelength range needs to be larger than fitting range, exiting")
        printStatus.warning(
            "Template wavelength range needs to be larger than fitting range, exiting"
        )
        return

    # Define file paths
    gas_cleaned_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + '_gas_cleaned_'+config["GAS"]["LEVEL"].lower()+'.fits'
    bin_spectra_file = os.path.join(config["GENERAL"]["OUTPUT"], config["GENERAL"]["RUN_ID"]) + "_bin_spectra.hdf5"

    # Check if emission-subtracted spectra file exists
    if (config["SFH"]["SPEC_EMICLEAN"] == True) and os.path.isfile(gas_cleaned_file):
        logging.info(f"Using emission-subtracted spectra at {gas_cleaned_file}")
        printStatus.done("Using emission-subtracted spectra")
        # Open the FITS file
        with fits.open(gas_cleaned_file, mem_map=True) as hdul:
            # Read the LOGLAM data from the file
            logLam = hdul[2].data["LOGLAM"]

            # Select the indices where the wavelength is within the specified range
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config["SFH"]["LMIN"], np.exp(logLam) < config["SFH"]["LMAX"]))[0]

            # Read the SPEC and ESPEC data from the file, only for the selected indices
            bin_data = hdul[1].data["SPEC"].T[idx_lam, :]
            bin_err = hdul[1].data["ESPEC"].T[idx_lam, :]
            logLam = logLam[idx_lam]
            nbins = bin_data.shape[1]
            npix = bin_data.shape[0]
    else:
        logging.info(f"Using regular spectra without any emission-correction at {bin_spectra_file}")
        printStatus.done("Using regular spectra without any emission-correction")
        with h5py.File(bin_spectra_file, 'r') as f:
            # Read the LOGLAM data from the file
            logLam = f["LOGLAM"][:]

            # Select the indices where the wavelength is within the specified range
            idx_lam = np.where(np.logical_and(np.exp(logLam) > config["SFH"]["LMIN"], np.exp(logLam) < config["SFH"]["LMAX"]))[0]

            # Read the SPEC and ESPEC data from the file, only for the selected indices
            bin_data = f["SPEC"][idx_lam, :]
            bin_err = f["ESPEC"][idx_lam, :]
            logLam = logLam[idx_lam]

    # Define additional variables
    nbins = bin_data.shape[1]
    npix = bin_data.shape[0]
    ubins = np.arange(nbins)
    dv = (np.log(lamRange_temp[0]) - logLam[0])*C

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    
    #check what type of noise should be passed on:
    if config["SFH"]["NOISE"] == 'variance': # use noise from cube 
        noise = bin_err  # already converted to noise, i.e. sqrt(variance)
    elif config["SFH"]["NOISE"] == 'constant': # use constant noise
        noise  = np.ones((npix,nbins))
        # while constant, the noise does need to be scaled to match the bin_err
        med_bin_err = np.nanmedian(bin_err, axis=0)
        noise *= med_bin_err        

    # No Monte Carlo realisations in this wrapper
    if config["SFH"].get("MC_PPXF", 0) > 0:
        logging.info("SFH.MC_PPXF is set but MC realisations are not implemented in this wrapper; ignoring it.")
        printStatus.warning("SFH.MC_PPXF is ignored: no MC realisations in this wrapper")

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config["SFH"]["FIXED"]:
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
        # Set fixed option to False and use initial guess from Config-file
        fixed = None
        start = np.zeros((nbins, config["SFH"]["MOM"]))
        for i in range(nbins):
            if config["SFH"]["MOM"] == 2:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"]])
            elif config["SFH"]["MOM"] == 4:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"],0.0,0.0])
            elif config["SFH"]["MOM"] == 6:
                start[i, :] = np.array([0.0, config["KIN"]["SIGMA"],0.0,0.0,0.0,0.0])

    # Define goodpixels
    goodPixels_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)

    # Check if a premask for the first step has been defined
    if 'SPEC_PREMASK' in config["SFH"]:
        goodPixels_premask_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_PREMASK"], logLam)
    else:
        goodPixels_premask_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)
    
    if 'SPEC_DUSTMASK' in config["SFH"]:
        goodPixels_dust_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_DUSTMASK"], logLam)
    else:
        goodPixels_dust_sfh = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_MASK"], logLam)
    
    # Feature-fit (short) window: bounding extent of the SFH.SPEC_FSLF good pixels
    # (SFH.SPEC_MASK if SPEC_FSLF is not set), as a contiguous crop of logLam.
    if 'SPEC_FSLF' in config["SFH"]:
        goodPixels_fslf_full = _auxiliary.spectralMasking(config, config["SFH"]["SPEC_FSLF"], logLam)
    else:
        goodPixels_fslf_full = goodPixels_sfh
    goodPixels_fslf_full = np.asarray(goodPixels_fslf_full)
    if len(goodPixels_fslf_full) == 0:
        raise ValueError("The SFH.SPEC_FSLF window contains no good pixels")
    i_left = int(np.min(goodPixels_fslf_full))
    i_right = int(np.max(goodPixels_fslf_full))
    npix_short = i_right - i_left + 1

    # Short-fit pixel mask: the SPEC_MASK good pixels cropped to the window
    goodPixels_sfh_arr = np.asarray(goodPixels_sfh)
    goodPixels_sfh_short = goodPixels_sfh_arr[
        (goodPixels_sfh_arr >= i_left) & (goodPixels_sfh_arr <= i_right)] - i_left
    if len(goodPixels_sfh_short) == 0:
        raise ValueError("No SFH.SPEC_MASK good pixels fall inside the SFH.SPEC_FSLF window")

    # Check if plot keyword is set:
    doplot = config["SFH"].get("PLOT", False)

    # define the regularisation value 
    sfh_cfg = config["SFH"]
    if "REGUL" in sfh_cfg:
        regul = sfh_cfg["REGUL"]
    elif "REGUL_ERR" in sfh_cfg:
        regul_err = sfh_cfg["REGUL_ERR"]
        regul = 0.0 if regul_err == 0 else 1.0 / regul_err
    else:
        raise KeyError("Either SFH.REGUL or SFH.REGUL_ERR must be set")

    # check if ADEGREE is set, otherwise make it -1
    if 'ADEG' in config["SFH"]:
        degree = config["SFH"]["ADEG"]
    else:
        degree = -1

    # Define output arrays
    ppxf_result = np.zeros((nbins,6    ))
    w_row = np.zeros((nbins,ncomb))
    ppxf_bestfit = np.zeros((nbins,npix))
    optimal_template = np.zeros((nbins,templates.shape[0]))
    formal_error = np.zeros((nbins,6))
    spectral_mask = np.zeros((nbins,bin_data.shape[0]))
    snr_postfit = np.zeros(nbins)
    red_chi2 = np.zeros(nbins)
    EBV = np.zeros(nbins)
    mpoly = np.zeros((nbins, bin_data.shape[0]))
    apoly = np.zeros((nbins, bin_data.shape[0]))

    # Define output arrays of the Step-5 (short) fit
    w_row_short = np.zeros((nbins,ncomb))
    ppxf_bestfit_short = np.zeros((nbins,npix_short))
    red_chi2_short = np.zeros(nbins)

    # ====================
    # If OPT_TEMP keyword set to 'galaxy_single' or 'galaxy_set' then
    # run PPXF once on combined mean spectrum to get a single or optimal template set

    if (config["SFH"]["OPT_TEMP"] == "galaxy_single") or (config["SFH"]["OPT_TEMP"] == "galaxy_set"):
        comb_spec = np.nanmean(bin_data[:,:],axis=1)
        comb_espec = np.nanmean(bin_err[:,:],axis=1)

        optimal_template_out, optimal_template_set = run_ppxf_firsttime(
            templates,
            comb_spec ,
            comb_espec,
            velscale,
            start[0,:],
            goodPixels_premask_sfh,
            config["SFH"]["MOM"],
            offset,
            degree,
            config["SFH"]["MDEG"],
            regul,
            velscale_ratio,
            ncomb,
        )

        # now define the optimal template that we'll use throughout
        if config["SFH"]["OPT_TEMP"] == 'galaxy_single':
            optimal_template_comb = optimal_template_out # single template
        if config["SFH"]["OPT_TEMP"] == 'galaxy_set':
            optimal_template_comb = optimal_template_set # selected set  from total galaxy fit
    else:
        optimal_template_comb = templates # all templates
 
    # ====================
    EBV_init = 0.1 # PHANGS value initial guess

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
        templates_filename_memmap = os.path.join(
            memmap_folder, "templates_memmap.tmp"
        )
        dump(templates, templates_filename_memmap)
        templates = load(templates_filename_memmap, mmap_mode="r")

        if config["SFH"]["OPT_TEMP"] == "default":
            optimal_template_comb = templates
        else:
            opt_temp_file = os.path.join(
                memmap_folder, "optimal_template_memmap.tmp"
            )
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
                    bin_data[:,i],
                    noise[:,i],
                    velscale,
                    start[i,:],
                    goodPixels_premask_sfh,
                    goodPixels_dust_sfh,
                    goodPixels_sfh,
                    config["SFH"]["MOM"],
                    offset,
                    degree,
                    config["SFH"]["MDEG"],
                    regul,
                    config["SFH"]["DOCLEAN"],
                    fixed,
                    velscale_ratio,
                    npix,
                    ncomb,
                    nbins,
                    i,
                    optimal_template_comb,
                    EBV_init,
                    logLam,
                    logAge_grid,
                    metal_grid,
                    alpha_grid,
                    config,
                    doplot,
                    i_left,
                    i_right,
                    goodPixels_sfh_short,
                    logLam_template,
                    nAges,
                    nMetal,
                    nAlpha,
                )
                results.append(result)
            return results

        # Use joblib to parallelize the work
        max_nbytes = "1M" # max array size before memory mapping is triggered
        chunk_size = max(1, nbins // (config["GENERAL"]["NCPU"] * 10))
        chunks = [range(i, min(i + chunk_size, nbins)) for i in range(0, nbins, chunk_size)]
        parallel_configs = {
            "n_jobs": config["GENERAL"]["NCPU"],
            "max_nbytes": None,
            "return_as": "generator",
        }

        #ppxf_tmp = list(tqdm(Parallel(**parallel_configs)(delayed(worker)(chunk, templates) for chunk in chunks),
        #                total=len(chunks), desc="Processing chunks", ascii=" #", unit="chunk"))

        with Parallel(**parallel_configs) as parallel:
            ppxf_tmp = list(tqdm(
                parallel(delayed(worker)(chunk, templates) for chunk in chunks),
                total=len(chunks), desc="Processing chunks",
                ascii=" #", unit="chunk"
            ))

        # Flatten the results
        ppxf_tmp = [result for chunk_results in ppxf_tmp for result in chunk_results]

        # Unpack results
        for i in range(0, nbins):
            ppxf_result[i,:config["SFH"]["MOM"]] = ppxf_tmp[i][0]
            w_row[i,:] = ppxf_tmp[i][1]
            ppxf_bestfit[i,:] = ppxf_tmp[i][2]
            optimal_template[i,:] = ppxf_tmp[i][3]
            formal_error[i,:config["SFH"]["MOM"]] = ppxf_tmp[i][4]
            spectral_mask[i,:] = ppxf_tmp[i][5]
            snr_postfit[i] = ppxf_tmp[i][6]
            red_chi2[i] = ppxf_tmp[i][7]
            EBV[i] = ppxf_tmp[i][8]
            mpoly[i,:] = ppxf_tmp[i][9]
            apoly[i,:] = ppxf_tmp[i][10]
            w_row_short[i,:] = ppxf_tmp[i][11]
            ppxf_bestfit_short[i,:] = ppxf_tmp[i][12]
            red_chi2_short[i] = ppxf_tmp[i][13]

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
                ppxf_result[i,:config["SFH"]["MOM"]],
                w_row[i,:],
                ppxf_bestfit[i,:],
                optimal_template[i,:],
                formal_error[i,:config["SFH"]["MOM"]],
                spectral_mask[i,:],
                snr_postfit[i],
                red_chi2[i],
                EBV[i],
                mpoly[i,:],
                apoly[i,:],
                w_row_short[i,:],
                ppxf_bestfit_short[i,:],
                red_chi2_short[i],
            ) = run_ppxf(
                templates,
                bin_data[:,i],
                noise[:,i],
                velscale,
                start[i,:],
                goodPixels_premask_sfh,
                goodPixels_dust_sfh,
                goodPixels_sfh,
                config["SFH"]["MOM"],
                offset,
                degree,
                config["SFH"]["MDEG"],
                regul,
                config["SFH"]["DOCLEAN"],
                fixed,
                velscale_ratio,
                npix,
                ncomb,
                nbins,
                i,
                optimal_template_comb,
                EBV_init,
                logLam,
                logAge_grid,
                metal_grid,
                alpha_grid,
                config,
                doplot,
                i_left,
                i_right,
                goodPixels_sfh_short,
                logLam_template,
                nAges,
                nMetal,
                nAlpha,
            )
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
    idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]

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

    # Calculate mean age, metallicity and alpha of the long (Step 4) and short (Step 5) fits
    mean_results_long = mean_agemetalalpha(
        w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins
    )
    mean_results_short = mean_agemetalalpha(
        w_row_short, 10**logAge_grid, metal_grid, alpha_grid, nbins
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
        ppxf_bestfit_short,
        logLam,
        goodPixels_sfh,
        optimal_template,
        logLam_template,
        npix,
        spectral_mask,
        bin_data,
        snr_postfit,
        red_chi2,
        red_chi2_short,
        EBV,
        mpoly,
        apoly,
        mean_results_long,
        mean_results_short,
        w_row,
        w_row_short,
        logAge_grid,
        metal_grid,
        alpha_grid,
        velscale,
        logLam,
        ncomb,
        nAges,
        nMetal,
        nAlpha,
        i_left,
        i_right,
    )

    if config["GENERAL"]["PARALLEL"] == True:
        templates._mmap.close()
        bin_data._mmap.close()
        noise._mmap.close()

        if config["SFH"]["OPT_TEMP"] != "default":
            optimal_template_comb._mmap.close()

        shutil.rmtree(memmap_folder)

    # Return
    return None