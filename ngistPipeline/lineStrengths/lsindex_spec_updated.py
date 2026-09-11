#!/usr/bin/env python
import optparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy
from astropy.io import ascii, fits


# ===============================================================================
#
# LSINDEX_SPEC
#
#  This function computes the Lick indices of a set of input spectra
#
#  NOTE: Input spectra is assumed to be in Angstroms.
#
# Jesus Falcon-Barroso, IAC, August 2016
# Modified to include integral EW calculation, Michelle Ding, Jan 2026

# ===============================================================================

def printProgress(iteration, total, prefix="", suffix="", decimals=2, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = "#" * filledLength + "-" * (barLength - filledLength)
    sys.stdout.write("\r%s [%s] %s%s %s\r" % (prefix, bar, percents, "%", suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")


# ==============================================================================
def load_inputlist(inlist):
    # Reading inputlist
    data = ascii.read(inlist, comment="\s*#")
    names = data["col1"]
    redshift = data["col2"]
    err_redshift = data["col3"]

    return names, redshift, err_redshift

# ==============================================================================
#
# FUNCTION: flux_density_integral() - Updated method
#
def flux_density_integral(wave_full, flux, wave1, wave2):
    # Create mask for pixels in or near the range
    mask = (wave_full >= wave1 - 10) & (wave_full <= wave2 + 10)
    
    if not numpy.any(mask):
        return 0.0
    
    wave_region = wave_full[mask]
    flux_region = flux[mask]
    
    # Interpolate flux at exact boundaries
    flux_at_wave1 = numpy.interp(wave1, wave_region, flux_region)
    flux_at_wave2 = numpy.interp(wave2, wave_region, flux_region)
    
    # Build integration arrays including boundaries
    wave_integrate = numpy.concatenate([[wave1], wave_region[
        (wave_region > wave1) & (wave_region < wave2)], [wave2]])
    flux_integrate = numpy.concatenate([[flux_at_wave1], flux_region[
        (wave_region > wave1) & (wave_region < wave2)], [flux_at_wave2]])
    
    # Integrate and return average
    integrated = numpy.trapz(flux_integrate, wave_integrate)
    return integrated / (wave2 - wave1)

# ==============================================================================
#
# FUNCTION: calc_index_integral() - New integral-based method
#
def calc_index_integral(bands, name, ll, counts, plot, plot_dir=None, bin_id=None, run_id=None):
    # Calculate continuum fluxes
    continuum_blue_flux = flux_density_integral(ll, counts, bands[0], bands[1])
    continuum_blue_midpoint = 0.5 * (bands[0] + bands[1])
    
    continuum_red_flux = flux_density_integral(ll, counts, bands[4], bands[5])
    continuum_red_midpoint = 0.5 * (bands[4] + bands[5])
    
    # Calculate continuum slope
    slope = (continuum_red_flux - continuum_blue_flux) / (continuum_red_midpoint - continuum_blue_midpoint)
    
    # Get feature region with proper boundaries
    mask = (ll >= bands[2] - 10) & (ll <= bands[3] + 10)
    wave_region = ll[mask]
    flux_region = counts[mask]
    
    # Interpolate at feature boundaries
    flux_at_feat_start = numpy.interp(bands[2], wave_region, flux_region)
    flux_at_feat_end = numpy.interp(bands[3], wave_region, flux_region)
    
    # Build feature arrays
    feature_wave = numpy.concatenate([[bands[2]], wave_region[
        (wave_region > bands[2]) & (wave_region < bands[3])], [bands[3]]])
    feature_flux = numpy.concatenate([[flux_at_feat_start], flux_region[
        (wave_region > bands[2]) & (wave_region < bands[3])], [flux_at_feat_end]])
    
    # Calculate continuum at each wavelength in feature
    continuum_at_feature = continuum_blue_flux + slope * (feature_wave - continuum_blue_midpoint)
    
    # Calculate equivalent width
    if bands[6] == 1.0:
        # Atomic index
        ind = numpy.trapz(1 - feature_flux / continuum_at_feature, feature_wave)
    elif bands[6] == 2.0:
        # Molecular index
        feature_integral = numpy.trapz(feature_flux, feature_wave)
        continuum_integral = numpy.trapz(continuum_at_feature, feature_wave)
        ind = -2.5 * numpy.log10(feature_integral / continuum_integral)
    else:
        ind = numpy.nan
    
    # Plotting
    if plot > 0:
        # Create wavelength mask for zoomed region
        mask = (ll >= bands[0] - 5) & (ll <= bands[5] + 5)
        
        fig = plt.figure(figsize=(7, 4))
        plt.plot(ll[mask], counts[mask], color='black', linewidth=1.5)
        
        # Draw the continuum line across the whole index region
        cont_left  = (slope * (bands[0] - continuum_blue_midpoint)) + continuum_blue_flux
        cont_right = (slope * (bands[5] - continuum_blue_midpoint)) + continuum_blue_flux
        plt.plot([bands[0], bands[5]], [cont_left, cont_right], color="red", linewidth=2, label="Continuum")

        # Shade pseudo-continua and feature
        plt.axvspan(bands[0], bands[1], color="skyblue", alpha=0.5, label="Pseudo-continua")
        plt.axvspan(bands[2], bands[3], color="grey", alpha=0.2, label="Central Bandpass")
        plt.axvspan(bands[4], bands[5], color="skyblue", alpha=0.5)

        # Labels and legend
        plt.xlabel('Wavelength [$\\AA$]')
        plt.ylabel('Flux')
        plt.title(f"BIN={bin_id} : {name}")

        handles, labels_legend = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        plt.legend(by_label.values(), by_label.keys(),
                title=f"EW = {float(ind):.4f} $\\AA$", loc="lower right")
        
        plt.tight_layout()
        
        # Save or close the figure
        if plot_dir is not None:
            import os
            
            # Create index-specific subdirectory: plot_dir/INDEX_NAME/
            index_plot_dir = os.path.join(plot_dir, name)
            if not os.path.exists(index_plot_dir):
                os.makedirs(index_plot_dir)
            
            # Create filename: RUN_ID_ls_INDEX_NAME_bin_XXXX.png
            if run_id is not None and bin_id is not None:
                filename = f"{run_id}_ls_{name}_bin_{bin_id:04d}.png"
            elif bin_id is not None:
                filename = f"bin_{bin_id:04d}.png"
            else:
                filename = f"{name}.png"
            
            filepath = os.path.join(index_plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.close(fig)

    return ind


# ==============================================================================
# purpose : Measure line-strength indices
#
# input : ll    - wavelength vector; assumed to be in *linear steps*
#         flux  - counts as a function of wavelength
#         noise - noise spectrum
#         z, z_err - redshift and error (in km/s)
#         lickfile - file listing the index definitions
#
# keywords  debug  - more than 0 gives some basic info
#           plot   - plot spectra
#           sims   - number of simulations for the errors (default: 100)
#
# output : names       - index names
#          index       - index values
#          index_error - index error values
#
# author : J. Falcon-Barroso
#
# version : 1.0  IAC (08/07/16) A re-coding of H. Kuntschner's IDL routine into python
# version : 2.0  Modified to include integral-based calculation method
# ==============================================================================
def lsindex(ll, flux_in, noise, z, lickfile, plot=0, sims=0, z_err=0,
            plot_dir=None, bin_id=None, run_id=None):
    # Deredshift spectrum to rest wavelength
    dll = (ll) / (z + 1.0)

    # Rebin to linear step
    flux = flux_in

    # Read index definition table
    tab = ascii.read(lickfile, comment="\s*#")
    names = tab["names"]
    bands = numpy.zeros((7, len(names)))
    bands[0, :] = tab["b1"]
    bands[1, :] = tab["b2"]
    bands[2, :] = tab["b3"]
    bands[3, :] = tab["b4"]
    bands[4, :] = tab["b5"]
    bands[5, :] = tab["b6"]
    bands[6, :] = tab["b7"]

    # Measure line indices
    num_ind = len(bands[0, :])
    index = numpy.zeros(num_ind)
    for k in range(num_ind):  # loop through all indices
        # check whether the wavelength range is o.k.
        if (dll[0] <= bands[0, k]) and (dll[len(dll) - 1] >= bands[5, k]):
            # calculate index value
            index[k] = calc_index_integral(bands[:, k], names[k], dll, flux, plot,
                                           plot_dir=plot_dir, bin_id=bin_id, run_id=run_id)
        else:
            # index outside wavelength range
            index[k] = numpy.nan

    # Calculate errors
    index_error = numpy.zeros(num_ind, dtype="D")
    index_error[:] = numpy.nan
    index_noise = numpy.zeros([num_ind, sims], dtype="D")

    if sims > 0:
        # Create redshift and sigma errors
        dz = numpy.random.randn(sims) * z_err

        # Loop through the simulations
        for i in range(sims):
            # resample spectrum according to noise
            ran = numpy.random.normal(0.0, 1.0, len(dll))
            flux_n = flux + ran * noise

            # loop through all indices
            for k in range(num_ind):
                # shift bands according to redshift error
                sz = z + dz[i]
                dll = ll / (sz + 1.0)
                bands2 = bands[:, k]
                if (dll[0] <= bands2[0]) and (dll[len(dll) - 1] >= bands2[5]):
                    index_noise[k, i] = calc_index_integral(
                        bands2, names[k], dll, flux_n, 0,
                        plot_dir=None, bin_id=None, run_id=None)
                else:
                    # index outside wavelength range
                    index_noise[k, i] = numpy.nan

        # Get STD of distribution (index error)
        index_error = numpy.std(index_noise, axis=1)

    return names, index, index_error, index_noise


# ==============================================================================
if __name__ == "__main__":
    os.system("clear")
    warnings.filterwarnings("ignore")
    print("========================")
    print("= Running LSINDEX_SPEC =")
    print("========================")
    print("")

    # Capturing the command line arguments
    parser = optparse.OptionParser(
        usage="%prog -i inputlist -l lickfile -o outfits -n nsims"
    )
    parser.add_option(
        "-i",
        "--inputlist",
        dest="inputlist",
        type="string",
        default="../config_files/miles_ku.inputlist",
        help="List of input spectra, redshift and err_redshift",
    )
    parser.add_option(
        "-l",
        "--lickfile",
        dest="lickfile",
        type="string",
        default="../config_files/lick_bands.conf",
        help="Lick file with index definitions",
    )
    parser.add_option(
        "-o",
        "--outfits",
        dest="outfits",
        type="string",
        default="../results/lick_indices.fits",
        help="Name of output FITS table with results",
    )
    parser.add_option(
        "-n",
        "--nsims",
        dest="nsims",
        type="int",
        default="0",
        help="Number of MC simulations for errors",
    )
    parser.add_option(
        "-p",
        "--plot",
        dest="plot",
        type="int",
        default="0",
        help="Plotting or not [0/1]",
    )

    (options, args) = parser.parse_args()
    inputlist = options.inputlist
    lickfile = options.lickfile
    outfits = options.outfits
    nsims = options.nsims
    plot_flag = options.plot

    # Getting the list of FITS files to process
    print("# Loading inputlist: " + inputlist)
    inlist, redshift, err_redshift = load_inputlist(inputlist)
    nfiles = len(inlist)
    print("- " + str(nfiles) + " files found")
    print("")

    # Computing the magnitudes for each input FITS file
    print("# Computing indices...")
    root = []
    for i in range(nfiles):
        # Opening the FITS file
        hdu = fits.open(inlist[i])
        flux = hdu[0].data
        npix = len(flux)
        crpix = hdu[0].header["CRPIX1"]
        crval = hdu[0].header["CRVAL1"]
        cdelt = hdu[0].header["CDELT1"]
        wave = ((numpy.arange(npix) + 1.0) - crpix) * cdelt + crval
        root = numpy.append(root, os.path.basename(inlist[i]))

        # Computing the indices
        names, indices, errors, _ = lsindex(
            wave,
            flux,
            flux * 0.1,
            redshift[i],
            lickfile,
            plot=plot_flag,
            sims=nsims,
            z_err=err_redshift[i],
        )

        if i == 0:
            outls = numpy.zeros((len(names), nfiles))
            outls_err = numpy.zeros((len(names), nfiles))

        outls[:, i] = indices
        outls_err[:, i] = errors

        printProgress(i + 1, nfiles, prefix=" ", suffix="Complete", barLength=50)

    # Saving the results to a FITS table
    if os.path.exists(outfits):
        os.remove(outfits)
    print("# Results will be stored in the FITS table: " + outfits)
    print("")
    cols = []
    cols.append(fits.Column("Files", format="100A", array=root))
    ndim = len(names)
    for i in range(ndim):
        cols.append(fits.Column(name=names[i], format="D", array=outls[i, :]))
        cols.append(
            fits.Column(name="ERR_" + names[i], format="D", array=outls_err[i, :])
        )
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.writeto(outfits)

    print("# DONE!")