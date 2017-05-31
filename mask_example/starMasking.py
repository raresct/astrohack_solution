import numpy as np
from astropy.io import fits
import subprocess

# Test case
name = "1237645879578460255-g"

# Read in csv and write to .fits file
input = np.genfromtxt(name+'.csv', delimiter=',', dtype='float32')

newhdu = fits.PrimaryHDU(input)
newhdu.writeto(name+'.fits', overwrite=True)

# Launch SExtractor
bashCommand = 'sex '+name+'.fits -c config.sex'
subprocess.check_call(bashCommand.split())

# Load extracted stars image and subtract it from the original
hdu = fits.open('stars.fits')
input = input - hdu[0].data

# Write out masked image
newhdu = fits.PrimaryHDU(input)
newhdu.writeto(name+'-masked.fits', overwrite=True)
