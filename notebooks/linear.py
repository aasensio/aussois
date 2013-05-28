import numpy as np
import matplotlib.pyplot as pl
from scipy import polyfit

cLight = 3e10
frequency = np.asarray([4.4, 4.9, 5.4, 6.7, 8.9, 11.0])*1e9
wavelength = cLight / frequency
ObservedBeta = 0.03 * wavelength**2 + 0.1*np.random.randn(6)

fig = pl.figure(0)
pl.errorbar(wavelength**2, ObservedBeta,yerr=0.2,marker='.',color='k',linestyle='none')
pl.xlabel(r'$\lambda^2$')
pl.ylabel(r'$\beta$')
pl.savefig("linearFunction.pdf",format="pdf")


InferredRM = np.sum(wavelength**2 * ObservedBeta) / np.sum(wavelength**4)

print InferredRM

fig2 = pl.figure(1)
pl.errorbar(wavelength**2, ObservedBeta,yerr=0.2,marker='.',color='k',linestyle='none')
pl.plot(wavelength**2, InferredRM*wavelength**2)
pl.xlabel(r'$\lambda^2$')
pl.ylabel(r'$\beta$')
pl.savefig("linearFunctionFit.pdf",format="pdf")

(a,b) = polyfit(wavelength**2, ObservedBeta, 1)

print a,b