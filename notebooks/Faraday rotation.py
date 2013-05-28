# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Rotation measure

# <codecell>

import numpy as np
import matplotlib.pylab as plt

# <markdowncell>

# The relation between the rotation angle and the rotation measure measure is linear:

# <markdowncell>

# $\beta = \mathrm{RM} \lambda^2$

# <markdowncell>

# Our aim is to compute the rotation measure from a measurement of different rotation angles at different wavelengths. We first define the set of wavelengths where we make the measurement. Let us assume that we observe at 

# <codecell>

cLight = 3e10
frequency = np.asarray([4.4, 4.9, 5.4, 6.7, 8.9, 11.0])*1e9
wavelength = cLight / frequency
ObservedBeta = 0.03 * wavelength**2 + 0.1*np.random.randn(6)

# <markdowncell>

# Let's do the plot of what we observe, noting that the points have some random noise added.

# <codecell>

plt.errorbar(wavelength**2, ObservedBeta,yerr=0.2,marker='.',color='k',linestyle='none')
plt.xlabel(r'$\lambda^2$')
plt.ylabel(r'$\beta$')

# <markdowncell>

# In order to get the value of the rotation measure, we need to minimize the following merit function:

# <markdowncell>

# \begin{equation}
# \chi^2 = \sum_{i=1}^N \frac{\left(\beta_i - \lambda_i^2 \mathrm{RM} \right)^2}{\sigma^2}
# \end{equation}

# <markdowncell>

# which can be done analytically by computing the first derivative and equating it to zero. We obtain:

# <markdowncell>

# \begin{equation}
# \mathrm{RM} = \frac{\sum_i \lambda_i^2 \beta_i}{\sum_i \lambda_i^4}
# \end{equation}

# <codecell>

InferredRM = np.sum(wavelength**2 * ObservedBeta) / np.sum(wavelength**4)

# <markdowncell>

# The inferred value of the rotation measure is, then:

# <codecell>

InferredRM

# <markdowncell>

# which is very close to the original one that we assumed. We can plot the resulting straight line:

# <codecell>

plt.errorbar(wavelength**2, ObservedBeta,yerr=0.2,marker='.',color='k',linestyle='none')
plt.plot(wavelength**2, InferredRM*wavelength**2)
plt.xlabel(r'$\lambda^2$')
plt.ylabel(r'$\beta$')

# <markdowncell>

# Note that the same cannot be done using standard linear regression routines, because the linear fit has to go through the (0,0) point. The result of such a fit gives a different slope to what we used at the beginning:

# <codecell>

(a,b) = polyfit(wavelength**2, ObservedBeta, 1)

# <codecell>

a,b

# <codecell>


