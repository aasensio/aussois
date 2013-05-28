# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Rotation measure

# <codecell>

import numpy as np
import matplotlib.pylab as plt
import emcee
from math import log
import IPython

# <codecell>

nPoints = 30
sigmaNoise = 1.2
xTrend = np.linspace(0,5,num=nPoints)
yTrend = 1.2 * xTrend + 0.5 + sigmaNoise*np.random.randn(nPoints)
xOutliers = np.asarray([1,3.5,4.2])
yOutliers = np.asarray([8,0.1,3])
xObs = np.append(xTrend, xOutliers)
yObs = np.append(yTrend, yOutliers)

# <codecell>

plt.plot(xTrend, yTrend, marker='.',color='k',linestyle='none')
plt.plot(xOutliers, yOutliers, marker='o',color='r',linestyle='none')

# <codecell>

class linearFit:
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        
    def logPosterior(self, x):
        model = x[0]+x[1]*self.x
        return -0.5 * np.sum((model-self.y)**2 / self.sigma**2)
    
    def __call__(self, x):
        return self.logPosterior(x)

# <markdowncell>

# First do the Bayesian inference without taking into account the outliers

# <codecell>

mcmcLinear = linearFit(xTrend, yTrend, sigmaNoise)

# <codecell>

ndim, nwalkers = 2, 10
p0 = [2*np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmcLinear)

# <codecell>

pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

# <codecell>

plt.subplot(2,2,1)
plt.plot(sampler.flatchain[:,0],marker='.',linestyle='none')
plt.subplot(2,2,2)
plt.plot(sampler.flatchain[:,1],marker='.',linestyle='none')
plt.subplot(2,2,3)
plt.hist(sampler.flatchain[:,0])
plt.subplot(2,2,4)
plt.hist(sampler.flatchain[:,1])

# <codecell>

plt.plot(xTrend, yTrend, marker='.',color='k',linestyle='none')
plt.plot(xTrend, 1.2*xTrend+0.5)
for i in range(10):
    plt.plot(xTrend, sampler.flatchain[i,0]+sampler.flatchain[i,1]*xTrend, color='r')

# <codecell>

sampler.flatchain[10,:]

# <codecell>

mcmcLinear = linearFit(xObs, yObs, sigmaNoise)
ndim, nwalkers = 2, 10
p0 = [2*np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmcLinear)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)
plt.subplot(2,2,1)
plt.plot(sampler.flatchain[:,0],marker='.',linestyle='none')
plt.subplot(2,2,2)
plt.plot(sampler.flatchain[:,1],marker='.',linestyle='none')
plt.subplot(2,2,3)
plt.hist(sampler.flatchain[:,0])
plt.subplot(2,2,4)
plt.hist(sampler.flatchain[:,1])

# <codecell>

plt.plot(xObs, yObs, marker='.',color='k',linestyle='none')
plt.plot(xTrend, 1.2*xTrend+0.5)
plt.xlim([0,5])
plt.ylim([-2,9])
for i in range(10):
    plt.plot(xTrend, sampler.flatchain[i,0]+sampler.flatchain[i,1]*xTrend, color='r')

# <codecell>

class linearFitWithOutliers:
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.nObs = x.size
        
    def logPrior(self,x):
        if (x[2] < 0.0 or x[2] > 1.0 or x[4] < 0.0):
            return -np.inf
        else:
            return 1.0
            
    def logLikelihood(self,x):
        a = x[0]
        b = x[1]
        lnPBad = log(x[2])
        lnMinusPBad = log(1.0-x[2])
        YBad = x[3]
        VBad = x[4]
        model = a+b*self.x
        termBad =  lnPBad - 0.5 * (model-self.y)**2 / self.sigma**2
        termNoBad = lnMinusPBad - 0.5 * (YBad-self.y)**2 / VBad**2 - 2.0 * log(VBad)
        #IPython.embed()
        
        return np.sum(np.logaddexp(termBad, termNoBad))
        #return np.sum(termBad)
          
    def logPosterior(self, x):
        t1 = self.logPrior(x)
        if np.isinf(t1):
            return t1
        
        t2 = self.logLikelihood(x)
        
        return t1 + t2
    
    def __call__(self, x):
        return self.logPosterior(x)

# <codecell>

mcmcLinearOutlier = linearFitWithOutliers(xObs, yObs, sigmaNoise)

# <codecell>

ndim, nwalkers = 5, 10
pInit = [1.0,1.0,0.5,np.mean(yObs),np.std(yObs)]
p0 = emcee.utils.sample_ball(pInit, 0.1*np.ones(ndim), nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmcLinearOutlier)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)
plt.subplot(2,2,1)
plt.plot(sampler.flatchain[:,0],marker='.',linestyle='none')
plt.subplot(2,2,2)
plt.plot(sampler.flatchain[:,1],marker='.',linestyle='none')
plt.subplot(2,2,3)
plt.hist(sampler.flatchain[:,0])
plt.subplot(2,2,4)
plt.hist(sampler.flatchain[:,1])



