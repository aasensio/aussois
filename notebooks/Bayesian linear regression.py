# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Bayesian linear regression

# <headingcell level=2>

# Initial steps

# <markdowncell>

# Some initial imports to make our code work. It uses NumPy, Matplotlib and emcee.

# <codecell>

import numpy as np
import matplotlib.pylab as plt
import emcee
import random
import IPython

# <markdowncell>

# Define our fake "observations", which is obtained from a linear function $y=1.2x+0.5$ with Gaussian noise with $\sigma=0.5$ and a few outliers added.

# <codecell>

nPoints = 30
sigmaNoise = 0.5
xTrend = np.linspace(0,5,num=nPoints)
yTrend = 1.2 * xTrend + 0.5 + sigmaNoise*np.random.randn(nPoints)
xOutliers = np.asarray([1,1,4.2])
yOutliers = np.asarray([8,6,3])
xObs = np.append(xTrend, xOutliers)
yObs = np.append(yTrend, yOutliers)

# <markdowncell>

# Plot the initial data.

# <codecell>

plt.plot(xTrend, yTrend, marker='.',color='k',linestyle='none')
plt.plot(xOutliers, yOutliers, marker='o',color='r',linestyle='none')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('originalData.png',format='png',dpi=600)
plt.show()

# <headingcell level=2>

# Standard regression

# <markdowncell>

# We want to do the full sampling of the likelihood (equivalent to a sampling of the posterior distribution with flat priors). Let's define a class which returns the $\log \mathcal{L}=-\chi^2/2$.

# <codecell>

class linearFit:
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        
    def logPosterior(self, x):
        model = x[1]+x[0]*self.x
        return -0.5 * np.sum((model-self.y)**2 / self.sigma**2)
    
    def __call__(self, x):
        return self.logPosterior(x)

# <markdowncell>

# First do the Bayesian inference without taking into account the outliers. To this end, we use the sampling package emcee (https://github.com/dfm/emcee).

# <codecell>

mcmcLinear = linearFit(xTrend, yTrend, sigmaNoise)

# <codecell>

ndim, nwalkers = 2, 10
pInit = [1.0,1.0]
p0 = emcee.utils.sample_ball(pInit, 0.1*np.ones(ndim), nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmcLinear)

# <markdowncell>

# Do an initial short sampling to avoid the transitory called the "burn-in". Then, do the final sampling.

# <codecell>

pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

# <markdowncell>

# Do some plots.

# <codecell>

fig = plt.figure(figsize=(8,6))
labels = ['b','m']
values = [1.2, 0.5]
loop = 1
for i in range(2):
    plt.subplot(2,2,loop)
    plt.plot(sampler.flatchain[:,i],marker='.',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel(labels[i])
    loop += 1
for i in range(2):
    plt.subplot(2,2,loop)
    plt.hist(sampler.flatchain[:,i])
    plt.xlabel(labels[i])
    plt.ylabel('Frequency')
    plt.axvline(values[i], color='red', linewidth=4)
    loop += 1
plt.tight_layout()
plt.savefig('mcmcNoOutlier.png',format='png')

# <markdowncell>

# Plot the original observations and a sample of the straight lines from the likelihood.

# <codecell>

plt.plot(xTrend, yTrend, marker='.',color='k',linestyle='none')
plt.plot(xTrend, 1.2*xTrend+0.5)
samples = range(nwalkers*1000)
random.shuffle(samples)
for i in range(10):
    plt.plot(xTrend, sampler.flatchain[samples[i],1]+sampler.flatchain[samples[i],0]*xTrend, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('fitNoOutlier.png',format='png',dpi=600)

# <markdowncell>

# Now do the same but including the outliers. Since the fit is not robust to ouliers, the fit will not be good. The reason is: 1) the model is not good for the ouliers, 2) the assumption of Gaussian noise with fixed variance is not correct.

# <codecell>

mcmcLinear = linearFit(xObs, yObs, sigmaNoise)
ndim, nwalkers = 2, 10
pInit = [1.0,1.0]
p0 = emcee.utils.sample_ball(pInit, 0.1*np.ones(ndim), nwalkers)

sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmcLinear)
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

fig = plt.figure(figsize=(8,6))
labels = ['b','m']
values = [1.2, 0.5]
loop = 1
for i in range(2):
    plt.subplot(2,2,loop)
    plt.plot(sampler.flatchain[:,i],marker='.',linestyle='none')
    plt.xlabel('Iteration')
    plt.ylabel(labels[i])
    loop += 1
for i in range(2):
    plt.subplot(2,2,loop)
    plt.hist(sampler.flatchain[:,i])
    plt.xlabel(labels[i])
    plt.ylabel('Frequency')
    plt.axvline(values[i], color='red', linewidth=4)
    loop += 1
plt.tight_layout()
plt.savefig('mcmcWithOutliers.png',format='png')

# <codecell>

plt.plot(xObs, yObs, marker='.',color='k',linestyle='none')
plt.plot(xTrend, 1.2*xTrend+0.5)
plt.xlim([0,5])
plt.ylim([-2,9])
samples = range(nwalkers*1000)
random.shuffle(samples)
for i in range(10):
    plt.plot(xTrend, sampler.flatchain[samples[i],1]+sampler.flatchain[samples[i],0]*xTrend, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('fitWithOutliers.png',format='png',dpi=600)

# <headingcell level=2>

# Modeling the outliers

# <markdowncell>

# If we want to have a robust fit, we have to model the outliers. We model them as coming from a broad Gaussian distribution and we let the method infer the value of the mean and variance of the distribution. Each point has a probability $P_\mathrm{bad}$ of being an outlier.

# <codecell>

class linearFitWithOutliers:
    def __init__(self, x, y, sigma):
        self.x = x
        self.y = y
        self.sigma = sigma
        self.nObs = x.size
        
    def logPrior(self,x):
        if (x[2] < 0.0 or x[2] > 1.0 or x[4] <= 0.0):
            return -np.inf
        else:
            return 1.0 / x[4]
            
    def logLikelihood(self,x):
        PBad = x[2]
        YBad = x[3]
        VBad = x[4]
        model = x[0]*self.x+x[1]
        termBad =  PBad * np.exp(-0.5 * (YBad-self.y)**2 / VBad**2) / (VBad*np.sqrt(2.0*np.pi))
        termNoBad = (1.0-PBad) * np.exp(-0.5 * (model-self.y)**2 / self.sigma**2) / (self.sigma*np.sqrt(2.0*np.pi))        
        
        return np.sum(np.log(termBad+termNoBad))
          
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
pos, prob, state = sampler.run_mcmc(p0, 300)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos, 1000)

# <codecell>

loop = 1
fig = plt.figure(figsize=(16, 6))
labels = ['b','m','Pbad','Ybad','Vbad']
values = [1.2, 0.5]
for i in range(5):
    plt.subplot(2,5,loop)
    plt.plot(sampler.flatchain[:,i],marker='.',linestyle='none')
    plt.ylabel(labels[i])
    plt.xlabel('Iteration')
    loop += 1
for i in range(5):
    plt.subplot(2,5,loop)
    plt.hist(sampler.flatchain[:,i])
    plt.xlabel(labels[i])
    plt.ylabel('Frequency')
    loop += 1
plt.tight_layout()
plt.savefig('mcmcWithOutliersRobust.png',format='png')

# <codecell>

plt.plot(xObs, yObs, marker='.',color='k',linestyle='none')
plt.plot(xTrend, 1.2*xTrend+0.5)
plt.xlim([0,5])
plt.ylim([-2,9])
samples = range(nwalkers*1000)
random.shuffle(samples)
for i in range(10):
    plt.plot(xTrend, sampler.flatchain[samples[i],1]+sampler.flatchain[samples[i],0]*xTrend, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('fitWithOutlierRobust.png',format='png',dpi=600)

# <codecell>

x = np.linspace(-10,15,200)
ygood = np.exp(-(x-3)**2/(2*0.3)**2)
ybad = np.exp(-(x-3)**2/(2*3.5)**2)
mixture = 0.5*ygood + 0.5*ybad
mixture2 = 0.1*ygood + 0.9*ybad
mixture3 = 0.9*ygood + 0.1*ybad
plt.plot(x,mixture,color='black')
plt.plot(x,mixture2,color='red')
plt.plot(x,mixture3,color='blue')
plt.savefig('mixture.png',format='png',dpi=600)

# <codecell>

p0

# <codecell>


