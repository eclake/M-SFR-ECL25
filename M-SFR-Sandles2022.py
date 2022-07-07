""" linmix -- A hierarchical Bayesian approach to linear regression with error in both X and Y.

Edited by E. Curtis-Lake 7th December 2020 to incorporate the Hogg mixture model to characterise
which objects are on or off the relation

Edited by L. Sandles for Sandles et al. 2022.
"""

from __future__ import print_function
from scipy.stats import norm,multivariate_normal
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from astropy.io import fits

def task_manager(conn):
    chain = None
    while True:
        message = conn.recv()
        if message['task'] == 'init':
            chain = Chain(**message['init_args'])
            chain.initial_guess()
        elif message['task'] == 'init_chain':
            chain.initialize_chain(message['minIter'],message['nBurn'],message['testIter'])
        elif message['task'] == 'step':
            chain.step(message['niter'])
        elif message['task'] == 'extend':
            chain.extend(message['niter'])
        elif message['task'] == 'fetch':
            conn.send(chain.__dict__[message['key']])
        elif message['task'] == 'kill':
            break
        else:
            print('tm_invalidtask')
            raise ValueError("Invalid task")

class Chain(object):
    def __init__(self, xArr, yArr, zArr, xsigArr, ysigArr, zsigArr, xycovArr, xzcovArr, yzcovArr, delta, nGaussXi, nChains, nGaussBeagle, piBeagle, proposalscale_xi, proposalscale_eta, proposalscale_zeta, proposalscale_alphaN_a, proposalscale_alphaN_b, proposalscale_alphaN_c, proposalscale_beta_a, proposalscale_beta_b, proposalscale_beta_c, proposalscale_sig0, proposalscale_k, proposalscale_pbad, proposalscale_outlier_mean, proposalscale_outlier_sigma, alphaNorm, z_lower, z_upper, rng=None):
        self.xArr = np.array(xArr, dtype=float)
        self.yArr = np.array(yArr, dtype=float)
        self.zArr = np.array(zArr, dtype=float)
        self.xsigArr = np.array(xsigArr, dtype=float)
        self.ysigArr = np.array(ysigArr, dtype=float)
        self.zsigArr = np.array(zsigArr, dtype=float)
        self.xycovArr = np.array(xycovArr, dtype=float)
        self.xzcovArr = np.array(xzcovArr, dtype=float)
        self.yzcovArr = np.array(yzcovArr, dtype=float)

        self.x = np.array(xArr[:,0], dtype=float)
        self.y = np.array(yArr[:,0], dtype=float)
        self.z = np.array(zArr[:,0], dtype=float)
        self.xsig = np.array(xsigArr[:,0], dtype=float)
        self.ysig = np.array(ysigArr[:,0], dtype=float)
        self.zsig = np.array(zsigArr[:,0], dtype=float)
        self.xycov = np.array(xycovArr[:,0], dtype=float)
        self.xzcov = np.array(xzcovArr[:,0], dtype=float)
        self.yzcov = np.array(yzcovArr[:,0], dtype=float)

        self.wxerr = (self.xsig != 0.0)
        self.wyerr = (self.ysig != 0.0)
        self.wzerr = (self.zsig != 0.0)
        self.werrs = werrs = self.wxerr & self.wyerr & self.wzerr

        self.xvar = self.xsig**2
        self.yvar = self.ysig**2
        self.zvar = self.zsig**2

        self.xycorr = np.zeros_like(self.xycov)
        self.xycorr[werrs] = self.xycov[werrs] / (self.xsig[werrs] * self.ysig[werrs])
        self.xzcorr = np.zeros_like(self.xzcov)
        self.xzcorr[werrs] = self.xzcov[werrs] / (self.xsig[werrs] * self.zsig[werrs])
        self.yzcorr = np.zeros_like(self.yzcov)
        self.yzcorr[werrs] = self.yzcov[werrs] / (self.ysig[werrs] * self.zsig[werrs])

        if delta is None: # for censorship, array of 1s means keep all objects
            self.delta = np.ones((len(self.x)), dtype=bool)
        else:
            self.delta = np.array(delta, dtype=bool)

        self.N = len(self.x) # number of objects
        self.nGaussXi = nGaussXi # number of gaussians modelling xi
        self.nChains = nChains # parallel chains
        self.nGaussBeagle = nGaussBeagle
        self.piBeagle = np.array(piBeagle, dtype=float) # weighting of nGaussBeagle

        #parameters for the MCMC chain on update_xi
        self.accept_xi = np.full(len(self.x), 0) # these get updated later on
        self.reject_xi = np.full(len(self.x), 0) # these get updated later on
        self.accept_eta = np.full(len(self.x), 0) # these get updated later on
        self.reject_eta = np.full(len(self.x), 0) # these get updated later on
        self.accept_zeta = np.full(len(self.x), 0) # these get updated later on
        self.reject_zeta = np.full(len(self.x), 0) # these get updated later on
        self.accept_pbad = 0 # these get updated later on
        self.reject_pbad = 0 # these get updated later on
        self.accept_outlier_mean = 0 # these get updated later on
        self.reject_outlier_mean = 0 # these get updated later on
        self.accept_outlier_sigma = 0 # these get updated later on
        self.reject_outlier_sigma = 0 # these get updated later on
        self.accept_sig0 = 0 # these get updated later on
        self.reject_sig0 = 0 # these get updated later on
        self.accept_k = 0 # these get updated later on
        self.reject_k = 0 # these get updated later on
        self.proposalscale_xi = np.full(self.N, proposalscale_xi)
        self.proposalscale_eta = np.full(self.N, proposalscale_eta)
        self.proposalscale_zeta = np.full(self.N, proposalscale_zeta)
        self.proposalscale_alphaN_a = proposalscale_alphaN_a
        self.proposalscale_alphaN_b = proposalscale_alphaN_b
        self.proposalscale_alphaN_c = proposalscale_alphaN_c
        self.proposalscale_beta_a = proposalscale_beta_a
        self.proposalscale_beta_b = proposalscale_beta_b
        self.proposalscale_beta_c = proposalscale_beta_c
        self.proposalscale_sig0 = proposalscale_sig0
        self.proposalscale_k = proposalscale_k
        self.proposalscale_pbad = proposalscale_pbad
        self.proposalscale_outlier_mean = proposalscale_outlier_mean
        self.proposalscale_outlier_sigma = proposalscale_outlier_sigma
        self.alphaNorm = alphaNorm

        self.z_lower = z_lower
        self.z_upper = z_upper

        #ECL variables for the Hogg mixture model
        #We describe the off-model objects with a Gaussian with unknown mean and standard deviation, sigma
        self.outlier_mean = 0.
        self.outlier_sigma = 0.
        self.pbad = np.empty_like(self.x)

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.initialized = False # still needs to make initial guesses as below, then set to True

    def initial_guess(self): # Step 1
        np.random.seed()
        self.chain_identifier = np.random.rand()

        N = self.N
        nGaussXi = self.nGaussXi
        nGaussBeagle = self.nGaussBeagle # 3
        piBeagle = self.piBeagle # the relative amplitudes of the Beagle posterior GMMs are fixed

        if nGaussBeagle == 1: # number of gaussians modelling BEAGLE posterior
            self.GBeagle = np.ones(N, dtype=int)
        else:
            self.GBeagle = np.zeros((N, nGaussBeagle), dtype=int)

            self.xi = np.zeros_like(self.z)
            self.eta = np.zeros_like(self.z)
            self.zeta = np.zeros_like(self.z)

            # just start by assigning GBeagle to random Gaussian (NOT MAX)
            for i in range(N): # for every object
                np.random.seed()

                while self.zeta[i] < self.z_lower or self.zeta[i] > self.z_upper:
                    maxind = np.random.choice(3, p=piBeagle[i])
                    self.GBeagle[i,maxind] = 1
                    self.x[i] = self.xArr[i,maxind]
                    self.y[i] = self.yArr[i,maxind]
                    self.z[i] = self.zArr[i,maxind]
                    self.xsig[i] = self.xsigArr[i,maxind]
                    self.ysig[i] = self.ysigArr[i,maxind]
                    self.zsig[i] = self.zsigArr[i,maxind]
                    self.xycov[i] = self.xycovArr[i,maxind]
                    self.xzcov[i] = self.xzcovArr[i,maxind]
                    self.yzcov[i] = self.yzcovArr[i,maxind]
                    self.xvar[i] = self.xsig[i]**2
                    self.yvar[i] = self.ysig[i]**2
                    self.zvar[i] = self.zsig[i]**2
                    self.xycorr[i] = self.xycov[i] / (self.xsig[i] * self.ysig[i])
                    self.xzcorr[i] = self.xzcov[i] / (self.xsig[i] * self.zsig[i])
                    self.yzcorr[i] = self.yzcov[i] / (self.ysig[i] * self.zsig[i])

                    mean = np.array([self.x[i],self.y[i],self.z[i]])
                    cov = np.array([[np.power(self.xsig[i],2), self.xycov[i], self.xzcov[i]], \
                                    [self.xycov[i], np.power(self.ysig[i],2), self.yzcov[i]], \
                                    [self.xzcov[i], self.yzcov[i], np.power(self.zsig[i],2)]])

                    xi_eta_zeta = np.random.multivariate_normal(mean, cov)

                    self.xi[i] = xi_eta_zeta[0]
                    self.eta[i] = xi_eta_zeta[1]
                    self.zeta[i] = xi_eta_zeta[2]

        self.xi_min = 8.5
        self.xi_max = 10.0

        # take a random sample of xs and ys, calculate alpha, beta and sigma
        idx_z_bin = (self.zeta > self.z_lower) & (self.zeta < self.z_upper)

        def straight_line(x, A, B): # this is your 'straight line' y=f(x)
            return B*(x-self.alphaNorm) + A

        xi_sampled = (self.xi[idx_z_bin]).copy()
        eta_sampled = (self.eta[idx_z_bin]).copy()
        zeta_sampled = (self.zeta[idx_z_bin]).copy()

        id_full = np.arange(len(self.xi))
        id_sampled = id_full[idx_z_bin]

        # =============================================================================
        # find initial MS using 2 sigma clipping
        # =============================================================================

        outliers = 1

        xi_outliers = []
        eta_outliers = []
        zeta_outliers = []
        id_outliers = []

        while outliers > 0:

            surface_fit = curve_fit(self.calc_sfr_surface, (xi_sampled,zeta_sampled), eta_sampled)

            self.beta_a = surface_fit[0][0]
            self.beta_b = surface_fit[0][1]
            self.beta_c = 0.0

            self.alphaN_a = surface_fit[0][2]
            self.alphaN_b = surface_fit[0][3]
            self.alphaN_c = 0.0

            # list of sfrs according to input values
            eta_from_relation = self.calc_sfr_surface((xi_sampled,zeta_sampled), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b)
            eta_residuals = eta_sampled - eta_from_relation
            eta_sigma = np.std(eta_residuals)
            eta_idx = (abs(eta_residuals)<2.0*eta_sigma)

            outliers = sum(~eta_idx)

            for oo in range(outliers):
                xi_outliers.append(xi_sampled[~eta_idx][oo])
                eta_outliers.append(eta_sampled[~eta_idx][oo])
                zeta_outliers.append(zeta_sampled[~eta_idx][oo])
                id_outliers.append(id_sampled[~eta_idx][oo])

            xi_sampled = xi_sampled[eta_idx]
            eta_sampled = eta_sampled[eta_idx]
            zeta_sampled = zeta_sampled[eta_idx]
            id_sampled = id_sampled[eta_idx]

        self.sig0 = eta_sigma
        self.k = 1.0

        xi_outliers = np.array(xi_outliers)
        eta_outliers = np.array(eta_outliers)
        zeta_outliers = np.array(zeta_outliers)
        id_outliers = np.array(id_outliers)

        idx_outlier_below = (eta_outliers < self.calc_sfr_surface((xi_outliers,zeta_outliers), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))
        self.id_outliers_below = id_outliers[idx_outlier_below]

        self.pbad = np.full_like(self.pbad, max(float(len(eta_outliers)) / float(sum(idx_z_bin)), 0.1))
        self.pbad[self.zeta>4.0] = 0.0

        if len(eta_outliers) > 1:
            self.outlier_mean = max(min(np.mean(eta_outliers), 10.0), -10.0)
            self.outlier_sigma = max(min(np.std(eta_outliers), 10.0), 1.0)
        else:
            self.outlier_mean = max(min(np.mean(eta_sampled), 10.0), -10.0)
            self.outlier_sigma = max(min(np.std(eta_sampled), 10.0), 1.0)

        '''
        plt.scatter(xi_sampled, eta_sampled)
        plt.scatter(xi_outliers, eta_outliers)
        plt.scatter(self.xi[self.id_outliers_below], self.eta[self.id_outliers_below], color='lime', marker='x', s=100)
        plt.plot((5.0, 11.0), self.calc_sfr_surface((np.array((5.0, 11.0)),np.array((3.0, 3.0))), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))

        plt.plot((5.0, 11.0), (self.outlier_mean, self.outlier_mean))
        plt.show()

        print('INITIAL GUESSES', self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b, self.sig0, self.k, max(self.pbad), self.outlier_mean, self.outlier_sigma)
        '''
        
        self.beta_b = 0.0

        x = self.x
        xvar = self.xvar
        y = self.y

        self.mu0 = np.median(x)
        self.wsqr = np.var(x, ddof=1) - np.median(xvar)
        self.wsqr = np.max([self.wsqr, 0.01*np.var(x, ddof=1)])

        # Now get the values for the mixture parameters, first do prior params
        self.mu0min = min(x)
        self.mu0max = max(x)

        mu0g = np.nan
        while not (mu0g > self.mu0min) & (mu0g < self.mu0max):
            mu0g = self.mu0 + (self.rng.normal(scale=np.sqrt(np.var(x, ddof=1) / N)) /
                               np.sqrt(self.nChains/self.rng.chisquare(self.nChains)))
        self.mu0 = mu0g

        # wsqr is the global scale
        self.wsqr *= 0.5 * N / self.rng.chisquare(0.5 * N)
        self.usqrmax = 1.5 * np.var(x, ddof=1)
        self.usqr = 0.5 * np.var(x, ddof=1)
        self.tausqr = 0.5 * self.wsqr * self.nChains / self.rng.chisquare(self.nChains, size=nGaussXi)
        self.mu = self.mu0 + self.rng.normal(scale=np.sqrt(self.wsqr), size=nGaussXi)

        # get initial group proportions and group labels
        pig = np.zeros(self.nGaussXi, dtype=float) # [0. 0. 0.]
        if nGaussXi == 1:
            self.G = np.ones(N, dtype=int)
            self.pi = np.array([1], dtype=float)
        else:
            self.G = np.zeros((N, nGaussXi), dtype=int)
            for i in range(N):
                minind = np.argmin(abs(x[i] - self.mu)) # index of the nearest mean of the xi gaussians
                pig[minind] += 1 # becomes something like [98. 34. 51.], number of items in each column
                self.G[i, minind] = 1
            self.pi = self.rng.dirichlet(pig+1) # eg [0.29385138 0.34768844 0.35846018]

        self.y_ul = y.copy()

        self.cens = np.nonzero(np.logical_not(self.delta))[0]
        self.initialized = True

    # =============================================================================
    # START OF STEP
    # =============================================================================
    
    def update_cens_y(self):  # Step 2
        todo = self.cens[:]
        while len(todo) > 0:
            self.y[todo] = self.rng.normal(loc=self.eta[todo],
                                           scale=np.sqrt(self.yvar[todo]),
                                           size=len(todo))
            todo = np.nonzero(np.logical_not(self.delta) & (self.y > self.y_ul))[0]

    def update_xi(self):  # Step 3
        wxerr = self.wxerr                                                              # boolean array where xsig != 0

        # P(xi|psi)
        xi_curr = self.xi                                                               # current xi values
        xi_prop = self.rng.normal(size=len(xi_curr),scale=self.proposalscale_xi)+xi_curr    # new proposal for each xi

        muArr = []                                                                      # mu are means of xi gaussians
        tausqrArr = []                                                                  # tausqr are sigsqr of xi gaussians
        for i in range(len(self.xi)):
            tempIdx = np.where(self.G[i] == 1)[0]                                       # [0], [1] or [2] decides which xi gaussian
            muArr.append(self.mu[tempIdx][0])                                           # chooses mu from [mu1, mu2, mu3]
            tausqrArr.append(self.tausqr[tempIdx][0])                                   # chooses tau from [ta1, ta2, ta3]

        # mass distribution
        log_p_xi_psi_curr = norm.logpdf(xi_curr,loc=muArr,scale=np.sqrt(tausqrArr))     # gives log( gaussian value at xi_curr ) per object
        log_p_xi_psi_prop = norm.logpdf(xi_prop,loc=muArr,scale=np.sqrt(tausqrArr))     # gives log( gaussian value at xi_prop ) per object

        # P(xi|eta,zeta,theta) - this is only really true for the proportionality, I'm not calculating the correct normalization
        # MS relation
        p_xi_eta_zeta_theta_curr = norm.pdf(self.eta, scale=np.sqrt(self.calc_sigsqr(xi_curr)), loc=self.calc_sfr_surface((xi_curr,self.zeta), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))
        p_xi_eta_zeta_theta_prop = norm.pdf(self.eta, scale=np.sqrt(self.calc_sigsqr(xi_prop)), loc=self.calc_sfr_surface((xi_prop,self.zeta), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))

        # outlier distribution
        # p_xi_om_os_curr = norm.pdf(self.eta, loc=self.outlier_mean, scale=self.outlier_sigma)
        p_xi_om_os_curr = self.calc_outlier_prob(self.eta, self.outlier_mean, self.outlier_sigma)
        p_xi_om_os_prop = self.calc_outlier_prob(self.eta, self.outlier_mean, self.outlier_sigma)

        for i in range(len(self.xi)):
            if wxerr[i] == True: # always: boolean array where xsig != 0

                # P(xi|eta,zeta,x,y)
                mean, cov = self.get_3d_mean_cov(i)

                # GMM measurement uncertainties
                log_p_xi_eta_zeta_x_y_curr = multivariate_normal.logpdf([xi_curr[i],self.eta[i],self.zeta[i]],mean,cov)
                log_p_xi_eta_zeta_x_y_prop = multivariate_normal.logpdf([xi_prop[i],self.eta[i],self.zeta[i]],mean,cov)

                log_target_curr = log_p_xi_eta_zeta_x_y_curr + log_p_xi_psi_curr[i] + np.log((((1.0-self.pbad[i])*p_xi_eta_zeta_theta_curr[i]) + (self.pbad[i]*p_xi_om_os_curr[i])))
                log_target_prop = log_p_xi_eta_zeta_x_y_prop + log_p_xi_psi_prop[i] + np.log((((1.0-self.pbad[i])*p_xi_eta_zeta_theta_prop[i]) + (self.pbad[i]*p_xi_om_os_prop[i])))

                acceptanceProb = min(1,np.exp(log_target_prop - log_target_curr)) # 1 if prop more likely than curr

                '''if acceptanceProb == 0:
                    print('xi acceptanceProb 0', self.ichain, i, log_target_prop, log_target_curr, np.exp(log_target_prop - log_target_curr))'''

                u = self.rng.uniform() # random between 0 and 1

                if u <= acceptanceProb: # accept proposal
                    self.xi[i] = xi_prop[i]
                    self.accept_xi[i] = self.accept_xi[i] + 1
                else:
                    self.reject_xi[i] = self.reject_xi[i] + 1

                test = self.accept_xi[i]+self.reject_xi[i]
                if self.ichain >= self.nBurn and test > 0 and test % self.testIter == 0:
                    if float(self.accept_xi[i])/float(self.accept_xi[i]+self.reject_xi[i]) > 0.5:
                        self.proposalscale_xi[i] = self.proposalscale_xi[i]*1.1
                    elif float(self.accept_xi[i])/float(self.accept_xi[i]+self.reject_xi[i]) < 0.4:
                        self.proposalscale_xi[i] = self.proposalscale_xi[i]*0.9

                    self.accept_xi[i] = 0
                    self.reject_xi[i] = 0

            else:
                print('wxerr[i] == FALSE')

    def update_eta(self):  # Step 4
        # Have to make this a MH update step - not sure if this could be done at the same time as the xi update - will have a think!

        wyerr = self.wyerr # boolean array where ysig != 0

        # update each object in turn
        eta_curr = self.eta # current eta values
        eta_prop = self.rng.normal(size=len(eta_curr),scale=self.proposalscale_eta)+eta_curr # new proposal for each eta

        # P(eta|xi,zeta,theta) - this is only really true for the proportionality, I'm not calculating the correct normalization
        p_eta_xi_zeta_theta_curr = norm.pdf(eta_curr, scale=np.sqrt(self.calc_sigsqr(self.xi)), loc=self.calc_sfr_surface((self.xi, self.zeta), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))
        p_eta_xi_zeta_theta_prop = norm.pdf(eta_prop, scale=np.sqrt(self.calc_sigsqr(self.xi)), loc=self.calc_sfr_surface((self.xi, self.zeta), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))

        # outlier distribution
        p_eta_om_os_curr = norm.pdf(eta_curr, loc=self.outlier_mean, scale=self.outlier_sigma)
        p_eta_om_os_prop = norm.pdf(eta_prop, loc=self.outlier_mean, scale=self.outlier_sigma)

        for i in range(len(self.eta)):
            if wyerr[i] == True: # always: boolean array where ysig != 0

                # P(eta|xi,zeta,x,y)
                mean, cov = self.get_3d_mean_cov(i)

                log_p_eta_xi_zeta_x_y_curr = multivariate_normal.logpdf([self.xi[i],eta_curr[i],self.zeta[i]],mean,cov)
                log_p_eta_xi_zeta_x_y_prop = multivariate_normal.logpdf([self.xi[i],eta_prop[i],self.zeta[i]],mean,cov)

                log_target_curr = log_p_eta_xi_zeta_x_y_curr + np.log((((1.0-self.pbad[i])*p_eta_xi_zeta_theta_curr[i]) + (self.pbad[i]*p_eta_om_os_curr[i])))
                log_target_prop = log_p_eta_xi_zeta_x_y_prop + np.log((((1.0-self.pbad[i])*p_eta_xi_zeta_theta_prop[i]) + (self.pbad[i]*p_eta_om_os_prop[i])))

                acceptanceProb = min(1,np.exp(log_target_prop - log_target_curr)) # 1 if prop more likely than curr

                '''if acceptanceProb == 0:
                    print('eta acceptanceProb 0', self.ichain, i, log_target_prop, log_target_curr, np.exp(log_target_prop - log_target_curr))'''

                u = self.rng.uniform() # random between 0 and 1

                if u <= acceptanceProb: # accept proposal
                    self.eta[i] = eta_prop[i]
                    self.accept_eta[i] = self.accept_eta[i] + 1
                else:
                    self.reject_eta[i] = self.reject_eta[i] + 1

                test = self.accept_eta[i]+self.reject_eta[i]
                if self.ichain >= self.nBurn and test > 0 and test % self.testIter == 0:
                    if float(self.accept_eta[i])/float(self.accept_eta[i]+self.reject_eta[i]) > 0.5:
                        self.proposalscale_eta[i] = self.proposalscale_eta[i]*1.1
                    elif float(self.accept_eta[i])/float(self.accept_eta[i]+self.reject_eta[i]) < 0.4:
                        self.proposalscale_eta[i] = self.proposalscale_eta[i]*0.9

                    self.accept_eta[i] = 0
                    self.reject_eta[i] = 0
            else:
                print('wyerr[i] == FALSE')

    def update_zeta(self): # Redshift Step
        wzerr = self.wzerr # boolean array where zsig != 0

        #update each object in turn
        zeta_curr = self.zeta # current zeta values

        zeta_prop = np.empty_like(zeta_curr)

        for i in range(len(self.zeta)):
            zeta_prop[i] = -1.0 # force the while loop
            test = 0
            while (zeta_prop[i] < self.z_lower or zeta_prop[i] > self.z_upper) and test<100:
                zeta_prop[i] = self.rng.normal(size=1,scale=self.proposalscale_zeta[i])+zeta_curr[i] # new proposal for each zeta
                test += 1

            '''if test > 10:
                print('zeta resampling issue', self.ichain, i, test, zeta_prop[i])'''

        # P(zeta|xi,eta,theta) - this is only really true for the proportionality, I'm not calculating the correct normalization
        p_zeta_xi_eta_theta_curr = norm.pdf(self.eta, scale=np.sqrt(self.calc_sigsqr(self.xi)), loc=self.calc_sfr_surface((self.xi,zeta_curr), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))
        p_zeta_xi_eta_theta_prop = norm.pdf(self.eta, scale=np.sqrt(self.calc_sigsqr(self.xi)), loc=self.calc_sfr_surface((self.xi,zeta_prop), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))

        # outlier distribution
        p_zeta_om_os_curr = self.calc_outlier_prob(self.eta, self.outlier_mean, self.outlier_sigma)
        p_zeta_om_os_prop = self.calc_outlier_prob(self.eta, self.outlier_mean, self.outlier_sigma)

        for i in range(len(self.zeta)):
            if wzerr[i] == True: # always: boolean array where ysig != 0

                # P(zeta|xi,eta,x,y)
                mean, cov = self.get_3d_mean_cov(i)

                log_p_zeta_xi_eta_x_y_curr = multivariate_normal.logpdf([self.xi[i],self.eta[i],zeta_curr[i]],mean,cov)
                log_p_zeta_xi_eta_x_y_prop = multivariate_normal.logpdf([self.xi[i],self.eta[i],zeta_prop[i]],mean,cov)

                log_target_curr = log_p_zeta_xi_eta_x_y_curr + np.log((((1.0-self.pbad[i])*p_zeta_xi_eta_theta_curr[i]) + (self.pbad[i]*p_zeta_om_os_curr[i])))
                log_target_prop = log_p_zeta_xi_eta_x_y_prop + np.log((((1.0-self.pbad[i])*p_zeta_xi_eta_theta_prop[i]) + (self.pbad[i]*p_zeta_om_os_prop[i])))

                r_zeta = np.exp(log_target_prop - log_target_curr)
                r_zeta = r_zeta * ((norm.cdf(zeta_curr[i]-self.z_lower, scale=self.proposalscale_zeta[i])-norm.cdf(zeta_curr[i]-self.z_upper, scale=self.proposalscale_zeta[i])) / (norm.cdf(zeta_prop[i]-self.z_lower, scale=self.proposalscale_zeta[i])-norm.cdf(zeta_prop[i]-self.z_upper, scale=self.proposalscale_zeta[i])))
                acceptanceProb = min(1,r_zeta) # 1 if prop more likely than curr

                '''if acceptanceProb == 0:
                    print('zeta acceptanceProb 0', self.ichain, i, log_target_prop, log_target_curr, np.exp(log_target_prop - log_target_curr))'''

                u = self.rng.uniform() # random between 0 and 1
                if u <= acceptanceProb:
                    self.zeta[i] = zeta_prop[i]
#                    if i == 0:
                    self.accept_zeta[i] = self.accept_zeta[i] + 1
                else:
#                    if i == 0:
                    self.reject_zeta[i] = self.reject_zeta[i] + 1

                test = self.accept_zeta[i]+self.reject_zeta[i]
                if self.ichain >= self.nBurn and test > 0 and test % self.testIter == 0:
                    if float(self.accept_zeta[i])/float(self.accept_zeta[i]+self.reject_zeta[i]) > 0.5:
                        self.proposalscale_zeta[i] = min(self.proposalscale_zeta[i]*1.1, ((self.z_upper + self.z_lower) / 2.0) - self.z_lower)
                        '''if self.proposalscale_zeta[i] == ((self.z_upper + self.z_lower) / 2.0) - self.z_lower:
                            print('PROP SCALE TOO LARGE', self.ichain, i, self.zeta[i])'''
                    elif float(self.accept_zeta[i])/float(self.accept_zeta[i]+self.reject_zeta[i]) < 0.4:
                        self.proposalscale_zeta[i] = self.proposalscale_zeta[i]*0.9

                    self.accept_zeta[i] = 0
                    self.reject_zeta[i] = 0

                self.pbad[self.zeta<=4.0] = max(self.pbad)
                self.pbad[self.zeta>4.0] = 0.0

            else:
                print('wzerr[i] == FALSE')

    def update_G(self):  # Step 5
        # Eqn (74)

        piNp = self.pi * (1.0/np.sqrt(2.0*np.pi*self.tausqr)
                          * np.exp(-0.5 * (self.xi[:, np.newaxis] - self.mu)**2 / self.tausqr))
        q_ki = piNp / np.sum(piNp, axis=1)[:, np.newaxis]
        # Eqn (73)
        for i in range(self.N):
            self.G[i] = self.rng.multinomial(1, q_ki[i])

    def update_alphaN_beta_sigma_outlier_model(self):  # Step 6

        alphaN_a_curr    = self.alphaN_a
        alphaN_b_curr    = self.alphaN_b
        alphaN_c_curr    = 0.0
        beta_a_curr     = self.beta_a

        # beta_b_curr     = self.beta_b
        beta_b_curr     = 0.0

        beta_c_curr     = 0.0
        sig0_curr       = self.sig0
        k_curr          = 1.0
        pbad_curr     = self.pbad
        outlier_mean_curr = self.outlier_mean
        outlier_sigma_curr = self.outlier_sigma

        idx_z_bin = (self.zeta > self.z_lower) & (self.zeta < self.z_upper)

        if self.ichain * 2 % self.nBurn == 0:

            # =============================================================================
            # redshift bin MS colour coded by MS probability - UPDATED FOR SSFR
            # =============================================================================

            '''
            print(self.ichain)
            p_eta_xi_zeta_theta = norm.pdf(self.eta[idx_z_bin], scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin])), loc=self.calc_sfr_surface((self.xi[idx_z_bin], self.zeta[idx_z_bin]), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b))
            p_eta_om_os = self.calc_outlier_prob(self.eta[idx_z_bin], self.outlier_mean, self.outlier_sigma)

            z_bad = self.pbad[idx_z_bin]*p_eta_om_os
            z_good = (1.0-self.pbad[idx_z_bin])*p_eta_xi_zeta_theta
            z_bad = np.where(z_bad==0, 1e-60, z_bad) # necessary if NO HOGG, ie pbad == 0
            idx_sort = np.argsort(z_good/z_bad)

            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_axes([0, 0, 0.85, 0.84]) #[left, bottom, width, height]
            scatter1 = ax1.scatter(self.xi[idx_z_bin][idx_sort], self.eta[idx_z_bin][idx_sort] - self.calc_sfr_surface((self.xi[idx_z_bin][idx_sort], self.zeta[idx_z_bin][idx_sort]), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b), c=np.log10((z_good/z_bad)[idx_sort]), cmap=mpl.cm.get_cmap('coolwarm'), alpha=1.0, vmin=-2.5, vmax=2.5, s=100)
            ax1.scatter(self.xi, self.eta - self.calc_sfr_surface((self.xi, self.zeta), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b), color='k', s=1)
            ax1.scatter(self.xi[5], self.eta[5] - self.calc_sfr_surface((self.xi[5], self.zeta[5]), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b), color='k', s=500)
            ax1.scatter(self.xi[~idx_z_bin], self.eta[~idx_z_bin] - self.calc_sfr_surface((self.xi[~idx_z_bin], self.zeta[~idx_z_bin]), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b), color='pink')
            idx_z_bin_outlier = (self.zeta[self.id_outliers_below] > self.z_lower) & (self.zeta[self.id_outliers_below] < self.z_upper) # outliers below within bin
            ax1.scatter(self.xi[self.id_outliers_below][idx_z_bin_outlier], self.eta[self.id_outliers_below][idx_z_bin_outlier] - self.calc_sfr_surface((self.xi[self.id_outliers_below][idx_z_bin_outlier], self.zeta[self.id_outliers_below][idx_z_bin_outlier]), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b), color='lime', s=100, marker='x')
            xlow = 6.5
            xhigh = 10.5
            ylow = -5.5
            yhigh = 2.5
            lw = 3
            ax1.plot((9.7,9.7), (ylow, yhigh), color='gray', linestyle='dashed', linewidth=2)
            x_tmp = np.array([xlow, xhigh])
            ax1.plot(x_tmp, (0,0), color='r', linewidth=lw)
            ax1.plot(x_tmp, (self.sig0,self.sig0), color='r', linestyle='dashed', linewidth=lw, label=r'Our Work, Intrinsic Scatter, {:.2f} {} sig0:{:.2f} pbad:{:.2f}'.format(self.chain_identifier, self.ichain, self.sig0, max(self.pbad)))
            ax1.plot(x_tmp, (-self.sig0,-self.sig0), color='r', linestyle='dashed', linewidth=lw)

            #OUTLIER DIST
            ax1.plot(x_tmp, np.array([self.outlier_mean, self.outlier_mean]) - self.calc_sfr_surface((x_tmp, np.array((3.0,3.0))), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b), color='b', linewidth=lw)
            ax1.plot(x_tmp, np.array([self.outlier_mean, self.outlier_mean]) - self.calc_sfr_surface((x_tmp, np.array((3.0,3.0))), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b)+self.outlier_sigma, color='b', linewidth=lw, linestyle='dashed')
            ax1.plot(x_tmp, np.array([self.outlier_mean, self.outlier_mean]) - self.calc_sfr_surface((x_tmp, np.array((3.0,3.0))), self.beta_a, self.beta_b, self.alphaN_a, self.alphaN_b)-self.outlier_sigma, color='b', linewidth=lw, linestyle='dashed')

            ax1.set_xlim(xlow, xhigh)
            ax1.set_xlabel(r'$\log(\mathrm{Stellar} \, \mathrm{Mass} \, / \, \mathrm{M_{\odot}})$', labelpad=10)
            ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
            ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
            ax1.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom='on', top='on')
            ax1.xaxis.set_tick_params(which='minor', size=5, width=2, direction='in', bottom='on', top='on')
            ax1.xaxis.set_tick_params(labelsize=10)
            ax1.set_ylim(ylow, yhigh)
            ax1.set_ylabel(r'$\log(\mathrm{SFR} \, / \, \mathrm{M_{\odot} \, yr^{-1}})$', labelpad=10)
            ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
            ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
            ax1.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left='on', right='on')
            ax1.yaxis.set_tick_params(which='minor', size=5, width=2, direction='in', left='on', right='on')
            ax1.yaxis.set_tick_params(labelsize=10)
            ax1.legend(loc='lower right', frameon=True, fontsize=10, framealpha=1.0)

            cbaxes = fig.add_axes([0.85, 0, 0.05, 0.84]) #[left, bottom, width, height]
            cb = plt.colorbar(scatter1, cax = cbaxes)
            cb.set_ticks(np.linspace(-2, 2, 5))
            cb.set_label(r'$\mathrm{MS}\longleftarrow \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, \longrightarrow \mathrm{Outliers}$', rotation=270, labelpad=30)
            cb.ax.tick_params(axis='y', size=0, width=2, direction='in', labelsize=0)

            ax2 = fig.add_axes([1.0, 0, 0.85, 0.84]) #[left, bottom, width, height]
            scatter2 = ax2.scatter(self.xi[idx_z_bin][idx_sort], self.eta[idx_z_bin][idx_sort], c=np.log10((z_good/z_bad)[idx_sort]), cmap=mpl.cm.get_cmap('coolwarm'), alpha=1.0, vmin=-2.5, vmax=2.5, s=100)
            ax2.scatter(self.xi, self.eta, color='k', s=1)
            ax2.scatter(self.xi[5], self.eta[5], color='k', s=500)
            ax2.scatter(self.xi[~idx_z_bin], self.eta[~idx_z_bin], color='pink')
            idx_z_bin_outlier = (self.zeta[self.id_outliers_below] > self.z_lower) & (self.zeta[self.id_outliers_below] < self.z_upper) # outliers below within bin
            ax2.scatter(self.xi[self.id_outliers_below][idx_z_bin_outlier], self.eta[self.id_outliers_below][idx_z_bin_outlier], color='lime', s=100, marker='x')
            xlow = 6.5
            xhigh = 10.5
            ylow = -5.5
            yhigh = 2.5
            lw = 3
            ax2.plot((9.7,9.7), (ylow, yhigh), color='gray', linestyle='dashed', linewidth=2)
            x_tmp = np.array([xlow, xhigh])
            ax2.plot(x_tmp, self.beta_a*(x_tmp-9.7)+   np.log10(10**self.alphaN_a * ((1+1.25)**self.alphaN_b))     + 0.7 , color='r', linewidth=lw)
            ax2.plot(x_tmp, self.beta_a*(x_tmp-9.7)+   np.log10(10**self.alphaN_a * ((1+6.0)**self.alphaN_b))     + 0.7 , color='k', linewidth=lw)
            ax2.plot(x_tmp, self.beta_a*(x_tmp-9.7)+   np.log10(10**self.alphaN_a * ((1+1.25)**self.alphaN_b)) + self.sig0 + 0.7, color='r', linestyle='dashed', linewidth=lw, label=r'Our Work, Intrinsic Scatter, {:.2f} {} sig0:{:.2f} pbad:{:.2f}'.format(self.chain_identifier, self.ichain, self.sig0, max(self.pbad)))
            ax2.plot(x_tmp, self.beta_a*(x_tmp-9.7)+   np.log10(10**self.alphaN_a * ((1+1.25)**self.alphaN_b)) - self.sig0 + 0.7, color='r', linestyle='dashed', linewidth=lw)

            #OUTLIER DIST
            ax2.plot(x_tmp, np.array([self.outlier_mean, self.outlier_mean]), color='b', linewidth=lw)
            ax2.plot(x_tmp, np.array([self.outlier_mean, self.outlier_mean])+self.outlier_sigma, color='b', linewidth=lw, linestyle='dashed')
            ax2.plot(x_tmp, np.array([self.outlier_mean, self.outlier_mean])-self.outlier_sigma, color='b', linewidth=lw, linestyle='dashed')
            ax2.set_xlabel(r'$\log(\mathrm{Stellar} \, \mathrm{Mass} \, / \, \mathrm{M_{\odot}})$', labelpad=10)
            ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
            ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
            ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', bottom='on', top='on')
            ax2.xaxis.set_tick_params(which='minor', size=5, width=2, direction='in', bottom='on', top='on')
            ax2.xaxis.set_tick_params(labelsize=10)
            ax2.set_ylabel(r'$\log(\mathrm{SFR} \, / \, \mathrm{M_{\odot} \, yr^{-1}})$', labelpad=10)
            ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
            ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
            ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', left='on', right='on')
            ax2.yaxis.set_tick_params(which='minor', size=5, width=2, direction='in', left='on', right='on')
            ax2.yaxis.set_tick_params(labelsize=10)
            ax2.legend(loc='lower right', frameon=True, fontsize=10, framealpha=1.0)
            cbaxes2 = fig.add_axes([1.85, 0, 0.05, 0.84]) #[left, bottom, width, height]
            cb2 = plt.colorbar(scatter2, cax = cbaxes2)
            cb2.set_ticks(np.linspace(-2, 2, 5))
            cb2.set_label(r'$\mathrm{MS}\longleftarrow \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, \longrightarrow \mathrm{Outliers}$', rotation=270, labelpad=30)
            cb2.ax.tick_params(axis='y', size=0, width=2, direction='in', labelsize=0)
            plt.show()
            '''
    
        # =============================================================================
        # alpha & beta
        # =============================================================================

        u = self.rng.uniform()

        if self.ichain <= self.nBurn:

            alphaN_a_prop,alphaN_b_prop = self.rng.normal(loc=[alphaN_a_curr,alphaN_b_curr], scale = [self.proposalscale_alphaN_a,self.proposalscale_alphaN_b])

            beta_a_prop,beta_b_prop = self.rng.normal(loc=[beta_a_curr,beta_b_curr], scale = [self.proposalscale_beta_a, self.proposalscale_beta_b])

        else:

            covProposal = np.cov(np.array((self.chain['alphaN_a'][max(0, self.ichain-self.nBurn):self.ichain], self.chain['alphaN_b'][max(0, self.ichain-self.nBurn):self.ichain], self.chain['beta_a'][max(0, self.ichain-self.nBurn):self.ichain], self.chain['beta_b'][max(0, self.ichain-self.nBurn):self.ichain])))

            alphaN_a_prop, alphaN_b_prop, beta_a_prop, beta_b_prop = self.rng.multivariate_normal([alphaN_a_curr, alphaN_b_curr, beta_a_curr, beta_b_curr], covProposal)

        alphaN_c_prop    = alphaN_c_curr
        beta_b_prop     = beta_b_curr
        beta_c_prop     = beta_c_curr

        prior_curr = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)
        prior_prop = self.prior_all(alphaN_a_prop, alphaN_b_prop, alphaN_c_prop, beta_a_prop, beta_b_prop, beta_c_prop, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)

        if prior_prop == 0:
            acceptanceProb = 0
            '''print('acceptanceProb 1')'''

        else:

            log_target_curr = np.sum(np.log(prior_curr*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_curr*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))
            

            log_target_prop = np.sum(np.log(prior_prop*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_prop, beta_b_prop, alphaN_a_prop, alphaN_b_prop), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_prop*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

            r = np.exp(log_target_prop-log_target_curr)
            acceptanceProb = min(1,r)

        if u <= acceptanceProb:
            self.alphaN_a = alphaN_a_prop
            self.alphaN_b = alphaN_b_prop
            self.alphaN_c = alphaN_c_prop
            self.beta_a = beta_a_prop
            self.beta_b = beta_b_prop
            self.beta_c = beta_c_prop

        alphaN_a_curr    = self.alphaN_a
        alphaN_b_curr    = self.alphaN_b
        alphaN_c_curr    = self.alphaN_c
        beta_a_curr     = self.beta_a
        beta_b_curr     = self.beta_b
        beta_c_curr     = self.beta_c

        self.alphaN_a_prop = alphaN_a_prop
        self.alphaN_b_prop = alphaN_b_prop
        self.beta_a_prop = beta_a_prop
        self.beta_b_prop = beta_b_prop

        if self.ichain >= self.nBurn:

            # =============================================================================
            # sig0
            # =============================================================================

            u = self.rng.uniform()

            sig0_prop = -1.0 # force the while loop
            while sig0_prop < 0.0:
                sig0_prop = self.rng.normal(loc=sig0_curr, scale = self.proposalscale_sig0)

            prior_sig0_curr = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)
            prior_sig0_prop = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_prop, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)

            if prior_sig0_prop == 0:
                acceptanceProb_sig0 = 0
                '''print('acceptanceProb_sig0 1')'''

            else:
                log_target_sig0_curr = np.sum(np.log(prior_sig0_curr*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_sig0_curr*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                log_target_sig0_prop = np.sum(np.log(prior_sig0_prop*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_prop, k=k_curr))) + prior_sig0_prop*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                r_sig0 = np.exp(log_target_sig0_prop-log_target_sig0_curr)
                r_sig0 = r_sig0 * (norm.cdf(sig0_curr, scale=self.proposalscale_sig0) / norm.cdf(sig0_prop, scale=self.proposalscale_sig0))
                acceptanceProb_sig0 = min(1,r_sig0)

            if u <= acceptanceProb_sig0:
                self.accept_sig0 += 1

            else:
                self.reject_sig0 += 1

            test_sig0 = self.accept_sig0+self.reject_sig0
            if self.ichain >= self.nBurn and test_sig0 > 0 and test_sig0 % self.testIter == 0:
                if float(self.accept_sig0)/float(self.accept_sig0+self.reject_sig0) > 0.5:
                    self.proposalscale_sig0 = self.proposalscale_sig0*1.1
                elif float(self.accept_sig0)/float(self.accept_sig0+self.reject_sig0) < 0.4:
                    self.proposalscale_sig0 = self.proposalscale_sig0*0.9

                self.accept_sig0 = 0
                self.reject_sig0 = 0

            if u <= acceptanceProb_sig0:
                self.sig0 = sig0_prop

            sig0_curr     = self.sig0
            self.sig0_prop = sig0_prop

            # =============================================================================
            # k
            # =============================================================================

            u = self.rng.uniform()

            k_prop = -1.0 # force the while loop
            while k_prop < 0.0:
                k_prop = self.rng.normal(loc=k_curr, scale = self.proposalscale_k)

            k_prop = k_curr

            prior_k_curr = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)
            prior_k_prop = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_prop, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)

            if prior_k_prop == 0:
                acceptanceProb_k = 0
                '''print('acceptanceProb_k 1')'''

            else:
                log_target_k_curr = np.sum(np.log(prior_k_curr*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_k_curr*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                log_target_k_prop = np.sum(np.log(prior_k_prop*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_prop))) + prior_k_prop*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                r_k = np.exp(log_target_k_prop-log_target_k_curr)
                r_k = r_k * (norm.cdf(k_curr, scale=self.proposalscale_k) / norm.cdf(k_prop, scale=self.proposalscale_k))
                acceptanceProb_k = min(1,r_k)

            if u <= acceptanceProb_k:
                self.accept_k += 1

            else:
                self.reject_k += 1

            test_k = self.accept_k+self.reject_k
            if self.ichain >= self.nBurn and test_k > 0 and test_k % self.testIter == 0:
                if float(self.accept_k)/float(self.accept_k+self.reject_k) > 0.5:
                    self.proposalscale_k = self.proposalscale_k*1.1
                elif float(self.accept_k)/float(self.accept_k+self.reject_k) < 0.4:
                    self.proposalscale_k = self.proposalscale_k*0.9

                self.accept_k = 0
                self.reject_k = 0

            if u <= acceptanceProb_k:
                self.k = k_prop

            k_curr = self.k

        if self.ichain >= 2*self.nBurn:

            # =============================================================================
            # pbad
            # =============================================================================

            u = self.rng.uniform()

            pbad_prop_value = -1.0 # force the while loop
            while pbad_prop_value < 0.0:
                pbad_prop_value = self.rng.normal(loc=max(pbad_curr), scale = self.proposalscale_pbad)
                pbad_prop = np.full_like(pbad_curr, 0.0)
                pbad_prop[pbad_curr>0.0] = pbad_prop_value

            prior_pbad_curr = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)
            prior_pbad_prop = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_prop), outlier_mean_curr, outlier_sigma_curr)

            if prior_pbad_prop == 0:
                acceptanceProb_pbad = 0
                '''print('acceptanceProb_pbad 1')'''

            else:
                log_target_pbad_curr = np.sum(np.log(prior_pbad_curr*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_pbad_curr*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                log_target_pbad_prop = np.sum(np.log(prior_pbad_prop*(1.0-pbad_prop[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_pbad_prop*pbad_prop[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                r_pbad = np.exp(log_target_pbad_prop-log_target_pbad_curr)
                r_pbad = r_pbad * (norm.cdf(max(pbad_curr), scale=self.proposalscale_pbad) / norm.cdf(max(pbad_prop), scale=self.proposalscale_pbad))
                acceptanceProb_pbad = min(1,r_pbad)

            if u <= acceptanceProb_pbad:
                self.accept_pbad += 1

            else:
                self.reject_pbad += 1

            test_pbad = self.accept_pbad+self.reject_pbad
            if self.ichain >= self.nBurn and test_pbad > 0 and test_pbad % self.testIter == 0:
                if float(self.accept_pbad)/float(self.accept_pbad+self.reject_pbad) > 0.5:
                    self.proposalscale_pbad = self.proposalscale_pbad*1.1
                elif float(self.accept_pbad)/float(self.accept_pbad+self.reject_pbad) < 0.4:
                    self.proposalscale_pbad = self.proposalscale_pbad*0.9

                self.accept_pbad = 0
                self.reject_pbad = 0

            if u <= acceptanceProb_pbad:
                self.pbad = pbad_prop

            pbad_curr     = self.pbad

            # =============================================================================
            # outlier_mean
            # =============================================================================

            u = self.rng.uniform()

            outlier_mean_prop = self.rng.normal(loc=outlier_mean_curr, scale = self.proposalscale_outlier_mean)

            prior_outlier_mean_curr = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)
            prior_outlier_mean_prop = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_prop, outlier_sigma_curr)

            if prior_outlier_mean_prop == 0:
                acceptanceProb_outlier_mean = 0
                '''print('acceptanceProb_outlier_mean 1')'''

            else:
                log_target_outlier_mean_curr = np.sum(np.log(prior_outlier_mean_curr*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + \
                                                     prior_outlier_mean_curr*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                log_target_outlier_mean_prop = np.sum(np.log(prior_outlier_mean_prop*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_outlier_mean_prop*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_prop, outlier_sigma_curr)))

                r_outlier_mean = np.exp(log_target_outlier_mean_prop-log_target_outlier_mean_curr)
                acceptanceProb_outlier_mean = min(1,r_outlier_mean)

            if u <= acceptanceProb_outlier_mean:
                self.accept_outlier_mean += 1

            else:
                self.reject_outlier_mean += 1

            test_outlier_mean = self.accept_outlier_mean+self.reject_outlier_mean
            if self.ichain >= self.nBurn and test_outlier_mean > 0 and test_outlier_mean % self.testIter == 0:
                if float(self.accept_outlier_mean)/float(self.accept_outlier_mean+self.reject_outlier_mean) > 0.5:
                    self.proposalscale_outlier_mean = self.proposalscale_outlier_mean*1.1
                elif float(self.accept_outlier_mean)/float(self.accept_outlier_mean+self.reject_outlier_mean) < 0.4:
                    self.proposalscale_outlier_mean = self.proposalscale_outlier_mean*0.9

                self.accept_outlier_mean = 0
                self.reject_outlier_mean = 0

            if u <= acceptanceProb_outlier_mean:
                self.outlier_mean = outlier_mean_prop

            outlier_mean_curr = self.outlier_mean

            # =============================================================================
            # outlier_sigma
            # =============================================================================

            u = self.rng.uniform()

            outlier_sigma_prop = -1.0 # force the while loop
            while outlier_sigma_prop < 0.0:
                outlier_sigma_prop = self.rng.normal(loc=outlier_sigma_curr, scale = self.proposalscale_outlier_sigma)

            prior_outlier_sigma_curr = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_curr)
            prior_outlier_sigma_prop = self.prior_all(alphaN_a_curr, alphaN_b_curr, alphaN_c_curr, beta_a_curr, beta_b_curr, beta_c_curr, sig0_curr, k_curr, max(pbad_curr), outlier_mean_curr, outlier_sigma_prop)

            if prior_outlier_sigma_prop == 0:
                acceptanceProb_outlier_sigma = 0
                '''print('acceptanceProb_outlier_sigma 1')'''

            else:
                log_target_outlier_sigma_curr = np.sum(np.log(prior_outlier_sigma_curr*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_outlier_sigma_curr*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_curr)))

                log_target_outlier_sigma_prop = np.sum(np.log(prior_outlier_sigma_prop*(1.0-pbad_curr[idx_z_bin])*norm.pdf(self.eta[idx_z_bin], loc=self.calc_sfr_surface((self.xi[idx_z_bin],self.zeta[idx_z_bin]),beta_a_curr, beta_b_curr, alphaN_a_curr, alphaN_b_curr), scale=np.sqrt(self.calc_sigsqr(self.xi[idx_z_bin], sig0=sig0_curr, k=k_curr))) + prior_outlier_sigma_prop*pbad_curr[idx_z_bin]*self.calc_outlier_prob(self.eta[idx_z_bin], outlier_mean_curr, outlier_sigma_prop)))

                r_outlier_sigma = np.exp(log_target_outlier_sigma_prop-log_target_outlier_sigma_curr)
                r_outlier_sigma = r_outlier_sigma * (norm.cdf(outlier_sigma_curr, scale=self.proposalscale_outlier_sigma) / norm.cdf(outlier_sigma_prop, scale=self.proposalscale_outlier_sigma))
                acceptanceProb_outlier_sigma = min(1,r_outlier_sigma)

            if u <= acceptanceProb_outlier_sigma:
                self.accept_outlier_sigma += 1

            else:
                self.reject_outlier_sigma += 1

            test_outlier_sigma = self.accept_outlier_sigma+self.reject_outlier_sigma
            if self.ichain >= self.nBurn and test_outlier_sigma > 0 and test_outlier_sigma % self.testIter == 0:
                if float(self.accept_outlier_sigma)/float(self.accept_outlier_sigma+self.reject_outlier_sigma) > 0.5:
                    self.proposalscale_outlier_sigma = self.proposalscale_outlier_sigma*1.1
                elif float(self.accept_outlier_sigma)/float(self.accept_outlier_sigma+self.reject_outlier_sigma) < 0.4:
                    self.proposalscale_outlier_sigma = self.proposalscale_outlier_sigma*0.9

                self.accept_outlier_sigma = 0
                self.reject_outlier_sigma = 0

            if u <= acceptanceProb_outlier_sigma:
                self.outlier_sigma = outlier_sigma_prop

        if self.ichain % self.nBurn == 0:
            self.sig0_propscale = self.proposalscale_sig0
            self.k_propscale = self.proposalscale_k
            self.pbad_propscale = self.proposalscale_pbad
            self.outlier_mean_propscale = self.proposalscale_outlier_mean
            self.outlier_sigma_propscale = self.proposalscale_outlier_sigma
            self.xi_propscale = self.proposalscale_xi
            self.eta_propscale = self.proposalscale_eta
            self.zeta_propscale = self.proposalscale_zeta

    def update_pi(self):  # Step 8
        # Eqn (82)

        self.nk = np.sum(self.G, axis=0)
        # Eqn (81)
        self.pi = self.rng.dirichlet(self.nk+1)

    def update_mu(self):  # Step 9
        Gsum = np.sum(self.G * self.xi[:, np.newaxis], axis=0)
        for k in range(self.nGaussXi):
            if self.nk[k] != 0:
                # Eqn (86)
                Sigma_muhat_k = 1.0/(1.0/self.usqr + self.nk[k]/self.tausqr[k])
                # Eqn (85)
                xibar_k = 1.0/self.nk[k] * Gsum[k]
                # Eqn (84)
                muhat_k = Sigma_muhat_k * (self.mu0/self.usqr + self.nk[k]/self.tausqr[k]*xibar_k)
                # Eqn (83)
                self.mu[k] = self.rng.normal(loc=muhat_k, scale=np.sqrt(Sigma_muhat_k))
            else:
                self.mu[k] = self.rng.normal(loc=self.mu0, scale=np.sqrt(self.usqr))

    def update_tausqr(self):  # Step 10
        # Eqn (88)
        nu_k = self.nk + 1
        # Eqn (89)
        tk_sqr = 1.0/nu_k * (self.wsqr + np.sum(self.G*(self.xi[:, np.newaxis]-self.mu)**2, axis=0))
        # Eqn (87)
        self.tausqr = tk_sqr * nu_k / self.rng.chisquare(nu_k, size=self.nGaussXi)

    def update_mu0(self):  # Step 11
        # Eqn (94)
        mubar = np.mean(self.mu)
        # Eqn (93)
        self.mu0 = self.rng.normal(loc=mubar, scale=np.sqrt(self.usqr/self.nGaussXi))

    def update_usqr(self):  # Step 12
        # Eqn (96)
        nu_u = self.nGaussXi + 1
        # Eqn (97)
        usqrhat = 1.0/nu_u * (self.wsqr + np.sum((self.mu - self.mu0)**2))
        usqr = np.inf
        while not usqr <= self.usqrmax:
            usqr = usqrhat * nu_u / self.rng.chisquare(nu_u)
        self.usqr = usqr

    def update_wsqr(self):  # Step 13
        # Eqn (102)
        a = 0.5 * (self.nGaussXi + 3)
        # Eqn (103)
        b = 0.5 * (1.0/self.usqr + np.sum(1.0/self.tausqr))
        # Eqn (101)
        self.wsqr = self.rng.gamma(a, 1.0/b)

    def update_GBeagle(self):
        for i in range(self.N):
            prob_arr = np.zeros(self.nGaussBeagle)
            for j in range(self.nGaussBeagle):
                if self.xsigArr[i][j] == 0 and self.ysigArr[i][j] == 0 and self.zsigArr[i][j] == 0: # hopefully never actually the case, these are the sigmas of the fixed beagle GMM fits
                    prob_arr[j] = 0.
                else:
                    values = [self.xi[i],self.eta[i],self.zeta[i]]
                    cov = [[self.xsigArr[i][j]**2,self.xycovArr[i][j],self.xzcovArr[i][j]],\
                              [self.xycovArr[i][j],self.ysigArr[i][j]**2,self.yzcovArr[i][j]],\
                              [self.xzcovArr[i][j],self.yzcovArr[i][j],self.zsigArr[i][j]**2]]
                    mean = [self.xArr[i,j],self.yArr[i,j],self.zArr[i,j]]
                    prob_arr[j] = np.log(self.piBeagle[i,j])+multivariate_normal.logpdf(values,mean,cov)
                    
            if (np.sum(prob_arr) == 0): # should never be the case
                for j in range(self.nGaussBeagle):
                    prob_arr[j] = self.piBeagle[i,j]
                sys.exit() # this kills the program, so this loop must have never been triggered
                #It looks like sometimes with small scatter at the position of some points, the probability is very close to zero
                #In these instances I'm drawing the next gaussian based on input weights... not great, but not sure I can do much else
            else:
                #sometimes the probabilities are really small so we want to be careful - if there is one that is so much smaller we should
                #set the probability to zero
                deltaProb = prob_arr-np.max(prob_arr)

                tempIdx = np.where(deltaProb < -50)[0] #remember this is a difference of 50 in log so it's really a huge difference!
                if len(tempIdx) > 0: # should never be the case (ie if some log probs are 50 < max prob), if so, sets them to 0
                  prob_arr[tempIdx] = 0.
                  tempIdx = np.where(deltaProb >= -50)[0] # then does usual calculations for the other probabilities
                  prob_arr[tempIdx] = np.exp(prob_arr[tempIdx]-np.min(prob_arr[tempIdx]))

                else:
                  prob_arr = np.exp(prob_arr - np.min(prob_arr))

            prob_arr = prob_arr/np.sum(prob_arr) # eg [2.50705025e-04 9.40316822e-01 5.94324733e-02]

            self.GBeagle[i] = self.rng.multinomial(1, prob_arr) # eg [0 1 0]
            tempIdx = np.where(self.GBeagle[i] == 1)[0] # eg [1]

            self.x[i] = self.xArr[i,tempIdx]
            self.y[i] = self.yArr[i,tempIdx]
            self.z[i] = self.zArr[i,tempIdx]
            self.xsig[i] = self.xsigArr[i,tempIdx]
            self.ysig[i] = self.ysigArr[i,tempIdx]
            self.zsig[i] = self.zsigArr[i,tempIdx]
            self.xycov[i] = self.xycovArr[i,tempIdx]
            self.xzcov[i] = self.xzcovArr[i,tempIdx]
            self.yzcov[i] = self.yzcovArr[i,tempIdx]
            self.xycorr[i] = self.xycov[i] / (self.xsig[i] * self.ysig[i])
            self.xzcorr[i] = self.xzcov[i] / (self.xsig[i] * self.zsig[i])
            self.yzcorr[i] = self.yzcov[i] / (self.ysig[i] * self.zsig[i])
            self.xvar[i] = self.xsig[i]**2
            self.yvar[i] = self.ysig[i]**2
            self.zvar[i] = self.zsig[i]**2

    def update_chain(self):
        ximean = np.sum(self.pi * self.mu)
        xisig = np.sqrt(np.sum(self.pi * (self.tausqr + self.mu**2)) - ximean**2)
        self.chain['alphaN_a'][self.ichain] = self.alphaN_a
        self.chain['alphaN_b'][self.ichain] = self.alphaN_b
        self.chain['alphaN_c'][self.ichain] = self.alphaN_c
        self.chain['beta_a'][self.ichain] = self.beta_a
        self.chain['beta_b'][self.ichain] = self.beta_b
        self.chain['beta_c'][self.ichain] = self.beta_c
        self.chain['sig0'][self.ichain] = self.sig0
        self.chain['k'][self.ichain] = self.k
        self.chain['xi_min'][self.ichain] = self.xi_min
        self.chain['xi_max'][self.ichain] = self.xi_max
        self.chain['xi'][self.ichain] = self.xi
        self.chain['eta'][self.ichain] = self.eta
        self.chain['zeta'][self.ichain] = self.zeta
        self.chain['pi'][self.ichain] = self.pi
        self.chain['mu'][self.ichain] = self.mu
        self.chain['tausqr'][self.ichain] = self.tausqr
        self.chain['mu0'][self.ichain] = self.mu0
        self.chain['usqr'][self.ichain] = self.usqr
        self.chain['wsqr'][self.ichain] = self.wsqr
        self.chain['ximean'][self.ichain] = ximean
        self.chain['xisig'][self.ichain] = xisig
        self.chain['outlier_mean'][self.ichain] = self.outlier_mean
        self.chain['outlier_sigma'][self.ichain] = self.outlier_sigma
        self.chain['pbad'][self.ichain] = max(self.pbad)

        self.chain['alphaN_a_prop'][self.ichain] = self.alphaN_a_prop
        self.chain['alphaN_b_prop'][self.ichain] = self.alphaN_b_prop
        self.chain['beta_a_prop'][self.ichain] = self.beta_a_prop
        self.chain['beta_b_prop'][self.ichain] = self.beta_b_prop

        if self.ichain % self.nBurn == 0:
            self.chain_propscales['sig0_propscale'][int(self.ichain / self.nBurn)] = self.sig0_propscale
            self.chain_propscales['k_propscale'][int(self.ichain / self.nBurn)] = self.k_propscale
            self.chain_propscales['pbad_propscale'][int(self.ichain / self.nBurn)] = self.pbad_propscale
            self.chain_propscales['outlier_mean_propscale'][int(self.ichain / self.nBurn)] = self.outlier_mean_propscale
            self.chain_propscales['outlier_sigma_propscale'][int(self.ichain / self.nBurn)] = self.outlier_sigma_propscale
            self.chain_propscales['xi_propscale'][int(self.ichain / self.nBurn)] = self.xi_propscale
            self.chain_propscales['eta_propscale'][int(self.ichain / self.nBurn)] = self.eta_propscale
            self.chain_propscales['zeta_propscale'][int(self.ichain / self.nBurn)] = self.zeta_propscale

        self.ichain += 1

        if self.ichain == minIter:
            pickle.dump(self.chain_propscales, open('./lm_chain_scenario_{}_{}_{}_{}x{}_30_101_propscales.p'.format(scenarioA, field, z_bin, nChains, minIter),'wb'))


    # =============================================================================
    # END OF STEP
    # =============================================================================

    # =============================================================================
    # INITIALISE, STEP and EXTEND
    # =============================================================================

    def initialize_chain(self, minIter, nBurn, testIter):
        self.chain_dtype = [('alphaN', (float, self.N)),
                            ('alphaN_a', float),
                            ('alphaN_b', float),
                            ('alphaN_c', float),
                            ('beta', (float, self.N)),
                            ('beta_a', float),
                            ('beta_b', float),
                            ('beta_c', float),
                            ('sig0', float),
                            ('k', float),
                            ('xi_min', float),
                            ('xi_max', float),
                            ('xi', (float, self.N)),
                            ('eta', (float, self.N)),
                            ('zeta', (float, self.N)),
                            ('outlier_mean', float),
                            ('outlier_sigma', float),
                            ('pbad', float),
                            ('pi', (float, self.nGaussXi)),
                            ('mu', (float, self.nGaussXi)),
                            ('tausqr', (float, self.nGaussXi)),
                            ('mu0', float),
                            ('usqr', float),
                            ('wsqr', float),
                            ('ximean', float),
                            ('xisig', float),
                            ('alphaN_a_prop', float),
                            ('alphaN_b_prop', float),
                            ('beta_a_prop', float),
                            ('beta_b_prop', float)

                            ]

        self.chain = np.empty((minIter), dtype=self.chain_dtype)
        self.ichain = 0
        self.nBurn = nBurn
        self.testIter = testIter

        self.chain_propscales_dtype = [
                            ('sig0_propscale', float),
                            ('k_propscale', float),
                            ('pbad_propscale', float),
                            ('outlier_mean_propscale', float),
                            ('outlier_sigma_propscale', float),
                            ('xi_propscale', (float, self.N)),
                            ('eta_propscale', (float, self.N)),
                            ('zeta_propscale', (float, self.N))

                            ]

        self.chain_propscales = np.empty(int(minIter / self.nBurn), dtype=self.chain_propscales_dtype)

    def step(self, niter):
        for i in range(niter):
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            np.seterr(**old_settings)

            self.update_cens_y()
            self.update_xi()
            self.update_eta()
            self.update_zeta() # new redshift step
            self.update_G()
            self.update_alphaN_beta_sigma_outlier_model()
            self.update_pi()
            self.update_mu()
            self.update_tausqr()
            self.update_mu0()
            self.update_usqr()
            self.update_wsqr()
            self.update_GBeagle() # THIS IS ADDED FOR GMM
            self.update_chain()

            if (i+1)%(niter/10.0) == 0:
                print(i+1, niter)

    def extend(self, length):
        extension = np.empty((length), dtype=self.chain_dtype)
        self.chain = np.hstack((self.chain, extension))

    # =============================================================================
    # ADDED FOR MH STEP
    # =============================================================================

    def dunif(self, val, min, max):
        if val > max or val < min:
           return 0.
        else:
           return 1./(max-min)

    def prior_alphaN(self,alphaN_a,alphaN_b,alphaN_c):
        min_alphaN_a = -3.0
        max_alphaN_a = 2.3
        
        min_alphaN_b = 0.0
        max_alphaN_b = 5.0

        min_alphaN_c = -1e6
        max_alphaN_c = 1e6
        
        a = self.dunif(alphaN_a, min_alphaN_a, max_alphaN_a)
        b = self.dunif(alphaN_b, min_alphaN_b, max_alphaN_b)
        c = self.dunif(alphaN_c, min_alphaN_c, max_alphaN_c)
        return a*b*c

    def prior_beta(self,beta_a,beta_b,beta_c):
        min_beta_a = -5.0
        min_beta_b = -5.0
        min_beta_c = -1e6
        max_beta_a = 5.0
        max_beta_b = 5.0
        max_beta_c = 1e6
        a = self.dunif(beta_a, min_beta_a, max_beta_a)
        b = self.dunif(beta_b, min_beta_b, max_beta_b)
        c = self.dunif(beta_c, min_beta_c, max_beta_c)
        return a*b*c

    def prior_sig0(self, sig0):
        min_sig0 = 0.05
        max_sig0 = 5.0
        return self.dunif(sig0, min_sig0, max_sig0)

    def prior_k(self, k):
        min_k = 0.0
        max_k = 5.0
        return self.dunif(k, min_k, max_k)

    def prior_outlier_mean(self, outlier_mean):
        min_outlier_mean = -10.0
        max_outlier_mean = 10.0
        return self.dunif(outlier_mean, min_outlier_mean, max_outlier_mean)

    def prior_outlier_sigma(self, outlier_sigma):
        min_outlier_sigma = 1.0
        max_outlier_sigma = 10.0
        return self.dunif(outlier_sigma, min_outlier_sigma, max_outlier_sigma)

    def prior_pbad(self, pbad):
        min_pbad = 0.0
        max_pbad = 0.5
        return self.dunif(pbad, min_pbad, max_pbad)

    def prior_all(self, alphaN_a, alphaN_b, alphaN_c, beta_a, beta_b, beta_c, sig0, k, pbad, outlier_mean, outlier_sigma):
        prior_all_output = self.prior_alphaN(alphaN_a,alphaN_b,alphaN_c)*self.prior_beta(beta_a,beta_b,beta_c)*self.prior_sig0(sig0)*self.prior_k(k)*self.prior_pbad(pbad)*self.prior_outlier_mean(outlier_mean)*self.prior_outlier_sigma(outlier_sigma)
        if prior_all_output == 0:
            print('prior_all_output == 0')

        return prior_all_output

    def calc_sigsqr(self, xi, sig0=None, k=None, xi_min=None, xi_max=None):
        if sig0 == None:
            sig0 = self.sig0
        if k == None:
            k = self.k
        if xi_min == None:
            xi_min = self.xi_min
        if xi_max == None:
            xi_max = self.xi_max

        sigsqr = ( sig0 * ( ((1.0-k)*(xi-xi_max)/(xi_max-xi_min)) + 1.0 ) ) ** 2.0
        return sigsqr
    
    def calc_outlier_prob(self, eta, loc, scale):
        outlier_low = -2.0
        outlier_high = 3.75
        outlier_norm = norm.cdf(outlier_high, loc, scale) - norm.cdf(outlier_low, loc, scale)
        outlier_prob = norm.pdf(eta, loc=loc, scale=scale) / outlier_norm
        ### CAN SET OUTLIER_NORM = 1 to remove this effect, and comment out both outlier_idx lines below
        outlier_idx = (eta < outlier_low) | (eta > outlier_high)
        outlier_prob[outlier_idx] = 0.0
        return outlier_prob
    
    # =============================================================================
    # REDSHIFT DEPENDENCE
    # =============================================================================

    def get_3d_mean_cov(self, i):

        x = self.x[i]
        y = self.y[i]
        z = self.z[i]

        xvar = self.xvar[i]
        yvar = self.yvar[i]
        zvar = self.zvar[i]

        xycov = self.xycov[i]
        xzcov = self.xzcov[i]
        yzcov = self.yzcov[i]

        mean = np.array([x,y,z])
        cov = np.array([[xvar, xycov, xzcov],[xycov, yvar, yzcov],[xzcov, yzcov, zvar]])

        return mean, cov

    # =============================================================================
    # added for SSFR fitting - SSFR @ mass alphaNorm = alphaN_a * (1 + z)**alphaN_b
    # =============================================================================

    def calc_sfr_surface(self, X, beta_a, beta_b, alphaN_a, alphaN_b):
        x,z = X

        return ((beta_a + beta_b*z)*(x-self.alphaNorm)) + alphaN_a + alphaN_b*np.log10(1.0+z) + self.alphaNorm - 9.0 # new alphaN_a = old np.log10(alphaN_a)

class LinMix(object):
    """ A class to perform linear regression of `y` on `x` when there are measurement errors in
    both variables.  The regression assumes:

    eta = alpha + beta * xi + epsilon

    x = xi + xerr

    y = eta + yerr

    Here, `alpha` and `beta` are the regression coefficients, `epsilon` is the intrinsic random
    scatter about the regression, `xerr` is the measurement error in `x`, and `yerr` is the
    measurement error in `y`.  `epsilon` is assumed to be normally-distributed with mean zero and
    variance `sigsqr`.  `xerr` and `yerr` are assumed to be normally-distributed with means equal
    to zero, variances `xsig`^2 and `ysig`^2, respectively, and covariance `xycov`. The
    distribution of `xi` is modelled as a mixture of normals, with group proportions `pi`, means
    `mu`, and variances `tausqr`.

    Args:
        x(array_like): The observed independent variable.
        y(array_like): The observed dependent variable.
        xsig(array_like): 1-sigma measurement errors in x.
        ysig(array_like): 1-sigma measurement errors in y.
        xycov(array_like): Covariance between the measurement errors in x and y.
        delta(array_like): Array indicating whether a data point is censored (i.e., not detected),
            or not.  If delta[i] == 1, then the ith source is detected.  If delta[i] == 0, then
            the ith source is not detected and y[i] will be interpreted as an upper limit.  Note
            that if there are censored data points, then the maximum-likelihood estimate
            (alpha, beta, sigsqr) is not valid.  By default, all data points are assumed to be
            detected.
        nGaussXi(int): The number of Gaussians to use in the mixture model for the distribution of xi.
        nChains(int): The number of Monte Carlo Markov Chains to instantiate.
        parallelize(bool): Use a separate thread for each chain.  Only makes sense for nChains > 1.
        seed(int): Random seed.  If `None`, then get seed from np.random.randint().

    Attributes:
        nChains(int): The number of instantiated MCMCs.
        chain(numpy recarray): The concatenated MCMCs themselves.  Actually, only the concatenation
            of the last half of each chain is stored here after convergence is reached.  The
            recarray has the following columns:
                - alpha(float): The regression intercept.
                - beta(float): The regression slope.
                - sigsqr(float): The regression intrinsic scatter.
                - pi(array_like): The mixture model component fractions.
                - mu(array_like): The mixture model component means.
                - tausqr(array_like): The mixture model component variances.
                - mu0(float): The hyperparameter describing the prior variance of the distribution
                    of mixture means.
                - usqr(float): The hyperparameter describing the prior variance of the distribution
                    of mixture variances.
                - wsqr(float): The hyperparameter describing the typical scale for the prior on
                    `usqr` and `tausqr`.
                - ximean(float): The mean of the distribution for the independent latent variable
                    `xi`.
                - xisig(float): The standard deviation of the distribution for the independent
                    latent variable `xi`.
                - corr(float): The linear correlation coefficient between the latent dependent and
                    independent variables `xi` and `eta`.
    """
    def __init__(self, xArr, yArr, zArr, xsigArr, ysigArr, zsigArr, xycovArr, xzcovArr, yzcovArr, proposalscale_xi, proposalscale_eta, proposalscale_zeta, proposalscale_alphaN_a, proposalscale_alphaN_b, proposalscale_alphaN_c, proposalscale_beta_a, proposalscale_beta_b, proposalscale_beta_c, proposalscale_sig0, proposalscale_k, proposalscale_pbad, proposalscale_outlier_mean, proposalscale_outlier_sigma, alphaNorm, z_lower, z_upper, delta=None, nGaussXi=3, nChains=4, parallelize=True, seed=None, nGaussBeagle=1, piBeagle=None):
        
        self.nChains = nChains
        self.parallelize = parallelize

        if seed is None:
            seed = np.random.randint(2**31)

        if piBeagle is None:
            piBeagle = np.ones(len(xArr)) # lester changed x to xArr
        if self.parallelize:
            # Will place 1 chain in 1 thread.
            from multiprocessing import Process, Pipe
            # Create a pipe for each thread.
            self.pipes = []
            slave_pipes = []
            for i in range(self.nChains):
                master_pipe, slave_pipe = Pipe()
                self.pipes.append(master_pipe)
                slave_pipes.append(slave_pipe)

            # Create chain pool.
            self.pool = []
            for sp in slave_pipes:
                self.pool.append(Process(target=task_manager, args=(sp,)))
                self.pool[-1].start()

            init_kwargs0 = {'xArr':xArr,
                            'yArr':yArr,
                            'zArr':zArr,
                            'xsigArr':xsigArr,
                            'ysigArr':ysigArr,
                            'zsigArr':zsigArr,
                            'xycovArr':xycovArr,
                            'xzcovArr':xzcovArr,
                            'yzcovArr':yzcovArr,
                            'proposalscale_xi':proposalscale_xi,
                            'proposalscale_eta':proposalscale_eta,
                            'proposalscale_zeta':proposalscale_zeta,
                            'proposalscale_alphaN_a':proposalscale_alphaN_a,
                            'proposalscale_alphaN_b':proposalscale_alphaN_b,
                            'proposalscale_alphaN_c':proposalscale_alphaN_c,
                            'proposalscale_beta_a':proposalscale_beta_a,
                            'proposalscale_beta_b':proposalscale_beta_b,
                            'proposalscale_beta_c':proposalscale_beta_c,
                            'proposalscale_sig0':proposalscale_sig0,
                            'proposalscale_k':proposalscale_k,
                            'proposalscale_pbad':proposalscale_pbad,
                            'proposalscale_outlier_mean':proposalscale_outlier_mean,
                            'proposalscale_outlier_sigma':proposalscale_outlier_sigma,
                            'delta':delta,
                            'nGaussXi':nGaussXi,
                            'nChains':self.nChains,
                            'nGaussBeagle':nGaussBeagle,
                            'piBeagle':piBeagle,
                            'alphaNorm':alphaNorm,
                            'z_lower':z_lower,
                            'z_upper':z_upper}

            for i, p in enumerate(self.pipes):
                init_kwargs = init_kwargs0.copy()
                init_kwargs['rng'] = np.random.RandomState(seed+i)
                p.send({'task':'init',
                        'init_args':init_kwargs})
        else:
            self._chains = []
            for i in range(self.nChains):
                self._chains.append(Chain(xArr, yArr, zArr, xsigArr, ysigArr, zsigArr, xycovArr, xzcovArr, yzcovArr, delta, nGaussXi, nChains, nGaussBeagle, piBeagle, proposalscale_xi, proposalscale_eta, proposalscale_zeta, proposalscale_alphaN_a, proposalscale_alphaN_b, proposalscale_alphaN_c, proposalscale_beta_a, proposalscale_beta_b, proposalscale_beta_c, proposalscale_sig0, proposalscale_k, proposalscale_pbad, proposalscale_outlier_mean, proposalscale_outlier_sigma, alphaNorm, z_lower, z_upper, rng=None))

                self._chains[-1].initial_guess()

    def _get_psi(self):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'fetch',
                        'key':'chain'})
            chains = [p.recv() for p in self.pipes]
            self.pipes[0].send({'task':'fetch',
                                'key':'ichain'})
            ndraw = int(self.pipes[0].recv()/2) # half length of total chain so far
        else:
            chains = [c.chain for c in self._chains]
            ndraw = int(self._chains[0].ichain/2) # half length of total chain so far
        psi={}
        keys=['alphaN_a','alphaN_b','alphaN_c','beta_a','beta_b','beta_c','sig0','k','pbad','outlier_mean','outlier_sigma']
        for key in keys:
            psi[key] = np.vstack([c[key][ndraw:2*ndraw] for c in chains]).T # c['alpha'] is minIter long

        # this multiplies [pi1, pi2, pi3] by [mu1, mu2, mu3] per iteration -> [pi1mu1, pi2mu2, pi3mu3]
        # then adds them up, to get pi1mu1 + pi2mu2 + pi3mu3 per iteration
        # then creates [ch1, ch2, ch3, ch4] x number of iterations
        psi['ximean'] = np.vstack([np.sum(c['pi'][ndraw:2*ndraw] * c['mu'][ndraw:2*ndraw], axis=1) for c in chains]).T

        # similar to above, but gets [pi1*(tausqr1+mu1), pi2*(tausqr2+mu2), pi3*(tausqr3+mu3)]
        # which becomes [pi1*(tausqr1+mu1) + pi2*(tausqr2+mu2) + pi3*(tausqr3+mu3)] x number of iterations
        psi['xivar'] = np.vstack([np.sum(c['pi'][ndraw:2*ndraw] * (c['tausqr'][ndraw:2*ndraw] + c['mu'][ndraw:2*ndraw]**2), axis=1) for c in chains]).T - psi['ximean']**2

        keys=['alphaN_a','alphaN_b','alphaN_c','beta_a','beta_b','beta_c','sig0','k','pbad','outlier_mean','outlier_sigma','ximean','xivar']
        for key in keys:
            psi[key] = np.hstack((psi[key][:int(ndraw/2)],psi[key][-int(ndraw/2):]))
        return psi

    def _get_Rhat(self):
        psi = self._get_psi()
        keys=['alphaN_a','alphaN_b','alphaN_c','beta_a','beta_b','beta_c','sig0','k','pbad','outlier_mean','outlier_sigma','ximean','xivar']
        n = len(psi[keys[0]]) # quarter of initial iterations, half discarded, then split in half
        m = len(psi[keys[0]][0]) # 2x number of chains, usually = 8
        psi_bar_dot_j = {}
        psi_bar_dot_dot = {}
        B = {}
        s_j_sqr = {}
        W = {}
        var_plus = {}
        Rhat = {}

        for key in keys:
            psi_bar_dot_j[key] = np.empty(m, dtype=float)
            psi_bar_dot_j[key] = np.mean(psi[key], axis=0) # 8x avg values of each chain
            psi_bar_dot_dot[key] = np.mean(psi_bar_dot_j[key]) # 1x average of averages
            B[key] = (m*n/(m-1.0)) * np.mean((psi_bar_dot_j[key]-psi_bar_dot_dot[key])**2) # between sequence
            s_j_sqr[key] = np.empty(m, dtype=float)
            s_j_sqr[key] = np.sum((psi[key]-psi_bar_dot_j[key])**2, axis=0) / (n-1.0)
            W[key] = np.mean(s_j_sqr[key]) # within sequence
            var_plus[key] = (((n-1.0)/n)*W[key]) + (B[key]/n)
            Rhat[key] = np.sqrt(var_plus[key]/W[key])
        return Rhat

    def _initialize_chains(self, minIter, nBurn, testIter):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'init_chain',
                        'minIter':minIter,'nBurn':nBurn,'testIter':testIter})
        else:
            for c in self._chains:
                c.initialize_chain(minIter,nBurn,testIter)

    def _step(self, niter):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'step',
                        'niter':niter})
        else:
            for c in self._chains:
                c.step(niter)

    def _extend(self, niter):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'extend',
                        'niter':niter})
        else:
            for c in self._chains:
                c.extend(niter)

    def _build_chain(self, ikeep):
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'fetch',
                        'key':'chain'})
            self.chain = np.hstack([p.recv()[ikeep:] for p in self.pipes])
        else:
            self.chain = np.hstack([c.chain[ikeep:] for c in self._chains])

    def run_mcmc(self, minIter=5000, maxIter=100000, silent=False, nBurn=500, checkIter=5000, RhatLimit=5000, testIter=200):
        """ Run the Markov Chain Monte Carlo for the LinMix object.

        Bayesian inference is employed, and a Markov chain containing random draws from the
        posterior is developed.  Convergence of the MCMC to the posterior is monitored using the
        potential scale reduction factor (RHAT, Gelman et al. 2004). In general, when RHAT < 1.1
        then approximate convergence is reached.  After convergence is reached, the second halves
        of all chains are concatenated and stored in the `.chain` attribute as a numpy recarray.

        Args:
            minIter(int): The minimum number of iterations to use.
            maxIter(int): The maximum number of iterations to use.
            silent(bool): If true, then suppress updates during sampling.
        """

        self._initialize_chains(minIter,nBurn,testIter)

        Rhat_array = []
        for i in range(0, minIter, checkIter): # 0 to minIter, whilst checking for convergence every "checkIter"
            self._step(checkIter)
            Rhat = self._get_Rhat()
            Rhat_array.append(Rhat)
            print()
            print('Iteration: ', i+checkIter)
            print('Rhat values:')
            print('alphaN:', Rhat['alphaN_a'], Rhat['alphaN_b'], Rhat['alphaN_c'])
            print('beta:', Rhat['beta_a'], Rhat['beta_b'], Rhat['beta_c'])
            print('sig0:', Rhat['sig0'])
            print('k:', Rhat['k'])
            print('ximean:', Rhat['ximean'])
            print('xivar:', Rhat['xivar'])
            print('pbad:', Rhat['pbad'])
            print('outlier_mean:', Rhat['outlier_mean'])
            print('outlier_sigma:', Rhat['outlier_sigma'])
            print()

        i += checkIter

        while (max(np.array(list(Rhat.values()))[~np.isnan(list(Rhat.values()))]) > RhatLimit) and (i < maxIter):
            self._extend(checkIter)
            self._step(checkIter)
            Rhat = self._get_Rhat()
            Rhat_array.append(Rhat)
            print()
            print('Extended Iteration: ', i+checkIter)
            print('Rhat values:')
            print('alphaN:', Rhat['alphaN_a'], Rhat['alphaN_b'], Rhat['alphaN_c'])
            print('beta:', Rhat['beta_a'], Rhat['beta_b'], Rhat['beta_c'])
            print('sig0:', Rhat['sig0'])
            print('k:', Rhat['k'])
            print('ximean:', Rhat['ximean'])
            print('xivar:', Rhat['xivar'])
            print('pbad:', Rhat['pbad'])
            print('outlier_mean:', Rhat['outlier_mean'])
            print('outlier_sigma:', Rhat['outlier_sigma'])
            print()
            i += checkIter

        self._build_chain(0) # this means build chain from 0th iteration opposed to int(i/2) etc.

        # Clean up threads
        if self.parallelize:
            for p in self.pipes:
                p.send({'task':'kill'})


#%%
# =============================================================================
# INPUTS
# =============================================================================

scenarioA = '34'
field = 'clusters'
z_bin = 'z1p25-6p0'
z_lower = 1.25
z_upper = 6.0

#with open('./scenario_{}_{}_{}.p'.format(scenarioA, field, z_bin), 'rb') as f:
#    data = pickle.load(f)

filepath = './scenario_{}_{}_{}.fits'.format(scenarioA, field, z_bin)
temp = fits.open(filepath)
temp = temp[1].data

data = {}
for name in temp.names:
    data[name] = temp[name]

piBeagle    = data['amp_GMM_3d']    # 3x probability of each posterior gaussian
GMMx        = data['x_GMM_3d']      # 3x posterior means per mass
GMMy        = data['y_GMM_3d']      # 3x posterior means per sfr
GMMxsig     = data['xsig_GMM_3d']   # 3x posterior sigmas per mass
GMMysig     = data['ysig_GMM_3d']   # 3x posterior sigmas per sfr
GMMxycov    = data['xycov_GMM_3d']  # 3x posterior covar per mass-sfr pair

GMMz        = data['z_GMM_3d']      # 3x posterior means per redshift
GMMzsig     = data['zsig_GMM_3d']   # 3x posterior sigmas per redshift
GMMxzcov    = data['xzcov_GMM_3d']  # 3x posterior covar per mass-redshift pair
GMMyzcov    = data['yzcov_GMM_3d']  # 3x posterior covar per sfr-redshift pair

nGaussXi                = 3         # 3 #gaussians modelling xi
nGaussBeagle            = 3         # 3 #gaussians modelling BEAGLE posterior

nChains                 = 4
minIter                 = 500      # if i%(niter/10.0) == 0:, TENTH OF THIS NEEDS TO DIVIDE BY 4, see yellow below
maxIter                 = minIter
checkIter               = minIter   # must be divisible by 4
nBurn                   = 50       # this is when the MH proposal distributions start adapting (xi, eta, zeta, alpha, beta, sig0 and k)
testIter                = 5        # xi, eta, zeta MH proposal checks every "testIter" iterations and adjusts proposal if needed
RhatLimit               = 1.01
alphaNorm               = 9.7       # value at which to normalise alpha, ie the sfr value the relation has given this mass, Santini use 9.7

proposalscale_xi                = 0.1
proposalscale_eta               = 0.1
proposalscale_zeta              = 0.1

proposalscale_beta_a            = 0.001
proposalscale_beta_b            = 0.0
proposalscale_alphaN_a          = 0.0005
proposalscale_alphaN_b          = 0.01
proposalscale_sig0              = 0.05

proposalscale_pbad              = 0.01
proposalscale_outlier_mean      = 0.1
proposalscale_outlier_sigma     = 0.1

proposalscale_alphaN_c          = 0.0
proposalscale_beta_c            = 0.0
proposalscale_k                 = 0.0

parallelize                     = True


# =============================================================================
# RUN 
# =============================================================================

lm = LinMix(GMMx, GMMy, GMMz, GMMxsig, GMMysig, GMMzsig, xycovArr=GMMxycov, xzcovArr=GMMxzcov, yzcovArr=GMMyzcov, nGaussXi=nGaussXi, nGaussBeagle=nGaussBeagle, piBeagle=piBeagle, nChains=nChains, parallelize=parallelize, proposalscale_xi=proposalscale_xi, proposalscale_eta=proposalscale_eta, proposalscale_zeta=proposalscale_zeta, proposalscale_alphaN_a=proposalscale_alphaN_a, proposalscale_alphaN_b=proposalscale_alphaN_b, proposalscale_alphaN_c=proposalscale_alphaN_c, proposalscale_beta_a=proposalscale_beta_a, proposalscale_beta_b=proposalscale_beta_b, proposalscale_beta_c=proposalscale_beta_c, proposalscale_sig0=proposalscale_sig0, proposalscale_k=proposalscale_k, proposalscale_pbad=proposalscale_pbad, proposalscale_outlier_mean=proposalscale_outlier_mean, proposalscale_outlier_sigma=proposalscale_outlier_sigma, alphaNorm=alphaNorm, z_lower=z_lower, z_upper=z_upper)

lm.run_mcmc(minIter=minIter, maxIter=maxIter, nBurn=nBurn, checkIter=checkIter, testIter=testIter, RhatLimit=RhatLimit)


# =============================================================================
# SAVE CHAINS
# =============================================================================
pickle.dump(lm.chain, open('./lm_chain_scenario_{}_{}_{}_{}x{}_30_101.p'.format(scenarioA, field, z_bin, nChains, minIter),'wb'))


