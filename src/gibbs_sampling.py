#Functions used for gibbs sampling
# Conor O'Sullivan
#02 April 2019

import pandas as pd
import numpy as np

class gibbs:
    def gibbs_difference(y, ind, mu0 = 50, tau0 = 1/625, del0 = 0, gamma0 = 1/625, a0 = 0.5, b0 = 50, maxiter = 5000):
        y1 = y[ind == 1]
        y2 = y[ind == 2]

        n1 = len(y1)
        n2 = len(y2)

        #initial values
        mu = (y1.mean() + y2.mean()) / 2
        delta = (y1.mean() - y2.mean()) / 2

        df_samples = pd.DataFrame(columns=["mu", "del", "tau",'theta'])

        ##### Gibbs sampler
        an = a0 + (n1 + n2)/2

        for i in range(maxiter):

            ##update tau
            bn = b0 + 0.5 * (sum((y1 - mu - delta)**2) + sum((y2 - mu + delta)**2))
            tau = np.random.gamma(an, 1/bn)
            ##

            ##update mu
            taun =  tau0 + tau * (n1 + n2)
            mun = (tau0 * mu0 + tau * (sum(y1 - delta) + sum(y2 + delta))) / taun
            mu = np.random.normal(mun, np.sqrt(1/taun))
            ##

            ##update delta
            gamman =  gamma0 + tau*(n1 + n2)
            deln = ( del0 * gamma0 + tau * (sum(y1 - mu) - sum(y2 - mu))) / gamman
            delta = np.random.normal(deln, np.sqrt(1/gamman))

            df_samples.loc[i] = [mu, delta, tau,1/np.sqrt(tau)]

        return df_samples

    def gibbs_m(y, ind, mu0 = 50, gamma0 = 1/25,eta0 = 1/2, t0 = 50, a0 = 1/2, b0 = 50, maxiter = 5000):

        ### starting values
        m = ind.nunique()
        ybar = theta = [np.mean(y[ind == i]) for i in ind.unique()]
        tau_w = np.mean([1/np.var(y[ind == i]) for i in ind.unique()]) ##within group precision
        mu = np.mean(theta)
        tau_b = 1/np.var(theta) ##between group precision
        n_m = [len(y[ind == i]) for i in ind.unique()]
        an = a0 + sum(n_m)/2


        ### setup MCMC
        theta_mat = pd.DataFrame(columns = list(ind.unique()))
        mat_store = pd.DataFrame(columns = ["mu", "tau_w", "tau_b","theta_w", "theta_b"])

        for i in range(maxiter):

            # sample new values of the thetas
            theta = []
            for j in range(m):

                taun = n_m[j]*tau_w + tau_b
                thetan = (ybar[j] * n_m[j] * tau_w + mu * tau_b) / taun
                theta.append(np.random.normal(thetan, np.sqrt(1/taun)))

            #sample new value of tau_w

            ss = 0
            for j in range(m):
                ss = ss + sum([ (x - theta[j])**2 for x in y[ind == j+1]])

            bn = b0 + ss/2
            tau_w = np.random.gamma(an, 1/bn)

            #sample a new value of mu
            gammam = m * tau_b + gamma0
            mum = (np.mean(theta) * m * tau_b + mu0 * gamma0) / gammam
            mu = np.random.normal(mum, np.sqrt(1/gammam))

            # sample a new value of tau_b
            etam = eta0 + m/2
            tm = t0 + sum([(t-mu)**2 for t in theta])/2
            tau_b = np.random.gamma(etam, 1/tm)

            #store results
            theta_mat.loc[i] = theta
            mat_store.loc[i] = [mu, tau_w, tau_b,1/np.sqrt(tau_w),1/np.sqrt(tau_b) ]

            if i%500 == 0: print("{}/{}".format(i,maxiter))

        return (theta_mat,mat_store)
