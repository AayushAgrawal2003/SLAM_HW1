'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import random


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        
        
        


        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        
        n = len(X_bar) 
        cumulative_sum = X_bar[0][3]
        r = random.uniform(1e-8,1/n)
        u_value = r
        
        X_bar_resampled = []
        num = 0
        for i in range(n):
            u_value = r + (i)/n
            
            while u_value > cumulative_sum and num < n-1:
                cumulative_sum += X_bar[num][3]
                num += 1
            X_bar_resampled.append(X_bar[num-1])
        wt_sum = sum(row[3] for row in X_bar_resampled)
        X_bar_resampled = [[a,b,c,d/wt_sum] for a,b,c,d in X_bar_resampled]
        # ipdb.set_trace()
        
        return np.array(X_bar_resampled)