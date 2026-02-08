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
        X_bar_resampled =  np.zeros_like(X_bar)
        weights = X_bar[:,3]
        sum_of_weights = sum(weights) + 1e-8
        new_weights = weights/sum_of_weights
        random_number = 0
        sum_ = 0
        index = 0
        for i in range(len(weights)):
            sum_+=new_weights[i]
            if sum_>=random_number:
                index = i
                break
        
        


        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        wt_sum = sum(row[3] for row in X_bar)
        n = len(X_bar) 
        cumulative_sum = 0
        r = random.uniform(0,1/n)
        u = r
        X_bar = [[a,b,c,d/wt_sum] for a,b,c,d in X_bar]
        X_bar_resampled = []
        num = 0
        for i in range(n):
            u = r + (i)/n
            while u > cumulative_sum and num < n:
                cumulative_sum += X_bar[num][3]
                num += 1
            X_bar_resampled.append(X_bar[min(num, n-1)])
        X_bar_resampled = np.array(X_bar_resampled)
        X_bar_resampled[:, 3] = 1.0 / n

        return X_bar_resampled
