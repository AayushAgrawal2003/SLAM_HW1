'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        self._max_range = 1000

        self._min_probability = 0.35

        self._subsampling = 2

        self._occupancy_map = occupancy_map

    def ray_casting(self, x_t1, angle):
        x, y, theta = x_t1

        laser_x = x + 25.0 * math.cos(theta)
        laser_y = y + 25.0 * math.sin(theta)

        # Ray direction: robot heading + beam angle
        ray_angle = theta + angle

        step_size = 5  # resolution in map units (cm)
        max_range = self._max_range

        dist = 0
        while dist < max_range:
            dist += step_size

            # Compute the endpoint of the ray at current distance
            ray_x = laser_x + dist * math.cos(ray_angle)
            ray_y = laser_y + dist * math.sin(ray_angle)

            map_x = int(ray_x / 10.0)
            map_y = int(ray_y / 10.0)

            # Check bounds
            if map_x < 0 or map_x >= self._occupancy_map.shape[1] or \
               map_y < 0 or map_y >= self._occupancy_map.shape[0]:
                return max_range

            # Check if this cell is occupied
            # Occupancy map: values < min_probability => occupied (obstacle)
            # Values close to 0 are obstacles, close to 1 are free, -1 is unknown
            cell_val = self._occupancy_map[map_y, map_x]
            if cell_val < self._min_probability and cell_val >= 0:
                return dist

        return max_range

    def beam_range_finder_model(self, z_t1_arr, x_t1):

        # Intrinsic parameters
        z_hit = self._z_hit
        z_short = self._z_short
        z_max = self._z_max
        z_rand = self._z_rand
        sigma_hit = self._sigma_hit
        lambda_short = self._lambda_short
        max_range = self._max_range

        # Normalize mixing weights
        w_sum = z_hit + z_short + z_max + z_rand
        z_hit /= w_sum
        z_short /= w_sum
        z_max /= w_sum
        z_rand /= w_sum

        # Log probability for numerical stability
        log_prob = 0.0

        # Subsample the 180 beams
        # Beam angles go from -90 to +90 degrees (pi radians), one per degree
        for k in range(0, 180, self._subsampling):
            z_k = z_t1_arr[k]  # actual measurement

            # Angle of this beam relative to robot heading
            # Beams are indexed 0..179 corresponding to -90..+89 degrees
            beam_angle = math.radians(-90 + k)

            # Ray cast to find expected (true) range z_k_star
            z_k_star = self.ray_casting(x_t1, beam_angle)

            # --- Compute the four density components ---

            # 1. p_hit: Gaussian centered at z_k_star, truncated to [0, max_range]
            if 0 <= z_k <= max_range:
                # Normalizer for truncated Gaussian
                eta_hit = 1.0 / (norm.cdf(max_range, loc=z_k_star, scale=sigma_hit)
                                 - norm.cdf(0, loc=z_k_star, scale=sigma_hit))
                p_hit = eta_hit * norm.pdf(z_k, loc=z_k_star, scale=sigma_hit)
            else:
                p_hit = 0.0

            if 0 <= z_k <= z_k_star and z_k_star > 0:
                eta_short = 1.0 / (1.0 - math.exp(-lambda_short * z_k_star))
                p_short = eta_short * lambda_short * math.exp(-lambda_short * z_k)
            else:
                p_short = 0.0

            if abs(z_k - max_range) < 1e-3:
                p_max_val = 1.0
            else:
                p_max_val = 0.0

            if 0 <= z_k < max_range:
                p_rand_val = 1.0 / max_range
            else:
                p_rand_val = 0.0

            p = z_hit * p_hit + z_short * p_short + z_max * p_max_val + z_rand * p_rand_val

            # Avoid log(0)
            if p > 0:
                log_prob += math.log(p)
            else:
                log_prob += math.log(1e-300)

        # Convert back from log space
        prob_zt1 = math.exp(log_prob)

        return prob_zt1