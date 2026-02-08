
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:

    def __init__(self, occupancy_map):
        self._z_hit = 35
        self._z_short = 3
        self._z_max = 0.5
        self._z_rand = 500

        self._sigma_hit = 150
        self._lambda_short = 0.01

        self._max_range = 1000
        self._min_probability = 0.35
        self._subsampling = 5

        self._occupancy_map = occupancy_map
        self._map_resolution = 10

    def _ray_cast(self, x, y, theta):
        x_map = x / self._map_resolution
        y_map = y / self._map_resolution

        step_size = 1.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        max_steps = int(self._max_range / self._map_resolution)

        for i in range(1, max_steps + 1):
            cx = x_map + i * step_size * cos_t
            cy = y_map + i * step_size * sin_t

            ix = int(round(cx))
            iy = int(round(cy))

            if ix < 0 or ix >= self._occupancy_map.shape[1] or \
               iy < 0 or iy >= self._occupancy_map.shape[0]:
                return self._max_range

            occ = self._occupancy_map[iy, ix]
            if occ < 0:
                continue
            if occ > self._min_probability:
                dist = i * step_size * self._map_resolution
                return min(dist, self._max_range)

        return self._max_range

    def _compute_p_hit(self, z_k, z_k_star):
        if 0 <= z_k <= self._max_range:
            eta = 1.0 / (
                norm.cdf(self._max_range, loc=z_k_star, scale=self._sigma_hit) -
                norm.cdf(0, loc=z_k_star, scale=self._sigma_hit)
            )
            p_hit = eta * norm.pdf(z_k, loc=z_k_star, scale=self._sigma_hit)
            return p_hit
        else:
            return 0.0

    def _compute_p_short(self, z_k, z_k_star):
        if 0 <= z_k <= z_k_star and z_k_star > 0:
            eta = 1.0 / (1.0 - math.exp(-self._lambda_short * z_k_star))
            p_short = eta * self._lambda_short * \
                      math.exp(-self._lambda_short * z_k)
            return p_short
        else:
            return 0.0

    def _compute_p_max(self, z_k):
        if abs(z_k - self._max_range) < 1.0:
            return 1.0
        else:
            return 0.0

    def _compute_p_rand(self, z_k):
        if 0 <= z_k < self._max_range:
            return 1.0 / self._max_range
        else:
            return 0.0

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        x, y, theta = x_t1

        x_laser = x + 25.0 * math.cos(theta)
        y_laser = y + 25.0 * math.sin(theta)

        q = 0.0

        for k in range(0, 180, self._subsampling):
            z_k = z_t1_arr[k]

            beam_angle = theta + math.radians(k - 90)
            z_k_star = self._ray_cast(x_laser, y_laser, beam_angle)

            p_hit = self._compute_p_hit(z_k, z_k_star)
            p_short = self._compute_p_short(z_k, z_k_star)
            p_max = self._compute_p_max(z_k)
            p_rand = self._compute_p_rand(z_k)

            p = (self._z_hit * p_hit +
                 self._z_short * p_short +
                 self._z_max * p_max +
                 self._z_rand * p_rand)

            if p > 0:
                q += math.log(p)

        prob_zt1 = math.exp(q)
        return prob_zt1