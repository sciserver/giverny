########################################################################
#
#  Copyright 2024 Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: turbulence@pha.jhu.edu
# Website: http://turbulence.pha.jhu.edu/
#
########################################################################

"""
calculate the interpolation weights for the non-constant step grids.
    - channel, channel5200, transition_bl datasets.
"""
import numpy as np
import math
import giverny.turbulence_gizmos.barycentric_weights as barycentric_weights
from giverny.turbulence_gizmos.variable_grids import *

class irregular_y_interpolation_weights():
    def __init__(self, lagrange_order, dataset_title):
        self.lagrange_order = lagrange_order
        
        self.y_values = np.array([], dtype = np.float64)
        if dataset_title == 'channel':
            self.y_values = get_channel_ys()
            self.find_stencil_bounds = self.find_stencil_bounds_for_channel
            self.find_stencil_endpoints = self.find_stencil_endpoints_for_channel
        elif dataset_title == 'channel5200':
            self.y_values = get_channel5200_ys()
            self.find_stencil_bounds = self.find_stencil_bounds_for_channel
            self.find_stencil_endpoints = self.find_stencil_endpoints_for_channel
        elif dataset_title == 'transition_bl':
            self.y_values = get_transition_bl_ys()
            self.find_stencil_bounds = self.find_stencil_bounds_for_transition_bl
            self.find_stencil_endpoints = self.find_stencil_endpoints_for_transition_bl

        dictionary_name = f"{dataset_title}_baryctrwt_y_lag{lagrange_order}"
        try:
            self.barycentric_weights = getattr(barycentric_weights, dictionary_name)
        except AttributeError:
            raise ValueError(f"barycentric weights not found for {dataset_title} lag{lagrange_order}")

    def find_stencil_bounds_for_channel(self, y_interpolate):
        """
        This method performs a binary search on 'self.y_values' to find an
        index 'n' such that 'y_interpolate' lies between neighboring grid
        points. The returned index is later used to determine stencil
        bounds for interpolation.

        The search domain is split into two regions to handle asymmetry
        around the grid center:
        - If 'y_interpolate <= 0', the search is restricted to the lower
        half of the grid.
        - If 'y_interpolate > 0', the search is restricted to the upper
        half of the grid.

        Parameters
        ----------
        y_interpolate : float
            The coordinate at which interpolation is requested.

        Returns
        -------
        int
            Index 'n' such that 'y_interpolate' lies between adjacent grid
            points:
            - For 'y_interpolate <= 0':
                self.y_values[n] <= y_interpolate < self.y_values[n + 1]
            - For 'y_interpolate > 0':
                self.y_values[n - 1] < y_interpolate <= self.y_values[n]

            Returns '-1' if no valid index is found.
        """
        Ny = len(self.y_values)
        
        if y_interpolate <= 0:
            low, high = 0, int(Ny/2) - 1
            while low <= high:

                mid = (low + high) // 2
                if self.y_values[mid] <= y_interpolate < self.y_values[mid + 1]:
                    return mid
                elif y_interpolate < self.y_values[mid]:
                    high = mid - 1
                else:
                    low = mid + 1

            return -1 
        else:
            low, high = int(Ny/2), Ny - 1 
            while low <= high:
                mid = (low + high) // 2

                if self.y_values[mid - 1] <= y_interpolate <= self.y_values[mid]:
                    return mid
                elif y_interpolate > self.y_values[mid]:
                    low = mid + 1
                else:
                    high = mid - 1
                    
            return -1 

    def find_stencil_bounds_for_transition_bl(self, y_interpolate):
        """
        This method performs a binary search on 'self.y_values' to find an
        index 'n' such that 'y_interpolate' lies between neighboring grid
        points. The returned index is later used to determine stencil
        bounds for interpolation.

        Parameters
        ----------
        y_interpolate : float
            The coordinate at which interpolation is requested.

        Returns
        -------
        int
            Index 'n' such that 'y_interpolate' lies between adjacent grid
            points:
            - self.y_values[n - 1] < y_interpolate <= self.y_values[n]
            - self.y_values[0] == y_interpolate

            Returns '-1' if no valid index is found.
        """
        Ny = len(self.y_values)
        low, high = 0, Ny - 1
        while low <= high:
            mid = (low + high) // 2

            if mid <= 0:
                if y_interpolate <= self.y_values[0]:
                    return 0
                else:
                    low = 1
                    continue

            if self.y_values[mid - 1] < y_interpolate <= self.y_values[mid]:
                return mid
            elif y_interpolate > self.y_values[mid]:
                low = mid + 1
            else:
                high = mid - 1

        return -1

    def find_stencil_endpoints_for_channel(self, n):   
        """
        Given a reference index `n` in grid points, this method computes
        the start index 'j_s', end index 'j_e', and origin offset 'j_o'
        defining a contiguous stencil of length 'lagrange_order' (q).

        The stencil is centered about 'n' when possible. Near domain
        boundaries, the stencil is shifted to remain within valid grid
        bounds, and the origin offset 'j_o' records the displacement of
        the reference index within the stencil.

        Parameters
        ----------
        n : int
            Reference grid index returned by the find_stencil_bounds function.

        Returns
        -------
        tuple of int
            (j_s, j_e, j_o) where:
            - 'j_s' : int
                Starting index of the stencil in grid points.
            - 'j_e' : int
                Ending index of the stencil in grid points.
            - 'j_o' : int
                Offset of the reference index within the stencil, such that
                the reference point corresponds to stencil index 'j_o'.
        """  
        j_s = None
        j_e = None
        j_o = None

        if n <= ((len(self.y_values)/2) - 1):
            j_o = max(math.ceil(self.lagrange_order/2) - n - 1, 0)
            j_s = n - math.ceil(self.lagrange_order/2) + 1 + j_o
        else:
            j_o = min(len(self.y_values) - n - math.ceil(self.lagrange_order/2), 0)
            j_s = n - math.floor(self.lagrange_order/2) + j_o

        j_e = j_s + self.lagrange_order - 1

        return (j_s,j_e,j_o)

    def find_stencil_endpoints_for_transition_bl(self, n):   
        """
        Given a reference index `n` in grid points, this method computes
        the start index 'j_s', end index 'j_e', and origin offset 'j_o'
        defining a contiguous stencil of length 'lagrange_order' (q).

        The stencil is centered about 'n' when possible. Near domain
        boundaries, the stencil is shifted to remain within valid grid
        bounds, and the origin offset 'j_o' records the displacement of
        the reference index within the stencil.

        Parameters
        ----------
        n : int
            Reference grid index returned by the find_stencil_bounds function.

        Returns
        -------
        tuple of int
            (j_s, j_e, j_o) where:
            - 'j_s' : int
                Starting index of the stencil in grid points.
            - 'j_e' : int
                Ending index of the stencil in grid points.
            - 'j_o' : int
                Offset of the reference index within the stencil, such that
                the reference point corresponds to stencil index 'j_o'.
        """  
        j_s = None
        j_e = None
        j_o = None

        left_threshold = math.floor(self.lagrange_order / 2)

        if n < left_threshold:
            j_o = max(math.ceil(self.lagrange_order/2) - n - 1, 0)
            j_s = n - math.ceil(self.lagrange_order/2) + 1 + j_o
        else:
            j_o = min(len(self.y_values) - n - math.ceil(self.lagrange_order/2), 0)
            j_s = n - math.floor(self.lagrange_order/2) + j_o

        j_e = j_s + self.lagrange_order - 1

        return (j_s,j_e,j_o)

    def calculate_interpolation_weights(self, stencil_points, barycentric_weights, y_interpolate):
        """
        Given a set of stencil grid points and their precomputed first-form
        barycentric weights, this function computes the interpolation weights.
        These weights can be applied to function values at the stencil points to
        obtain the interpolated value at 'y_interpolate'.

        The interpolation weights λ_j are given by:

            λ_j(y) = (w_j / (y - x_j)) / Σ_k (w_k / (y - x_k))

        where:
            - x_j are the stencil grid points
            - w_j are the barycentric weights

        Parameters
        ----------
        stencil_points : np.ndarray
            One-dimensional array of grid points forming the interpolation
            stencil.
        barycentric_weights : np.ndarray
            First-form barycentric weights corresponding to `stencil_points`.
        y_interpolate : float
            The coordinate at which the interpolant is evaluated.

        Returns
        -------
        np.ndarray
            Array of interpolation weights of the same length as 'stencil_points'. 
        """

        differences = y_interpolate - stencil_points

        numerators = np.divide(
            barycentric_weights,
            differences,
            out=np.sign(barycentric_weights) * np.finfo(np.float64).max,
            where=differences != 0
        )
        
        denominator = np.sum(numerators)

        weights = numerators / denominator

        return weights

    def get_stencil_weights(self, y_interpolate):
        n = self.find_stencil_bounds(y_interpolate)

        j_s, j_e, j_o = self.find_stencil_endpoints(n) 

        stencil_points = self.y_values[j_s : j_e + 1]
        
        barycentric_weights = self.barycentric_weights[(j_s, j_o)]

        return self.calculate_interpolation_weights(stencil_points, barycentric_weights, y_interpolate)