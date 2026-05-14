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

import math
import numpy as np

class irregular_y_differentiation_vectors():
    def __init__(self, lookup_table):
        """
        Initialize the differentiation helper.

        Parameters
        ----------
        lookup_table : dict
            Precomputed differentiation lookup table.
        """
        self.differentiation_lookup_table = lookup_table

    def calculate_grid_points_derivative(self, j_s, j_e, field_values, derivative_order):
        """
        Compute derivatives at grid points within a stencil range.

        This method uses the precomputed differentiation weights from the
        lookup table to evaluate derivatives of a field defined on the grid.

        Parameters
        ----------
        j_s : int
            Starting index of the grid region where derivatives are required.

        j_e : int
            Ending index of the grid region where derivatives are required.

        field_values : ndarray
            Values of the field defined on the full grid.

        derivative_order : int
            Order of derivative to compute 
            (1 for first derivative, 2 for second derivative).

        Returns
        -------
        ndarray
            Array containing the computed derivatives at grid points
            from j_s to j_e.

        Notes
        -----
        For each grid point in the range, the corresponding stencil and
        differentiation weights are retrieved from the lookup table.
        The derivative is computed as:

            f'(y_i) = Σ D_ij * f(y_j)

        where D_ij are differentiation matrices.
        """
        start_idx_lt, end_idx_lt, differentiation_matrix_lt = self.differentiation_lookup_table[derivative_order - 1]

        stencil_starts = start_idx_lt[j_s : j_e + 1]
        stencil_ends = end_idx_lt[j_s : j_e + 1]
        differentiation_matrix = differentiation_matrix_lt[j_s : j_e + 1]

        field_matrix = np.array([field_values[s:e+1] for s, e in zip(stencil_starts, stencil_ends)])

        return np.einsum('ij,ij->i', differentiation_matrix, field_matrix)
    
    def get_interpolation_vectors(self, y_interpolate, interpolation_class, derivative_order, field_values):
        """
        Compute derivative values and interpolation weights for a target point.

        This function prepares the vectors needed to evaluate interpolated
        derivatives at an arbitrary location between grid points.

        Parameters
        ----------
        y_interpolate : float
            Location where interpolation is required.

        interpolation_class : YInterpolationWeights

        derivative_order : int
            Order of derivative to compute.

        field_values : ndarray
            Field values defined on the full grid.

        Returns
        -------
        grid_points_derivative : ndarray
            Derivative values evaluated at stencil grid points.

        interpolation_weights : ndarray
            Weights used to interpolate the derivative from grid points
            to the target location y_interpolate.

        Notes
        -----
        The `interpolation_class` must be created using the same dataset and
        Lagrange order that were used to generate the differentiation lookup table.
        For instance, if the lookup table was built with `ORDER_OF_DIFFERENCING = 4`,
        then the interpolation class must also use a Lagrange order of 4.
        
        The final interpolated derivative can then be computed as:

            f'(y) = Σ w_i * f'(y_i)
        """
        n = interpolation_class.find_stencil_bounds(y_interpolate)

        j_s, j_e, j_o = interpolation_class.find_stencil_endpoints(n)

        y_stencil = np.asarray(interpolation_class.y_values[j_s : j_e + 1])
        w_stencil = np.asarray(interpolation_class.barycentric_weights[(j_s, j_o)])

        grid_points_derivative = self.calculate_grid_points_derivative(j_s, j_e, field_values, derivative_order)
        interpolation_weights = interpolation_class.calculate_interpolation_weights(y_stencil, w_stencil, y_interpolate)

        return grid_points_derivative, interpolation_weights


