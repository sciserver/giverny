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
differentiation matrix lookup table generator
-
this module generates lookup tables for first and second order numerical
differentiation matrices using barycentric lagrange interpolation weights.

a dictionary is generated:

    {
        0 : (start_idx, end_idx, d1_matrix),
        1 : (start_idx, end_idx, d2_matrix)
    }

where:
    0 : first derivative lookup table
    1 : second derivative lookup table

"""
import numpy as np
from giverny.turbulence_gizmos.variable_grids_interpolation_weights import irregular_y_interpolation_weights

def compute_differentiation_matrix_order_1(center_idx, y_stencil, w_stencil):
    difference = y_stencil[center_idx] - y_stencil
        
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        row = (w_stencil / w_stencil[center_idx]) / difference
        
    row[center_idx] = 0
    row[center_idx] = -np.sum(row)

    return row

def compute_differentiation_matrix_order_2(center_idx, y_stencil, w_stencil):
    difference = y_stencil[center_idx] - y_stencil
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        off_diag_1 = (w_stencil / w_stencil[center_idx]) / difference

        sum_term = np.sum(np.nan_to_num(off_diag_1, nan = 0, posinf = 0, neginf = 0))
        
        row = -2 * off_diag_1 * (sum_term + (1.0 / difference))

    row[center_idx] = 0
    row[center_idx] = -np.sum(row)

    return row

def create_lookup_table(derivative_order, interpolation_class):
    y_values = interpolation_class.y_values
    barycentric_weights = interpolation_class.barycentric_weights

    n = len(y_values)
    stencil_size = interpolation_class.lagrange_order

    lt_start_idx = np.zeros(n, dtype=np.int32)
    lt_end_idx = np.zeros(n, dtype=np.int32)
    lt_differentiation_matrix = np.zeros((n, stencil_size))

    for i in range(n):
        j_s, j_e, j_o = interpolation_class.find_stencil_endpoints(i)
        
        center_idx = i - j_s
        
        y_stencil = np.asarray(y_values[j_s : j_e + 1])
        w_stencil = np.asarray(barycentric_weights[(j_s, j_o)])

        row = np.zeros(stencil_size)

        if derivative_order == 1:
            row = compute_differentiation_matrix_order_1(center_idx, y_stencil, w_stencil)
        elif derivative_order == 2:
            row = compute_differentiation_matrix_order_2(center_idx, y_stencil, w_stencil)
    
        lt_start_idx[i] = j_s
        lt_end_idx[i] = j_e
        lt_differentiation_matrix[i] = row

    return lt_start_idx, lt_end_idx, lt_differentiation_matrix

if __name__ == "__main__":
    # Define dataset and path to grid points
    dataset = 'channel'

    # Example FD4 has Order of Differencing = 4
    ORDER_OF_DIFFERENCING = 4

    #The code below remains unchanged for different ORDER_OF_DIFFERENCING and datasets.

    differentiation_lookup_table = {}

    #Compute differentiation matrices for first and second derivatives
    for i in range(2):
        derivative_order = i + 1
        q = ORDER_OF_DIFFERENCING + derivative_order
        interpolation_class = irregular_y_interpolation_weights(q, dataset)
        lt_start_idx, lt_end_idx, lt_differentiation_matrix = create_lookup_table(derivative_order, interpolation_class)
        differentiation_lookup_table[i] = (lt_start_idx, lt_end_idx, lt_differentiation_matrix)

    with open(f"{dataset}-baryctrwt-diffmat-y-fd{ORDER_OF_DIFFERENCING}.pkl", "wb") as f:
        pickle.dump(differentiation_lookup_table, f)
        
