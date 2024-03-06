import os
import csv
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
# tqdm is used in the getdata jupyter notebook single point time series.
from tqdm import tqdm
from collections import defaultdict

"""
user-input checking gizmos.
"""
def check_dataset_title(dataset_title):
    # check that dataset_title is a valid dataset title. 'isotropic4096' dataset is not yet converted to zarr.
    valid_dataset_titles = ['isotropic8192', 'isotropic4096', 'isotropic1024coarse', 'isotropic1024fine',
                            'sabl2048low', 'sabl2048high',
                            'rotstrat4096',
                            'mhd1024', 'mixing',
                            'channel', 'channel5200',
                            'transition_bl']
    
    if dataset_title not in valid_dataset_titles:
        raise Exception(f"'{dataset_title}' (case-sensitive) is not a valid dataset title:\n{valid_dataset_titles}")
        
    return

def check_variable(variable, dataset_title):
    # check that variable is a valid variable name.
    valid_variables = {
        'isotropic1024coarse':['pressure', 'velocity', 'force', 'position'],
        'isotropic1024fine':['pressure', 'velocity', 'force', 'position'],
        'isotropic4096':['pressure', 'velocity'],
        'isotropic8192':['pressure', 'velocity'],
        'sabl2048low':['pressure', 'velocity', 'temperature', 'energy'],
        'sabl2048high':['pressure', 'velocity', 'temperature', 'energy'],
        'rotstrat4096':['velocity', 'temperature'],
        'mhd1024':['pressure', 'velocity', 'magneticfield', 'vectorpotential', 'force', 'position'],
        'mixing':['pressure', 'velocity', 'density', 'position'],
        'channel':['pressure', 'velocity', 'position'],
        'channel5200':['pressure', 'velocity'],
        'transition_bl':['pressure', 'velocity', 'position']
    }[dataset_title]
    
    if variable not in valid_variables:
        raise Exception(f"'{variable}' (case-sensitive) is not a valid variable for '{dataset_title}':\n{valid_variables}")
        
    return

def check_timepoint(timepoint, dataset_title, query_type):
    # check that timepoint is a valid timepoint for the dataset.
    valid_timepoints = {
        'isotropic1024coarse': {'getdata': (0.0, 10.056, 0.002), 'getcutout': range(1, 5024 + 1)},
        'isotropic1024fine': {'getdata': (0.0, 0.0198, 0.0002), 'getcutout': range(1, 100 + 1)},
        'isotropic4096': {'getdata': range(1, 1 + 1), 'getcutout': range(1, 1 + 1)},
        'isotropic8192': {'getdata': range(1, 6 + 1), 'getcutout': range(1, 6 + 1)},
        'sabl2048low': {'getdata': range(1, 20 + 1), 'getcutout': range(1, 20 + 1)},
        'sabl2048high': {'getdata': (0.0, 7.2, 0.075), 'getcutout': range(1, 97 + 1)},
        'rotstrat4096': {'getdata': range(1, 5 + 1), 'getcutout': range(1, 5 + 1)},
        'mhd1024': {'getdata': (0.0, 2.56, 0.0025), 'getcutout': range(1, 1025 + 1)},
        'mixing': {'getdata': (0.0, 40.44, 0.04), 'getcutout': range(1, 1012 + 1)},
        'channel': {'getdata': (0.0, 25.9935, 0.0065), 'getcutout': range(1, 4000 + 1)},
        'channel5200': {'getdata': range(1, 11 + 1), 'getcutout': range(1, 11 + 1)},
        'transition_bl': {'getdata': (0.0, 1175.0, 0.25), 'getcutout': range(1, 4701 + 1)}
    }[dataset_title][query_type]
    
    # list of datasets which are low-resolution and thus the timepoint is specified as a time index for getdata processing.
    time_index_datasets = ['isotropic4096', 'isotropic8192', 'sabl2048low', 'rotstrat4096', 'channel5200']
    
    if query_type == 'getcutout' or dataset_title in time_index_datasets: 
        # handles checking datasets with time indices.
        if timepoint not in valid_timepoints:
            raise Exception(f"{timepoint} is not a valid timepoint for '{dataset_title}': must be an integer and in the inclusive range of " +
                            f'[{valid_timepoints[0]}, {valid_timepoints[-1]}]')
    else:
        # handles checking datasets with real times.
        if timepoint < valid_timepoints[0] or timepoint > valid_timepoints[1]:
            raise Exception(f"{timepoint} is not a valid timepoint for '{dataset_title}': must be in the inclusive range of " +
                            f'[{valid_timepoints[0]}, {valid_timepoints[1]}]')
        
    return

def check_points_domain(dataset_title, points):
    # check that the points are inside the acceptable domain along each axis for the dataset. 'modulo' placeholder values
    # are used for cases where points outside the domain are allowed as modulo(domain range).
    axes_domain = {
        'isotropic1024coarse': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'isotropic1024fine': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'isotropic4096': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'isotropic8192': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'sabl2048low': ['modulo[0, 399.8046875]', 'modulo[0, 399.8046875]', [0.09765625, 399.90234375]],
        'sabl2048high': ['modulo[0, 399.8046875]', 'modulo[0, 399.8046875]', [0.09765625, 399.90234375]],
        'rotstrat4096': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'mhd1024': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'mixing': ['modulo[0, 2pi]', 'modulo[0, 2pi]', 'modulo[0, 2pi]'],
        'channel': ['modulo[0, 8pi]', [-1, 1], 'modulo[0, 3pi]'],
        'channel5200': ['modulo[0, 8pi]', [-1, 1], 'modulo[0, 3pi]'],
        'transition_bl': [[30.2185, 1000.065], [0, 26.48795], 'modulo[0, 240]']
    }[dataset_title]

    # minimum and maximum point values along each axis.
    points_min = np.min(points, axis = 0)
    points_max = np.max(points, axis = 0)

    # checks if all points are within the axes domain for the dataset.
    points_domain_check = np.all([points_min[axis] >= axes_domain[axis][0] and points_max[axis] <= axes_domain[axis][1]
                                  if type(axes_domain[axis]) == list else True for axis in range(len(axes_domain))])

    if not points_domain_check:
        raise Exception(f"all points are not within the allowed domain [minimum, maximum] for '{dataset_title}':\n" +
                        f"x: {axes_domain[0]}\ny: {axes_domain[1]}\nz: {axes_domain[2]}")
    
    return

def check_operator(operator, variable):
    # check that the interpolation operator is a valid operator.
    valid_operators = {'velocity':['function', 'gradient', 'hessian', 'laplacian'],
                       'pressure':['function', 'gradient', 'hessian'],
                       'energy':['function', 'gradient', 'hessian'],
                       'temperature':['function', 'gradient', 'hessian'],
                       'force':['function'],
                       'magneticfield':['function', 'gradient', 'hessian', 'laplacian'],
                       'vectorpotential':['function', 'gradient', 'hessian', 'laplacian'],
                       'density':['function', 'gradient', 'hessian'],
                       'position':['function']}[variable]
    
    if operator not in valid_operators:
        raise Exception(f"'{operator}' (case-sensitive) is not a valid interpolation operator for '{variable}':\n{valid_operators}")
        
    return

def check_spatial_interpolation(dataset_title, variable, sint, operator):
    # check that the interpolation method is a valid method.
    if operator == 'function':
        valid_sints = {
            'isotropic1024coarse': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14'], {'position': ['lag4', 'lag6', 'lag8']}),
            'isotropic1024fine': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14'], {'position': ['lag4', 'lag6', 'lag8']}),
            'isotropic4096': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14']),
            'isotropic8192': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8']),
            'sabl2048low': defaultdict(lambda: ['lag4', 'lag6', 'lag8', 'm1q4', 'm2q8']),
            'sabl2048high': defaultdict(lambda: ['lag4', 'lag6', 'lag8', 'm1q4', 'm2q8']),
            'rotstrat4096': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14']),
            'mhd1024': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14'], {'position': ['lag4', 'lag6', 'lag8']}),
            'mixing': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14'], {'position': ['lag4', 'lag6', 'lag8']}),
            'channel': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14'], {'position': ['lag4', 'lag6', 'lag8']}),
            'channel5200': defaultdict(lambda: ['none', 'lag4', 'lag6', 'lag8', 'm1q4', 'm2q8', 'm2q14']),
            'transition_bl': defaultdict(lambda: ['none', 'lag4'], {'position': ['lag4']})
        }[dataset_title][variable]
        sint_suffix = ''
    elif operator == 'gradient':
        valid_sints = {
            'isotropic1024coarse': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'isotropic1024fine': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'isotropic4096': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'isotropic8192': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8'],
            'sabl2048low': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8'],
            'sabl2048high': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8'],
            'rotstrat4096': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'mhd1024': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'mixing': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'channel': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'channel5200': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm1q4', 'm2q8', 'm2q14'],
            'transition_bl': ['fd4noint', 'fd4lag4']
        }[dataset_title]
        sint_suffix = '_g'
    elif operator == 'hessian':
        valid_sints = {
            'isotropic1024coarse': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'isotropic1024fine': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'isotropic4096': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'isotropic8192': ['fd4noint', 'fd6noint', 'fd8noint', 'm2q8'],
            'sabl2048low': ['fd4noint', 'fd6noint', 'fd8noint', 'm2q8'],
            'sabl2048high': ['fd4noint', 'fd6noint', 'fd8noint', 'm2q8'],
            'rotstrat4096': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'mhd1024': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'mixing': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'channel': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'channel5200': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4', 'm2q8', 'm2q14'],
            'transition_bl': ['fd4noint', 'fd4lag4']
        }[dataset_title]
        sint_suffix = '_h'
    elif operator == 'laplacian':
        valid_sints = {
            'isotropic1024coarse': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'isotropic1024fine': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'isotropic4096': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'isotropic8192': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'sabl2048low': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'sabl2048high': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'rotstrat4096': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'mhd1024': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'mixing': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'channel': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'channel5200': ['fd4noint', 'fd6noint', 'fd8noint', 'fd4lag4'],
            'transition_bl': ['fd4noint', 'fd4lag4']
        }[dataset_title]
        sint_suffix = '_l'
        
    if sint not in valid_sints:
        raise Exception(f"'{sint}' (case-sensitive) is not a valid spatial interpolation method for ('{dataset_title}', '{variable}', '{operator}'):\n{valid_sints}")
        
    return sint + sint_suffix

def check_temporal_interpolation(dataset_title, variable, tint):
    # check that the interpolation method is a valid method.
    valid_tints = {
            'isotropic1024coarse': defaultdict(lambda: ['none', 'pchip'], {'position': ['none']}),
            'isotropic1024fine': defaultdict(lambda: ['none', 'pchip'], {'position': ['none']}),
            'isotropic4096': defaultdict(lambda: ['none']),
            'isotropic8192': defaultdict(lambda: ['none']),
            'sabl2048low': defaultdict(lambda: ['none']),
            'sabl2048high': defaultdict(lambda: ['none', 'pchip']),
            'rotstrat4096': defaultdict(lambda: ['none']),
            'mhd1024': defaultdict(lambda: ['none', 'pchip'], {'position': ['none']}),
            'mixing': defaultdict(lambda: ['none', 'pchip'], {'position': ['none']}),
            'channel': defaultdict(lambda: ['none', 'pchip'], {'position': ['none']}),
            'channel5200': defaultdict(lambda: ['none']),
            'transition_bl': defaultdict(lambda: ['none', 'pchip'], {'position': ['none']})
        }[dataset_title][variable]
        
    if tint not in valid_tints:
        raise Exception(f"'{tint}' (case-sensitive) is not a valid temporal interpolation method for ('{dataset_title}', '{variable}'):\n{valid_tints}")
        
    return

def check_option_parameter(option, timepoint_start):
    # check that the 'option' parameter used by getPosition was correctly specified.
    timepoint_end, delta_t = option

    if (timepoint_start + delta_t) > timepoint_end:
        raise Exception(f"'timepoint' + 'delta_t' is greater than 'timepoint_end': {timepoint_start} + {delta_t} = {timepoint_start + delta_t} > {timepoint_end}")

    # # steps_to_keep was removed as a user-modifiable parameter, so this check is no longer needed. it is commented out in case this is changed in the future.
    # if steps_to_keep > ((timepoint_end - timepoint_start) / delta_t):
    #     raise Exception(f"'steps_to_keep' ({steps_to_keep}) is greater than the number of delta_t ({delta_t}) time steps between " +
    #                     f"timepoint ({timepoint_start}) and timepoint_end ({timepoint_end}): " +
    #                     f"{steps_to_keep} > {((timepoint_end - timepoint_start) / delta_t)} = (({timepoint_end} - {timepoint_start}) / {delta_t})")
        
    return

def check_axes_ranges(axes_ranges):
    # check that the axis ranges are all specified as minimum and maximum integer values.
    for axis_range in axes_ranges:
        if len(axis_range) != 2:
            raise Exception(f'axis range, {axis_range}, is not correctly specified as [minimum, maximum]')
            
        for val in axis_range:
            if type(val) != int:
                raise Exception(f'{val} in axis range, {list(axis_range)}, is not an integer')
                
def check_strides(strides):
    # check that the strides are all positive integer values.
    for stride in strides:
        if type(stride) != int or stride < 1:
            raise Exception(f'stride, {stride}, is not an integer value >= 1')

"""
mapping gizmos.
"""
def get_dataset_resolution(dataset_title):
    # get the number of datapoints (resolution) along each axis of the dataset. non-giverny datasets are not supported 
    # by this code at this time, so a placeholder value (the largest dimension) is used.
    return {
        'isotropic1024coarse': 1024,
        'isotropic1024fine': 1024,
        'isotropic4096': 4096,
        'isotropic8192': 8192,
        'sabl2048low': 2048,
        'sabl2048high': 2048,
        'rotstrat4096': 4096,
        'mhd1024': 1024,
        'mixing': 1024,
        'channel': 2048, #[2048, 512, 1536],
        'channel5200': 10240, #[10240, 1536, 7680],
        'transition_bl': 3320 #[3320, 224, 2048]
    }[dataset_title]

def get_dataset_dx(dataset_title, query_type):
    # get the dataset spacing between datapoints along each axis of the dataset. non-giverny datasets are not supported 
    # by this code at this time, so a placeholder value (the largest dimension) is used.
    return {
        'isotropic1024coarse': defaultdict(lambda: 2 * np.pi / 1024),
        'isotropic1024fine': defaultdict(lambda: 2 * np.pi / 1024),
        'isotropic4096': defaultdict(lambda: 2 * np.pi / 4096),
        'isotropic8192': defaultdict(lambda: 2 * np.pi / 8192),
        'sabl2048low': defaultdict(lambda: 2 * np.pi / 2048, {'getdata': 0.1953125}),
        'sabl2048high': defaultdict(lambda: 2 * np.pi / 2048, {'getdata': 0.1953125}),
        'rotstrat4096': defaultdict(lambda: 2 * np.pi / 4096),
        'mhd1024': defaultdict(lambda: 2 * np.pi / 1024),
        'mixing': defaultdict(lambda: 2 * np.pi / 1024),
        'channel': defaultdict(lambda: 2 * np.pi / 2048), #[2048, 512, 1536],
        'channel5200': defaultdict(lambda: 2 * np.pi / 10240), #[10240, 1536, 7680],
        'transition_bl': defaultdict(lambda: 2 * np.pi / 3320) #[3320, 224, 2048]
    }[dataset_title][query_type]

def get_dataset_dimension_offsets(dataset_title, variable):
    # get the dataset dimension offset between each axis of the dataset. non-giverny datasets are not supported 
    # by this code at this time, so a placeholder value (0) is used.
    return {
        'isotropic1024coarse': defaultdict(lambda: 0),
        'isotropic1024fine': defaultdict(lambda: 0),
        'isotropic4096': defaultdict(lambda: 0),
        'isotropic8192': defaultdict(lambda: 0),
        'sabl2048low': defaultdict(lambda: np.array([0, 0, -0.5]), {'energy': np.array([0, 0, -1.0]), 'velocity_uv': np.array([0, 0, -0.5]), 'velocity_w': np.array([0, 0, -1.0])}),
        'sabl2048high': defaultdict(lambda: np.array([0, 0, -0.5]), {'energy': np.array([0, 0, -1.0]), 'velocity_uv': np.array([0, 0, -0.5]), 'velocity_w': np.array([0, 0, -1.0])}),
        'rotstrat4096': defaultdict(lambda: 0),
        'mhd1024': defaultdict(lambda: 0),
        'mixing': defaultdict(lambda: 0),
        'channel': defaultdict(lambda: 0),
        'channel5200': defaultdict(lambda: 0),
        'transition_bl': defaultdict(lambda: 0)
    }[dataset_title][variable]

def get_sabl_points_map(cube, points):
    """
    separate points into different interpolation methods for the 'sabl2048low' and 'sabl2048high' datasets if they are near the boundary.
    """
    # create dictionaries for storing the points and their corresponding ordering.
    points_map = {}
    original_indices_map = {}

    # dataset paramters.
    specified_sint = cube.sint
    dataset_dx = cube.dx
    dataset_var_name = cube.var_name

    # constants.
    points_len = len(points)
    # save the original indices for points, which corresponds to the orderering of the user-specified
    # points. these indices will be used for sorting output_data back to the user-specified points ordering.
    original_points_indices = np.arange(points_len)

    # map the step down interpolation methods for each interpolation method that can be specified by the user. the methods are listed in order of decreasing bucket size.
    interpolation_step_down_method_map = {'lag8': ['lag6', 'lag4', 'linear'], 'lag6': ['lag4', 'linear'], 'lag4': ['linear'],
                                          'm2q8': ['m1q4', 'linear'], 'm1q4': ['linear'],
                                          'm2q8_g': ['m1q4_g', 'linear_g'], 'm1q4_g': ['linear_g'],
                                          'm2q8_h': ['fd6noint_h', 'fd4noint_h', 'linear_h'],
                                          'fd8noint_g': ['fd6noint_g', 'fd4noint_g', 'linear_g'], 'fd6noint_g': ['fd4noint_g', 'linear_g'], 'fd4noint_g': ['linear_g'],
                                          'fd8noint_h': ['fd6noint_h', 'fd4noint_h', 'linear_h'], 'fd6noint_h': ['fd4noint_h', 'linear_h'], 'fd4noint_h': ['linear_h'],
                                          'fd8noint_l': ['fd6noint_l', 'fd4noint_l', 'linear_l'], 'fd6noint_l': ['fd4noint_l', 'linear_l'], 'fd4noint_l': ['linear_l'],
                                          'fd4lag4_g': ['m1q4_g', 'linear_g'],
                                          'fd4lag4_l': ['fd6noint_l', 'fd4noint_l', 'linear_l']}

    # map the minium z-axis index (dx multiplier) for each variable and interpolation method. default values correspond to 'energy' and 'velocity' variables.
    # the (w) component of 'velocity' is more restrictive on the interpolation method at the lower z-axis boundary. 'linear' index is the 0 index because
    # the comparison is >=, not >. the points are constrained by the check_points_domain function such that user cannot specify points outside the domain.
    interpolation_min_index_map = defaultdict(lambda: {'lag8': 4, 'lag6': 3, 'lag4': 2,
                                                       'm2q8': 4, 'm1q4': 2,
                                                       'm2q8_g': 4, 'm1q4_g': 2,
                                                       'm2q8_h': 4,
                                                       'fd8noint_g': 4.5, 'fd6noint_g': 3.5, 'fd4noint_g': 2.5,
                                                       'fd8noint_h': 4.5, 'fd6noint_h': 3.5, 'fd4noint_h': 2.5,
                                                       'fd8noint_l': 4.5, 'fd6noint_l': 3.5, 'fd4noint_l': 2.5,
                                                       'fd4lag4_g': 4,
                                                       'fd4lag4_l': 4,
                                                       'linear': 0, 'linear_g': 0, 'linear_h': 0, 'linear_l': 0
                                                      },
                                                      {'temperature': {'lag8': 3.5, 'lag6': 2.5, 'lag4': 1.5,
                                                                       'm2q8': 3.5, 'm1q4': 1.5,
                                                                       'm2q8_g': 3.5, 'm1q4_g': 1.5,
                                                                       'm2q8_h': 3.5,
                                                                       'fd8noint_g': 4, 'fd6noint_g': 3, 'fd4noint_g': 2,
                                                                       'fd8noint_h': 4, 'fd6noint_h': 3, 'fd4noint_h': 2,
                                                                       'fd8noint_l': 4, 'fd6noint_l': 3, 'fd4noint_l': 2,
                                                                       'fd4lag4_g': 3.5,
                                                                       'fd4lag4_l': 3.5,
                                                                       'linear': 0, 'linear_g': 0, 'linear_h': 0, 'linear_l': 0
                                                                      },
                                                       'pressure': {'lag8': 3.5, 'lag6': 2.5, 'lag4': 1.5,
                                                                    'm2q8': 3.5, 'm1q4': 1.5,
                                                                    'm2q8_g': 3.5, 'm1q4_g': 1.5,
                                                                    'm2q8_h': 3.5,
                                                                    'fd8noint_g': 4, 'fd6noint_g': 3, 'fd4noint_g': 2,
                                                                    'fd8noint_h': 4, 'fd6noint_h': 3, 'fd4noint_h': 2,
                                                                    'fd8noint_l': 4, 'fd6noint_l': 3, 'fd4noint_l': 2,
                                                                    'fd4lag4_g': 3.5,
                                                                    'fd4lag4_l': 3.5,
                                                                    'linear': 0, 'linear_g': 0, 'linear_h': 0, 'linear_l': 0
                                                                   }
                                                      }
                                             )

    # map the minium z-axis index (dx multiplier) for each variable and interpolation method. default values correspond to 'temperature', 'pressure', and 'velocity' variables.
    # the (u, v) components of 'velocity' are more restrictive on the interpolation method at the upper z-axis boundary. 'linear' index is the domain (2048) + 1 because
    # the comparison is <, not <=. the points are constrained by the check_points_domain function such that user cannot specify points outside the domain.
    interpolation_max_index_map = defaultdict(lambda: {'lag8': 2044.5, 'lag6': 2045.5, 'lag4': 2046.5,
                                                       'm2q8': 2044.5, 'm1q4': 2046.5,
                                                       'm2q8_g': 2044.5, 'm1q4_g': 2046.5,
                                                       'm2q8_h': 2044.5,
                                                       'fd8noint_g': 2043, 'fd6noint_g': 2044, 'fd4noint_g': 2045,
                                                       'fd8noint_h': 2043, 'fd6noint_h': 2044, 'fd4noint_h': 2045,
                                                       'fd8noint_l': 2043, 'fd6noint_l': 2044, 'fd4noint_l': 2045,
                                                       'fd4lag4_g': 2044.5,
                                                       'fd4lag4_l': 2044.5,
                                                       'linear': 2049, 'linear_g': 2049, 'linear_h': 2049, 'linear_l': 2049
                                                      },
                                                      {'energy': {'lag8': 2045, 'lag6': 2046, 'lag4': 2047,
                                                                  'm2q8': 2045, 'm1q4': 2047,
                                                                  'm2q8_g': 2045, 'm1q4_g': 2047,
                                                                  'm2q8_h': 2045,
                                                                  'fd8noint_g': 2043.5, 'fd6noint_g': 2044.5, 'fd4noint_g': 2045.5,
                                                                  'fd8noint_h': 2043.5, 'fd6noint_h': 2044.5, 'fd4noint_h': 2045.5,
                                                                  'fd8noint_l': 2043.5, 'fd6noint_l': 2044.5, 'fd4noint_l': 2045.5,
                                                                  'fd4lag4_g': 2045,
                                                                  'fd4lag4_l': 2045,
                                                                  'linear': 2049, 'linear_g': 2049, 'linear_h': 2049, 'linear_l': 2049
                                                                 },
                                                      }
                                             )

    # minimum and maximum positions along the z-axis for the user-specified sint method.
    min_z = interpolation_min_index_map[dataset_var_name][specified_sint] * dataset_dx
    max_z = interpolation_max_index_map[dataset_var_name][specified_sint] * dataset_dx

    # determine the points that are between min_z and max_z.
    sint_points = np.logical_and(points[:, 2] >= min_z, points[:, 2] < max_z)
    # save the points and their corresponding ordering indices for the specified_sint method. these will be added to points_map last to make sure
    # that sint in the cube class variable is what the user specified.
    specified_sint_points = points[sint_points]
    specified_sint_original_indices = original_points_indices[sint_points]
    
    # get the step-down interpolation methods to check near the z-axis boundary.
    step_down_method_sints = interpolation_step_down_method_map[specified_sint]
    # count how many points have been assigned an interpolation method.
    mapped_points_count = len(specified_sint_points)
    # keep track of the previous interpolation method minimum and maximum z-axis positions so that points are not assigned
    # to multiple interpolation methods.
    previous_min_z = min_z
    previous_max_z = max_z
    # counter for iterating over the step-down interpolation methods.
    sint_counter = 0

    # iterate through step-down interpolation methods until all points are assigned an interpolation method.
    while mapped_points_count < points_len:
        step_down_method_sint = step_down_method_sints[sint_counter]

        step_down_min_z = interpolation_min_index_map[dataset_var_name][step_down_method_sint] * dataset_dx
        step_down_max_z = interpolation_max_index_map[dataset_var_name][step_down_method_sint] * dataset_dx

        # determine the points that are between step_down_min_z and step_down_max_z, but also were not assigned to a larger bucket interpolation method.
        sint_points = np.logical_and(np.logical_and(points[:, 2] >= step_down_min_z, points[:, 2] < step_down_max_z),
                                     np.logical_or(points[:, 2] < previous_min_z, points[:, 2] >= previous_max_z))

        # points array mapped to step_down_method_sint.
        step_down_points = points[sint_points]
        step_down_points_len = len(step_down_points)
        
        # add the points and their corresponding ordering indices to their respective dictionaries.
        if step_down_points_len > 0:
            points_map[step_down_method_sint] = step_down_points
            original_indices_map[step_down_method_sint] = original_points_indices[sint_points]
            mapped_points_count += step_down_points_len

        previous_min_z = step_down_min_z
        previous_max_z = step_down_max_z
        sint_counter += 1

    # add the specified_sint points and their corresponding ordering indices to their respective dictionaries.
    if len(specified_sint_points) != 0:
        points_map[specified_sint] = specified_sint_points
        original_indices_map[specified_sint] = specified_sint_original_indices
    
    return points_map, original_indices_map

def get_time_dt(dataset_title, query_type):
    """
    dt between timepoints. placeholder '1' dt values used for legacy datasets that are processed by the pyJHTDB library and datasets processed by giverny
    for which the time is specified by the user as a time index.
    """
    return {
        'isotropic1024coarse': defaultdict(lambda: 1, {'getdata': 0.002}),
        'isotropic1024fine': defaultdict(lambda: 1, {'getdata': 0.0002}),
        'isotropic4096': defaultdict(lambda: 1),
        'isotropic8192': defaultdict(lambda: 1),
        'sabl2048low': defaultdict(lambda: 1),
        'sabl2048high': defaultdict(lambda: 1, {'getdata': 0.075}),
        'rotstrat4096': defaultdict(lambda: 1),
        'mhd1024': defaultdict(lambda: 1, {'getdata': 0.0025}),
        'mixing': defaultdict(lambda: 1, {'getdata': 0.04}),
        'channel': defaultdict(lambda: 1, {'getdata': 0.0065}),
        'channel5200': defaultdict(lambda: 1),
        'transition_bl': defaultdict(lambda: 1, {'getdata': 0.25})
    }[dataset_title][query_type]
    
def get_time_index_shift(dataset_title, query_type):
    """
    addition to map the time to a correct time index in the filenames. e.g. the first time index in the 'sabl2048high' filenames
    is '000'; this time index is disallowed for queries to handle 'pchip' time interpolation queries. so, time '0' specified by the user corresponds
    to the filename time index '001'. for 'sabl2048high' this means the time index shift is +1 for 'getdata' queries. the defaults of -1 (getcutout)
    and +0 (getdata*) are used for converting low-resolution datasets time indices to 0-based time indices and also as a placeholder value for
    the pyJHTDB datasets. *it is assumed that the pyJHTDB datasets have a preceding timepoint ('-1') for 'pchip' interpolation.
    """
    return {
        'isotropic1024coarse': defaultdict(lambda: -1, {'getdata': 0}),
        'isotropic1024fine': defaultdict(lambda: -1, {'getdata': 0}),
        'isotropic4096': defaultdict(lambda: -1),
        'isotropic8192': defaultdict(lambda: -1),
        'sabl2048low': defaultdict(lambda: -1),
        'sabl2048high': defaultdict(lambda: 0, {'getdata': 1}),
        'rotstrat4096': defaultdict(lambda: -1),
        'mhd1024': defaultdict(lambda: -1, {'getdata': 0}),
        'mixing': defaultdict(lambda: -1, {'getdata': 0}),
        'channel': defaultdict(lambda: -1, {'getdata': 0}),
        'channel5200': defaultdict(lambda: -1),
        'transition_bl': defaultdict(lambda: -1, {'getdata': 0})
    }[dataset_title][query_type]

def get_time_index_from_timepoint(dataset_title, timepoint, tint, query_type):
    """
    get the corresponding time index for this dataset from the specified timepoint. handles datasets that allow 'pchip' time interpolation, which
    requires 2 timepoints worth of data on either side of the timepoint specified by the user.
    """
    # dt between timepoints.
    dt = get_time_dt(dataset_title, query_type)
    # addition to map the time to a correct time index in the filename.
    time_index_shift = get_time_index_shift(dataset_title, query_type)
    
    # convert the timepoint to a time index.
    time_index = (timepoint / dt) + time_index_shift
    # round the time index the nearest time index grid point if 'none' time interpolation was specified.
    if tint == 'none':
        time_index = int(math.floor(time_index + 0.5))
    
    return time_index

def get_giverny_datasets():
    # get the dataset titles that are processed by the giverny code (this backend code, *not* the legacy pyJHTDB code).
    return ['isotropic8192', 'sabl2048low', 'sabl2048high']

def get_manual_sql_metadata_datasets():
    # get the dataset titles for which the sql metadata table needs to be manually created.
    return ['sabl2048low', 'sabl2048high']

def get_filename_prefix(dataset_title):
    # get the common filename prefix for each database file in the dataset. some filename prefixes are placeholders because they are processed
    # by the legacy code.
    return {
        'isotropic1024coarse': 'iso1024coarse',
        'isotropic1024fine': 'iso1024fine',
        'isotropic4096': 'iso4096',
        'isotropic8192': 'iso8192',
        'sabl2048low': 'sabl2048a',
        'sabl2048high': 'sabl2048b',
        'rotstrat4096': 'rotstrat4096',
        'mhd1024': 'mhd1024',
        'mixing': 'mixing',
        'channel': 'channel',
        'channel5200': 'channel5200',
        'transition_bl': 'transitionbl'
    }[dataset_title]

def get_value_names():
    # map of the value names for each variable, e.g. "ux" is the x-component of the velocity. 'vel': velocity, 'pr': pressure,
    # 'en': energy, 'temp': temperature, 'frc': force, 'mgnt': magnetic field, 'ptnt': vector potential, 'dnst': density, 'pst': position.
    return {
        'vel':{1:'ux', 2:'uy', 3:'uz'},
        'pr':{1:'p'},
        'en':{1:'e'},
        'temp':{1:'t'},
        'frc':{1:'fx', 2:'fy', 3:'fz'},
        'mgnt':{1:'bx', 2:'by', 3:'bz'},
        'ptnt':{1:'ax', 2:'ay', 3:'az'},
        'dnst':{1:'rho'},
        'pst':{1:'x', 2:'y', 3:'z'}
    }

def get_num_values_per_datapoint(variable_id):
    # get the number of values per datapoint for the user-specified variable.
    return {
        'vel':3,
        'pr':1,
        'en':1,
        'temp':1,
        'frc':3,
        'mgnt':3,
        'ptnt':3,
        'dnst':1,
        'pst':3
    }[variable_id]

def get_variable_identifier(variable_name):
    # convert the variable name to its corresponding identifier, e.g. convert "velocity" to "vel".
    return {
        'velocity':'vel',
        'pressure':'pr',
        'energy':'en',
        'temperature':'temp',
        'force':'frc',
        'magneticfield':'mgnt',
        'vectorpotential':'ptnt',
        'density':'dnst',
        'position':'pst'
    }[variable_name]

def get_output_title(dataset_title):
    # format the dataset title string for the contour plot titles.
    return {
        'isotropic1024coarse':'Isotropic 1024 (coarse)',
        'isotropic1024fine':'Isotropic 1024 (fine)',
        'isotropic4096':'Isotropic 4096',
        'isotropic8192':'Isotropic 8192',
        'sabl2048low':'Stable Atmospheric Boundary Layer 2048 (low-rate)',
        'sabl2048high':'Stable Atmospheric Boundary Layer 2048 (high-rate)',
        'rotstrat4096':'Rotating Stratified 4096',
        'mhd1024':'Magneto-hydrodynamic Isotropic 1024',
        'mixing':'Homogeneous Buoyancy Driven 1024',
        'channel':'Channel Flow',
        'channel5200':'Channel Flow (Re = 5200)',
        'transition_bl':'Transitional Boundary Layer'
    }[dataset_title]

def get_output_variable_name(variable_name):
    # format the variable name string for the HDF5 output file dataset name.
    return {
        'velocity':'Velocity',
        'pressure':'Pressure',
        'energy':'Energy',
        'temperature':'Temperature',
        'force':'Force',
        'magneticfield':'MagneticField',
        'vectorpotential':'VectorPotential',
        'density':'Density',
        'position':'Position'
    }[variable_name]

def get_attribute_type(variable_name):
    # get the attribute type of the output data.
    return {
        'velocity':'Vector',
        'pressure':'Scalar',
        'energy':'Scalar',
        'temperature':'Scalar',
        'force':'Vector',
        'magneticfield':'Vector',
        'vectorpotential':'Vector',
        'density':'Scalar',
        'position':'Vector'
    }[variable_name]

def get_timepoint_digits(dataset_title):
    # get the number of digits in the timepoint in order to add leading zeros. some timepoint digits are placeholders because they are processed
    # by the legacy code.
    return {
        'isotropic1024coarse':1,
        'isotropic1024fine':1,
        'isotropic4096':1,
        'isotropic8192':1,
        'sabl2048low':3,
        'sabl2048high':3,
        'rotstrat4096':1,
        'mhd1024':1,
        'mixing':1,
        'channel':1,
        'channel5200':2,
        'transition_bl':1
    }[dataset_title]

def get_interpolation_tsv_header(dataset_title, variable_name, timepoint, timepoint_end, delta_t, sint, tint):
    """
    get the interpolation tsv header.
    """
    # parse the interpolation method and operator from sint.
    sint_split = sint.split('_')
    
    # interpolation method (e.g. 'lag4', 'm2q8', 'fd6noint').
    method = sint.replace('_g', '').replace('_h', '').replace('_l', '')
    
    # interpolation operator (function, gradient, hessian, or laplacian).
    operator = 'function'
    if any(operator_key == sint_split[-1] for operator_key in ['g', 'h', 'l']):
        operator = {
            'g':'gradient',
            'h':'hessian',
            'l':'laplacian'
        }[sint_split[-1]]
    
    if variable_name == 'position':
        point_header = f'dataset: {dataset_title}, variable: {variable_name}, timepoint: {timepoint}, timepoint end: {timepoint_end}, delta t: {delta_t}, temporal method: {tint}, ' + \
                       f'spatial method: {method}, spatial operator: {operator}\n'
    else:
        point_header = f'dataset: {dataset_title}, variable: {variable_name}, timepoint: {timepoint}, temporal method: {tint}, spatial method: {method}, spatial operator: {operator}\n'
    point_header += 'x_point\ty_point\tz_point'
    
    return {
        'velocity_function':point_header + '\tux\tuy\tuz',
        'velocity_gradient':point_header + '\tduxdx\tduxdy\tduxdz\tduydx\tduydy\tduydz\tduzdx\tduzdy\tduzdz',
        'velocity_hessian':point_header + '\td2uxdxdx\td2uxdxdy\td2uxdxdz\td2uxdydy\td2uxdydz\td2uxdzdz' + \
                                          '\td2uydxdx\td2uydxdy\td2uydxdz\td2uydydy\td2uydydz\td2uydzdz' + \
                                          '\td2uzdxdx\td2uzdxdy\td2uzdxdz\td2uzdydy\td2uzdydz\td2uzdzdz',
        'velocity_laplacian':point_header + '\tgrad2ux\tgrad2uy\tgrad2uz',
        'pressure_function':point_header + '\tp',
        'pressure_gradient':point_header + '\tdpdx\tdpdy\tdpdz',
        'pressure_hessian':point_header + '\td2pdxdx\td2pdxdy\td2pdxdz\td2pdydy\td2pdydz\td2pdzdz',
        'energy_function':point_header + '\te',
        'energy_gradient':point_header + '\tdedx\tdedy\tdedz',
        'energy_hessian':point_header + '\td2edxdx\td2edxdy\td2edxdz\td2edydy\td2edydz\td2edzdz',
        'temperature_function':point_header + '\tt',
        'temperature_gradient':point_header + '\tdtdx\tdtdy\tdtdz',
        'temperature_hessian':point_header + '\td2tdxdx\td2tdxdy\td2tdxdz\td2tdydy\td2tdydz\td2tdzdz',
        'force_function':point_header + '\tfx\tfy\tfz',
        'magneticfield_function':point_header + '\tbx\tby\tbz',
        'magneticfield_gradient':point_header + '\tdbxdx\tdbxdy\tdbxdz\tdbydx\tdbydy\tdbydz\tdbzdx\tdbzdy\tdbzdz',
        'magneticfield_hessian':point_header + '\td2bxdxdx\td2bxdxdy\td2bxdxdz\td2bxdydy\td2bxdydz\td2bxdzdz' + \
                                               '\td2bydxdx\td2bydxdy\td2bydxdz\td2bydydy\td2bydydz\td2bydzdz' + \
                                               '\td2bzdxdx\td2bzdxdy\td2bzdxdz\td2bzdydy\td2bzdydz\td2bzdzdz',
        'magneticfield_laplacian':point_header + '\tgrad2bx\tgrad2by\tgrad2bz',
        'vectorpotential_function':point_header + '\tax\tay\taz',
        'vectorpotential_gradient':point_header + '\tdaxdx\tdaxdy\tdaxdz\tdaydx\tdaydy\tdaydz\tdazdx\tdazdy\tdazdz',
        'vectorpotential_hessian':point_header + '\td2axdxdx\td2axdxdy\td2axdxdz\td2axdydy\td2axdydz\td2axdzdz' + \
                                                 '\td2aydxdx\td2aydxdy\td2aydxdz\td2aydydy\td2aydydz\td2aydzdz' + \
                                                 '\td2azdxdx\td2azdxdy\td2azdxdz\td2azdydy\td2azdydz\td2azdzdz',
        'vectorpotential_laplacian':point_header + '\tgrad2ax\tgrad2ay\tgrad2az',
        'density_function':point_header + '\trho',
        'density_gradient':point_header + '\tdrhodx\tdrhody\tdrhodz',
        'density_hessian':point_header + '\td2rhodxdx\td2rhodxdy\td2rhodxdz\td2rhodydy\td2rhodydz\td2rhodzdz',
        'position_function':point_header + '\tx\ty\tz'
    }[f'{variable_name}_{operator}']

"""
processing gizmos.
"""
def assemble_axis_data(axes_data):
    # assemble all of the axis data together into one numpy array.
    return np.array(axes_data, dtype = np.ndarray)

def convert_to_0_based_value(value):
    # convert the 1-based value to a 0-based value.
    return value - 1

def get_axes_ranges_num_datapoints(axes_ranges):
    # number of datapoints along each axis.
    return axes_ranges[:, 1] - axes_ranges[:, 0] + 1

def convert_to_0_based_ranges(axes_ranges, cube_resolution):
    # calculate the number of datapoints along each axis range.
    axes_lengths = get_axes_ranges_num_datapoints(axes_ranges)
    
    # convert the 1-based axes ranges to 0-based axes ranges.
    axes_ranges = axes_ranges - 1
    
    # truncate the axes range if necessary.
    for axis_index, axis_range in enumerate(axes_ranges):
        if axes_lengths[axis_index] > cube_resolution:
            axes_ranges[axis_index, 1] = axes_ranges[axis_index, 0] + cube_resolution - 1
    
    return axes_ranges
    
"""
output folder gizmos.
"""
def create_output_folder(output_path):
    # create the output folder directory.
    os.makedirs(output_path, exist_ok = True)
        
"""
user-interfacing gizmos.
"""
def write_interpolation_tsv_file(cube, points, interpolation_data, output_filename):
    """
    write the interpolation results to a tsv file.
    """
    print('Writing the interpolation .tsv file...')
    sys.stdout.flush()

    # create the output folder if it does not already exist.
    create_output_folder(cube.output_path)

    # concatenate the points and interpolation_data matrices together.
    output_data = np.concatenate((points, interpolation_data), axis = 1)

    # write the tsv file.
    with open(cube.output_path.joinpath(output_filename + '.tsv'), 'w', newline = '') as output_file:
        # output header.
        output_header = [header_row.split('\t') 
                         for header_row in get_interpolation_tsv_header(cube.dataset_title, cube.var_name, cube.timepoint_original, cube.timepoint_end, cube.delta_t,
                                                                        cube.sint, cube.tint).split('\n')]

        # create a csv writer object with tab delimiter.
        writer = csv.writer(output_file, delimiter = '\t')
        # write the header row to the tsv file.
        writer.writerows(output_header)
        # write output_data to the tsv file.
        writer.writerows(output_data)

    print('\nFile written successfully.')
    print('-----')
    sys.stdout.flush()
            
def write_cutout_hdf5_and_xmf_files(cube, output_data, axes_ranges, output_filename):
    """
    write the hdf5 and xmf files for the getCutout result.
    """
    print('Writing the cutout .h5 and .xmf files...')
    sys.stdout.flush()
    
    # write output_data to a hdf5 file.
    # -----
    output_data.to_netcdf(cube.output_path.joinpath(output_filename + '.h5'),
                          format = "NETCDF4", mode = "w")
    
    # write the xmf file.
    # -----
    # get the dataset name used for the hdf5 file.
    h5_var_name = get_output_variable_name(cube.var_name)
    h5_attribute_type = get_attribute_type(cube.var_name)
    h5_dataset_name = cube.dataset_name
    
    # the shape of the cutout.
    shape = axes_ranges[:, 1] - axes_ranges[:, 0] + 1
    
    # get the output timepoint.
    xmf_timepoint = cube.timepoint_original
    
    output_str = f"""<?xml version=\"1.0\" ?>
    <!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>
    <Xdmf Version=\"2.0\">
      <Domain>
          <Grid Name=\"Structured Grid\" GridType=\"Uniform\">
            <Time Value=\"{xmf_timepoint}\"/>
            <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"{shape[2]} {shape[1]} {shape[0]}\"/>
            <Geometry GeometryType=\"VXVYVZ\">
              <DataItem Name=\"Xcoor\" Dimensions=\"{shape[0]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                {output_filename}.h5:/xcoor
              </DataItem>
              <DataItem Name=\"Ycoor\" Dimensions=\"{shape[1]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                {output_filename}.h5:/ycoor
              </DataItem>
              <DataItem Name=\"Zcoor\" Dimensions=\"{shape[2]}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                {output_filename}.h5:/zcoor
              </DataItem>
            </Geometry>
            <Attribute Name=\"{h5_var_name}\" AttributeType=\"{h5_attribute_type}\" Center=\"Node\">
              <DataItem Dimensions=\"{shape[2]} {shape[1]} {shape[0]} {cube.num_values_per_datapoint}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">
                {output_filename}.h5:/{h5_dataset_name}
              </DataItem>
            </Attribute>
          </Grid>
      </Domain>
    </Xdmf>"""

    with open(cube.output_path.joinpath(output_filename + '.xmf'), 'w') as output_file:
        output_file.write(output_str)
    
    print('\nFiles written successfully.')
    print('-----')
    sys.stdout.flush()

def contour_plot(cube, value_index_original, cutout_data, plot_ranges, axes_ranges, strides, output_filename,
                colormap = 'inferno'):
    """
    create a contour plot from the getCutout output.
    """
    # dictionaries.
    # -----
    variable = cube.var_name
    
    # variable identifier, e.g. "vel" for "velocity".
    variable_id = get_variable_identifier(variable)
    # names for each value, e.g. value index 0 for velocity data corresponds to the x-component of the velocity ("ux").
    value_name_map = get_value_names()
    
    # create the output folder if it does not already exist.
    create_output_folder(cube.output_path)
    
    # exception handling.
    # -----
    # check that the user-input x-, y-, and z-axis plot ranges are all specified correctly as [minimum, maximum] integer values.
    check_axes_ranges(plot_ranges)
    
    # check that the user specified a valid value index.
    if value_index_original not in value_name_map[variable_id]:
        raise Exception(f"{value_index_original} is not a valid value_index: {list(value_name_map[variable_id].keys())}")
        
    # transposed minimum and maximum arrays for both plot_ranges and axes_ranges.
    plot_ranges_min = plot_ranges[:, 0]
    plot_ranges_max = plot_ranges[:, 1]
    
    axes_min = axes_ranges[:, 0]
    axes_max = axes_ranges[:, 1]
    
    # raise exception if all of the plot datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_min <= plot_ranges_min) and np.all(plot_ranges_max <= axes_max)):
        raise Exception(f'the specified plot ranges are not all bounded by the box volume defined by:\n{axes_ranges}')
        
    # determine how many of the axis minimum values are equal to their corresponding axis maximum value.
    num_axes_equal_min_max = plot_ranges_min == plot_ranges_max
    # raise exception if the data being plotted is not 2-dimensional.
    if np.count_nonzero(num_axes_equal_min_max == True) != 1:
        raise Exception(f'only one axis (x, y, or z) should be specified as a single point, e.g. z_plot_range = [3, 3], to create a contour plot')
        
    # convert the requested plot ranges to the domain [0, 2 * pi].
    xcoor_values = np.around(np.arange(plot_ranges_min[0] - 1, plot_ranges_max[0], strides[0], dtype = np.float32) * cube.dx, cube.decimals)
    ycoor_values = np.around(np.arange(plot_ranges_min[1] - 1, plot_ranges_max[1], strides[1], dtype = np.float32) * cube.dx, cube.decimals)
    zcoor_values = np.around(np.arange(plot_ranges_min[2] - 1, plot_ranges_max[2], strides[2], dtype = np.float32) * cube.dx, cube.decimals)

    # generate the plot.
    print('Generating contour plot...')
    print('-----')
    sys.stdout.flush()
    
    # -----
    # convert the 1-based value_index_original to a 0-based index for python.
    value_index = convert_to_0_based_value(value_index_original)
    
    # name of the value that is being plotted.
    value_name = value_name_map[variable_id][value_index_original]
    
    # get the formatted dataset title for use in the plot title.
    output_dataset_title = get_output_title(cube.dataset_title)
    
    # specify the subset (or full) axes ranges to use for plotting. cutout_data is of the format [z-range, y-range, x-range, output value index].
    plot_data = cutout_data[cube.dataset_name].sel(xcoor = xcoor_values, 
                                                   ycoor = ycoor_values, 
                                                   zcoor = zcoor_values,
                                                   values = value_index)
    
    # raise exception if only one point is going to be plotted along more than 1 axis. a contour plot requires more 
    # than 1 point along 2 axes. this check is required in case the user specifies a stride along an axis that 
    # is >= number of datapoints along that axis.
    if plot_data.shape.count(1) > 1:
        raise Exception('the contour plot could not be created because more than 1 axis only had 1 datapoint')
    
    # create the figure.
    fig = plt.figure(figsize = (11, 8.5), dpi = 300)
    fig.set_facecolor('white')
    ax = fig.add_subplot(111)
    cf = plot_data.plot(ax = ax, cmap = colormap, center = False,
                        cbar_kwargs = {'label': None})

    # plot labels.
    # convert the [0, 2 * pi] domain plot label to the integer index in the [1, 8192] domain.
    original_axis_title = ax.get_title().replace('coor', '').split('=')
    plane_axis = original_axis_title[0].strip()
    plane_point = int(np.round(float(original_axis_title[-1].strip()) / cube.dx)) + 1
    axis_title = plane_axis + ' = ' + str(plane_point) + ', t = ' + str(cutout_data.attrs['t_start'])
    title_str = f'{output_dataset_title}\n{variable} ({value_name}) contour plot ({axis_title})'
    # get the x-axis and y-axis variable names (e.g. 'x' and 'y') before the axis labels are appended to.
    x_axis_variable = ax.get_xlabel()
    y_axis_variable = ax.get_ylabel()
    x_label = str(x_axis_variable).split('coor')[0].strip()
    y_label = str(y_axis_variable).split('coor')[0].strip()
    x_axis_stride = cutout_data.attrs[f'{x_label}_step']
    y_axis_stride = cutout_data.attrs[f'{y_label}_step']
    plt.title(title_str, fontsize = 16, weight = 'bold')
    plt.xlabel(f'{x_label} (stride = {x_axis_stride})', fontsize = 14, weight = 'bold')
    plt.ylabel(f'{y_label} (stride = {y_axis_stride})', fontsize = 14, weight = 'bold')
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # adjust the axis ticks to the center of each datapoint.
    x_ticks = [xlims[0] + ((x_axis_stride * cube.dx) / 2), xlims[1] - ((x_axis_stride * cube.dx) / 2)]
    y_ticks = [ylims[0] + ((y_axis_stride * cube.dx) / 2), ylims[1] - ((y_axis_stride * cube.dx) / 2)]
    # axis datapoints.
    x_axis_points = np.rint(plot_data.coords[x_axis_variable].values / cube.dx) + 1
    y_axis_points = np.rint(plot_data.coords[y_axis_variable].values / cube.dx) + 1
    plt.xticks(x_ticks, [int(x_axis_points[0]), int(x_axis_points[-1])])
    plt.yticks(y_ticks, [int(y_axis_points[0]), int(y_axis_points[-1])])
    
    # save the figure.
    plt.tight_layout()
    plt.savefig(cube.output_path.joinpath(output_filename + '.png'))
    
    # show the figure in the notebook, and shrinks the dpi to make it easily visible.
    fig.set_dpi(67)
    plt.tight_layout()
    plt.show()
    
    # close the figure.
    plt.close()
    
    print('-----')
    print('Contour plot created successfully.')
    sys.stdout.flush()

def cutout_values(cube, x, y, z, output_data, axes_ranges, strides):
    """
    retrieve data values for all of the specified points.
    """
    
    # minimum and maximum endpoints along each axis for the points the user requested.
    endpoints_min = np.array([np.min(x), np.min(y), np.min(z)], dtype = np.int32)
    endpoints_max = np.array([np.max(x), np.max(y), np.max(z)], dtype = np.int32)
    
    # exception_handling.
    # -----
    # raise exception if all of the user requested datapoints are not inside the bounds of the user box volume.
    if not(np.all(axes_ranges[:, 0] <= endpoints_min) and np.all(endpoints_max <= axes_ranges[:, 1])):
        raise Exception(f'the specified point(s) are not all bounded by the box volume defined by:\n{axes_ranges}')
    
    # convert the requested plot ranges to 0-based indices and then to their corresponding values in the domain [0, 2 * pi].
    xcoor_values = np.around(np.arange(endpoints_min[0] - 1, endpoints_max[0], strides[0], dtype = np.float32) * cube.dx, cube.decimals)
    ycoor_values = np.around(np.arange(endpoints_min[1] - 1, endpoints_max[1], strides[1], dtype = np.float32) * cube.dx, cube.decimals)
    zcoor_values = np.around(np.arange(endpoints_min[2] - 1, endpoints_max[2], strides[2], dtype = np.float32) * cube.dx, cube.decimals)

    # value(s) corresponding to the specified (x, y, z) datapoint(s).
    return output_data[cube.dataset_name].sel(xcoor = xcoor_values,
                                              ycoor = ycoor_values,
                                              zcoor = zcoor_values)
