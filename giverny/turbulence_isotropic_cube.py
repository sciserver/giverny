import os
import sys
import h5py
import math
#import time
#import psutil
import shutil
#import struct
#import tracemalloc
import numpy as np
import SciServer.CasJobs as cj
from dask.distributed import Client, LocalCluster
# installs morton-py if necessary.
try:
    import morton
except ImportError as e:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'morton-py'])
    #%pip install morton-py
finally:
    import morton

class iso_cube:
    def __init__(self, cube_num, cube_dimensions = 3, cube_title = ''):
        # cube size.
        self.N = cube_num
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.cube_title = cube_title
        
        # setting up Morton curve.
        bits = int(math.log(self.N, 2))
        self.mortoncurve = morton.Morton(dimensions = cube_dimensions, bits = bits)
        
        self.init_cache()
        
    def init_cache(self):
        # read SQL metadata for all of the turbulence data files into the cache
        sql = f"""
        select dbm.ProductionMachineName
        , dbm.ProductionDatabaseName
        , dbm.minLim, dbm.maxLim
        , dbm.minTime, dbm.maxTime
        , dp.path
        from databasemap dbm
           join datapath{str(self.N)} dp
             on dp.datasetid=dbm.datasetid
           and dp.productionmachinename=dbm.productionmachinename
           and dp.ProductionDatabaseName=dbm.ProductionDatabaseName
        where dbm.datasetname = '{self.cube_title}'
        order by minlim
        """
        df = cj.executeQuery(sql, "turbinfo")
        
        x, y, z = self.mortoncurve.unpack(df['minLim'].values)
        df['x_min'] = x
        df['y_min'] = y
        df['z_min'] = z
        
        x, y, z = self.mortoncurve.unpack(df['maxLim'].values)
        df['x_max'] = x
        df['y_max'] = y 
        df['z_max'] = z
        
        self.cache = df
    
    # defines some helper functions, all hardcoded (double-check this when other datasets are available)
    def parse_corner_points(self, x_min, x_max, y_min, y_max, z_min, z_max):
        # only points 1, 2, 4, and 5 are required for finding the correct sub-boxes.
        # corner 1 is the bottom left back side origin point.
        # corner 2 is the bottom right back side corner point (same as corner 1 except at the maximum x-position).
        # corner 4 is the bottom left front side corner point (same as corner 1 except at the maximum y-positon).
        # corner 5 is the top left back corner point (same as corner 1 except at the maximum z-positon).
        # corners 2, 3, and 4 travel around the bottom plane of the box clockwise from corner 1.
        # corners 6, 7, and 8 travel around the top plane of the box clockwise from corner 5.
        c1 = (x_min, y_min, z_min)
        c2 = (x_max, y_min, z_min)
        #c3 = (x_max, y_max, z_min)
        c4 = (x_min, y_max, z_min)
        c5 = (x_min, y_min, z_max)
        #c6 = (x_max, y_min, z_max)
        #c7 = (x_max, y_max, z_max)
        #c8 = (x_min, y_max, z_max)
        
        corner_points = (c1, c2, c4, c5)
        
        return corner_points
        
    def get_files_for_corner_points(self, x_range, y_range, z_range, var, timepoint):
        # define the corner points.
        x_min = x_range[0]; x_max = x_range[1];
        y_min = y_range[0]; y_max = y_range[1];
        z_min = z_range[0]; z_max = z_range[1];
        
        # retrieve the corner points.
        c_points = self.parse_corner_points(x_min, x_max, y_min, y_max, z_min, z_max)
        
        database_files = []
        
        # only points 1, 2, 4, and 5 are required for finding the correct sub-boxes.
        c1_info = self.get_file_for_point(c_points[0][0], c_points[0][1], c_points[0][2], var, timepoint)
        c1_file = c1_info[0]
        database_files.append(c1_file)
        
        c2_info = self.get_file_for_point(c_points[1][0], c_points[1][1], c_points[1][2], var, timepoint)
        c2_file = c2_info[0]
        database_files.append(c2_file)
        
        c4_info = self.get_file_for_point(c_points[2][0], c_points[2][1], c_points[2][2], var, timepoint)
        c4_file = c4_info[0]
        database_files.append(c4_file)
        
        c5_info = self.get_file_for_point(c_points[3][0], c_points[3][1], c_points[3][2], var, timepoint)
        c5_file = c5_info[0]
        database_files.append(c5_file)
        
        return database_files
    
    def find_sub_box_end_point(self, axis_range, datapoint, axis_position, db_file_comparison, var, timepoint):
        # placeholder end point value. 
        end_point = -1
        # if the difference between the axis range end points is <= to this value, then the end_point
        # has been found.
        axis_range_difference = 2
        
        end_point_found = False
        while not end_point_found:
            mid_point = math.floor((axis_range[0] + axis_range[1]) / 2)
            
            # stops recursively shrinking the box once the difference between the two end points is <= axis_range_difference.
            if (axis_range[1] - axis_range[0]) <= axis_range_difference:
                end_point_found = True
            
            # updates the datapoint to the new mid point.
            datapoint[axis_position] = mid_point
            
            # gets the db file for the new datapoint.
            datapoint_info = self.get_file_for_point(datapoint[0], datapoint[1], datapoint[2], var, timepoint)
            datapoint_file = datapoint_info[0]
            
            # compares the db file for datapoint to the origin point.
            if datapoint_file == db_file_comparison:
                end_point = mid_point
                axis_range[0] = mid_point
            else:
                end_point = mid_point - 1
                axis_range[1] = mid_point
            
            # used for checking that there were no redundant calculations
            #print(f'midpoint = {mid_point}')
            #print(f'endpoint = {end_point}')
            #print('-')
                
        return end_point
    
    def recursive_single_database_file_sub_boxes(self, box, var, timepoint, single_file_boxes):
        db_files = self.get_files_for_corner_points(box[0], box[1], box[2], var, timepoint)
        num_db_files = len(set(db_files))
        
        # checks that the x-, y-, and z- range minimum and maximum points are all within one length of the database cube size. if they are 
        # not, then this algorithm will continue to search for the database file representative boxes, even if the minimum and maximum
        # points are in the same database file (e.g. x_max = x_min + self.N).
        x_min_cube_multiple = math.floor(float(box[0][0]) / float(self.N))
        x_max_cube_multiple = math.floor(float(box[0][1]) / float(self.N))
        y_min_cube_multiple = math.floor(float(box[1][0]) / float(self.N))
        y_max_cube_multiple = math.floor(float(box[1][1]) / float(self.N))
        z_min_cube_multiple = math.floor(float(box[2][0]) / float(self.N))
        z_max_cube_multiple = math.floor(float(box[2][1]) / float(self.N))
        
        x_multiples = len(set([x_min_cube_multiple, x_max_cube_multiple]))
        y_multiples = len(set([y_min_cube_multiple, y_max_cube_multiple]))
        z_multiples = len(set([z_min_cube_multiple, z_max_cube_multiple]))

        if num_db_files == 1 and x_multiples == 1 and y_multiples == 1 and z_multiples == 1:
            unique_db_file = list(set(db_files))[0]
            # this if statement is deprecated.
            #if unique_db_file in single_file_boxes:
            #    raise Exception(f'{unique_db_file} is already in single_file_boxes')
            
            # stores the minLim of the box for use later when reading in the data.
            box_info = self.get_file_for_point(box[0][0], box[1][0], box[2][0], var, timepoint)
            box_minLim = box_info[3]
            
            if unique_db_file not in single_file_boxes:
                single_file_boxes[unique_db_file] = [(box, box_minLim)]
            else:
                single_file_boxes[unique_db_file].append((box, box_minLim))
            
            return
        elif db_files[0] != db_files[1] or x_multiples > 1:
            # this means that the x_range was sufficiently large such that all of the points were
            # not contained in a singular database file.  i.e. the database files were different for
            # corners 1 and 2.  the data x_range will now be recursively split in half to find the first databse file endpoint
            # along this axis.

            # this value is specified as 0 because the x-axis index is 0.  this is used for determing which 
            # point (X, Y, or Z) the midpoint is going to be tested for.  in this case, this section of code
            # is adjusting only the x-axis.
            axis_position = 0
            # stores the c1 corner point (X, Y, Z) of the box to be used for finding the first box end point
            # when shrinking the x-axis into sub-boxes.
            datapoint = [box[0][0], box[1][0], box[2][0]]
            # which axis is sub-divided, in this case it is the x-axis.
            axis_range = list(box[0])
            # determine where the end x-axis point is for the first sub-box.
            first_box_end_point = self.find_sub_box_end_point(axis_range, datapoint, axis_position, db_files[0], \
                                                                       var, timepoint)

            first_sub_box = [[box[0][0], first_box_end_point], box[1], box[2]]
            second_sub_box = [[first_box_end_point + 1, box[0][1]], box[1], box[2]]
            
            sub_boxes = []
            sub_boxes.append(first_sub_box)
            sub_boxes.append(second_sub_box)
            
            for sub_box in sub_boxes:
                self.recursive_single_database_file_sub_boxes(sub_box, var, timepoint, single_file_boxes)
        elif db_files[0] != db_files[2] or y_multiples > 1:
            # this means that the y_range was sufficiently large such that all of the points were
            # not contained in a singular database file.  i.e. the database files were different for
            # corners 1 and 4.  the data y_range will now be recursively split in half to find the first databse file endpoint
            # along this axis.

            # this value is specified as 1 because the y-axis index is 1.  this is used for determing which 
            # point (X, Y, or Z) the midpoint is going to be tested for.  in this case, this section of code
            # is adjusting only the y-axis.
            axis_position = 1
            # stores the c1 corner point (X, Y, Z) of the box to be used for finding the first box end point 
            # when shrinking the y-axis into sub-boxes.
            datapoint = [box[0][0], box[1][0], box[2][0]]
            # which axis is sub-divided, in this case it is the y-axis.
            axis_range = list(box[1])
            # determine where the end y-axis point is for the first sub-box.
            first_box_end_point = self.find_sub_box_end_point(axis_range, datapoint, axis_position, db_files[0], \
                                                                       var, timepoint)

            first_sub_box = [box[0], [box[1][0], first_box_end_point], box[2]]
            second_sub_box = [box[0], [first_box_end_point + 1, box[1][1]], box[2]]

            sub_boxes = []
            sub_boxes.append(first_sub_box)
            sub_boxes.append(second_sub_box)
            
            for sub_box in sub_boxes:
                self.recursive_single_database_file_sub_boxes(sub_box, var, timepoint, single_file_boxes)
        elif db_files[0] != db_files[3] or z_multiples > 1:
            # this means that the z_range was sufficiently large such that all of the points were
            # not contained in a singular database file.  i.e. the database files were different for
            # corners 1 and 5.  the data z_range will now be recursively split in half to find the first databse file endpoint
            # along this axis.

            # this value is specified as 2 because the z-axis index is 2.  this is used for determing which 
            # point (X, Y, or Z) the midpoint is going to be tested for.  in this case, this section of code
            # is adjusting only the z-axis.
            axis_position = 2
            # stores the c1 corner point (X, Y, Z) of the box to be used for finding the first box end point 
            # when shrinking the z-axis into sub-boxes.
            datapoint = [box[0][0], box[1][0], box[2][0]]
            # which axis is sub-divided, in this case it is the z-axis.
            axis_range = list(box[2])
            # determine where the end z-axis point is for the first sub-box.
            first_box_end_point = self.find_sub_box_end_point(axis_range, datapoint, axis_position, db_files[0], \
                                                          var, timepoint)

            first_sub_box = [box[0], box[1], [box[2][0], first_box_end_point]]
            second_sub_box = [box[0], box[1], [first_box_end_point + 1, box[2][1]]]
            
            sub_boxes = []
            sub_boxes.append(first_sub_box)
            sub_boxes.append(second_sub_box)
            
            for sub_box in sub_boxes:
                self.recursive_single_database_file_sub_boxes(sub_box, var, timepoint, single_file_boxes)
    
    def identify_single_database_file_sub_boxes(self, x_range, y_range, z_range, var, timepoint):
        # initially assumes the user specified box contains points in different files. the boxes will be split up until all the points
        # in each box are from a single database file.
        box = [x_range, y_range, z_range]
        single_file_boxes = {}
        self.recursive_single_database_file_sub_boxes(box, var, timepoint, single_file_boxes)
            
        return single_file_boxes
    
    def boxes_contained(self, sub_box, user_box):
        contained = False
        # checks if the sub-divided box is fully contained within the user-specified box.
        if (sub_box[0][0] >= user_box[0][0] and sub_box[0][1] <= user_box[0][1]) and \
            (sub_box[1][0] >= user_box[1][0] and sub_box[1][1] <= user_box[1][1]) and \
            (sub_box[2][0] >= user_box[2][0] and sub_box[2][1] <= user_box[2][1]):
            contained = True
        
        return contained
    
    def boxes_overlap(self, sub_box, user_box):
        overlap = False
        # checks if the sub-divided box and the user-specified box overlap on all 3 axes
        if (sub_box[0][0] <= user_box[0][1] and user_box[0][0] <= sub_box[0][1]) and \
            (sub_box[1][0] <= user_box[1][1] and user_box[1][0] <= sub_box[1][1]) and \
            (sub_box[2][0] <= user_box[2][1] and user_box[2][0] <= sub_box[2][1]):
            overlap = True
            
        return overlap
    
    def determine_min_overlap_point(self, voxel, user_box, axis):
        min_point = None
        
        # checks if the user-specified box minimum value along the given axis is <= the voxel minimum value along the same axis.  if so, then the minimum
        # value is stored as voxel minimum value.  otherwise, the minimum value is stored as the user-specified box minimum value.
        if user_box[axis][0] <= voxel[axis][0]:
            min_point = voxel[axis][0]
        else:
            min_point = user_box[axis][0]
            
        return min_point
    
    def determine_max_overlap_point(self, voxel, user_box, axis):
        max_point = None
        
        # checks if the user-specified box maximum value along the given axis is >= the voxel maximum value along the same axis.  if so, then the maximum
        # value is stored as voxel maximum value.  otherwise, the maximum value is stored as the user-specified box maximum value.
        if user_box[axis][1] >= voxel[axis][1]:
            max_point = voxel[axis][1]
        else:
            max_point = user_box[axis][1]
            
        return max_point
    
    def voxel_ranges_in_user_box(self, voxel, user_box):
        # determine the minimum and maximum values of the overlap, along each axis, between voxel and the user-specified box for a partially overlapped voxel.
        # axis 0 corresponds to the x-axis.
        # axis 1 corresponds to the y-axis.
        # axis 2 corresponds to the z-axis.
        voxel_x_min = self.determine_min_overlap_point(voxel, user_box, axis = 0)
        voxel_x_max = self.determine_max_overlap_point(voxel, user_box, axis = 0)
        
        voxel_y_min = self.determine_min_overlap_point(voxel, user_box, axis = 1)
        voxel_y_max = self.determine_max_overlap_point(voxel, user_box, axis = 1)
        
        voxel_z_min = self.determine_min_overlap_point(voxel, user_box, axis = 2)
        voxel_z_max = self.determine_max_overlap_point(voxel, user_box, axis = 2)
        
        voxel_data = [[voxel_x_min, voxel_x_max], [voxel_y_min, voxel_y_max], [voxel_z_min, voxel_z_max]]
        
        return voxel_data
        
    def recursive_sub_boxes_in_file(self, box, user_db_box, morton_voxels_to_read, voxel_side_length = 8):
        # recursively sub-divides the database file cube until the entire user-specified box is mapped by morton cubes.
        box_x_range = box[0]
        box_y_range = box[1]
        box_z_range = box[2]
        
        # only need to check one axes since each sub-box is a cube. this value will be compared to voxel_side_length to limit the recursive
        # shrinking algorithm.
        sub_box_axes_length = box_x_range[1] - box_x_range[0] + 1
        
        # checks if the sub-box corner points are all inside the portion of the user-specified box in the database file.
        box_fully_contained = self.boxes_contained(box, user_db_box)
        box_partially_contained = self.boxes_overlap(box, user_db_box)
        # recursively shrinks to voxel-sized boxes (8 x 8 x 8), and stores all of the necessary information regarding these boxes
        # that will be used when reading in data from the database file. if a sub-box is fully inside the user-box before getting down 
        # to the voxel level, then the information is stored for this box without shrinking further to save time.
        if box_fully_contained:
            # converts the box (X, Y, Z) minimum and maximum points to morton indices for the database file.
            morton_index_min = self.mortoncurve.pack(box[0][0], box[1][0], box[2][0])
            morton_index_max = self.mortoncurve.pack(box[0][1], box[1][1], box[2][1])
            
            # calculate the number of voxel lengths that fit along each axis, and then from these values calculate the total number
            # of voxels that are inside of this sub-box. each of these values should be an integer because all of the values are divisible.
            num_voxels_x = int((box[0][1] - box[0][0] + 1) / voxel_side_length)
            num_voxels_y = int((box[1][1] - box[1][0] + 1) / voxel_side_length)
            num_voxels_z = int((box[2][1] - box[2][0] + 1) / voxel_side_length)
            num_voxels = num_voxels_x * num_voxels_y * num_voxels_z
            
            # whether the voxel type is fully contained in the user-box ('f') or partially contained ('p') for each voxel.
            voxel_type = ['f'] * num_voxels
            
            # stores the morton indices for reading from the database file efficiently and voxel type information for parsing out the voxel information.
            if morton_voxels_to_read == list():
                morton_voxel_info = [[morton_index_min, morton_index_max], num_voxels, voxel_type]
                morton_voxels_to_read.append(morton_voxel_info)
            else:
                # check if the most recent sub-box maximum is 1 index less than the new sub-box minimum.  if so, then 
                # extend the range of the previous sub-box morton maximum to stitch these two boxes together. also, add the number of voxels 
                # and append the new voxel type information so that the voxel boundaries can be parsed correctly when reading the data.
                if morton_voxels_to_read[-1][0][1] == (morton_index_min - 1):
                    morton_voxels_to_read[-1][0][1] = morton_index_max
                    morton_voxels_to_read[-1][1] += num_voxels
                    morton_voxels_to_read[-1][2] += voxel_type
                else:
                    # start a new morton sequence that will be read in separately.
                    morton_voxel_info = [[morton_index_min, morton_index_max], num_voxels, voxel_type]
                    morton_voxels_to_read.append(morton_voxel_info)
                    
            return
        elif sub_box_axes_length == voxel_side_length:
            if box_fully_contained or box_partially_contained:
                # converts the box (X, Y, Z) minimum and maximum points to morton indices for the database file.
                morton_index_min = self.mortoncurve.pack(box[0][0], box[1][0], box[2][0])
                morton_index_max = self.mortoncurve.pack(box[0][1], box[1][1], box[2][1])
                
                # calculate the number of voxel lengths that fit along each axis, and then from these values calculate the total number
                # of voxels that are inside of this sub-box. each of these values should be an integer because all of the values are divisible.
                num_voxels_x = int((box[0][1] - box[0][0] + 1) / voxel_side_length)
                num_voxels_y = int((box[1][1] - box[1][0] + 1) / voxel_side_length)
                num_voxels_z = int((box[2][1] - box[2][0] + 1) / voxel_side_length)
                num_voxels = num_voxels_x * num_voxels_y * num_voxels_z
                
                # whether the voxel type is fully contained in the user-box ('f') or partially contained ('p') for each voxel.
                voxel_type = 'p'
                if box_fully_contained:
                    voxel_type = 'f'
                
                voxel_type = [voxel_type] * num_voxels
                
                # stores the morton indices for reading from the database file efficiently and voxel info for parsing out the voxel information.
                if morton_voxels_to_read == list():
                    morton_voxel_info = [[morton_index_min, morton_index_max], num_voxels, voxel_type]
                    morton_voxels_to_read.append(morton_voxel_info)
                else:
                    # check if the most recent sub-box maximum is 1 index less than the new sub-box minimum.  if so, then 
                    # extend the range of the previous sub-box morton maximum to stitch these two boxes together. also, add the number of voxels 
                    # and append the new voxel type information so that the voxel boundaries can be parsed correctly when reading the data.
                    if morton_voxels_to_read[-1][0][1] == (morton_index_min - 1):
                        morton_voxels_to_read[-1][0][1] = morton_index_max
                        morton_voxels_to_read[-1][1] += num_voxels
                        morton_voxels_to_read[-1][2] += voxel_type
                    else:
                        # start a new morton sequence that will be read in separately.
                        morton_voxel_info = [[morton_index_min, morton_index_max], num_voxels, voxel_type]
                        morton_voxels_to_read.append(morton_voxel_info)

            return
        else:
            if box_partially_contained:
                # sub-divide the box into 8 sub-cubes (divide the x-, y-, and z- axes in half) and recursively check each box if 
                # it is inside the user-specified box, if necessary.
                box_x_range_midpoint = math.floor((box_x_range[0] + box_x_range[1]) / 2)
                box_y_range_midpoint = math.floor((box_y_range[0] + box_y_range[1]) / 2)
                box_z_range_midpoint = math.floor((box_z_range[0] + box_z_range[1]) / 2)
                
                # ordering sub-boxes 1-8 below in this order maintains the morton-curve index structure, such that 
                # the minimum (X, Y, Z) morton index for a new box only needs to be compared to the last 
                # sub-boxes' maximum (X, Y, Z) morton index to see if they can be stitched together.
                
                # new_sub_box_1 is the sub-box bounded by [x_min, x_midpoint], [y_min, y_midpoint], and [z_min, z_midpoint]
                # new_sub_box_2 is the sub-box bounded by [x_midpoint + 1, x_max], [y_min, y_midpoint], and [z_min, z_midpoint]
                # new_sub_box_3 is the sub-box bounded by [x_min, x_midpoint], [y_midpoint + 1, y_max], and [z_min, z_midpoint]
                # new_sub_box_4 is the sub-box bounded by [x_midpoint + 1, x_max], [y_midpoint + 1, y_max], and [z_min, z_midpoint]
                new_sub_box_1 = [[box_x_range[0], box_x_range_midpoint], [box_y_range[0], box_y_range_midpoint], [box_z_range[0], box_z_range_midpoint]]
                new_sub_box_2 = [[box_x_range_midpoint + 1, box_x_range[1]], [box_y_range[0], box_y_range_midpoint], [box_z_range[0], box_z_range_midpoint]]
                new_sub_box_3 = [[box_x_range[0], box_x_range_midpoint], [box_y_range_midpoint + 1, box_y_range[1]], [box_z_range[0], box_z_range_midpoint]]
                new_sub_box_4 = [[box_x_range_midpoint + 1, box_x_range[1]], [box_y_range_midpoint + 1, box_y_range[1]], [box_z_range[0], box_z_range_midpoint]]
                
                # new_sub_box_5 is the sub-box bounded by [x_min, x_midpoint], [y_min, y_midpoint], and [z_midpoint + 1, z_max]
                # new_sub_box_6 is the sub-box bounded by [x_midpoint + 1, x_max], [y_min, y_midpoint], and [z_midpoint + 1, z_max]
                # new_sub_box_7 is the sub-box bounded by [x_min, x_midpoint], [y_midpoint + 1, y_max], and [z_midpoint + 1, z_max]
                # new_sub_box_8 is the sub-box bounded by [x_midpoint + 1, x_max], [y_midpoint + 1, y_max], and [z_midpoint + 1, z_max]
                new_sub_box_5 = [[box_x_range[0], box_x_range_midpoint], [box_y_range[0], box_y_range_midpoint], [box_z_range_midpoint + 1, box_z_range[1]]]
                new_sub_box_6 = [[box_x_range_midpoint + 1, box_x_range[1]], [box_y_range[0], box_y_range_midpoint], [box_z_range_midpoint + 1, box_z_range[1]]]
                new_sub_box_7 = [[box_x_range[0], box_x_range_midpoint], [box_y_range_midpoint + 1, box_y_range[1]], [box_z_range_midpoint + 1, box_z_range[1]]]
                new_sub_box_8 = [[box_x_range_midpoint + 1, box_x_range[1]], [box_y_range_midpoint + 1, box_y_range[1]], [box_z_range_midpoint + 1, box_z_range[1]]]

                new_sub_boxes = []
                new_sub_boxes.append(new_sub_box_1)
                new_sub_boxes.append(new_sub_box_2)
                new_sub_boxes.append(new_sub_box_3)
                new_sub_boxes.append(new_sub_box_4)
                new_sub_boxes.append(new_sub_box_5)
                new_sub_boxes.append(new_sub_box_6)
                new_sub_boxes.append(new_sub_box_7)
                new_sub_boxes.append(new_sub_box_8)
                    
                for new_sub_box in new_sub_boxes:
                    # checks if a sub-box is at least partially contained inside the user-specified box. if so, then the sub-box will 
                    # be recursively searched until an entire sub-box is inside the user-specified box.
                    new_sub_box_partially_contained = self.boxes_overlap(new_sub_box, user_db_box)

                    if new_sub_box_partially_contained:
                        self.recursive_sub_boxes_in_file(new_sub_box, user_db_box, morton_voxels_to_read, voxel_side_length)
        return
        
    def identify_sub_boxes_in_file(self, user_db_box_original, var, timepoint, voxel_side_length = 8):
        # take the modulus of the axes end points to account for periodic boundary conditions.
        user_db_box = [[user_db_box_axis_range[0] % self.N, user_db_box_axis_range[1] % self.N] for user_db_box_axis_range in user_db_box_original]
        
        # initially assumes the user-specified box in the file is not the entire box representing the file. the database file box will 
        # be sub-divided into morton cubes until the user-specified box is completely mapped by all of these sub-cubes.
        user_db_box_x_range = user_db_box[0]
        user_db_box_y_range = user_db_box[1]
        user_db_box_z_range = user_db_box[2]
        
        user_db_box_x_min = user_db_box_x_range[0]
        user_db_box_y_min = user_db_box_y_range[0]
        user_db_box_z_min = user_db_box_z_range[0]
        
        # retrieve the morton index limits (minLim, maxLim) of the cube representing the whole database file
        f, cornercode, offset, minLim, maxLim = self.get_file_for_point(user_db_box_x_min, user_db_box_y_min, user_db_box_z_min, var, timepoint)
        minLim_xyz = self.mortoncurve.unpack(minLim)
        maxLim_xyz = self.mortoncurve.unpack(maxLim)
        
        # get the box for the entire database file so that it can be recursively broken down into cubes
        db_box = [[minLim_xyz[0], maxLim_xyz[0]], [minLim_xyz[1], maxLim_xyz[1]], [minLim_xyz[2], maxLim_xyz[2]]]
        
        # these are the constituent file sub-cubes that make up the part of the user-specified box in the database file
        morton_voxels_to_read = []
        self.recursive_sub_boxes_in_file(db_box, user_db_box, morton_voxels_to_read, voxel_side_length)

        return morton_voxels_to_read
        
    def get_velocities_for_all_points(self, x_range, y_range, z_range, min_step = 1):
        # manually retrieves the velocities for all points inside the box. this is computationally expensive and not efficient, and 
        # this function is deprecated.
        x_min = x_range[0]; x_max = x_range[1];
        y_min = y_range[0]; y_max = y_range[1];
        z_min = z_range[0]; z_max = z_range[1];
        
        current_x_max = x_max
        current_y_max = y_max
        current_z_max = z_max
        
        velocity_map = {}
        velocity_data = np.array([-1, -1, -1])
        for x_point in np.arange(x_min, x_max + 1, min_step):
            for y_point in np.arange(y_min, y_max + 1, min_step):
                for z_point in np.arange(z_min, z_max + 1, min_step):
                    velocity_data = self.getISO_Point(x_point, y_point, z_point, var = 'vel', timepoint = 0, verbose = False)
                    
                    velocity_map[(x_point, y_point, z_point)] = velocity_data
                    #print(x_point)
                    #print(cornercode, offset)
        
        return velocity_map
        
    def get_offset(self, X, Y, Z):
        """
        TODO is this code correct for velocity as well?  YES
        """
        # morton curve index corresponding to the user specified X, Y, and Z values
        code = self.mortoncurve.pack(X, Y, Z)
        # always looking at an 8 x 8 x 8 box around the grid point, so the shift is always 9 bits to determine 
        # the bottom left corner of the box. the cornercode (bottom left corner of the 8 x 8 x 8 box) is always 
        # in the same file as the user-specified grid point.
        # equivalent to 512 * (math.floor(code / 512))
        cornercode = (code >> 9) << 9
        corner = np.array(self.mortoncurve.unpack(cornercode))
        # calculates the offset between the grid point and corner of the box and converts it to a 4-byte float.
        offset = np.sum((np.array([X, Y, Z]) - corner) * np.array([1, 8, 64]))
        
        return cornercode, offset
    
    def get_file_for_point(self, X, Y, Z, var = 'pr', timepoint = 0):
        """
        querying the cached SQL metadata for the file for the user specified grid point
        """
        # use periodic boundary conditions to adjust the X, Y, and Z values if they are outside the range of the whole dataset cube.
        if X < 0 or X > (self.N - 1):
            X = X % self.N
        
        if Y < 0 or Y > (self.N - 1):
            Y = Y % self.N
            
        if Z < 0 or Z > (self.N - 1):
            Z = Z % self.N
        
        # query the cached SQL metadata for the user-specified grid point.
        cornercode, offset = self.get_offset(X, Y, Z)
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        dataN = t.path.split("/")
        f = f'/home/idies/workspace/turb/data{t.ProductionMachineName[-2:]}_{dataN[2][-2:]}/{dataN[-1]}/{t.ProductionDatabaseName}_{var}_{timepoint}.bin'
        return f, cornercode, offset, t.minLim, t.maxLim
        
    def read_database_files_sequentially(self, sub_db_boxes, \
                                         x_min, y_min, z_min, x_range, y_range, z_range, \
                                         num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder):
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in sub_db_boxes:
            sub_db_boxes_disk_data = sub_db_boxes[database_file_disk]
            
            # read in the voxel data from all of the database files on this disk.
            disk_result_output_data = self.get_iso_points(sub_db_boxes_disk_data, \
                                                          x_min, y_min, z_min, x_range, y_range, z_range, \
                                                          num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder, \
                                                          verbose = False)
            
            result_output_data += disk_result_output_data
            
            # clear disk_result_output_data to free up memory.
            disk_result_output_data = None
            
        return result_output_data
        
    def read_database_files_in_parallel_with_dask(self, sub_db_boxes, \
                                                  x_min, y_min, z_min, x_range, y_range, z_range, \
                                                  num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder, \
                                                  num_processes, dask_cluster_ip):
        # start the dask client for parallel processing.
        if dask_cluster_ip != '':
            # using a premade distributed cluster.
            client = Client(dask_cluster_ip)
            # get the current working directory for saving the zip file of turbulence processing functions to.
            data_dir = os.getcwd() + '/'
            try:
                # upload the turbulence processing functions in the giverny folder to the workers.
                shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny/')
                client.upload_file(data_dir + 'giverny.zip')
            except:
                # close the dask client.
                client.close()
                
                # delete the giverny.zip file if using a premade cluster.
                if os.path.exists(data_dir + 'giverny.zip'):
                    os.remove(data_dir + 'giverny.zip')
                
                # raise an Exception, informing the user of how to fix the issue.
                raise Exception('one of the python files in the "giverny" folder was modified. please shutdown the dask cluster and create a new one.')
        else:
            # using a local cluster.
            cluster = LocalCluster(n_workers = num_processes, processes = True)
            client = Client(cluster)
        
            # old client method. this version is deprecated.
            #client = Client(n_workers = num_processes, processes = True)
        
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in sub_db_boxes:
            sub_db_boxes_disk_data = sub_db_boxes[database_file_disk]
            
            # scatter the data across the distributed workers.
            sub_db_boxes_disk_data_scatter = client.scatter(sub_db_boxes_disk_data)
            # read in the voxel data from all of the database files on this disk.
            disk_result_output_data = client.submit(self.get_iso_points, sub_db_boxes_disk_data_scatter, \
                                                    x_min, y_min, z_min, x_range, y_range, z_range, \
                                                    num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder, \
                                                    verbose = False)
            
            result_output_data.append(disk_result_output_data)
            
            # clear disk_result_output_data to free up memory.
            disk_result_output_data = None
        
        # gather all of the results once they are finished being run in parallel by dask.
        result_output_data = client.gather(result_output_data)
        # flattens result_output_data to match the formatting as when the data is processed sequentially.
        result_output_data = [element for result in result_output_data for element in result]
        
        # close the dask client.
        client.close()
        
        if dask_cluster_ip != '':
            # delete the giverny.zip file if using a premade cluster.
            if os.path.exists(data_dir + 'giverny.zip'):
                os.remove(data_dir + 'giverny.zip')
        else:
            # close the cluster if a local cluster was created.
            cluster.close()
        
        return result_output_data
    
    def get_iso_points(self, sub_db_boxes_disk_data, \
                       x_min, y_min, z_min, x_range, y_range, z_range, \
                       num_values_per_datapoint = 1, bytes_per_datapoint = 4, voxel_side_length = 8, missing_value_placeholder = -999.9, verbose = False):
        """
        retrieve the values for the specified var(iable) in the user-specified box and at the specified timepoint.
        """
        # used to check the memory usage so that it could be minimized.
        #process = psutil.Process(os.getpid())
        #print(f'memory usage 1 (gigabytes) = {(process.memory_info().rss) / (1024**3)}')  # in bytes. 

        # volume of the voxel cube.
        voxel_cube_size = voxel_side_length**3
        
        # defines the user-specified box.
        user_box = [x_range, y_range, z_range]
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files and morton sub-boxes to read the data from.
        for db_file in sub_db_boxes_disk_data:
            # iterate over the user box ranges corresponding to the morton voxels that will be read from this database file.
            for user_box_key in sub_db_boxes_disk_data[db_file]:
                user_box_ranges = user_box_key[0]
                db_minLim = user_box_key[1]
            
                morton_voxels_to_read = sub_db_boxes_disk_data[db_file][user_box_key]

                # retrieve the minimum and maximum (X, Y, Z) coordinates of the morton sub-box that is going to be read in.
                min_xyz = [user_box_ranges[0][0], user_box_ranges[1][0], user_box_ranges[2][0]]
                max_xyz = [user_box_ranges[0][1], user_box_ranges[1][1], user_box_ranges[2][1]]
                
                # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                cube_x_multiple = math.floor(float(min_xyz[0]) / float(self.N)) * self.N
                cube_y_multiple = math.floor(float(min_xyz[1]) / float(self.N)) * self.N
                cube_z_multiple = math.floor(float(min_xyz[2]) / float(self.N)) * self.N

                # create the local output array for this box that will be filled and returned.
                local_output_array = np.full((max_xyz[2] - min_xyz[2] + 1, \
                                              max_xyz[1] - min_xyz[1] + 1, \
                                              max_xyz[0] - min_xyz[0] + 1, \
                                              num_values_per_datapoint), fill_value = missing_value_placeholder, dtype = 'f')

                # iterates over the groups of morton adjacent voxels to minimize the number I/O operations when reading the data.
                for morton_data in morton_voxels_to_read:
                    # the continuous range of morton indices compiled from adjacent voxels that can be read in from the file at the same time.
                    morton_index_range = morton_data[0]
                    # the voxels that will be parsed out from the data that is read in. the voxels need to be parsed separately because the data is sequentially
                    # ordered within a voxel as opposed to morton ordered outside a voxel.
                    num_voxels = morton_data[1]

                    # morton_index_min is equivalent to "cornercode + offset" because morton_index_min is defined as the corner of a voxel.
                    morton_index_min = morton_index_range[0]
                    morton_index_max = morton_index_range[1]
                    morton_index_diff = (morton_index_max - morton_index_min) + 1

                    # the point to seek to in order to start reading the file for this morton index range.
                    seek_distance = num_values_per_datapoint * bytes_per_datapoint * (morton_index_min - db_minLim)
                    # number of bytes to read in from the database file.
                    read_length = num_values_per_datapoint * bytes_per_datapoint * morton_index_diff

                    # read the data. this method is deprecated because it is slow.
                    #with open(db_file, 'rb') as b:
                    #    b.seek(seek_distance)
                    #    xraw = b.read(read_length)
                    #
                    # unpack the data as 4-byte floats.
                    #l = struct.unpack('f' * num_values_per_datapoint * morton_index_diff, xraw)

                    # used to check the memory usage so that it could be minimized.
                    #print(f'memory usage 1a (gigabytes) = {(process.memory_info().rss) / (1024**3)}')  # in bytes. 

                    # read the data efficiently.
                    l = np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance)
                    l = l[np.arange(0, l.size - num_values_per_datapoint + 1, num_values_per_datapoint)[:, None] + np.arange(num_values_per_datapoint)]

                    # used to check the memory usage so that it could be minimized.
                    #print(f'memory usage 1b (gigabytes) = {(process.memory_info().rss) / (1024**3)}')  # in bytes. 

                    # iterate over each voxel in voxel_data.
                    for voxel_count in np.arange(0, num_voxels, 1):
                        # get the origin point (X, Y, Z) for the voxel.
                        voxel_origin_point = self.mortoncurve.unpack(morton_index_min + (voxel_count * voxel_cube_size))

                        # voxel axes ranges.
                        voxel_ranges = [[voxel_origin_point[0] + cube_x_multiple, voxel_origin_point[0] + cube_x_multiple + voxel_side_length - 1], \
                                        [voxel_origin_point[1] + cube_y_multiple, voxel_origin_point[1] + cube_y_multiple + voxel_side_length - 1], \
                                        [voxel_origin_point[2] + cube_z_multiple, voxel_origin_point[2] + cube_z_multiple + voxel_side_length - 1]]

                        # assumed to be a voxel that is fully inside the user-specified box. if the box was only partially contained inside the user-specified
                        # box, then the voxel ranges are corrected to the edges of the user-specified box.
                        voxel_type = morton_data[2][voxel_count]
                        if voxel_type == 'p':
                            voxel_ranges = self.voxel_ranges_in_user_box(voxel_ranges, [x_range, y_range, z_range])

                        # retrieve the x-, y-, and z-ranges for the voxel.
                        voxel_x_range = voxel_ranges[0]
                        voxel_y_range = voxel_ranges[1]
                        voxel_z_range = voxel_ranges[2]

                        # pull out the data that corresponds to this voxel.
                        sub_l_array = l[voxel_count * voxel_cube_size : (voxel_count + 1) * voxel_cube_size]

                        # reshape the sub_l array into a voxel matrix.
                        sub_l_array = sub_l_array.reshape(voxel_side_length, voxel_side_length, voxel_side_length, num_values_per_datapoint)
                        
                        # swap the x- and z- axes to maintain the correct structure. turned this off to leave as (Z, Y, X) for hdf5 file format.
                        #sub_l_array = np.swapaxes(sub_l_array, 0, 2)
                        # remove parts of the voxel that are outside of the user-specified box.
                        if voxel_type == 'p':
                            sub_l_array = sub_l_array[voxel_z_range[0] % voxel_side_length : (voxel_z_range[1] % voxel_side_length) + 1, \
                                                      voxel_y_range[0] % voxel_side_length : (voxel_y_range[1] % voxel_side_length) + 1, \
                                                      voxel_x_range[0] % voxel_side_length : (voxel_x_range[1] % voxel_side_length) + 1]
                        
                        # insert sub_l_array into local_output_data.
                        local_output_array[voxel_z_range[0] - min_xyz[2] : voxel_z_range[1] - min_xyz[2] + 1, \
                                           voxel_y_range[0] - min_xyz[1] : voxel_y_range[1] - min_xyz[1] + 1, \
                                           voxel_x_range[0] - min_xyz[0] : voxel_x_range[1] - min_xyz[0] + 1] = sub_l_array

                        # clear sub_l_array to free up memory.
                        sub_l_array = None

                    # clear l to free up memory.
                    l = None

                # checks to make sure that data was read in for all points.
                if missing_value_placeholder in local_output_array:
                    raise Exception(f'local_output_array was not filled correctly')
                
                # append the filled local_output_array into local_output_data.
                local_output_data.append((local_output_array, min_xyz, max_xyz))

                # clear local_output_array to free up memory.
                local_output_array = None
        
        # used to check the memory usage so that it could be minimized.
        #print(f'memory usage 2 (gigabytes) = {(process.memory_info().rss) / (1024**3)}')  # in bytes. 
        
        return local_output_data
            
    def get_iso_point_original(self, X, Y, Z, var = 'pr', timepoint = 0, verbose = False):
        """
        find the value for the specified var(iable) at the specified point XYZ and specified time.
        Position is assumed to be a point of the grid, i.e. should be integers, and time should be an integer between 0 and 5.
        """
        f, cornercode, offset, minLim, maxLim = self.get_file_for_point(X, Y, Z, var, timepoint)
        if verbose:
            print(f'filename : {f}')
            print(f'cornercode : {cornercode}')
            print(f'corner : {np.array(self.mortoncurve.unpack(cornercode))}')
            print(f'offset : {offset}')
            print(f'minLim : {minLim}')
            print(f'maxLim : {maxLim}')
            #print(f, cornercode, offset, minLim, maxLim)
        
        N = 1
        if var == 'vel':
            N = 3
        
        with open(f, 'rb') as b:
            b.seek(N * 4 * (cornercode + offset - minLim))
            xraw = b.read(4 * N)
        
        l = struct.unpack('f' * N, xraw)
        
        return l
    
    def write_output_matrix_to_hdf5(self, output_data, output_path, output_filename, dataset_name):
        # write output_data to a hdf5 file.
        with h5py.File(output_path + output_filename + '.h5', 'w') as h5f:
            h5f.create_dataset(dataset_name, data = output_data)