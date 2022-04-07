import os
import sys
import glob
import h5py
import math
import shutil
import pathlib
import subprocess
import numpy as np
import SciServer.CasJobs as cj
from dask.distributed import Client, LocalCluster
from giverny.turbulence_gizmos.basic_gizmos import *
# installs morton-py if necessary.
try:
    import morton
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'morton-py'])
finally:
    import morton

class iso_cube:
    def __init__(self, dataset_title = '', cube_dimensions = 3):
        # check that dataset_title is a valid dataset title.
        check_dataset_title(dataset_title)
        
        # cube size.
        self.N = get_dataset_resolution(dataset_title)
        
        # turbulence dataset name, e.g. "isotropic8192" or "isotropic1024fine".
        self.dataset_title = dataset_title
        
        # setting up Morton curve.
        bits = int(math.log(self.N, 2))
        self.mortoncurve = morton.Morton(dimensions = cube_dimensions, bits = bits)
        
        self.init_cache()
        
    def init_cache(self):
        # read SQL metadata for all of the turbulence data files into the cache.
        sql = f"""
        select dbm.ProductionDatabaseName
        , dbm.minLim
        , dbm.maxLim
        from databasemap dbm
        where dbm.datasetname = '{self.dataset_title}'
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
        
        # get map of the filepaths for all of the dataset binary files.
        self.filepaths = self.get_filepaths()
        
        self.cache = df
    
    def get_filepaths(self):
        # map the filepaths to the part of each filename that matches "ProductionDatabaseName" in the SQL metadata for this dataset.
        
        # get the common filename prefix for all files in this dataset, e.g. "iso8192" for the isotropic8192 dataset.
        dataset_filename_prefix = get_filename_prefix(self.dataset_title)
        # recursively search all sub-directories in the turbulence filedb system for the dataset binary files.
        filepaths = sorted(glob.glob(f'/home/idies/workspace/turb/**/{dataset_filename_prefix}*.bin', recursive = True))
        
        # map the filepaths to the filenames so that they can be easily retrieved.
        filepaths_map = {}
        for filepath in filepaths:
            # part of the filenames that exactly matches the "ProductionDatabaseName" column stored in the SQL metadata, plus the variable
            # identifer (e.g. 'vel'), plus the timepoint.
            filename = filepath.split(os.sep)[-1].replace('.bin', '').strip()
            # only add the filepath to the dictionary once since there could be backup copies of the files.
            if filename not in filepaths_map:
                filepaths_map[filename] = filepath
        
        return filepaths_map
    
    # defines some helper functions, all hardcoded (double-check this when other datasets are available)
    def parse_corner_points(self, box):
        # corner 1 is the bottom left back side origin point.
        # corner 2 is the bottom right back side corner point (same as corner 1 except at the maximum x-position).
        # corner 3 is the bottom right front side corner point (same as corner 1 except at maximum x- and y-positions).
        # corner 4 is the bottom left front side corner point (same as corner 1 except at the maximum y-positon).
        # corner 5 is the top left back corner point (same as corner 1 except at the maximum z-positon).
        # corners 2, 3, and 4 travel around the bottom plane of the box clockwise from corner 1.
        # corners 6, 7, and 8 travel around the top plane of the box clockwise from corner 5.
        
        # box minimum and maximum points.
        box_min = [axis_range[0] for axis_range in box]
        box_max = [axis_range[1] for axis_range in box]
        
        # only points 1, 2, 4, and 5 are required for finding the correct sub-boxes. the corner points are returned in order.
        return (
            (box_min[0], box_min[1], box_min[2]),
            (box_max[0], box_min[1], box_min[2]),
            (box_min[0], box_max[1], box_min[2]),
            (box_min[0], box_min[1], box_max[2])
        )
        
    def get_files_for_corner_points(self, box, var, timepoint):
        # retrieve the corner points.
        corner_points = self.parse_corner_points(box)
        
        database_files = []
        # only points 1, 2, 4, and 5 are required for finding the correct sub-boxes. corner_points is ordered.
        for corner_point in corner_points:
            point_info = self.get_file_for_point(corner_point, var, timepoint)
            point_file = point_info[0]
            
            database_files.append(point_file)
        
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
            datapoint_info = self.get_file_for_point(datapoint, var, timepoint)
            datapoint_file = datapoint_info[0]
            
            # compares the db file for datapoint to the origin point.
            if datapoint_file == db_file_comparison:
                end_point = mid_point
                axis_range[0] = mid_point
            else:
                end_point = mid_point - 1
                axis_range[1] = mid_point
                
        return end_point
    
    def recursive_single_database_file_sub_boxes(self, box, var, timepoint, single_file_boxes):
        db_files = self.get_files_for_corner_points(box, var, timepoint)
        num_db_files = len(set(db_files))
        
        # checks that the x-, y-, and z- range minimum and maximum points are all within the same multiple of the database cube size. if they are 
        # not, then this algorithm will continue to search for the database file representative boxes, even if the starting minimum and maximum
        # points are in the same database file (e.g. x_max = x_min + self.N).
        box_multiples = [math.floor(axis_range[0] / self.N) == math.floor(axis_range[1] / self.N) for axis_range in box]

        if num_db_files == 1 and all(box_multiples):
            unique_db_file = list(set(db_files))[0]
            
            # stores the minLim of the box for use later when reading in the data.
            box_info = self.get_file_for_point([axis_range[0] for axis_range in box], var, timepoint)
            box_minLim = box_info[3]
            
            if unique_db_file not in single_file_boxes:
                single_file_boxes[unique_db_file] = [(box, box_minLim)]
            else:
                single_file_boxes[unique_db_file].append((box, box_minLim))
            
            return
        elif db_files[0] != db_files[1] or not box_multiples[0]:
            # this means that the x_range was sufficiently large such that all of the points were
            # not contained in a singular database file.  i.e. the database files were different for
            # corners 1 and 2.  the data x_range will now be recursively split in half to find the first databse file endpoint
            # along this axis.

            # this value is specified as 0 because the x-axis index is 0.  this is used for determing which 
            # point (x, y, or z) the midpoint is going to be tested for.  in this case, this section of code
            # is adjusting only the x-axis.
            axis_position = 0
            # stores the c1 corner point (x, y, z) of the box to be used for finding the first box end point
            # when shrinking the x-axis into sub-boxes.
            datapoint = [axis_range[0] for axis_range in box]
            # which axis is sub-divided, in this case it is the x-axis.
            axis_range = list(box[0])
            # determine where the end x-axis point is for the first sub-box.
            first_box_end_point = self.find_sub_box_end_point(axis_range, datapoint, axis_position, db_files[0],
                                                              var, timepoint)
            
            # append the first and second sub-boxes.
            first_sub_box = [[box[0][0], first_box_end_point], [box[1][0], box[1][1]], [box[2][0], box[2][1]]]
            second_sub_box = [[first_box_end_point + 1, box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]]]
            
            sub_boxes = [first_sub_box, second_sub_box]
            
            for sub_box in sub_boxes:
                self.recursive_single_database_file_sub_boxes(sub_box, var, timepoint, single_file_boxes)
        elif db_files[0] != db_files[2] or not box_multiples[1]:
            # this means that the y_range was sufficiently large such that all of the points were
            # not contained in a singular database file.  i.e. the database files were different for
            # corners 1 and 4.  the data y_range will now be recursively split in half to find the first databse file endpoint
            # along this axis.

            # this value is specified as 1 because the y-axis index is 1.  this is used for determing which 
            # point (x, y, or z) the midpoint is going to be tested for.  in this case, this section of code
            # is adjusting only the y-axis.
            axis_position = 1
            # stores the c1 corner point (x, y, z) of the box to be used for finding the first box end point 
            # when shrinking the y-axis into sub-boxes.
            datapoint = [axis_range[0] for axis_range in box]
            # which axis is sub-divided, in this case it is the y-axis.
            axis_range = list(box[1])
            # determine where the end y-axis point is for the first sub-box.
            first_box_end_point = self.find_sub_box_end_point(axis_range, datapoint, axis_position, db_files[0],
                                                              var, timepoint)
            
            # append the first and second sub-boxes.
            first_sub_box = [[box[0][0], box[0][1]], [box[1][0], first_box_end_point], [box[2][0], box[2][1]]]
            second_sub_box = [[box[0][0], box[0][1]], [first_box_end_point + 1, box[1][1]], [box[2][0], box[2][1]]]
            
            sub_boxes = [first_sub_box, second_sub_box]
            
            for sub_box in sub_boxes:
                self.recursive_single_database_file_sub_boxes(sub_box, var, timepoint, single_file_boxes)
        elif db_files[0] != db_files[3] or not box_multiples[2]:
            # this means that the z_range was sufficiently large such that all of the points were
            # not contained in a singular database file.  i.e. the database files were different for
            # corners 1 and 5.  the data z_range will now be recursively split in half to find the first databse file endpoint
            # along this axis.

            # this value is specified as 2 because the z-axis index is 2.  this is used for determing which 
            # point (x, y, or z) the midpoint is going to be tested for.  in this case, this section of code
            # is adjusting only the z-axis.
            axis_position = 2
            # stores the c1 corner point (x, y, z) of the box to be used for finding the first box end point 
            # when shrinking the z-axis into sub-boxes.
            datapoint = [axis_range[0] for axis_range in box]
            # which axis is sub-divided, in this case it is the z-axis.
            axis_range = list(box[2])
            # determine where the end z-axis point is for the first sub-box.
            first_box_end_point = self.find_sub_box_end_point(axis_range, datapoint, axis_position, db_files[0],
                                                              var, timepoint)
            
            # append the first and second sub-boxes.
            first_sub_box = [[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], first_box_end_point]]
            second_sub_box = [[box[0][0], box[0][1]], [box[1][0], box[1][1]], [first_box_end_point + 1, box[2][1]]]
            
            sub_boxes = [first_sub_box, second_sub_box]
            
            for sub_box in sub_boxes:
                self.recursive_single_database_file_sub_boxes(sub_box, var, timepoint, single_file_boxes)
    
    def identify_single_database_file_sub_boxes(self, box, var, timepoint):
        # initially assumes the user specified box contains points in different files. the boxes will be split up until all the points
        # in each box are from a single database file.
        box = [list(axis_range) for axis_range in box]
        
        single_file_boxes = {}
        self.recursive_single_database_file_sub_boxes(box, var, timepoint, single_file_boxes)
            
        return single_file_boxes
    
    def boxes_contained(self, sub_box, user_box):
        # note: using list comprehension instead of explicit comparisons in the if statement increases the time by approximately 25 percent.
        
        # checks if the sub-divided box is fully contained within the user-specified box.
        if (sub_box[0][0] >= user_box[0][0] and sub_box[0][1] <= user_box[0][1]) and \
           (sub_box[1][0] >= user_box[1][0] and sub_box[1][1] <= user_box[1][1]) and \
           (sub_box[2][0] >= user_box[2][0] and sub_box[2][1] <= user_box[2][1]):
            return True
        
        return False
    
    def boxes_overlap(self, sub_box, user_box):
        # note: using list comprehension instead of explicit comparisons in the if statement increases the time by approximately 25 percent.
        
        # checks if the sub-divided box and the user-specified box overlap on all 3 axes.
        if (sub_box[0][0] <= user_box[0][1] and sub_box[0][1] >= user_box[0][0]) and \
           (sub_box[1][0] <= user_box[1][1] and sub_box[1][1] >= user_box[1][0]) and \
           (sub_box[2][0] <= user_box[2][1] and sub_box[2][1] >= user_box[2][0]):
            return True
            
        return False
    
    def voxel_ranges_in_user_box(self, voxel, user_box):
        # determine the minimum and maximum values of the overlap, along each axis, between voxel and the user-specified box 
        # for a partially overlapped voxel.
        return [[voxel[q][0] if user_box[q][0] <= voxel[q][0] else user_box[q][0], 
                 voxel[q][1] if user_box[q][1] >= voxel[q][1] else user_box[q][1]] for q in range(len(user_box))]
        
    def recursive_sub_boxes_in_file(self, box, user_db_box, morton_voxels_to_read, voxel_side_length = 8):
        # recursively sub-divides the database file cube until the entire user-specified box is mapped by morton cubes.
        
        # only need to check one axes since each sub-box is a cube. this value will be compared to voxel_side_length to limit the recursive
        # shrinking algorithm.
        sub_box_axes_length = box[0][1] - box[0][0] + 1
        
        # checks if the sub-box corner points are all inside the portion of the user-specified box in the database file.
        box_fully_contained = self.boxes_contained(box, user_db_box)
        # recursively shrinks to voxel-sized boxes (8 x 8 x 8), and stores all of the necessary information regarding these boxes
        # that will be used when reading in data from the database file. if a sub-box is fully inside the user-box before getting down 
        # to the voxel level, then the information is stored for this box without shrinking further to save time.
        if box_fully_contained:
            # converts the box (x, y, z) minimum and maximum points to morton indices for the database file.
            morton_index_min = self.mortoncurve.pack(box[0][0], box[1][0], box[2][0])
            morton_index_max = self.mortoncurve.pack(box[0][1], box[1][1], box[2][1])
            
            # calculate the number of voxel lengths that fit along each axis, and then from these values calculate the total number
            # of voxels that are inside of this sub-box. each of these values should be an integer because all of the values are divisible.
            num_voxels = int((box[0][1] - box[0][0] + 1) / voxel_side_length) * \
                         int((box[1][1] - box[1][0] + 1) / voxel_side_length) * \
                         int((box[2][1] - box[2][0] + 1) / voxel_side_length)
            
            # whether the voxel type is fully contained in the user-box ('f') or partially contained ('p') for each voxel.
            voxel_type = ['f'] * num_voxels
            
            # stores the morton indices for reading from the database file efficiently and voxel type information for parsing out the 
            # voxel information.
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
            # converts the box (x, y, z) minimum and maximum points to morton indices for the database file.
            morton_index_min = self.mortoncurve.pack(box[0][0], box[1][0], box[2][0])
            morton_index_max = self.mortoncurve.pack(box[0][1], box[1][1], box[2][1])

            # calculate the number of voxel lengths that fit along each axis, and then from these values calculate the total number
            # of voxels that are inside of this sub-box. each of these values should be an integer because all of the values are divisible.
            num_voxels = int((box[0][1] - box[0][0] + 1) / voxel_side_length) * \
                         int((box[1][1] - box[1][0] + 1) / voxel_side_length) * \
                         int((box[2][1] - box[2][0] + 1) / voxel_side_length)

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
            # sub-divide the box into 8 sub-cubes (divide the x-, y-, and z- axes in half) and recursively check each box if 
            # it is inside the user-specified box, if necessary.
            box_midpoints = [math.floor((axis_range[0] + axis_range[1]) / 2) for axis_range in box]

            # ordering sub-boxes 1-8 below in this order maintains the morton-curve index structure, such that 
            # the minimum (x, y, z) morton index for a new box only needs to be compared to the last 
            # sub-boxes' maximum (x, y, z) morton index to see if they can be stitched together.

            # sub_box_1 is the sub-box bounded by [x_min, x_midpoint], [y_min, y_midpoint], and [z_min, z_midpoint]
            # sub_box_2 is the sub-box bounded by [x_midpoint + 1, x_max], [y_min, y_midpoint], and [z_min, z_midpoint]
            # sub_box_3 is the sub-box bounded by [x_min, x_midpoint], [y_midpoint + 1, y_max], and [z_min, z_midpoint]
            # sub_box_4 is the sub-box bounded by [x_midpoint + 1, x_max], [y_midpoint + 1, y_max], and [z_min, z_midpoint]
            sub_box_1 = [[box[0][0], box_midpoints[0]], [box[1][0], box_midpoints[1]], [box[2][0], box_midpoints[2]]]
            sub_box_2 = [[box_midpoints[0] + 1, box[0][1]], [box[1][0], box_midpoints[1]], [box[2][0], box_midpoints[2]]]
            sub_box_3 = [[box[0][0], box_midpoints[0]], [box_midpoints[1] + 1, box[1][1]], [box[2][0], box_midpoints[2]]]
            sub_box_4 = [[box_midpoints[0] + 1, box[0][1]], [box_midpoints[1] + 1, box[1][1]], [box[2][0], box_midpoints[2]]]

            # sub_box_5 is the sub-box bounded by [x_min, x_midpoint], [y_min, y_midpoint], and [z_midpoint + 1, z_max]
            # sub_box_6 is the sub-box bounded by [x_midpoint + 1, x_max], [y_min, y_midpoint], and [z_midpoint + 1, z_max]
            # sub_box_7 is the sub-box bounded by [x_min, x_midpoint], [y_midpoint + 1, y_max], and [z_midpoint + 1, z_max]
            # sub_box_8 is the sub-box bounded by [x_midpoint + 1, x_max], [y_midpoint + 1, y_max], and [z_midpoint + 1, z_max]
            sub_box_5 = [[box[0][0], box_midpoints[0]], [box[1][0], box_midpoints[1]], [box_midpoints[2] + 1, box[2][1]]]
            sub_box_6 = [[box_midpoints[0] + 1, box[0][1]], [box[1][0], box_midpoints[1]], [box_midpoints[2] + 1, box[2][1]]]
            sub_box_7 = [[box[0][0], box_midpoints[0]], [box_midpoints[1] + 1, box[1][1]], [box_midpoints[2] + 1, box[2][1]]]
            sub_box_8 = [[box_midpoints[0] + 1, box[0][1]], [box_midpoints[1] + 1, box[1][1]], [box_midpoints[2] + 1, box[2][1]]]
            
            new_sub_boxes = [sub_box_1, sub_box_2, sub_box_3, sub_box_4, sub_box_5, sub_box_6, sub_box_7, sub_box_8]

            for new_sub_box in new_sub_boxes:
                # checks if a sub-box is at least partially contained inside the user-specified box. if so, then the sub-box will 
                # be recursively searched until an entire sub-box is inside the user-specified box.
                new_sub_box_partially_contained = self.boxes_overlap(new_sub_box, user_db_box)

                if new_sub_box_partially_contained:
                    self.recursive_sub_boxes_in_file(new_sub_box, user_db_box, morton_voxels_to_read, voxel_side_length)
        return
        
    def identify_sub_boxes_in_file(self, user_db_box_original, var, timepoint, voxel_side_length = 8):
        # initially assumes the user-specified box in the file is not the entire box representing the file. the database file box will 
        # be sub-divided into morton cubes until the user-specified box is completely mapped by all of these sub-cubes.
        
        # take the modulus of the axes end points to account for periodic boundary conditions.
        user_db_box = [[axis_range[0] % self.N, axis_range[1] % self.N] for axis_range in user_db_box_original]
        
        # retrieve the morton index limits (minLim, maxLim) of the cube representing the whole database file.
        f, cornercode, offset, minLim, maxLim = self.get_file_for_point([axis_range[0] for axis_range in user_db_box], var, timepoint)
        minLim_xyz = self.mortoncurve.unpack(minLim)
        maxLim_xyz = self.mortoncurve.unpack(maxLim)
        
        # get the box for the entire database file so that it can be recursively broken down into cubes.
        db_box = [[minLim_xyz[q], maxLim_xyz[q]] for q in range(len(minLim_xyz))]
        
        # these are the constituent file sub-cubes that make up the part of the user-specified box in the database file.
        morton_voxels_to_read = []
        self.recursive_sub_boxes_in_file(db_box, user_db_box, morton_voxels_to_read, voxel_side_length)

        return morton_voxels_to_read
        
    def get_offset(self, datapoint):
        """
        todo: is this code correct for velocity as well?  yes.
        """
        # morton curve index corresponding to the user specified x, y, and z values
        code = self.mortoncurve.pack(datapoint[0], datapoint[1], datapoint[2])
        # always looking at an 8 x 8 x 8 box around the grid point, so the shift is always 9 bits to determine 
        # the bottom left corner of the box. the cornercode (bottom left corner of the 8 x 8 x 8 box) is always 
        # in the same file as the user-specified grid point.
        # equivalent to 512 * (math.floor(code / 512))
        cornercode = (code >> 9) << 9
        corner = np.array(self.mortoncurve.unpack(cornercode))
        # calculates the offset between the grid point and corner of the box and converts it to a 4-byte float.
        offset = np.sum((np.array(datapoint) - corner) * np.array([1, 8, 64]))
        
        return cornercode, offset
    
    def get_file_for_point(self, datapoint, var = 'pr', timepoint = 0):
        """
        querying the cached SQL metadata for the file for the user specified grid point
        """
        # use periodic boundary conditions to adjust the x, y, and z values if they are outside the range of the whole dataset cube.
        datapoint = [point % self.N for point in datapoint]
        
        # query the cached SQL metadata for the user-specified grid point.
        cornercode, offset = self.get_offset(datapoint)
        t = self.cache[(self.cache['minLim'] <= cornercode) & (self.cache['maxLim'] >= cornercode)]
        t = t.iloc[0]
        f = self.filepaths[f'{t.ProductionDatabaseName}_{var}_{timepoint}']
        return f, cornercode, offset, t.minLim, t.maxLim
        
    def read_database_files_sequentially(self, sub_db_boxes,
                                         axes_ranges,
                                         num_values_per_datapoint, bytes_per_datapoint, voxel_side_length,
                                         missing_value_placeholder):
        result_output_data = []
        # iterate over the hard disk drives that the database files are stored on.
        for database_file_disk in sub_db_boxes:
            sub_db_boxes_disk_data = sub_db_boxes[database_file_disk]
            
            # read in the voxel data from all of the database files on this disk.
            disk_result_output_data = self.get_iso_points(sub_db_boxes_disk_data,
                                                          axes_ranges,
                                                          num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder,
                                                          verbose = False)
            
            result_output_data += disk_result_output_data
            
        return result_output_data
        
    def sort_hard_disks(self, sub_db_boxes):
        # sort the hard disks from largest to smallest in terms of the total volume of data (and number of i/o operations) that will
        # be read from each hard disk.
        sub_db_box_sizes = {}
        for database_file_disk in sub_db_boxes:
            # initialize the volume of data to be read ('read volume') and number of i/o operations ('num reads') for this hard disk.
            sub_db_box_sizes[database_file_disk] = {'read volume': 0, 'num reads': 0}
            
            for database_file in sub_db_boxes[database_file_disk]:
                for user_box in sub_db_boxes[database_file_disk][database_file]:
                    # calculate the volume of the box in the database file.
                    read_volume = np.prod([user_box[0][q][1] - user_box[0][q][0] + 1 for q in range(len(user_box[0]))], dtype = int)
                    # retrieve the number of distinct voxel groups that need to be read in separately for this box.
                    num_reads = len(sub_db_boxes[database_file_disk][database_file][user_box])
                    
                    # update 'read volume' and 'num reads' for this hard disk.
                    sub_db_box_sizes[database_file_disk]['read volume'] += read_volume
                    sub_db_box_sizes[database_file_disk]['num reads'] += num_reads
                    
        # sort the hard disks from largest to smallest.
        ordered_sub_db_box_keys = sorted(sub_db_box_sizes,
                                         key = lambda x: (sub_db_box_sizes[x]['read volume'], sub_db_box_sizes[x]['num reads']),
                                         reverse = True)
        
        return ordered_sub_db_box_keys, sub_db_box_sizes
    
    def read_database_files_in_parallel_with_dask(self, sub_db_boxes,
                                                  axes_ranges,
                                                  num_values_per_datapoint, bytes_per_datapoint, voxel_side_length,
                                                  missing_value_placeholder, num_processes):
        # start the dask client for parallel processing.
        # -----
        # flag specifying if the cluster is a premade distributed cluster. assumed to be True to start.
        distributed_cluster = True
        try:
            # using a premade distributed cluster.
            import SciServer.Dask
            
            # attached cluster (when a cluster is created together with a new container).
            client = SciServer.Dask.getClient()
            # deletes data on the network and restarts the workers.
            client.restart()
            
            # get the current working directory for saving the zip file of turbulence processing functions to.
            data_dir = os.getcwd() + os.sep
            
            # upload the turbulence processing functions in the giverny folder to the workers.
            shutil.make_archive(data_dir + 'giverny', 'zip', root_dir = data_dir, base_dir = 'giverny' + os.sep)
            client.upload_file(data_dir + 'giverny.zip')
        except FileNotFoundError:
            # update the distributed_cluster flag to False.
            distributed_cluster = False
            
            # using a local cluster if there is no premade distributed cluster.
            cluster = LocalCluster(n_workers = num_processes, processes = True)
            client = Client(cluster)
        
        # number of hard disks that the database files being read are stored on.
        num_db_disks = len(sub_db_boxes)
        
        # available workers.
        workers = list(client.scheduler_info()['workers'])
        num_workers = len(workers)
        
        # determine how many workers will be utilized for processing the data.
        utilized_workers = num_workers
        if utilized_workers > num_db_disks:
            utilized_workers = num_db_disks
        
        print(f'Database files are being read in parallel ({utilized_workers} workers utilized)...')
        sys.stdout.flush()
        
        # sort sub_db_boxes keys (distinct hard disks) from largest to smallest in terms of the amount of data to be read.
        ordered_sub_db_box_keys, sub_db_box_sizes = self.sort_hard_disks(sub_db_boxes)
        
        result_output_data = []
        worker_size_map = {}
        # iterate over the hard disk drives that the database files are stored on.
        for file_disk_index, database_file_disk in enumerate(ordered_sub_db_box_keys):
            sub_db_boxes_disk_data = sub_db_boxes[database_file_disk]
            
            # identify the worker that has the smallest load assigned to it.
            worker = None
            if len(worker_size_map) < num_workers:
                # use the next available worker and keep track of how much data this worker will read.
                worker = workers[file_disk_index % num_workers]
                worker_size_map[worker] = {'read volume': sub_db_box_sizes[database_file_disk]['read volume'],
                                           'num reads': sub_db_box_sizes[database_file_disk]['num reads']}
            else:
                # choose the worker that has the least amount of work assigned to it. this choice minimizes box volume first, and
                # then the number of i/o operations second if two or more workers are currently assigned to read the same box volume.
                worker = sorted(worker_size_map, key = lambda x: (worker_size_map[x]['read volume'], worker_size_map[x]['num reads']))[0]
                
                # update how much data this worker will read.
                worker_size_map[worker]['read volume'] += sub_db_box_sizes[database_file_disk]['read volume']
                worker_size_map[worker]['num reads'] += sub_db_box_sizes[database_file_disk]['num reads']
            
            # scatter the data across the distributed workers.
            sub_db_boxes_disk_data_scatter = client.scatter(sub_db_boxes_disk_data, workers = worker)
            # read in the voxel data from all of the database files on this disk.
            disk_result_output_data = client.submit(self.get_iso_points, sub_db_boxes_disk_data_scatter,
                                                    axes_ranges,
                                                    num_values_per_datapoint, bytes_per_datapoint, voxel_side_length, missing_value_placeholder,
                                                    verbose = False,
                                                    workers = worker)
            
            result_output_data.append(disk_result_output_data)
        
        # gather all of the results once they are finished being run in parallel by dask.
        result_output_data = client.gather(result_output_data)        
        # flattens result_output_data to match the formatting as when the data is processed sequentially.
        result_output_data = [element for result in result_output_data for element in result]
        
        # close the dask client.
        client.close()
        
        if distributed_cluster:
            # delete the giverny.zip file if using a premade distributed cluster.
            if os.path.exists(data_dir + 'giverny.zip'):
                os.remove(data_dir + 'giverny.zip')
        else:
            # close the cluster if a local cluster was created.
            cluster.close()
        
        return result_output_data
    
    def get_iso_points(self, sub_db_boxes_disk_data,
                       user_box,
                       num_values_per_datapoint = 1, bytes_per_datapoint = 4, voxel_side_length = 8, missing_value_placeholder = -999.9,
                       verbose = False):
        """
        retrieve the values for the specified var(iable) in the user-specified box and at the specified timepoint.
        """
        # volume of the voxel cube.
        voxel_cube_size = voxel_side_length**3
        
        # the collection of local output data that will be returned to fill the complete output_data array.
        local_output_data = []
        # iterate over the database files and morton sub-boxes to read the data from.
        for db_file in sub_db_boxes_disk_data:
            # iterate over the user box ranges corresponding to the morton voxels that will be read from this database file.
            for user_box_key in sub_db_boxes_disk_data[db_file]:
                user_box_ranges = user_box_key[0]
                db_minLim = user_box_key[1]
            
                morton_voxels_to_read = sub_db_boxes_disk_data[db_file][user_box_key]

                # retrieve the minimum and maximum (x, y, z) coordinates of the morton sub-box that is going to be read in.
                min_xyz = [axis_range[0] for axis_range in user_box_ranges]
                max_xyz = [axis_range[1] for axis_range in user_box_ranges]
                
                # calculate the number of periodic cubes that the user-specified box expands into beyond the boundary along each axis.
                cube_multiples = [math.floor(float(min_xyz[q]) / float(self.N)) * self.N for q in range(len(min_xyz))]

                # create the local output array for this box that will be filled and returned.
                local_output_array = np.full((max_xyz[2] - min_xyz[2] + 1,
                                              max_xyz[1] - min_xyz[1] + 1,
                                              max_xyz[0] - min_xyz[0] + 1,
                                              num_values_per_datapoint), fill_value = missing_value_placeholder, dtype = 'f')

                # iterates over the groups of morton adjacent voxels to minimize the number I/O operations when reading the data.
                for morton_data in morton_voxels_to_read:
                    # the continuous range of morton indices compiled from adjacent voxels that can be read in from the file at the same time.
                    # the minimum morton index is equivalent to "cornercode + offset" because it is defined as the corner of a voxel.
                    morton_index_range = morton_data[0]
                    # the voxels that will be parsed out from the data that is read in. the voxels need to be parsed separately because the data 
                    # is sequentially ordered within a voxel as opposed to morton ordered outside a voxel.
                    num_voxels = morton_data[1]

                    # the point to seek to in order to start reading the file for this morton index range.
                    seek_distance = num_values_per_datapoint * bytes_per_datapoint * (morton_index_range[0] - db_minLim)
                    # number of bytes to read in from the database file.
                    read_length = num_values_per_datapoint * bytes_per_datapoint * (morton_index_range[1] - morton_index_range[0] + 1)

                    # read the data efficiently.
                    l = np.fromfile(db_file, dtype = 'f', count = read_length, offset = seek_distance)
                    # group the value(s) for each datapoint together.
                    l = l.reshape(int(len(l) / num_values_per_datapoint), num_values_per_datapoint)
                    
                    # iterate over each voxel in voxel_data.
                    for voxel_count in np.arange(0, num_voxels, 1):
                        # get the origin point (x, y, z) for the voxel.
                        voxel_origin_point = self.mortoncurve.unpack(morton_index_range[0] + (voxel_count * voxel_cube_size))

                        # voxel axes ranges.
                        voxel_ranges = [[voxel_origin_point[0] + cube_multiples[0], voxel_origin_point[0] + cube_multiples[0] + voxel_side_length - 1],
                                        [voxel_origin_point[1] + cube_multiples[1], voxel_origin_point[1] + cube_multiples[1] + voxel_side_length - 1],
                                        [voxel_origin_point[2] + cube_multiples[2], voxel_origin_point[2] + cube_multiples[2] + voxel_side_length - 1]]

                        # assumed to be a voxel that is fully inside the user-specified box. if the box was only partially contained inside the 
                        # user-specified box, then the voxel ranges are corrected to the edges of the user-specified box.
                        voxel_type = morton_data[2][voxel_count]
                        if voxel_type == 'p':
                            voxel_ranges = self.voxel_ranges_in_user_box(voxel_ranges, user_box)

                        # pull out the data that corresponds to this voxel.
                        sub_l_array = l[voxel_count * voxel_cube_size : (voxel_count + 1) * voxel_cube_size]

                        # reshape the sub_l array into a voxel matrix.
                        sub_l_array = sub_l_array.reshape(voxel_side_length, voxel_side_length, voxel_side_length, num_values_per_datapoint)
                        
                        # remove parts of the voxel that are outside of the user-specified box.
                        if voxel_type == 'p':
                            sub_l_array = sub_l_array[voxel_ranges[2][0] % voxel_side_length : (voxel_ranges[2][1] % voxel_side_length) + 1,
                                                      voxel_ranges[1][0] % voxel_side_length : (voxel_ranges[1][1] % voxel_side_length) + 1,
                                                      voxel_ranges[0][0] % voxel_side_length : (voxel_ranges[0][1] % voxel_side_length) + 1]
                        
                        # insert sub_l_array into local_output_data.
                        local_output_array[voxel_ranges[2][0] - min_xyz[2] : voxel_ranges[2][1] - min_xyz[2] + 1,
                                           voxel_ranges[1][0] - min_xyz[1] : voxel_ranges[1][1] - min_xyz[1] + 1,
                                           voxel_ranges[0][0] - min_xyz[0] : voxel_ranges[0][1] - min_xyz[0] + 1] = sub_l_array

                # checks to make sure that data was read in for all points.
                if missing_value_placeholder in local_output_array:
                    raise Exception(f'local_output_array was not filled correctly')
                
                # append the filled local_output_array into local_output_data.
                local_output_data.append((local_output_array, min_xyz, max_xyz)) 
        
        return local_output_data
    
    def write_output_matrix_to_hdf5(self, output_data, output_path, output_filename, dataset_name):
        # write output_data to a hdf5 file.
        with h5py.File(output_path.joinpath(output_filename + '.h5'), 'w') as h5f:
            h5f.create_dataset(dataset_name, data = output_data)
            