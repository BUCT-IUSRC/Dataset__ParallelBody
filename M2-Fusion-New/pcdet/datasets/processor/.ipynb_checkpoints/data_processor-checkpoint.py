from functools import partial

import numpy as np

from ...utils import box_utils, common_utils
import time


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        # mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        lidar_mask = common_utils.mask_points_by_range(data_dict['lidar_points'], self.point_cloud_range)
        radar_mask = common_utils.mask_points_by_range(data_dict['radar_points'], self.point_cloud_range)
        # data_dict['points'] = data_dict['points'][mask]
        data_dict['lidar_points'] = data_dict['lidar_points'][lidar_mask]
        data_dict['radar_points'] = data_dict['radar_points'][radar_mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            # points = data_dict['points']
            lidar_points = data_dict['lidar_points']
            radar_points = data_dict['radar_points']
            # shuffle_idx = np.random.permutation(points.shape[0])
            lidar_shuffle_idx = np.random.permutation(lidar_points.shape[0])
            radar_shuffle_idx = np.random.permutation(radar_points.shape[0])
            # points = points[shuffle_idx]
            lidar_points = lidar_points[lidar_shuffle_idx]
            radar_points = radar_points[radar_shuffle_idx]
            # data_dict['points'] = points
            data_dict['lidar_points'] = lidar_points
            data_dict['radar_points'] = radar_points

        return data_dict

    ### change
    # def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None, voxel_generator_008=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE

            ############################################ 008voxel ########################################
            # 修改
            POINT_CLOUD_RANGE_008 = [-25.6, -10, -3, 25.6, 41.2,  1]
            POINT_CLOUD_RANGE_008 = [-38.4, -10, -3, 38.4, 66.8,  1]
            voxel_generator_008 = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE_008,
                point_cloud_range=POINT_CLOUD_RANGE_008,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            ############################################ 008voxel ########################################

            # return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator, voxel_generator_008=voxel_generator_008)

        # points = data_dict['points']
        lidar_points = data_dict['lidar_points']
        radar_points = data_dict['radar_points']
        # voxel_output = voxel_generator.generate(points)
        lidar_voxel_output = voxel_generator.generate(lidar_points)
        radar_voxel_output = voxel_generator.generate(radar_points)

        ### 008
        lidar_voxel_output_008 = voxel_generator_008.generate(lidar_points)
        radar_voxel_output_008 = voxel_generator_008.generate(radar_points)
        # if isinstance(voxel_output, dict):
        #     voxels, coordinates, num_points = \
        #         voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        # else:
        #     voxels, coordinates, num_points = voxel_output
        if isinstance(lidar_voxel_output, dict):
            lidar_voxels, lidar_coordinates, lidar_num_points = \
                lidar_voxel_output['voxels'], lidar_voxel_output['coordinates'], lidar_voxel_output['num_points_per_voxel']
            radar_voxels, radar_coordinates, radar_num_points = \
                radar_voxel_output['voxels'], radar_voxel_output['coordinates'], radar_voxel_output['num_points_per_voxel']

            ### _008
            lidar_voxels_008, lidar_coordinates_008, lidar_num_points_008 = \
                lidar_voxel_output_008['voxels'], lidar_voxel_output_008['coordinates'], lidar_voxel_output_008['num_points_per_voxel']
            radar_voxels_008, radar_coordinates_008, radar_num_points_008 = \
                radar_voxel_output_008['voxels'], radar_voxel_output_008['coordinates'], radar_voxel_output_008['num_points_per_voxel']
        else:
            lidar_voxels, lidar_coordinates, lidar_num_points = lidar_voxel_output
            radar_voxels, radar_coordinates, radar_num_points = radar_voxel_output

            ### _008
            lidar_voxels_008, lidar_coordinates_008, lidar_num_points_008 = lidar_voxel_output_008
            radar_voxels_008, radar_coordinates_008, radar_num_points_008 = radar_voxel_output_008

        if not data_dict['use_lead_xyz']:
            # voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
            lidar_voxels = lidar_voxels[..., 3:]  # remove xyz in voxels(N, 3)
            radar_voxels = radar_voxels[..., 3:]  # remove xyz in voxels(N, 3)

            ### _008
            lidar_voxels_008 = lidar_voxels_008[..., 3:]  # remove xyz in voxels(N, 3)
            radar_voxels_008 = radar_voxels_008[..., 3:]  # remove xyz in voxels(N, 3)

        # data_dict['voxels'] = voxels
        data_dict['lidar_voxels'] = lidar_voxels
        data_dict['radar_voxels'] = radar_voxels
        # data_dict['voxel_coords'] = coordinates
        data_dict['lidar_voxel_coords'] = lidar_coordinates
        data_dict['radar_voxel_coords'] = radar_coordinates
        # data_dict['voxel_num_points'] = num_points
        data_dict['lidar_voxel_num_points'] = lidar_num_points
        data_dict['radar_voxel_num_points'] = radar_num_points

        ### _008
        data_dict['lidar_voxels_008'] = lidar_voxels_008
        data_dict['lidar_voxel_coords_008'] = lidar_coordinates_008
        data_dict['lidar_voxel_num_points_008'] = lidar_num_points_008
        data_dict['radar_voxels_008'] = radar_voxels_008
        data_dict['radar_voxel_coords_008'] = radar_coordinates_008
        data_dict['radar_voxel_num_points_008'] = radar_num_points_008

        matrix_index_size = 2000
        radar_pillars_matrix_index = np.ones((matrix_index_size, matrix_index_size), dtype=np.int64)
        lidar_pillars_matrix_index = np.ones((matrix_index_size, matrix_index_size), dtype=np.int64)
        radar_pillars_matrix_index = -radar_pillars_matrix_index
        lidar_pillars_matrix_index = -lidar_pillars_matrix_index
        for i in range(len(radar_coordinates_008)):
            radar_pillars_matrix_index[int(radar_coordinates_008[i][1])][int(radar_coordinates_008[i][2])] = i
        data_dict['radar_pillars_matrix_index'] = radar_pillars_matrix_index
        
        for i in range(len(lidar_coordinates_008)):
            lidar_pillars_matrix_index[int(lidar_coordinates_008[i][1])][int(lidar_coordinates_008[i][2])] = i
        data_dict['lidar_pillars_matrix_index'] = lidar_pillars_matrix_index


        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        # points = data_dict['points']
        lidar_points = data_dict['lidar_points']
        radar_points = data_dict['radar_points']
        # if num_points < len(points):
        #     pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        #     pts_near_flag = pts_depth < 40.0
        #     far_idxs_choice = np.where(pts_near_flag == 0)[0]
        #     near_idxs = np.where(pts_near_flag == 1)[0]
        #     # near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
        #     choice = []
        #     if num_points > len(far_idxs_choice):
        #         near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
        #         choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
        #             if len(far_idxs_choice) > 0 else near_idxs_choice
        #     else: 
        #         choice = np.arange(0, len(points), dtype=np.int32)
        #         choice = np.random.choice(choice, num_points, replace=False)
        #     np.random.shuffle(choice)
        # else:
        #     choice = np.arange(0, len(points), dtype=np.int32)
        #     if num_points > len(points):
        #         if num_points >= len(points)*2:
        #             extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
        #         else:
        #             extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
        #         choice = np.concatenate((choice, extra_choice), axis=0)
        #     np.random.shuffle(choice)
        if num_points < len(lidar_points):
            lidar_pts_depth = np.linalg.norm(lidar_points[:, 0:3], axis=1)
            lidar_pts_near_flag = lidar_pts_depth < 40.0
            lidar_far_idxs_choice = np.where(lidar_pts_near_flag == 0)[0]
            lidar_near_idxs = np.where(lidar_pts_near_flag == 1)[0]
            lidar_choice = []
            if num_points > len(lidar_far_idxs_choice):
                lidar_near_idxs_choice = np.random.choice(lidar_near_idxs, num_points - len(lidar_far_idxs_choice), replace=False)
                lidar_choice = np.concatenate((lidar_near_idxs_choice, lidar_far_idxs_choice), axis=0) \
                    if len(lidar_far_idxs_choice) > 0 else lidar_near_idxs_choice
            else: 
                lidar_choice = np.arange(0, len(lidar_points), dtype=np.int32)
                lidar_choice = np.random.choice(lidar_choice, num_points, replace=False)
            np.random.shuffle(lidar_choice)
        else:
            lidar_choice = np.arange(0, len(lidar_points), dtype=np.int32)
            if num_points > len(lidar_points):
                if num_points >= len(lidar_points)*2:
                    lidar_extra_choice = np.random.choice(lidar_choice, num_points - len(lidar_points), replace=True)
                else:
                    lidar_extra_choice = np.random.choice(lidar_choice, num_points - len(lidar_points), replace=False)
                lidar_choice = np.concatenate((lidar_choice, lidar_extra_choice), axis=0)
            np.random.shuffle(lidar_choice)
        if num_points < len(radar_points):
            radar_pts_depth = np.linalg.norm(radar_points[:, 0:3], axis=1)
            radar_pts_near_flag = radar_pts_depth < 40.0
            radar_far_idxs_choice = np.where(radar_pts_near_flag == 0)[0]
            radar_near_idxs = np.where(radar_pts_near_flag == 1)[0]
            radar_choice = []
            if num_points > len(radar_far_idxs_choice):
                radar_near_idxs_choice = np.random.choice(radar_near_idxs, num_points - len(radar_far_idxs_choice), replace=False)
                radar_choice = np.concatenate((radar_near_idxs_choice, radar_far_idxs_choice), axis=0) \
                    if len(radar_far_idxs_choice) > 0 else radar_near_idxs_choice
            else: 
                radar_choice = np.arange(0, len(radar_points), dtype=np.int32)
                radar_choice = np.random.choice(radar_choice, num_points, replace=False)
            np.random.shuffle(radar_choice)
        else:
            radar_choice = np.arange(0, len(radar_points), dtype=np.int32)
            if num_points > len(radar_points):
                if num_points >= len(radar_points)*2:
                    radar_extra_choice = np.random.choice(radar_choice, num_points - len(radar_points), replace=True)
                else:
                    radar_extra_choice = np.random.choice(radar_choice, num_points - len(radar_points), replace=False)
                radar_choice = np.concatenate((radar_choice, radar_extra_choice), axis=0)
            np.random.shuffle(radar_choice)
        # data_dict['points'] = points[choice]
        data_dict['lidar_points'] = lidar_points[lidar_choice]
        data_dict['radar_points'] = radar_points[radar_choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
