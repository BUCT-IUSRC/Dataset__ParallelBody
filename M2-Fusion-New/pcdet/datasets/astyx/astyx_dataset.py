import copy
import pickle
import json

import numpy as np
from skimage import io

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils
from pcdet.datasets.astyx.object3d_astyx import Object3dAstyx, inv_trans
from pcdet.datasets.dataset import DatasetTemplate
###
from copy import deepcopy
import math
import argparse
from easydict import EasyDict as edict
import torch
import os

def filter_det_range(dets, close, far,k1):
    dets = deepcopy(dets)
    if dets['location'].shape[0] == 0:
        return dets
    valid_idx = (np.abs(dets['location'][:, 2]) > close) * \
        (np.abs(dets['location'][:, 2]) <= far)
    for k in dets:
        if k == k1 :
            continue
        dets[k] = dets[k][valid_idx]
    return dets

class kitti_config():
    CLASS_NAME_TO_ID = {
        'Pedestrian': -99,
        'Car': 0,
        'Cyclist': -99,
        'Van': 0,
        'Truck': 0,
        'Person_sitting': -99,
        'Tram': -99,
        'Misc': -99,
        'DontCare': -1
    }

    colors = [[0, 0, 255], [0, 255, 255], [255, 0, 0], [255, 120, 0],
            [255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

    boundary = {
        "minX": 0,
        "maxX": 69.12,
        "minY": -34.56,
        "maxY": 34.56,
        "minZ": -2.73,
        "maxZ": 1.27
    }

    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    bound_size_z = boundary['maxZ'] - boundary['minZ']

    boundary_back = {
        "minX": -50,
        "maxX": 0,
        "minY": -25,
        "maxY": 25,
        "minZ": -2.73,
        "maxZ": 1.27
    }

    BEV_WIDTH = 480  # across y axis -25m ~ 25m
    BEV_HEIGHT = 480  # across x axis 0m ~ 50m
    DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT

    # maximum number of points per voxel
    T = 35

    # voxel size
    vd = 0.1  # z
    vh = 0.05  # y
    vw = 0.05  # x

    # voxel grid
    W = math.ceil(bound_size_x / vw)
    H = math.ceil(bound_size_y / vh)
    D = math.ceil(bound_size_z / vd)

    # Following parameters are calculated as an average from KITTI dataset for simplicity
    Tr_velo_to_cam = np.array([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ])

    # cal mean from train set
    R0 = np.array([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ])

    P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
                [0., 719.787081, 174.545111, 0.1066855],
                [0., 0., 1., 3.0106472e-03],
                [0., 0., 0., 0]
                ])

    R0_inv = np.linalg.inv(R0)
    Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
    P2_inv = np.linalg.pinv(P2)
    #####################################################################################

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation using PyTorch')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')

    parser.add_argument('--root-dir', type=str, default='../', metavar='PATH',
                        help='The ROOT working directory')
    ####################################################################
    ##############     Model configs            ########################
    ####################################################################
    parser.add_argument('--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--hflip_prob', type=float, default=0.5,
                        help='The probability of horizontal flip')
    parser.add_argument('--no-val', action='store_true',
                        help='If true, dont evaluate the model on the val set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='mini-batch size (default: 16), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--print_freq', type=int, default=50, metavar='N',
                        help='print frequency (default: 50)')
    parser.add_argument('--tensorboard_freq', type=int, default=50, metavar='N',
                        help='frequency of saving tensorboard (default: 50)')
    parser.add_argument('--checkpoint_freq', type=int, default=2, metavar='N',
                        help='frequency of saving checkpoints (default: 5)')
    ####################################################################
    ##############     Training strategy            ####################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=300, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr_type', type=str, default='cosin',
                        help='the type of learning rate scheduler (cosin or multi_step or one_cycle)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.949, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0., metavar='WD',
                        help='weight decay (default: 0.)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--steps', nargs='*', default=[150, 180],
                        help='number of burn in step')

    ####################################################################
    ##############     Loss weight            ##########################
    ####################################################################

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')

    configs = edict(vars(parser.parse_args(args=[])))

    ####################################################################
    ############## Hardware configurations #############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    cnf = kitti_config

    configs.pin_memory = True
    configs.input_size = (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)
    configs.down_ratio = 2
    configs.hm_size = (cnf.BEV_WIDTH/configs.down_ratio, cnf.BEV_HEIGHT/configs.down_ratio)
    configs.max_objects = 50

    configs.imagenet_pretrained = True
    configs.head_conv = 256
    configs.num_classes = 1
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos 8 for bin cos sin
    configs.voxel_size = [0.16, 0.16, 4]
    # configs.point_cloud_range =[0, -34.56, -2.73, 69.12, 34.56, 1.27]
    configs.point_cloud_range = [0, -38.4, -3, 76.8, 38.4, 1]
    configs.max_number_of_points_per_voxel = 32


    configs.heads = {
        'hm_cen': configs.num_classes, #1
        'cen_offset': configs.num_center_offset, #2
        'direction': configs.num_direction, #2
        'z_coor': configs.num_z, #1
        'dim': configs.num_dim #3
    }

    configs.num_input_features = 4

    ####################################################################
    ############## Dataset, logs, Checkpoints dir ######################
    ####################################################################
    configs.dataset_dir = '/media/wx/File/data/kittidata'
    configs.checkpoints_dir = os.path.join(configs.root_dir, 'checkpoints', configs.saved_fn)
    configs.logs_dir = os.path.join(configs.root_dir, 'logs', configs.saved_fn)

    if not os.path.isdir(configs.checkpoints_dir):
        os.makedirs(configs.checkpoints_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    return configs



###

class AstyxDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.astyx_infos = []
        self.include_astyx_data(self.mode)

        #self.pc_type = self.dataset_cfg.POINT_CLOUD_TYPE[0]
        if 'radar' in self.dataset_cfg.POINT_CLOUD_TYPE and 'lidar' in self.dataset_cfg.POINT_CLOUD_TYPE :
            self.pc_type = 'fusion'
        else:
            self.pc_type = self.dataset_cfg.POINT_CLOUD_TYPE[0]

        #######
        configs = parse_train_configs()
        self.max_objects = configs.max_objects
        self.hm_size = configs.hm_size
        self.num_classes = configs.num_classes

    def include_astyx_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Astyx dataset')
        astyx_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                astyx_infos.extend(infos)

        self.astyx_infos.extend(astyx_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Astyx dataset: %d' % (len(astyx_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
        assert lidar_file.exists()
        return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0,1,2,3))

    def get_radar(self, idx):
        radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
        assert radar_file.exists()
        return np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0,1,2,3))

    def get_pointcloud(self, idx, pc_type, T_from_radar_to_lidar):
        rotate = T_from_radar_to_lidar[0:3, 0:3]
        translation = T_from_radar_to_lidar[0:3, 3]
        if pc_type == 'lidar':
            lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
            assert lidar_file.exists()
            return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0, 1, 2, 3))
        elif pc_type == 'radar':
            radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
            assert radar_file.exists()
            return np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 4))
        elif pc_type == 'fusion':
            lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
            assert lidar_file.exists()
            lidar = np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0, 1, 2, 3))
            radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
            assert radar_file.exists()
            radar = np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 4))
            lidar[:, :3] = np.dot(lidar[:, :3], rotate)
            lidar[:, :3] += translation
            x = radar[:, 0]
            z = radar[:, 2]
            x = np.sqrt(x*x + z*z)*np.cos(20/96*np.arctan2(z, x))
            z = np.sqrt(x*x + z*z)*np.sin(20/96*np.arctan2(z, x))
            radar[:, 0] = x
            radar[:, 2] = z
            return lidar, radar
        else:
            pass

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'camera_front' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'groundtruth_obj3d' / ('%s.json' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            data = json.load(f)
        objects = [Object3dAstyx.from_label(obj) for obj in data['objects']]
        return objects

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calibration' / ('%s.json' % idx)
        assert calib_file.exists()
        with open(calib_file, 'r') as f:
            data = json.load(f)

        T_from_lidar_to_radar = np.array(data['sensors'][1]['calib_data']['T_to_ref_COS'])
        T_from_camera_to_radar = np.array(data['sensors'][2]['calib_data']['T_to_ref_COS'])
        K = np.array(data['sensors'][2]['calib_data']['K'])
        T_from_radar_to_lidar = inv_trans(T_from_lidar_to_radar)
        T_from_radar_to_camera = inv_trans(T_from_camera_to_radar)
        return {'T_from_radar_to_lidar': T_from_radar_to_lidar,
                'T_from_radar_to_camera': T_from_radar_to_camera,
                'T_from_lidar_to_radar': T_from_lidar_to_radar,
                'T_from_camera_to_radar': T_from_camera_to_radar,
                'K': K}

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'pc_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)
            info['calib'] = calib

            if has_label:
                obj_list = self.get_label(sample_idx)
                for obj in obj_list:
                    obj.from_radar_to_camera(calib)
                    obj.from_radar_to_image(calib)

                annotations = {'name': np.array([obj.cls_type for obj in obj_list]),
                               'occluded': np.array([obj.occlusion for obj in obj_list]),
                               'alpha': np.array([-np.arctan2(obj.loc[1], obj.loc[0])
                                                  + obj.rot_camera for obj in obj_list]),
                               'bbox': np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0),
                               'dimensions': np.array([[obj.l, obj.h, obj.w] for obj in obj_list]),
                               'location': np.concatenate([obj.loc_camera.reshape(1, 3) for obj in obj_list], axis=0),
                               'rotation_y': np.array([obj.rot_camera for obj in obj_list]),
                               'score': np.array([obj.score for obj in obj_list]),
                               'difficulty': np.array([obj.level for obj in obj_list], np.int32),
                               'truncated': -np.ones(len(obj_list))}

                # num_objects标签数不会变
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                if self.pc_type == 'radar':
                    gt_boxes = np.array([[*obj.loc, obj.w, obj.l, obj.h, obj.rot] for obj in obj_list])
                elif self.pc_type == 'fusion':
                    gt_boxes = np.array([[*obj.loc, obj.w, obj.l, obj.h, obj.rot] for obj in obj_list])

                    ### add
                    cen_labels = np.array([[obj.cat_id, *obj.loc, obj.h, obj.w, obj.l, obj.rot] for obj in obj_list])
                else:
                    for obj in obj_list:
                        obj.from_radar_to_lidar(calib)
                    # fang The astyx annotation is based on the radar coordinate system.
                    gt_boxes = np.array([[*obj.loc_lidar, obj.w, obj.l, obj.h, obj.rot_lidar] for obj in obj_list])
                annotations['gt_boxes'] = gt_boxes

                ###
                try:
                    cen_labels
                except NameError:
                    var_exists = False
                else:
                    var_exists = True
                if var_exists:
                    annotations['cen_labels'] = cen_labels
                ###

                info['annos'] = annotations

                if count_inside_pts:
                    calib = self.get_calib(sample_idx)
                    T_from_radar_to_lidar = calib['T_from_radar_to_lidar']
                    #fang Returns the laser point cloud in the laser coordinate system and the radar point cloud in the radar coordinate system.
                    lidar_points, radar_points = self.get_pointcloud(sample_idx, self.pc_type, T_from_radar_to_lidar)

                    corners = box_utils.boxes_to_corners_3d(gt_boxes)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    lidar_num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    radar_num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    # for k in range(num_objects):
                    #     flag = box_utils.in_hull(points[:, 0:3], corners[k])
                    #     num_points_in_gt[k] = flag.sum()
                    for k in range(num_objects):
                        lidar_flag = box_utils.in_hull(lidar_points[:, 0:3], corners[k])
                        radar_flag = box_utils.in_hull(radar_points[:, 0:3], corners[k])
                        lidar_num_points_in_gt[k] = lidar_flag.sum()
                        radar_num_points_in_gt[k] = radar_flag.sum()
                    num_points_in_gt = lidar_num_points_in_gt + radar_num_points_in_gt
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('astyx_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['pc_idx']
            calib = self.get_calib(sample_idx)
            # points = self.get_pointcloud(sample_idx, self.pc_type)
            T_from_radar_to_lidar = calib['T_from_radar_to_lidar']
            lidar_points, radar_points = self.get_pointcloud(sample_idx, self.pc_type, T_from_radar_to_lidar)
            # points = 
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes']

            num_obj = gt_boxes.shape[0]
            # point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            #     torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            # ).numpy()  # (nboxes, npoints)
            lidar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(lidar_points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
            radar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(radar_points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                # gt_points = points[point_indices[i] > 0]
                lidar_gt_points = lidar_points[lidar_point_indices[i] > 0]
                radar_gt_points = radar_points[radar_point_indices[i] > 0]

                # gt_points[:, :3] -= gt_boxes[i, :3]
                lidar_gt_points[:, :3] -= gt_boxes[i, :3]
                radar_gt_points[:, :3] -= gt_boxes[i, :3]
                gt_points = np.concatenate((lidar_gt_points, radar_gt_points), axis=0)
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_3d': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]

            obj_list = [Object3dAstyx.from_prediction(box, label, score, self.pc_type) for box, label, score
                        in zip(pred_boxes,pred_labels,pred_scores)]
            for i, obj in enumerate(obj_list):
                if self.pc_type == 'radar':
                    obj.from_radar_to_camera(calib)
                    obj.from_radar_to_image(calib)
                    loc = obj.loc
                elif self.pc_type == 'fusion':
                    obj.from_radar_to_camera(calib)
                    obj.from_radar_to_image(calib)
                    loc = obj.loc
                else:
                    obj.from_lidar_to_camera(calib)
                    obj.from_lidar_to_image(calib)
                    loc = obj.loc_lidar
                pred_dict['dimensions'][i, :] = np.array([obj.l, obj.h, obj.w])
                pred_dict['location'][i, :] = np.array(obj.loc_camera)
                pred_dict['rotation_y'][i] = np.array(obj.rot_camera)
                pred_dict['alpha'][i] = -np.arctan2(loc[1], loc[0]) + obj.rot_camera
                pred_dict['bbox'][i, :] = np.array(obj.box2d)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.astyx_infos[0].keys():
            return None, {}

        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # use the same kitti eval package

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.astyx_infos]

        dt_annos_range = [filter_det_range(dets1, 60,76.8,'frame_id') for dets1 in eval_det_annos]
        # # print(dt_annos_range)

        gt_annos_range = [filter_det_range(dets2, 60,76.8,'frame_id') for dets2 in eval_gt_annos]

        ##################################################################
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(gt_annos_range, dt_annos_range, class_names)
        # ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

########################## targets #######################################
    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    def gen_hm_radius(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap

    def compute_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

    cnf = kitti_config
    # def get_cen_label():
    def build_targets(self, labels, hflipped):
        cnf = kitti_config
        minX = cnf.boundary['minX']
        maxX = cnf.boundary['maxX']
        minY = cnf.boundary['minY']
        maxY = cnf.boundary['maxY']
        minZ = cnf.boundary['minZ']
        maxZ = cnf.boundary['maxZ']

        num_objects = min(len(labels), self.max_objects)
        hm_l, hm_w = self.hm_size

        hm_main_center = np.zeros((self.num_classes,int(hm_l), int(hm_w)), dtype=np.float32)
        cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
        direction = np.zeros((self.max_objects, 2), dtype=np.float32)
        z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
        dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

        #anglebin = np.zeros((self.max_objects, 2), dtype=np.float32)
        #angleoffset = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices_center = np.zeros((self.max_objects), dtype=np.int64)
        obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

        for k in range(num_objects):
            cls_id, x, y, z, h, w, l, yaw = labels[k]
            cls_id = int(cls_id)
            # Invert yaw angle
            yaw = -yaw
            if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
                continue
            if (h <= 0) or (w <= 0) or (l <= 0):
                continue

            bbox_l = l / cnf.bound_size_x * hm_l
            bbox_w = w / cnf.bound_size_y * hm_w
            radius = self.compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
            radius = max(0, int(radius))

            center_x = (x - minX) / cnf.bound_size_x * hm_w  # x --> y (invert to 2D image space)
            center_y = (y - minY) / cnf.bound_size_y * hm_l  # y --> x
            center = np.array([center_x, center_y], dtype=np.float32)

            if hflipped:
                center[1] = hm_l - center[1] - 1

            center_int = center.astype(np.int32)
            if cls_id < 0:
                # ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
                # ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [cls_id]
                # # Consider to make mask ignore
                # for cls_ig in ignore_ids:
                #     self.gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
                # hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
                continue

            # Generate heatmaps for main center
            self.gen_hm_radius(hm_main_center[cls_id], center, radius)

            # Index of the center
            indices_center[k] = center_int[1] * hm_w + center_int[0]

            # targets for center offset
            cen_offset[k] = center - center_int

            # targets for dimension
            dimension[k, 0] = h
            dimension[k, 1] = w
            dimension[k, 2] = l

            # targets for direction
            direction[k, 0] = math.sin(float(yaw))  # im
            direction[k, 1] = math.cos(float(yaw))  # re
            # im -->> -im
            if hflipped:
                direction[k, 0] = - direction[k, 0]

            # targets for depth
            z_coor[k] = z

            # Generate object masks
            obj_mask[k] = 1


            '''if yaw < np.pi / 6. and yaw > -7 * np.pi / 6.:
                anglebin[k, 0] = 1
                angleoffset[k, 0] = yaw - (-0.5 * np.pi)
            if yaw > -7np.pi / 6. and yaw <  np.pi / 6.:
                anglebin[k, 1] = 1
                angleoffset[k, 1] = yaw - (0.5 * np.pi)'''


        '''targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'anglebin': anglebin,
            'angleoffset': angleoffset,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }'''

        targets = {
            'hm_cen': hm_main_center,
            'cen_offset': cen_offset,
            'direction': direction,
            'z_coor': z_coor,
            'dim': dimension,
            'indices_center': indices_center,
            'obj_mask': obj_mask,
        }

        '''img = np.zeros_like(targets['hm_cen'], np.uint8)

        for i in range(108):
            for j in range(108):
                for k in range(3):
                    if  targets['hm_cen'][k ,i,j] > 0:
                        print( targets['hm_cen'][k,i,j])
                img[:,i,j] = targets['hm_cen'][:,i,j]*100

        hetmap = img
        print(hetmap.shape)
        hetmap = hetmap.transpose(1,2,0)
        print(hetmap.shape)
        hetmap = cv2.resize(hetmap,(800,800))
        print(hetmap.shape)
        cv2.imshow('x',hetmap)

        cv2.waitKey(0)'''

        return targets
    ##########################targets#######################################

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.astyx_infos) * self.total_epochs

        return len(self.astyx_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.astyx_infos)

        info = copy.deepcopy(self.astyx_infos[index])

        ##################################################################
        # print(f'info annos:')
        # print(len(info['point_cloud']))
        # for key, value in info['point_cloud'].items():
        #     print(key)
        #################################################################
        sample_idx = info['point_cloud']['pc_idx']

        # points = self.get_pointcloud(sample_idx, self.pc_type)
        calib = info['calib']
        T_from_radar_to_lidar = calib['T_from_radar_to_lidar']
        lidar_points, radar_points = self.get_pointcloud(sample_idx, self.pc_type, T_from_radar_to_lidar)

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            # pts_rect = calib.lidar_to_rect(points[:, 0:3])
            # fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            # points = points[fov_flag]

            ### add ###
            lidar_pts_rect = calib.lidar_to_rect(lidar_points[:, 0:3])
            lidar_fov_flag = self.get_fov_flag(lidar_pts_rect, img_shape, calib)
            lidar_points = lidar_points[lidar_fov_flag]

            radar_pts_rect = calib.lidar_to_rect(radar_points[:, 0:3])
            radar_fov_flag = self.get_fov_flag(radar_pts_rect, img_shape, calib)
            radar_points = radar_points[radar_fov_flag]
            ### add ###

        input_dict = {
            # 'points': points,
            'lidar_points': lidar_points,
            'radar_points': radar_points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            # ##################################################################
            # print(f'info annos:')
            # print(len(annos))
            # for key, value in annos.items():
            #     print(key, type(value), value.shape)
            # ##################################################################
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            # loc, dims, rots = annos['location'], annos['dimensions'], annos['orientation']
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            gt_boxes = annos['gt_boxes']
            cen_labels = annos['cen_labels']

            ###
            hflipped = False
            targets = self.build_targets(cen_labels, hflipped)
            ###

            # input_dict.update({
            #     'gt_names': gt_names,
            #     'gt_boxes': gt_boxes
            # })
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes,
                'hm_cen': targets['hm_cen'],
                'cen_offset': targets['cen_offset'],
                'direction': targets['direction'],
                'z_coor': targets['z_coor'],
                'dim': targets['dim'],
                'indices_center': targets['indices_center'],
                'obj_mask': targets['obj_mask']
            })

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict


def create_astyx_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = AstyxDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('astyx_infos_%s.pkl' % train_split)
    val_filename = save_path / ('astyx_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'astyx_infos_trainval.pkl'
    test_filename = save_path / 'astyx_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    astyx_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(astyx_infos_train, f)
    print('Astyx info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    astyx_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(astyx_infos_val, f)
    print('Astyx info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(astyx_infos_train + astyx_infos_val, f)
    print('Astyx info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    astyx_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(astyx_infos_test, f)
    print('Astyx info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_astyx_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.full_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_astyx_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'astyx',
            save_path=ROOT_DIR / 'data' / 'astyx'
        )
