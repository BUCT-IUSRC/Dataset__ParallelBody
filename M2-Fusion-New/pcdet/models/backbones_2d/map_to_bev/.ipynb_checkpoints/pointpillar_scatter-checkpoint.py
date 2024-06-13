import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        self.adap_max_pool = nn.AdaptiveMaxPool2d(480)

    def forward(self, batch_dict, **kwargs):
        # control_bit = batch_dict['control_bit']
        batch_size = batch_dict['batch_size']
        ################################## radar CenterPillar的特征进行scatter ####################################
        radar_cen_pillar_features, radar_cen_coords = batch_dict['radar_cen_pillar_features'], batch_dict['radar_cen_voxel_coords']
        radar_cen_batch_spatial_features = []
        radar_cen_batch_size = radar_cen_coords[:, 0].max().int().item() + 1
        for radar_cen_batch_idx in range(radar_cen_batch_size):
            radar_cen_spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * 960 * 960,
                dtype=radar_cen_pillar_features.dtype,
                device=radar_cen_pillar_features.device)

            radar_cen_batch_mask = radar_cen_coords[:, 0] == radar_cen_batch_idx
            radar_cen_this_coords = radar_cen_coords[radar_cen_batch_mask, :]
            radar_cen_indices = radar_cen_this_coords[:, 1] + radar_cen_this_coords[:, 2] * 960 + radar_cen_this_coords[:, 3]
            radar_cen_indices = radar_cen_indices.type(torch.long)
            radar_cen_pillars = radar_cen_pillar_features[radar_cen_batch_mask, :]
            radar_cen_pillars = radar_cen_pillars.t()
            radar_cen_spatial_feature[:, radar_cen_indices] = radar_cen_pillars
            radar_cen_batch_spatial_features.append(radar_cen_spatial_feature)

        radar_cen_batch_spatial_features = torch.stack(radar_cen_batch_spatial_features, 0)
        radar_cen_batch_spatial_features = radar_cen_batch_spatial_features.view(radar_cen_batch_size, self.num_bev_features * self.nz, 960, 960)
        
        # for batch in range(batch_size):
        #     if control_bit[batch] == False:
        #         radar_cen_batch_spatial_features[batch] = torch.zeros_like(radar_cen_batch_spatial_features[0])

        radar_cen_batch_spatial_features = self.adap_max_pool(radar_cen_batch_spatial_features)
        ################################## radar CenterPillar的特征进行scatter ####################################

        ################################## lidar CenterPillar的特征进行scatter ####################################
        lidar_cen_pillar_features, lidar_cen_coords = batch_dict['lidar_cen_pillar_features'], batch_dict['lidar_cen_voxel_coords']
        lidar_cen_batch_spatial_features = []
        lidar_cen_batch_size = lidar_cen_coords[:, 0].max().int().item() + 1
        for lidar_cen_batch_idx in range(lidar_cen_batch_size):
            lidar_cen_spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * 960 * 960,
                dtype=lidar_cen_pillar_features.dtype,
                device=lidar_cen_pillar_features.device)

            lidar_cen_batch_mask = lidar_cen_coords[:, 0] == lidar_cen_batch_idx
            lidar_cen_this_coords = lidar_cen_coords[lidar_cen_batch_mask, :]
            lidar_cen_indices = lidar_cen_this_coords[:, 1] + lidar_cen_this_coords[:, 2] * 960 + lidar_cen_this_coords[:, 3]
            lidar_cen_indices = lidar_cen_indices.type(torch.long)
            lidar_cen_pillars = lidar_cen_pillar_features[lidar_cen_batch_mask, :]
            lidar_cen_pillars = lidar_cen_pillars.t()
            lidar_cen_spatial_feature[:, lidar_cen_indices] = lidar_cen_pillars
            lidar_cen_batch_spatial_features.append(lidar_cen_spatial_feature)

        lidar_cen_batch_spatial_features = torch.stack(lidar_cen_batch_spatial_features, 0)
        lidar_cen_batch_spatial_features = lidar_cen_batch_spatial_features.view(lidar_cen_batch_size, self.num_bev_features * self.nz, 960, 960)
                
        # for batch in range(batch_size):
        #     if control_bit[batch] == False:
        #         lidar_cen_batch_spatial_features[batch] = torch.zeros_like(lidar_cen_batch_spatial_features[0])

        lidar_cen_batch_spatial_features = self.adap_max_pool(lidar_cen_batch_spatial_features)
        ################################## lidar CenterPillar的特征进行scatter ####################################

        ################################## lidar和radar的特征进行scatter ####################################
        lidar_pillar_features, lidar_coords = batch_dict['lidar_pillar_features'], batch_dict['lidar_voxel_coords']
        radar_pillar_features, radar_coords = batch_dict['radar_pillar_features'], batch_dict['radar_voxel_coords']
        lidar_batch_spatial_features = []
        radar_batch_spatial_features = []
        lidar_batch_size = lidar_coords[:, 0].max().int().item() + 1
        radar_batch_size = radar_coords[:, 0].max().int().item() + 1
        for lidar_batch_idx in range(lidar_batch_size):
            lidar_spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=lidar_pillar_features.dtype,
                device=lidar_pillar_features.device)

            lidar_batch_mask = lidar_coords[:, 0] == lidar_batch_idx
            lidar_this_coords = lidar_coords[lidar_batch_mask, :]
            lidar_indices = lidar_this_coords[:, 1] + lidar_this_coords[:, 2] * self.nx + lidar_this_coords[:, 3]
            lidar_indices = lidar_indices.type(torch.long)
            lidar_pillars = lidar_pillar_features[lidar_batch_mask, :]
            lidar_pillars = lidar_pillars.t()
            lidar_spatial_feature[:, lidar_indices] = lidar_pillars
            lidar_batch_spatial_features.append(lidar_spatial_feature)
        for radar_batch_idx in range(radar_batch_size):
            radar_spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=radar_pillar_features.dtype,
                device=radar_pillar_features.device)

            radar_batch_mask = radar_coords[:, 0] == radar_batch_idx
            radar_this_coords = radar_coords[radar_batch_mask, :]
            radar_indices = radar_this_coords[:, 1] + radar_this_coords[:, 2] * self.nx + radar_this_coords[:, 3]
            radar_indices = radar_indices.type(torch.long)
            radar_pillars = radar_pillar_features[radar_batch_mask, :]
            radar_pillars = radar_pillars.t()
            radar_spatial_feature[:, radar_indices] = radar_pillars
            radar_batch_spatial_features.append(radar_spatial_feature)

        lidar_batch_spatial_features = torch.stack(lidar_batch_spatial_features, 0)
        radar_batch_spatial_features = torch.stack(radar_batch_spatial_features, 0)
        lidar_batch_spatial_features = lidar_batch_spatial_features.view(lidar_batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        radar_batch_spatial_features = radar_batch_spatial_features.view(radar_batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['lidar_spatial_features'] = lidar_batch_spatial_features
        batch_dict['radar_spatial_features'] = radar_batch_spatial_features
        ################################## lidar和radar的特征进行scatter ####################################

        ########################################### cat操作 ##############################################
        lidar_cat_batch_spatial_features = torch.cat((lidar_batch_spatial_features, lidar_cen_batch_spatial_features), 1)
        radar_cat_batch_spatial_features = torch.cat((radar_batch_spatial_features, radar_cen_batch_spatial_features), 1)
        cat_batch_spatial_features = torch.cat((lidar_cat_batch_spatial_features, radar_cat_batch_spatial_features), 1)
        batch_dict['cat_spatial_features'] = cat_batch_spatial_features
        ########################################### cat操作 ##############################################
        return batch_dict
