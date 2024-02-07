import torch
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import random
from einops import rearrange
from utils import train_utils, geo_utils
from process_data_normal import process_point_cloud, perform_pca_on_gradients, get_curvature



def preprocess_data(voxel_coords, expand_radius, test_sdf):
    '''
    Inputs:
        voxels_coords: voxel coordinates in shape [R^3, 3]
        expand_radius: the expand radius to calculate the boundary consistency loss
        test_sdf: a list with points with positive and negative sdf values
    '''
    pos_data, neg_data = test_sdf[0], test_sdf[1]
    sdf_all = torch.cat([pos_data, neg_data], dim=0) # [K,4]
    sdf_tree_all = cKDTree(sdf_all[:,:3])

    sdf_surface = sdf_all[np.abs(sdf_all[:,-1]) < 0.1]
    sdf_tree_surface = cKDTree(sdf_surface[:,:3])

    sdf_tree_pos = cKDTree(pos_data[:,:3])
    sdf_tree_neg = cKDTree(neg_data[:,:3])

    valid_voxels_idx, valid_voxels_coords, voxel_to_points_mapping = [], [], []
    valid_voxels_surface = []
    valid_transforms = []
    for cur_voxel_idx, cur_voxel_coordinate in enumerate(voxel_coords):
        near_sample_indices_surface = sdf_tree_surface.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
        
        # # store the sdf samples of valid voxel
        # if len(near_sample_indices_surface) > 0:
        #     near_sample_indices = sdf_tree_all.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
        #     valid_voxels_idx.append(cur_voxel_idx)
        #     valid_voxels_coords.append(cur_voxel_coordinate)
        #     voxel_to_points_mapping.append(near_sample_indices)

        # pos_samples_indices_surface = sdf_tree_pos.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
        # neg_samples_indices_surface = sdf_tree_neg.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
        # if len(pos_samples_indices_surface) > 0 and len(neg_samples_indices_surface) > 0:
        #     valid_voxels_surface.append(True)
        # else:
        #     valid_voxels_surface.append(False)

        pos_samples_indices_surface = sdf_tree_pos.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
        neg_samples_indices_surface = sdf_tree_neg.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
        if len(pos_samples_indices_surface) > 0 and len(neg_samples_indices_surface) > 0:
            valid_voxels_surface.append(True)
            near_sample_indices = sdf_tree_all.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
            valid_voxels_idx.append(cur_voxel_idx)
            valid_voxels_coords.append(cur_voxel_coordinate)
            voxel_to_points_mapping.append(near_sample_indices)
        else:
            valid_voxels_surface.append(False)

        # compute the coordinate field
        n1, m = 200, 20
        #if len(near_sample_indices_surface) > 0:
        if len(pos_samples_indices_surface) > 0 and len(neg_samples_indices_surface) > 0:
            if len(near_sample_indices_surface) > 10:
                near_sample_indices_all = sdf_tree_all.query_ball_point(x=cur_voxel_coordinate.numpy(), r=expand_radius, p=np.inf)
                point_retrived = sdf_all[near_sample_indices_all]   # [n1,4]
                subsampled_points, sdf_gradients = process_point_cloud(point_retrived[:,:3].numpy(), point_retrived[:,-1].numpy(), n1, m)
                normal_direction = sdf_gradients.mean(axis=0) / np.linalg.norm(sdf_gradients.mean(axis=0))  # [3]
                pca_components, pca_explained_variance = perform_pca_on_gradients(sdf_gradients, n=3)
                curve_direction = get_curvature(normal_direction, pca_components[1])    # [3]
                normal_direction /= np.linalg.norm(normal_direction)
                curve_direction /= np.linalg.norm(curve_direction)
                z = np.cross(normal_direction, curve_direction)
                # transform = np.concatenate([normal_direction.reshape((3,1)), curve_direction.reshape((3,1)), z.reshape((3,1))], axis=1)
                transform = np.concatenate([normal_direction.reshape((1,3)), curve_direction.reshape((1,3)), z.reshape((1,3))], axis=0)
                transform = torch.from_numpy(transform)
            else:
                transform = torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            valid_transforms.append(transform)

    return valid_voxels_idx, valid_voxels_coords, voxel_to_points_mapping, valid_voxels_surface, valid_transforms


class TestData_instance:
    def __init__(self, sdf_data, valid_voxels_idx, valid_voxels_coords, voxel_to_point_mapping, valid_voxels_surface, valid_transforms):
        self.sdf_data = sdf_data
        self.valid_voxels_idx = valid_voxels_idx
        self.valid_voxels_coords = valid_voxels_coords
        self.voxel_to_point_mapping = voxel_to_point_mapping
        self.valid_voxels_surface = valid_voxels_surface
        self.valid_transforms = valid_transforms    # computed coordinate field for valid voxels

        self.num_voxels_per_scene = 3000    # N
        self.num_points_per_voxel = 1000 #16      # M
    

    def random_sample(self, iter, total_iter):
        data = torch.cat([self.sdf_data[0], self.sdf_data[1]], dim=0)   # [K,4]

        # select a fixed number of valid voxels
        num_valid_voxels = len(self.valid_voxels_idx)
        # selected_indices = random.choices(np.arange(num_valid_voxels), k=self.num_voxels_per_scene)
        selected_indices = np.arange(num_valid_voxels)
        
        # if iter > (total_iter // 2):
        #     selected_indices = selected_indices[self.valid_voxels_surface]

        selected_voxel_index = [self.valid_voxels_idx[it] for it in selected_indices]                   # [N]
        selected_voxel_to_point_mapping = [self.voxel_to_point_mapping[it] for it in selected_indices]  # [N,?]
        selected_voxel_coords = torch.stack([self.valid_voxels_coords[it] for it in selected_indices])  # [N,3]
        selected_voxel_transform = torch.stack([self.valid_transforms[it] for it in selected_indices])  # [N,3,3]

        # for each valid voxel, sample a fixed number of points
        selected_voxel_points, all_mask = [], []
        for it in selected_voxel_to_point_mapping:
            #num_points_per_voxel = min(self.num_points_per_voxel, len(it))
            selected_points_idx = random.choices(it, k=self.num_points_per_voxel)
            mask = torch.tensor(find_first_occurrences(selected_points_idx)).float()
            selected_points_idx = torch.tensor(selected_points_idx).long()  # [M]
            all_mask.append(mask)
            selected_voxel_points.append(data[selected_points_idx])         # list [M,4]
        
        all_mask = torch.stack(all_mask)                            # [N,M]
        selected_voxel_points = torch.stack(selected_voxel_points)  # [N,M,4]

        res = {
            'voxel_location': selected_voxel_coords,  # [N,3]
            'voxel_local_index': torch.tensor(selected_indices).long(), # [N]
            'sdf': selected_voxel_points, # [N,M,4]
            'mask': all_mask, # [N,M]
            'transform': selected_voxel_transform,  # [N,3,3]
        }
        return self.batchify_sample(res)
    

    def batchify_sample(self, sample):
        voxel_location = sample['voxel_location']           # [N,3]
        voxel_local_index = sample['voxel_local_index']     # [N]
        sdf = sample['sdf']                                 # [N,M,4]
        mask = sample['mask']                               # [N,M]
        transform = sample['transform']                     # [N,3,3]

        N, M, _ = sdf.shape
        voxel_location = voxel_location.unsqueeze(1).repeat(1,M,1)      # [N,M,3]
        voxel_local_index = voxel_local_index.unsqueeze(1).repeat(1,M)  # [N,M]
        transform = transform.unsqueeze(1).repeat(1,M,1,1)              # [N,M,3,3]

        voxel_location = rearrange(voxel_location, 'n m c -> (n m) c')
        voxel_local_index = rearrange(voxel_local_index, 'n m -> (n m)')
        sdf = rearrange(sdf, 'n m c -> (n m) c')
        mask = rearrange(mask, 'n m -> (n m)').unsqueeze(-1)
        transform = rearrange(transform, 'n m a b -> (n m) a b')

        sample_new = {
            'voxel_location': voxel_location,
            'voxel_local_index': voxel_local_index,
            'sdf': sdf,
            'mask': mask,
            'transform': transform
        }
        return sample_new



def transform_to_local(config, points, voxel_location, codes_frame, transform):
    coord_field_mode = train_utils.get_spec_with_default(config, 'CoordinateFieldMode', 'none')
    if coord_field_mode == 'none':
        # deepls default coordinate field definition
        # local coordinate is defined at the center of voxels and has axis-aligned rotation
        points_local = points - voxel_location
    elif coord_field_mode == 'rotation_only' or coord_field_mode == 'computed_rotation_only':
        # local coordinate is defined at the center of voxels and has grid-specific rotation
        assert codes_frame.shape[-1] == 4   # rotation is represented by the 4-dim quaternion
        points_local = points - voxel_location           # [B,3]
        coord_rot_mat = geo_utils.quat2mat_transform(codes_frame)  # quaternion [B,4] -> rotation matrix [B,3,3]
        # points_local = torch.einsum('ijk,ik->ik', coord_rot_mat, points_local)   # [B,3]
        points_local = torch.einsum('bij,bj->bi', coord_rot_mat, points_local)   # [B,3]
    elif coord_field_mode == 'computed':
        points_local = points - voxel_location    # [B,3]
        coord_rot_mat = transform     # [B,3,3] 
        # points_local = torch.einsum('ijk,ik->ik', coord_rot_mat, points_local)   # [B,3]
        points_local = torch.einsum('bij,bj->bi', coord_rot_mat, points_local)   # [B,3]
    elif coord_field_mode == 'rotation_location' or coord_field_mode == 'computed_rotation_location':
        # local coordinate is defined at the center of voxels and has grid-specific rotation
        assert codes_frame.shape[-1] == 7
        points_local = points - voxel_location            # [B,3]
        origin = 2 * (torch.sigmoid(codes_frame[:,4:]) - 0.5)
        origin *= (train_utils.get_spec_with_default(config, "VolumeSizeHalf", 1.0) / 
                   train_utils.get_spec_with_default(config, "VoxelResolution", 1.0))
        points_local -= origin
        coord_rot_mat = geo_utils.quat2mat_transform(codes_frame[:,:4])  # quaternion [B,4] -> rotation matrix [B,3,3]
        #breakpoint()
        # points_local = torch.einsum('ijk,ik->ik', coord_rot_mat, points_local)   # [B,3]
        points_local = torch.einsum('bij,bj->bi', coord_rot_mat, points_local)   # [B,3]
    else:
        raise NotImplementedError

    return points_local
        

def find_first_occurrences(lst):
    seen = set()
    output = []
    for element in lst:
        if element not in seen:
            output.append(1)
            seen.add(element)
        else:
            output.append(0)
    return output