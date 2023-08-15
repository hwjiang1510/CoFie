import functools
import json
import logging
import math
import os
import signal
import sys
import time
import warnings
import deep_ls
import deep_ls.workspace as ws
import torch
import numpy as np
from deep_ls.data import remove_nans
from tqdm import tqdm
from scipy.spatial import cKDTree


def generate_grid_center_indices(voxel_resolution=32, volume_size_half=1.2):
    '''
    return voxel coordinate in shape [R^3,3], R is cube_size
    '''
    # Divide space into equally spaced subspaces and calculate center position of subspace
    voxel_centers = np.linspace(-volume_size_half, volume_size_half, voxel_resolution, endpoint=False)
    voxel_centers += volume_size_half / voxel_resolution
    # Create grid indices
    all_voxel_centers = np.vstack(np.meshgrid(voxel_centers, voxel_centers, voxel_centers)).reshape(3, -1).T
    all_voxel_centers = torch.tensor(all_voxel_centers).reshape(voxel_resolution, voxel_resolution, voxel_resolution, 3).permute(1,0,2,3)
    return all_voxel_centers


def get_voxel_indices(points, voxel_resolution, volume_size_half):
    # Normalize points to [0, 1]
    normalized_points = (points[:, :3] + volume_size_half) / (2 * volume_size_half)
    # Convert to voxel indices
    voxel_indices = (normalized_points * voxel_resolution).floor().long()
    return voxel_indices


def calculate_occupancy(instance_name, category, voxel_resolution, volume_size_half, voxel_coordinate, expand_rate=1.5):
    root = os.path.join('./data/SdfSamples', 'ShapeNetV2', category)
    instance_path = os.path.join(root, instance_name + '.npz')

    save_root = root.replace('SdfSamples', 'SdfSamplesToVoxelIndices')
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, '{}_vol-{}_res-{}_expand-{}.json'.format(instance_name,
                                                                                volume_size_half,
                                                                                voxel_resolution,
                                                                                expand_rate))
    
    # get all sampled sdf for the instance
    npz = np.load(instance_path)
    sdf_pos = remove_nans(torch.from_numpy(npz["pos"]))
    sdf_neg = remove_nans(torch.from_numpy(npz["neg"]))
    sdf_all = torch.cat([sdf_pos, sdf_neg], dim=0) # [K,4]

    # get sampled points near the surface
    sdf_surface = sdf_all[np.abs(sdf_all[:,-1]) < 0.1]   # voxel size is 2.4/32 = 0.075
    #print(sdf_surface[:,:3].max(), sdf_surface[:,:3].min())

    # get the corresponding voxel indices of the surface points
    voxel_indices = get_voxel_indices(sdf_surface[:,:3], voxel_resolution, volume_size_half)
    voxel_indices_unique = torch.unique(voxel_indices, dim=0)   # [n,3]
    #print(sdf_surface[:5], voxel_indices[:5])

    # get the coordinate of surface voxels, in shape [n,3]
    voxel_coordinates_unique = voxel_coordinate[voxel_indices_unique[:,0], voxel_indices_unique[:,1], voxel_indices_unique[:,2], :]

    # build the KD three and find point indices within each voxel
    sdf_tree = cKDTree(sdf_all[:,:3])
    voxel_size = (volume_size_half * 2) / voxel_resolution
    sdf_grid_radius = expand_rate * voxel_size
    results = {}
    for cur_voxel_idx, cur_voxel_coordinate in zip(voxel_indices_unique, voxel_coordinates_unique):
        near_sample_indices = sdf_tree.query_ball_point(x=cur_voxel_coordinate.numpy(), r=sdf_grid_radius, p=np.inf)
        key_name = str(cur_voxel_idx.numpy().tolist()).replace('[', '').replace(']','')
        results[key_name] = near_sample_indices
    with open(save_path, 'w') as f:
        json.dump(results, f)
    
    return results


def process_instances():
    split = './examples/splits/sv2_tables_train.json'
    category = "04379243"
    with open(split, "r") as f:
        info = json.load(f)
    all_instances = info["ShapeNetV2"][category]
    print(len(all_instances))

    voxel_resolution = 32
    volume_size_half = 1.2
    expand_rate = 1.5 # for calculating shape boundary consistency loss
    voxel_coordinate = generate_grid_center_indices(voxel_resolution, volume_size_half) # [R^3, 3]

    for instance_idx, instance in tqdm(enumerate(all_instances)):
        cur_result = calculate_occupancy(instance,
                                        category,
                                        voxel_resolution,
                                        volume_size_half, 
                                        voxel_coordinate, 
                                        expand_rate)
        num_valid_voxels = len(cur_result.keys())


if __name__ == '__main__':
    process_instances()