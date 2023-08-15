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
from deep_ls.data_voxel import generate_grid_center_indices
from tqdm import tqdm
from scipy.spatial import cKDTree


def calculate_occupancy(category_instance, voxel_resolution, volume_size_half, voxel_coordinate, expand_rate=1.5):
    print(category_instance)
    category, instance_name = category_instance
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
    sdf_surface = sdf_all[np.abs(sdf_all[:,-1]) < 0.075]   # voxel size is 2.4/32 = 0.075
    sdf_tree_all = cKDTree(sdf_all[:,:3])
    sdf_tree_surface = cKDTree(sdf_surface[:,:3])
    
    # get query radius
    voxel_size = (volume_size_half * 2) / voxel_resolution
    sdf_grid_radius = expand_rate * voxel_size

    # find neighbouring points
    results = {}
    for cur_voxel_idx, cur_voxel_coordinate in enumerate(voxel_coordinate):
        near_sample_indices_surface = sdf_tree_surface.query_ball_point(x=cur_voxel_coordinate.numpy(), r=sdf_grid_radius, p=np.inf)
        if len(near_sample_indices_surface) > 0:
            near_sample_indices_all = sdf_tree_all.query_ball_point(x=cur_voxel_coordinate.numpy(), r=sdf_grid_radius, p=np.inf)
            results[cur_voxel_idx] = near_sample_indices_all
    with open(save_path, 'w') as f:
        json.dump(results, f)

    num_valid_voxel = len(list(results.keys()))
    print('{} voxels valid ({}%)'.format(num_valid_voxel, 100 * num_valid_voxel / (32 * 32 * 32)))
    
    return results


def process_instances():
    split_file = 'examples/splits/sv2_all_train.json'
    with open(split_file, 'r') as f:
        split = json.load(f)
    
    all_category_instance = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                cur_category_instance = [class_name, instance_name]
                all_category_instance.append(cur_category_instance)

    print(len(all_category_instance))

    voxel_resolution = 32
    volume_size_half = 1.2
    expand_rate = 1.5 # for calculating shape boundary consistency loss
    voxel_coordinate = generate_grid_center_indices(voxel_resolution, volume_size_half).reshape(-1,3) # [R^3, 3]

    for instance_idx, category_instance in tqdm(enumerate(all_category_instance)):
        cur_result = calculate_occupancy(category_instance,
                                        voxel_resolution,
                                        volume_size_half, 
                                        voxel_coordinate, 
                                        expand_rate)
        num_valid_voxels = len(cur_result.keys())


if __name__ == '__main__':
    process_instances()