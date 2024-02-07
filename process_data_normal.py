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
from sklearn.decomposition import PCA

def process_point_cloud(points, sdf_values, n1, m):
    """
    Process a point cloud by subsampling and computing derivatives of SDF values.

    :param points: Array of shape [N, 3] representing the point cloud.
    :param sdf_values: Array of length N representing the SDF values of each point.
    :param n1: Number of points to subsample.
    :param m: Number of nearest neighbors to consider.
    :return: Array of subsampled points, their nearest neighbors, and SDF derivatives.
    """

    # Ensure n1 and m are not larger than the number of points
    n1 = min(n1, len(points))
    m = min(m, len(points) - 1)

    # Subsample n1 points
    np.random.seed(0)  # For reproducibility
    subsample_indices = np.random.choice(len(points), n1, replace=False)
    subsampled_points = points[subsample_indices]

    # Build a KD-Tree for efficient nearest neighbor search
    kd_tree = cKDTree(points)

    # Initialize array to store derivatives
    sdf_derivatives = np.zeros(n1)
    
    sdf_gradients = np.zeros((n1, 3))

    # For each subsampled point
    for i, point in enumerate(subsampled_points):
        # Find m nearest neighbors
        _, nn_indices = kd_tree.query(point, k=m+1)  # +1 because the point itself is included
        nn_indices = nn_indices[1:]  # Exclude the point itself

        # Compute spatial and SDF differences
        spatial_diff = points[nn_indices] - point.reshape((1,3)) # [m,3]
        sdf_diff = sdf_values[nn_indices] - sdf_values[subsample_indices[i]] # in shape [m]

        # Compute the weighted sum of spatial differences
        # Weighted by the SDF differences
        weights = sdf_diff[:, np.newaxis] # [m,1]
        gradient = np.sum(weights * spatial_diff, axis=0) / m
        
        norm = np.linalg.norm(gradient)
        if norm != 0:
            sdf_gradients[i] = gradient / norm
        else:
            sdf_gradients[i] = np.array([0.0, 0.0, 1.0]) 

    return subsampled_points, sdf_gradients


def perform_pca_on_gradients(gradients, n=3):
    """
    Perform PCA on the gradient vectors.

    :param gradients: Array of shape [n1, 3] representing the gradient vectors.
    :return: PCA components and explained variance.
    """
    pca = PCA(n_components=n)
    pca.fit(gradients)

    # The principal components and the explained variance
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_

    return components, explained_variance


def get_curvature(normal, vec2):
    normal_to_plane = np.cross(normal, vec2)
    return np.cross(normal_to_plane, normal)


def calculate_occupancy(category_instance, voxel_resolution, volume_size_half, voxel_coordinate, expand_rate=1.5):
    print(category_instance)
    category, instance_name = category_instance
    root = os.path.join('./data/SdfSamples', 'ShapeNetV2', category)
    instance_path = os.path.join(root, instance_name + '.npz')

    save_root = root.replace('SdfSamples', 'SdfSamplesToVoxelIndices_normal')
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
    sdf_surface = sdf_all[np.abs(sdf_all[:,-1]) < 0.1]   # voxel size is 2.0/32 = 0.0625
    #sdf_surface = sdf_all.clone()
    sdf_tree_all = cKDTree(sdf_all[:,:3])
    sdf_tree_surface = cKDTree(sdf_surface[:,:3])
    
    # get query radius
    voxel_size = (volume_size_half * 2) / voxel_resolution
    # sdf_grid_radius = expand_rate * voxel_size / 2.0    # radius is half of voxel size
    sdf_grid_radius = voxel_size / 2.0    # no exand rate for calculating normal

    n1 = 200
    m = 20

    # find neighbouring points
    results = {}
    for cur_voxel_idx, cur_voxel_coordinate in enumerate(voxel_coordinate):
        near_sample_indices_surface = sdf_tree_surface.query_ball_point(x=cur_voxel_coordinate.numpy(), r=sdf_grid_radius, p=np.inf)
        if len(near_sample_indices_surface) > 10:
            near_sample_indices_all = sdf_tree_all.query_ball_point(x=cur_voxel_coordinate.numpy(), r=sdf_grid_radius, p=np.inf)
            point_retrived = sdf_all[near_sample_indices_all]   # [n1,4]
            subsampled_points, sdf_gradients = process_point_cloud(point_retrived[:,:3].numpy(), point_retrived[:,-1].numpy(), n1, m)
            normal_direction = sdf_gradients.mean(axis=0) / np.linalg.norm(sdf_gradients.mean(axis=0))  # [3]
            pca_components, pca_explained_variance = perform_pca_on_gradients(sdf_gradients, n=3)
            curve_direction = get_curvature(normal_direction, pca_components[1])    # [3]
            results[cur_voxel_idx] = np.concatenate((normal_direction, curve_direction)).tolist()
        elif len(near_sample_indices_surface) > 0:
            results[cur_voxel_idx] = np.array((0.0, 0, 1, 0, 1, 0)).tolist()
    with open(save_path, 'w') as f:
        json.dump(results, f)

    num_valid_voxel = len(list(results.keys()))
    #print('{} voxels valid ({}%)'.format(num_valid_voxel, 100 * num_valid_voxel / (32 * 32 * 32)))
    
    return results


def process_instances():
    split_file = 'examples/splits/sv2_all_test.json'
    with open(split_file, 'r') as f:
        split = json.load(f)
    
    all_category_instance = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                cur_category_instance = [class_name, instance_name]
                all_category_instance.append(cur_category_instance)

    print(len(all_category_instance))

    voxel_resolution = 32 #32
    volume_size_half = 1.0 # 1.2
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