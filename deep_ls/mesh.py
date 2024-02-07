#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.

from json import decoder
import logging
from random import sample
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_ls.utils
from deep_ls.data_voxel import generate_grid_center_indices
from utils import vis_utils, train_utils, recon_utils

from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from tqdm import tqdm


def create_mesh(config, decoder, latent_vec, latent_vec_valid, latent_vec_valid_idx, transforms_valid,
                cube_size, box_size, filename, N=256, max_batch=32 ** 3, offset=None, scale=None):
    '''
    Inputs:
        box_size: is actually half box size
    '''
    # get args
    coord_field_mode = train_utils.get_spec_with_default(config, 'CoordinateFieldMode', 'none')
    code_length = config["CodeLength"]
    if coord_field_mode == 'rotation_only':
        code_additional = 4    # 4-d rotation quaternion
    elif coord_field_mode == 'none':
        code_additional = 0
    elif coord_field_mode == 'computed':
        code_additional = 0
    elif coord_field_mode == 'computed_rotation_only':
        code_additional = 4
    elif coord_field_mode == 'rotation_location':
        code_additional = 7
    elif coord_field_mode == 'computed_rotation_location':
        code_additional = 7
    else:
        raise NotImplementedError

    start = time.time()
    ply_filename = filename

    decoder.eval()

    # create testing grids (with higher resolution, e.g. 128 or 256)
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.ones(N ** 3, 4)
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples.requires_grad = False

    #grid_radius = (box_size * 2) / cube_size
    grid_radius = box_size / cube_size

    samples_counter = np.zeros((samples.shape[0], 1), dtype=np.int_)
    tree_samples = samples[:,:3].cpu().numpy()
    samples_tree = cKDTree(tree_samples)

    #sdf_grid_indices = deep_ls.data.generate_grid_center_indices(cube_size=cube_size, box_size=box_size)
    sdf_grid_indices = deep_ls.data_voxel.generate_grid_center_indices(cube_size, box_size).reshape(-1,3)
    sdf_grid_indices_valid = sdf_grid_indices[latent_vec_valid_idx]

    logging.info('working on generating mesh')
    for center_point_index in tqdm(range(len(sdf_grid_indices_valid))):
        near_sample_indices = samples_tree.query_ball_point(x=[sdf_grid_indices_valid[center_point_index].numpy()], r=grid_radius, p=np.inf)
        num_sdf_samples = len(near_sample_indices[0])
        if num_sdf_samples < 1:
            continue
        near_sample_indices = near_sample_indices[0]

        # get code
        code = latent_vec_valid[center_point_index].cuda()
        code_frame = code[code_length:]    # [C']
        code = code[:code_length]          # [C]
        transform = transforms_valid[center_point_index].cuda()

        transformed_sample = recon_utils.transform_to_local(config,
                                                            points=samples[near_sample_indices, 0:3].float().cuda(),
                                                            voxel_location=sdf_grid_indices_valid[center_point_index].float().cuda(),
                                                            codes_frame=code_frame.unsqueeze(0).repeat(num_sdf_samples,1).float().cuda(),
                                                            transform=transform.unsqueeze(0).repeat(num_sdf_samples,1,1).float().cuda())

        #transformed_sample = samples[near_sample_indices, 0:3] - sdf_grid_indices_valid[center_point_index] 
        code = code.unsqueeze(0)
        code = code.repeat(transformed_sample.shape[0], 1)  # [N,C]
        decoder_input = torch.cat([code, transformed_sample.cuda()], dim=1).float().cuda()
        samples[near_sample_indices, 3] = decoder(decoder_input).squeeze(1).detach().cpu()
        samples_counter[near_sample_indices, 0] += 1

    sdf_values = samples[:, 3]
    sdf_values = torch.atanh(sdf_values)
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )



def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    vis_utils.vis_recon(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
