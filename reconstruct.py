#!/usr/bin/env python3
# Based on: https://github.com/facebookresearch/DeepSDF using MIT LICENSE (https://github.com/facebookresearch/DeepSDF/blob/master/LICENSE)
# Copyright 2021-present Philipp Friedrich, Josef Kamysek. All Rights Reserved.
# python reconstruct.py -e examples/all -c 15 --split examples/splits/sv2_all_test_small.json -d data --skip

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import deep_ls
import deep_ls.workspace as ws

from train_deep_ls import get_spec_with_default

from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import tqdm

from utils import recon_utils, train_utils, geo_utils
from deep_ls.data import remove_nans

def reconstruct(
    config,
    decoder,
    num_iterations,
    latent_size,
    voxel_resolution, #cube_size,
    volume_size_half, #box_size,
    expand_radius, #voxel_radius,
    code_additional,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    decreased_by = 3
    adjust_lr_every = int(num_iterations / 4)

    # voxel size setting
    voxel_coords = deep_ls.data_voxel.generate_grid_center_indices(voxel_resolution, volume_size_half).reshape(-1,3)
    #expand_radius = expand_radius * ((box_size * 2) / cube_size)
    expand_radius = expand_radius * (box_size / cube_size)

    valid_voxels_idx, valid_voxels_coords, voxel_to_point_mapping, valid_voxels_surface, valid_transforms = \
        recon_utils.preprocess_data(voxel_coords, expand_radius, test_sdf)
    num_voxels_valid = len(valid_voxels_idx)
    logging.info('Valid voxel number {} ({}%)'.format(num_voxels_valid, 100 * num_voxels_valid / len(voxel_coords)))
    # num_voxels_surface = torch.tensor(valid_voxels_surface).sum().int().item()
    
    # sdf_grid_indices = deep_ls.data.generate_grid_center_indices(cube_size=cube_size, box_size=box_size)
    # sdf_grid_indices = deep_ls.data_voxel.generate_grid_center_indices(cube_size, box_size)
    # sdf_grid_indices = sdf_grid_indices.reshape(-1,3)       # voxel locations
    # sdf_grid_radius = voxel_radius * ((box_size * 2) / cube_size)
    # find valid voxels that contains at least one point
    # sdf_grid_indices_valid, sdf_grid_idx_valid = [], []
    # pos_data, neg_data = test_sdf[0], test_sdf[1]
    # all_data = torch.cat([torch.tensor(pos_data), torch.tensor(neg_data)], dim=0)   # [N,4]
    # all_data_surface = all_data[torch.abs(all_data[:,-1]) < 0.1]
    # print(all_data.shape, all_data_surface.shape)
    # sdf_tree = cKDTree(all_data_surface[:,:3].numpy())
    # for cur_voxel_idx, cur_voxel_coordinate in enumerate(sdf_grid_indices):
    #     near_sample_indices = sdf_tree.query_ball_point(x=cur_voxel_coordinate.numpy(), r=sdf_grid_radius, p=np.inf)
    #     if len(near_sample_indices) > 0:
    #         sdf_grid_indices_valid.append(cur_voxel_coordinate)
    #         sdf_grid_idx_valid.append(cur_voxel_idx)
    # sdf_grid_indices_valid = torch.stack(sdf_grid_indices_valid)
    # sdf_grid_idx_valid = torch.tensor(sdf_grid_idx_valid).long()
    # num_voxels_valid = sdf_grid_indices_valid.shape[0]
    # logging.info('Valid voxel number {} ({}%)'.format(num_voxels_valid, 100 * num_voxels_valid / len(sdf_grid_indices)))

    # build latent code
    if type(stat) == type(0.1):
        latent = torch.ones(num_voxels_valid, latent_size + code_additional).normal_(mean=0, std=1.0/latent_size).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()
        raise NotImplementedError # TODO
    
    # initialize coordinate field related code
    coord_field_mode = train_utils.get_spec_with_default(config, 'CoordinateFieldMode', 'none')
    if coord_field_mode == 'rotation_only':
        rot_init = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).view(1,3,3)
        quat_init = geo_utils.mat2quat_transform(rot_init)   # [1,4]
        latent[:, latent_size:] = quat_init.repeat(num_voxels_valid, 1)
    elif coord_field_mode == 'none':
        pass
    elif coord_field_mode == 'computed':
        pass
    elif coord_field_mode == 'computed_rotation_only':
        #breakpoint()
        rot_init = torch.stack(valid_transforms).view(-1,3,3)
        assert rot_init.shape[0] == num_voxels_valid
        quat_init = geo_utils.mat2quat_transform(rot_init)
        latent[:, latent_size:] = quat_init
    elif coord_field_mode == 'rotation_location':
        rot_init = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).view(1,3,3)
        quat_init = geo_utils.mat2quat_transform(rot_init)   # [1,4]
        latent[:, latent_size:latent_size+4] = quat_init.repeat(num_voxels_valid, 1)
        latent[:, latent_size+4:] = torch.tensor([[0.0, 0, 0]]).repeat(num_voxels_valid,1)
    elif coord_field_mode == 'computed_rotation_location':
        rot_init = torch.stack(valid_transforms).view(-1,3,3)
        assert rot_init.shape[0] == num_voxels_valid
        quat_init = geo_utils.mat2quat_transform(rot_init)
        latent[:, latent_size:latent_size+4] = quat_init
        latent[:, latent_size+4:] = torch.tensor([[0.0, 0, 0]]).repeat(num_voxels_valid,1)
    else:
        raise NotImplementedError
    
    use_computed_coordinate = True if coord_field_mode == 'computed' else False

    # initialize optimizer
    latent.requires_grad = True
    latent_all = torch.ones(len(voxel_coords), latent_size + code_additional).normal_(mean=0, std=1.0/latent_size).cuda()
    # optimizer = torch.optim.Adam([latent], lr=lr, betas=(0.9, 0.999))
    optimizer = torch.optim.Adam(
        [{"params": decoder.parameters(), "lr": 0.0001,},
         {"params": latent, "lr": lr,},], betas=(0.9, 0.999))
    loss_num = 0
    loss_l1 = torch.nn.L1Loss(reduction='none')

    # start optimization epochs
    for e in tqdm.tqdm(range(num_iterations)):
        decoder.eval()
        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)
        optimizer.zero_grad()
        loss = 0.0

        # sdf_data = deep_ls.data.unpack_sdf_samples_from_ram(test_sdf, num_samples)
        # sdf_tree = cKDTree(sdf_data[:,:3])
        sdf_all = torch.cat([test_sdf[0], test_sdf[1]], dim=0) # [K,4]

        # for each voxel
        for i, (voxel_idx, voxel_location, points_idx) in enumerate(zip(valid_voxels_idx, valid_voxels_coords, voxel_to_point_mapping)):
            # near_sample_indices = sdf_tree.query_ball_point([voxel_coords[voxel_idx].numpy()], r=expand_radius, p=np.inf)
            # num_sdf_samples = len(near_sample_indices[0])
            # if num_sdf_samples < 1: 
            #     continue
            num_points = len(points_idx)
            if num_points == 0:
                continue

            codes = latent[i].cuda().unsqueeze(0)    # [1,N+4]
            codes = codes.repeat(num_points, 1)
            codes_frame = codes[:, latent_size:]    # [N,C']
            codes = codes[:, :latent_size]          # [N,C]

            points = sdf_all[points_idx, :3].float().cuda()  # [N,3]
            voxel_location_repeat = voxel_location.unsqueeze(0).repeat(num_points,1).float().cuda()
            transform = torch.tensor([[1.0,0,0], [0,1,0], [0,0,1]]).unsqueeze(0).repeat(num_points,1,1).float().cuda()    # [N,3,3]

            # transform points to local
            points_local = recon_utils.transform_to_local(config, points, voxel_location_repeat, codes_frame, transform)

            input = torch.cat([codes, points_local], dim=1)
            preds = decoder(input)

            sdf_gt = sdf_all[points_idx, 3].unsqueeze(1).cuda()
            sdf_gt = torch.tanh(sdf_gt)
            loss += loss_l1(preds, sdf_gt).mean() / len(valid_voxels_idx)

            # transformed_sample = sdf_data[near_sample_indices[0], :3] - sdf_grid_indices_valid[center_point_index] 
            # transformed_sample.requires_grad = False
            # code = code.unsqueeze(0)
            # code = code.repeat(transformed_sample.shape[0], 1)
            # decoder_input = torch.cat([code, transformed_sample.cuda()], dim=1).float().cuda()
            # pred_sdf = decoder(decoder_input)
            # loss += loss_l1(pred_sdf, sdf_gt.cuda()) / len(sdf_grid_indices_valid)
        # if l2reg:
        #     loss += torch.sum(torch.norm(code, dim=0)) / len(sdf_grid_indices_valid)
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            logging.info('epoch {}, loss {}, lr {}, latent norm {}'.format(e, loss.item(), 
                                                                           optimizer.param_groups[0]['lr'],
                                                                           latent.norm(dim=1).mean().item()))
        loss_num = loss.cpu().data.numpy()

    # update latent_all
    latent_all[valid_voxels_idx] = latent.detach()

    return loss_num, latent_all, latent, valid_voxels_idx, valid_transforms

# python reconstruct.py -e examples/all -c 18 --split examples/splits/sv2_all_test_small.json -d data --skip
if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepLS decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--exp_idx",
        "-idx",
        dest="exp_idx",
        type=int,
        default=0,
        help="Manually split the evaluation with multiple threads.",
    )
    arg_parser.add_argument(
        "--num_exp",
        "-num",
        dest="num_exp",
        type=int,
        default=1,
        help="Manually split the evaluation with multiple threads.",
    )
    deep_ls.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_ls.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]
    cube_size = specs["VoxelResolution"]
    box_size = specs["VolumeSizeHalf"]
    voxel_radius = specs["ConsistenyLossExpandRadius"]

    # get coordinate field configuration
    coord_field_mode = train_utils.get_spec_with_default(specs, 'CoordinateFieldMode', 'none')
    if coord_field_mode == 'rotation_only':
        code_additional = 4    # 4-d rotation quaternion
    elif coord_field_mode == 'none':
        code_additional = 0
    elif coord_field_mode == 'computed':
        code_additional = 0
    elif coord_field_mode == 'computed_rotation_only':
        code_additional = 4    # 4-d rotation quaternion
    elif coord_field_mode == 'rotation_location':
        code_additional = 7
    elif coord_field_mode == 'computed_rotation_location':
        code_additional = 7
    else:
        raise NotImplementedError
    logging.info("Using coordinate field mode: {}".format(coord_field_mode))
    
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    # if torch.cuda.device_count() > 1:
    #     decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]
    state_dict = {}
    for k, v in saved_model_state["model_state_dict"].items():
        state_dict[k.replace('module.', '')] = v
    decoder.load_state_dict(state_dict)

    # if torch.cuda.device_count() > 1:
    #     decoder = decoder.module.cuda()
    # else:
    #     decoder = decoder.cuda()

    decoder = decoder.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = deep_ls.data.get_instance_filenames(args.data_source, split)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir, exist_ok=True)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir, exist_ok=True)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir, exist_ok=True)

    print('{} experiments, current {}'.format(args.num_exp, args.exp_idx))
    # start performing reconstruct
    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue
        # print(ii, args.exp_idx, ii % args.num_exp, ii % args.num_exp != args.exp_idx, npz)
        if ii % args.num_exp != args.exp_idx:
            continue
        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        data_sdf = deep_ls.data.read_sdf_samples_into_ram(full_filename)
        data_sdf = [remove_nans(data_sdf[0]), remove_nans(data_sdf[1])]

        for k in range(repeat):
            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, npz[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                # print('skip', npz)
                continue

            if not os.path.isfile(latent_filename):
                logging.info("reconstructing {}".format(npz))

                # data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
                # data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

                start = time.time()
                err, latent, latent_valid, latent_valid_idx, transforms_valid = reconstruct(
                    specs,
                    decoder,
                    int(args.iterations),
                    latent_size,
                    cube_size,
                    box_size,
                    voxel_radius,
                    code_additional,
                    data_sdf,
                    0.01,  # [emp_mean,emp_var],
                    0.1,
                    num_samples=8000,
                    lr=5e-3,
                    l2reg=True,
                )
                logging.debug("reconstruct time: {}".format(time.time() - start))
                err_sum += err
                logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
                logging.debug(ii)

                if not os.path.exists(os.path.dirname(latent_filename)):
                    os.makedirs(os.path.dirname(latent_filename), exist_ok=True)
                save_dict = {
                    'latent_valid': latent_valid,
                    'latent_valid_idx': latent_valid_idx,
                    'latent_all_num': latent.shape[0]
                }
                torch.save(save_dict, latent_filename)
                logging.info('latent saved at {}'.format(latent_filename))
            else:
                logging.info('use saved latent at {}'.format(latent_filename))
                save_dict = torch.load(latent_filename)
                latent_valid = save_dict['latent_valid']
                latent_valid_idx = save_dict['latent_valid_idx']
                latent_all_num = save_dict['latent_all_num']
                latent = None

            decoder.eval()
            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename), exist_ok=True)

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    deep_ls.mesh.create_mesh(
                        specs, decoder, latent, latent_valid, latent_valid_idx, transforms_valid, cube_size, box_size, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.debug("total time: {}".format(time.time() - start))