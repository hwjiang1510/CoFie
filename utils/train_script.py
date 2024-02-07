import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
import itertools
from torch.cuda.amp import autocast
from einops import rearrange
from utils import train_utils, geo_utils


logger = logging.getLogger(__name__)


def adjust_lr(optimizer_all, epoch, decrease_lr_epochs):
    print('current lr', optimizer_all.param_groups[0]["lr"])
    if epoch in decrease_lr_epochs:
        for i, param_group in enumerate(optimizer_all.param_groups):
            param_group["lr"] *= 0.5
            print('Decrease lr')


def train_epoch(config, dataset, loader, decoder, codes_all, loss, optimizer_all, lr_schedules, epoch, device, local_rank, loss_log, decrease_lr_epochs):
    time_meters = train_utils.AverageMeters()
    loss_meters = train_utils.AverageMeters()

    decoder.train()
    adjust_lr(optimizer_all, epoch, decrease_lr_epochs)
    do_code_regularization = train_utils.get_spec_with_default(config, "CodeRegularization", True)
    code_reg_lambda = train_utils.get_spec_with_default(config, "CodeRegularizationLambda", 1e-4)
    sdf_chunk_value = train_utils.get_spec_with_default(config, 'ClampingDistance', 0.1)
    code_length = config["CodeLength"]

    batch_end = time.time()
    for batch_idx, sample_unbactched in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch
        # print(iter_num)
        
        # make sample into 2D inputs
        # print('1')
        sample = batchify_sample(config, sample_unbactched)
        sample = train_utils.dict_to_cuda(sample, device)

        # get latent code
        # print('2')
        codes = codes_all.module(sample['voxel_global_index'].long())   # [N,C+C']
        codes_frame = codes[:, code_length:]                            # [N,C']
        codes = codes[:, :code_length]                                  # [N,C]

        # get points and their sdf values (in world coordinate)
        # print('3')
        sdf = sample['sdf']
        points = sdf[:,:3]      # [B,3]
        sdf_values = sdf[:,3:]  # [B,1]
        sdf_values = torch.tanh(sdf_values)

        # chunk sdf values
        # sdf_values = sdf_values.clamp(min=-sdf_chunk_value, max=sdf_chunk_value)

        # transform points into local coordinate
        # print('4')
        points_local = transform_to_local(config, points, sample, codes_frame)
        # points_local.requires_grad = False

        # get inputs
        # print('5')
        #breakpoint()
        inputs = torch.cat([codes, points_local], dim=1).float()

        # get prediction
        preds_raw = decoder(inputs)
        #preds = preds_raw.clamp(min=-sdf_chunk_value, max=sdf_chunk_value)

        # get loss
        loss_sdf = loss(preds_raw, sdf_values).mean() 
        #loss_sdf = (loss_sdf * sample['weights'].float()).sum() / sample['weights'].sum().item()
        loss_meters.add_loss_value('sdf loss', loss_sdf.detach().item())
        if do_code_regularization:
            l2_size_loss = torch.sum(torch.norm(codes, dim=1))
            loss_reg = (code_reg_lambda * min(1.0, iter_num / 500) * l2_size_loss) / len(sdf)
            loss_sdf = loss_sdf + loss_reg  #l2_size_loss
            loss_meters.add_loss_value('reg loss', loss_reg.detach().item())
        loss_log.append(loss_sdf.detach().item())

        loss_sdf.backward()
        optimizer_all.step()
        optimizer_all.zero_grad()

        time_meters.add_loss_value('Batch time', time.time() - batch_end)
        if iter_num % 10 == 0 and local_rank == 0:
            msg = 'Epoch {0}, iter {1}, rank {2}, Time {data_time:.3f}s, Point Location {point_loc:.3f}, SDF mean GT {inputs_sdf_mean:.4f}, pred {preds_sdf_mean:.4f}, Loss:'.format(
                epoch, iter_num, local_rank, 
                point_loc=points_local[sample['weights'].long().squeeze()].mean().detach().item(),
                inputs_sdf_mean=sdf_values[sample['weights'].long().squeeze()].mean().detach().item(),
                preds_sdf_mean=preds_raw[sample['weights'].long().squeeze()].mean().detach().item(),
                data_time=time_meters.average_meters['Batch time'].avg
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        #dist.barrier()
        batch_end = time.time()

    return loss_log


def batchify_sample(config, sample):
    '''
    sample:
        'voxel_location': [B,N,3], voxel location in world
        'voxel_local_index': [B,N], voxel local index
        'instance_global_index': [B], global index of the scene/instance
        'sdf': [B,N,M,4]
        'weights': [B,N,M,1]
        'transform': [B,N,3,3]
    '''
    use_computed_coordinate = 'computed' in train_utils.get_spec_with_default(config, 'CoordinateFieldMode', 'none')

    voxel_location = sample['voxel_location']
    voxel_local_index = sample['voxel_local_index']
    instance_global_index = sample['instance_global_index']
    sdf = sample['sdf']
    weight = sample['weights']
    if use_computed_coordinate:
        transform = sample['transform']

    B,N,M,_ = sdf.shape
    voxel_location = voxel_location.reshape(B,N,1,3).repeat(1,1,M,1)    # [B,N,M,3]
    voxel_global_index = voxel_local_index + instance_global_index.reshape(B,1)     # [B,N]
    voxel_global_index = voxel_global_index.reshape(B,N,1).repeat(1,1,M)  # [B,N,M]
    if use_computed_coordinate:
        transform = transform.reshape(B,N,1,3,3).repeat(1,1,M,1,1)  # [B,N,M,3,3]

    voxel_location = rearrange(voxel_location, 'b n m c -> (b n m) c')
    voxel_global_index = rearrange(voxel_global_index, 'b n m -> (b n m)')
    sdf = rearrange(sdf, 'b n m c -> (b n m) c')
    weight = rearrange(weight, 'b n m c -> (b n m) c')
    if use_computed_coordinate:
        transform = rearrange(transform, 'b n m a c -> (b n m) a c')

    sample_new = {
        'voxel_location': voxel_location.float(),
        'voxel_global_index': voxel_global_index,
        'sdf': sdf.float(),
        'weights': weight.float()
    }
    if use_computed_coordinate:
        sample_new['transform'] = transform.float()
    
    return sample_new


def transform_to_local(config, points, sample, codes_frame):
    coord_field_mode = train_utils.get_spec_with_default(config, 'CoordinateFieldMode', 'none')
    if coord_field_mode == 'none':
        # deepls default coordinate field definition
        # local coordinate is defined at the center of voxels and has axis-aligned rotation
        points_local = points - sample['voxel_location']
    elif coord_field_mode == 'rotation_only' or coord_field_mode == 'computed_rotation_only':
        # local coordinate is defined at the center of voxels and has grid-specific rotation
        assert codes_frame.shape[-1] == 4   # rotation is represented by the 4-dim quaternion
        points_local = points - sample['voxel_location']           # [B,3]
        coord_rot_mat = geo_utils.quat2mat_transform(codes_frame)  # quaternion [B,4] -> rotation matrix [B,3,3]
        #breakpoint()
        # points_local = torch.einsum('ijk,ik->ik', coord_rot_mat, points_local)   # [B,3]
        points_local = torch.einsum('bij,bj->bi', coord_rot_mat, points_local)   # [B,3]
    elif coord_field_mode == 'rotation_scale':
        # local coordinate is defined at the center of voxels and has grid-specific rotation
        assert codes_frame.shape[-1] == 7   # rotation is represented by the 4-dim quaternion
        points_local = points - sample['voxel_location']           # [B,3]
        codes_frame_rot = codes_frame[:,:4]
        coord_rot_mat = geo_utils.quat2mat_transform(codes_frame_rot)  # quaternion [B,4] -> rotation matrix [B,3,3]
        #breakpoint()
        # points_local = torch.einsum('ijk,ik->ik', coord_rot_mat, points_local)   # [B,3]
        points_local = torch.einsum('bij,bj->bi', coord_rot_mat, points_local)   # [B,3]
        codes_frame_scale = codes_frame[:,4:]
        points_local *= codes_frame_scale
    elif coord_field_mode == 'computed':
        points_local = points - sample['voxel_location']    # [B,3]
        coord_rot_mat = sample['transform']     # [B,3,3] 
        # points_local = torch.einsum('ijk,ik->ik', coord_rot_mat, points_local)   # [B,3]
        points_local = torch.einsum('bij,bj->bi', coord_rot_mat, points_local)   # [B,3]
    elif coord_field_mode == 'rotation_location' or coord_field_mode == 'computed_rotation_location':
        # local coordinate is defined at the center of voxels and has grid-specific rotation
        assert codes_frame.shape[-1] == 7
        points_local = points - sample['voxel_location']            # [B,3]
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


