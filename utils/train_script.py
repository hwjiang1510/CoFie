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
from utils import train_utils


def adjust_lr(optimizer_all, epoch, decrease_lr_epochs):
    if epoch in decrease_lr_epochs:
        for i, param_group in enumerate(optimizer_all.param_groups):
            param_group["lr"] *= 0.5


def train_epoch(config, dataset, loader, decoder, codes_all, loss, optimizer_all, lr_schedules, epoch, device, local_rank, loss_log, decrease_lr_epochs):
    time_meters = train_utils.AverageMeters()
    loss_meters = train_utils.AverageMeters()

    decoder.train()
    #train_utils.adjust_learning_rate(lr_schedules, optimizer_all, epoch)
    adjust_lr(optimizer_all, epoch, decrease_lr_epochs)
    do_code_regularization = train_utils.get_spec_with_default(config, "CodeRegularization", True)
    code_reg_lambda = train_utils.get_spec_with_default(config, "CodeRegularizationLambda", 1e-4)

    batch_end = time.time()
    for batch_idx, sample_unbactched in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch
        #__import__('pdb').set_trace()
        
        # make sample into 2D inputs
        sample = batchify_sample(sample_unbactched)
        sample = train_utils.dict_to_cuda(sample, device)

        #__import__('pdb').set_trace()

        # get latent code
        codes = codes_all.module(sample['voxel_global_index'].long())

        # get points and their sdf values (in world coordinate)
        sdf = sample['sdf']
        points = sdf[:,:3]      # [B,3]
        sdf_values = sdf[:,3:]  # [B,1]
        #sdf_values = torch.tanh(sdf_values)

        # transform points into local coordinate
        points_local = transform_to_local(points, sample)
        points_local.requires_grad = False

        # multiply with a factor
        sdf_values *= 100
        points_local *= 100

        # get inputs
        inputs = torch.cat([codes, points_local], dim=1).float()

        # get prediction
        preds = decoder(inputs)

        #__import__('pdb').set_trace()

        # get loss
        loss_sdf = (loss(preds, sdf_values)[sample['weights'].long().squeeze()]).mean()
        loss_meters.add_loss_value('sdf loss', loss_sdf.detach().item())
        if do_code_regularization:
            l2_size_loss = torch.sum(torch.norm(codes, dim=0)) / len(sdf)
            loss_reg = (code_reg_lambda * min(1.0, epoch / 76) * l2_size_loss) / len(sdf)
            loss_sdf = loss_sdf + loss_reg  #l2_size_loss
            loss_meters.add_loss_value('reg loss', loss_reg.detach().item())
        loss_log.append(loss_sdf.detach().item())

        loss_sdf.backward()
        # for name, param in decoder.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter: {name}, Gradient mean: {param.grad.mean().item()}")
        #__import__('pdb').set_trace()
        
        optimizer_all.step()
        optimizer_all.zero_grad()

        #__import__('pdb').set_trace()

        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % 10 == 0 and local_rank == 0:
            msg = 'Epoch {0}, iter {1}, rank {2}, Time {data_time:.3f}s, Point Location {point_loc:.3f}, SDF mean GT {inputs_sdf_mean:.4f}, pred {preds_sdf_mean:.4f}, Loss:'.format(
                epoch, iter_num, local_rank, 
                point_loc=points_local[sample['weights'].long().squeeze()].mean().detach().item(),
                inputs_sdf_mean=sdf_values[sample['weights'].long().squeeze()].mean().detach().item(),
                preds_sdf_mean=preds[sample['weights'].long().squeeze()].mean().detach().item(),
                data_time=time_meters.average_meters['Batch time'].avg
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logging.info(msg)

        #dist.barrier()
        batch_end = time.time()

    return loss_log


def batchify_sample(sample):
    '''
    sample:
        'voxel_location': [B,N,3], voxel location in world
        'voxel_local_index': [B,N], voxel local index
        'instance_global_index': [B], global index of the scene/instance
        'sdf': [B,N,M,4]
        'weights': [B,N,M,1]
    '''
    voxel_location = sample['voxel_location']
    voxel_local_index = sample['voxel_local_index']
    instance_global_index = sample['instance_global_index']
    sdf = sample['sdf']
    weight = sample['weights']

    #print(voxel_location.shape, voxel_local_index.shape, instance_global_index.shape, sdf.shape, weight.shape)

    B,N,M,_ = sdf.shape
    voxel_location = voxel_location.reshape(B,N,1,3).repeat(1,1,M,1)    # [B,N,M,3]
    voxel_global_index = voxel_local_index + instance_global_index.reshape(B,1)     # [B,N]
    voxel_global_index = voxel_global_index.reshape(B,N,1).repeat(1,1,M)  # [B,N,M]

    voxel_location = rearrange(voxel_location, 'b n m c -> (b n m) c')
    voxel_global_index = rearrange(voxel_global_index, 'b n m -> (b n m)')
    sdf = rearrange(sdf, 'b n m c -> (b n m) c')
    weight = rearrange(weight, 'b n m c -> (b n m) c')

    sample_new = {
        'voxel_location': voxel_location,
        'voxel_global_index': voxel_global_index,
        'sdf': sdf,
        'weights': weight
    }
    return sample_new


def transform_to_local(points, sample):
    points_local = points - sample['voxel_location']
    return points_local


