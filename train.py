import functools
import json
import logging
import math
import os
import signal
import sys
import time
import random
import pprint
import warnings
import deep_ls
import deep_ls.workspace as ws
import torch
import torch.utils.data as data_utils
import torch.distributed as dist
import numpy as np
from utils import train_utils, dist_utils, train_script, geo_utils, loss_utils
from deep_ls.data_voxel import VoxelBased_SDFSamples
from deep_ls.data_voxel_global import SDFSamples



def signal_handler(sig, frame):
    logging.info("Stopping early...")
    sys.exit(0)


def main(exp_dir, continue_from):
    logger = train_utils.get_logger(exp_dir)

    logger.info("running " + exp_dir)
    logger.info("training with {} GPU(s)".format(torch.cuda.device_count()))
    signal.signal(signal.SIGINT, signal_handler)

    # get training specifications
    config = ws.load_experiment_specifications(exp_dir)
    logger.info(pprint.pformat(config))

    # get experiment description
    logger.info("Experiment description: \n" + str(config["Description"]))

    # set random seeds
    torch.cuda.manual_seed_all(config["Seed"])
    torch.manual_seed(config["Seed"])
    np.random.seed(config["Seed"])
    random.seed(config["Seed"])

    # set device
    gpus = range(torch.cuda.device_count())
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')
    local_rank = 0
    # if "LOCAL_RANK" in os.environ:
    #     dist_utils.dist_init(int(os.environ["LOCAL_RANK"]))
    # local_rank = 0 #dist.get_rank()
    # torch.cuda.set_device(local_rank)

    # parse saving epoch numbers
    #saving_epochs = train_utils.get_saving_epochs(config)

    # get voxel coordinates
    voxel_coordinates = deep_ls.data_voxel.generate_grid_center_indices(config["VoxelResolution"],
                                                                        config["VolumeSizeHalf"]).reshape(-1,3)

    # get dataset
    dataset = VoxelBased_SDFSamples(config, voxel_coordinates, split='train', load_ram=True)
    #dataset = SDFSamples(config, split='train', load_ram=False)
    #data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=config["SampleScenePerBatch"],
                                               shuffle=True,
                                               num_workers=int(config["DataLoaderThreads"]), 
                                               pin_memory=True, 
                                               drop_last=True)
                                               #sampler=data_sampler)

    # get number of valid voxels
    num_valid_voxels = dataset.num_valid_voxel
    logger.info("Valid voxel number: " + str(num_valid_voxels))

    # get coordinate field configuration
    coord_field_mode = train_utils.get_spec_with_default(config, 'CoordinateFieldMode', 'none')
    code_length = config["CodeLength"]
    if coord_field_mode == 'rotation_only':
        code_additional = 4     # 4-d rotation quaternion
    elif coord_field_mode == 'none':
        code_additional = 0
    elif coord_field_mode == 'rotation_scale':
        code_additional = 4 + 3 # 4-d rotation quaternion & 3-d scale
    elif coord_field_mode == 'computed':
        code_additional = 0
    elif coord_field_mode == 'computed_rotation_only':
        code_additional = 4
    elif coord_field_mode == 'rotation_location':
        code_additional = 7 # 4+3
    elif coord_field_mode == 'computed_rotation_location':
        code_additional = 7 # 4+3
    else:
        raise NotImplementedError
    logger.info("Using coordinate field mode: {}".format(coord_field_mode))

    # get latent code
    lat_vecs = torch.nn.Embedding(num_valid_voxels, code_length + code_additional, max_norm=config["CodeBound"])
    torch.nn.init.normal_(lat_vecs.weight.data, 0.0, 1.0 / math.sqrt(config["CodeLength"]))
    if coord_field_mode == 'rotation_only':
        rot_init = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).view(1,3,3)
        quat_init = geo_utils.mat2quat_transform(rot_init)   # [1,4]
        quat_init = quat_init.repeat(num_valid_voxels, 1)
        lat_vecs.weight.data[:, code_length:] = quat_init
    elif coord_field_mode == 'none':
        pass
    elif coord_field_mode == 'rotation_scale':
        rot_init = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).view(1,3,3)
        quat_init = geo_utils.mat2quat_transform(rot_init)   # [1,4]
        quat_init = quat_init.repeat(num_valid_voxels, 1)
        lat_vecs.weight.data[:, code_length:code_length+4] = quat_init
        lat_vecs.weight.data[:, code_length+4] = 1.0
    elif coord_field_mode == 'computed':
        pass
    elif coord_field_mode == 'computed_rotation_only':
        rot_all = dataset.computed_rotation # [num_valid_voxels, 3, 3]
        assert rot_all.shape[0] == num_valid_voxels
        quat_all = geo_utils.mat2quat_transform(rot_all)   # [num_valid_voxels, 4]
        lat_vecs.weight.data[:, code_length:] = quat_all
    elif coord_field_mode == 'rotation_location':
        rot_init = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]).view(1,3,3)
        quat_init = geo_utils.mat2quat_transform(rot_init)   # [1,4]
        quat_init = quat_init.repeat(num_valid_voxels, 1)
        loc_init = torch.tensor([[0.0, 0, 0]]).repeat(num_valid_voxels, 1)
        lat_vecs.weight.data[:, code_length:code_length+4] = quat_init
        lat_vecs.weight.data[:, code_length+4:] = loc_init
    elif coord_field_mode == 'computed_rotation_location':
        rot_all = dataset.computed_rotation # [num_valid_voxels, 3, 3]
        assert rot_all.shape[0] == num_valid_voxels
        quat_all = geo_utils.mat2quat_transform(rot_all)   # [num_valid_voxels, 4]
        loc_init = torch.tensor([[0.0, 0, 0]]).repeat(num_valid_voxels, 1)
        lat_vecs.weight.data[:, code_length:code_length+4] = quat_all
        lat_vecs.weight.data[:, code_length+4:] = loc_init
    else:
        raise NotImplementedError

    # build decoder
    arch = __import__("networks." + config["NetworkArch"], fromlist=["Decoder"])
    decoder = arch.Decoder(config["CodeLength"], **config["NetworkSpecs"])
    print(decoder)

    # get optimizer
    lr_schedules = train_utils.get_learning_rate_schedules(config)
    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    # get loss
    loss_type = train_utils.get_spec_with_default(config, 'Loss', 'L1')
    logger.info('Using loss {}'.format(loss_type))
    if loss_type == 'L1':
        loss = torch.nn.L1Loss(reduction="none")
    elif loss_type == 'L2':
        loss = torch.nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError

    # resume
    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}
    start_epoch = 0
    if continue_from is not None:
        logger.info('continuing from "{}"'.format(continue_from))
        lat_epoch = train_utils.load_latent_vectors(exp_dir, continue_from + ".pth", lat_vecs)
        model_epoch = ws.load_model_parameters(exp_dir, continue_from, decoder)
        optimizer_epoch = train_utils.load_optimizer(exp_dir, continue_from + ".pth", optimizer_all)
        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = train_utils.load_logs(exp_dir)
        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = train_utils.clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch)
        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError("epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch))
        start_epoch = model_epoch + 1

    logger.info("starting from epoch {}".format(start_epoch))
    logger.info("Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())))
    logger.info("Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings, lat_vecs.embedding_dim,))
    logger.info("initialized with mean magnitude {}".format(
            train_utils.get_mean_latent_vector_magnitude(lat_vecs, code_length)))
    logger.info("code length {}, additional coordinate field code length {}".format(
            code_length, code_additional))

    # to cuda
    # decoder = decoder.to(device)
    # decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
    # lat_vecs = lat_vecs.to(device)
    # if device == torch.device("cuda"):
    #     torch.backends.cudnn.benchmark = True
    #     device_ids = range(torch.cuda.device_count())
    #     print("using {} cuda".format(len(device_ids)))
    #     decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank], find_unused_parameters=False)
    #     lat_vecs = torch.nn.parallel.DistributedDataParallel(lat_vecs, device_ids=[local_rank], find_unused_parameters=False)
    #     device_num = len(device_ids)
    #     ddp = True
    #breakpoint()
    decoder = decoder.to(device)
    lat_vecs = lat_vecs.to(device)
    decoder = torch.nn.parallel.DataParallel(decoder)
    lat_vecs = torch.nn.parallel.DataParallel(lat_vecs)

    # get number of epochs
    iteration_each_epoch = len(dataset) // config["SampleScenePerBatch"]
    epoch_end = config['NumIterations'] // iteration_each_epoch
    print('Train for {} epochs, {} iterations per epoch'.format(epoch_end, iteration_each_epoch))

    # decrease lr epoch number
    decrease_lr_epoch0 = epoch_end // 8
    decrease_lr_epochs = [(it+1) * decrease_lr_epoch0 for it in range(8)]
    print('Decrease learning rate at {} epoches'.format(decrease_lr_epochs))

    # train
    for epoch in range(start_epoch, epoch_end+1):
        print('Epoch {}'.format(epoch))
        start = time.time()

        loss_log = train_script.train_epoch(config, dataset, data_loader, decoder, lat_vecs, 
                                            loss, optimizer_all, lr_schedules, 
                                            epoch, device, local_rank, loss_log, decrease_lr_epochs)

        end = time.time()
        timing_log.append(end - start)
        #lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
        lr_log.append(optimizer_all.param_groups[0]["lr"])

        train_utils.save_latest(exp_dir, epoch, decoder, optimizer_all, lat_vecs)

        lat_mag_log.append(train_utils.get_mean_latent_vector_magnitude(lat_vecs.module, code_length))
        train_utils.append_parameter_magnitudes(param_mag_log, decoder)
        
        #if epoch in saving_epochs:
        train_utils.save_checkpoints(exp_dir, epoch, decoder, optimizer_all, lat_vecs)

        #if epoch % train_utils.get_spec_with_default(config, "LogFrequency", 10) == 0:
        train_utils.save_logs(
            exp_dir,
            loss_log,
            lr_log,
            timing_log,
            lat_mag_log,
            param_mag_log,
            epoch,
        )


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepLS autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
             + "from the latest running snapshot, or an integer corresponding to "
             + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')

    deep_ls.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    #deep_ls.configure_logging(args)

    main(args.experiment_directory, args.continue_from)