import glob
import logging
import numpy as np
import os
import random
import json
import tqdm
import torch
import torch.utils.data
import deep_ls.workspace as ws
from deep_ls.data import get_instance_filenames, remove_nans


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


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)    # [N_subsample,3]

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        split,
        load_ram=False,
    ):  
        self.config = config

        self.data_source = config["DataSource"]
        self.subsample = config["SamplePointPerVoxel"]
        self.split = split

        if split == 'train':
            self.split_file = config["TrainSplit"]
        else:
            self.split_file = config["TestSplit"]

        with open(self.split_file, "r") as f:
            self.split_file_loaded = json.load(f)

        self.npyfiles = get_instance_filenames(self.data_source, self.split_file_loaded)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + self.data_source
        )

        self.num_valid_voxel = len(self.npyfiles)

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            res = unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample)
        else:
            res = unpack_sdf_samples(filename, self.subsample)  # [N_subsample, 4]
        return_data = self.process_data(res, idx)
        return return_data
    
    def process_data(self, res, idx):
        res = res.unsqueeze(0)  # [M,4] -> [N=1,M,4]
        data = {
            'voxel_location': torch.tensor([0.0, 0.0, 0.0]).reshape(1,3),  # [N,3]
            'voxel_local_index': torch.tensor([0]).long(), # [N]
            'instance_global_index': idx,
            'sdf': res,
            'weights': torch.ones_like(res[:,:,:1]).float(), # [N,M,1]
            'idx': idx,
        }
        return data

