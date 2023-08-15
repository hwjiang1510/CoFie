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


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_idx = torch.tensor(random.choices(np.arange(len(pos_size)), k=half)).long()
    neg_idx = torch.tensor(random.choices(np.arange(len(pos_size)), k=half)).long()

    pos_sampled = pos_tensor[pos_idx]
    neg_sampled = neg_tensor[neg_idx]
    samples = torch.cat([pos_sampled, neg_sampled], 0)

    # pos_start_ind = random.randint(0, pos_size - half)
    # sample_pos = pos_tensor[pos_start_ind: (pos_start_ind + half)]

    # if neg_size <= half:
    #     random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    #     sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    # else:
    #     neg_start_ind = random.randint(0, neg_size - half)
    #     sample_neg = neg_tensor[neg_start_ind: (neg_start_ind + half)]

    # samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class VoxelBased_SDFSamples(torch.utils.data.Dataset):
    def __init__(
            self,
            config,
            voxel_coordinates,
            split='train',
            load_ram=True,
    ):
        self.config = config
        self.split = split
        self.voxel_coordinates = voxel_coordinates  # [R,R,R,3]

        self.data_path = config["DataSource"]
        if split == 'train':
            self.split_file = config["TrainSplit"]
        else:
            self.split_file = config["TestSplit"]

        with open(self.split_file, "r") as f:
            self.split_file_loaded = json.load(f)

        self.data_source = config["DataSource"]
        self.npyfiles = get_instance_filenames(self.data_source, self.split_file_loaded)
        logging.info("Training on {} scenes".format(len(self.npyfiles)))

        self.num_voxels_per_scene = config["SampleVoxelPerScene"]
        self.num_points_per_voxel = config["SamplePointPerVoxel"]
        self.volume_size_half = config["VolumeSizeHalf"]
        self.voxel_resolution = config["VoxelResolution"]
        self.expand_ratio = config["ConsistenyLossExpandRadius"]

        self.farthest = (self.volume_size_half * 2) / self.voxel_resolution * self.expand_ratio

        self.load_ram = load_ram
        self.num_valid_voxel = 0
        self.num_valid_voxel_accumu = []
        
        # loading number of valid voxel and its accumulation
        file_path = './data/cache/cache_vol-{}_res-{}_expand-{}.npy'.format(
                        self.volume_size_half, int(self.voxel_resolution), self.expand_ratio)
        print('loading number of valid voxel accumulation')
        if not os.path.isfile(file_path):
            self.num_valid_voxel = 0
            self.num_valid_voxel_accumu = []
            for f in tqdm.tqdm(self.npyfiles):
                self.num_valid_voxel_accumu.append(self.num_valid_voxel)
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                association_filename = filename.replace('SdfSamples', 'SdfSamplesToVoxelIndices')
                association_filename = association_filename.replace('.npz', '_vol-{}_res-{}_expand-{}.json'.format(
                    self.volume_size_half, int(self.voxel_resolution), self.expand_ratio))
                with open(association_filename, 'r') as f_association:
                        cur_association = json.load(f_association)
                self.num_valid_voxel += len(cur_association.keys())
            data = {
                'num_valid_voxel': self.num_valid_voxel,
                'num_valid_voxel_accumu': self.num_valid_voxel_accumu
                }
            with open(file_path, 'w') as write_f:
                json.dump(data, write_f)
        else:
            with open(file_path, 'r') as read_f:
                data = json.load(read_f)
            self.num_valid_voxel = data['num_valid_voxel']
            self.num_valid_voxel_accumu = data['num_valid_voxel_accumu']
        
        print('{} local shapes ({}% valid)'.format(self.num_valid_voxel, 100 * self.num_valid_voxel / (self.voxel_resolution**3 * len(self.npyfiles))))
        print('Accumulation list', self.num_valid_voxel_accumu)
        
        # if load_ram:
        #     file_path = './data/cache/cache_vol-{}_res-{}_expand-{}.npy'
        #     if not os.path.isfile(file_path):
        #         self.loaded_sdf, self.loaded_association = [], []
        #         for f in tqdm.tqdm(self.npyfiles):
        #             self.num_valid_voxel_accumu.append(self.num_valid_voxel)
        #             filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
        #             npz = np.load(filename)
        #             pos_tensor = torch.from_numpy(npz["pos"])
        #             neg_tensor = torch.from_numpy(npz["neg"])
        #             all_points = torch.cat([pos_tensor, neg_tensor], dim=0)
        #             self.loaded_sdf.append(all_points)
        #             association_filename = filename.replace('SdfSamples', 'SdfSamplesToVoxelIndices')
        #             association_filename = association_filename.replace('.npz', '_vol-{}_res-{}_expand-{}.json'.format(
        #                 self.volume_size_half, int(self.voxel_resolution), self.expand_ratio))
        #             with open(association_filename, 'r') as f_association:
        #                 cur_association = json.load(f_association)
        #             self.loaded_association.append(cur_association)
        #             self.num_valid_voxel += len(cur_association.keys())
        #         # data = {
        #         #     'num_valid_voxel': self.num_valid_voxel,
        #         #     'num_valid_voxel_accumu': self.num_valid_voxel_accumu,
        #         #     'all_points': self.loaded_sdf.tolist(),
        #         #     'association': self.loaded_association}
        #         # with open(file_path, 'w') as write_f:
        #         #     json.dum(data, write_f)
        #     else:
        #         with open(file_path, 'r') as read_f:
        #             data = json.load(read_f)
        #         self.num_valid_voxel = data['num_valid_voxel']
        #         self.num_valid_voxel_accumu = data['num_valid_voxel_accumu']
        #         self.loaded_sdf = data['all_points']
        #         self.loaded_association = data['association']

        print('Loading finished')


    def __len__(self):
        return 100000 #len(self.npyfiles)
    
    def process_data(self, sdf, association, idx):
        '''
        sdf: [K,4] tensor of point location and its sdf value
        association: the key is the voxel index, the value is the point index within is expasion range

        For every instance, we sampled N voxels; Each of the voxel is associated with M points (balanced)
        '''
        # sample N voxels
        all_voxel_index = list(association.keys())
        num_voxels = len(all_voxel_index)
        selected_voxel_key_index = random.choices(np.arange(num_voxels), k=self.num_voxels_per_scene)
        selected_voxel_index = [all_voxel_index[it] for it in selected_voxel_key_index]
        selected_voxel_index_int = [int(it) for it in selected_voxel_index]
        selected_voxel_location = self.voxel_coordinates[selected_voxel_index_int]  # [N,3]

        # selected_voxel_index = []
        # for it in selected_voxel_index_string:
        #     ii, jj, kk = it.split(',')
        #     ii, jj, kk = int(ii.replace(' ', '')), int(jj.replace(' ', '')), int(kk.replace(' ', ''))
        #     selected_voxel_index.append([ii,jj,kk])
        # selected_voxel_index = torch.tensor(selected_voxel_index).long()
        # i, j, k = selected_voxel_index[:,0], selected_voxel_index[:,1], selected_voxel_index[:,2]
        # selected_voxel_location = self.voxel_coordinates[i, j, k]   # [N,3]

        # for each voxel, sample M points (balanced)
        all_points_index = [association[it] for it in selected_voxel_index] # [N,?]
        all_points = [sdf[it] for it in all_points_index]   # [N,?,4]
        all_points_balanced, all_points_weight = [], []
        for (it, voxel_center) in zip(all_points, selected_voxel_location):
            # pos = it[it[:,-1] >= 0]
            # neg = it[it[:,-1] < 0]
            # pos_sampled, pos_weights = self.sample_points(pos, voxel_center)  # [M/2,4], [M/2,1]
            # neg_sampled, neg_weights = self.sample_points(neg, voxel_center)  # [M/2,4], [M/2,1]
            # cur_sampled = torch.cat([pos_sampled, neg_sampled], dim=0)  # [M,4]
            # cur_weights = torch.cat([pos_weights, neg_weights], dim=0)  # [M,1]
            cur_sampled, cur_weights = self.sample_points(it, voxel_center)
            all_points_balanced.append(cur_sampled)
            all_points_weight.append(cur_weights)
        all_points_balanced = torch.stack(all_points_balanced)  # [N,M,4]
        all_points_weight = torch.stack(all_points_weight)      # [N,M,1]

        res = {
            'voxel_location': selected_voxel_location,  # [N,3]
            'voxel_local_index': torch.tensor(selected_voxel_key_index).long(), # [N]
            'instance_global_index': self.num_valid_voxel_accumu[idx],
            'sdf': all_points_balanced, # [N,M,4]
            'weights': all_points_weight, # [N,M,1]
            'idx': idx,
        }
        return res
    
    
    def sample_points(self, pts, voxel_center):
        num_sample = self.config['SamplePointPerVoxel'] #// 2
        # pts = remove_nans(pts)
        #pts = self.remove_far(pts, voxel_center)

        if len(pts) == 0:
            sample_pts = torch.zeros(num_sample, 4)
            weights = torch.zeros(num_sample, 1)
        else:
            sample_idx = torch.tensor(random.choices(np.arange(len(pts)), k=num_sample)).long()
            sample_pts = pts[sample_idx]
            weights = torch.ones(num_sample, 1)
        return sample_pts, weights
    

    def remove_far(self, pts, voxel_center):
        pts_centered_abs = torch.abs(pts[:,:3] - voxel_center.unsqueeze(0))
        mask = torch.logical_and(pts_centered_abs[:,0]<self.farthest, pts_centered_abs[:,1]<self.farthest)
        mask = torch.logical_and(mask, pts_centered_abs[:,2]<self.farthest)
        #print(mask.float().mean())
        pts = pts[mask]
        return pts


    def load_data(self, idx):
        f = self.npyfiles[idx]
        filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
        npz = np.load(filename)
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
        sdf = torch.cat([pos_tensor, neg_tensor], dim=0)

        association_filename = filename.replace('SdfSamples', 'SdfSamplesToVoxelIndices')
        association_filename = association_filename.replace('.npz', '_vol-{}_res-{}_expand-{}.json'.format(
            self.volume_size_half, int(self.voxel_resolution), self.expand_ratio))
        with open(association_filename, 'r') as f_association:
            cur_association = json.load(f_association)

        return sdf, cur_association


    def __getitem__(self, idx):
        #sdf = self.loaded_sdf[idx]
        #association = self.loaded_association[idx]
        real_idx = idx % len(self.npyfiles)

        sdf, association = self.load_data(real_idx)

        data = self.process_data(sdf, association, real_idx)
        return data


