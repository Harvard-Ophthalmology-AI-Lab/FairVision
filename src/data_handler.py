import sys, os
from PIL import Image
import blobfile as bf
import numpy as np
import random
import csv
import pickle
from datetime import datetime
import scipy.stats as stats
from skimage.transform import resize
from glob import glob
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

def find_all_files_(folder, str_pattern='*.npz'):
    files = [os.path.basename(y) for x in os.walk(folder) for y in glob(os.path.join(x[0], str_pattern))]
    return files

def get_all_pids(data_dir):
    pids = []
    dict_pid_fid = {}
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        pid = raw_data['pid'].item()
        pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid

def get_all_pids_filter(data_dir, keep_list=None):
    race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

    pids = []
    dict_pid_fid = {}
    files = []
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        race = raw_data['race'].item()
        if keep_list is not None and race not in keep_list:
            continue

        if not hasattr(raw_data, 'pid'):
            pid = f[f.find('_')+1:f.find('.')]
        else:
            pid = raw_data['pid'].item()
            pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
        files.append(f)
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid, files

def vf_to_matrix(vec, fill_in=-50):
    mat = np.empty((8,9))
    mat[:] = fill_in

    mat[0, 3:7] = vec[0:4]
    mat[1, 2:8] = vec[4:10]
    mat[2, 1:] = vec[10:18]
    mat[3, :7] = vec[18:25]
    mat[3, 8] = vec[25]
    mat[4, :7] = vec[26:33]
    mat[4, 8] = vec[33]
    mat[5, 1:] = vec[34:42]
    mat[6, 2:8] = vec[42:48]
    mat[7, 3:7] = vec[48:52]

    return mat

def endless_loader(dataloader):
    while True:
        for data in dataloader:
            yield data

class FairVision_Glaucoma(Dataset):
    # subset: train | val | test
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0,
                    depth=3, indices=None, attribute_type='race', transform=None, needBalance=False, dataset_proportion=1., select_group=-1):

        self.data_path = data_path
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform
        self.needBalance = needBalance
        self.dataset_proportion = dataset_proportion

        self.data_files = find_all_files(self.data_path, suffix='npz')
        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]
        tmp_data_files = []
        if self.attribute_type == 'gender' or self.attribute_type == 'maritalstatus' or self.attribute_type == 'hispanic' or self.attribute_type == 'language':
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if self.attribute_type == 'gender':
                    attr = raw_data['male'].item()
                else:
                    attr = raw_data[self.attribute_type].item()
                if attr > -1:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        if self.modality_type == 'oct_bscans' or self.modality_type == 'oct_bscans_3d':
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if len(raw_data['oct_bscans']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        elif self.modality_type == 'slo_fundus':
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if len(raw_data['slo_fundus']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        # Oversampling
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        if self.subset == 'train' and self.needBalance:
            all_attrs = []
            for idx in range(0, len(self.data_files)):
                data_file = os.path.join(self.data_path, self.data_files[idx])
                raw_data = np.load(data_file, allow_pickle=True)
                cur_label = raw_data[self.attribute_type].item()
                all_attrs.append(cur_label)

                _, nums_attrs = np.unique(all_attrs, return_counts=True)

                if cur_label not in self.label_samples:
                    self.label_samples[cur_label] = list()
                self.label_samples[cur_label].append(self.data_files[idx])
                self.balanced_max = len(self.label_samples[cur_label]) \
                    if len(self.label_samples[cur_label]) > self.balanced_max else self.balanced_max
            ttl_num_samples = 0
            self.class_samples_num = [0]*len(list(self.label_samples.keys()))
            for i, (k,v) in enumerate(self.label_samples.items()):
                self.class_samples_num[int(k)] = len(v)
                ttl_num_samples += len(v)
                print(f'{k}-th identity training samples: {len(v)}')
            print(f'total number of training samples: {ttl_num_samples}')
            self.class_samples_num = np.array(self.class_samples_num)

            # Oversample the classes with fewer elements than the max
            for i_label in self.label_samples:
                while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
                    self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))

            data_files = []
            for i, (k,v) in enumerate(self.label_samples.items()):
                data_files = data_files + v
            self.data_files = data_files
        
        if self.subset == 'train' and self.dataset_proportion < 1.:
            num_samples = int(len(self.data_files) * self.dataset_proportion)
            self.data_files = random.sample(self.data_files, num_samples)
        
        # -1=unknown
        # 1="American Indian or Alaska Native", 
        # 2="Asian", 
        # 3="Black or African American", 
        # 4="Hispanic or Latino", 
        # 5="Native Hawaiian or Other Pacific Islander", 
        # 6="Other", 
        # 7="White or Caucasian”
        # self.race_mapping = {2: 0, 3: 1, 7: 2}
        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

        if select_group>=0:
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if self.attribute_type == 'race':
                    tmp_key = raw_data['race'].item()
                    # tmp_key = int(self.race_mapping[tmp_key])
                elif self.attribute_type == 'gender':
                    tmp_key = int(raw_data['male'].item())
                elif self.attribute_type == 'hispanic':
                    tmp_key = int(raw_data['hispanic'].item())

                if tmp_key == select_group:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        data_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(data_file, allow_pickle=True)

        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        elif self.modality_type == 'oct_bscans':

            oct_img = raw_data['oct_bscans']
            if oct_img.shape[1] != self.resolution:
                oct_img_array = []
                for img in oct_img:
                    tmp_img = resize(img, (self.resolution, self.resolution))
                    oct_img_array.append(tmp_img[None,:,:])
                oct_img = np.concatenate(oct_img_array, axis=0)
            data_sample = oct_img.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        elif self.modality_type == 'oct_bscans_3d':
            oct_img = raw_data['oct_bscans']
            data_sample = oct_img
            data_sample = data_sample[None, :, :, :]
            data_sample = data_sample.astype(int)
            data_sample = data_sample.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample)#.float()

        elif self.modality_type == 'slo_fundus':
            slo_fundus = raw_data['slo_fundus']
            slo_fundus = np.transpose(slo_fundus)
            slo_fundus = slo_fundus[None,:,:]
            if slo_fundus.shape[1] != self.resolution:
                img_array = []
                for img in slo_fundus:
                    tmp_img = resize(img, (self.resolution, self.resolution))
                    img_array.append(tmp_img[None,:,:])
                slo_fundus = np.concatenate(img_array, axis=0)
            if self.depth>1:
                slo_fundus = np.repeat(slo_fundus, self.depth, axis=0)
            data_sample = slo_fundus.astype(np.float32)        

        elif self.modality_type == 'ilm':
            ilm_sample = raw_data[self.modality_type]
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            data_sample = ilm_sample.astype(np.float32)
        elif self.modality_type == 'rnflt+ilm':
            rnflt_sample = raw_data['rnflt']
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            
            ilm_sample = raw_data['ilm']
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            
            data_sample = np.concatenate((rnflt_sample, ilm_sample), axis=0)
            data_sample = data_sample.astype(np.float32)
            
        elif self.modality_type == 'clockhours':
            data_sample = raw_data[self.modality_type].astype(np.float32)

        if self.task == 'md':
            y = torch.tensor(float(raw_data['md'].item()))
        elif self.task == 'tds':
            y = torch.tensor(float(raw_data['glaucoma'].item()))
        elif self.task == 'cls':
            y = torch.tensor(float(raw_data['glaucoma'].item()))

        attr = []
        tmp_key = raw_data['race'].item()
        if type(raw_data['race'].item()).__name__ == 'str':
            tmp_key = raw_data['race'].item()
            tmp_key = self.race_mapping[tmp_key]
        attr.append(torch.tensor(tmp_key).int())
        attr.append(torch.tensor(raw_data['male'].item()).int())
        attr.append(torch.tensor(raw_data['hispanic'].item()).int())
        

        return data_sample, y, attr

class FairVision_DR(Dataset):
    # subset: train | val | test
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0,
                    depth=3, indices=None, attribute_type='race', transform=None, needBalance=False, dataset_proportion=1., select_group=-1):

        self.data_path = data_path
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform
        self.needBalance = needBalance
        self.dataset_proportion = dataset_proportion

        self.disease_mapping = {'not.in.icd.table': 0.,
                    'no.dr.diagnosis': 0.,
                    'mild.npdr': 0.,
                    'moderate.npdr': 0.,
                    'severe.npdr': 1.,
                    'pdr': 1.}

        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

        self.data_files = find_all_files(self.data_path, suffix='npz')
        # self.data_files = self.data_files[:100]
        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]
        tmp_data_files = []
        if self.attribute_type == 'gender' or self.attribute_type == 'maritalstatus' or self.attribute_type == 'hispanic' or self.attribute_type == 'language':
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if self.attribute_type == 'gender':
                    attr = raw_data['male'].item()
                else:
                    attr = raw_data[self.attribute_type].item()
                if attr > -1:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        if self.modality_type == 'oct_bscans' or self.modality_type == 'oct_bscans_3d':
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if len(raw_data['oct_bscans']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        elif self.modality_type == 'slo_fundus':
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if len(raw_data['slo_fundus']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        # Oversampling
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        if self.subset == 'train' and self.needBalance:
            for idx in range(0, len(self.data_files)):
                data_file = os.path.join(self.data_path, self.data_files[idx])
                raw_data = np.load(data_file, allow_pickle=True)
                cur_label = raw_data[self.attribute_type].item()
                if cur_label not in self.label_samples:
                    self.label_samples[cur_label] = list()
                self.label_samples[cur_label].append(self.data_files[idx])
                self.balanced_max = len(self.label_samples[cur_label]) \
                    if len(self.label_samples[cur_label]) > self.balanced_max else self.balanced_max
            ttl_num_samples = 0
            self.class_samples_num = [0]*len(list(self.label_samples.keys()))
            for i, (k,v) in enumerate(self.label_samples.items()):
                self.class_samples_num[int(k)] = len(v)
                ttl_num_samples += len(v)
                print(f'{k}-th identity training samples: {len(v)}')
            print(f'total number of training samples: {ttl_num_samples}')
            self.class_samples_num = np.array(self.class_samples_num)

            # Oversample the classes with fewer elements than the max
            for i_label in self.label_samples:
                while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
                    self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))

            data_files = []
            for i, (k,v) in enumerate(self.label_samples.items()):
                data_files = data_files + v
            self.data_files = data_files
        
        if self.subset == 'train' and self.dataset_proportion < 1.:
            num_samples = int(len(self.data_files) * self.dataset_proportion)
            self.data_files = random.sample(self.data_files, num_samples)

        if select_group>=0:
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if self.attribute_type == 'race':
                    tmp_key = raw_data['race'].item()
                    # tmp_key = int(self.race_mapping[tmp_key])
                elif self.attribute_type == 'gender':
                    tmp_key = int(raw_data['male'].item())
                elif self.attribute_type == 'hispanic':
                    tmp_key = int(raw_data['hispanic'].item())

                if tmp_key == select_group:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        data_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(data_file, allow_pickle=True)

        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        elif self.modality_type == 'oct_bscans':

            oct_img = raw_data['oct_bscans']
            if oct_img.shape[1] != self.resolution:
                oct_img_array = []
                for img in oct_img:
                    # oct_img_array.append(resize(img, (self.resolution, self.resolution)))
                    tmp_img = resize(img, (self.resolution, self.resolution))
                    oct_img_array.append(tmp_img[None,:,:])
                oct_img = np.concatenate(oct_img_array, axis=0)
            # data_sample = np.stack(oct_img_array, axis=0)
            data_sample = oct_img.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        elif self.modality_type == 'oct_bscans_3d':
            oct_img = raw_data['oct_bscans']
            data_sample = oct_img
            data_sample = data_sample[None, :, :, :]
            data_sample = data_sample.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample)#.float()

        elif self.modality_type == 'slo_fundus':
            slo_fundus = raw_data['slo_fundus']
            slo_fundus = np.transpose(slo_fundus)
            slo_fundus = slo_fundus[None,:,:]
            if slo_fundus.shape[1] != self.resolution:
                img_array = []
                for img in slo_fundus:
                    tmp_img = resize(img, (self.resolution, self.resolution))
                    img_array.append(tmp_img[None,:,:])
                slo_fundus = np.concatenate(img_array, axis=0)
            if self.depth>1:
                slo_fundus = np.repeat(slo_fundus, self.depth, axis=0)
            data_sample = slo_fundus.astype(np.float32)        

        elif self.modality_type == 'ilm':
            ilm_sample = raw_data[self.modality_type]
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            data_sample = ilm_sample.astype(np.float32)
        elif self.modality_type == 'rnflt+ilm':
            rnflt_sample = raw_data['rnflt']
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            
            ilm_sample = raw_data['ilm']
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)

            data_sample = np.concatenate((rnflt_sample, ilm_sample), axis=0)
            data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'clockhours':
            data_sample = raw_data[self.modality_type].astype(np.float32)

        if self.task == 'md':
            y = torch.tensor(float(raw_data['md'].item()))
        elif self.task == 'tds':
            y = torch.tensor(float(raw_data['glaucoma'].item()))
        elif self.task == 'cls':
            y = torch.tensor(self.disease_mapping[raw_data['dr_subtype'].item()])

        attr = []
        tmp_key = raw_data['race'].item()
        if type(raw_data['race'].item()).__name__ == 'str':
            tmp_key = raw_data['race'].item()
            tmp_key = self.race_mapping[tmp_key]
        attr.append(torch.tensor(tmp_key).int())
        attr.append(torch.tensor(raw_data['male'].item()).int())
        attr.append(torch.tensor(raw_data['hispanic'].item()).int())
        

        return data_sample, y, attr

class FairVision_AMD(Dataset):
    # subset: train | val | test
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0,
                    depth=3, indices=None, attribute_type='race', transform=None, needBalance=False, dataset_proportion=1., select_group=-1):

        self.data_path = data_path
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform
        self.needBalance = needBalance
        self.dataset_proportion = dataset_proportion

        self.disease_mapping = {'not.in.icd.table': 0.,
                        'no.amd.diagnosis': 0.,
                        'early.dry': 1.,
                        'intermediate.dry': 2.,
                        'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
                        'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
                        'wet.amd.active.choroidal.neovascularization': 3.,
                        'wet.amd.inactive.choroidal.neovascularization': 3.,
                        'wet.amd.inactive.scar': 3.}

        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

        self.data_files = find_all_files(self.data_path, suffix='npz')
        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]
        tmp_data_files = []
        if self.attribute_type == 'gender' or self.attribute_type == 'maritalstatus' or self.attribute_type == 'hispanic' or self.attribute_type == 'language':
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if self.attribute_type == 'gender':
                    attr = raw_data['male'].item()
                else:
                    attr = raw_data[self.attribute_type].item()
                if attr > -1:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        if self.modality_type == 'oct_bscans' or self.modality_type == 'oct_bscans_3d':
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if len(raw_data['oct_bscans']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        elif self.modality_type == 'slo_fundus':
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if len(raw_data['slo_fundus']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        # Oversampling
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        if self.subset == 'train' and self.needBalance:
            for idx in range(0, len(self.data_files)):
                data_file = os.path.join(self.data_path, self.data_files[idx])
                raw_data = np.load(data_file, allow_pickle=True)
                cur_label = raw_data[self.attribute_type].item()
                if cur_label not in self.label_samples:
                    self.label_samples[cur_label] = list()
                self.label_samples[cur_label].append(self.data_files[idx])
                self.balanced_max = len(self.label_samples[cur_label]) \
                    if len(self.label_samples[cur_label]) > self.balanced_max else self.balanced_max
            ttl_num_samples = 0
            self.class_samples_num = [0]*len(list(self.label_samples.keys()))
            for i, (k,v) in enumerate(self.label_samples.items()):
                self.class_samples_num[int(k)] = len(v)
                ttl_num_samples += len(v)
                print(f'{k}-th identity training samples: {len(v)}')
            print(f'total number of training samples: {ttl_num_samples}')
            self.class_samples_num = np.array(self.class_samples_num)

            # Oversample the classes with fewer elements than the max
            for i_label in self.label_samples:
                while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
                    self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))

            data_files = []
            for i, (k,v) in enumerate(self.label_samples.items()):
                data_files = data_files + v
            self.data_files = data_files
        
        if self.subset == 'train' and self.dataset_proportion < 1.:
            num_samples = int(len(self.data_files) * self.dataset_proportion)
            self.data_files = random.sample(self.data_files, num_samples)

        if select_group>=0:
            tmp_data_files = []
            for x in self.data_files:
                data_file = os.path.join(self.data_path, x)
                raw_data = np.load(data_file, allow_pickle=True)
                if self.attribute_type == 'race':
                    tmp_key = raw_data['race'].item()
                    # tmp_key = int(self.race_mapping[tmp_key])
                elif self.attribute_type == 'gender':
                    tmp_key = int(raw_data['male'].item())
                elif self.attribute_type == 'hispanic':
                    tmp_key = int(raw_data['hispanic'].item())

                if tmp_key == select_group:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        data_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(data_file, allow_pickle=True)

        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        elif self.modality_type == 'oct_bscans':

            oct_img = raw_data['oct_bscans']
            if oct_img.shape[1] != self.resolution:
                oct_img_array = []
                for img in oct_img:
                    tmp_img = resize(img, (self.resolution, self.resolution))
                    oct_img_array.append(tmp_img[None,:,:])
                oct_img = np.concatenate(oct_img_array, axis=0)
            data_sample = oct_img.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        elif self.modality_type == 'oct_bscans_3d':
            oct_img = raw_data['oct_bscans']
            data_sample = oct_img
            data_sample = data_sample[None, :, :, :]
            data_sample = data_sample.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample)#.float()

        elif self.modality_type == 'slo_fundus':
            slo_fundus = raw_data['slo_fundus']
            slo_fundus = np.transpose(slo_fundus)
            slo_fundus = slo_fundus[None,:,:]
            

            if slo_fundus.shape[1] != self.resolution:
                # img_array = []
                img_array = []
                for img in slo_fundus:
                    tmp_img = resize(img, (self.resolution, self.resolution))
                    img_array.append(tmp_img[None,:,:])
                slo_fundus = np.concatenate(img_array, axis=0)
            
            if self.depth>1:
                slo_fundus = np.repeat(slo_fundus, self.depth, axis=0)
            data_sample = slo_fundus.astype(np.float32)        
            
        elif self.modality_type == 'ilm':
            ilm_sample = raw_data[self.modality_type]
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            data_sample = ilm_sample.astype(np.float32)
        elif self.modality_type == 'rnflt+ilm':
            rnflt_sample = raw_data['rnflt']
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            
            ilm_sample = raw_data['ilm']
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)

            data_sample = np.concatenate((rnflt_sample, ilm_sample), axis=0)
            data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'clockhours':
            data_sample = raw_data[self.modality_type].astype(np.float32)

        if self.task == 'md':
            y = torch.tensor(float(raw_data['md'].item()))
        elif self.task == 'tds':
            y = torch.tensor(float(raw_data['glaucoma'].item()))
        elif self.task == 'cls':
            y = torch.tensor(self.disease_mapping[raw_data['amd_condition'].item()])

        attr = []
        tmp_key = raw_data['race'].item()
        if type(raw_data['race'].item()).__name__ == 'str':
            tmp_key = raw_data['race'].item()
            tmp_key = self.race_mapping[tmp_key]
        attr.append(torch.tensor(tmp_key).int())
        attr.append(torch.tensor(raw_data['male'].item()).int())
        attr.append(torch.tensor(raw_data['hispanic'].item()).int())
        

        return data_sample, y, attr
