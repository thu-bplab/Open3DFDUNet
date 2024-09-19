# Copyright (c) 2024-2025, Di Kong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#-*- coding:utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import re
import os

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            dataset_type: str,
            apply_transform=None,
        ):
        self.input_folder = input_folder
        self.sparse_image = os.path.join(input_folder, dataset_type, 'sparse_image')
        self.gt_image = os.path.join(input_folder, dataset_type, 'gt_image')
        self.pair_files = self.pair_file()
        self.transform = apply_transform

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.sparse_image, '*')))
        target_files = sorted(glob(os.path.join(self.gt_image, '*')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)

        if self.transform is not None:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)


        return {'input':input_img, 'target':target_img}

def main():
    transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.unsqueeze(0)),
    ])
    dataset = NiftiPairImageGenerator(input_folder='/data/bml/KD/Dataset/brain', dataset_type='train', apply_transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(len(dataset)) 
    
    for batch in dataloader:
            sparse = batch['input']
            gt = batch['target']
            print(sparse.shape)
            print(gt.shape)

if __name__ == '__main__':
    main()