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


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
import torch
import h5py
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from models import FullyDenseUNet3D
from datasets.dataset import NiftiPairImageGenerator
from eval.eval_nii import Evaluator, RMSE, PSNR, SSIM2D, SSIM3D

def reconstruct_from_blocks(blocks, num_batches, original_shape):
    """
    Reassemble the cubes into the original size of 3D image
    :param blocks: Tensor containing all the small cubes
    :param num_batches: The quantity of each batch
    :param original_shape: The shape of the original image
    :return: Reconstructed image
    """
    reconstructed_images = []
    block_size = 128
    blocks_per_image = 8  # 每张图像被分成8个块

    for i in range(0, blocks.shape[0], blocks_per_image * num_batches):
        reconstructed = torch.zeros((num_batches,) + original_shape)  # 创建空图像

        for b in range(num_batches):
            index = 0
            for z in range(2):
                for y in range(2):
                    for x in range(2):
                        z_start, z_end = z * block_size, (z + 1) * block_size
                        y_start, y_end = y * block_size, (y + 1) * block_size
                        x_start, x_end = x * block_size, (x + 1) * block_size
                        reconstructed[b, :, z_start:z_end, y_start:y_end, x_start:x_end] = blocks[i + b * blocks_per_image + index]
                        index += 1
            reconstructed_images.append(reconstructed[b])

    return torch.stack(reconstructed_images, dim=0)

def save_as_hdf5(data, filename):
    
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('dataset', data=data.numpy())


def main(args):
    # Configs
    dataroot = args.dataroot
    cate = args.cate
    batch_size = args.batch_size
    model_path = args.model_path
    cuda = args.cuda
    dp = args.dp
    results_dir = args.results_dir

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    # Init model
    model = FullyDenseUNet3D(in_channels=1, out_channels=1)
    # DataParallel: Multi GPU
    if dp:
        model = nn.DataParallel(model)
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dataloader
    print("Initializing Test Datasets and DataLoaders...")
    inputfolder = os.path.join(dataroot, cate)
    transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.unsqueeze(0)),
    ]) 
    # Test dataset
    test_dataset = NiftiPairImageGenerator(
        inputfolder,
        dataset_type='test',
        apply_transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=40)

    # Evaluation
    evaluator = Evaluator(model, RMSE(), PSNR(), SSIM2D(), SSIM3D(window_size=11), device) 
    test_rmse, test_psnr, test_ssim2d, test_ssim3d = evaluator.evaluate(test_dataloader)
    print(f"Test Results: RMSE: {test_rmse}, PSNR: {test_psnr}, SSIM2D: {test_ssim2d}, SSIM3D: {test_ssim3d}")

    # Save quantitative test results
    if results_dir != '':
        test_results = {
            'test_rmse': test_rmse, 
            'test_psnr': test_psnr,
            'test_ssim2d': test_ssim2d,
            'test_ssim3d': test_ssim3d
        }
        df = pd.DataFrame([test_results])
        df.to_csv(f'{results_dir}/test_results.csv', index=False)
    
    # List to store all outputs
    outputs_list = []

    # Testing loop
    for batch in test_dataloader:
        # Input test data
        sparse_inputs = batch['input'].to(device).float()

        # To do the inference
        with torch.no_grad():
            outputs = model(sparse_inputs)

        # Add the output to the list
        outputs_list.append(outputs.cpu())

    # Combine all outputs into one Tensor
    all_outputs = torch.cat(outputs_list, dim=0)
    
    # Reconstruct the output cubes to its original size
    recon_batch_size = 1
    reconstructed_image = reconstruct_from_blocks(all_outputs, recon_batch_size, (1, 256, 256, 256))

    # Save the reconstructed 3D images as an HDF5 file
    if results_dir != '':
        save_as_hdf5(reconstructed_image, f'{results_dir}/reconstructed_image_best.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Medical Image Testing Script')
    parser.add_argument('--dataroot', type=str, default='/data/bml/Dataset', help='root directory of the dataset')
    parser.add_argument('--cate', type=str, default='vessel', help='category of the input dataset')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model file')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
    parser.add_argument('--dp',action='store_true',default=True)
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to save test results (default: none)')

    args = parser.parse_args()
    main(args)
