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
import time
import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from torch.optim import Adam, AdamW
from models import FullyDenseUNet3D
from datasets.dataset import NiftiPairImageGenerator
from losses.losses import MSELoss
from eval.eval_nii import Evaluator, RMSE, PSNR, SSIM2D, SSIM3D
from utils import LambdaLR

def main(args):
    print("Starting training session with configurations:")
    print(args)

    # Configs
    epochs = args.epochs
    decay_epochs = args.decay_epochs
    decay_factors = args.decay_factors
    eval_epoch = args.eval_epoch
    batch_size = args.batch_size
    dataroot = args.dataroot
    cate = args.cate
    lr = args.lr
    weight_decay = args.weight_decay
    results_dir = args.results_dir
    dp = args.dp
    cuda = args.cuda

    # Preparation and store the hyperparameters
    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if results_dir == '':
        results_dir = 'output/cache-'+datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"No results_dir provided, using default directory: {results_dir}")

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    with open(results_dir+'/args.json','w') as fid:
        json.dump(args.__dict__,fid,indent=2)
        print(f"Training configurations saved to {results_dir}/args.json")

    # TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=results_dir)
    print(f"TensorBoard SummaryWriter initialized at {results_dir}")

    # Dataloader
    print("Initializing Datasets and DataLoaders...")
    inputfolder = os.path.join(dataroot, cate)
    transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.unsqueeze(0)),
    ]) 
    # train
    train_dataset = NiftiPairImageGenerator(
        inputfolder,
        dataset_type='train',
        apply_transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=40)
    # val
    val_dataset = NiftiPairImageGenerator(
        inputfolder,
        dataset_type='val',
        apply_transform=transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
    print(len(train_dataset))
    
    # Model Init
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyDenseUNet3D(in_channels=1, out_channels=1).to(device)

    # DataParallel: Multi GPU
    if dp:
        model = nn.DataParallel(model)
        print("DataParallel mode enabled for multi-GPU training.")

    # Losses
    criterion_mse = MSELoss()
    
    # Evaluation Metrics
    evaluator = Evaluator(model, RMSE(), PSNR(), SSIM2D(), SSIM3D(window_size=11), device) 
    print("Evaluation metrics initialized: RMSE, PSNR, SSIM2D, SSIM3D.")

    # Optimize & LR schedulers
    # optimizer = Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    # Default: beta1=0.9, beta2=0.999
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(epochs, 0, decay_epochs, decay_factors).step)

    # Record
    results = {'valid_rmse':[], 'valid_psnr':[], 'valid_ssim2d':[], 'valid_ssim3d':[]}
    best_val_rmse = float('inf')
    best_epoch = 0
    global_step = 0

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()

        print(f"Epoch [{epoch+1}/{epochs}] - Training...")
        for batch in train_dataloader:
            sparse_inputs = batch['input'].to(device).float()
            gt = batch['target'].to(device).float()

            optimizer.zero_grad()
            # Forward
            outputs = model(sparse_inputs)
            # Cal loss
            loss_mse = criterion_mse(outputs, gt)
            loss = loss_mse

            # Backward
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], L2Loss: {loss_mse.item()}")

            # TensorBoard: Record the training loss of each Step
            writer.add_scalar('L2Loss/train', loss_mse.item(), global_step)
            # Update global step
            global_step += 1


        # val every {eval_epoch} number epochs
        if (epoch + 1) % eval_epoch == 0:
            val_rmse, val_psnr, val_ssim2d, val_ssim3d = evaluator.evaluate(val_dataloader)
            print(f"Validation - Epoch {epoch+1}: RMSE: {val_rmse}, PSNR: {val_psnr}, SSIM2D: {val_ssim2d}, SSIM3D: {val_ssim3d}")

            # Record Validation Results
            results['valid_rmse'].append(val_rmse)
            results['valid_psnr'].append(val_psnr)
            results['valid_ssim2d'].append(val_ssim2d)
            results['valid_ssim3d'].append(val_ssim3d)

            # Calculate number of validations performed so far
            num_validations = (epoch + 1) // eval_epoch
            data_frame = pd.DataFrame(data=results, index=range(0, num_validations))
            data_frame.to_csv(results_dir+'/log.csv', index_label='epoch')

            # TensorBoard: Record verification indicators
            writer.add_scalar('Metric/RMSE', val_rmse, epoch)
            writer.add_scalar('Metric/PSNR', val_psnr, epoch)
            writer.add_scalar('Metric/SSIM2D', val_ssim2d, epoch)
            writer.add_scalar('Metric/SSIM3D', val_ssim3d, epoch)

            # Save the Intermediate Model
            weights_dir = os.path.join(results_dir, 'weights')
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            torch.save(model.state_dict(), f'{weights_dir}/model_epoch_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}")

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), f'{weights_dir}/model_best.pth')
                print(f"Best model saved at epoch {epoch+1} with Validation Loss: {val_rmse}")

        # Update learning rates
        lr_scheduler.step()
        
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] Current Learning Rate: {current_lr}")

    writer.close()
    print(f"Training completed. Best model was at epoch {best_epoch + 1} with Validation Loss: {best_val_rmse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Medical Image Training Script')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--decay_epochs', nargs='+', type=int, default=[400, 500], help='Epochs to decay learning rate')
    parser.add_argument('--decay_factors', nargs='+', type=float, default=[0.5, 0.1], help='Factors to decay learning rate')
    parser.add_argument('--eval_epoch', type=int, default=2, help='Eval after every number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--dataroot', type=str, default='/data/bml/Dataset', help='root directory of the dataset')
    parser.add_argument('--cate', type=str, default='vessel', help='category of the input dataset')
    
    ### training options
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=0.2, help="minimal learning rate ratio")
    parser.add_argument('--weight_decay', type=float, default=0.0)
    
    parser.add_argument('--dp',action='store_true',default=True)
    parser.add_argument('--cuda', action='store_true',default=True, help='use GPU computation')
    parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
    parser.add_argument('--alpha', type=float, default=0.01, help='weight for the L1 Loss in the combined loss function')

    args = parser.parse_args()
    main(args)
    