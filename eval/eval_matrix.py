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


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from math import exp

def normalize_data(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_true, y_pred):
        
        mse = F.mse_loss(y_true, y_pred)
        rmse_value = torch.sqrt(mse)
        return rmse_value

# PSNR for val and test, for 3D image super-resolution
class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, inputs, targets, max_pixel=1.0):
        mse = F.mse_loss(inputs, targets)
        return 20 * torch.log10(max_pixel / torch.sqrt(mse))

# SSIM2D for val and test, for 3D image super-resolution
class SSIM2D(nn.Module):
    def __init__(self):
        super(SSIM2D, self).__init__()

    def forward(self, inputs, targets, data_range):
        return ssim(inputs, targets, data_range=data_range)

class Evaluator:
    def __init__(self, model, criterion_mse, criterion_psnr, criterion_ssim, criterion_ssim3d, device):
        self.model = model
        self.criterion_mse = criterion_mse
        self.criterion_psnr = criterion_psnr
        self.criterion_ssim = criterion_ssim
        self.criterion_ssim3d = criterion_ssim3d
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        final_total_ssim3d = 0
        final_total_ssim = 0
        final_total_psnr = 0

        with torch.no_grad():
            for batch in dataloader:
                sparse_inputs = batch['input'].to(self.device).float()
                gt = batch['target'].to(self.device).float()

                outputs = self.model(sparse_inputs)
                # Normalized to [0, 1]
                outputs = normalize_data(outputs)
                gt_norm = normalize_data(gt)
                mse_loss = self.criterion_mse(outputs, gt_norm)
                total_loss += mse_loss.item()

                final_total_ssim3d += self.criterion_ssim3d(outputs, gt_norm).item()

                total_ssim = 0
                total_psnr = 0
                for i in range(outputs.shape[0]):
                    total_psnr += self.criterion_psnr(outputs[i], gt_norm[i]).item()

                    ssim_sum = 0
                    for j in range(outputs.shape[2]):  # Iterate through each slice
                        output_slice = outputs[i, 0, j].cpu().numpy()
                        gt_slice = gt_norm[i, 0, j].cpu().numpy()
                        ssim_val = ssim(output_slice, gt_slice, data_range=1.0)
                        ssim_sum += ssim_val
                    
                    total_ssim += ssim_sum / outputs.shape[2]
                
                final_total_ssim += total_ssim / outputs.shape[0]
                final_total_psnr += total_psnr / outputs.shape[0] 
        
        avg_loss = total_loss / len(dataloader)
        avg_ssim3d = final_total_ssim3d / len(dataloader)
        avg_psnr = final_total_psnr / len(dataloader)
        avg_ssim = final_total_ssim / len(dataloader)

        return avg_loss, avg_psnr, avg_ssim, avg_ssim3d

# SSIM3D for val and test, for 3D image super-resolution
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM3D(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)
