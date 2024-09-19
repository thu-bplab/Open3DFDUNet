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
import torch.nn as nn
import torch.nn.functional as F

# Define the Convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm3d(out_channels)
        # self.silu = nn.SiLU()
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.4) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

# Define the Dense Block
class DenseBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock3D, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=1)
        self.conv2 = ConvBlock(in_channels, in_channels // 2, kernel_size=3, dropout=True)
        self.conv3 = ConvBlock(in_channels * 3 // 2, in_channels, kernel_size=1)
        self.conv4 = ConvBlock(in_channels, in_channels // 2, kernel_size=3, dropout=True)

    def forward(self, x):
        # black line: Apply the first convolution
        x1 = self.conv1(x)
        
        # blue line: Apply the second convolution
        x2 = self.conv2(x1)
        
        # orange line: Concatenate the outputs with the input
        x3 = torch.cat([x, x2], 1)

        # black line: Apply the third convolution 
        x4 = self.conv3(x3) 

        # blue line: Apply the fourth convolution
        x5 = self.conv4(x4)

        # orange line: Concatenate the outputs with the input
        out = torch.cat([x3, x5], 1)
        return out

class TransitionDown3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        # self.silu = nn.SiLU()
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        return self.elu(self.norm(self.conv(x)))

class TransitionUp3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp3D, self).__init__()
        self.conv_trans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        # self.silu = nn.SiLU()
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        return self.elu(self.norm(self.conv_trans(x)))

class FullyDenseUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FullyDenseUNet3D, self).__init__()
        
        # first light purple line: 3x3x3 Conv + BN + ELU (replaced by SiLU) 
        self.init_conv = ConvBlock(in_channels, 8, kernel_size=3)

        # dark orange line: First Dense Block
        self.dense_block1 = DenseBlock3D(8)
        # green line: First Down Block, transition down
        self.trans_down1 = TransitionDown3D(16, 16)

        # Second dense block and transition down
        self.dense_block2 = DenseBlock3D(16)
        self.trans_down2 = TransitionDown3D(32, 32)

        # Third dense block and transition down
        self.dense_block3 = DenseBlock3D(32)
        self.trans_down3 = TransitionDown3D(64, 64)

        # Fourth dense block
        self.dense_block4 = DenseBlock3D(64)

        # Transition up and fourth dense block
        self.trans_up1 = TransitionUp3D(128, 64)
        # first black line: 1x1x1 Conv 
        self.one_conv1 = nn.Conv3d(128, 32, kernel_size=1, bias=False) 
        self.dense_block5 = DenseBlock3D(32)  # Concatenated with skip connection

        # Transition up and fifth dense block
        self.trans_up2 = TransitionUp3D(64, 32)
        # second black line: 1x1x1 Conv 
        self.one_conv2 = nn.Conv3d(64, 16, kernel_size=1, bias=False) 
        self.dense_block6 = DenseBlock3D(16)  # Concatenated with skip connection

        # Transition up and sixth dense block
        self.trans_up3 = TransitionUp3D(32, 16)
        # thrid black line: 1x1x1 Conv 
        self.one_conv3 = nn.Conv3d(32, 16, kernel_size=1, bias=False) 
        self.dense_block7 = DenseBlock3D(16)  # Concatenated with skip connection

        # second light purple line: 3x3x3 Conv, Additional convolution at the end of decoding
        self.refinement_conv = ConvBlock(32, 16, kernel_size=3)

        # grey line: Final convolution
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1, bias=False)

        # Residual connection
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Initial convolution
        x_init = self.init_conv(x)

        # Encoder path
        out_dense1 = self.dense_block1(x_init)
        out_trans1 = self.trans_down1(out_dense1)

        out_dense2 = self.dense_block2(out_trans1)
        out_trans2 = self.trans_down2(out_dense2)

        out_dense3 = self.dense_block3(out_trans2)
        out_trans3 = self.trans_down3(out_dense3)

        out_dense4 = self.dense_block4(out_trans3)

        # Decoder path
        out_up1 = self.trans_up1(out_dense4)
        out_up1 = torch.cat([out_up1, out_dense3], 1)  # Skip connection from trans_down2
        out_one_conv1 = self.one_conv1(out_up1)
        out_dense5 = self.dense_block5(out_one_conv1)

        out_up2 = self.trans_up2(out_dense5)
        out_up2 = torch.cat([out_up2, out_dense2], 1)  # Skip connection from trans_down1
        out_one_conv2 = self.one_conv2(out_up2)
        out_dense6 = self.dense_block6(out_one_conv2)

        out_up3 = self.trans_up3(out_dense6)
        out_up3 = torch.cat([out_up3, out_dense1], 1)  # Skip connection from init_conv
        out_one_conv3 = self.one_conv3(out_up3)
        out_dense7 = self.dense_block7(out_one_conv3)

        # Refinement path
        out_refinement = self.refinement_conv(out_dense7)

        # Final convolution
        out_final = self.final_conv(out_refinement)

        # Add the residual image from the input
        out_residual = self.residual_conv(x)
        out_final = out_final + out_residual

        return out_final

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullyDenseUNet3D().to(device)

model = nn.DataParallel(model)

# model.to("cuda:0")

input_tensor = torch.randn(32, 1, 128, 128, 128).to(device)

output = model(input_tensor)
'''