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


import random
import time
import datetime
import sys
import torch
import numpy as np
from torch.autograd import Variable
from visdom import Visdom

    
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_epochs, decay_factors):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_epochs = decay_epochs
        self.decay_factors = decay_factors

    def step(self, epoch):
        if epoch < self.decay_epochs[0]:
            return 1.0
        elif epoch < self.decay_epochs[1]:
            return self.decay_factors[0]
        else:
            return self.decay_factors[1]
