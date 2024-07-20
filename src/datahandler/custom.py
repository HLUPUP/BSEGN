import os

import h5py

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset
import re
import numpy as np
import scipy




@regist_dataset
class CustomSample(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # check if the dataset exists
        self.dataset_path = './dataset/real_data_5bin'
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path

        # WRITE YOUR CODE FOR SCANNING DATA
        # example:
        for root, _, files in os.walk(self.dataset_path):
            
            for file_name in files:
                
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        # WRITE YOUR CODE FOR LOADING DATA FROM DATA INDEX
        # example:
        file_name = self.img_paths[data_idx]
        
        noisy_img_wave = self._load_wave( file_name)
        noisy_img_wave_label= noisy_img_wave[:,:,:]
        noisy_img_wave = noisy_img_wave[:,:,:]
        return {'real_noisy': noisy_img_wave,'noisy_img_wave_label':noisy_img_wave_label} # only noisy image dataset


    
