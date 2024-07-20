
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import regist_model
from .DBSNl import DBSNl
from .DBSNl import DBSN_gaussian


@regist_model
class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, 
                    bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a    = pd_a
        self.pd_b    = pd_b
        self.pd_pad  = pd_pad
        self.R3      = R3
        self.R3_T    = R3_T
        self.R3_p    = R3_p
        
        # define network
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, 1, bsn_base_ch, bsn_num_module)
            
        else:
            raise NotImplementedError('bsn %s is not implemented'%bsn)

    def forward(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
       
        pd_img_denoised= self.bsn(img)

        return pd_img_denoised

    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        b,c,h,w = x.shape

        img_pd_bsn= self.forward(img=x, pd=self.pd_b)
        
        
        return img_pd_bsn

        
   