import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from . import regist_loss


eps = 1e-6

# ============================ #
#  Self-reconstruction loss    #
# ============================ #

# def calculate_var(output,data):
#     tensor_len=output.shape[0]
#     aa=torch.zeros(tensor_len)
    
    
#     for i in range(output.shape[0]):
#         aa[i]=(torch.var(output[i,0,:,:]))
#     mm=torch.mean(aa)
#     return mm
def wave_img(self,sigMat,denoised,row_sums):
    mm_shape = torch.tensor(sigMat.shape[-1])
    for i in range(sigMat.shape[0]):
        if row_sums[i]>mm_shape*3/10 or row_sums[i]<mm_shape*(-3)/10:
            x = torch.arange(0, mm_shape).cpu().numpy()
            y = sigMat[i,3,0,:].cpu().numpy()
            y1 = sigMat[i,3,1,:].cpu().numpy()
            y2 = sigMat[i,3,2,:].cpu().numpy()
            y3= denoised[i,3,0,:].cpu().numpy()


def gaussian_kernel(size, center_index, sigma=1):
    kernel = torch.zeros(size)
    center = center_index  # 中心位置
    for i in range(size[1]):
        distance = torch.abs(i - center)
        kernel[0,i] = torch.exp(-(distance)**2 / (2 * sigma**2))
    return kernel
    
def calculate_var(output,data):
    tensor_len=output.shape[2]
    data_index=torch.zeros(tensor_len)
    aa=torch.zeros(tensor_len)
    for i in range(output.shape[0]):
        aa[i]=(torch.var(output[i,0,:,:]))
    mm = torch.mean(aa)
    return mm

def different_2transmiss_loss(output):
    cf_num = (torch.sum(output,dim=(1,2,3)))
    cf_num =cf_num **2
    a=output.shape[-1]**2*2
    b=torch.sum(torch.abs(output)**2,dim=(1,2,3))
    cf_den = a * b
    transmiss_loss = cf_num /cf_den
    transmiss_loss = 1-torch.mean(transmiss_loss)
    transmiss_loss*=10
    return transmiss_loss

def different_3transmiss_loss(output):
    variance_loss = torch.var(output, dim=(0,1), keepdim=True)
    variance_loss = torch.sum(variance_loss)
    transmiss_loss = F.l1_loss(output[:,0,:,:]-output[:,1,:,:])+F.l1_loss(output[:,1,:,:]-output[:,2,:,:])+0.5*F.l1_loss(output[:,2,:,:]-output[:,3,:,:])
    return variance_loss



def cf_loss(output):
    cf_num = (torch.sum(output,dim=(1,2,3)))
    cf_num =cf_num **2

    a=output.shape[-1]
    b=torch.sum(torch.abs(output)**2,dim=(1,2,3))
    cf_den = a * b

    cf_factor = cf_num /cf_den
    cf_factor = torch.mean(cf_factor)
    return (1-cf_factor )


@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module,epoch):
        output = model_output['recon'][:,:,:,:]
        target_noisy = data['noisy_img_wave_label'][:,3:4,:,:]
        loss=F.l1_loss(output, target_noisy)
        
            
        return loss

@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon'][:,3:4,:,:]
        output=calculate_var(output,data)
        target_noisy = data['noisy_img_wave_label'] 
        target_noisy=calculate_var(target_noisy,data)
        print("pre_var:",target_noisy,"   post_var",output,"   sub_var",target_noisy-output)
        return (target_noisy-output) 
@regist_loss
class cf_num():
    def __call__(self, input_data, model_output, data, module,epoch):
        output = model_output['recon'][:,:,:,:]
        output = output[:,0,0,:]
        cf_num = (torch.sum(output,dim=(1)))
        cf_num =cf_num **2
        a=output.shape[-1]//2
        b=torch.sum(torch.abs(output)**2,dim=(1))
        cf_den = a * b
        cf_factor = cf_num /cf_den
        cf_factor = torch.mean(cf_factor)
        
        return 1-cf_factor

@regist_loss
class cf_num_1():
    def __call__(self, input_data, model_output, data, module,epoch):
        
        output = model_output['recon'][:,3:4,:,:]
        target_noisy = data['noisy_img_wave_label'][:,3:4,:,:]
        loss1 = F.l1_loss(output, target_noisy) 
        cf_factor = loss1
        return cf_factor 
        
        
@regist_loss
class cf_num_2():
    def __call__(self, input_data, model_output, data, module,epoch):
        output = model_output['recon'][:,1:2,:,:]
        cf_num = (torch.sum(output,dim=(1,2,3)))
        cf_num =cf_num **2
        a=output.shape[1]*output.shape[2]*output.shape[3]
        b=torch.sum(torch.abs(output)**2,dim=(1,2,3))
        cf_den = a * b
        cf_factor = cf_num /cf_den
        cf_factor = torch.mean(cf_factor)
        return cf_factor

