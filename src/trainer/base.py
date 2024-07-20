import os
import math
import time, datetime

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ..util.dnd_submission.bundle_submissions import bundle_submissions_srgb
from ..util.dnd_submission.dnd_denoise import denoise_srgb
from ..util.dnd_submission.pytorch_wrapper import pytorch_denoiser

from ..loss import Loss
from ..datahandler import get_dataset_class
from ..util.file_manager import FileManager
from ..util.logger import Logger
from ..util.util import human_format, np2tensor, rot_hflip_img, psnr, ssim, tensor2np, imread_tensor 
from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
import h5py
import PIL.Image as img

#builduct
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import os
from scipy.signal import firwin, filtfilt
import time
import PIL.Image as img
from val_img import val_img
import torchvision.models as models
import copy
from scipy.io import loadmat
from scipy.io import savemat
import random
import torch.nn.functional as F
import scipy.io as sio



status_len = 13
mean_cf_factor_noise = 0
mean_cf_factor_denoise = 0
mean_std_denoise = 0
mean_std_noise = 0
cf_count = 0
cf_count_noise = 0
cf_count_denoise = 0


class BaseTrainer():
    '''
    Base trainer class to implement other trainer classes.
    below function should be implemented in each of trainer class.
    '''
    def test(self):
        raise NotImplementedError('define this function for each trainer')
    def validation(self):
        raise NotImplementedError('define this function for each trainer')
    def _set_module(self):
        # return dict form with model name.
        raise NotImplementedError('define this function for each trainer')
    def _set_optimizer(self):
        # return dict form with each coresponding model name.
        raise NotImplementedError('define this function for each trainer')
    def _forward_fn(self, module, loss, data):
        # forward with model, loss function and data.
        # return output of loss function.
        raise NotImplementedError('define this function for each trainer')

    #----------------------------#
    #    Train/Test functions    #
    #----------------------------#   
    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        # get file manager and logger class
        self.file_manager = FileManager(self.session_name)
        self.logger = Logger()
        
        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg   = cfg['validation']
        self.test_cfg  = cfg['test']
        self.ckpt_cfg  = cfg['checkpoint']

    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 1 and self.train_cfg['warmup']:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch+1):
            self._before_epoch()
            # if self.epoch<12:
            #     for param in self.model['denoiser'].module.bsn_gaussian.parameters():
            #         param.requires_grad = False
            #     # for gpu_model in self.model['denoiser'].children():
            #     #     for name, param in gpu_model.named_parameters():
            #     #         print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')
            # else :
            #     for param in self.model['denoiser'].module.bsn.parameters():
            #         param.requires_grad = False
            #     for param in self.model['denoiser'].module.bsn_gaussian.parameters():
            #         param.requires_grad = True
                
                # for gpu_model in self.model['denoiser'].children():
                #     for name, param in gpu_model.named_parameters():
                #         print(f'Parameter: {name}, Requires Gradient: {param.requires_grad}')

            self._run_epoch()
            
            self._after_epoch()
        
        self._after_train()

    def _warmup(self):
        self._set_status('warmup')

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.max_iter:
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
                % (warmup_iter, self.max_iter))
            warmup_iter = self.max_iter

        for self.iter in range(1, warmup_iter+1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()

    def _before_test(self, dataset_load):
        # initialing
        self.module = self._set_module()
        self._set_status('test')

        # load checkpoint file
        ckpt_epoch = self._find_last_epoch() if self.cfg['ckpt_epoch'] == -1 else self.cfg['ckpt_epoch']
        ckpt_name  = self.cfg['pretrained'] if self.cfg['pretrained'] is not None else None
        self.load_checkpoint(ckpt_epoch, name=ckpt_name)
        self.epoch = self.cfg['ckpt_epoch'] # for print or saving file name.

        # test dataset loader
        if dataset_load:
            self.test_dataloader = self._set_dataloader(self.test_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}
        

        # evaluation mode and set status
        self._eval_mode()
        self._set_status('test %03d'%self.epoch)

        # start message
        self.logger.highlight(self.logger.get_start_msg())

        # set denoiser
        self._set_denoiser()
        
        # wrapping denoiser w/ self_ensemble
        if self.cfg['self_en']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.self_ensemble(denoiser_fn, *input_data)

        # wrapping denoiser w/ crop test
        if 'crop' in self.cfg['test']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.crop_test(denoiser_fn, *input_data, size=self.cfg['test']['crop'])
            
    def _before_train(self):
        # cudnn
        torch.backends.cudnn.benchmark = False

        # initialing
        self.module = self._set_module()
        


        # training dataset loader
        self.train_dataloader = self._set_dataloader(self.train_cfg, batch_size=self.train_cfg['batch_size'], shuffle=True, num_workers=self.cfg['thread'])

        # validation dataset loader
        if self.val_cfg['val']:
            self.val_dataloader = self._set_dataloader(self.val_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # other configuration
        self.max_epoch = self.train_cfg['max_epoch']
        self.epoch = self.start_epoch = 1
        #hl_len=len(self.train_dataloader['dataset'].dataset)
        max_len = self.train_dataloader['dataset'].dataset.__len__() # base number of iteration works for dataset named 'dataset'
        self.max_iter = math.ceil(max_len / self.train_cfg['batch_size'])

        self.loss = Loss(self.train_cfg['loss'], self.train_cfg['tmp_info'])
        self.loss_dict = {'count':0}
        self.loss_var={}
        self.tmp_info = {}
        self.loss_log = []
        self.cf_num_dict= {'count':0}
        self.cf_num_1_dict= {'count':0}
        self.cf_num_2_dict= {'count':0}

        # set optimizer
        self.optimizer = self._set_optimizer()
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # resume
        
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch+1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='w')

        # tensorboard
        tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
        self.tboard = SummaryWriter(log_dir=self.file_manager.get_dir('tboard/%s'%tboard_time))

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()


            
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.info(self.summary())
        self.logger.start((self.epoch-1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self._set_status('epoch %03d/%03d'%(self.epoch, self.max_epoch))

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        # model training mode
        self._train_mode()

    def _run_epoch(self):
        for self.iter in range(1, self.max_iter+1):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # save checkpoint
        if self.epoch >= self.ckpt_cfg['start_epoch']:
            if (self.epoch-self.ckpt_cfg['start_epoch'])%self.ckpt_cfg['interval_epoch'] == 0:
                self.save_checkpoint()

        # validation
        if self.val_cfg['val']:
            if self.epoch >= self.val_cfg['start_epoch'] and self.val_cfg['val']:
                if (self.epoch-self.val_cfg['start_epoch']) % self.val_cfg['interval_epoch'] == 0:
                    self._eval_mode()
                    self._set_status('val %03d'%self.epoch)
                    self.validation()

    def _before_step(self):
        pass

    def _run_step(self):
        # get data (data should be dictionary of Tensors)
        data = {}
        for key in self.train_dataloader_iter:
            data[key] = next(self.train_dataloader_iter[key])

        # to device
        if self.cfg['gpu'] != 'None':
            for dataset_key in data:#data is dataset include clean(epoch  3 120 120) real_noisy noise
                for key in data[dataset_key]:
                    data[dataset_key][key] = data[dataset_key][key].cuda() #get  clean real_noisy noise

        # forward, cal losses, backward)
        # losses, tmp_info ,var_loss= self._forward_fn(self.model, self.loss, data)
        losses, tmp_info, cf_num,cf_num_1,cf_num_2 = self._forward_fn(self.model, self.loss, data)
        


        #var_loss = {key: var_loss[key].mean()   for key in var_loss}
        #print(var_loss)
        losses   = {key: losses[key].mean()   for key in losses}
        tmp_info = {key: tmp_info[key].mean() for key in tmp_info}
        cf_num = {key: cf_num[key].mean()   for key in cf_num}
        cf_num_1 = {key: cf_num_1[key].mean()   for key in cf_num_1}
        cf_num_2 = {key: cf_num_2[key].mean()   for key in cf_num_2}

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.requires_grad_(True)
        total_loss.backward()

        # for name, param in self.model['denoiser'].module.bsn_correction.named_parameters():
        #     print(f"Parameter: {name}, Gradient: {param.grad}")

        # optimizer step
        for opt in self.optimizer.values():
            opt.step()

        # zero grad
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True) 

        # save losses and tmp_info
        for key in losses:
            if key != 'count':
                if key in self.loss_dict:
                    self.loss_dict[key] += float(losses[key])
                else:
                    self.loss_dict[key] = float(losses[key])

        for key in cf_num:
            if key != 'count':
                if key in self.cf_num_dict:
                    self.cf_num_dict[key] += float(cf_num[key])
                else:
                    self.cf_num_dict[key] = float(cf_num[key])

        for key in cf_num_1:
            if key != 'count':
                if key in self.cf_num_1_dict:
                    self.cf_num_1_dict[key] += float(cf_num_1[key])
                else:
                    self.cf_num_1_dict[key] = float(cf_num_1[key])
        for key in cf_num_2:
            if key != 'count':
                if key in self.cf_num_2_dict:
                    self.cf_num_2_dict[key] += float(cf_num_2[key])
                else:
                    self.cf_num_2_dict[key] = float(cf_num_2[key])

        for key in tmp_info:
            if key in self.tmp_info:
                self.tmp_info[key] += float(tmp_info[key])
            else:
                self.tmp_info[key] = float(tmp_info[key])
        # for key in var_loss:
            
        #         if key in self.loss_var:
        #             self.loss_var[key] += float(self.loss_var[key])
        #         else:
        #             self.loss_var[key] = float(self.loss_var[key])
        self.loss_dict['count'] += 1
        self.cf_num_dict['count'] += 1
        self.cf_num_1_dict['count'] += 1
        self.cf_num_2_dict['count'] += 1

    def _after_step(self):
        # adjust learning rate
        self._adjust_lr()

        # print loss
        if (self.iter%self.cfg['log']['interval_iter']==0 and self.iter!=0) or (self.iter == self.max_iter):
            self.print_loss()

        # print progress 
        if (self.iter%self.cfg['log']['interval_iter']==0 and self.iter!=0) or (self.iter == self.max_iter):
            self.logger.print_prog_msg((self.epoch-1, self.iter-1))

    def test_dataloader_process(self, dataloader, add_con=0., floor=False, img_save=True, img_save_path=None, info=True):
        '''
        do test or evaluation process for each dataloader
        include following steps:
            1. denoise image
            2. calculate PSNR & SSIM
            3. (optional) save denoised image
        Args:
            dataloader : dataloader to be tested.
            add_con : add constant to denoised image.
            floor : floor denoised image. (default range is [0, 255])
            img_save : whether to save denoised and clean images.
            img_save_path (optional) : path to save denoised images.
            info (optional) : whether to print info.
        Returns:
            psnr : total PSNR score of dataloaer results or None (if clean image is not available)
            ssim : total SSIM score of dataloder results or None (if clean image is not available)
        '''
        # make directory
        self.file_manager.make_dir(img_save_path)

        # test start
        psnr_sum = 0.
        ssim_sum = 0.
        count = 0
        for idx, data in enumerate(dataloader):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]
            denoised_image = self.denoiser(*input_data)

            # add constant and floor (if floor is on)
            denoised_image += add_con
            if floor: denoised_image = torch.floor(denoised_image)

            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                ssim_value = ssim(denoised_image, data['clean'])

                psnr_sum += psnr_value
                ssim_sum += ssim_value
                count += 1

            # image save
            img_save=False
            if img_save:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze(0).cpu()
                if 'real_noisy' in self.cfg['model_input']: noisy_img = data['real_noisy']
                elif 'syn_noisy' in self.cfg['model_input']: noisy_img = data['syn_noisy']
                elif 'noisy' in self.cfg['model_input']: noisy_img = data['noisy']
                else: noisy_img = None
                if noisy_img is not None: noisy_img = noisy_img.squeeze(0).cpu()
                denoi_img = denoised_image.squeeze(0).cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f'%(idx, psnr_value) if 'clean' in data else '%04d_DN'%idx

                # imwrite
                if 'clean' in data:         self.file_manager.save_img_tensor(img_save_path, '%04d_CL'%idx, clean_img)
                if noisy_img is not None: self.file_manager.save_img_tensor(img_save_path, '%04d_N'%idx, noisy_img)
                self.file_manager.save_img_tensor(img_save_path, denoi_name, denoi_img)

            # procedure log msg
            if info:
                if 'clean' in data:
                    self.logger.note('[%s] testing... %04d/%04d. PSNR : %.2f dB'%(self.status, idx, dataloader.__len__(), psnr_value), end='\r')
                else:
                    self.logger.note('[%s] testing... %04d/%04d.'%(self.status, idx, dataloader.__len__()), end='\r')

        # final log msg
        if count > 0:
            self.logger.val('[%s] Done! PSNR : %.2f dB, SSIM : %.3f'%(self.status, psnr_sum/count, ssim_sum/count))
        else:
            self.logger.val('[%s] Done!'%self.status)

        # return
        if count == 0:
            return None, None
        else:
            return psnr_sum/count, ssim_sum/count

    def test_img(self, image_dir, save_dir='./'):
        '''
        Inference a single image.
        '''
        # load image
        img=cv2.imread(image_dir) #hl
        noisy = np2tensor(cv2.imread(image_dir))
        noisy = noisy.unsqueeze(0).float()

        # to device
        if self.cfg['gpu'] != 'None':
            noisy = noisy.cuda()

        # forward
        denoised = self.denoiser(noisy)

        # post-process
        denoised += self.test_cfg['add_con']
        if self.test_cfg['floor']: denoised = torch.floor(denoised)

        # save image
        denoised = tensor2np(denoised)
        denoised = denoised.squeeze(0)
        
        name = image_dir.split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(save_dir, name+'_ourdata_20epoch.png'), denoised)#

        # print message
        self.logger.note('[%s] saved : %s'%(self.status, os.path.join(save_dir, name+'_DN.png')))

    def test_dir(self, direc):
        '''
        Inference all images in the directory.
        '''
        for ff in [f for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))]:
            os.makedirs(os.path.join(direc, 'results'), exist_ok=True)
            self.test_img(os.path.join(direc, ff), os.path.join(direc, 'results'))


    def caculate_cf(self,sigMat,denoised):
        
        #noise_factor
        sigMat = sigMat[:,3:4,:,:]
        cf_num = (torch.sum(sigMat,dim=(1,2,3)))
        cf_num =cf_num **2
        a=sigMat.shape[-1]
        b=torch.sum(torch.abs(sigMat)**2,dim=(1,2,3))
        cf_den = a * b
        cf_factor = cf_num /cf_den
        noise_cf_factor = torch.mean(cf_factor)
        #std
        std_noise = torch.var(sigMat)

        #denoise_factor
        cf_num = (torch.sum(denoised,dim=(1,2,3)))
        cf_num =cf_num **2
        a=denoised.shape[-1]
        b=torch.sum(torch.abs(denoised)**2,dim=(1,2,3))
        cf_den = a * b
        cf_factor = cf_num /cf_den
        denoise_cf_factor = torch.mean(cf_factor)

        std_denoise = torch.var(denoised)


        return noise_cf_factor ,denoise_cf_factor,std_noise,std_denoise
        


    
    def test_tensor_wave(self,sigMat): 
        denoised = self.denoiser(sigMat) #batch channel width height
        # self.caculate_cf(sigMat,denoised)
        mm = copy.deepcopy(denoised[:,0,0,:])
        mm[mm<=0]=-1
        mm[mm>0]=1
        mm_shape = torch.tensor(mm.shape[1])
        row_sums = torch.sum(mm, dim=1)
        # self.wave_img(sigMat,denoised,row_sums)
        result = torch.zeros_like(mm)
        ratio = 3 #0.65-0.35
        result[row_sums <= mm_shape*ratio/10, :] = 1
        result[row_sums > mm_shape*ratio/10, :] = mm[row_sums > mm_shape*ratio/10, :]
        result[row_sums < mm_shape*(-ratio)/10, :] = mm[row_sums < mm_shape*(-ratio)/10, :]
        denoised_ratio3 = denoised[:,0,0,:]
        denoised_ratio3 *= result
        denoised_ratio3 = torch.sum(denoised_ratio3,dim=1)
        result = torch.zeros_like(mm)
        ratio = 4  #0.7-0.3 = 0.4
        result[row_sums <= mm_shape*ratio/10, :] = 1
        result[row_sums > mm_shape*ratio/10, :] = mm[row_sums > mm_shape*ratio/10, :]
        result[row_sums < mm_shape*(-ratio)/10, :] = mm[row_sums < mm_shape*(-ratio)/10, :]
        denoised_ratio4 = denoised[:,0,0,:]
        denoised_ratio4 *= result
        denoised_ratio4 = torch.sum(denoised_ratio4,dim=1)

        denoised_base=torch.sum(denoised[:,:,:,:],dim=(1,2,3))
        return denoised_base, denoised_ratio3,denoised_ratio4

    def prove_estimation(self,sigMat):
        denoised = self.denoiser(sigMat)
        
        ratio_input = copy.deepcopy(sigMat[:,3,0,:])
        ratio_input[ratio_input<=0]=-1
        ratio_input[ratio_input>0]=1
        input_shape = torch.tensor(ratio_input.shape[1])
        ratio_sums = torch.sum(ratio_input, dim=1)
        ratio_cal = torch.abs(ratio_sums/input_shape)#(torch.abs(ratio_sums/input_shape)+1)/2
        ratio_cal = (ratio_cal+1)/2
        if ratio_cal >0.7:
            noise_cf_factor ,denoise_cf_factor = self.caculate_cf(sigMat,denoised)
            print(noise_cf_factor ,denoise_cf_factor)

 

    def test_wave(self, img_name): 
        signalFile_base       = h5py.File(img_name, 'r')
        sigMat      = np.array(signalFile_base['pre_beamform_data'])
        sigMat = np.transpose(sigMat, (2,0,1))
        sigMat = torch.from_numpy(np.ascontiguousarray(sigMat ).astype(np.float32))
        
        sigMat = sigMat.unsqueeze(0).float()
        sigMat = torch.cat([sigMat, sigMat], dim=0)
        # to device
        if self.cfg['gpu'] != 'None':
            sigMat = sigMat.cuda()
        # forward
        denoised = self.denoiser(sigMat)
        denoised = denoised[:,3:4,:,:]
        denoised=torch.sum(denoised)
        return denoised

    def test_dir_wave(self, direc):
        '''
        Inference all images in the directory.
        '''
        file= os.listdir(direc)
        file.sort()
        a=file[-1]
        i=0
        pixel_x=0
        pixel_y=0
        image=torch.zeros(64,64)
        for f in file :
            denoised = self.test_wave(os.path.join(direc, f))
            image[int(i/64),int(i%64)]=denoised
            i=i+1
            if(i%500==0):
                print(i)

    def test_dir_wave_for_tr(self, direc):
        '''
        Inference all images in the directory.
        '''
        file= os.listdir(direc)
        file.sort()
        i=0
        pixel_x=0
        pixel_y=0
        image=torch.zeros(64,64)
        for f in file :
            denoised = self.test_wave(os.path.join(direc, f))
            image[int(i/64),int(i%64)]=denoised
            i=i+1
            if(i%500==0):
                print(i)
    
   

        image=image.cpu().detach()
        image=image.numpy()
        
        imageRecon = 20*np.log10((np.abs(image))/(np.amax(np.abs(image))))
        imageRecon[imageRecon<-30]=-30

        maxValue = np.amax(imageRecon)
        minValue = np.amin(imageRecon)

        # normalize image
        normRecon = 255*(imageRecon-minValue)/(maxValue-minValue)

        # convert to rgb
        arrayRecon = img.fromarray(normRecon)
        if arrayRecon.mode != 'RGB':
            rgbRecon = arrayRecon.convert('RGB')
        
        rgbRecon.save( self.cfg['save_wave_img_name'])




    def builduct_dynamic_all_bin(self):
        ELE_NO = 2048
        M = 256
        R0 = 112e-3#%112
        SOUND_SPEED = 1490
        FS = 25e6
        NSAMPLE = 3750
        NS0 = 998#187
        # hl_wave_base rect32 pixel 64
        RECT_LENGTH = 150e-3#100
        PIXEL_NO = 2048#512
        grid_step = np.linspace(-RECT_LENGTH/2, RECT_LENGTH/2, PIXEL_NO)
        x_pos, z_pos = np.meshgrid(grid_step,grid_step)
        x_pos = x_pos.flatten('F')
        z_pos = z_pos.flatten('F')

        
        ele_x_pos = np.zeros(ELE_NO)
        ele_z_pos = np.zeros(ELE_NO)
        first_one = 2*np.pi/360*(45-43.4695)
        ele_angle = 2*np.pi/360*(43.4695/255)

        for npart in range (1,9):
            begin_npart = np.pi/4*(npart-1)
            for nele in range (1,257):
                ele_polor = first_one+begin_npart+ele_angle*(nele-1)
                ele_x_pos[256*(npart-1)+nele-1] = R0*np.cos(ele_polor)
                ele_z_pos[256*(npart-1)+nele-1] = -R0*np.sin(ele_polor)


        FS = 25e6
        CUTOFF_FREQ1 = 1e6
        CUTOFF_FREQ2 = 4.5e6
        FILTER_ORDER = 72

        
        nyquist = 0.5 * FS
        low = CUTOFF_FREQ1 / nyquist
        high = CUTOFF_FREQ2 / nyquist
        fir_coeff = firwin(FILTER_ORDER + 1, [low, high], pass_zero=False)
        # d1 = filtfilt(fir_coeff, [1.0], [1.0])  
        folder_path='/mnt/nvme_sda1/uct_data/'
        
        for folder_name in os.listdir(folder_path):
            #folder_name = '20211203001_TAGU_R_20211203-102918_16L'
            if folder_name[:8] == "3_1_4090":
                file_path = '/mnt/nvme_sda1/uct_data/'+folder_name + '/slice2.bin'
            else :
                continue
            
            with open(file_path, 'rb') as file:
                
                step_list = np.array([64])#8
                for step in step_list:
                    
                    image = np.zeros(PIXEL_NO * PIXEL_NO)
                    image_ratio3 = np.zeros(PIXEL_NO * PIXEL_NO)
                    image_ratio4 = np.zeros(PIXEL_NO * PIXEL_NO)

                    S_counter = np.zeros(PIXEL_NO * PIXEL_NO)
                    for emit_i in range(0, ELE_NO, step):#ELE_NO
                        start_time = time.time()
                        print(f'Emitter No {emit_i}',str(folder_name) )
                        skip = NSAMPLE * ELE_NO * 2 * emit_i
                        file.seek(skip,0)
                        
                        data = np.fromfile(file, dtype=np.int16, count=4*ELE_NO * NSAMPLE).reshape(( NSAMPLE,4*ELE_NO))
                        data = data[:,:2048]
                        fdata = filtfilt(fir_coeff,[1.0], data.T).T                      
                        fdata[:100, :] = 0
                        
                        emit_x_pos = ele_x_pos[emit_i]
                        emit_z_pos = ele_z_pos[emit_i]

                        depth_penetration = (emit_x_pos - x_pos)**2 + (emit_z_pos - z_pos)**2
                        
            
                        r_square = x_pos**2 + z_pos**2
                        select_point = (np.sqrt(depth_penetration) < 1.0 * R0) & \
                                        (np.arccos((R0**2 + depth_penetration - r_square) / (2 * R0 * np.sqrt(depth_penetration))) < np.pi / 9)

                        select_point=select_point*1
                        
                        index_down = emit_i - M
                        index_up = emit_i + M 
                        if index_down < 0:
                            sub_apture_index = np.concatenate([np.arange(ELE_NO + index_down, ELE_NO), np.arange(0, index_up)])
                            sub_fdata = np.concatenate([fdata[:, ELE_NO + index_down: ELE_NO], fdata[:, :index_up ]], axis=1)
                        elif index_up > ELE_NO:
                            sub_apture_index = np.concatenate([np.arange(index_down, ELE_NO), np.arange(0, index_up - ELE_NO )])
                            sub_fdata = np.concatenate([fdata[:, index_down: ELE_NO], fdata[:, :index_up - ELE_NO ]], axis=1)
                        else:
                            sub_apture_index = np.arange(index_down, index_up )
                            sub_fdata = fdata[:, index_down: index_up]

                        
                        
                        single_sensor_img = np.zeros(PIXEL_NO * PIXEL_NO)

                        width_dynamic_recv = np.round(np.sqrt(depth_penetration) * select_point / R0 * (M - 50)) + 50
                        begin_dynamic_recv = M - width_dynamic_recv
                        end_dynamic_recv = begin_dynamic_recv + width_dynamic_recv * 2
                        begin_dynamic_recv[begin_dynamic_recv < 0] = 0
                        end_dynamic_recv[end_dynamic_recv > (2 * M-1)] = 2 * M-1

                        begin_dynamic_recv = begin_dynamic_recv* select_point
                        end_dynamic_recv = end_dynamic_recv* select_point

                        select_point_remote50 = width_dynamic_recv*2 
                        find_select_point = np.where(select_point)[0] 
                        
                        recv_x_pos = ele_x_pos[sub_apture_index]
                        recv_z_pos = ele_z_pos[sub_apture_index]


                        batch_size = 1
                        patch_size = 81
                        
                        single_sensor_img,select_point_remote50,single_sensor_img_ratio3,single_sensor_img_ratio4 = self.load_batch_pixel_gpu_conv1d_all_bin(sub_fdata, depth_penetration,batch_size,recv_x_pos,recv_z_pos,x_pos,z_pos,begin_dynamic_recv,end_dynamic_recv,SOUND_SPEED,FS,NS0,select_point,select_point_remote50,find_select_point,PIXEL_NO,patch_size)
                        
                        
                        S_counter += select_point_remote50
                        end_time = time.time()
                        print(end_time-start_time)
                        image += single_sensor_img
                        image_ratio3 += single_sensor_img_ratio3
                        image_ratio4 += single_sensor_img_ratio4
                        

                    image = image/S_counter
                    image = image.reshape(PIXEL_NO,PIXEL_NO)
                    image = np.rot90(image, k=1)
                    image[np.isnan(image)] = 0
                    image = 20*np.log10(np.abs(image) / np.max(np.abs(image)))
                    
                    db_range =-50
                    for db_range in range(-100,-30,10):
                        image_db = image
                        image_db[image_db<db_range] = db_range
                        image_db = image_db[::-1]
                        maxValue = np.amax(image_db)
                        minValue = np.amin(image_db)

                        # normalize image
                        normRecon = np.round((255*(image_db-minValue)/(maxValue-minValue))).astype(np.uint8)

                        # convert to rgb
                        arrayRecon = img.fromarray(normRecon)
                        if arrayRecon.mode != 'RGB':
                            rgbRecon = arrayRecon.convert('RGB')
                        
                        
                        File_Path='./test_result/5bin_train_all_UCTData_1in_1out_epoch15/'+folder_name +'/1in_1out_conv/'+str(step)+'_pixel_'+str(PIXEL_NO)
                        if not os.path.exists(File_Path):
                            os.makedirs(File_Path)
                        rgbRecon.save( File_Path+'/dB'+str(db_range)+'.png')   

                    image_ratio3 = image_ratio3/S_counter
                    image_ratio3 = image_ratio3.reshape(PIXEL_NO,PIXEL_NO)
                    image_ratio3 = np.rot90(image_ratio3, k=1)
                    image_ratio3[np.isnan(image_ratio3)] = 0
                    image_ratio3 = 20*np.log10(np.abs(image_ratio3) / np.max(np.abs(image_ratio3)))
                    
                    db_range = -50
                    for db_range in range(-100,-30,10):
                        image_db = image_ratio3
                        image_db[image_db<db_range] = db_range
                        image_db = image_db[::-1]
                        maxValue = np.amax(image_db)
                        minValue = np.amin(image_db)

                        # normalize image
                        normRecon = np.round((255*(image_db-minValue)/(maxValue-minValue))).astype(np.uint8)

                        # convert to rgb
                        arrayRecon = img.fromarray(normRecon)
                        if arrayRecon.mode != 'RGB':
                            rgbRecon = arrayRecon.convert('RGB')
                        
                        File_Path='./test_result/5bin_train_all_UCTData_1in_1out_epoch15/'+folder_name +'/1in_1out_conv_ratio3/'+str(step)+'_pixel_'+str(PIXEL_NO)
                        if not os.path.exists(File_Path):
                            os.makedirs(File_Path)
                        rgbRecon.save( File_Path+'/dB'+str(db_range)+'.png')    

                    image_ratio4 = image_ratio4/S_counter
                    image_ratio4 = image_ratio4.reshape(PIXEL_NO,PIXEL_NO)
                    image_ratio4 = np.rot90(image_ratio4, k=1)
                    image_ratio4[np.isnan(image_ratio4)] = 0
                    image_ratio4 = 20*np.log10(np.abs(image_ratio4) / np.max(np.abs(image_ratio4)))
                    
                    db_range=-50
                    for db_range in range(-100,-30,10):
                        image_db = image_ratio4
                        image_db[image_db<db_range] = db_range
                        image_db = image_db[::-1]
                        maxValue = np.amax(image_db)
                        minValue = np.amin(image_db)

                        # normalize image
                        normRecon = np.round((255*(image_db-minValue)/(maxValue-minValue))).astype(np.uint8)

                        # convert to rgb
                        arrayRecon = img.fromarray(normRecon)
                        if arrayRecon.mode != 'RGB':
                            rgbRecon = arrayRecon.convert('RGB')
                        
                        File_Path='./test_result/5bin_train_all_UCTData_1in_1out_epoch15/'+folder_name +'/1in_1out_conv_ratio4/'+str(step)+'_pixel_'+str(PIXEL_NO)
                        if not os.path.exists(File_Path):
                            os.makedirs(File_Path)
                        rgbRecon.save( File_Path+'/dB'+str(db_range)+'.png')  

                

    

    

    def load_batch_pixel_gpu_conv1d_all_bin(self,sub_fdata, depth_penetration,batch_size,recv_x_pos,recv_z_pos,x_pos,z_pos,begin_dynamic_recv,end_dynamic_recv,SOUND_SPEED,FS,NS0,select_point,select_point_remote50,find_select_point,PIXEL_NO, patch_size):#center_x, center_y, radius, start_angle, end_angle
       
        start_time = time.time()
        sub_fdata = sub_fdata.copy()
        sub_fdata = torch.tensor(sub_fdata).cuda()
        select_point = torch.tensor(select_point).cuda()
        end_dynamic_recv = torch.tensor(end_dynamic_recv).cuda()
        begin_dynamic_recv = torch.tensor(begin_dynamic_recv).cuda()
        depth_penetration = torch.tensor(depth_penetration).cuda()
        recv_x_pos = torch.tensor(recv_x_pos).cuda()
        x_pos = torch.tensor(x_pos).cuda()
        recv_z_pos = torch.tensor(recv_z_pos).cuda()
        z_pos = torch.tensor(z_pos).cuda()
        select_point_remote50 = torch.tensor(select_point_remote50).cuda()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        select_point_2d = select_point.reshape((PIXEL_NO,PIXEL_NO))
        select_point_remote50_2d = select_point_remote50.reshape((PIXEL_NO,PIXEL_NO))
        select_point_flag = select_point
        batch_subtract_recv = (end_dynamic_recv-begin_dynamic_recv+1)*select_point
        single_sensor_img = np.zeros(PIXEL_NO * PIXEL_NO)
        single_sensor_img_ratio3 = np.zeros(PIXEL_NO * PIXEL_NO)
        single_sensor_img_ratio4 = np.zeros(PIXEL_NO * PIXEL_NO)
        
        end_time1 = time.time()
        
        
        for x  in range (0,PIXEL_NO-patch_size+1,patch_size):
            for y in range (0,PIXEL_NO-patch_size+1,patch_size):
                if select_point_2d[x,y]==1 and select_point_2d[x+patch_size-1,y]==1 and select_point_2d[x,y+patch_size-1]==1  and select_point_2d[x+patch_size-1,y+patch_size-1]==1 :
                    
                    max_batch_subtract_recv = 0
                    delay_batch_min = 5000
                    delay_batch_max = 0
                    signal_batch_num_max = 0
                    ceil_signal_num = 0
                    patch_indices = torch.stack(torch.meshgrid([torch.arange(patch_size), torch.arange(patch_size)])).reshape(2,-1)
                    patch_indices[0] += x
                    patch_indices[1] += y
                    replace_index=patch_indices[0]*PIXEL_NO+patch_indices[1]
                    

                    
                    delay=(torch.sqrt(depth_penetration[replace_index]).unsqueeze(1)+\
                        torch.sqrt((- x_pos[replace_index].unsqueeze(1)+recv_x_pos.unsqueeze(0).expand(replace_index.shape[0], -1))**2+\
                                (- z_pos[replace_index].unsqueeze(1)+recv_z_pos.unsqueeze(0).expand(replace_index.shape[0], -1))**2))/SOUND_SPEED * FS - NS0
                    
                   
                    delay[delay < 0] = 0
                    delay[delay > 3749] = 0
                    delay_raw = torch.round(delay).to(torch.int)
                    
                    delay_col = torch.arange(512)
                    
                    
                    delay_batch_min=torch.min(delay_raw)
                    delay_batch_max=torch.max(delay_raw)
                    
                    max_batch_subtract_recv=torch.max(batch_subtract_recv[replace_index])
                    
                    

                    if(delay_batch_min>104 and delay_batch_max<3746):
                        batch_pre_beamform_data_hl_test2 = []

                        wave_index = torch.arange(-3, 4).unsqueeze(0).unsqueeze(2).expand(delay_raw.shape[0],7, delay_raw.shape[1] ).cuda()
                        delay_raw = delay_raw.unsqueeze(1)-wave_index
                        pre_beamform_data = sub_fdata[delay_raw , delay_col]#.permute(0, 2, 1)

                        
                       
                        pre_beamform_data_hl_test2=pre_beamform_data[:,:,256-int(max_batch_subtract_recv//2):256+int(max_batch_subtract_recv//2)]
                       
                        select_point_remote50_2d[x:x+patch_size,y:y+patch_size]=max_batch_subtract_recv

                        
                        pre_beamform_data_hl_test2 = pre_beamform_data_hl_test2.float()
                        pre_beamform_data_hl_test2 = pre_beamform_data_hl_test2.unsqueeze(2)
                        pixel_int_batch ,denoised_ratio3,denoised_ratio4=self.test_tensor_wave(pre_beamform_data_hl_test2[:,3:4,:,:])
                        pixel_int_batch = pixel_int_batch.cpu().numpy()
                        denoised_ratio3 = denoised_ratio3.cpu().numpy()
                        denoised_ratio4 = denoised_ratio4.cpu().numpy()

                        single_sensor_img[replace_index]=pixel_int_batch
                        single_sensor_img_ratio3[replace_index]=denoised_ratio3
                        single_sensor_img_ratio4[replace_index]=denoised_ratio4
                        
                        
                        select_point_flag[replace_index]=2
                        
        end_time2 = time.time()
        
        select_point_flag=select_point_flag.cpu().numpy()
        sub_fdata = sub_fdata.cpu().numpy()
        select_point = select_point.cpu().numpy()
        end_dynamic_recv=end_dynamic_recv.cpu().numpy()
        begin_dynamic_recv=begin_dynamic_recv.cpu().numpy()
        depth_penetration=depth_penetration.cpu().numpy()
        recv_x_pos=recv_x_pos.cpu().numpy()
        x_pos=x_pos.cpu().numpy()
        recv_z_pos=recv_z_pos.cpu().numpy()
        z_pos=z_pos.cpu().numpy()
        select_point_remote50=select_point_remote50_2d.view(-1).cpu().numpy()
        

        find_remain_select_point=np.where(select_point_flag == 1) [0]
        for pixel_index in range (0,len(find_remain_select_point)):
            pixel_num=find_remain_select_point[pixel_index ]
           
            delay = (np.sqrt(depth_penetration[pixel_num]) +
                                                np.sqrt((recv_x_pos - x_pos[pixel_num])**2 + (recv_z_pos - z_pos[pixel_num])**2)) / \
                                                SOUND_SPEED * FS - NS0
            delay[delay < 0] = 0
            delay[delay > 3749] = 0
            delay_raw = np.round(delay).astype(int)
            
            delay_col = np.arange(512)

            
            pre_beamform_data=np.array([sub_fdata[delay_raw,delay_col]])
            begin_recv_in_this_pixel=int(begin_dynamic_recv[pixel_num])
            end_recv_in_this_pixel=int(end_dynamic_recv[pixel_num])
            pre_beamform_data=pre_beamform_data[:,begin_recv_in_this_pixel:end_recv_in_this_pixel+1]
            # pixel_int_batch[pixel_num_in_batch]=np.sum(pre_beamform_data)
            single_sensor_img[pixel_num]=np.sum(pre_beamform_data)
            single_sensor_img_ratio3[pixel_num] = np.sum(pre_beamform_data)
            single_sensor_img_ratio4[pixel_num] = np.sum(pre_beamform_data)
            
        end_time3=time.time()

        return single_sensor_img,select_point_remote50,single_sensor_img_ratio3,single_sensor_img_ratio4
    



    def test_DND(self, img_save_path):
        '''
        Benchmarking DND dataset.
        '''
        # make directories for .mat & image saving 
        self.file_manager.make_dir(img_save_path)
        self.file_manager.make_dir(img_save_path + '/mat')
        if self.test_cfg['save_image']: self.file_manager.make_dir(img_save_path + '/img')

        def wrap_denoiser(Inoisy, nlf, idx, kidx):
            noisy = 255 * torch.from_numpy(Inoisy)

            # to device
            if self.cfg['gpu'] != 'None':
                noisy = noisy.cuda()

            noisy = autograd.Variable(noisy)

            # processing
            noisy = noisy.permute(2,0,1)
            noisy = self.test_dataloader['dataset'].dataset._pre_processing({'real_noisy': noisy})['real_noisy']

            noisy = noisy.view(1,noisy.shape[0], noisy.shape[1], noisy.shape[2])

            denoised = self.denoiser(noisy)

            denoised += self.test_cfg['add_con']
            if self.test_cfg['floor']: denoised = torch.floor(denoised)

            denoised = denoised[0,...].cpu().numpy()
            denoised = np.transpose(denoised, [1,2,0])

            # image save
            if self.test_cfg['save_image'] and False:
                self.file_manager.save_img_numpy(img_save_path+'/img', '%02d_%02d_N'%(idx, kidx),  255*Inoisy)
                self.file_manager.save_img_numpy(img_save_path+'/img', '%02d_%02d_DN'%(idx, kidx), denoised)

            return denoised / 255

        denoise_srgb(wrap_denoiser, './dataset/DND/dnd_2017', self.file_manager.get_dir(img_save_path+'/mat'))

        bundle_submissions_srgb(self.file_manager.get_dir(img_save_path+'/mat'))

        # info 
        self.logger.val('[%s] Done!'%self.status)

    def _set_denoiser(self):
        if hasattr(self.model['denoiser'].module, 'denoise'):
            self.denoiser = self.model['denoiser'].module.denoise
        else:
            self.denoiser = self.model['denoiser'].module

    @torch.no_grad()
    def crop_test(self, fn, x, size=512, overlap=0):
        '''
        crop test image and inference due to memory problem
        '''
        b,c,h,w = x.shape
        denoised = torch.zeros_like(x)
        for i in range(0,h,size-overlap):
            for j in range(0,w,size-overlap):
                end_i = min(i+size, h)
                end_j = min(j+size, w)
                x_crop = x[...,i:end_i,j:end_j]
                denoised_crop = fn(x_crop)
                
                start_i = overlap if i != 0 else 0
                start_j = overlap if j != 0 else 0

                denoised[..., i+start_i:end_i, j+start_j:end_j] = denoised_crop[..., start_i:, start_j:]

        return denoised

    @torch.no_grad()
    def self_ensemble(self, fn, x):
        '''
        Geomery self-ensemble function
        Note that in this function there is no gradient calculation.
        Args:
            fn : denoiser function
            x : input image
        Return:
            result : self-ensembled image
        '''
        result = torch.zeros_like(x)

        for i in range(8):
            tmp = fn(rot_hflip_img(x, rot_times=i%4, hflip=i//4))
            tmp = rot_hflip_img(tmp, rot_times=4-i%4)
            result += rot_hflip_img(tmp, hflip=i//4)
        return result / 8

    #----------------------------#
    #      Utility functions     #
    #----------------------------# 
    def print_loss(self):
        temporal_loss = 0.
        for key in self.loss_dict:
            if key != 'count':
                    temporal_loss += self.loss_dict[key]/self.loss_dict['count']
        self.loss_log += [temporal_loss]
        if len(self.loss_log) > 100: self.loss_log.pop(0)

        # print status and learning rate
        loss_out_str = '[%s] %04d/%04d, lr:%s ∣ '%(self.status, self.iter, self.max_iter, "{:.1e}".format(self._get_current_lr()))
        global_iter = (self.epoch-1)*self.max_iter + self.iter

        # print losses
        avg_loss = np.mean(self.loss_log)
        loss_out_str += 'avg_100 : %.6f ∣ '%(avg_loss)
        self.tboard.add_scalar('loss/avg_100', avg_loss, global_iter)

        for key in self.loss_dict:
            if key != 'count':
                loss = self.loss_dict[key]/self.loss_dict['count']
                loss_out_str += '%s : %.6f ∣ '%(key, loss)
                self.tboard.add_scalar('loss/%s'%key, loss, global_iter)
                self.loss_dict[key] = 0.
        
        for key in self.cf_num_dict:
            if key != 'count':
                cf_num = self.cf_num_dict[key]/self.cf_num_dict['count']
                loss_out_str += '%s : %.6f ∣ '%(key, cf_num)
                # self.tboard.add_scalar('cf_num/%s'%key, cf_num, global_iter)
                self.cf_num_dict[key] = 0.
        for key in self.cf_num_1_dict:
            if key != 'count':
                cf_num = self.cf_num_1_dict[key]/self.cf_num_1_dict['count']
                loss_out_str += '%s : %.6f ∣ '%(key, cf_num)
                # self.tboard.add_scalar('cf_num/%s'%key, cf_num, global_iter)
                self.cf_num_1_dict[key] = 0.
        for key in self.cf_num_2_dict:
            if key != 'count':
                cf_num = self.cf_num_2_dict[key]/self.cf_num_2_dict['count']
                loss_out_str += '%s : %.6f ∣ '%(key, cf_num)
                # self.tboard.add_scalar('cf_num/%s'%key, cf_num, global_iter)
                self.cf_num_2_dict[key] = 0.

        # print temporal information
        if len(self.tmp_info) > 0:
            loss_out_str += '\t['
            for key in self.tmp_info:
                loss_out_str += '  %s : %.2f'%(key, self.tmp_info[key]/self.loss_dict['count'])
                self.tmp_info[key] = 0.
            loss_out_str += ' ]'

        # reset
        self.loss_dict['count'] = 0
        self.cf_num_dict['count'] = 0
        self.cf_num_1_dict['count'] = 0
        self.cf_num_2_dict['count'] = 0
        self.logger.info(loss_out_str)

    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        aa=os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name)
        torch.save({'epoch': self.epoch,
                    'model_weight': {key:self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                    os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name))
        torch.save({'epoch': self.epoch,
                    'model_weight': {key:self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                    os.path.join('./ckpt/', checkpoint_name))
        if((self.epoch-1) %5 ==0):
            a=1
            # val_img(checkpoint_name)
        

    def load_checkpoint(self, load_epoch=0, name=None):
        if name is None:
            # if scratch, return
            if load_epoch == 0: return
            # load from local checkpoint folder
            file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        else:
            # load from global checkpoint folder
            file_name = os.path.join('./ckpt', name)
    
        # check file exist
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name

        # load checkpoint (epoch, model_weight, optimizer_weight)
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.optimizer:
                self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

        # print message
        # self.logger.note('[%s] model loaded : %s'%(self.status, file_name))

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d'%epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.'%self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for first_optim in self.optimizer.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']

    def _set_dataloader(self, dataset_cfg, batch_size, shuffle, num_workers):
        dataloader = {}
        dataset_dict = dataset_cfg['dataset']
        if not isinstance(dataset_dict, dict):
            dataset_dict = {'dataset': dataset_dict}

        for key in dataset_dict:
            args = dataset_cfg[key + '_args']
            dataset = get_dataset_class(dataset_dict[key])(**args)
            dataloader[key] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

        return dataloader

    def _set_one_optimizer(self, opt, parameters, lr):
        lr = float(self.train_cfg['init_lr'])

        if opt['type'] == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt['SGD']['momentum']), weight_decay=float(opt['SGD']['weight_decay']))
        elif opt['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt['Adam']['betas'])
        elif opt['type'] == 'AdamW':
            return optim.Adam(parameters, lr=lr, betas=opt['AdamW']['betas'], weight_decay=float(opt['AdamW']['weight_decay']))
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _adjust_lr(self):
        sched = self.train_cfg['scheduler']

        if sched['type'] == 'step':
            '''
            step decreasing scheduler
            Args:
                step_size: step size(epoch) to decay the learning rate
                gamma: decay rate
            '''
            if self.iter == self.max_iter:
                args = sched['step']
                if self.epoch % args['step_size'] == 0:
                    for optimizer in self.optimizer.values():
                        lr_before = optimizer.param_groups[0]['lr']
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_before * float(args['gamma'])
        elif sched['type'] == 'linear':
            '''
            linear decreasing scheduler
            Args:
                step_size: step size(epoch) to decrease the learning rate
                gamma: decay rate for reset learning rate
            '''
            args = sched['linear']
            if not hasattr(self, 'reset_lr'):
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args['gamma'])**((self.epoch-1)//args['step_size'])

            # reset lr to initial value
            if self.epoch % args['step_size'] == 0 and self.iter == self.max_iter:
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args['gamma'])**(self.epoch//args['step_size'])
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.reset_lr
            # linear decaying
            else:
                ratio = ((self.epoch + (self.iter)/self.max_iter - 1) % args['step_size']) / args['step_size']
                curr_lr = (1-ratio) * self.reset_lr
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))

    def _adjust_warmup_lr(self, warmup_iter):
        init_lr = float(self.train_cfg['init_lr'])
        warmup_lr = init_lr * self.iter / warmup_iter

        for optimizer in self.optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def _train_mode(self):
        for key in self.model:
            self.model[key].train()

    def _eval_mode(self):
        for key in self.model:
            self.model[key].eval()

    def _set_status(self, status:str):
        assert len(status) <= status_len, 'status string cannot exceed %d characters, (now %d)'%(status_len, len(status))

        if len(status.split(' ')) == 2:
            s0, s1 = status.split(' ')
            self.status = '%s'%s0.rjust(status_len//2) + ' '\
                          '%s'%s1.ljust(status_len//2)
        else:
            sp = status_len - len(status)
            self.status = ''.ljust(sp//2) + status + ''.ljust((sp+1)//2)

    def summary(self):
        summary = ''

        summary += '-'*100 + '\n'
        # model
        for k, v in self.module.items():
            # get parameter number
            param_num = sum(p.numel() for p in v.parameters())

            # get information about architecture and parameter number
            summary += '[%s] paramters: %s -->'%(k, human_format(param_num)) + '\n'
            summary += str(v) + '\n\n'
        
        # optim

        # Hardware

        summary += '-'*100 + '\n'

        return summary



