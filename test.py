import argparse, os

import torch


from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class
import numpy as np


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name', default=None,  type=str)
    args.add_argument('-c', '--config',       default='CUSTOM',  type=str)#defalt= None CUSTOM APBSN_DND
    args.add_argument('-e', '--ckpt_epoch',   default=0,     type=int)
    args.add_argument('-g', '--gpu',          default='0',  type=str)#defalt= None
    # args.add_argument(      '--pretrained',   default='CUSTOM_040_channel14_2_2channel.pth',  type=str)#defalt= None CUSTOM_040_channel7_2_1channel
    args.add_argument(      '--pretrained',   default='CUSTOM_015.pth',  type=str)#defalt= None CUSTOM_040_channel7_2_1channel
    args.add_argument(      '--thread',       default=4,     type=int)
    args.add_argument(      '--self_en',      action='store_true')
    args.add_argument(      '--test_img',     default=None,  type=str)#defalt=  './figs/test_hl_img/512/0033.png' 
    args.add_argument(      '--test_dir',     default=None,  type=str) #None './dataset/make_wave_pixel100'
    args.add_argument(      '--builduct',     default='./',  type=str) #None
    args.add_argument(      '--save_wave_img_name',     default='./output/CUSTOM/wave_img/db-30_channel16_wave_img.jpg',  type=str) #None


    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config # set session name to config file name

    cfg = ConfigParser(args)
    

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    # test
    trainer.test()


if __name__ == '__main__':
    print("notgc_model")
    main()
