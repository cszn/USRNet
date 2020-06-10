import os.path
import cv2
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat
from scipy import ndimage
import scipy.io as scio

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util

from models.network_usrnet import USRNet as net

'''
Spyder (Python 3.7)
PyTorch 1.4.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/USRNet
        https://github.com/cszn/KAIR

If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)

by Kai Zhang (12/March/2020)
'''

"""
# --------------------------------------------
testing code of USRNet for the Table 1 in the paper
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3217--3226},
  year={2020}
}
# --------------------------------------------
|--model_zoo                # model_zoo
   |--usrgan                # model_name, optimized for perceptual quality
   |--usrnet                # model_name, optimized for PSNR
   |--usrgan_tiny           # model_name, tiny model optimized for perceptual quality
   |--usrnet_tiny           # model_name, tiny model optimized for PSNR
|--testsets                 # testsets
   |--real_set              # testset_name, contain 3 real images
|--results                  # results
   |--real_set_usrnet       # result_name = testset_name + '_' + model_name
   |--real_set_usrnet_tiny
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'usrnet'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
    testset_name = 'set_real'  # test set,  'set_real'
    test_image = 'chip.png'    # 'chip.png', 'comic.png'
    #test_image = 'comic.png'

    sf = 4                     # scale factor, only from {1, 2, 3, 4}
    show_img = False           # default: False
    save_E = True              # save estimated image
    save_LE = True             # save zoomed LR, Estimated images

    # ----------------------------------------
    # set noise level and kernel
    # ----------------------------------------
    if 'chip' in test_image:
        noise_level_img = 15       # noise level for LR image, 15 for chip
        kernel_width_default_x1234 = [0.6, 0.9, 1.7, 2.2] # Gaussian kernel widths for x1, x2, x3, x4
    else:
        noise_level_img = 2       # noise level for LR image, 0.5~3 for clean images
        kernel_width_default_x1234 = [0.4, 0.7, 1.5, 2.0] # default Gaussian kernel widths of clean/sharp images for x1, x2, x3, x4

    noise_level_model = noise_level_img/255.  # noise level of model
    kernel_width = kernel_width_default_x1234[sf-1]

    # set your own kernel width
    # kernel_width = 2.2

    k = utils_deblur.fspecial('gaussian', 25, kernel_width)
    k = sr.shift_pixel(k, sf)  # shift the kernel
    k /= np.sum(k)
    util.surf(k) if show_img else None
    # scio.savemat('kernel_realapplication.mat', {'kernel':k})

    # load approximated bicubic kernels
    #kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernel_bicubicx234.mat'))['kernels']
#    kernels = loadmat(os.path.join('kernels', 'kernel_bicubicx234.mat'))['kernels']
#    kernel = kernels[0, sf-2].astype(np.float64)

    kernel = util.single2tensor4(k[..., np.newaxis])


    n_channels = 1 if 'gray' in  model_name else 3  # 3 for color image, 1 for grayscale image
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, fixed, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    if 'tiny' in model_name:
        model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    else:
        model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                    nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for key, v in model.named_parameters():
        v.requires_grad = False

    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)

    img = os.path.join(L_path, test_image)
    # ------------------------------------
    # (1) img_L
    # ------------------------------------
    img_name, ext = os.path.splitext(os.path.basename(img))
    img_L = util.imread_uint(img, n_channels=n_channels)
    img_L = util.uint2single(img_L)

    util.imshow(img_L) if show_img else None
    w, h = img_L.shape[:2]
    logger.info('{:>10s}--> ({:>4d}x{:<4d})'.format(img_name+ext, w, h))

    # boundary handling
    boarder = 8     # default setting for kernel size 25x25
    img = cv2.resize(img_L, (sf*h, sf*w), interpolation=cv2.INTER_NEAREST)
    img = utils_deblur.wrap_boundary_liu(img, [int(np.ceil(sf*w/boarder+2)*boarder), int(np.ceil(sf*h/boarder+2)*boarder)])
    img_wrap = sr.downsample_np(img, sf, center=False)
    img_wrap[:w, :h, :] = img_L
    img_L = img_wrap

    util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------
    sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])
    [img_L, kernel, sigma] = [el.to(device) for el in [img_L, kernel, sigma]]

    img_E = model(img_L, kernel, sf, sigma)

    img_E = util.tensor2uint(img_E)[:sf*w, :sf*h, ...]

    if save_E:
        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'.png'))

    # --------------------------------
    # (3) save img_LE
    # --------------------------------
    if save_LE:
        k_v = k/np.max(k)*1.2
        k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
        k_factor = 3
        k_v = cv2.resize(k_v, (k_factor*k_v.shape[1], k_factor*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_L = util.tensor2uint(img_L)[:w, :h, ...]
        img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_I[:k_v.shape[0], :k_v.shape[1], :] = k_v
        util.imshow(np.concatenate([img_I, img_E], axis=1), title='LR / Recovered') if show_img else None
        util.imsave(np.concatenate([img_I, img_E], axis=1), os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'_LE.png'))


if __name__ == '__main__':

    main()
