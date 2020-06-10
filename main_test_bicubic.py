import os.path
import logging
import cv2
import numpy as np
from collections import OrderedDict
from scipy.io import loadmat
#import hdf5storage
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_deblur
from utils import utils_sisr as sr

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
   |--set5                  # testset_name
   |--set14
   |--urban100
   |--bsd100
   |--srbsd68               # already cropped
|--results                  # results
   |--set5_usrnet_bicubic   # result_name = testset_name + '_' + model_name + '_bicubic'
   |--set5_usrgan_bicubic
   |--set5_usrnet_tiny_bicubic
   |--set5_usrgan_tiny_bicubic
# --------------------------------------------
"""



def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'usrnet'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
    testset_name = 'set5'     # test set,  'set5' | 'srbsd68'
    need_degradation = True    # default: True
    sf = 4                     # scale factor, only from {2, 3, 4}
    show_img = False           # default: False
    save_L = True              # save LR image
    save_E = True              # save estimated image

    # load approximated bicubic kernels
    #kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels_bicubicx234.mat'))['kernels']
    kernels = loadmat(os.path.join('kernels', 'kernels_bicubicx234.mat'))['kernels']
    kernel = kernels[0, sf-2].astype(np.float64)
    kernel = util.single2tensor4(kernel[..., np.newaxis])

    task_current = 'sr'       # fixed, 'sr' for super-resolution
    n_channels = 3            # fixed, 3 for color image
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    noise_level_img = 0       # fixed: 0, noise level for LR image
    noise_level_model = noise_level_img  # fixed, noise level of model, default 0
    result_name = testset_name + '_' + model_name + '_bicubic'
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, fixed, for Low-quality images
    H_path = L_path                               # H_path, 'None' | L_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_usrnet import USRNet as net

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

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        # degradation process, bicubic downsampling
        if need_degradation:
            img_L = util.modcrop(img_L, sf)
            img_L = util.imresize_np(img_L, 1/sf)

            # img_L = util.uint2single(util.single2uint(img_L))
            # np.random.seed(seed=0)  # for reproducibility
            # img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        w, h = img_L.shape[:2]

        if save_L:
            util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name+'_LR_x'+str(sf)+'.png'))

        img = cv2.resize(img_L, (sf*h, sf*w), interpolation=cv2.INTER_NEAREST)
        img = utils_deblur.wrap_boundary_liu(img, [int(np.ceil(sf*w/8+2)*8), int(np.ceil(sf*h/8+2)*8)])
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

        img_E = util.tensor2uint(img_E)
        img_E = img_E[:sf*w, :sf*h, :]

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()
            img_H = util.modcrop(img_H, sf)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

            if np.ndim(img_H) == 3:  # RGB image
                img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

        # ------------------------------------
        # save results
        # ------------------------------------
        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'.png'))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - x{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr, ave_ssim))
        if np.ndim(img_H) == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('Average PSNR/SSIM( Y ) - {} - x{} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr_y, ave_ssim_y))


if __name__ == '__main__':

    main()
