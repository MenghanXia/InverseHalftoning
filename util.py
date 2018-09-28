from __future__ import division
from __future__ import print_function
import os, glob, shutil, math
from scipy import signal
import tensorflow as tf
import numpy as np
from PIL import Image



def exists_or_mkdir(path, need_remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif need_remove:
        shutil.rmtree(path)
        os.makedirs(path)
    return None


def show_image_summary(img_batch):
    ## from [-1,1] to [0,1]
    return tf.image.convert_image_dtype((img_batch), dtype=tf.float32, saturate=True)


def save_list(save_path, data_list):
    n = len(data_list)
    with open(save_path, 'w') as f:
        f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None


def save_images_from_batch(img_batch, save_dir, init_no):
    if img_batch.shape[-1] == 3:
        ## rgb color image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, :]+1)+0.5).astype(np.uint8))
            image.save(os.path.join(save_dir, 'result_%05d.png' % (init_no + i)), 'PNG')
    else:
        ## single-channel gray image
        for i in range(img_batch.shape[0]):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i, :, :, 0]+1)+0.5).astype(np.uint8))
            image.save(os.path.join(save_dir, 'result_%05d.png' % (init_no + i)), 'PNG')
    return None


def measure_quality(im_batch1, im_batch2):
    mean_psnr = 0
    mean_ssim = 0
    im_batch1 = im_batch1.squeeze()
    im_batch2 = im_batch2.squeeze()
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        psnr = calc_psnr(im1, im2)
        ssim = calc_ssim(im1, im2)
        mean_psnr += psnr
        mean_ssim += ssim
    return mean_psnr/num, mean_ssim/num


def measure_psnr(im_batch1, im_batch2):
    mean_psnr = 0
    num = im_batch1.shape[0]
    for i in range(num):
        # Convert pixel value to [0,255]
        im1 = 127.5 * (im_batch1[i]+1)
        im2 = 127.5 * (im_batch2[i]+1)
        psnr = calc_psnr(im1, im2)
        mean_psnr += psnr
    return mean_psnr/num


def calc_psnr(im1, im2):
    '''
    Notice: Pixel value should be convert to [0,255]
    '''
    if im1.shape[-1] != 3:
        g_im1 = im1.astype(np.float32)
        g_im2 = im2.astype(np.float32)
    else:
        g_im1 = np.array(Image.fromarray(im1).convert('L'), np.float32)
        g_im2 = np.array(Image.fromarray(im2).convert('L'), np.float32)

    mse = np.mean((g_im1 - g_im2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_ssim(im1, im2):
    """
    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    '''
    Notice: Pixel value should be convert to [0,255]
    '''
    if np.max(im1) <= 1.0:
        print('Error: pixel value should be converted to [0,255] !')
        return None
    if im1.shape[-1] != 3:
        g_im1 = im1.astype(np.float32)
        g_im2 = im2.astype(np.float32)
    else:
        g_im1 = np.array(Image.fromarray(im1).convert('L'), np.float32)
        g_im2 = np.array(Image.fromarray(im2).convert('L'), np.float32)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, g_im1, mode='valid')
    mu2 = signal.fftconvolve(window, g_im2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, g_im1 * g_im1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, g_im2 * g_im2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, g_im1 * g_im2, mode='valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()