from __future__ import division
import os
import time
from glob import glob
from utils import *
from sklearn import preprocessing
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def read_images(c_dim,config):
    is_grayscale = (c_dim == 1)
    real_data = glob(os.path.join("../data/denoise/grayOriginal", "*.jpg"))
    noise_data = glob(os.path.join("../data/denoise/graygauss", "*.jpg"))

    real = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in real_data]
    noise = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in noise_data]

    if is_grayscale:
        reals = np.array(real).astype(np.float32)[:,:,:,None]
        noises = np.array(noise).astype(np.float32)[:,:,:,None]
    else:
        reals = np.array(real).astype(np.float32)
        noises = np.array(noise).astype(np.float32)

    return reals, noises

def read_images2(c_dim,config):
    is_grayscale = (c_dim == 1)
    # real_data = glob(os.path.join("./data/simsun", "*.npy"))
    # noise_data = glob(os.path.join("./data/jg", "*.npy"))
    #
    # real = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in real_data]
    # noise = [get_image(img_file, config.image_size, is_crop=False, is_grayscale=is_grayscale) for img_file in noise_data]

    if is_grayscale:
        noises = np.load('../data/simsun80.npy').astype(np.float32)[:,:,:,None]
        reals = np.load('../data/jg.npy').astype(np.float32)[:,:,:,None]

        # noises = np.load('./data/simsun80_norm.npy').astype(np.float32)[:, :, :, None]
        # reals = np.load('./data/jgnorm.npy').astype(np.float32)[:, :, :, None]
        #reals = np.array(real).astype(np.float32)[:,:,:,None]
        #noises = np.array(noise).astype(np.float32)[:,:,:,None]
    else:
        reals = np.array(real).astype(np.float32)
        noises = np.array(noise).astype(np.float32)

    return reals, noises

def resize_mnist(path, shape):
    data = input_data.read_data_sets(path,reshape=False)
    imgs = data.train.images
    r = np.random.permutation(len(imgs))
    imgs = imgs[r,:]

    imgs_new = []
    for img in imgs:
        imgs_new.append(resize(img,shape).astype(np.float32))

    np.save('mnist_resize_train.npy',imgs_new)
    print ('file saved')


def resize_imgs(path,filename,shape,shuffle=True):
    imgs = np.load(path+filename).astype(np.float32)
    imgs = np.ones_like(imgs,dtype=np.float32)-imgs     # 变成黑底白字
    if shuffle:
        np.random.shuffle(imgs)
    imgs_new = []
    for img in imgs:
        imgs_new.append(resize(img,shape).astype(np.float32))
    np.save(path+'resize_'+filename,imgs_new)
    print ('file saved')

def load_mnist(path):
    return np.load('mnist_resize_train.npy')

def load_imgs(path):
    return np.load(path)

# resize_mnist('./MNIST_data',shape=(32,32))
# resize_imgs('../data/','simsun80_norm.npy',(32,32,1))