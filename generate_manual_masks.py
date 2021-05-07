from __future__ import print_function
import numpy as np
from skimage.transform import rescale
from skimage.morphology import disk
from skimage.measure import label, regionprops
from scipy import ndimage, misc
import os
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#############
# Load data #
#############

NB_IMAGES = 25
PATH = './manual_masks/'
X_DIM = 1460
Y_DIM = 1920

inputs_masks = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
x_dim_round = 128 * (X_DIM // 128)
y_dim_round = 128 * (Y_DIM // 128)

for i in range(NB_IMAGES):
    im = Image.open(PATH + str(i) + '.png')
    image_array = np.array(im)
    im_gray = rgb2gray(image_array)
    im_thr = np.zeros(im_gray.shape)
    im_thr[im_gray<0.7] = 1
    inputs_masks[i, :, :] = im_thr

inputs_masks_cropped = inputs_masks[:, 0:x_dim_round, 0:y_dim_round]

print('--- Data loading finished ---')

#############
# Save data #
#############

PATH_SMP = './samples/'
if not os.path.exists(PATH_SMP):
    os.mkdir(PATH_SMP)

inputs_masks = inputs_masks.astype(np.uint8)
path = os.path.join(PATH_SMP, 'labels_nucleoli_manual_fullsize.npy')
np.save(path, inputs_masks)
inputs_masks_cropped = inputs_masks_cropped.astype(np.uint8)
path = os.path.join(PATH_SMP, 'labels_nucleoli_manual.npy')
np.save(path, inputs_masks_cropped)

print('--- Samples saving finished ---')

##############
# Shift data #
##############

labels_nucleoli_manual = np.load('./samples/labels_nucleoli_manual.npy')
labels_nucleoli_manual_shifted_0 = ndimage.shift(labels_nucleoli_manual[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)
labels_nucleoli_manual_shifted = np.copy(labels_nucleoli_manual)
labels_nucleoli_manual_shifted[0] = labels_nucleoli_manual_shifted_0
np.save('./samples/labels_nucleoli_manual_shifted_0.npy', labels_nucleoli_manual_shifted)
