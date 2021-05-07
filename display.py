
from __future__ import print_function
import numpy as np
from skimage.transform import rescale
from skimage.morphology import disk
from skimage.measure import label, regionprops
from scipy import ndimage
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion
from skimage.morphology.selem import disk
from skimage import segmentation
from scipy import misc
#from preprocessing import *
from postprocessing import *
from skimage.color import rgb2gray
from imageio import imwrite

#########################
# DHM images
#########################
# phase_image = np.load('./samples/images_phase_shifted.npy')
# mask_1 = np.load(
#     './results/model_unet_2d_n_layers_5_modality_labels_nucleoli_RG_t015_v18_wo_holes_dilated2/predictions.npy')
# mask_2 = np.load('./samples/labels_nucleoli_manual_shifted_0.npy')
# mask_3 = np.load(
#     './results/model_unet_2d_n_layers_5_modality_labels_dapi_filled/predictions.npy')
# border_mask = generate_border_mask(mask_3)
# cluster_mask = generate_cluster_mask(mask_3)
# mask_4 = np.multiply(np.multiply(np.multiply(mask_3, 1 - border_mask), 1 - cluster_mask), mask_3)
#
# for i in range(25):
#     mask_improved_i_contour = segmentation.mark_boundaries(phase_image[i], np.multiply(mask_1[i], mask_3[i]), color=[1, 0, 0])                # automatic segmentation of nucleoli
#     mask_improved_i_contour2 = segmentation.mark_boundaries(mask_improved_i_contour, mask_2[i], color=[1, 1, 0])    # manual contours
#     mask_improved_i_contour3 = segmentation.mark_boundaries(mask_improved_i_contour2, mask_3[i], color=[1, 1, 1])           # discarded cells
#     mask_improved_i_contour4 = segmentation.mark_boundaries(mask_improved_i_contour3, mask_4[i], color=[0, 1, 0])     # cell nuclei
#     mask_improved_i_contour4 = mask_improved_i_contour4*255
#     mask_improved_i_contour4_uint8 = mask_improved_i_contour4.astype(np.uint8)
#     imwrite('./figures/image_' + str(i) + '_dhm_contours.png', mask_improved_i_contour4_uint8)


################################
# Fluo images
################################
gfp_image = np.load('./samples/images_gfp_shifted_0.npy')
mask_1 = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0.npy')
mask_11 = np.load('./samples/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0.npy')
mask_2 = np.load('./samples/labels_nucleoli_manual_shifted_0.npy')
mask_3 = np.load('./samples/labels_dapi_close.npy')  # to be checked
border_mask = generate_border_mask(mask_3)
cluster_mask = generate_cluster_mask(mask_3)
mask_4 = np.multiply(np.multiply(np.multiply(mask_3, 1 - border_mask), 1 - cluster_mask), mask_3)
gfp_image[gfp_image > 85] = 85
gfp_image = gfp_image.astype(np.double)*255/85
gfp_image = gfp_image.astype(np.uint8)
for i in range(25):
    mask_improved_i_contour = segmentation.mark_boundaries(gfp_image[i], np.multiply(mask_1[i], mask_3[i]), color=[1, 0, 0])
    mask_improved_i_contour11 = segmentation.mark_boundaries(mask_improved_i_contour, np.multiply(mask_11[i], mask_3[i]), color=[0, 0, 1])
    mask_improved_i_contour2 = segmentation.mark_boundaries(mask_improved_i_contour11, mask_2[i], color=[1, 1, 0])
    mask_improved_i_contour3 = segmentation.mark_boundaries(mask_improved_i_contour2, mask_3[i], color=[1, 1, 1])
    mask_improved_i_contour4 = segmentation.mark_boundaries(mask_improved_i_contour3, mask_4[i], color=[0, 1, 0])
    mask_improved_i_contour4 = mask_improved_i_contour4 * 255
    mask_improved_i_contour4_uint8 = mask_improved_i_contour4.astype(np.uint8)
    imwrite('./figures/image_' + str(i) + '_fluo_contours.png', mask_improved_i_contour4_uint8)


