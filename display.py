
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



from scipy.misc import imsave

#########################
# DHM images
#########################
gfp_image = np.load('./samples/images_phase_shifted.npy')
mask_1 = np.load(
    '/export/home/jleger/Documents/segmentation/microscopy/results/model_unet_2d_n_layers_5_modality_labels_nucleoli_RG_t0125_v17_wo_holes_dilated1/predictions.npy')
# mask_2 = np.load('./samples/labels_nucleoli_manual_shifted_0.npy')
# mask_3 = np.load(
#     '/export/home/jleger/Documents/segmentation/microscopy/results/model_unet_2d_n_layers_5_modality_labels_dapi_filled/predictions.npy')  # to be checked
#
# border_mask = generate_border_mask(mask_3)
# cluster_mask = generate_cluster_mask(mask_3)
#
# mask_4 = np.multiply(np.multiply(np.multiply(mask_3, 1 - border_mask), 1 - cluster_mask), mask_3)
# mask_5 = np.load(
#     './samples/labels_dapi_close.npy')
# mask_5_shifted = ndimage.shift(mask_5, [0, 0, 8], order=0, mode='reflect', prefilter=False)
# mask_5_shifted_uint8 = mask_5_shifted.astype(np.uint8)
# np.save('./samples/labels_dapi_close_shifted.npy', mask_5_shifted_uint8)
# sh = mask_5_shifted.shape
# mask_5_shifted_eroded = np.zeros(sh)
# for i in range(sh[0]):
#     mask_5_shifted_i = mask_5_shifted[i]
#     mask_5_shifted_eroded[i] = ndimage.binary_erosion(mask_5_shifted_i, structure=disk(10))
# mask_5_shifted_eroded = mask_5_shifted_eroded.astype(np.uint8)
# np.save('./samples/labels_dapi_close_shifted_eroded10.npy', mask_5_shifted_eroded)
#
# samples_nucleoli = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0.npy')
# samples_nucleoli_cropped = np.multiply(samples_nucleoli, mask_5_shifted_eroded)
# samples_nucleoli_cropped = samples_nucleoli_cropped.astype(np.uint8)
# np.save('./samples/samples_nucleoli_cropped.npy', samples_nucleoli_cropped)
#
# mask_6 = np.load(
#     '/export/home/jleger/Documents/segmentation/microscopy/results/model_unet_2d_n_layers_5_modality_labels_dapi_close_shifted/predictions.npy')

for i, j in zip(range(25), range(25)):

    mask_improved_i_contour = segmentation.mark_boundaries(gfp_image[i], mask_1[i], color=[255, 0, 0])
    #mask_improved_i_contour2 = segmentation.mark_boundaries(mask_improved_i_contour, mask_2[i], color=[255, 255, 0])
    #mask_improved_i_contour3 = segmentation.mark_boundaries(gfp_image[i], mask_3[i], color=[255, 255, 255])
    #mask_improved_i_contour4 = segmentation.mark_boundaries(mask_improved_i_contour3, mask_4[i], color=[0, 255, 0])
    #mask_improved_i_contour5 = segmentation.mark_boundaries(mask_improved_i_contour4, mask_5[i], color=[0, 255, 255])
    #mask_improved_i_contour6 = segmentation.mark_boundaries(mask_improved_i_contour2, mask_5_shifted[i], color=[255, 0, 255])

    #mask_improved_i_contour7 = segmentation.mark_boundaries(mask_improved_i_contour6, mask_5_shifted_eroded[i], color=[255, 0, 255])
    #mask_improved_i_contour8 = segmentation.mark_boundaries(gfp_image[i], mask_6[i], color=[255, 0, 255])

    misc.toimage(mask_improved_i_contour * 255, cmin=0, cmax=255).save(
    './figures_clean/image_' + str(i) + '_dhm_predictions17.png')


################################
# Fluo images
################################
# gfp_image = np.load('./samples/images_gfp_shifted_0.npy')
# mask_1 = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0.npy')
# mask_11 = np.load('./samples/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0.npy')
# mask_2 = np.load('./samples/labels_nucleoli_manual_shifted_0.npy')
# mask_3 = np.load('./samples/labels_dapi_close.npy')  # to be checked
#
# border_mask = generate_border_mask(mask_3)
# cluster_mask = generate_cluster_mask(mask_3)
#
# mask_4 = np.multiply(np.multiply(np.multiply(mask_3, 1 - border_mask), 1 - cluster_mask), mask_3)
#
# for i, j in zip(range(25), range(25)):
#
#     mask_improved_i_contour = segmentation.mark_boundaries(gfp_image[i], mask_2[i], color=[255, 0, 0])
#     mask_improved_i_contour11 = segmentation.mark_boundaries(mask_improved_i_contour, mask_11[i], color=[0, 0, 255])
#     mask_improved_i_contour2 = segmentation.mark_boundaries(mask_improved_i_contour11, mask_2[i], color=[255, 255, 0])
#     mask_improved_i_contour3 = segmentation.mark_boundaries(mask_improved_i_contour2, mask_3[i], color=[255, 255, 255])
#     mask_improved_i_contour4 = segmentation.mark_boundaries(mask_improved_i_contour3, mask_4[i], color=[0, 255, 0])
#
#     misc.toimage(mask_improved_i_contour * 255, cmin=0, cmax=85).save(
#     './figures_clean/image_' + str(i) + '_fluo_v18_manual.png')
#
