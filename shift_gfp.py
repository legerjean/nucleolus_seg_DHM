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
from skimage.color import rgb2gray

#images_gfp = np.load('./samples/images_gfp.npy')
#images_gfp_shifted_0 = ndimage.shift(images_gfp[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)

nucleoli_RG_rt045_v18_wo_holes = np.load('./samples/labels_nucleoli_RG_rt045_v18_wo_holes.npy')
nucleoli_RG_rt045_v18_wo_holes_shifted_0 = ndimage.shift(nucleoli_RG_rt045_v18_wo_holes[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)
nucleoli_RG_t015_v18_wo_holes_dilated2 = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2.npy')
nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0 = ndimage.shift(nucleoli_RG_t015_v18_wo_holes_dilated2[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)

nucleoli_RG_rt045_v17_wo_holes = np.load('./samples/labels_nucleoli_RG_rt045_v17_wo_holes.npy')
nucleoli_RG_rt045_v17_wo_holes_shifted_0 = ndimage.shift(nucleoli_RG_rt045_v17_wo_holes[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)
nucleoli_RG_t0125_v17_wo_holes_dilated1 = np.load('./samples/labels_nucleoli_RG_t0125_v17_wo_holes_dilated1.npy')
nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted_0 = ndimage.shift(nucleoli_RG_t0125_v17_wo_holes_dilated1[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)

#labels_nucleoli_manual = np.load('./samples/labels_nucleoli_manual.npy')
#labels_nucleoli_manual_shifted_0 = ndimage.shift(labels_nucleoli_manual[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)

#images_gfp_shifted = np.copy(images_gfp)
#images_gfp_shifted[0] = images_gfp_shifted_0
nucleoli_RG_rt045_v18_wo_holes_shifted = np.copy(nucleoli_RG_rt045_v18_wo_holes)
nucleoli_RG_rt045_v18_wo_holes_shifted[0] = nucleoli_RG_rt045_v18_wo_holes_shifted_0
nucleoli_RG_t015_v18_wo_holes_dilated2_shifted = np.copy(nucleoli_RG_t015_v18_wo_holes_dilated2)
nucleoli_RG_t015_v18_wo_holes_dilated2_shifted[0] = nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0
nucleoli_RG_rt045_v17_wo_holes_shifted = np.copy(nucleoli_RG_rt045_v17_wo_holes)
nucleoli_RG_rt045_v17_wo_holes_shifted[0] = nucleoli_RG_rt045_v17_wo_holes_shifted_0
nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted = np.copy(nucleoli_RG_t0125_v17_wo_holes_dilated1)
nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted[0] = nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted_0
#labels_nucleoli_manual_shifted = np.copy(labels_nucleoli_manual)
#labels_nucleoli_manual_shifted[0] = labels_nucleoli_manual_shifted_0


#np.save('./samples/images_gfp_shifted_0.npy', images_gfp_shifted)
np.save('./samples/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0.npy', nucleoli_RG_rt045_v18_wo_holes_shifted)
np.save('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0.npy', nucleoli_RG_t015_v18_wo_holes_dilated2_shifted)
np.save('./samples/labels_nucleoli_RG_rt045_v17_wo_holes_shifted_0.npy', nucleoli_RG_rt045_v17_wo_holes_shifted)
np.save('./samples/labels_nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted_0.npy', nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted)
#np.save('./samples/labels_nucleoli_manual_shifted_0.npy', labels_nucleoli_manual_shifted)
