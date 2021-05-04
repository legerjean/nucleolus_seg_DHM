##############################
# IMPORTS
##############################

from __future__ import print_function
import numpy as np
from skimage.transform import rescale
from skimage.morphology import disk
from skimage.measure import label, regionprops
from scipy import ndimage, misc
import os
from PIL import Image

##############################
# LOAD DATA
##############################

NB_IMAGES = 75
PATH = '/export/home/jleger/Documents/segmentation/microscopy/raw/'
X_DIM = 1460
Y_DIM = 1920
X_DIM_PHASE = 364
Y_DIM_PHASE = 480

inputs_dapi_init = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
inputs_holo_init = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
inputs_phase_init = np.zeros((NB_IMAGES, X_DIM_PHASE, Y_DIM_PHASE))
inputs_gfp_init = np.zeros((NB_IMAGES, X_DIM, Y_DIM))

list_of_dir = os.listdir(PATH)
list_of_dir.sort()

for i in range(NB_IMAGES*4):
    im = Image.open(PATH + list_of_dir[i])
    image_array = np.array(im)
    image_type = i % 4
    image_index = int(i/4.0)
    if image_type == 0:
        inputs_dapi_init[image_index, :, :] = image_array
    elif image_type == 1:
        inputs_gfp_init[image_index, :, :] = image_array
    elif image_type == 2:
        inputs_holo_init[image_index, :, :] = image_array
    elif image_type == 3:
        inputs_phase_init[image_index, :, :] = image_array

print('--- Data loading finished ---')


##############################
# PREPROCESS DATA
##############################

# Normalization
vmax = 1
dapi_max = np.amax(inputs_dapi_init)
dapi_min = np.amin(inputs_dapi_init)
print(dapi_max)
print(dapi_min)
holo_max = np.amax(inputs_holo_init)
holo_min = np.amin(inputs_holo_init)
phase_max = np.amax(inputs_phase_init)
phase_min = np.amin(inputs_phase_init)
print(phase_max)
print(phase_min)
gfp_max = np.amax(inputs_gfp_init)
gfp_min = np.amin(inputs_gfp_init)
print(gfp_max)
print(gfp_min)

inputs_dapi = (inputs_dapi_init-dapi_min)*vmax / (dapi_max-dapi_min)
inputs_holo = (inputs_holo_init-holo_min)*vmax / (holo_max-holo_min)
inputs_gfp = (inputs_gfp_init-gfp_min)*vmax / (gfp_max-gfp_min)
inputs_phase = (inputs_phase_init-phase_min)*vmax / (phase_max-phase_min)

# Upsampling the data phase image
inputs_phase_up = []
sh = inputs_phase.shape
for i in range(sh[0]):
    inputs_phase_i = rescale(inputs_phase[i, :, :], 4, order=3, mode='reflect')
    inputs_phase_i = ndimage.shift(inputs_phase_i, -2, order=0, mode='reflect', prefilter=False)
    inputs_phase_up.append(inputs_phase_i)
inputs_phase = np.array(inputs_phase_up)

print('--- Preprocessing finished ---')


##############################
# GENERATE GT DAPI
##############################

PATH_GT = './samples/'
if not os.path.exists(PATH_GT):
    os.mkdir(PATH_GT)

# Nuclei GT for segmentation
labels_dapi = np.zeros(inputs_dapi.shape)
labels_dapi[inputs_dapi > 0.035] = 1
sh_dapi = labels_dapi.shape
labels_dapi_filtered = np.zeros(sh_dapi)

for i in range(sh_dapi[0]):
    labels_dapi_i = labels_dapi[i, :, :]

    # Remove small regions
    labels = label(labels_dapi_i, background=0)
    props = regionprops(labels)
    nuclei_clean = np.copy(labels_dapi_i)
    for region in props:
        label_index = region.label
        if region.area < 2000:
            nuclei_clean[labels == label_index] = 0

    # Closing
    sh = labels_dapi_i.shape
    margin = 5
    labels_dapi_extended_i = np.zeros((sh[0] + margin * 2, sh[1] + margin * 2))
    labels_dapi_extended_i[margin:margin + sh[0], margin:margin + sh[1]] = nuclei_clean
    se = disk(2)
    labels_dapi_close_i = ndimage.binary_closing(labels_dapi_extended_i, structure=se)
    labels_dapi_close_i = labels_dapi_close_i[margin:margin + sh[0], margin:margin + sh[1]]

    # Fill holes
    # mask = np.logical_not(nuclei_clean)
    # tmp = np.zeros(mask.shape, bool)
    # output = ndimage.binary_dilation(tmp, None, -1, mask, None, 1, 0)
    # np.logical_not(output, output)

    # Accumulate
    labels_dapi_filtered[i] = labels_dapi_close_i

labels_dapi_filtered = labels_dapi_filtered.astype(np.uint8)

print('--- GT generation finished ---')


##############################
# GENERATE SAMPLES
##############################

PATH_SMP = './samples/'

# Crop to power of 2^7
x_dim_round = 128 * (X_DIM // 128)
y_dim_round = 128 * (Y_DIM // 128)
inputs_dapi_cropped = inputs_dapi[:, 0:x_dim_round, 0:y_dim_round]
inputs_gfp_cropped = inputs_gfp[:, 0:x_dim_round, 0:y_dim_round]
inputs_holo_cropped = inputs_holo[:, 0:x_dim_round, 0:y_dim_round]
inputs_phase_cropped = inputs_phase[:, 0:x_dim_round, 0:y_dim_round]
labels_dapi_filtered_cropped = labels_dapi_filtered[:, 0:x_dim_round, 0:y_dim_round]


inputs_phase = inputs_phase*255
inputs_holo = inputs_holo*255
inputs_dapi = inputs_dapi*255
inputs_gfp = inputs_gfp*255
inputs_phase_cropped = inputs_phase_cropped*255
inputs_holo_cropped = inputs_holo_cropped*255
inputs_dapi_cropped = inputs_dapi_cropped*255
inputs_gfp_cropped = inputs_gfp_cropped*255

# inputs_phase = inputs_phase.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_phase_shifted_fullsize.npy')
# np.save(path,inputs_phase)
# inputs_phase_cropped = inputs_phase_cropped.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_phase_shifted.npy')
# np.save(path,inputs_phase_cropped)
#
# inputs_holo = inputs_holo.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_holo_fullsize.npy')
# np.save(path,inputs_holo)
# inputs_holo_cropped = inputs_holo_cropped.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_holo.npy')
# np.save(path,inputs_holo_cropped)
#
# inputs_dapi = inputs_dapi.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_dapi_fullsize.npy')
# np.save(path,inputs_dapi)
# inputs_dapi_cropped = inputs_dapi_cropped.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_dapi.npy')
# np.save(path,inputs_dapi_cropped)
#
# inputs_gfp = inputs_gfp.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_gfp_fullsize.npy')
# np.save(path,inputs_gfp)
# inputs_gfp_cropped = inputs_gfp_cropped.astype(np.uint8)
# path = os.path.join(PATH_SMP, 'images_gfp.npy')
# np.save(path,inputs_gfp_cropped)

labels_dapi = labels_dapi_filtered.astype(np.uint8)
path = os.path.join(PATH_GT, 'labels_dapi_close_fullsize.npy')
np.save(path,labels_dapi)
labels_dapi_cropped = labels_dapi_filtered_cropped.astype(np.uint8)
path = os.path.join(PATH_GT, 'labels_dapi_close.npy')
np.save(path,labels_dapi_cropped)

print('--- Samples saving finished ---')
print('--- Data generation successfully completed ---')