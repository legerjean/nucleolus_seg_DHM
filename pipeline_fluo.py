from __future__ import print_function
import numpy as np
from skimage.transform import rescale
from skimage.morphology import disk, flood
from skimage.measure import label, regionprops
from scipy import ndimage, misc
import os
from PIL import Image


def load_and_preprocess_data():
    #############
    # Load data #
    #############
    NB_IMAGES = 75
    PATH = './raw/'
    X_DIM = 1460
    Y_DIM = 1920
    X_DIM_PHASE = 364
    Y_DIM_PHASE = 480

    inputs_dapi_init = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
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
        elif image_type == 3:
            inputs_phase_init[image_index, :, :] = image_array

    print('--- Data loading finished ---')

    ##################
    # Normalize data #
    ##################
    vmax = 1
    dapi_max = np.amax(inputs_dapi_init)
    dapi_min = np.amin(inputs_dapi_init)
    phase_max = np.amax(inputs_phase_init)
    phase_min = np.amin(inputs_phase_init)
    gfp_max = np.amax(inputs_gfp_init)
    gfp_min = np.amin(inputs_gfp_init)

    inputs_dapi = (inputs_dapi_init-dapi_min)*vmax / (dapi_max-dapi_min)
    inputs_gfp = (inputs_gfp_init-gfp_min)*vmax / (gfp_max-gfp_min)
    inputs_phase = (inputs_phase_init-phase_min)*vmax / (phase_max-phase_min)

    #################################
    # Upsample the data phase image #
    #################################
    # Phase images are slightly shifted in order to be better aligned with the fluorescence images
    inputs_phase_up = []
    sh = inputs_phase.shape
    for i in range(sh[0]):
        inputs_phase_i = rescale(inputs_phase[i, :, :], 4, order=3, mode='reflect')
        inputs_phase_i = ndimage.shift(inputs_phase_i, -2, order=0, mode='reflect', prefilter=False)
        inputs_phase_up.append(inputs_phase_i)
    inputs_phase = np.array(inputs_phase_up)

    print('--- Preprocessing finished ---')

    #############
    # Save data #
    #############
    PATH_SMP = './samples/'

    # Crop to power of 2^7
    x_dim_round = 128 * (X_DIM // 128)
    y_dim_round = 128 * (Y_DIM // 128)
    inputs_dapi_cropped = inputs_dapi[:, 0:x_dim_round, 0:y_dim_round]
    inputs_gfp_cropped = inputs_gfp[:, 0:x_dim_round, 0:y_dim_round]
    inputs_phase_cropped = inputs_phase[:, 0:x_dim_round, 0:y_dim_round]

    inputs_phase = inputs_phase * 255
    inputs_dapi = inputs_dapi * 255
    inputs_gfp = inputs_gfp * 255
    inputs_phase_cropped = inputs_phase_cropped * 255
    inputs_dapi_cropped = inputs_dapi_cropped * 255
    inputs_gfp_cropped = inputs_gfp_cropped * 255

    inputs_phase = inputs_phase.astype(np.uint8)
    path = os.path.join(PATH_SMP, 'images_phase_shifted_fullsize.npy')
    np.save(path, inputs_phase)
    inputs_phase_cropped = inputs_phase_cropped.astype(np.uint8)
    path = os.path.join(PATH_SMP, 'images_phase_shifted.npy')
    np.save(path, inputs_phase_cropped)

    inputs_dapi = inputs_dapi.astype(np.uint8)
    path = os.path.join(PATH_SMP, 'images_dapi_fullsize.npy')
    np.save(path, inputs_dapi)
    inputs_dapi_cropped = inputs_dapi_cropped.astype(np.uint8)
    path = os.path.join(PATH_SMP, 'images_dapi.npy')
    np.save(path, inputs_dapi_cropped)

    inputs_gfp = inputs_gfp.astype(np.uint8)
    path = os.path.join(PATH_SMP, 'images_gfp_fullsize.npy')
    np.save(path, inputs_gfp)
    inputs_gfp_cropped = inputs_gfp_cropped.astype(np.uint8)
    path = os.path.join(PATH_SMP, 'images_gfp.npy')
    np.save(path, inputs_gfp_cropped)

    print('--- Samples saved ---')


def dapi_segmentation():
    PATH_GT = './samples/'
    X_DIM = 1460
    Y_DIM = 1920
    if not os.path.exists(PATH_GT):
        os.mkdir(PATH_GT)

    inputs_dapi = np.load('./samples/images_dapi_fullsize.npy')
    inputs_dapi = inputs_dapi.astype(np.double)/255
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

        # Accumulate
        labels_dapi_filtered[i] = labels_dapi_close_i

    labels_dapi_filtered = labels_dapi_filtered.astype(np.uint8)

    # Crop and save
    x_dim_round = 128 * (X_DIM // 128)
    y_dim_round = 128 * (Y_DIM // 128)
    labels_dapi_filtered_cropped = labels_dapi_filtered[:, 0:x_dim_round, 0:y_dim_round]
    labels_dapi = labels_dapi_filtered.astype(np.uint8)
    path = os.path.join(PATH_GT, 'labels_dapi_close_fullsize.npy')
    np.save(path, labels_dapi)
    labels_dapi_cropped = labels_dapi_filtered_cropped.astype(np.uint8)
    path = os.path.join(PATH_GT, 'labels_dapi_close.npy')
    np.save(path, labels_dapi_cropped)

    print('--- Nuclei segmentation finished ---')


def seeds_generation():
    inputs_gfp = np.load('./samples/images_gfp_fullsize.npy')
    inputs_gfp = inputs_gfp.astype(np.double)
    inputs_gfp = np.divide(inputs_gfp, 255)

    # Nucleoli GT for counting
    thr_min = 0.08  # 0.1
    thr_max = 0.23  # 0.5
    thr_list = np.arange(thr_min, thr_max, 0.025)

    NB_IMAGES = 75
    X_DIM = 1460
    Y_DIM = 1920
    gfp_centers_init = np.zeros((NB_IMAGES, X_DIM, Y_DIM))

    for i in range(75):
        my_gfp = inputs_gfp[i]
        gfp_labels = label(np.ones((X_DIM, Y_DIM)), background=0)
        gfp_regions = regionprops(gfp_labels, intensity_image=my_gfp)

        for counter, thr in enumerate(thr_list):
            # Thresholding and morphological operation
            gfp_thr = np.zeros((X_DIM, Y_DIM))
            gfp_thr[my_gfp > thr] = 1
            se = disk(2)
            gfp_open = ndimage.binary_opening(gfp_thr, structure=se)  # retirer les pixels isolés
            se = disk(3)
            gfp_close = ndimage.binary_closing(gfp_open, structure=se)  # relier les petits pixels adjacents
            gfp_final = np.copy(gfp_close)

            # Assign the previous label at the new subregions
            gfp_new_labels = np.multiply(gfp_labels, gfp_open)

            # Update the labels
            for reg in gfp_regions:
                # Get only the subregions with the same label
                lab = reg.label
                gfp_indiv = np.zeros(my_gfp.shape)
                gfp_indiv[gfp_new_labels == lab] = 1
                gfp_indiv_labels = label(gfp_indiv, background=0)
                gfp_indiv_regions = regionprops(gfp_indiv_labels, my_gfp)
                nb_subregions = len(gfp_indiv_regions)

                # If an existing region disappears, rewrite it
                if nb_subregions == 0:
                    gfp_final[gfp_labels == lab] = 1

            gfp_labels = label(gfp_final, background=0)
            gfp_regions = regionprops(gfp_labels, intensity_image=my_gfp)

        gfp_centers_init[i] = gfp_final

    gfp_centers_init = gfp_centers_init.astype(np.uint8)
    np.save('./samples/seeds_min008_max023_step0025_open2_close3.npy', gfp_centers_init)

    print('--- Seeds generation finished ---')


def region_growing():
    # Load images
    gfp_images = np.load('./samples/images_gfp_fullsize.npy')
    masks_manual = np.load('./samples/seeds_min008_max023_step0025_open2_close3.npy')
    sh = masks_manual.shape
    RG_045_improved = np.zeros((75, sh[1], sh[2]))
    RG_015_improved = np.zeros((75, sh[1], sh[2]))

    counter = 0
    for i in range(75):
        masks_manual_i = masks_manual[i]
        gfp_image_i = gfp_images[i]
        RG_045_improved_i = np.zeros(masks_manual_i.shape)
        RG_015_improved_i = np.zeros(masks_manual_i.shape)

        gfp_labels = label(masks_manual_i, background=0)
        gfp_regions = regionprops(gfp_labels, intensity_image=gfp_image_i)

        for reg in gfp_regions:
            lab = reg.label

            # Get maximum
            region_mask = np.zeros(gfp_image_i.shape)
            region_mask[gfp_labels == lab] = 1
            gfp_masked_on_region = np.multiply(gfp_image_i, region_mask)

            maxi = np.amax(gfp_masked_on_region)
            index_maxi = np.unravel_index(np.argmax(gfp_masked_on_region, axis=None), gfp_masked_on_region.shape)

            # Region growing
            #tol_045 = max(0, min(maxi * 0.45, maxi - 0.1 * 255))
            tol_045 = max(0, min(maxi * 0.45, maxi - 0.15 * 255))
            tol_015 = max(0, maxi - 0.15 * 255)
            tol_010 = max(0, maxi - 0.1 * 255)
            region_flood_045 = flood(gfp_image_i, index_maxi, connectivity=2, tolerance=tol_045)
            region_flood_010 = flood(gfp_image_i, index_maxi, connectivity=2, tolerance=tol_010)
            region_flood_015 = flood(gfp_image_i, index_maxi, connectivity=2, tolerance=tol_015)

            area_flood = np.sum(region_flood_010)
            max_flood = max(gfp_image_i[region_flood_010])

            current_mask_045 = np.zeros(gfp_image_i.shape)
            current_mask_015 = np.zeros(gfp_image_i.shape)
            if area_flood > 4500:  # mich too big
                current_mask_015[region_flood_015 == 1] = 0
                current_mask_045[region_flood_045 == 1] = 0
                RG_015_improved_i = np.logical_or(RG_015_improved_i, current_mask_015)
                RG_045_improved_i = np.logical_or(RG_045_improved_i, current_mask_045)
            if area_flood > 1000:
                if max_flood < 65:  # mitose
                    current_mask_015[region_flood_015 == 1] = 0
                    current_mask_045[region_flood_045 == 1] = 0
                    RG_015_improved_i = np.logical_or(RG_015_improved_i, current_mask_015)
                    RG_045_improved_i = np.logical_or(RG_045_improved_i, current_mask_045)
                else:
                    current_mask_015[region_flood_015 == 1] = 1
                    current_mask_045[region_flood_045 == 1] = 1
                    RG_015_improved_i = np.logical_or(RG_015_improved_i, current_mask_015)
                    RG_045_improved_i = np.logical_or(RG_045_improved_i, current_mask_045)
            else:
                current_mask_015[region_flood_015 == 1] = 1
                current_mask_045[region_flood_045 == 1] = 1
                RG_015_improved_i = np.logical_or(RG_015_improved_i, current_mask_015)
                RG_045_improved_i = np.logical_or(RG_045_improved_i, current_mask_045)

        RG_015_improved[counter] = RG_015_improved_i
        RG_045_improved[counter] = RG_045_improved_i

        counter = counter + 1

    X_DIM = 1460
    Y_DIM = 1920
    x_dim_round = 128 * (X_DIM // 128)
    y_dim_round = 128 * (Y_DIM // 128)
    RG_015_improved_cropped = RG_015_improved[:, 0:x_dim_round, 0:y_dim_round]
    RG_045_improved_cropped = RG_045_improved[:, 0:x_dim_round, 0:y_dim_round]

    RG_015_improved_cropped = RG_015_improved_cropped.astype(np.uint8)
    np.save('./samples/labels_nucleoli_RG_t015_v18.npy', RG_015_improved_cropped)
    RG_015_improved = RG_015_improved.astype(np.uint8)
    np.save('./samples/labels_nucleoli_RG_t015_v18_fullsize.npy', RG_015_improved)
    RG_045_improved_cropped = RG_045_improved_cropped.astype(np.uint8)
    np.save('./samples/labels_nucleoli_RG_rt045_v18.npy', RG_045_improved_cropped)
    RG_045_improved = RG_045_improved.astype(np.uint8)
    np.save('./samples/labels_nucleoli_RG_rt045_v18_fullsize.npy', RG_045_improved)


def fill_holes(input, folder_name, structure=None, origin=0):
    sh = input.shape
    input_filled = np.zeros(sh)
    for i in range(sh[0]):
        mask = np.logical_not(input[i])
        tmp = np.zeros(mask.shape, bool)
        output = ndimage.binary_dilation(tmp, structure, -1, mask, None, 1, origin)
        np.logical_not(output, output)
        input_filled[i] = output

    input_filled = input_filled.astype(np.uint8)
    np.save(folder_name, input_filled)


def dilate_thresholding():
    thr_results = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes.npy')
    sh = thr_results.shape
    dilated_images = np.zeros(sh)
    for i in range(sh[0]):
        thr_results_i = thr_results[i]
        dilated_images[i] = ndimage.binary_dilation(thr_results_i, structure=disk(2))
    dilated_images = dilated_images.astype(np.uint8)
    np.save('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2.npy', dilated_images)


def shift_gfp():
    images_gfp = np.load('./samples/images_gfp.npy')
    images_gfp_shifted_0 = ndimage.shift(images_gfp[0], np.array([-5, -10]), order=0, mode='reflect', prefilter=False)
    nucleoli_RG_rt045_v18_wo_holes = np.load('./samples/labels_nucleoli_RG_rt045_v18_wo_holes.npy')
    nucleoli_RG_rt045_v18_wo_holes_shifted_0 = ndimage.shift(nucleoli_RG_rt045_v18_wo_holes[0], np.array([-5, -10]),
                                                             order=0, mode='reflect', prefilter=False)
    nucleoli_RG_t015_v18_wo_holes_dilated2 = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2.npy')
    nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0 = ndimage.shift(nucleoli_RG_t015_v18_wo_holes_dilated2[0],
                                                                     np.array([-5, -10]), order=0, mode='reflect',
                                                                     prefilter=False)

    images_gfp_shifted = np.copy(images_gfp)
    images_gfp_shifted[0] = images_gfp_shifted_0
    nucleoli_RG_rt045_v18_wo_holes_shifted = np.copy(nucleoli_RG_rt045_v18_wo_holes)
    nucleoli_RG_rt045_v18_wo_holes_shifted[0] = nucleoli_RG_rt045_v18_wo_holes_shifted_0
    nucleoli_RG_t015_v18_wo_holes_dilated2_shifted = np.copy(nucleoli_RG_t015_v18_wo_holes_dilated2)
    nucleoli_RG_t015_v18_wo_holes_dilated2_shifted[0] = nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0

    np.save('./samples/images_gfp_shifted_0.npy', images_gfp_shifted)
    np.save('./samples/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0.npy', nucleoli_RG_rt045_v18_wo_holes_shifted)
    np.save('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0.npy',
            nucleoli_RG_t015_v18_wo_holes_dilated2_shifted)


def gfp_segmentation():
    seeds_generation()
    region_growing()
    fill_holes(np.load('./samples/labels_nucleoli_RG_rt045_v18.npy'),
               './samples/labels_nucleoli_RG_rt045_v18_wo_holes.npy')
    fill_holes(np.load('./samples/labels_nucleoli_RG_t015_v18.npy'),
               './samples/labels_nucleoli_RG_t015_v18_wo_holes.npy')
    dilate_thresholding()
    # The first GFP image in not perfectly aligned with the other modalities, so here is a quick fix
    shift_gfp()


load_and_preprocess_data()
dapi_segmentation()
gfp_segmentation()


