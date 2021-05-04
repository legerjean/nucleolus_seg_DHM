from core.utils_core import *
from core.train import *
from core.predict import *
import numpy as np
from skimage.measure import label, regionprops
# from postprocessing import label_simple
from scipy import ndimage, misc
from skimage.segmentation import active_contour
from skimage.morphology import disk
from skimage.color import label2rgb
from skimage import segmentation


# #####################################
# # Improve manual masks
# #####################################

def improve_manual_masks():
    # Load images
    gfp_images = np.load('./samples/images_gfp_fullsize.npy')
    masks_manual = np.load('./samples/labels_nucleoli_manual_fullsize.npy')
    masks_manual_improved = np.zeros(masks_manual.shape)
    sh = masks_manual.shape

    for i in range(sh[0]):
        gfp_image_i = gfp_images[i]
        masks_manual_i = masks_manual[i]
        mask_improved_i = np.zeros(masks_manual_i.shape)

        # Statistics
        gfp_labels = label(masks_manual_i, background=0)
        gfp_regions = regionprops(gfp_labels, intensity_image=gfp_image_i)

        # ax = plt.figure(figsize=(14, 5))
        for reg in gfp_regions:
            lab = reg.label
            # bb_intensities = reg.intensity_image
            # bb_mask = reg.image

            # Compute threshold for each nucleoli
            region_intensities = gfp_image_i[gfp_labels == lab]
            region_intensities = np.sort(region_intensities)
            max = np.mean(region_intensities[-10:])  # average on 10 largest values
            thr = 0.5 * max

            # Perform thresholding for each nucleoli
            region_mask = np.zeros(gfp_image_i.shape)
            region_mask[gfp_labels == lab] = 1
            gfp_masked_on_region = np.multiply(gfp_image_i, region_mask)
            mask_improved_i[gfp_masked_on_region > thr] = 1

            # region_intensities_n = np.divide(region_intensities, max)
            # ax = plt.subplot(5, 14, lab)
            # ax.hist(region_intensities_n, bins=25, histtype='bar')
            # ax.set_xlim([0, 1.1])
            # ax.set_ylim([0, 100])

        se = disk(3)
        mask_improved_i_close = ndimage.binary_closing(mask_improved_i, structure=se)  # relier les petits pixels adjacents
        mask_improved_i_final = np.multiply(mask_improved_i_close, masks_manual_i)
        mask_improved_i_contour = segmentation.mark_boundaries(gfp_image_i, mask_improved_i_final)
        mask_i_contour = segmentation.mark_boundaries(gfp_image_i, masks_manual_i)

        misc.imsave('./figures/gfp_thresholding/mask_manual_v2_' + str(i) + '.png', mask_improved_i_contour)
        misc.imsave('./figures/gfp_thresholding/mask_manual_init_' + str(i) + '.png', mask_i_contour)

        masks_manual_improved[i] = mask_improved_i_final

    # X_DIM = 1460
    # Y_DIM = 1920
    # x_dim_round = 128 * (X_DIM // 128)
    # y_dim_round = 128 * (Y_DIM // 128)
    # masks_manual_improved_cropped = masks_manual_improved[:, 0:x_dim_round, 0:y_dim_round]
    #
    # PATH_SMP = './samples/'
    # masks_manual_improved = masks_manual_improved.astype(np.uint8)
    # path = os.path.join(PATH_SMP, 'labels_nucleoli_manual_improved_fullsize.npy')
    # np.save(path, masks_manual_improved)
    # masks_manual_improved_cropped = masks_manual_improved_cropped.astype(np.uint8)
    # path = os.path.join(PATH_SMP, 'labels_nucleoli_manual_improved.npy')
    # np.save(path, masks_manual_improved_cropped)
    #
    #
    #     # plt.savefig('./figures/gfp_thresholding/histograms/histogram_all_n.png')
    #     # plt.close()
    #
    # gfp_image = np.load('./samples/images_gfp.npy')
    # # mask_improved_final = np.load('./samples/labels_nucleoli_v2_fullsize.npy')
    # mask_improved_final = np.load('./results/model_unet_2d_n_layers_5_modality_nucleoli_reg_regiongrowing/predictions.npy')
    #
    # for i in range(25):
    #     mask_improved_i_contour = segmentation.mark_boundaries(gfp_image[i], mask_improved_final[i])
    #     misc.imsave('./figures/model_unet_2d_n_layers_5_modality_nucleoli_reg_regiongrowing/labels_nucleoli_' + str(i) + '.png', mask_improved_i_contour)
    #

#####################################
# Mitose removal in DAPI
#####################################
#
# # Load nuclei and seeds segmentations
# labels_nuclei = np.load('./samples/labels_nuclei.npy')
# seeds_nucleoli = np.load('./samples/dilated2_seeds_nucleoli_pred.npy')
#
# # Assign labels to nuclei and corresponding seeds
# instance_labels_nuclei, instance_labels_nucleoli = label_simple(seeds_nucleoli,labels_nuclei)
#
# # Loop on images to remove mitose cells
# labels_nuclei_corrected = np.zeros(labels_nuclei.shape)
# sh = labels_nuclei.shape
# for i in range(sh[0]):
#     nuclei_props = regionprops(instance_labels_nuclei)
#     nucleoli_labels_i = instance_labels_nucleoli[i]
#     nuclei_labels_i = instance_labels_nuclei[i]
#
#     # Loop on cells within one image
#     for region_nuclei_j in nuclei_props:
#         j = region_nuclei_j.label
#         nucleoli_one_cell = np.zeros(nucleoli_labels_i.shape)
#         nucleoli_one_cell[nucleoli_labels_i == j] = 1
#
#         # If no seed in one nucleus, remove the nucleus
#         if np.sum(nucleoli_one_cell) != 0:
#             labels_nuclei_corrected[i, nuclei_labels_i == j] = 1
#
# # Save new nuclei without mitose
# labels_nuclei_corrected = labels_nuclei_corrected.astype(np.uint8)
# np.save('./samples/labels_nuclei_corrected.npy', labels_nuclei_corrected)


def mitose_detection(masks_nuclei):
    #masks_nuclei = np.load('./samples/labels_nuclei_v2.npy')

    images_phase = np.load('./samples/images_phase.npy')
    images_dapi = np.load('./samples/images_dapi.npy')
    images_gfp = np.load('./samples/images_gfp.npy')

    sh = masks_nuclei.shape
    mask_mitose_phase = np.zeros(sh)
    mask_mitose_dapi = np.zeros(sh)
    mask_mitose_gfp = np.zeros(sh)
    mask_mitose = np.zeros(sh)

    metric_phase = []
    metric_gfp = []
    metric_dapi = []
    metric_tot = []
    class_tot = []

    for i in range(75):
        masks_nuclei_i = masks_nuclei[i]
        images_phase_i = images_phase[i]
        images_gfp_i = images_gfp[i]
        images_dapi_i = images_dapi[i]
        mask_mitose_phase_i = np.zeros(images_phase_i.shape)
        mask_mitose_dapi_i = np.zeros(images_phase_i.shape)
        mask_mitose_gfp_i = np.zeros(images_phase_i.shape)
        mask_mitose_i = np.zeros(images_phase_i.shape)

        #cells_labels = label(masks_cells_i, background=0)
        cells_labels = label(masks_nuclei_i, background=0)

        nuclei_labels = np.multiply(cells_labels, masks_nuclei_i)
        regions_phase = regionprops(cells_labels, intensity_image=images_phase_i)
        regions_gfp = regionprops(cells_labels, intensity_image=images_gfp_i)
        regions_dapi = regionprops(nuclei_labels, intensity_image=images_dapi_i)

        for (reg_phase, reg_gfp, reg_dapi) in zip(regions_phase, regions_gfp, regions_dapi):
            lab = reg_phase.label
            reg_phase_metric = reg_phase.mean_intensity
            reg_dapi_metric = reg_dapi.mean_intensity
            reg_gfp_metric = reg_gfp.max_intensity
            score = (reg_phase_metric - 150) + (reg_dapi_metric - 35) - (reg_gfp_metric - 75)

            metric_phase.append(reg_phase_metric)
            metric_dapi.append(reg_dapi_metric)
            metric_gfp.append(reg_gfp_metric)
            metric_tot.append(score)

            if reg_phase_metric > 150:
                mask_mitose_phase_i[cells_labels == lab] = 1
            if reg_dapi_metric > 35:
                mask_mitose_dapi_i[nuclei_labels == lab] = 1
            if reg_gfp_metric < 75:
                mask_mitose_gfp_i[cells_labels == lab] = 1

            if score > 0:
                mask_mitose_i[cells_labels == lab] = 1
                class_tot.append(1)
            else:
                class_tot.append(0)

            mask_mitose_phase[i] = mask_mitose_phase_i
            mask_mitose_dapi[i] = mask_mitose_dapi_i
            mask_mitose_gfp[i] = mask_mitose_gfp_i
            mask_mitose[i] = mask_mitose_i

    metric_phase_array = np.array(metric_phase)
    metric_dapi_array = np.array(metric_dapi)
    metric_gfp_array = np.array(metric_gfp)

    # mask_mitose_phase = mask_mitose_phase.astype(np.uint8)
    # np.save('./samples/labels_mitose_phase.npy', mask_mitose_phase)
    # mask_mitose_dapi = mask_mitose_dapi.astype(np.uint8)
    # np.save('./samples/labels_mitose_dapi.npy', mask_mitose_dapi)
    # mask_mitose_gfp = mask_mitose_gfp.astype(np.uint8)
    # np.save('./samples/labels_mitose_gfp.npy', mask_mitose_gfp)
    # mask_mitose = mask_mitose.astype(np.uint8)
    # np.save('./samples/labels_mitose.npy', mask_mitose)

    # n_phase = np.arange(100, 200, 1)
    # gfp_cst = 75
    # front_dapi = 35 + (gfp_cst - 75) - (n_phase - 150)
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(metric_phase_array, metric_dapi_array,
    #            linewidths=1, alpha=.7,
    #            edgecolor='k',
    #            s = 30,
    #            c=class_tot)
    # plt.xlabel('Mean nuclear intensity on DHM image', fontsize=12)
    # plt.ylabel('Mean nuclear intensity on DAPI image', fontsize=12)
    # plt.plot(n_phase, front_dapi, 'k')
    # plt.title('Mitose detection')
    # plt.savefig('./figures/gfp_thresholding/histograms/scatter1.png')
    #
    # n_dapi = np.arange(10, 80, 1)
    # phase_cst = 150
    # front_gfp = 75 + (n_dapi - 35) + (phase_cst - 150)
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(metric_dapi_array, metric_gfp_array,
    #            linewidths=1, alpha=.7,
    #            edgecolor='k',
    #            s = 50,
    #            c=class_tot)
    # plt.xlabel('Mean nuclear intensity on DAPI image', fontsize=12)
    # plt.ylabel('Maximum nucleolar intensity on GFP image', fontsize=12)
    # plt.plot(n_dapi, front_gfp, 'k')
    # plt.title('Mitose detection')
    # plt.savefig('./figures/gfp_thresholding/histograms/scatter2.png')
    #
    # n_gfp = np.arange(5, 200, 1)
    # dapi_cst = 35
    # front_phase = -(dapi_cst - 35) + (n_gfp - 75) + 150
    # fig = plt.figure(figsize=(6, 6))
    # plt.scatter(metric_gfp_array, metric_phase_array,
    #            linewidths=1, alpha=.7,
    #            edgecolor='k',
    #            s = 30,
    #            c=class_tot)
    # plt.xlabel('Maximum nucleolar intensity on GFP image', fontsize=12)
    # plt.ylabel('Mean nuclear intensity on DHM image', fontsize=12)
    # plt.plot(n_gfp, front_phase, 'k')
    # plt.title('Mitose detection')
    # plt.savefig('./figures/gfp_thresholding/histograms/scatter3.png')
    #
    # plt.close()

    return mask_mitose


def virtual_annotations_generation():
    labels_mitose = np.load('./samples/labels_mitose.npy')
    labels_nuclei = np.load('./samples/labels_nuclei_v2.npy')

    dapi_virtual_annotations = np.multiply(1-labels_mitose, labels_nuclei)
    np.save('./samples/dapi_virtual_annotations.npy', dapi_virtual_annotations)
    print(np.amax(dapi_virtual_annotations))
    # for i in range(75):
    #     misc.imsave('./figures/virtual_annotations/dapi_' + str(i) + '.png', dapi_virtual_annotations[i] * 255)

    labels_nucleoli_init = np.load('./results/model_unet_2d_n_layers_5_modality_nucleoli_reg_unique/predictions.npy')
    sh = labels_nucleoli_init.shape
    labels_nucleoli = np.zeros(sh)
    for i in range(sh[0]):
        se = disk(2)
        labels_nucleoli[i] = ndimage.binary_opening(labels_nucleoli_init[i], structure=se)
    labels_nucleoli = labels_nucleoli.astype(np.uint8)
    #gfp_virtual_annotations = np.multiply(labels_nucleoli, dapi_virtual_annotations)
    gfp_virtual_annotations = np.copy(labels_nucleoli)
    np.save('./samples/gfp_virtual_annotations_thr.npy', gfp_virtual_annotations)
    print(np.amax(gfp_virtual_annotations))
    # for i in range(75):
    #     misc.imsave('./figures/virtual_annotations/gfp_' + str(i) + '.png', gfp_virtual_annotations[i] * 255)


def filter_labels_nuclei():

    #labels_nuclei = np.load('./samples/labels_nuclei.npy')
    #labels_nuclei = np.load('./results/model_unet_2d_n_layers_5_modality_nuclei_phase_labels_nuclei_v2_wo_mitose/predictions.npy')
    labels_nuclei = np.load('./results/model_unet_2d_n_layers_5_modality_labels_nucleoli_RG_t015_v18_wo_holes_dilated2/predictions.npy')

    labels_dapi_filtered = np.zeros(labels_nuclei.shape)
    print(labels_dapi_filtered.shape)
    for i in range(75):
        # Close holes in masks
        labels_dapi_i = labels_nuclei[i, :, :]
        sh = labels_dapi_i.shape
        margin = 5
        labels_dapi_extended_i = np.zeros((sh[0] + margin*2, sh[1] + margin*2))
        labels_dapi_extended_i[margin:margin+sh[0], margin:margin+sh[1]] = labels_dapi_i
        # se = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
        se = disk(2)
        labels_dapi_close_i = ndimage.binary_closing(labels_dapi_extended_i, structure=se)
        labels_dapi_close_i = labels_dapi_close_i[margin:margin+sh[0], margin:margin+sh[1]]
        labels_dapi_close_i = np.copy(labels_dapi_close_i)
        labels_dapi_close_i = np.copy(labels_dapi_i)

        # Remove small regions
        labels = label(labels_dapi_close_i, background=0)
        props = regionprops(labels)
        nuclei_clean = np.copy(labels_dapi_close_i)
        for region in props:
            label_index = region.label
            #if region.area < 2000:
            if region.area < 12:
                nuclei_clean[labels == label_index] = 0

        # Accumulate
        labels_dapi_filtered[i] = nuclei_clean

    labels_dapi_filtered = labels_dapi_filtered.astype(np.uint8)

    X_DIM = 1460
    Y_DIM = 1920
    x_dim_round = 128 * (X_DIM // 128)
    y_dim_round = 128 * (Y_DIM // 128)
    print(x_dim_round)
    print(y_dim_round)
    labels_dapi_filtered_cropped = labels_dapi_filtered[:, 0:x_dim_round, 0:y_dim_round]

    PATH_GT = './samples/'
    # labels_dapi = labels_dapi_filtered.astype(np.uint8)
    # path = os.path.join(PATH_GT, 'labels_nuclei_fullsize_v2.npy')
    # np.save(path, labels_dapi)
    # labels_dapi_cropped = labels_dapi_filtered_cropped.astype(np.uint8)
    # path = os.path.join(PATH_GT, 'labels_nuclei_v2.npy')
    # np.save(path, labels_dapi_cropped)

    labels_dapi_cropped = labels_dapi_filtered_cropped.astype(np.uint8)
    path = os.path.join(PATH_GT, 'model_unet_2d_n_layers_5_modality_labels_nucleoli_RG_t015_v18_wo_holes_dilated2_postprocessed12.npy')
    np.save(path, labels_dapi_cropped)


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
    thr_results = np.load('./samples/labels_nucleoli_RG_t0125_v19_wo_holes.npy')
    sh = thr_results.shape
    dilated_images = np.zeros(sh)
    for i in range(sh[0]):
        thr_results_i = thr_results[i]
        dilated_images[i] = ndimage.binary_dilation(thr_results_i, structure=disk(1))
    dilated_images = dilated_images.astype(np.uint8)
    np.save('./samples/labels_nucleoli_RG_t0125_v19_wo_holes_dilated1.npy', dilated_images)

#improve_manual_masks()
filter_labels_nuclei()
#mitose_detection()
#virtual_annotations_generation()
#fill_holes(np.load('./samples/labels_nucleoli_RG_rt045_v21.npy'), './samples/labels_nucleoli_RG_rt045_v21_wo_holes.npy')
#dilate_thresholding()
