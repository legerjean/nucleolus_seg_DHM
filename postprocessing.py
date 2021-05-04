import numpy as np
import csv
import pickle
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy import ndimage, misc
from skimage.morphology import disk


def generate_border_mask(nuclei_mask):
    sh = nuclei_mask.shape
    border_mask = np.zeros(sh, dtype=np.uint8)

    for i in range(sh[0]):  # loop on images of the database
        nuclei_im = nuclei_mask[i, :, :]
        nuclei_crop = clear_border(nuclei_im)
        border_mask[i] = nuclei_im - nuclei_crop

    return border_mask


def generate_cluster_mask(nuclei_mask):
    sh = nuclei_mask.shape
    cluster_mask = np.zeros(sh, dtype=np.uint8)
    cluster_list = []

    for i in range(sh[0]):  # loop on images of the database
        nuclei_im = nuclei_mask[i, :, :]

        # Set labels
        nuclei_label = label(nuclei_im, background=0)

        # Clean regions
        props = regionprops(nuclei_label)
        nuclei_clean = np.copy(nuclei_im)
        for region in props:
            label_index = region.label
            # Discard non convex regions
            if region.solidity < 0.95:
                nuclei_clean[nuclei_label == label_index] = 0
            else:
                cluster_list.append(region.solidity)

        # Accumulate
        cluster_mask[i] = nuclei_im - nuclei_clean

    return cluster_mask


def generate_manual_mask(nuclei_mask, manual_annotations):
    dapi_virtual_annotations = nuclei_mask[0:25]
    dapi_virtual_annotations_filtered_mask = np.zeros(manual_annotations.shape)

    for i in range(25):
        manual_annotations_i = manual_annotations[i]
        dapi_virtual_annotations_i = dapi_virtual_annotations[i]

        # Remove unannotated regions
        labels = label(dapi_virtual_annotations_i, background=0)
        labelled_manual_annotations_i = np.multiply(labels, manual_annotations_i)
        dapi_virtual_annotations_filtered_i = np.copy(dapi_virtual_annotations_i)
        for label_index in range(1, np.amax(labels) + 1):
            if len(labelled_manual_annotations_i[labelled_manual_annotations_i == label_index]) == 0:
                dapi_virtual_annotations_filtered_i[labels == label_index] = 0

        dapi_virtual_annotations_filtered_mask[i] = dapi_virtual_annotations_i - dapi_virtual_annotations_filtered_i

    return dapi_virtual_annotations_filtered_mask
