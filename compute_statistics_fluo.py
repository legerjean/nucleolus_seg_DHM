from postprocessing import *
import csv
from itertools import zip_longest
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from math import log10, floor, pi, sqrt
import math
import os
import matplotlib.pyplot as plt


def compute_features(directory, gfp_virtual_annotations_filename, dapi_virtual_annotations_filename, manual_mode):

    # Data cleaning
    dapi_virtual_annotations = np.load(dapi_virtual_annotations_filename)
    gfp_virtual_annotations = np.load(gfp_virtual_annotations_filename)
    labels_nucleoli_manual_improved = np.load(dapi_virtual_annotations_filename)  # ignored input
    labels_nucleoli_manual_improved = labels_nucleoli_manual_improved[0:25]

    border_mask = generate_border_mask(dapi_virtual_annotations)
    cluster_mask = generate_cluster_mask(dapi_virtual_annotations)
    manual_mask = generate_manual_mask(dapi_virtual_annotations, labels_nucleoli_manual_improved)

    if manual_mode:
        dapi_virtual_annotations_clean = np.multiply(np.multiply(dapi_virtual_annotations[0:25], 1 - border_mask[0:25]),
                                                     1 - cluster_mask[0:25])
        gfp_virtual_annotations_clean = np.multiply(np.multiply(gfp_virtual_annotations[0:25], 1 - border_mask[0:25]),
                                                    1 - cluster_mask[0:25])
        labels_nucleoli_manual_improved_wo_border = np.multiply(labels_nucleoli_manual_improved[0:25],
                                                                1 - border_mask[0:25])
        gfp_virtual_annotations_manual = np.multiply(np.multiply(gfp_virtual_annotations[0:25], 1 - manual_mask[0:25]),
                                                     1 - border_mask[0:25])
    else:
        dapi_virtual_annotations_clean = np.multiply(np.multiply(dapi_virtual_annotations, 1 - border_mask),
                                                     1 - cluster_mask)
        gfp_virtual_annotations_clean = np.multiply(np.multiply(gfp_virtual_annotations, 1 - border_mask),
                                                    1 - cluster_mask)
        labels_nucleoli_manual_improved_wo_border = np.multiply(labels_nucleoli_manual_improved, 1 - border_mask[0:25])
        gfp_virtual_annotations_manual = np.multiply(np.multiply(gfp_virtual_annotations[0:25], 1 - manual_mask),
                                                     1 - border_mask[0:25])

    # Define statistics inputs
    if manual_mode:
        nb_im = 25
    else:
        nb_im = 75
    nb_nucleoli_inputs = 1
    nuclei_input = dapi_virtual_annotations_clean[0:nb_im]
    sh = nuclei_input.shape
    nucleoli_inputs = np.zeros((nb_nucleoli_inputs, sh[0], sh[1], sh[2]))
    nucleoli_inputs[0] = gfp_virtual_annotations_clean[0:nb_im]
    if nb_nucleoli_inputs > 1:
        nucleoli_inputs[1] = labels_nucleoli_manual_improved_wo_border[0:nb_im]
        nucleoli_inputs[2] = gfp_virtual_annotations_manual[0:nb_im]

    phase_images = np.load('./samples/images_phase_shifted.npy')
    gfp_images = np.load('./samples/images_gfp_shifted_0.npy')

    parent_dir = "./statistics"

    # Initialize accumulators
    nb_nucleoli = []
    volume_nucleoli = []
    area_nuclei = []
    area_nucleoli = []

    for i in range(nb_nucleoli_inputs):
        nb_nucleoli.append([])
        volume_nucleoli.append([])
        area_nuclei.append([])
        area_nucleoli.append([])

    for inputs_i in range(nb_nucleoli_inputs):

        nucleoli_input = nucleoli_inputs[inputs_i]
        counter = 0
        for i in range(nb_im):
            nuclei_labels_i = label(nuclei_input[i], background=0)
            nucleoli_labels_i = np.multiply(nuclei_labels_i, nucleoli_input[i])

            # Get labels
            gfp_image_i = gfp_images[i]
            phase_image_i = phase_images[i]

            # Compute the number of cells in one image
            nuclei_props = regionprops(nuclei_labels_i, phase_image_i)
            nb_cells_i = len(nuclei_props) - 1  # background is not a cell

            # Loop on the cells in one image
            for region_nuclei_j in nuclei_props:
                j = region_nuclei_j.label

                # Area of the nucleus
                area_nucleus_j = region_nuclei_j.area
                area_nuclei[inputs_i].append(area_nucleus_j)

                # Get only the nucleoli with label j
                nucleoli_one_cell = np.zeros(nucleoli_labels_i.shape)
                nucleoli_one_cell[nucleoli_labels_i == j] = 1
                nucleoli_one_cell_labels = label(nucleoli_one_cell, background=0)
                nucleoli_one_cell_props = regionprops(nucleoli_one_cell_labels, phase_image_i)

                # Count number of nucleoli per cell
                nb_nucleoli[inputs_i].append(len(nucleoli_one_cell_props))  # list of arrays

                #if (len(nucleoli_one_cell_props) == 1) or (len(nucleoli_one_cell_props) == 2):
                #    nb_manual = manual_nb_nucleoli[0][counter]
                #    if nb_manual != len(nucleoli_one_cell_props):
                #        misc.imsave('./figures/update_14May/image_' + str(i) + '_error' + str(j) + '.png', nucleoli_one_cell * 255)
                #        misc.imsave('./figures/update_14May/image_' + str(i) + '_gfp.png', gfp_image_i)
                #        misc.imsave('./figures/update_14May/image_' + str(i) + '_phase.png', phase_image_i)

                # Volume and area of nucleoli per cell
                area_nucleoli_j = 0
                pseudo_volume_nucleoli_j = 0
                for region_nucleoli in nucleoli_one_cell_props:  # loop on the nucleoli in a cell
                    mean_intensity = region_nucleoli.mean_intensity
                    area = region_nucleoli.area
                    area_increment = area
                    pseudo_volume_increment = mean_intensity * area
                    area_nucleoli_j = area_nucleoli_j + area_increment
                    pseudo_volume_nucleoli_j = pseudo_volume_nucleoli_j + pseudo_volume_increment
                area_nucleoli[inputs_i].append(area_nucleoli_j)
                volume_nucleoli[inputs_i].append(pseudo_volume_nucleoli_j)  # list of arrays

                counter = counter + 1

    nb_nucleoli_array = np.array(nb_nucleoli)
    area_nucleoli_array = np.array(area_nucleoli)
    area_nuclei_array = np.array(area_nuclei)
    volume_nucleoli_array = np.array(volume_nucleoli)

    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'nb_nucleoli.npy'), nb_nucleoli_array)
    np.save(os.path.join(path, 'area_nucleoli.npy'), area_nucleoli_array)
    np.save(os.path.join(path, 'area_nuclei.npy'), area_nuclei_array)
    np.save(os.path.join(path, 'volume_nucleoli.npy'), volume_nucleoli_array)


def remove_zeros(directory, folder):
    # Load features
    nb_nucleoli_array = np.load(folder + 'nb_nucleoli.npy')
    area_nucleoli_array = np.load(folder + 'area_nucleoli.npy')
    area_nuclei_array = np.load(folder + 'area_nuclei.npy')
    volume_nucleoli_array = np.load(folder + 'volume_nucleoli.npy')

    nb_nucleoli_new = []
    area_nucleoli_new = []
    area_nuclei_new = []
    volume_nucleoli_new = []
    nb_nucleoli_new.append([])
    area_nucleoli_new.append([])
    area_nuclei_new.append([])
    volume_nucleoli_new.append([])

    pop_index = 0
    nb_cells_tot = len(nb_nucleoli_array[pop_index])
    nb_nucleoli_array = nb_nucleoli_array[pop_index]
    area_nucleoli_array = area_nucleoli_array[pop_index]
    area_nuclei_array = area_nuclei_array[pop_index]
    volume_nucleoli_array = volume_nucleoli_array[pop_index]
    for i in range(nb_cells_tot):
        if nb_nucleoli_array[i] != 0:
            nb_nucleoli_new[pop_index].append(nb_nucleoli_array[i])
            area_nucleoli_new[pop_index].append(area_nucleoli_array[i])
            area_nuclei_new[pop_index].append(area_nuclei_array[i])
            volume_nucleoli_new[pop_index].append(volume_nucleoli_array[i])

    nb_nucleoli_array_new = np.array(nb_nucleoli_new)
    area_nucleoli_array_new = np.array(area_nucleoli_new)
    area_nuclei_array_new = np.array(area_nuclei_new)
    volume_nucleoli_array_new = np.array(volume_nucleoli_new)

    print(len(nb_nucleoli_array_new[0]))
    print(len(nb_nucleoli_array))

    parent_dir = "./statistics"
    path = os.path.join(parent_dir, directory)

    np.save(os.path.join(path, 'nb_nucleoli_wo_0.npy'), nb_nucleoli_array_new)
    np.save(os.path.join(path, 'area_nucleoli_wo_0.npy'), area_nucleoli_array_new)
    np.save(os.path.join(path, 'area_nuclei_wo_0.npy'), area_nuclei_array_new)
    np.save(os.path.join(path, 'volume_nucleoli_wo_0.npy'), volume_nucleoli_array_new)


def remove_zeros_2_inputs(directory1, folder1, directory2, folder2):
    # Load features
    nb_nucleoli_array1 = np.load(folder1 + 'nb_nucleoli.npy')
    area_nucleoli_array1 = np.load(folder1 + 'area_nucleoli.npy')
    area_nuclei_array1 = np.load(folder1 + 'area_nuclei.npy')
    volume_nucleoli_array1 = np.load(folder1 + 'volume_nucleoli.npy')
    nb_nucleoli_array2 = np.load(folder2 + 'nb_nucleoli.npy')
    area_nucleoli_array2 = np.load(folder2 + 'area_nucleoli.npy')
    area_nuclei_array2 = np.load(folder2 + 'area_nuclei.npy')
    volume_nucleoli_array2 = np.load(folder2 + 'volume_nucleoli.npy')

    nb_nucleoli_new_1 = []
    area_nucleoli_new_1 = []
    area_nuclei_new_1 = []
    volume_nucleoli_new_1 = []
    nb_nucleoli_new_1.append([])
    area_nucleoli_new_1.append([])
    area_nuclei_new_1.append([])
    volume_nucleoli_new_1.append([])
    nb_nucleoli_new_2 = []
    area_nucleoli_new_2 = []
    area_nuclei_new_2 = []
    volume_nucleoli_new_2 = []
    nb_nucleoli_new_2.append([])
    area_nucleoli_new_2.append([])
    area_nuclei_new_2.append([])
    volume_nucleoli_new_2.append([])

    pop_index = 0
    nb_cells_tot = len(nb_nucleoli_array1[pop_index])
    nb_nucleoli_array1 = nb_nucleoli_array1[pop_index]
    area_nucleoli_array1 = area_nucleoli_array1[pop_index]
    area_nuclei_array1 = area_nuclei_array1[pop_index]
    volume_nucleoli_array1 = volume_nucleoli_array1[pop_index]
    nb_nucleoli_array2 = nb_nucleoli_array2[pop_index]
    area_nucleoli_array2 = area_nucleoli_array2[pop_index]
    area_nuclei_array2 = area_nuclei_array2[pop_index]
    volume_nucleoli_array2 = volume_nucleoli_array2[pop_index]

    for i in range(nb_cells_tot):
        if nb_nucleoli_array1[i] != 0 and nb_nucleoli_array2[i] != 0:
            nb_nucleoli_new_1[pop_index].append(nb_nucleoli_array1[i])
            area_nucleoli_new_1[pop_index].append(area_nucleoli_array1[i])
            area_nuclei_new_1[pop_index].append(area_nuclei_array1[i])
            volume_nucleoli_new_1[pop_index].append(volume_nucleoli_array1[i])
            nb_nucleoli_new_2[pop_index].append(nb_nucleoli_array2[i])
            area_nucleoli_new_2[pop_index].append(area_nucleoli_array2[i])
            area_nuclei_new_2[pop_index].append(area_nuclei_array2[i])
            volume_nucleoli_new_2[pop_index].append(volume_nucleoli_array2[i])

    nb_nucleoli_array_new_1 = np.array(nb_nucleoli_new_1)
    area_nucleoli_array_new_1 = np.array(area_nucleoli_new_1)
    area_nuclei_array_new_1 = np.array(area_nuclei_new_1)
    volume_nucleoli_array_new_1 = np.array(volume_nucleoli_new_1)
    nb_nucleoli_array_new_2 = np.array(nb_nucleoli_new_2)
    area_nucleoli_array_new_2 = np.array(area_nucleoli_new_2)
    area_nuclei_array_new_2 = np.array(area_nuclei_new_2)
    volume_nucleoli_array_new_2 = np.array(volume_nucleoli_new_2)

    print(len(nb_nucleoli_array_new_1[0]))
    print(len(nb_nucleoli_array1))
    print(len(nb_nucleoli_array_new_2[0]))
    print(len(nb_nucleoli_array2))

    parent_dir = "./statistics"
    path1 = os.path.join(parent_dir, directory1)
    np.save(os.path.join(path1, 'nb_nucleoli_wo_0.npy'), nb_nucleoli_array_new_1)
    np.save(os.path.join(path1, 'area_nucleoli_wo_0.npy'), area_nucleoli_array_new_1)
    np.save(os.path.join(path1, 'area_nuclei_wo_0.npy'), area_nuclei_array_new_1)
    np.save(os.path.join(path1, 'volume_nucleoli_wo_0.npy'), volume_nucleoli_array_new_1)
    path2 = os.path.join(parent_dir, directory2)
    np.save(os.path.join(path2, 'nb_nucleoli_wo_0.npy'), nb_nucleoli_array_new_2)
    np.save(os.path.join(path2, 'area_nucleoli_wo_0.npy'), area_nucleoli_array_new_2)
    np.save(os.path.join(path2, 'area_nuclei_wo_0.npy'), area_nuclei_array_new_2)
    np.save(os.path.join(path2, 'volume_nucleoli_wo_0.npy'), volume_nucleoli_array_new_2)


def intra_population_statistics(folder):

    # Load features
    nb_nucleoli_array = np.load(folder + 'nb_nucleoli_wo_0.npy')
    area_nucleoli_array = np.load(folder + 'area_nucleoli_wo_0.npy')
    area_nuclei_array = np.load(folder + 'area_nuclei_wo_0.npy')
    volume_nucleoli_array = np.load(folder + 'volume_nucleoli_wo_0.npy')
    pop_index = 0
    extension_name = 'std'

    # General
    nb_cells_tot = len(nb_nucleoli_array[pop_index])
    nb_nucleoli_total = np.sum(nb_nucleoli_array[pop_index])

    # Nucleolus
    mean_area_nucleoli = np.mean(area_nucleoli_array[pop_index])
    std_area_nucleoli = np.std(area_nucleoli_array[pop_index], ddof=1)
    area_nucleoli_ac = np.multiply(area_nucleoli_array[pop_index], 0.022801)
    area_nucleoli_am = np.multiply(area_nucleoli_array[pop_index], 0.022201)
    mean_area_nucleoli_ac = np.mean(area_nucleoli_ac)
    std_area_nucleoli_ac = np.std(area_nucleoli_ac, ddof=1)
    mean_area_nucleoli_am = np.mean(area_nucleoli_am)
    std_area_nucleoli_am = np.std(area_nucleoli_am, ddof=1)
    mean_volume_nucleoli = np.mean(volume_nucleoli_array[pop_index])
    std_volume_nucleoli = np.std(volume_nucleoli_array[pop_index], ddof=1)
    mean_nb_nucleoli = np.mean(nb_nucleoli_array[pop_index])
    std_nb_nucleoli = np.std(nb_nucleoli_array[pop_index], ddof=1)
    ratio = np.divide(area_nucleoli_array[pop_index], area_nuclei_array[pop_index])
    mean_ratio_nucleoli = np.mean(ratio)
    std_ratio_nucleoli = np.std(ratio, ddof=1)
    diameter_nucleoli = 2*np.sqrt(np.multiply(area_nucleoli_array[pop_index], 0.022201/math.pi))
    mean_diameter_nucleoli = np.mean(diameter_nucleoli)
    std_diameter_nucleoli = np.std(diameter_nucleoli, ddof=1)

    # Nucleus
    mean_area_nuclei = np.mean(area_nuclei_array[pop_index])
    std_area_nuclei = np.std(area_nuclei_array[pop_index], ddof=1)
    area_nuclei_ac = np.multiply(area_nuclei_array[pop_index], 0.022801)
    area_nuclei_am = np.multiply(area_nuclei_array[pop_index], 0.022201)
    mean_area_nuclei_ac = np.mean(area_nuclei_ac)
    std_area_nuclei_ac = np.std(area_nuclei_ac, ddof=1)
    mean_area_nuclei_am = np.mean(area_nuclei_am)
    std_area_nuclei_am = np.std(area_nuclei_am, ddof=1)

    # Analysis per number of nucleoli
    nb_nucleoli_range = np.unique(nb_nucleoli_array[pop_index])
    mean_area_nucleoli_separated = np.zeros(len(nb_nucleoli_range))
    std_area_nuclei_separated = np.zeros(len(nb_nucleoli_range))
    nb_cells_separated = np.zeros(len(nb_nucleoli_range))

    mean_volume_nucleoli_separated = np.zeros(len(nb_nucleoli_range))
    std_volume_nuclei_separated = np.zeros(len(nb_nucleoli_range))

    # Area nucleoli non-normalized
    for counter, value in enumerate(nb_nucleoli_range):
        indices = np.where(nb_nucleoli_array[pop_index] == value)
        nb_cells_separated[counter] = len(indices[0])
        mean_area_nucleoli_separated[counter] = np.mean(area_nucleoli_array[pop_index][indices[0]])
        std_area_nuclei_separated[counter] = np.std(area_nucleoli_array[pop_index][indices[0]], ddof=1)

        mean_volume_nucleoli_separated[counter] = np.mean(volume_nucleoli_array[pop_index][indices[0]])
        std_volume_nuclei_separated[counter] = np.std(volume_nucleoli_array[pop_index][indices[0]], ddof=1)

    # Print statistics
    print('----------- Statistics on full dataset -------------')
    print('nb_cells             = ' + str("%.2f" % nb_cells_tot))
    print('nb_nucleoli          = ' + str("%.2f" % nb_nucleoli_total))
    print('')
    print('mean_area_nucleoli_ac   = ' + str("%.2f" % mean_area_nucleoli_ac))
    print('std_area_nucleoli_ac    = ' + str("%.2f" % std_area_nucleoli_ac))
    print('mean_area_nucleoli_am   = ' + str("%.2f" % mean_area_nucleoli_am))
    print('std_area_nucleoli_am    = ' + str("%.2f" % std_area_nucleoli_am))
    print('mean_volume_nucleoli = ' + str("%.2f" % mean_volume_nucleoli))
    print('std_volume_nucleoli  = ' + str("%.2f" % std_volume_nucleoli))
    print('mean_nb_nucleoli     = ' + str("%.2f" % mean_nb_nucleoli))
    print('std_nb_nucleoli      = ' + str("%.2f" % std_nb_nucleoli))
    print('mean_area_ratio      = ' + str("%.2f" % mean_ratio_nucleoli))
    print('std_area_ratio       = ' + str("%.2f" % std_ratio_nucleoli))
    print('mean_area_nuclei_ac   = ' + str("%.2f" % mean_area_nuclei_ac))
    print('std_area_nuclei_ac    = ' + str("%.2f" % std_area_nuclei_ac))
    print('mean_area_nuclei_am   = ' + str("%.2f" % mean_area_nuclei_am))
    print('std_area_nuclei_am    = ' + str("%.2f" % std_area_nuclei_am))
    print('\n')

    for counter, value in enumerate(nb_nucleoli_range):
        print('----------- Statistics on subset with ' + str(value) + ' nucleoli -------------')
        print('nb_cells             = ' + str("%.2f" % nb_cells_separated[counter]))
        print('mean_area_nucleoli   = ' + str("%.2f" % mean_area_nucleoli_separated[counter]))
        print('std_area_nucleoli    = ' + str("%.2f" % std_area_nuclei_separated[counter]))
        print('mean_area_nucleoli   = ' + str("%.2f" % np.multiply(mean_area_nucleoli_separated[counter], 0.022201)))
        print('std_area_nucleoli    = ' + str("%.2f" % np.multiply(std_area_nuclei_separated[counter], 0.022201)))

        print('mean_volume_nucleoli   = ' + str("%.2f" % mean_volume_nucleoli_separated[counter]))
        print('std_volume_nucleoli    = ' + str("%.2f" % std_volume_nuclei_separated[counter]))
        print('\n')
        #print('')

    my_dict = {'nb_cells': nb_cells_tot,
               'nb_nucleoli': nb_nucleoli_total,
               'mean_nb_nucleoli': mean_nb_nucleoli,
               'std_nb_nucleoli': std_nb_nucleoli,
               'mean_area_nucleoli_am': mean_area_nucleoli_am,
               'std_area_nucleoli_am': std_area_nucleoli_am,
               'mean_diameter_nucleoli': mean_diameter_nucleoli,
               'std_diameter_nucleoli': std_diameter_nucleoli,
               #'mean_area_nucleoli_ac': mean_area_nucleoli_ac,
               #'std_area_nucleoli_ac': std_area_nucleoli_ac,
               'mean_volume_nucleoli': mean_volume_nucleoli,
               'std_volume_nucleoli': std_volume_nucleoli,
               'mean_area_ratio': mean_ratio_nucleoli,
               'std_area_ratio': std_ratio_nucleoli,
               'mean_area_nuclei_am': mean_area_nuclei_am,
               'std_area_nuclei_am': std_area_nuclei_am
               #'mean_area_nuclei_ac': mean_area_nuclei_ac,
               #'std_area_nuclei_ac': std_area_nuclei_ac
               }

    pickle_out = open(os.path.join(folder, 'statistics.pkl'), "wb")
    pickle.dump(my_dict, pickle_out)
    pickle_out.close()

    my_file = open(os.path.join(folder, 'statistics.txt'), "w")
    for k, v in my_dict.items():
        my_file.write(k + ' , ' + str(v) + '\n')
    my_file.close()

    # Number of nucleoli per cell
    nbins = 9
    plt.figure(figsize=(12,12))
    plt.hist([nb_nucleoli_array[pop_index]], bins = nbins, range = (0,nbins), label = ['Prediction'], align = 'left', alpha = 0.7)
    plt.ylim(0, 180*3)
    plt.ylabel('Number of occurences', fontsize = 18)
    plt.xlabel('Number nucleoli per nucleus', fontsize = 18)
    plt.title('Number of nucleoli per nucleus histogram', fontsize = 24)
    # plt.legend(fontsize = 14)
    # plt.annotate('Number of nucleoli = %i' % nb_total_nucleoli, (0.55,0.75), fontsize=14,xycoords='figure fraction')
    # plt.annotate('Number of nuclei = %i' % nb_total_nuclei, (0.55,0.7), fontsize=14,xycoords='figure fraction')
    plt.savefig(folder + 'nb_nucleoli_hist_' + extension_name + '.png')
    plt.show()
    plt.close()

    # Number of nuclei and nucleoli
    nb_total_nucleoli = np.sum(nb_nucleoli_array[pop_index])
    nb_total_nuclei = len(nb_nucleoli_array[pop_index])

    # Area nucleoli non-normalized
    nbins = 75
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    area_nucleoli_array_div = area_nucleoli_am
    plt.hist(area_nucleoli_array_div, bins=nbins, range = (0,50), histtype = 'bar', alpha = 0.7, density=False)
    # plt.hist(volume_nucleoli_array_n_GT, bins=nbins, range = (0,3), histtype = 'bar', alpha = 0.7, label = 'Estimated GT')
    plt.ylim(0, 35*3)
    plt.ylabel(r'Frequency', fontsize = 14)
    plt.xlabel(r'Nucleolar area per cell [$\mu m^2$]', fontsize = 14)
    plt.title(r'Nucleolar area distribution', fontsize = 18)
    #plt.legend(fontsize = 12)
    #plt.annotate(r'Number of cells = %i' % nb_cells_tot, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nuclei = %i' % nb_total_nuclei, (0.55,0.7), fontsize=15,xycoords='figure fraction')

    # Volume nucleoli non-normalized
    plt.subplot(2,2,2)
    nbins = 75
    # plt.figure(figsize=(10,10))
    volume_nucleoli_array_div = np.multiply(volume_nucleoli_array[pop_index], 0.00001)
    plt.hist(volume_nucleoli_array_div, bins=nbins, range = (0,4), histtype = 'bar', alpha = 0.7, density=False)
    # plt.hist(volume_nucleoli_array_n_GT, bins=nbins, range = (0,3), histtype = 'bar', alpha = 0.7, label = 'Estimated GT')
    plt.ylim(0, 35*3)
    plt.ylabel(r'Frequency', fontsize = 14)
    plt.xlabel(r'Nucleolar pseudo-volume per cell [$10^{-5}$]', fontsize = 14)
    plt.title(r'Nucleolar pseudo-volume distribution', fontsize = 18)
    #plt.legend(fontsize = 12)
    #plt.annotate(r'Number of cells = %i' % nb_cells_tot, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nucleoli = %i' % nb_total_nucleoli, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nuclei = %i' % nb_total_nuclei, (0.55,0.7), fontsize=15,xycoords='figure fraction')

    # Volume ratio
    plt.subplot(2,2,3)
    nbins = 75
    # plt.figure(figsize=(10,10))
    plt.hist(np.divide(area_nucleoli_array[pop_index],area_nuclei_array[pop_index]), bins=nbins, range = (0,0.4), histtype = 'bar', alpha = 0.7, density=False)
    # plt.hist(volume_nucleoli_array_n_GT, bins=nbins, range = (0,3), histtype = 'bar', alpha = 0.7, label = 'Estimated GT')
    plt.ylim(0, 50*3)
    plt.ylabel(r'Frequency', fontsize = 14)
    plt.xlabel(r'Nucleolar/nuclear area ratio', fontsize = 14)
    plt.title(r'Nucleolar/nuclear area ratio distribution', fontsize = 18)
    #plt.legend(fontsize = 12)
    #plt.annotate(r'Number of cells = %i' % nb_cells_tot, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nucleoli = %i' % nb_total_nucleoli, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nuclei = %i' % nb_total_nuclei, (0.55,0.7), fontsize=15,xycoords='figure fraction')

    # Area nuclei
    plt.subplot(2,2,4)
    nbins = 75
    # plt.figure(figsize=(10,10))
    area_nuclei_array_div = np.multiply(area_nuclei_array[pop_index], 0.0225)
    plt.hist(area_nuclei_array_div, bins=nbins, range = (50, 300), histtype = 'bar', alpha = 0.7, density=False)
    # plt.hist(volume_nucleoli_array_n_GT, bins=nbins, range = (0,3), histtype = 'bar', alpha = 0.7, label = 'Estimated GT')
    plt.ylim(0, 25*3)
    plt.ylabel(r'Frequency', fontsize = 14)
    plt.xlabel(r'Nuclear area per cell [$\mu m^2$]', fontsize = 14)
    plt.title(r'Nuclear area distribution', fontsize = 18)
    #plt.legend(fontsize = 12)
    #plt.annotate(r'Number of cells = %i' % nb_cells_tot, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nucleoli = %i' % nb_total_nucleoli, (0.55,0.75), fontsize=15,xycoords='figure fraction')
    # plt.annotate('Number of nuclei = %i' % nb_total_nuclei, (0.55,0.7), fontsize=15,xycoords='figure fraction')

    plt.suptitle('For %i cells' % nb_cells_tot, fontsize = 18)

    plt.savefig(folder + 'features_hist_' + extension_name + '.png')


def plot_details_histograms(folder_area, folder_counting):

    extension_name = 'std'

    nb_nucleoli_array = np.load(folder_counting + '/nb_nucleoli_wo_0.npy')
    area_nucleoli_array = np.load(folder_area + '/area_nucleoli_wo_0.npy')
    volume_nucleoli_array = np.load(folder_area + '/volume_nucleoli_wo_0.npy')

    nb_nucleoli_range = np.unique(nb_nucleoli_array[0])
    mean_area_nucleoli_separated = np.zeros(len(nb_nucleoli_range))
    std_area_nuclei_separated = np.zeros(len(nb_nucleoli_range))
    nb_cells_separated = np.zeros(len(nb_nucleoli_range))

    mean_volume_nucleoli_separated = np.zeros(len(nb_nucleoli_range))
    std_volume_nuclei_separated = np.zeros(len(nb_nucleoli_range))

    nbins = 30
    plt.figure(figsize=(12, 12))
    for counter, value in enumerate(nb_nucleoli_range):
        indices = np.where(nb_nucleoli_array[0] == value)
        nb_cells_separated[counter] = len(indices[0])
        mean_area_nucleoli_separated[counter] = np.mean(np.multiply(area_nucleoli_array[0][indices[0]], 0.022201))
        std_area_nuclei_separated[counter] = np.std(np.multiply(area_nucleoli_array[0][indices[0]], 0.022201), ddof=1)
        mean_volume_nucleoli_separated[counter] = np.mean(volume_nucleoli_array[0][indices[0]])
        std_volume_nuclei_separated[counter] = np.std(volume_nucleoli_array[0][indices[0]], ddof=1)
        print('----------- Statistics on subset with ' + str(value) + ' nucleoli -------------')
        print('nb_cells             = ' + str("%.2f" % nb_cells_separated[counter]))
        print('mean_area_nucleoli   = ' + str("%.2f" % mean_area_nucleoli_separated[counter]))
        print('std_area_nucleoli    = ' + str("%.2f" % std_area_nuclei_separated[counter]))
        print('mean_volume_nucleoli   = ' + str("%.2f" % mean_volume_nucleoli_separated[counter]))
        print('std_volume_nucleoli    = ' + str("%.2f" % std_volume_nuclei_separated[counter]))
        print('\n')

        # if counter < 4:
        #     plt.subplot(2, 2, counter + 1)
        #     plt.hist(np.multiply(area_nucleoli_array[0][indices[0]], 0.022201), bins=nbins, range=(0, 50),
        #              histtype='bar', alpha=0.7, density=False)
        #     plt.ylabel(r'Frequency', fontsize=14)
        #     plt.xlabel(r'Nucleolar area per cell [$\mu m^2$]', fontsize=14)
        #     plt.title(str(value) + ' nucleoli', fontsize=18)
    #plt.savefig(folder_area + '/area_hist_details_' + extension_name + '.png')


def plot_confusion_matrices(folder):

    entry_true = np.load('./statistics/labels_nucleoli_manual_shifted_0_new/nb_nucleoli.npy')
    entry_true = entry_true[0]
    entry_pred = np.load(folder + '/nb_nucleoli.npy')
    entry_pred = entry_pred[0]
    entry_pred = entry_pred[0:len(entry_true)]

    # Remove zeros in true and pred
    non_zeros_true = np.nonzero(entry_true)
    entry_true = entry_true[non_zeros_true]
    entry_pred = entry_pred[non_zeros_true]
    non_zeros_pred = np.nonzero(entry_pred)
    entry_true = entry_true[non_zeros_pred]
    entry_pred = entry_pred[non_zeros_pred]

    # Concatenate to generate the labels
    entry_concat = np.array([entry_true, entry_pred])

    cm_virtual_manual = confusion_matrix(entry_true, entry_pred)
    sh = cm_virtual_manual.shape
    cm_virtual_manual_n = cm_virtual_manual.astype('float') / cm_virtual_manual.sum(axis=0)[np.newaxis, :]
    df_cm = pd.DataFrame(cm_virtual_manual, range(sh[0]), range(sh[0]))
    df_cm_n = pd.DataFrame(cm_virtual_manual_n, range(sh[0]), range(sh[0]))

    plt.figure(figsize=(20,7))
    sn.set(font_scale=1.4) # for label size
    plt.subplot(1,2,1)
    x_axis_labels = np.unique(entry_concat)
    y_axis_labels = np.unique(entry_concat)
    sn.heatmap(df_cm_n, annot=True, annot_kws={"size": 14}, cmap='Purples', xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size
    plt.ylabel(r'Manual annotations', fontsize = 14)
    plt.xlabel(r'Automatic predictions', fontsize = 14)
    plt.title(r'Normalized confusion matrix', fontsize = 18)
    plt.subplot(1,2,2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, cmap='Purples', fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels) # font size
    plt.ylabel(r'Manual annotations', fontsize = 14)
    plt.xlabel(r'Automatic predictions', fontsize = 14)
    plt.title(r'Confusion matrix', fontsize = 18)
    plt.savefig(folder + '/confusion_matrix.png')
    plt.show()


def round_to_n(x, n_digits):
    return round(x, n_digits-(1+int(floor(log10(abs(x))))))


def round_metrics():
    parent_dir = './statistics'

    folder_names = [#'labels_nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted_0',
                    #'labels_nucleoli_RG_rt045_v17_wo_holes_shifted_0',
                    'labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0_new',
                    'labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0_new'
                    ]

    for i in range(len(folder_names)):
        pickle_in = open(parent_dir + '/' + folder_names[i] + '/statistics.pkl', "rb")
        my_dict = pickle.load(pickle_in)
        my_dict['mean_area_nucleoli_am'] = round_to_n(my_dict['mean_area_nucleoli_am'], 4)
        my_dict['std_area_nucleoli_am'] = round_to_n(my_dict['std_area_nucleoli_am'], 3)
        #my_dict['mean_area_nucleoli_ac'] = round_to_n(my_dict['mean_area_nucleoli_ac'], 4)
        #my_dict['std_area_nucleoli_ac'] = round_to_n(my_dict['std_area_nucleoli_ac'], 3)
        my_dict['mean_volume_nucleoli'] = round_to_n(my_dict['mean_volume_nucleoli'], 6)
        my_dict['std_volume_nucleoli'] = round_to_n(my_dict['std_volume_nucleoli'], 5)
        my_dict['mean_diameter_nucleoli'] = round_to_n(my_dict['mean_diameter_nucleoli'], 3),
        my_dict['std_diameter_nucleoli'] = round_to_n(my_dict['std_diameter_nucleoli'], 3),
        my_dict['mean_nb_nucleoli'] = round_to_n(my_dict['mean_nb_nucleoli'], 3)
        my_dict['std_nb_nucleoli'] = round_to_n(my_dict['std_nb_nucleoli'], 3)
        my_dict['mean_area_ratio'] = round_to_n(my_dict['mean_area_ratio'], 2)
        my_dict['std_area_ratio'] = round_to_n(my_dict['std_area_ratio'], 2)
        my_dict['mean_area_nuclei_am'] = round_to_n(my_dict['mean_area_nuclei_am'], 4)
        my_dict['std_area_nuclei_am'] = round_to_n(my_dict['std_area_nuclei_am'], 3)
        #my_dict['mean_area_nuclei_ac'] = round_to_n(my_dict['mean_area_nuclei_ac'], 4)
        #my_dict['std_area_nuclei_ac'] = round_to_n(my_dict['std_area_nuclei_ac'], 3)

        pickle_out = open(parent_dir + '/' + folder_names[i] + '/statistics_rounded.pkl', "wb")
        pickle.dump(my_dict, pickle_out)
        pickle_out.close()


def print_table_all():

    parent_dir = './statistics'
    #pickle_in = open(parent_dir + '/labels_nucleoli_RG_t0125_v17_wo_holes_dilated1_shifted_0/statistics_rounded.pkl', "rb")
    #my_dict_1 = pickle.load(pickle_in)
    #pickle_in = open(parent_dir + '/labels_nucleoli_RG_rt045_v17_wo_holes_shifted_0/statistics_rounded.pkl', "rb")
    #my_dict_2 = pickle.load(pickle_in)
    pickle_in = open(parent_dir + '/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0_new/statistics_rounded.pkl', "rb")
    my_dict_3 = pickle.load(pickle_in)
    pickle_in = open(parent_dir + '/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0_new/statistics_rounded.pkl', "rb")
    my_dict_4 = pickle.load(pickle_in)
    #pickle_in = open(parent_dir + '/dhm_RG_t010_v7_wo_holes/statistics_rounded.pkl', "rb")
    #my_dict_5 = pickle.load(pickle_in)
    #pickle_in = open(parent_dir + '/labels_nucleoli_fiji/statistics_rounded.pkl', "rb")
    #my_dict_4 = pickle.load(pickle_in)

    my_file = open(parent_dir + '/statistics_fluo_new.txt', "w")
    my_file.write('Metric' + ' '
                  #+ 't-0125-v17' + ' '
                  #+ 'rt-045-v17' + ' '
                  + 't-015-v18' + ' '
                  + 'rt-045-v18' + '\n')

    # for (k_1, v_1), (k_2, v_2), (k_3, v_3), (k_4, v_4)\
    #         in zip(my_dict_1.items(), my_dict_2.items(), my_dict_3.items(), my_dict_4.items()):
    #     my_file.write(k_2 + ' '
    #                   + str(v_1).replace('.', ',') + ' '
    #                   + str(v_2).replace('.', ',') + ' '
    #                   + str(v_3).replace('.', ',') + ' '
    #                   + str(v_4).replace('.', ',') + '\n')

    for (k_3, v_3), (k_4, v_4)\
            in zip(my_dict_3.items(), my_dict_4.items()):
        my_file.write(k_3 + ' '
                      + str(v_3).replace('.', ',') + ' '
                      + str(v_4).replace('.', ',') + '\n')

    my_file.close()


# Set 'run_manual' to True to compute the statistics on the manual contours (useful for the confusion matrix).
# Then, set to False, to get the results for the automatic segmentation.
run_manual = False

if run_manual:
    folder_name1 = 'labels_nucleoli_manual_shifted_0_new'
    compute_features(folder_name1,
                     './samples/labels_nucleoli_manual_shifted_0.npy',
                     './samples/labels_dapi_close.npy', True)
    remove_zeros(folder_name1, './statistics/' + folder_name1 + '/')
    intra_population_statistics('./statistics/' + folder_name1 + '/')
    plot_details_histograms('./statistics/labels_nucleoli_manual_shifted_0_new',
                            './statistics/labels_nucleoli_manual_shifted_0_new')
else:
    folder_name1 = 'labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0_new'
    folder_name2 = 'labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0_new'
    compute_features(folder_name1,
                    './samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0.npy',
                    './samples/labels_dapi_close.npy', False)
    compute_features(folder_name2,
                    './samples/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0.npy',
                    './samples/labels_dapi_close.npy', False)
    remove_zeros_2_inputs(folder_name1, './statistics/' + folder_name1 + '/', folder_name2, './statistics/' + folder_name2 + '/')
    intra_population_statistics('./statistics/' + folder_name1 + '/')
    intra_population_statistics('./statistics/' + folder_name2 + '/')
    plot_details_histograms('./statistics/labels_nucleoli_RG_t015_v18_wo_holes_dilated2_shifted_0_new',
                            './statistics/labels_nucleoli_RG_rt045_v18_wo_holes_shifted_0_new')
    plot_confusion_matrices('./statistics/' + folder_name1 + '/')
    plot_confusion_matrices('./statistics/' + folder_name2 + '/')
    round_metrics()
    print_table_all()
