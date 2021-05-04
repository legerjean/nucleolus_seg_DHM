inputs_gfp = np.load('./samples/images_gfp_fullsize.npy')
inputs_gfp = np.divide(inputs_gfp, 255)

# Nucleoli GT for counting
#thr_min = 0.05
thr_min = 0.08 # 0.1
thr_max = 0.23#0.5
thr_list = np.arange(thr_min, thr_max, 0.025) # thr_list = np.arange(thr_min, thr_max, 0.025)

NB_IMAGES = 75
X_DIM = 1460
Y_DIM = 1920
gfp_centers_init = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
gfp_centers = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
gfp_centers_filtered = np.zeros((NB_IMAGES, X_DIM, Y_DIM))
sh = gfp_centers.shape

for i in range(75):
    my_gfp = inputs_gfp[i]
    gfp_labels = label(np.ones((X_DIM, Y_DIM)), background=0)
    gfp_regions = regionprops(gfp_labels, intensity_image=my_gfp)

    for counter, thr in enumerate(thr_list):
        # Thresholding and morphological operation
        gfp_thr = np.zeros((X_DIM, Y_DIM))
        gfp_thr[my_gfp > thr] = 1
        se = disk(2)
        gfp_open = ndimage.binary_opening(gfp_thr, structure=se) # retirer les pixels isolés
        se = disk(3)
        gfp_close = ndimage.binary_closing(gfp_open, structure=se) # relier les petits pixels adjacents
        gfp_final = np.copy(gfp_close)
        #se = disk(3)
        #gfp_close = ndimage.binary_closing(gfp_thr, structure=se) # relier les petits pixels adjacents
        #se = disk(2)
        #gfp_open = ndimage.binary_opening(gfp_close, structure=se) # retirer les pixels isolés
        #gfp_final = np.copy(gfp_open)

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

    # Dilation of the seeds
    se = disk(2)
    gfp_final_dilated = ndimage.binary_dilation(gfp_final, structure=se)
    gfp_centers[i] = gfp_final_dilated

    # Filtering of the big centers
    labels = label(gfp_final_dilated, background=0)
    regions = regionprops(labels, gfp_final_dilated)
    nb_regions = len(regions)
    for j in range(nb_regions):
        reg = regions[j]
        lab = reg.label
        area = reg.area
        if area > 500:
            gfp_centers_i = gfp_final_dilated
            gfp_centers_i[labels == lab] = 0
            gfp_final_dilated = gfp_centers_i

    gfp_centers_filtered[i] = gfp_final_dilated
    print(i)

gfp_centers_init = gfp_centers_init.astype(np.uint8)
gfp_centers = gfp_centers.astype(np.uint8)
gfp_centers_filtered = gfp_centers_filtered.astype(np.uint8)

#np.save('./samples/seeds_dilation2.npy',gfp_centers)
#np.save('./samples/seeds_dilation2_filtered.npy',gfp_centers_filtered)
np.save('./samples/seeds_min008_max023_step0025_open2_close3.npy', gfp_centers_init)

#for i in range(3):
#    plt.figure(figsize=(15,15))
#    plt.imshow(inputs_gfp[i]*255,cmap='gray')
#    plt.figure(figsize=(15,15))
#    plt.imshow(gfp_centers[i]*255,cmap='gray')
#plt.show()
print('--- Seeds generation finished ---')
