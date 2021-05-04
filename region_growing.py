# Load images
gfp_images = np.load('./samples/images_gfp_fullsize.npy')
masks_manual = np.load('./samples/seeds_min008_max023_step0025_open2_close3.npy')
# masks_manual = np.load('./samples/labels_nucleoli_cn_035_improved_v2_fullsize.npy')
sh = masks_manual.shape
RG_010_improved = np.zeros((75, sh[1], sh[2]))
RG_035_improved = np.zeros((75, sh[1], sh[2]))
RG_0125_improved = np.zeros((75, sh[1], sh[2]))

counter = 0
for i in range(75):

    masks_manual_i = masks_manual[i]
    gfp_image_i = gfp_images[i]
    RG_010_improved_i = np.zeros(masks_manual_i.shape)
    RG_035_improved_i = np.zeros(masks_manual_i.shape)
    RG_0125_improved_i = np.zeros(masks_manual_i.shape)

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
        tol_035 = max(0, min(maxi * 0.45, maxi - 0.1 * 255))
        tol_0125 = max(0, maxi - 0.15 * 255)
        tol_010 = max(0, maxi - 0.1 * 255)
        region_flood_035 = morphology.flood(gfp_image_i, index_maxi, connectivity=2, tolerance=tol_035)
        region_flood_010 = morphology.flood(gfp_image_i, index_maxi, connectivity=2, tolerance=tol_010)
        region_flood_0125 = morphology.flood(gfp_image_i, index_maxi, connectivity=2, tolerance=tol_0125)

        area_flood = np.sum(region_flood_010)
        area_flood_035 = np.sum(region_flood_035)
        max_flood = max(gfp_image_i[region_flood_010])

        current_mask_010 = np.zeros(gfp_image_i.shape)
        current_mask_035 = np.zeros(gfp_image_i.shape)
        current_mask_0125 = np.zeros(gfp_image_i.shape)
        if area_flood > 4500:  # mich too big
            current_mask_0125[region_flood_0125 == 1] = 0
            current_mask_035[region_flood_035 == 1] = 0
            RG_0125_improved_i = np.logical_or(RG_0125_improved_i, current_mask_0125)
            RG_035_improved_i = np.logical_or(RG_035_improved_i, current_mask_035)
        elif area_flood > 1000:
            if max_flood < 75:  # mitose
                current_mask_0125[region_flood_0125 == 1] = 0
                current_mask_035[region_flood_035 == 1] = 0
                RG_0125_improved_i = np.logical_or(RG_0125_improved_i, current_mask_0125)
                RG_035_improved_i = np.logical_or(RG_035_improved_i, current_mask_035)
            # elif area_flood/area_flood_035 > 5:
            #    region_flood_x = morphology.flood(gfp_image_i, index_maxi, connectivity=2, tolerance=(tol_010+tol_035)/2)
            #    current_mask_010[region_flood_x == 1] = 1
            #    current_mask_035[region_flood_035 == 1] = 1
            #    RG_010_improved_i = np.logical_or(RG_010_improved_i, current_mask_010)
            #    RG_035_improved_i = np.logical_or(RG_035_improved_i, current_mask_035)
            else:
                current_mask_0125[region_flood_0125 == 1] = 1
                current_mask_035[region_flood_035 == 1] = 1
                RG_0125_improved_i = np.logical_or(RG_0125_improved_i, current_mask_0125)
                RG_035_improved_i = np.logical_or(RG_035_improved_i, current_mask_035)
        else:
            current_mask_0125[region_flood_0125 == 1] = 1
            current_mask_035[region_flood_035 == 1] = 1
            RG_0125_improved_i = np.logical_or(RG_0125_improved_i, current_mask_0125)
            RG_035_improved_i = np.logical_or(RG_035_improved_i, current_mask_035)

    RG_0125_improved[counter] = RG_0125_improved_i
    RG_035_improved[counter] = RG_035_improved_i

    counter = counter + 1

X_DIM = 1460
Y_DIM = 1920
x_dim_round = 128 * (X_DIM // 128)
y_dim_round = 128 * (Y_DIM // 128)
RG_0125_improved_cropped = RG_0125_improved[:, 0:x_dim_round, 0:y_dim_round]
RG_035_improved_cropped = RG_035_improved[:, 0:x_dim_round, 0:y_dim_round]

RG_0125_improved_cropped = RG_0125_improved_cropped.astype(np.uint8)
np.save('./samples/labels_nucleoli_RG_t015_v18.npy', RG_0125_improved_cropped)
RG_0125_improved = RG_0125_improved.astype(np.uint8)
np.save('./samples/labels_nucleoli_RG_t015_v18_fullsize.npy', RG_0125_improved)
RG_035_improved_cropped = RG_035_improved_cropped.astype(np.uint8)
np.save('./samples/labels_nucleoli_RG_rt045_v18.npy', RG_035_improved_cropped)
RG_035_improved = RG_035_improved.astype(np.uint8)
np.save('./samples/labels_nucleoli_RG_rt045_v18_fullsize.npy', RG_035_improved)