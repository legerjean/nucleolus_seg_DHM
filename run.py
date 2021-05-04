from core.utils_core import *
from core.train import *
from core.predict import *
from postprocessing import *
import numpy as np
import csv
import pickle

gpu = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


#####################################
# Segmenters
#####################################

def train_nucleoli_segmenter():
    parameters = {'nTrain':             50, # without data augmentation
                  'nVal':               25,
                  'nTest':              0,
                  'model':              'unet_2d',
                  'n_layers':           5,
                  'n_feat_maps':        16,
                  'batch_size':         1,
                  'nb_epoch':           150,
                  'lr':                 1e-4,
                  'loss':               'dice_loss_2d',
                  'en_online':          0,
                  'wd':                 0,
                  'dropout':            0,
                  'bn':                 0,
                  'factor_augment':     1,
                  'mitose_removal':     0,
                  'init':               'he_uniform',
                  'modality':           'labels_nucleoli_RG_t015_v18_wo_holes'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('./samples/images_phase_shifted.npy')

    if parameters['mitose_removal']:
        output_masks = np.load('./samples/labels_nucleoli_wo_mitose.npy')
    else:
        output_masks = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)


def train_nuclei_segmenter():
    parameters = {'nTrain':             50*4, # without data augmentation
                  'nVal':               25*4,
                  'nTest':              0,
                  'model':              'unet_2d',
                  'n_layers':           3,
                  'n_feat_maps':        16,
                  'batch_size':         1,
                  'nb_epoch':           10,
                  'lr':                 1e-4,
                  'loss':               'dice_loss_2d',
                  'en_online':          0,
                  'wd':                 0,
                  'dropout':            0,
                  'bn':                 0,
                  'factor_augment':     1,
                  'mitose_removal':     0,
                  'init':               'he_uniform',
                  'modality':           'interleaved_3layers'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/images_25_first_filtered_interleaved_uint8.npy')

    if parameters['mitose_removal']:
        output_masks = np.load('./samples/labels_nucleoli_wo_mitose.npy')
    else:
        output_masks = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/thr_all_close_25_first_filtered_interleaved.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)

def train_nuclei_segmenter2():
    parameters = {'nTrain':             50*4, # without data augmentation
                  'nVal':               25*4,
                  'nTest':              0,
                  'model':              'unet_2d_weighted',
                  'n_layers':           5,
                  'n_feat_maps':        16,
                  'batch_size':         1,
                  'nb_epoch':           10,
                  'lr':                 1e-4,
                  'loss':               'weighted_dice_loss',
                  'en_online':          0,
                  'wd':                 0,
                  'dropout':            0,
                  'bn':                 0,
                  'factor_augment':     1,
                  'mitose_removal':     0,
                  'init':               'he_uniform',
                  'modality':           'interleaved_weighted'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/images_25_first_filtered_interleaved_uint8.npy')

    if parameters['mitose_removal']:
        output_masks = np.load('./samples/labels_nucleoli_wo_mitose.npy')
    else:
        output_masks = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/thr_all_close_25_first_filtered_interleaved.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)

def train_nuclei_segmenter2_online():
    parameters = {'nTrain':             50*4, # without data augmentation
                  'nVal':               25*4,
                  'nTest':              0,
                  'model':              'unet_2d',
                  'n_layers':           5,
                  'n_feat_maps':        16,
                  'batch_size':         2,
                  'nb_epoch':           150,
                  'lr':                 1e-4,
                  'loss':               'dice_loss_2d',
                  'en_online':          1,
                  'wd':                 0,
                  'dropout':            0,
                  'bn':                 0,
                  'factor_augment':     1,
                  'mitose_removal':     0,
                  'init':               'he_uniform',
                  'modality':           'nucleoli_interleaved_online'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/images_25_first_filtered_interleaved_uint8.npy')

    if parameters['mitose_removal']:
        output_masks = np.load('./samples/labels_nucleoli_wo_mitose.npy')
    else:
        output_masks = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/thr_all_close_25_first_filtered_interleaved.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)

def train_seeds_nucleoli_segmenter_gfp():
    parameters = {'nTrain': 50,  # without data augmentation
                 'nVal': 25,
                 'nTest': 0,
                 'model': 'unet_2d',
                 'n_layers': 3,
                 'n_feat_maps': 16,
                 'batch_size': 1,
                 'nb_epoch': 100,
                 'lr': 1e-4,
                 'loss': 'dice_loss_2d',
                 'en_online': 0,
                 'wd': 0,
                 'dropout': 0,
                 'bn': 0,
                 'factor_augment': 1,
                 'mitose_removal': 0,
                 'init': 'he_uniform',
                 'modality': 'dilated2_seeds_nucleoli_gfp_boost'}

    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('./samples/images_gfp.npy')
    output_masks = np.load('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_gfp/predictions.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)


def train_seeds_nucleoli_segmenter_phase():
    parameters = {'nTrain': 50,  # without data augmentation
                  'nVal': 25,
                  'nTest': 0,
                  'model': 'unet_2d',
                  'n_layers': 5,
                  'n_feat_maps': 16,
                  'batch_size': 1,
                  'nb_epoch': 100,
                  'lr': 1e-4,
                  'loss': 'dice_loss_2d',
                  'en_online': 0,
                  'wd': 0,
                  'dropout': 0,
                  'bn': 0,
                  'factor_augment': 1,
                  'mitose_removal': 0,
                  'init': 'he_uniform',
                  'modality': 'dilated2_seeds_nucleoli_filtered_phase_reg'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('./samples/images_phase.npy')
    output_masks = np.load('./results/model_unet_2d_n_layers_5_modality_dilated2_seeds_nucleoli_gfp/predictions.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)


def predict_nucleoli_segmenter():
    #input_images = np.load('./samples/images_gfp.npy')
    input_images_0 = np.load('./samples/images_phase.npy')
    sh = input_images_0.shape
    input_images = np.zeros((75, sh[1], sh[2], 3))
    input_images[:, :, :, 0] = input_images_0
    input_images[:, :, :, 1] = np.load('./samples/images_gfp.npy')
    input_images[:, :, :, 2] = np.load('./samples/images_dapi.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_3channels_n_layers_5_modality_3channels_virtual_annotations'
    run_cv_predict_3channels(input_images, predict_set, results_name_short)
    #concatenate_predictions(predict_set, results_name_short, input_images.shape)
    concatenate_predictions(predict_set, results_name_short, (75, sh[1], sh[2]))

    
def predict_nuclei_segmenter():
    input_images = np.load('./samples/images_phase_shifted.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_n_layers_5_modality_samples_nucleoli_cropped'
    run_cv_predict(input_images, predict_set, results_name_short)
    concatenate_predictions(predict_set, results_name_short, input_images.shape)


def predict_nuclei_segmenter2():
    input_images = np.load('/export/home/jleger/Documents/segmentation/microscopy_regularization/samples/images_25_first_filtered_interleaved_uint8.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_n_layers_5_modality_thr012'
    run_cv_predict(input_images, predict_set, results_name_short)
    concatenate_predictions(predict_set, results_name_short, input_images.shape)


def predict_seeds_nucleoli_segmenter():
    input_images = np.load('./samples/images_gfp.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_n_layers_5_modality_dilated2_seeds_nucleoli_gfp_boost'
    run_cv_predict(input_images, predict_set, results_name_short)
    concatenate_predictions(predict_set, results_name_short, input_images.shape)


def predict_seeds_nucleoli_segmenter_phase():
    input_images = np.load('./samples/images_dapi.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_n_layers_5_modality_nucleoli_mult_from_dapi'
    run_cv_predict(input_images, predict_set, results_name_short)
    concatenate_predictions(predict_set, results_name_short, input_images.shape)



#####################################
# Utils
#####################################


def run_cv_train(params, images, masks, params_entries):
    # Save parameters and create results folder path
    results_name_short = params2name({k: params[k] for k in params_entries})
    results_path = './results/' + results_name_short
    save_params(params, results_path)
    params['nTrain'] = params['nTrain'] * params['factor_augment']

    # Run the cross validation training
    for i in [1]:#[0, 1, 2]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], results_path, i, True)

        #model = unet_2d(params)
        #model.load_weights('./results/model_unet_2d_n_layers_5_modality_nucleoli/firstval' + str(cv['val'][0]) + '/weights.h5')
        #train_model_weighted(params, cv, images, masks, 0, 1, results_path, True)
        train_model(params, cv, images, masks, 0, 1, results_path, True)


def run_cv_train_3channels(params, images, masks, params_entries):
    # Save parameters and create results folder path
    results_name_short = params2name({k: params[k] for k in params_entries})
    results_path = './results/' + results_name_short
    save_params(params, results_path)
    params['nTrain'] = params['nTrain'] * params['factor_augment']

    # Run the cross validation training
    for i in [0, 1,2]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], results_path, i, True)
        #model = unet_2d(params)
        #model.load_weights('./results/model_unet_2d_n_layers_5_modality_nucleoli/firstval' + str(cv['val'][0]) + '/weights.h5')
        train_model_3channels(params, cv, images, masks, 0, 1, results_path, True)

        
def run_cv_predict(images, predict_set, results_name_short):
    # Load params and create results folder path
    results_path = './results/' + results_name_short
    with open(results_path + '/params.p', 'rb') as handle:
        params = pickle.load(handle)
    nb_classes = 1  # excluding the background

    # Run the cross validation prediction
    for i in [1]:#[0, 1, 2]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], results_path, i, True)
        if predict_set == 'val':
            selected_indices = cv['val']
        else:
            print('ERROR: prediction not on the validation set is not yet supported')
        #predict_model_weighted(images[selected_indices], params, nb_classes, cv, results_path, True)
        predict_model(images[selected_indices], params, nb_classes, cv, results_path, True)


def run_cv_predict_3channels(images, predict_set, results_name_short):
    # Load params and create results folder path
    results_path = './results/' + results_name_short
    with open(results_path + '/params.p', 'rb') as handle:
        params = pickle.load(handle)
    nb_classes = 1  # excluding the background

    # Run the cross validation prediction
    for i in [0, 1, 2]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], results_path, i, True)
        if predict_set == 'val':
            selected_indices = cv['val']
        else:
            print('ERROR: prediction not on the validation set is not yet supported')
        predict_model_3channels(images[selected_indices], params, nb_classes, cv, results_path, True)


def concatenate_predictions(predict_set, results_name_short, shape_images):
    # Load params
    results_path = './results/' + results_name_short
    with open(results_path + '/params.p', 'rb') as handle:
        params = pickle.load(handle)

    # Run the cross validation prediction
    predictions = np.zeros(shape_images)
    for i in [0, 1, 2]:
        print('======================================== cvNum ' + str(i) + ' ========================================')
        cv = cv_index_generator(params, params['nTrain'], results_path, i, True)
        if predict_set == 'val':
            predictions_fold = np.load(results_path + '/firstval' + str(cv['val'][0]) + '/predictions.npy')
            predictions[cv['val']] = predictions_fold
        else:
            print('ERROR: concatenation not on the validation set is not yet supported')
    predictions = predictions.astype(np.uint8)
    np.save(results_path + '/predictions.npy', predictions)

#####################################
# Run segmenters
#####################################

#train_nucleoli_segmenter_3channels()
#train_nucleoli_segmenter()
#train_nucleoli_segmenterbis()
#train_nucleoli_segmenterbisbis()
#train_nuclei_segmenter2()
#train_nuclei_segmenter()
#train_nuclei_segmenter2_online()

# train_seeds_nucleoli_segmenter_phase()
# predict_seeds_nucleoli_segmenter_phase()
# train_seeds_nucleoli_segmenter_gfp()
#predict_nucleoli_segmenter()
# predict_seeds_nucleoli_segmenter_phase()
predict_nuclei_segmenter2()
# predict_seeds_nucleoli_segmenter()


#####################################
# Clean predictions
#####################################

# nuclei_pred_thr = np.load('./samples/dapi_virtual_annotations.npy')
#
# nucleoli_pred_thr = np.load('./samples/gfp_virtual_annotations.npy')
# # seeds_pred_thr = np.load('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_filtered_phase/predictions.npy')
# # # seeds_pred_reg = np.load('./results/model_unet_2d_n_layers_5_modality_dilated2_seeds_nucleoli_filtered_phase_reg/predictions.npy')
# # seeds_pred_gfp = np.load('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_gfp/predictions.npy')
# # seeds_pred_gfp_boost = np.load('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_gfp_boost/predictions.npy')
# #
# nuclei_clean_acc, nucleoli_clean_acc = clean_predictions(nucleoli_pred_thr, nuclei_pred_thr)
# # nuclei_clean_acc, seeds_clean_acc = clean_predictions(seeds_pred_thr, nuclei_pred_thr)
# # # nuclei_clean_acc, seeds_clean_reg_acc = clean_predictions(seeds_pred_reg, nuclei_pred_thr)
# # nuclei_clean_acc, seeds_clean_gfp_acc = clean_predictions(seeds_pred_gfp, nuclei_pred_thr)
# # nuclei_clean_acc, seeds_clean_gfp_boost_acc = clean_predictions(seeds_pred_gfp_boost, nuclei_pred_thr)
# #
# np.save('./samples/dapi_virtual_annotations_wo_cluster.npy', nuclei_clean_acc)
#
# #np.save('./samples/labels_cells.npy', nuclei_clean_acc)
# np.save('./samples/gfp_virtual_annotations_wo_cluster.npy', nucleoli_clean_acc)
# # np.save('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_filtered_phase/predictions_clean.npy', seeds_clean_acc)
# # # np.save('./results/model_unet_2d_n_layers_5_modality_dilated2_seeds_nucleoli_filtered_phase_reg/predictions_clean.npy', seeds_clean_reg_acc)
# # np.save('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_gfp/predictions_clean.npy', seeds_clean_gfp_acc)
# # np.save('./results/model_unet_2d_n_layers_3_modality_dilated2_seeds_nucleoli_gfp_boost/predictions_clean.npy', seeds_clean_gfp_boost_acc)
