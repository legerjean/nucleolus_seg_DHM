from core.utils_core import *
from core.train import *
from core.predict import *
from postprocessing import *
import numpy as np
import csv
import pickle

gpu = 0
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
                  'modality':           'labels_nucleoli_RG_t015_v18_wo_holes_dilated2'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('./samples/images_phase_shifted.npy')
    output_masks = np.load('./samples/labels_nucleoli_RG_t015_v18_wo_holes_dilated2.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)


def train_nuclei_segmenter():
    parameters = {'nTrain': 50,  # without data augmentation
                  'nVal': 25,
                  'nTest': 0,
                  'model': 'unet_2d',
                  'n_layers': 5,
                  'n_feat_maps': 16,
                  'batch_size': 1,
                  'nb_epoch': 150,
                  'lr': 1e-4,
                  'loss': 'dice_loss_2d',
                  'en_online': 0,
                  'wd': 0,
                  'dropout': 0,
                  'bn': 0,
                  'factor_augment': 1,
                  'mitose_removal': 0,
                  'init': 'he_uniform',
                  'modality': 'labels_dapi_filled'}
    parameters_entries = ('model', 'n_layers', 'modality')
    input_images = np.load('./samples/images_phase_shifted.npy')
    output_masks = np.load('./samples/labels_dapi_close.npy')
    run_cv_train(parameters, input_images, output_masks, parameters_entries)


def predict_nucleoli_segmenter():
    input_images = np.load('./samples/images_phase_shifted.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_n_layers_5_modality_labels_nucleoli_RG_t015_v18_wo_holes_dilated2'
    run_cv_predict(input_images, predict_set, results_name_short)
    concatenate_predictions(predict_set, results_name_short, input_images.shape)


def predict_nuclei_segmenter():
    input_images = np.load('./samples/images_phase_shifted.npy')
    predict_set = 'val'
    results_name_short = 'model_unet_2d_n_layers_5_modality_labels_dapi_filled'
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

        
def run_cv_predict(images, predict_set, results_name_short):
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
        #predict_model_weighted(images[selected_indices], params, nb_classes, cv, results_path, True)
        predict_model(images[selected_indices], params, nb_classes, cv, results_path, True)


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


train_nucleoli_segmenter()
train_nuclei_segmenter()
predict_nucleoli_segmenter()
predict_nuclei_segmenter()
