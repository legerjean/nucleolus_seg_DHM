from core.utils_core import *
from core.train import *
import numpy as np

gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def predict_model_3channels(images, params, nb_classes, cv, results_path, en_save=True):

    # Get model
    if params['model'] == 'unet_2d':
        model = unet_2d(params)
    elif params['model'] == 'unet_2d_v2':
        model = get_2dunet_v2(params)
    elif params['model'] == 'unet_2d_v3':
        model = get_2dunet_v3(params)
    elif params['model'] == 'unet_3d':
        model = unet_3d(params)
    elif params['model'] == 'unet_2d_3channels':
        model = unet_2d_3channels(params)
    model.load_weights(results_path + '/firstval' + str(cv['val'][0]) + '/weights.h5')
    
    # Load normalization parameters
    pickle_in = open(results_path + '/firstval' + str(cv['val'][0]) + '/norm_params.p', "rb")
    norm_params = pickle.load(pickle_in)
    mu = norm_params['mu']
    sigma = norm_params['sigma']

    # Perform prediction
    sh = images.shape
    if nb_classes > 1:
        predictions = np.zeros((sh[0], sh[1], sh[2], nb_classes))
    else:
        predictions = np.zeros((sh[0], sh[1], sh[2]))
    predictions_thr = np.copy(predictions)

    for i in range(sh[0]):
        test_inputs_i_0 = (images[i, :, :, 0] - mu[0]) / sigma[0]
        test_inputs_i_0 = np.expand_dims(test_inputs_i_0, axis=-1)
        test_inputs_i_0 = np.expand_dims(test_inputs_i_0, axis=0)
        test_inputs_i_1 = (images[i, :, :, 1] - mu[1]) / sigma[1]
        test_inputs_i_1 = np.expand_dims(test_inputs_i_1, axis=-1)
        test_inputs_i_1 = np.expand_dims(test_inputs_i_1, axis=0)
        test_inputs_i_2 = (images[i, :, :, 2] - mu[2]) / sigma[2]
        test_inputs_i_2 = np.expand_dims(test_inputs_i_2, axis=-1)
        test_inputs_i_2 = np.expand_dims(test_inputs_i_2, axis=0)
        test_predictions_i = model.predict([test_inputs_i_0, test_inputs_i_1, test_inputs_i_2], batch_size=params['batch_size'], verbose=0)
        predictions[i] = np.squeeze(test_predictions_i)

    predictions_thr[predictions > 0.5] = 1
    predictions_thr = predictions_thr.astype(np.uint8)
    
    if en_save:
        np.save(results_path + '/firstval' + str(cv['val'][0]) + '/predictions.npy', predictions_thr)


def predict_model(images, params, nb_classes, cv, results_path, en_save=True):
    # Get model
    if params['model'] == 'unet_2d':
        model = unet_2d(params)
    elif params['model'] == 'unet_2d_v2':
        model = get_2dunet_v2(params)
    elif params['model'] == 'unet_2d_v3':
        model = get_2dunet_v3(params)
    elif params['model'] == 'unet_3d':
        model = unet_3d(params)
    model.load_weights(results_path + '/firstval' + str(cv['val'][0]) + '/weights.h5')

    # Load normalization parameters
    pickle_in = open(results_path + '/firstval' + str(cv['val'][0]) + '/norm_params.p', "rb")
    norm_params = pickle.load(pickle_in)

    # Perform prediction
    sh = images.shape
    if nb_classes > 1:
        predictions = np.zeros((sh[0], sh[1], sh[2], nb_classes))
    else:
        predictions = np.zeros((sh[0], sh[1], sh[2]))
    predictions_thr = np.copy(predictions)

    for i in range(sh[0]):
        test_inputs_i = (images[i, :, :] - norm_params['mu']) / norm_params['sigma']
        test_inputs_i = np.expand_dims(test_inputs_i, axis=-1)
        test_inputs_i = np.expand_dims(test_inputs_i, axis=0)
        test_predictions_i = model.predict(test_inputs_i, batch_size=params['batch_size'], verbose=0)
        predictions[i] = np.squeeze(test_predictions_i)

    predictions_thr[predictions > 0.5] = 1
    predictions_thr = predictions_thr.astype(np.uint8)

    if en_save:
        np.save(results_path + '/firstval' + str(cv['val'][0]) + '/predictions.npy', predictions_thr)