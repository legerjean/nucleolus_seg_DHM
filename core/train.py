
import numpy as np
from core.utils_core import *
from keras.callbacks import ModelCheckpoint
import os
import time
from core.online_data_augmentation import *
import pickle

from tensorflow import set_random_seed
set_random_seed(3)

def train_model(params, cv, images, masks, pretrained_model, gpu, results_path, en_test):
    time_start = time.clock()

    # Fix random seed
    seed = 1
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if not os.path.exists(results_path + '/firstval' + str(cv['val'][0])):
        os.mkdir(results_path + '/firstval' + str(cv['val'][0]))

    # Load data
    list_train = cv['train']
    list_val = cv['val']
    if en_test:
        list_test = cv['test']

    train_images = images[list_train]
    train_masks = masks[list_train]
    val_images = images[list_val]
    val_masks = masks[list_val]
    if en_test:
        test_images = images[list_test]
        test_masks = masks[list_test]

    # Normalize data
    norm_params = {}
    norm_params['mu'] = np.mean(train_images)
    norm_params['sigma'] = np.std(train_images)
    pickle.dump(norm_params, open(results_path + '/firstval' + str(cv['val'][0]) + '/norm_params.p', "wb"))
    train_images = (train_images - norm_params['mu']) / norm_params['sigma']
    val_images = (val_images - norm_params['mu']) / norm_params['sigma']
    if en_test:
        test_images = (test_images - norm_params['mu']) / norm_params['sigma']

    # Add dimension for channel
    if not params['en_online']:
        train_images = np.expand_dims(train_images, axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)
        val_images = np.expand_dims(val_images, axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)
        if en_test:
            test_images = np.expand_dims(test_images, axis=-1)
            test_masks = np.expand_dims(test_masks, axis=-1)

    # Train model and save best
    if pretrained_model != 0:
        model = pretrained_model
    elif params['model'] == 'unet_2d':
        model = unet_2d(params)
    elif params['model'] == 'unet_2d_v2':
        model = get_2dunet_v2(params)
    elif params['model'] == 'unet_2d_v3':
        model = get_2dunet_v3(params)
    elif params['model'] == 'unet_2d_v4':
        model = unet_2d_v4(params)
    elif params['model'] == 'unet_3d':
        model = unet_3d(params)
    elif params['model'] == 'unet_2d_3channels':
        model = unet_2d_3channels(params)

    model_checkpoint = ModelCheckpoint(results_path + '/firstval' + str(cv['val'][0]) + '/weights.h5',
                                       verbose=1,
                                       monitor='val_' + params['loss'],
                                       save_best_only=False,
                                       save_weights_only=True,
                                       period=2)

    if params['en_online']:
        training_generator = DataGeneratorTrain(params, train_images, train_masks)
        validation_generator = DataGeneratorVal(params, val_images, val_masks)
        hist = model.fit_generator(generator=training_generator,
                                   validation_data=validation_generator,
                                   use_multiprocessing=False,
                                   workers=1,
                                   steps_per_epoch=len(list_train),
                                   validation_steps=len(list_val),
                                   verbose=1,
                                   epochs=params['nb_epoch'],
                                   callbacks=[model_checkpoint])
    else:
        hist = model.fit(train_images,
                     train_masks,
                     batch_size=params['batch_size'],
                     nb_epoch=params['nb_epoch'],
                     verbose=2,
                     shuffle=True,
                     validation_data=(val_images, val_masks),
                     callbacks=[model_checkpoint])

    # Save training stats
    train_time = (time.clock() - time_start)
    np.save(results_path + '/firstval' + str(cv['val'][0]) + '/train_time.npy', train_time)
    save_history(hist.history, params, cv, results_path)


def train_model_3channels(params, cv, images, masks, pretrained_model, gpu, results_path, en_test):
    time_start = time.clock()

    # Fix random seed
    seed = 1
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if not os.path.exists(results_path + '/firstval' + str(cv['val'][0])):
        os.mkdir(results_path + '/firstval' + str(cv['val'][0]))

    # Load data
    list_train = cv['train']
    list_val = cv['val']
    if en_test:
        list_test = cv['test']

    train_images = images[list_train]
    train_masks = masks[list_train]
    val_images = images[list_val]
    val_masks = masks[list_val]
    if en_test:
        test_images = images[list_test]
        test_masks = masks[list_test]



    # Normalize data
    norm_params = {}
    mu = np.zeros(3)
    sigma = np.zeros(3)
    for i in range(3):
        mu[i] = np.mean(train_images[:, :, :, i])
        sigma[i] = np.std(train_images[:, :, :, i])
    norm_params['mu'] = mu
    norm_params['sigma'] = sigma
    pickle.dump(norm_params, open(results_path + '/firstval' + str(cv['val'][0]) + '/norm_params.p', "wb"))
    for i in range(3):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mu[i]) / sigma[i]
        val_images[:, :, :, i] = (val_images[:, :, :, i] - mu[i]) / sigma[i]
    if en_test:
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mu[i]) / sigma[i]


    # Train model and save best
    if pretrained_model != 0:
        model = pretrained_model
    elif params['model'] == 'unet_2d':
        model = unet_2d(params)
    elif params['model'] == 'unet_2d_v2':
        model = get_2dunet_v2(params)
    elif params['model'] == 'unet_2d_v3':
        model = get_2dunet_v3(params)
    elif params['model'] == 'unet_2d_v4':
        model = unet_2d_v4(params)
    elif params['model'] == 'unet_3d':
        model = unet_3d(params)
    elif params['model'] == 'unet_2d_3channels':
        model = unet_2d_3channels(params)

    model_checkpoint = ModelCheckpoint(results_path + '/firstval' + str(cv['val'][0]) + '/weights.h5',
                                       verbose=1,
                                       monitor='val_' + params['loss'],
                                       save_best_only=False,
                                       save_weights_only=True,
                                       period=2)

    if params['en_online']:
        training_generator = DataGeneratorTrain(params, train_images, train_masks)
        validation_generator = DataGeneratorVal(params, val_images, val_masks)
        hist = model.fit_generator(generator=training_generator,
                                   validation_data=validation_generator,
                                   use_multiprocessing=False,
                                   workers=1,
                                   steps_per_epoch=len(list_train),
                                   validation_steps=len(list_val),
                                   verbose=1,
                                   epochs=params['nb_epoch'],
                                   callbacks=[model_checkpoint])
    else:
        input0 = np.expand_dims(train_images[:, :, :, 0], axis=-1)
        input1 = np.expand_dims(train_images[:, :, :, 1], axis=-1)
        input2 = np.expand_dims(train_images[:, :, :, 2], axis=-1)
        train_masks = np.expand_dims(train_masks, axis=-1)
        val0 = np.expand_dims(val_images[:, :, :, 0], axis=-1)
        val1 = np.expand_dims(val_images[:, :, :, 1], axis=-1)
        val2 = np.expand_dims(val_images[:, :, :, 2], axis=-1)
        val_masks = np.expand_dims(val_masks, axis=-1)

        hist = model.fit([input0, input1, input2],
                     train_masks,
                     batch_size=params['batch_size'],
                     nb_epoch=params['nb_epoch'],
                     verbose=2,
                     shuffle=True,
                     validation_data=([val0, val1, val2], val_masks),
                     callbacks=[model_checkpoint])

    # Save training stats
    train_time = (time.clock() - time_start)
    np.save(results_path + '/firstval' + str(cv['val'][0]) + '/train_time.npy', train_time)
    save_history(hist.history, params, cv, results_path)



