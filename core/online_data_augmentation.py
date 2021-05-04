import os
import numpy as np
import keras
import random
from scipy import ndimage

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def image_transform_2d(image, shears, angles, shifts, order, dims):
    shear_matrix = np.array([[1,            shears[0],    0],
                             [shears[1],    1,            0],
                             [0,            0,            1]])

    shift_matrix = np.array([[1, 0,  shifts[0]],
                             [0, 1,  shifts[1]],
                             [0, 0,  1]])

    offset = np.array([[1, 0, dims[0]],
                       [0, 1, dims[1]],
                       [0, 0, 1]])

    offset_opp = np.array([[1, 0, -dims[0]],
                           [0, 1, -dims[1]],
                           [0, 0, 1]])

    angle = np.deg2rad(angles)
  
    rotz = np.array([[np.cos(angle),    -np.sin(angle),  0],
                     [np.sin(angle),    np.cos(angle),   0],
                     [0,                    0,                   1]])

    rotation_matrix = offset_opp.dot(rotz).dot(offset)
    affine_matrix = shift_matrix.dot(rotation_matrix).dot(shear_matrix)

    return ndimage.interpolation.affine_transform(image, affine_matrix, order=order, mode='nearest')


class DataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, params, training_images, training_masks):
        'Initialization'
        self.sh = training_images.shape
        self.batch_size = params['batch_size']
        self.list_IDs = np.arange(self.sh[0])
        self.n_channels = 1
        self.shuffle = True
        self.on_epoch_end()
        self.training_images = training_images
        self.training_masks = training_masks

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        dim = self.sh[1:]
        X = np.empty((self.batch_size, *dim, self.n_channels))
        Y = np.empty((self.batch_size, *dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            im = self.training_images[ID]
            if len(dim) == 2:
                shears = np.array([0.02 * random.uniform(-1, 1) for _ in range(2)])
                angle = np.array([5 * random.uniform(-1, 1)])
                shifts = np.array([0.05 * random.uniform(-1, 1) * dim[j] for j in range(2)])
                im = image_transform_2d(im, shears, angle, shifts, order=3, dims=dim)
            else:
                print('Online data augmentation - input dimensions not supported')
            X[i,] = np.expand_dims(im, axis=-1)

            # Store class
            mask = self.training_masks[ID]
            Y[i,] = np.expand_dims(mask, axis=-1)

        return X, Y


class DataGeneratorVal(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, params, validation_images, validation_masks):
        'Initialization'
        self.sh = validation_images.shape
        self.batch_size = params['batch_size']
        self.list_IDs = np.arange(self.sh[0])
        self.n_channels = 1
        self.shuffle = True
        self.on_epoch_end()
        self.validation_images = validation_images
        self.validation_masks = validation_masks

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        dim = self.sh[1:]
        X = np.empty((self.batch_size, *dim, self.n_channels))
        Y = np.empty((self.batch_size, *dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            im = self.validation_images[ID]
            X[i,] = np.expand_dims(im, axis=-1)

            # Store class
            mask = self.validation_masks[ID]
            Y[i,] = np.expand_dims(mask, axis=-1)

        return X, Y
