from __future__ import print_function

import numpy as np
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import h5py, hashlib
from ezprogbar import ProgressBar
from keras.utils import np_utils
from dlutils import *

# Load training and test data
def load_data(train_data_file, test_data_file, **opt):
    normalize = opt.get('normalize', True)
    flatten = opt.get('flatten', False)
    train_class_size = opt.get('train_class_size', None)
    test_class_size = opt.get('test_class_size', None)

    print("Loading training data set:", train_data_file)
    X_train, y_train, Y_train, features = load_set(
        train_data_file,
        normalize=normalize,
        flatten=flatten,
        class_size=train_class_size,
        verbose=False,
    )

    print("Loading validation data set:", test_data_file)
    X_test, y_test, Y_test, _ = load_set(
        test_data_file,
        normalize=normalize,
        flatten=flatten,
        class_size=test_class_size,
        verbose=False,
    )

    data_shape = X_train.shape
    item_shape = data_shape[1:]
    print(data_shape[0], 'training samples')
    print(X_test.shape[0], 'validation samples')
    print('Image shape:', item_shape)
    return X_train, y_train, Y_train, X_test, y_test, Y_test

def load_set(data_file, **opt):
    normalize = opt.get('normalize', True)
    flatten = opt.get('flatten', False)
    class_size = opt.get('class_size', None)
    sample = opt.get('sample', None)
    verbose = opt.get('verbose', True)
    X, y, features = load_h5(data_file, class_size=class_size, sample=sample)
    num_classes = len(features)
    num_samples = X.shape[0]
    data_shape = X.shape
    item_shape = data_shape[1:]
    ndim = len(item_shape)
    if verbose:
        print(X.shape[0], 'samples')
        print("Item shape:", item_shape)

    # The original data of each image is a NxNx3 matrix of integers from 0 to 255.
    # We need to scale it down to floats in in a balanced unit interval
    if normalize:
        if verbose:
            print("Normalizing data")
        X = data_normalization(X)
    if flatten:
        if verbose:
            print("Flattening data")
        X = X.reshape((num_samples, np.prod(item_shape)))

    Y = np_utils.to_categorical(y, num_classes)
    return X, y, Y, features

def load_h5(h5file, **opt):
    sample = opt.get('sample', None)
    class_size = opt.get('class_size', None)
    images = []
    classes = []
    f = h5py.File(h5file, 'r')
    num_images = int(f.get('num_images').value)
    features = f.get('features').value.tolist()
    print("Total num images in file:", num_images)

    I = range(0, num_images)
    if sample is None:
        N = num_images
    elif sample > num_images:
        raise Exception("sample exceeds num_images")
    else:
        N = sample
        I = random.sample(I, N)
        print("Sampling %d images from %d" % (N, num_images))

    pm = ProgressBar(N, prompt="Load progress: ")
    for i in I:
        img_key = 'img_' + str(i)
        cls_key = 'cls_' + str(i)
        img = np.array(f.get(img_key))
        cls = f.get(cls_key).value
        images.append(img)
        classes.append(cls)
        pm.advance()
    f.close()

    X = np.array(images)
    y = np.array(classes)
    if class_size != None:
        X, y = balanced_sample(X, y, features, class_size)
    return X, y, features 

def data_normalization(X):
    data_shape = X.shape
    ndim = len(data_shape)
    # gray scale image, must be reshaped for Keras Convolution API ...
    if ndim == 3:
        data_shape += (1,)
        X = X.reshape(data_shape)
    X = X.astype('float32')
    X = X - np.mean(X, axis = 0)
    X = X / 255
    #X /= np.std(X, axis = 0) # normalize
    #X = X / (X.max(axis=0) - X.min(axis=0))
    return X

def save_h5_from_data(savefile, X, y, features):
    print("Writing file:", savefile)
    pm = ProgressBar(len(X))
    f = h5py.File(savefile, 'w')
    i = 0
    for img,cls in zip(X, y):
        f.create_dataset('img_' + str(i), data=img, compression='gzip')
        f.create_dataset('cls_' + str(i), data=cls)
        pm.advance()
        i += 1
    f.create_dataset('num_images', data=i)
    f.create_dataset('features', data=features)
    f.close()
    return savefile

def save_h5_from_file(inpfile, savefile, class_size):
    X, y, features = load_h5(inpfile, class_size=class_size)
    save_h5_from_data(savefile, X, y, features)
    return savefile

def check_data_set(model, h5file, sample=None):
    X, y, features = load_h5(h5file, sample=sample)
    num_classes = len(features)
    print("Loaded %d images" % (X.shape[0],))
    X = data_normalization(X)
    Y = np_utils.to_categorical(y, num_classes)
    loss, acc = model.evaluate(X, Y)
    print("Data shape:", X.shape)
    print("accuracy   = %.6f loss = %.6f" % (acc, loss))
    return acc, loss

def stddev_scaling(X):
    X -= np.mean(X, axis = 0) # zero-center
    X /= np.std(X, axis = 0) # std normalize

# old code: should be done with h5py
# save a subset of data to an npz numpy file
# make sure the sampled data is balanced
# def sample_data(train_class_size, test_class_size, classes, npfile):
#     X_train, y_train, Y_train, X_test, y_test, Y_test, num_classes = load_data()
#     X_train, y_train = balanced_sample(X_train, y_train, classes, train_class_size)
#     X_test, y_test = balanced_sample(X_test, y_test, classes, test_class_size)
#     Y_train = np_utils.to_categorical(y_train, nb_classes)
#     Y_test = np_utils.to_categorical(y_test, nb_classes)
#     np_save_data(npfile, X_train, y_train, Y_train, X_test, y_test, Y_test)
#     return npfile

# old code: should be done with h5py
# def np_save_data(npfile, X_train, y_train, Y_train, X_test, y_test, Y_test):
#     print("Saving data to file:", npfile)
#     np.savez(npfile, X_train=X_train, y_train=y_train, Y_train=Y_train, X_test=X_test, y_test=y_test, Y_test=Y_test)
#     return npfile
#
# old code: should be done with h5py
# def np_restore_data(npfile):
#     print("Restoring data from npfile:", npfile)
#     d = np.load(npfile)
#     names = ['X_train', 'y_train', 'Y_train', 'X_test', 'y_test', 'Y_test']
#     return (d[key] for key in names)

def h5_append(h5file, key, value):
    f = h5py.File(h5file, 'a')
    f.create_dataset(key, data=value)
    f.close()

def h5_overwrite(h5file, key, value):
    f = h5py.File(fl, 'a')
    del f[key]
    f.create_dataset(key, data=value)
    f.close()

def h5_get(h5file, key):
    f = h5py.File(h5file, 'r')
    value = f.get(key).value
    f.close()
    return value

def h5_imshow(h5file, i):
    im = h5_get(h5file, 'img_' + str(i))
    print("Class:", h5_get(h5file, 'cls_' + str(i)))
    print("Shape:", im.shape)
    plt.imshow(im, cmap='jet', interpolation='none')
    plt.show()

def view_false_predictions(model, X, Y, offset=0, interpolation='nearest'):
    print("Computing false_pred and y_pred")
    y_pred, false_preds = get_false_predictions(model, X, Y)
    print("Total false predictions:", len(false_preds))
    for i,(x,y,p) in enumerate(false_preds[offset:offset+15]):
        plt.subplot(3, 5, i+1)
        plt.imshow(x, cmap='jet', interpolation=interpolation)
        plt.title("y: %s\np: %s" % (y, p), fontsize=10, loc='left')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.6, hspace=0.2)
    plt.show()

# read images from an HDF5 file into a Numpy array
def read_images(h5file):
    images = []
    classes = []
    f = h5py.File(h5file, 'r')
    num_images = int(f.get('num_images').value)
    for i in range(num_images):
        img_key = 'img_' + str(i)
        cls_key = 'cls_' + str(i)
        img = np.array(f.get(img_key))
        cls = f.get(cls_key).value
        images.append(img)
        classes.append(cls)
    f.close()
    images = np.array(images)
    classes = np.array(classes)
    return images, classes

def check_images(model, img_files):
   imgs = load_and_scale_imgs(img_files)
   predictions = model.predict_classes(imgs)
   print(predictions)
 
def load_and_scale_images(img_files, shape):
    imgs = [rescale_image(img_file, shape) for img_file in img_files]
    return np.array(imgs)

# example: rescale_image("cat.jpg", (32,32,3))
def rescale_image(image_file, shape):
    im = imresize(imread(image_file, 0, 'RGB'), shape)
    return im

def classify_images(model, img_files, shape):
   imgs = load_and_scale_imgs(img_files, shape)
   predictions = model.predict_classes(imgs)
   return predictions

def save_numpy_array_as_image(arr, outfile):
    imsave(outfile, image_array)
    return outfile

def rgb2gray(rgb):
    img = np.dot(rgb[..., 0:3], [0.299, 0.587, 0.114])
    img = img.astype(int)
    return img

# get images from generator g and flatten them to
# be input for a flat layer
def flat_gen(g):
    for X_batch, Y_batch in g:
        X_batch_flat = X_batch.reshape(X_batch.shape[0], np.prod(X_batch.shape[1:]))
        yield X_batch_flat, Y_batch

def check_img_dups(h5file_list):
    dig = {}
    for h5file in h5file_list:
        f = h5py.File(h5file, 'r')
        num_images = int(f.get('num_images').value)
        pm = ProgressBar(num_images)
        for i in range(num_images):
            img_key = 'img_' + str(i)
            a = np.array(f.get(img_key))
            a.flags.writeable = False
            s = str(a.data)
            md = hashlib.sha512()
            md.update(s)
            key = md.hexdigest()
            if not key in dig:
                dig[key] = []
            dig[key].append(i)
            pm.advance()
        f.close()

    print("dict length =", len(dig))

    for key in dig:
        count = len(dig[key])
        if count >= 2:
            print("Dups: ", count, dig[key])


