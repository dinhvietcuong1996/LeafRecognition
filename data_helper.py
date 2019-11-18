import os
import numpy as np


__features__ = ['image', 'vein', 'xyprojection', 'color', 'texture', 'fourier', 'shape']

__feature_shape__ = {
	'image': [300,300,3],	
	'vein': [300,300],	
	'xyprojection': [60,],
	'color': [36,],
	'texture': [18,],
	'fourier': [17,],
	'shape': [33,],
}

__feature_files__ = {
    'image': "features_data/images.npy",
	'vein': "features_data/vein.npy",
	'xyprojection': "features_data/xyprojection.npy",
	'color': "features_data/color.npy",
	'texture': "features_data/texture.npy",
	'fourier': "features_data/fourier.npy",
	'shape': "features_data/shape.npy",
}

__label_file__ = "features_data/labels.npy"

__model_file__ = "ENCODER-{}-l2rate{}-dropout{}-fold{}.h5"
__decoder_file__ = "DECODER-fold{}.pickle"

__normalizing_features__ = ['color', 'texture', 'fourier', 'shape', "combine"]

def load_features(feature):
    if type(feature) == list:
        return [np.load(__feature_files__[f]) for f in feature]
    return np.load(__feature_files__[feature])

def load_labels():
    return np.load(__label_file__)


def normalize_feature_data(feature, X_train, X_valid, X_test):
    """normalize data
    any feature in __normalizing_features__ is normalized, otherwise kept intact
    """
    if type(feature) == list:
        for i, f in enumerate(feature):
            
            if f in __normalizing_features__:
                stds = np.std(X_train[i], axis=0)
                stds[stds==0.0] = 1.0
                means = np.mean(X_train[i], axis=0)
                X_train[i] = (X_train[i]-means)/stds
                X_valid[i] = (X_valid[i]-means)/stds
                X_test[i] = (X_test[i]-means)/stds
    else:
        if feature in __normalizing_features__:
            stds = np.std(X_train, axis=0)
            stds[stds==0.0] = 1.0
            means = np.mean(X_train, axis=0)
            X_train = (X_train-means)/stds
            X_valid = (X_valid-means)/stds
            X_test = (X_test-means)/stds
            
    return X_train, X_valid, X_test

def split_train_test_valid(feature, Kfold, fold, X, y):
    """split dataset X, y into train, valid, test sets based on kfold and fold
    """
    fold = "Fold_" + str(fold)
    
    train_index = Kfold[fold] == "Train"
    valid_index = Kfold[fold] == "Valid"
    test_index = Kfold[fold] == "Test"

    if type(feature) == list:
        X_train = [x[train_index] for x in X]
        X_valid = [x[valid_index] for x in X]
        X_test = [x[test_index] for x in X]
    else:
        X_train = X[train_index]
        X_valid = X[valid_index]
        X_test = X[test_index]

    ## normalize handcrafted features
    X_train, X_valid, X_test = normalize_feature_data(feature, X_train, X_valid, X_test)

    y_train = y[train_index]
    y_valid = y[valid_index]
    y_test = y[test_index]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
