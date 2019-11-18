
import argparse
parser = argparse.ArgumentParser(description='Select best encoders.')

parser.add_argument('kfoldfile', type=str,
					help='10fold cross validation file.')

import os
from modeling import get_training_model
import pandas as pd
import numpy as np
import re
from data_helper import __features__, load_features, load_labels, split_train_test_valid



def get_encoders_performances(kfold_file):
    """get all encoders' performances from kfold's model directory, i. e., kfold[:-4] + "_models/"
    save to file 'kfold[:-4] + "_encoders_performances.csv"'
    """
    modeldir = kfold_file[:-4] + "_models/"
    model_files = os.listdir(modeldir)
    model_files = [os.path.join(modeldir, f) for f in model_files]
    encoder_performances = pd.DataFrame(columns=["kfold_file", "feature", "fold", "model_file", "val_acc", "test_acc"])
    i = 0
    kfold = pd.read_csv(kfold_file)
    for feature in __features__:
        X = load_features(feature)
        y = load_labels()
        model = get_training_model(feature)
        for fold in range(1,11):
            _, (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(feature, kfold, fold, X, y)
            for model_file in model_files:
                if re.search("ENCODER-{}-.*-fold{}\.".format(feature, fold), model_file):
                    model.load_weights(model_file)
                    _, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
                    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
                    encoder_performances.loc[i] = [kfold_file, feature, fold, model_file, val_acc, test_acc]
                    i += 1
                    print("File {} - val_acc {} - test_acc {}".format(model_file, val_acc, test_acc))
    encoder_performances.to_csv(kfold_file[:-4] + "_encoders_performances.csv", index=False)
    return encoder_performances

def select_best_encoders(kfold_file):
    performances = pd.read_csv(kfold_file[:-4] + "_encoders_performances.csv")
    best_encoders = pd.DataFrame(columns=performances.columns)
    i = 0
    for fold in range(1,11):
        for feature in __features__:
            encoders = performances[np.logical_and(performances['feature'] == feature, performances['fold'] == fold)]
            best_encoder = encoders.loc[encoders['val_acc'].idxmax()]
            best_encoders.loc[i] = best_encoder
            i += 1
    best_encoders.to_csv(kfold_file[:-4] + "_best_encoders.csv", index = False)
    return best_encoders

if __name__ == "__main__":
    args = parser.parse_args()
    encoders_performances_file = args.kfoldfile[:-4] + "_encoders_performances.csv"
    if not os.path.exists(encoders_performances_file):
        get_encoders_performances(args.kfoldfile)
    best_encoders_file = args.kfoldfile[:-4] + "_best_encoders.csv"
    if not os.path.exists(best_encoders_file):
        select_best_encoders(args.kfoldfile)