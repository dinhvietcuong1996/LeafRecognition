
import argparse
parser = argparse.ArgumentParser(description='Train encoders.')

parser.add_argument('expfile', type=str,
					help='Experiment file.')


from modeling import get_training_model, CheckpointCallback
import pandas as pd
import numpy as np
import os

__model_file_pattern__ = "ENCODER-{}-l2rate{}-dropout{}-fold{}.h5"

__feature_data__ = {
	'image': "features_data/images.npy",
	'vein': "features_data/vein.npy",
	'xyprojection': "features_data/xyprojection.npy",
	'color': "features_data/color.npy",
	'texture': "features_data/texture.npy",
	'fourier': "features_data/fourier.npy",
	'shape': "features_data/shape.npy",
}
__label__ = 'features_data/labels.npy'

def load_feature_data(feature):
	return np.load(__feature_data__[feature])
def load_labels():
	return np.load(__label__)


__normalizing_features__ = ['color', 'texture', 'fourier', 'shape', "combine"]
def normalize_feature_data(feature, X_train, X_valid, X_test):
    
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


def run_training(feature, kfold, folds, l2_rate, dropout):
	print("==========================")
	print("Training feature {} - l2_rate {} - dropout {}".format(feature, l2_rate, dropout))
	X = load_feature_data(feature)
	y = load_labels()

	outdir = kfold[:-4] + "_models/"
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	kfold = pd.read_csv(kfold)
	folds = list(map(int, folds.split(',')))

	if feature in ['vein', 'image']:
		max_epochs = 200
	else:
		max_epochs = 1000000

	valid_accuracies, test_accuracies = [], []

	for fold in folds:
		(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(feature, kfold, fold, X, y)
		
		model_path = __model_file_pattern__.format(feature, l2_rate, dropout, fold)
		model_path = os.path.join(outdir, model_path)

		checkpoint = CheckpointCallback(verbose=False)
		model = get_training_model(feature, l2_rate=l2_rate, dropout=dropout)
		model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
				  verbose=0,
				  epochs=max_epochs,
				  batch_size=100,
				  callbacks=[checkpoint])
		model.save_weights(model_path)

		_, train_acc = model.evaluate(X_train, y_train, verbose=0)
		_, val_acc = model.evaluate(X_valid, y_valid, verbose=0)
		_, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
		print("Fold {}: train_time {:.4f}, train_acc {:.4f}, val_acc {:.4f}, test_acc {:.4f}".format(fold, checkpoint.training_time, train_acc, val_acc, test_acc))

		valid_accuracies.append(val_acc)
		test_accuracies.append(test_acc)

	val_acc_mean = np.mean(valid_accuracies)
	val_acc_std = np.std(valid_accuracies)
	test_acc_mean = np.mean(test_accuracies)
	test_acc_std = np.std(test_accuracies)
	print("End of 10-fold CV")
	print("Valid accuracy: {:.4f} +- {:.4f}".format(val_acc_mean, val_acc_std))
	print("Test accuracy: {:.4f} +- {:.4f}".format(test_acc_mean, test_acc_std))

	return val_acc_mean, val_acc_std, test_acc_mean, test_acc_std


def run_training_encoders(experiment_file):
	experiments = pd.read_csv(experiment_file)
	for exp_i in range(len(experiments)):
		kfold_file, folds, feature, l2_rate, dropout = experiments.iloc[exp_i][['kfold_file','fold','feature','l2_rate','dropout_rate']]
		val_acc_mean, val_acc_std, test_acc_mean, test_acc_std = run_training(feature, kfold_file, folds, l2_rate, dropout)

		experiments.at[exp_i, 'val_acc_mean'] = val_acc_mean
		experiments.at[exp_i, 'val_acc_std'] = val_acc_std
		experiments.at[exp_i, 'test_acc_mean'] = test_acc_mean
		experiments.at[exp_i, 'test_acc_std'] = test_acc_std

		experiments.to_csv(experiment_file, index=False)
	print("Complete training ", experiment_file)

if __name__ == "__main__":
	

	args = parser.parse_args()
	run_training_encoders(args.expfile)




