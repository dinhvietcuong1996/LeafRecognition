
import argparse
parser = argparse.ArgumentParser(description='Train encoders.')

parser.add_argument('expfile', type=str,
					help='Experiment file.')

from data_helper import load_features, load_labels, __model_file__, split_train_test_valid
from modeling import get_training_model, CheckpointCallback
import pandas as pd
import numpy as np
import os

def run_training(feature, kfold, fold, l2_rate, dropout):
	"""run a single training
	"""
	print("==========================")
	print("Training feature {} - l2_rate {} - dropout {} - fold {}".format(feature, l2_rate, dropout, fold))
	X = load_features(feature)
	y = load_labels()

	outdir = kfold[:-4] + "_models/"
	if not os.path.exists(outdir):
		os.mkdir(outdir)

	kfold = pd.read_csv(kfold)

	if feature in ['vein', 'image']:
		max_epochs = 200
	else:
		max_epochs = 1000000

	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_train_test_valid(feature, kfold, fold, X, y)

	model_path = __model_file__.format(feature, l2_rate, dropout, fold)
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

	print("Train_time {:.4f}, train_acc {:.4f}, val_acc {:.4f}, test_acc {:.4f}".format(checkpoint.training_time, train_acc, val_acc, test_acc))

	return val_acc, test_acc

def run_training_encoders(experiment_file):
	"""run training from a csv file,
    save trained models to 'kfold[:-4]+"_models/"' directory
	"""
	experiments = pd.read_csv(experiment_file)
	for exp_i in range(len(experiments)):
		kfold_file, folds, feature, l2_rate, dropout = experiments.iloc[exp_i][['kfold_file','fold','feature','l2_rate','dropout']]
		val_acc, test_acc = run_training(feature, kfold_file, folds, l2_rate, dropout)

		experiments.at[exp_i, 'val_acc'] = val_acc
		experiments.at[exp_i, 'test_acc'] = test_acc

		experiments.to_csv(experiment_file, index=False)
	print("Complete training ", experiment_file)

if __name__ == "__main__":
	args = parser.parse_args()
	run_training_encoders(args.expfile)




