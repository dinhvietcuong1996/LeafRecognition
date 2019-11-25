# LeafRecoginition
This repository contains codes in paper A combination of Deep and Handcrafted Features for Leaf Recognition.

## Preperation


### Libraries
Install required python packages by pip3 

```console
pip3 install requirements.txt
```

### Dataset

Dataset is downloaded from https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download, all images are then extracted to folder "Leaves/".
We prepare labels sorted by filenames in features_data/labels.npy. You can confirm the correction of the labels in http://flavia.sourceforge.net/.

### Kfold_file

Kfold_file is a .csv file that follows the format of our Dataset_10FoldCV_indexed.csv, which must contain columns "Filename", "Fold_1", "Fold_2", ..., "Fold_10" . The order of leaves must be sorted by their filenames (numercial order). Values of "Fold" columns must be "Train", "Valid" or "Test".


## Train encoders

Training-encoder experiment file must be prepared beforehand. This file is to run training encoders with predefined l2 regularization and dropout rates. The file follows the format of train_encoders_example.csv and contains columns "kfold_file", "fold", "feature", "l2_rate", "dropout", "val_acc", "test_acc". 'feature' must be one of values 'image', 'vein', 'xyprojection', 'shape', 'color', 'texture', 'fourier'. 'fold' must in range 1 to 10.

To train encoders, run
```console
python3 run_training_encoders.py train_encoders_example.csv
```
This command creates a "{kfold_file}\_models/ folder, trains and saves encoders to the folder in .h5 format. Filenames of the model follows the format "ENCODER-{feature}-l2rate{}-dropout{}-fold{}.h5" (defined in data_helper.py).

## Select best encoders

Copy all trained encoders into the "{kfold_file}\_models/" folder and run
```console
python3 run_selecting_best_encoders.py Kfoldfile.csv
```
This command looks for all encoders in the directory "{kfold_file}\_models/", lists out their performances into "{Kfoldfile}\_encoders\_performances.csv" and select among them the best one each feature and each fold. The select encoder filenames are saved in "{Kfoldfile}\_best\_encoders.csv".

## Train decoders

Run 
```console
python3 run_training_decoders.py Kfoldfile.csv 
```

This command reads best encoders' filenames in "{Kfoldfile}\_best\_encoders.csv" and loads the corresponding saved .h5 files, trains decoders each fold to "{kfold_file}\_models/DECODER-fold{}.pickle". 

## Our results
Our experiments' results on Dataset_10FoldCV_indexed.csv are saved on "LEAF_v20". In summary, we reached the result
Valid accuracy: 0.9979 +- 0.0035
Test accuracy: 0.9953 +- 0.0037

