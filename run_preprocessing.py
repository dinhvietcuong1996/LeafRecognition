import numpy as np
from preprocessing_and_handfeature_extraction import get_features
from data_helper import __feature_files__
import os
import cv2

import sys
def progressBar(value, endvalue, bar_length=20):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}% {2}/{3} ".format(arrow + spaces, int(round(percent * 100)), value, endvalue))
    sys.stdout.flush()


def run_leaf_rotations(save_images=False):
    indir = "Leaves"
    if save_images:
        outdir = "Processed_Leaves/"
        outdir_vein = "Vein_Images/"
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    files = os.listdir(indir)
    files = sorted([f for f in files if f[-4:] == ".jpg"])
    ## processed images and extracted features are sorted according to files' order
    ## we require kfold and labels are sorted in this order too, otherwise they will be mismatched

    f_images = np.empty((1907,300,300,3), dtype=np.uint8)
    f_veins = np.empty((1907,300,300), dtype=np.uint8)
    f_colors = np.empty((1907,36), dtype=np.float32)
    f_shapes = np.empty((1907,38), dtype=np.float32)
    f_textures = np.empty((1907,13), dtype=np.float32)
    f_fouriers = np.empty((1907,40), dtype=np.float32)
    f_xyprojections = np.empty((1907, 60), dtype=np.float32)
    for i, filename in enumerate(files):
        filepath = os.path.join(indir, filename)
        img = cv2.imread(filepath)
        resized, vein, color, shape, texture, fourier, xyprojection = get_features(img)
        f_images[i] = resized
        f_veins[i] = vein
        f_colors[i] = color
        f_shapes[i] = shape
        f_textures[i] = texture
        f_fouriers[i] = fourier
        f_xyprojections[i] = xyprojection
        progressBar(i+1, 1907)

        if save_images:
            processed_path = os.path.join(outdir, filename)
            cv2.imwrite(processed_path, resized)
            vein_path = os.path.join(outdir_vein, filename)
            cv2.imwrite(vein_path, vein)

    np.save(__feature_files__['image'], f_images)
    np.save(__feature_files__['vein'], f_veins)
    np.save(__feature_files__['shape'], f_shapes)
    np.save(__feature_files__['color'], f_colors)
    np.save(__feature_files__['texture'], f_textures)
    np.save(__feature_files__['fourier'], f_fouriers)
    np.save(__feature_files__['xyprojection'], f_xyprojections)
    print()
if __name__ == "__main__":
    from time import time
    tt = time()
    run_leaf_rotations(save_images=False)
    print("Processing time: ", time() - tt)