import os
import cv2
import numpy as np
from data_helper import __feature_files__
from preprocessing_and_handfeature_extraction import rotate_img


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
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    files = os.listdir(indir)
    files = sorted([f for f in files if f[-4:] == ".jpg"])
    resized_images = np.empty((1907,300,300,3), dtype=np.uint8)
    for i, filename in enumerate(files):
        # filename = "1124.jpg"
        # if int(filename[:-4]) < 2231 or int(filename[:-4]) > 2290:
        #     continue
        filepath = os.path.join(indir, filename)

        img = cv2.imread(filepath)
        rotated_img = rotate_img(img)
        resized = cv2.resize(rotated_img, (300,300))
        resized_images[i] = resized
        ## save file

        if save_images:
            outpath = os.path.join(outdir, filename)
            cv2.imwrite(outpath, resized)

        progressBar(i, 1907)
        # break
    np.save(__feature_files__['image'], resized_images)

if __name__ == "__main__":
    run_leaf_rotations(save_images=False)
        
    