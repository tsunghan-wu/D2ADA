"""
Unorganized codes for edge detection information.
The detected edge is for ReDAL active learning.
Feel free to modify the code if you need it.

See more details in "Baseline Active Learning Methods" section
in the supplementary material.
"""
import os
import cv2
import numpy as np
from PIL import Image


def edge_detection(img_fname):
    img = cv2.imread(img_fname, 0)  # process gray-scale image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    # print(img_fname)
    # x = cv2.Sobel(img, cv2.CV_64F, dx=2, dy=0, ksize=5)
    # y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=2, ksize=5)
    # combine = np.array(x) + np.array(y)
    return canny


root_dir = "/project/project-dataset2/Cityscapes/leftImg8bit/train"
dst_dir = "/project/project-dataset2/edge_information/train"
os.makedirs(dst_dir, exist_ok=True)
for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in dirs:
        print('dir:', name)
        path = os.path.join(root, name)
        path = path.replace(root_dir, dst_dir)
        os.makedirs(path, exist_ok=True)


for root, dirs, files in os.walk(root_dir, topdown=False):
    for name in files:
        print('image:', name)
        path = os.path.join(root, name)
        edge_img = edge_detection(path)
        path = path.replace(root_dir, dst_dir)[:-4]
        path = path.replace('leftImg8bit', 'edge')
        im = Image.fromarray(edge_img.astype(np.uint8))
        im.save(path+'.png')
