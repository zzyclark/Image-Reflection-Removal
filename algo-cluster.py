import cv2
import numpy as np
import os

files = [f for f in os.listdir('.') if os.path.isfile(f)]
imgs = []

#read input
for f in files:
    if 'png' in f and 'background' not in f:
        imgs.append(cv2.imread(f))

#generate output
h, w = imgs[0].shape[:2]
img_out = np.zeros((h, w, 3), np.uint8)
num = len(imgs)

def main_cluster_mean(i, j):
    mat = np.empty([num, 3])
    for index in range(num):
        mat[index,:] = imgs[index][i, j]
    mat = np.float32(mat)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    ret,label,center=cv2.kmeans(mat,2,criteria,20,cv2.KMEANS_PP_CENTERS)
    label = label.flatten()
    label_sort = np.bincount(label)
    main_cluster = np.argmax(label_sort)
    return np.round(center[main_cluster,:])

for i in range(h):
    for j in range(w):
        img_out[i, j] = main_cluster_mean(i, j)

cv2.imwrite('backgroud_algo2.png', img_out)
