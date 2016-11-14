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

for i in range(h): 
    for j in range(w):
        r = g = b = 0
        
        for img in imgs:
            r = r + img[i, j][0]
            g = g + img[i, j][1]
            b = b + img[i, j][2]
#            print "original: ", img[i, j]

        img_out[i, j] = [r/num, g/num, b/num]        
#        print "background: ", img_out[i, j]

cv2.imwrite('backgroud_algo1.png', img_out)
