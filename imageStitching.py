import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from stitch import stitch
import argparse

img=[]
for i in range(1,len(sys.argv)):
    #img.append(sys.argv[i])
    img1 = cv2.imread(sys.argv[i])
    img.append(img1)

panoroma=img[0]
for i in range(1,len(img)):
    panoroma=stitch(panoroma,img[i])
    #cv2.imshow('p',panoroma)

cv2.imshow("final",panoroma)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
img1=cv2.imread('rightImage.png')
img2=cv2.imread('leftImage.png')
d=stitch(img1,img2)
cv2.imshow('tp',d)
'''