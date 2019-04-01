import os
from PIL import Image, ImageOps
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import time
import copy
import cv2

def getColor(num, toLAB = False, to_plt = False):
    Rname = 'FLIR_{:05d}.jpg'.format(num)
    path_rgb = os.path.expanduser('~/Documents/DB/FLIR/training/RGB/')
    path = os.path.join(path_rgb, Rname)
    Rim = Image.open(path).resize((640,512))
    if toLAB:
        arr = np.array(color.rgb2lab(Rim))
    else:
        arr = np.array(Rim)
    return arr, path


def getTerm(num, to255 = True, to_plt = False):
    Tname = "FLIR_{:05d}.tiff".format(num)
    path_thermal = os.path.expanduser('~/Documents/DB/FLIR/training/Data/')
    path = os.path.join(path_thermal, Tname)
    Tim = Image.open(path)
    # norm image
    imArr = np.array(Tim)
    if to255:
        imArr = (imArr - imArr.min())
        imArr = np.int_(imArr/imArr.max()*255)
    return imArr, path


def plotboth(num):
    Rarr = getRGB(num)
    Tarr = getTerm(num)
    
    plt.figure(2)
    plt.imshow(Rarr)
    plt.pause(0.0001)
    plt.figure(1)
    plt.imshow(Tarr)
    plt.pause(0.0001)


def connectPics(color, therm):
    l0 = color[:,:,0]
    


def present(num = 5, pause = 0.5):
    for i in range(num):
        try:
            plotboth(i+1)
            plt.pause(pause)
        except:
            a = 1 # do nothing

img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)