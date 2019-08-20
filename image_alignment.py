# from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy
import scipy.io


MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15

image1_path = '/Users/DrorBlum/Desktop/ll_K/rgb/FLIR_00003.jpg'
image2_path = '/Users/DrorBlum/Desktop/TermalIm/matlab/lkk.mat'
output_path = '/Users/DrorBlum/Desktop/ll_K/output'

def alignImages(im, imReference):
    # Convert images to grayscale
    # im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
    # print(im1Gray)
    im1Gray = im
    im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Match features.
    # matcher = cv2.DescriptorMatcher_create()
    # matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False)



    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # print(matches,1)

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = imReference.shape
    im1Reg = cv2.warpPerspective(im, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = image1_path
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.COLOR_BGR2GRAY)
    imArr = np.array(imReference)
    imArr = (imArr - imArr.min())
    imArr = np.int_(imArr.astype(float) / imArr.max() * 255)
    imReference = imArr
    imReference = imReference.astype(np.uint8)

    # Read image to be aligned
    imFilename = image2_path
    print("Reading image to align : ", imFilename)
    # im = cv2.imread(imFilename,cv2.COLOR_BGR2GRAY)
    # im = im.astype(np.uint8)
    mat = scipy.io.loadmat(image2_path)
    im = mat

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)

