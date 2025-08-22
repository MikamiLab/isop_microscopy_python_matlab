import cv2
import sys, os, datetime
import numpy as np
import pandas as pd
import re
import time
import matplotlib.pyplot as plt

now = datetime.datetime.now()

video_path = "img/bright img.avi"
path = "img"


img = cv2.imread("img/bright img.jpg", cv2.IMREAD_GRAYSCALE)  # queryiamge
print(img.shape)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("読み込み失敗")
    sys.exit()

# Features
sift = cv2.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)


filename = path+"/video_LocalOptical.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 115
# size = (480, 640)
# writer = cv2.VideoWriter(filename, fourcc, fps, size)
flag = True

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    if flag==True:
        writer = cv2.VideoWriter(filename, fourcc, fps, (frame.shape[1], frame.shape[0]))
        time.sleep(1)
        flag = False


    if ret:
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
        # img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
        # Homography
        if len(good_points) > 10:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            # Perspective transform
            h, w = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            # cv2.imshow("Homography", homography)
            # writer.write(homography)
        else:
            # cv2.imshow("Homography", grayframe)
            pass
        writer.write(grayframe)
        plt.imshow(grayframe)

    else:
        break

cap.release()
writer.release()

print("Finished")
