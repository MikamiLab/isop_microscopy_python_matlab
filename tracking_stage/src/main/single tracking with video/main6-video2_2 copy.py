# -*- coding: utf-8 -*-
from re import S
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtCore import Qt
import cv2
import time
import datetime
import os

import glob

import csv
import math
import image_operation22


CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

#動画のパスを入力
VIDEO_DATA = r"E:\0307_0308_c.elegans\2.0%_M9_lite-1\20240308162024\rec.avi"

now = datetime.datetime.now()
try:
    os.mkdir(os.getcwd() + "/module/VideoTracking_module/log/" + now.strftime("%Y%m%d%H%M%S"))
except FileExistsError:
    pass

log_name = now.strftime(os.getcwd()+ "/module/VideoTracking_module/log/" + "%Y%m%d%H%M%S/")+"log.csv"
log_file = open(log_name,"a")
log_file.write("frame,detect_x,detect_y,delta\n")


flag = False
detect_flag = False
reset_flag = True

centerx = 320
centery = 240
lightsheet_x = 104
lightsheet_y = 141
asobi = 10
radius = 10
move_scale = 1

fps = 50


def run():
    eatures = None
    frame = None
    gray_next = None
    gray_prev = None

    detection_x,detection_y = 351,133
    delta = 0
    frame_num = 0

    org = cv2.VideoCapture(VIDEO_DATA)
    time.sleep(1)
    end_flag, frame = org.read()

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_operation22.addFeature(detection_x,detection_y, gray_next)

    while end_flag:
        frame = cv2.resize(frame , (512,512))

        # if (flag == True) :
        #     RecThread().video.write(self.frame)

        radius = radius

        gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if(detect_flag == True):
            image_operation22.onMouse(detection_x,detection_y, radius, features, status, gray_next)
            detect_flag = False

        # if(reset_flag == True):
        #     self.features = None
        #     reset_flag = False
        #     detection_x = None
        #     detection_y = None
        #     previous_x = None


        if detection_x != None and detection_y != None:
            frame = cv2.drawMarker(frame, (detection_x, detection_y), (0, 0, 255))
        else:
            pass

        #self.frame = cv2.rectangle(self.frame, (lightsheet_x, lightsheet_y),  (lightsheet_x+292, lightsheet_y+191), (0, 255, 255))

        if features is not None:
            if  len(features) != 0:
                # オプティカルフローの計算(詳細：http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)
                features, status, err = cv2.calcOpticalFlowPyrLK(
                                                    gray_prev, # 前フレームのグレースケール画像
                                                    gray_next, # 現在のフレームのグレースケール画像
                                                    features, # 特徴点のベクトル prevPts
                                                    None,# 出力される特徴点のベクトル nextPts
                                                    winSize = (radius, radius), # 各ピラミッドレベルにおける探索窓のサイズ(Lucas-Kanade法参照)
                                                    maxLevel = 3, # 画像ピラミッドの最大レベル数(Lucas-Kanade法参照)
                                                    criteria = CRITERIA, # 反復探索アルゴリズムの停止基準
                                                    flags = 0) # 初期値設定フラグ(prevPts→nextPtsが適用される)

                image_operation22.refreshFeatures(features, status)

                for feature in features:
                    detection_x = int(feature[0][0])
                    detection_y = int(feature[0][1])

            else:
                print("再検出")
                # image_operation22.onMouse(detection_x,detection_y)

        else:
            pass

        #self.frame = cv2.circle(self.frame, (dx, dy), self.radius*2, (255, 0, 255))


        frame_num = frame_num +1
        log_file.write(str(frame_num)+ ","
                    + str(detection_x) + ","
                    + str(detection_y) + ","
                    + str(delta)+ "\n"
                    )

        # previous_x = self.detection_x
        # previous_y = self.detection_y

        gray_prev = gray_next

        end_flag, frame = org.read()

        if end_flag:
            gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            #RecThread(self).video.release()
            log_file.close()

    print(frame_num)



run()
