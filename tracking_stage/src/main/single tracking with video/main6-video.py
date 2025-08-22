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
import image_operation2


CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

#動画のパスを入力
VIDEO_DATA = r"E:\C.elegans\2023-09-21\20230921-195239tdTomato\20230921195108\rec-match-115fps.avi"
flag = False
flag_VH = True
flag_dia = False

radius = 10

move_scale = 1
detect_up = False
detect_down = False
detect_left = False
detect_right = False
detect_flag = False
reset_flag = True

now = datetime.datetime.now()
try:
    os.mkdir(os.getcwd() + "/module/VideoTracking_module/log/" + now.strftime("%Y%m%d%H%M%S"))
except FileExistsError:
    pass

log_name = now.strftime(os.getcwd()+ "/module/VideoTracking_module/log/" + "%Y%m%d%H%M%S/")+"log.csv"
log_file = open(log_name,"a")
log_file.write("frame,rec_x,rec_y,detect_x,detect_y,delta_x,delta_y,delta\n")

centerx = 320
centery = 240

top_speed = 1
acce_time = 1
asobi = 10

detection_x = 0
detection_y = 0

lightsheet_x = 144
lightsheet_y = 148

delta_x0 = 0
delta_y0 = 0

fps = 115

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Measurement")
        self.setFixedSize(1000, 600)

        scene = QtWidgets.QGraphicsScene()

        self.graphicsView = QtWidgets.QGraphicsView(scene)

        self.graphicsView.setMouseTracking(True)
        self.graphicsView.installEventFilter(self)
        self.graphicsView.setGeometry(20, 20,640,480 )
        self.setCentralWidget(self.graphicsView)
        #self.graphicsView.move(350, 200)

        self.l1 = QtWidgets.QLabel(self)
        self.l1.setText("探索範囲")
        self.l1.setGeometry(QtCore.QRect(270, 550, 75, 23))

        self.sp = QtWidgets.QSpinBox(self)
        self.sp.setMinimum(5)
        self.sp.setValue(radius)
        self.sp.valueChanged.connect(self.valuechange)
        self.sp.setGeometry(QtCore.QRect(350, 550, 50, 23))

        self.pushButton_3 = QtWidgets.QPushButton(self)
        self.pushButton_3.setGeometry(QtCore.QRect(50, 290, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("Track On")

        pushButton_6 = QtWidgets.QPushButton(self)
        pushButton_6.setGeometry(QtCore.QRect(50, 350, 75, 23))
        pushButton_6.setObjectName("pushButton_6")
        pushButton_6.setText("Track Off")

        pushButton_8 = QtWidgets.QPushButton(self)
        pushButton_8.setGeometry(QtCore.QRect(50, 450, 75, 23))
        pushButton_8.setObjectName("pushButton_8")
        pushButton_8.setText("Quit")

        pushButton_12 = QtWidgets.QPushButton(self)
        pushButton_12.setGeometry(QtCore.QRect(870, 90, 75, 23))
        pushButton_12.setObjectName("pushButton_12")
        pushButton_12.setText("up")
        pushButton_12.setShortcut(Qt.Key_Up)

        pushButton_13 = QtWidgets.QPushButton(self)
        pushButton_13.setGeometry(QtCore.QRect(870, 190, 75, 23))
        pushButton_13.setObjectName("pushButton_13")
        pushButton_13.setText("down")
        pushButton_13.setShortcut(Qt.Key_Down)

        pushButton_14 = QtWidgets.QPushButton(self)
        pushButton_14.setGeometry(QtCore.QRect(825, 140, 75, 23))
        pushButton_14.setObjectName("pushButton_14")
        pushButton_14.setText("left")
        pushButton_14.setShortcut(Qt.Key_Left)

        pushButton_15 = QtWidgets.QPushButton(self)
        pushButton_15.setGeometry(QtCore.QRect(915, 140, 75, 23))
        pushButton_15.setObjectName("pushButton_15")
        pushButton_15.setText("right")
        pushButton_15.setShortcut(Qt.Key_Right)

        self.pushButton_3.pressed.connect(self.track)
        pushButton_6.pressed.connect(self.trackoff)
        pushButton_8.pressed.connect(self.quit)
        pushButton_12.pressed.connect(self.detect_up)
        pushButton_13.pressed.connect(self.detect_down)
        pushButton_14.pressed.connect(self.detect_left)
        pushButton_15.pressed.connect(self.detect_right)

        #メイン画像
        self._item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._item)
        thread = CapThread(self)
        thread.mainFrameGrabbed.connect(self.main_image)
        thread.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def main_image(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self._item.setPixmap(pixmap)

    def track(self):
        global flag
        flag = True
        self.pushButton_3.setStyleSheet("background-color: red")
        RecThread(self).start()

    def trackoff(self):
        global flag,log_file,log_name
        flag = False
        self.pushButton_3.setStyleSheet("background-color: none")
        RecThread(self).video.release()
        RecThread(self).quit()
        print("log saved")
        log_file.close()

    def quit(self):
        files = glob.glob(os.getcwd() + "/save/*")
        for file in files:
            if(os.path.getsize(file) < 6000):
                RecThread(self).video.release()
                RecThread(self).quit()
                os.remove(file)
        app.quit()

    def set_zero(self):
        CapThread.rec_x = 0
        CapThread.rec_y = 0

    def eventFilter(self, source, event):
        global detect_flag,reset_flag
        global click_x,click_y

        if event.type() == QtCore.QEvent.MouseButtonPress and source is self.graphicsView and self.graphicsView.itemAt(event.pos()):
            if event.button() == QtCore.Qt.LeftButton:
                detect_flag = True
                pos = self.graphicsView.mapToScene(event.pos())
                click_x = pos.x()
                click_y = pos.y()
            if event.button() == QtCore.Qt.RightButton:
                reset_flag = True
        return QWidget.eventFilter(self, source, event)

    def valuechange(self):
        global radius
        radius = self.sp.value()

    def detect_up(self):
        global detect_up
        detect_up = True

    def detect_down(self):
        global detect_down
        detect_down = True

    def detect_left(self):
        global detect_left
        detect_left = True

    def detect_right(self):
        global detect_right
        detect_right = True

class RecThread(QtCore.QThread):
        filename = now.strftime(os.getcwd()+ "/module/VideoTracking_module/log/" + "%Y%m%d%H%M%S/")+"rec.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(filename, fourcc, 38, (640,480))

class CapThread(QtCore.QThread):
    mainFrameGrabbed = QtCore.pyqtSignal(QtGui.QImage)

    rec_x = 0
    rec_y = 0

    def run(self):

        global detect_flag,reset_flag,detect_up,detect_down,detect_left,detect_right

        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None
        detection_x = 0
        detection_y = 0
        delta = 0

        self.frame_num = 0

        org = cv2.VideoCapture(VIDEO_DATA)
        time.sleep(1)
        end_flag, self.frame = org.read()

        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        while end_flag:
            self.frame = cv2.resize(self.frame , (640,480))

            if (flag == True) :
                RecThread().video.write(self.frame)

            self.radius = radius

            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            if(detect_flag == True):
                image_operation2.onMouse(self,click_x,click_y)
                detect_flag = False
                self.rec_x = 0
                self.rec_y = 0

            if(reset_flag == True):
                self.features = None
                reset_flag = False
                detection_x = None
                detection_y = None
                previous_x = None
                previous_y = None
                delta_x = 0
                delta_y = 0

            #print(detect_up)

            if(detect_up == True):
                for feature in self.features:
                        feature[0][1] -= 3
                detect_up = False
            elif(detect_down == True):
                for feature in self.features:
                        feature[0][1] += 3
                detect_down = False
            elif(detect_left == True):
                for feature in self.features:
                        feature[0][0] -= 3
                detect_left = False
            elif(detect_right == True):
                for feature in self.features:
                        feature[0][0] += 3
                detect_right = False
            else:
                pass

            if detection_x != None and detection_y != None:
                self.frame = cv2.drawMarker(self.frame, (detection_x, detection_y), (0, 0, 255))
            else:
                pass

            self.frame = cv2.rectangle(self.frame, (lightsheet_x, lightsheet_y),  (lightsheet_x+292, lightsheet_y+191), (0, 255, 255))

            if self.features is not None or (detection_x is not None and detection_y is not None):
                if  len(self.features) != 0:
                    # オプティカルフローの計算(詳細：http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)
                    self.features, self.status, err = cv2.calcOpticalFlowPyrLK(
                                                        self.gray_prev, # 前フレームのグレースケール画像
                                                        self.gray_next, # 現在のフレームのグレースケール画像
                                                        self.features, # 特徴点のベクトル prevPts
                                                        None,# 出力される特徴点のベクトル nextPts
                                                        winSize = (self.radius, self.radius), # 各ピラミッドレベルにおける探索窓のサイズ(Lucas-Kanade法参照)
                                                        maxLevel = 3, # 画像ピラミッドの最大レベル数(Lucas-Kanade法参照)
                                                        criteria = CRITERIA, # 反復探索アルゴリズムの停止基準
                                                        flags = 0) # 初期値設定フラグ(prevPts→nextPtsが適用される)

                    image_operation2.refreshFeatures(self)

                    for feature in self.features:
                        detection_x = int(feature[0][0])
                        detection_y = int(feature[0][1])

                else:
                    print("再検出")
                    image_operation2.onMouse(self,detection_x,detection_y)


                if detection_x == None or previous_x == None:
                    delta_x = 0
                else:
                    delta_x = detection_x - previous_x

                if detection_y == None or previous_y == None:
                    delta_y = 0
                else:
                    delta_y = detection_y - previous_y

                delta = math.sqrt(pow(delta_x,2)+pow(delta_y,2))

                self.frame = cv2.circle(self.frame, (detection_x, detection_y), self.radius*2, (255, 0, 255))
                self.frame = cv2.circle(self.frame, (detection_x, detection_y), asobi*2, (0, 255, 255))


                if (previous_y is not None) and (previous_x is not None):
                    self.frame = cv2.circle(self.frame, (previous_x, previous_y), self.radius*2, (0, 0, 255))
                    self.frame = cv2.circle(self.frame, (previous_x, previous_y), asobi*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (previous_x, previous_y), (0, 255, 0))

            else:
                pass


            self.frame_num = self.frame_num +1
            log_file.write(str(self.frame_num)+ ","
                        + str(self.rec_x) + "," + str(self.rec_y) + ","
                        + str(detection_x) + "," + str(detection_y) + ","
                        + str(delta_x) + "," + str(delta_y) + ","
                        + str(delta)
                        + "\n")

            # 表示
            mainImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            previous_x = detection_x
            previous_y = detection_y
            self.gray_prev = self.gray_next

            end_flag, self.frame = org.read()

            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            else:
                RecThread(self).video.release()
                log_file.close()

            mainQImage = QtGui.QImage(mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            self.mainFrameGrabbed.emit(mainQImage)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())