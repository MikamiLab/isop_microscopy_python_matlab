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
VIDEO_DATA = r"C:\Users\Hikaru Shishido\Desktop\2023-11-01\20231101-182842tdTomato\20231101181917\analysis\rec_matching.avi"

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

        self.pushButton_3.pressed.connect(self.track)
        pushButton_6.pressed.connect(self.trackoff)
        pushButton_8.pressed.connect(self.quit)

        #メイン画像
        self._item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._item)
        thread = CapThread(self)
        thread.mainFrameGrabbed.connect(self.main_image)
        thread.start()

    def valuechange(self):
        global radius
        radius = self.sp.value()

    @QtCore.pyqtSlot(QtGui.QImage)
    def main_image(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self._item.setPixmap(pixmap)

    def track(self):
        global flag
        flag = True
        self.pushButton_3.setStyleSheet("background-color: red")
        #RecThread(self).start()

    def trackoff(self):
        global flag
        flag = False
        self.pushButton_3.setStyleSheet("background-color: none")
        #RecThread(self).video.release()
        #RecThread(self).quit()
        print("log saved")

    def quit(self):
        files = glob.glob(os.getcwd() + "/save/*")
        for file in files:
            if(os.path.getsize(file) < 6000):
                #RecThread(self).video.release()
                #RecThread(self).quit()
                os.remove(file)
        app.quit()

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




# class RecThread(QtCore.QThread):
#         filename = now.strftime(os.getcwd()+ "/module/VideoTracking_module/log/" + "%Y%m%d%H%M%S/")+"rec.avi"
#         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#         video = cv2.VideoWriter(filename, fourcc, 38, (640,480))



class CapThread(QtCore.QThread):
    mainFrameGrabbed = QtCore.pyqtSignal(QtGui.QImage)

    def run(self):

        global detect_flag,reset_flag

        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None

        self.detection_x,self.detection_y = 276,230
        


        delta = 0

        self.frame_num = 0

        org = cv2.VideoCapture(VIDEO_DATA)
        time.sleep(1)
        end_flag, self.frame = org.read()

        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        image_operation22.addFeature(self,self.detection_x,self.detection_y)

        while end_flag:
            self.frame = cv2.resize(self.frame , (640,480))

            # if (flag == True) :
            #     RecThread().video.write(self.frame)

            self.radius = radius

            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            if(detect_flag == True):
                image_operation22.onMouse(self,self.detection_x,self.detection_y)
                detect_flag = False

            # if(reset_flag == True):
            #     self.features = None
            #     reset_flag = False
            #     detection_x = None
            #     detection_y = None
            #     previous_x = None


            if self.detection_x != None and self.detection_y != None:
                self.frame = cv2.drawMarker(self.frame, (self.detection_x, self.detection_y), (0, 0, 255))
            else:
                pass

            #self.frame = cv2.rectangle(self.frame, (lightsheet_x, lightsheet_y),  (lightsheet_x+292, lightsheet_y+191), (0, 255, 255))

            if self.features is not None:
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

                    image_operation22.refreshFeatures(self)

                    for feature in self.features:
                        self.detection_x = int(feature[0][0])
                        self.detection_y = int(feature[0][1])

                else:
                    print("再検出")
                    image_operation22.onMouse(self,self.detection_x,self.detection_y)

            else:
                pass

            #self.frame = cv2.circle(self.frame, (dx, dy), self.radius*2, (255, 0, 255))


            self.frame_num = self.frame_num +1
            log_file.write(str(self.frame_num)+ ","
                        + str(self.detection_x) + ","
                        + str(self.detection_y) + ","
                        + str(delta)+ "\n"
                        )

            # 表示
            mainImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # previous_x = self.detection_x
            # previous_y = self.detection_y

            self.gray_prev = self.gray_next

            end_flag, self.frame = org.read()

            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            else:
                #RecThread(self).video.release()
                log_file.close()

            mainQImage = QtGui.QImage(mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            self.mainFrameGrabbed.emit(mainQImage)
        print(self.frame_num)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())