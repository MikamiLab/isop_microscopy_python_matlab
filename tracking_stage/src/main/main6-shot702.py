# -*- coding: utf-8 -*-
#from fileinput import filename
from re import S
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtCore import Qt
import cv2
import serial
#import serial.tools.list_ports
import time
import datetime
import os
import math
import glob
import csv
import common.image_operation2 as image_operation2
import pandas
import sys
from matplotlib import pyplot as plt
import numpy as np

import common.basler as basler
from pypylon import pylon


CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

flag = False
flag_detect = False
flag_VH = True
flag_dia = False
track_flag = False
stage_flag = True
save_flag = False

detect_up = False
detect_down = False
detect_left = False
detect_right = False

detect_flag = False
reset_flag = True
writer = None #オリジナル画像の保存先動画。
writer_detect = None #検出画像の保存先動画

size = (640, 480)#480 

centerx = 320  
centery = 240
low_speed = 10000  #1000~8000
top_speed = 500000
acce_time = 200   #30~60
pulse = 5000     #分割数250,2040x1530,15000 / 250,1020x765,5000
m= None

acce = (top_speed-low_speed)/(acce_time/1000)

kernel = 1 #7
sigma = 1 #5

pu_m = 0.01590033
pi_m = 0.616291862

fps = 115 # 2022/07/22 average 0.0422 s, 2022/07/27 average 0.0120 s, 

radius = 16
asobi = 2
move_scale = 38.85873036

mark = 5

delta_x0 = 0
delta_y0 = 0

now = datetime.datetime.now()
try:
    os.mkdir(os.getcwd()+ "/log/"+now.strftime("%Y%m%d%H%M%S"))
except FileExistsError:
    pass
#logの保存先を作成
log_name = now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"log.csv"
log_file = open(log_name,"a")
log_file.write("frame,time,x(pixel),y(pixel),radius(pixel),asobi(pixel),Δt(s),Δd(μm),track_v(μm/s),err(μm),sum_x(pixel),sum_y(pixel)\n")
log_path = open(r"F:\tracking stage\path.txt","w")
log_path.write(log_name)
log_path.close()
#parameter情報の保存
parameter_name = now.strftime(os.getcwd()+"/log/"+"%Y%m%d%H%M%S/")+"parameter.txt"
parameter_file = open(parameter_name,"a")
parameter_file.write("fps:\t"+str(fps)+"\npixsel:\t"+str(size)+"\nlow_speed:\t"+str(low_speed)+"\ntop_speed:\t"+str(top_speed)+"\nacce_time:\t"+str(acce_time)+"\n")
parameter_file.close()
#write(オリジナル画像)の保存先を作成
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
filename = "rec.avi"
writer = cv2.VideoWriter(os.getcwd()+"/log/"+now.strftime("%Y%m%d%H%M%S")+"/"+filename, fourcc, fps, size)
                

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Measurement")
        self.setFixedSize(1100, 650)
        
        scene = QtWidgets.QGraphicsScene()
        
        self.graphicsView = QtWidgets.QGraphicsView(scene)
        #self.graphicsView.setGeometry(QtCore.QRect(80, 65, 500, 300))   #機能していない（画像表示の位置が設定できない）
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.installEventFilter(self)
        self.setCentralWidget(self.graphicsView)

        self.l = QtWidgets.QLabel(self)
        self.l.setText("X")
        self.l.setGeometry(QtCore.QRect(40, 600, 20, 23))

        self.sp = QtWidgets.QSpinBox(self)
        self.sp.setMinimum(-320)
        self.sp.setMaximum(320)
        self.sp.setValue(centerx-320)
        self.sp.setSingleStep(1)
        self.sp.valueChanged.connect(self.valuechange)
        self.sp.setGeometry(QtCore.QRect(70, 600, 50, 23))

        self.l2 = QtWidgets.QLabel(self)
        self.l2.setText("Y")
        self.l2.setGeometry(QtCore.QRect(140, 600, 20, 23))

        self.sp2 = QtWidgets.QSpinBox(self)
        self.sp2.setMinimum(-240)
        self.sp2.setMaximum(240)
        self.sp2.setValue(centery-240)
        self.sp2.setSingleStep(1)
        self.sp2.valueChanged.connect(self.valuechange2)
        self.sp2.setGeometry(QtCore.QRect(160, 600, 50, 23))
        
        self.l3 = QtWidgets.QLabel(self)
        self.l3.setText("探索範囲")
        self.l3.setGeometry(QtCore.QRect(235, 600, 100, 23))

        self.sp3 = QtWidgets.QSpinBox(self)
        self.sp3.setMinimum(5)
        self.sp3.setValue(radius)
        self.sp3.valueChanged.connect(self.valuechange3)
        self.sp3.setGeometry(QtCore.QRect(325, 600, 55, 23))

        self.l4 = QtWidgets.QLabel(self)
        self.l4.setText("遊び")
        self.l4.setGeometry(QtCore.QRect(400, 600, 75, 23))

        self.sp4 = QtWidgets.QSpinBox(self)
        self.sp4.setMinimum(1)
        self.sp4.setMaximum(100)
        self.sp4.setValue(asobi)
        self.sp4.setSingleStep(1)
        self.sp4.valueChanged.connect(self.valuechange4)
        self.sp4.setGeometry(QtCore.QRect(450, 600, 55, 23))

        self.l5 = QtWidgets.QLabel(self)
        self.l5.setText("移動倍率")
        self.l5.setGeometry(QtCore.QRect(530, 600, 100, 23))

        self.sp5 = QtWidgets.QDoubleSpinBox(self)
        self.sp5.setMinimum(0)
        self.sp5.setSingleStep(1)
        self.sp5.setValue(move_scale)
        self.sp5.valueChanged.connect(self.valuechange5)
        self.sp5.setGeometry(QtCore.QRect(620, 600, 65, 23))

        self.l6 = QtWidgets.QLabel(self)
        self.l6.setText("最低速pps")
        self.l6.setGeometry(QtCore.QRect(710, 585, 100, 23))

        self.sp6 = QtWidgets.QSpinBox(self)
        self.sp6.setMinimum(1)
        self.sp6.setMaximum(500000)
        self.sp6.setValue(low_speed)
        self.sp6.setSingleStep(5)
        self.sp6.valueChanged.connect(self.valuechange6)
        self.sp6.setGeometry(QtCore.QRect(800, 585, 120, 23))

        self.l7 = QtWidgets.QLabel(self)
        self.l7.setText("最高速pps")
        self.l7.setGeometry(QtCore.QRect(710, 615, 100, 23))

        self.sp7 = QtWidgets.QSpinBox(self)
        self.sp7.setMinimum(1)
        self.sp7.setMaximum(500000)
        self.sp7.setValue(top_speed)
        self.sp7.setSingleStep(5)
        self.sp7.valueChanged.connect(self.valuechange7)
        self.sp7.setGeometry(QtCore.QRect(800, 615, 120, 23))

        self.l8 = QtWidgets.QLabel(self)
        self.l8.setText("加速mS")
        self.l8.setGeometry(QtCore.QRect(940, 600, 80, 23))

        self.sp8 = QtWidgets.QSpinBox(self)
        self.sp8.setMinimum(1)
        self.sp8.setMaximum(1000)
        self.sp8.setValue(acce_time)
        self.sp8.setSingleStep(1)
        self.sp8.valueChanged.connect(self.valuechange8)
        self.sp8.setGeometry(QtCore.QRect(1010, 600, 50, 23))

        self.l9 = QtWidgets.QLabel(self)
        self.l9.setText("Marker")
        self.l9.setGeometry(QtCore.QRect(970, 300, 50, 23))

        self.sp9 = QtWidgets.QSpinBox(self)
        self.sp9.setMinimum(1)
        self.sp9.setMaximum(10)
        self.sp9.setValue(mark)
        self.sp9.setSingleStep(1)
        self.sp9.valueChanged.connect(self.valuechange9)
        self.sp9.setGeometry(QtCore.QRect(1010, 300, 50, 23))


        pushButton = QtWidgets.QPushButton(self)
        pushButton.setAutoRepeat(True)
        pushButton.setGeometry(QtCore.QRect(75, 150, 75, 23))
        pushButton.setObjectName("pushButton")
        pushButton.setText("Up")
        pushButton.setShortcut(Qt.Key_W)

        pushButton_2 = QtWidgets.QPushButton(self)
        pushButton_2.setAutoRepeat(True)
        pushButton_2.setGeometry(QtCore.QRect(75, 250, 75, 23))
        pushButton_2.setObjectName("pushButton_2")
        pushButton_2.setText("Down")
        pushButton_2.setShortcut(Qt.Key_S)

        pushButton_3 = QtWidgets.QPushButton(self)
        pushButton_3.setAutoRepeat(True)
        pushButton_3.setGeometry(QtCore.QRect(30, 200, 75, 23))
        pushButton_3.setObjectName("pushButton_3")
        pushButton_3.setText("Left")
        pushButton_3.setShortcut(Qt.Key_A)

        pushButton_4 = QtWidgets.QPushButton(self)
        pushButton_4.setAutoRepeat(True)
        pushButton_4.setGeometry(QtCore.QRect(120, 200, 75, 23))
        pushButton_4.setObjectName("pushButton_4")
        pushButton_4.setText("Right")
        pushButton_4.setShortcut(Qt.Key_D)

        self.pushButton_5 = QtWidgets.QPushButton(self)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 380, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setText("Track")
        self.pushButton_5.setShortcut(Qt.Key_I)

        pushButton_6 = QtWidgets.QPushButton(self)
        pushButton_6.setGeometry(QtCore.QRect(120, 380, 75, 23))
        pushButton_6.setObjectName("pushButton_6")
        pushButton_6.setText("Off")
        pushButton_6.setShortcut(Qt.Key_O)
        
        pushButton_7 = QtWidgets.QPushButton(self)
        pushButton_7.setGeometry(QtCore.QRect(30, 410, 75, 23))
        pushButton_7.setObjectName("pushButton_7")
        pushButton_7.setText("Quit") 
        pushButton_7.setShortcut(Qt.Key_Q)

        self.pushButton_8 = QtWidgets.QPushButton(self)
        self.pushButton_8.setGeometry(QtCore.QRect(30, 500, 75, 23))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.setText("Manual")
        self.pushButton_8.setShortcut(Qt.Key_M)

        self.pushButton_9 = QtWidgets.QPushButton(self)
        self.pushButton_9.setGeometry(QtCore.QRect(120, 500, 75, 23))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.setText("Off")
        self.pushButton_9.setShortcut(Qt.Key_N)

        """
        self.pushButton_10 = QtWidgets.QPushButton(self)
        self.pushButton_10.setGeometry(QtCore.QRect(920, 350, 75, 23))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.setText("Rec")

        pushButton_11 = QtWidgets.QPushButton(self)
        pushButton_11.setGeometry(QtCore.QRect(1010, 350, 75, 23))
        pushButton_11.setObjectName("pushButton_11")
        pushButton_11.setText("Off")
        """

        pushButton_12 = QtWidgets.QPushButton(self)
        pushButton_12.setGeometry(QtCore.QRect(945, 150, 75, 23))
        pushButton_12.setObjectName("pushButton_12")
        pushButton_12.setText("up")
        pushButton_12.setShortcut(Qt.Key_Up)

        pushButton_13 = QtWidgets.QPushButton(self)
        pushButton_13.setGeometry(QtCore.QRect(945, 250, 75, 23))
        pushButton_13.setObjectName("pushButton_13")
        pushButton_13.setText("down")
        pushButton_13.setShortcut(Qt.Key_Down)

        pushButton_14 = QtWidgets.QPushButton(self)
        pushButton_14.setGeometry(QtCore.QRect(900, 200, 75, 23))
        pushButton_14.setObjectName("pushButton_14")
        pushButton_14.setText("left")
        pushButton_14.setShortcut(Qt.Key_Left)

        pushButton_15 = QtWidgets.QPushButton(self)
        pushButton_15.setGeometry(QtCore.QRect(990, 200, 75, 23))
        pushButton_15.setObjectName("pushButton_15")
        pushButton_15.setText("right")
        pushButton_15.setShortcut(Qt.Key_Right)

        pushButton.pressed.connect(self.up)
        pushButton_2.pressed.connect(self.down)
        pushButton_3.pressed.connect(self.left)
        pushButton_4.pressed.connect(self.right)
        self.pushButton_5.pressed.connect(self.track)
        pushButton_6.pressed.connect(self.trackoff)
        pushButton_7.pressed.connect(self.quit)
        self.pushButton_8.pressed.connect(self.manual)
        self.pushButton_9.pressed.connect(self.manualoff)
        #self.pushButton_10.pressed.connect(self.Rec)
        #pushButton_11.pressed.connect(self.Recoff)
        pushButton_12.pressed.connect(self.detect_up)
        pushButton_13.pressed.connect(self.detect_down)
        pushButton_14.pressed.connect(self.detect_left)
        pushButton_15.pressed.connect(self.detect_right)
        

        global ser
        try:
            ser = serial.Serial(port=list(serial.tools.list_ports.grep("0403:6001"))[0][0], baudrate=38400, timeout=0.5)
        except IndexError:
            QMessageBox.information(self,"Message","No Motor Connection")
            sys.exit()

        ser.write(("D:WS" + str(low_speed) + "F" + str(top_speed)+ "R" + str(acce_time) + "S" + str(low_speed) + "F" + str(top_speed) + "R" + str(acce_time) + "\r\n").encode())
        ser.write(b"S:1250\r\n")
        ser.write(b"S:2250\r\n")
        ser.write(b"C:W1\r\n")
    

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


    def up(self): 
        ser.write(("M:2+P"+ str(pulse) + "\r\n").encode())
        ser.write(b"G\r\n")
    def down(self):
        ser.write(("M:2-P"+ str(pulse) + "\r\n").encode())
        ser.write(b"G\r\n")
    def right(self): 
        ser.write(("M:1+P"+ str(pulse) + "\r\n").encode())
        ser.write(b"G\r\n")
    def left(self):
        ser.write(("M:1-P"+ str(pulse) + "\r\n").encode())
        ser.write(b"G\r\n")


    def upperRight(self, delta_x0, delta_y0, m): 
        ser.write(("M:W+P"+ str(math.floor(move_scale*abs(delta_x0)*m)) + "+P"+ str(math.floor(move_scale*abs(delta_y0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def upperLeft(self, delta_x0, delta_y0, m):
        ser.write(("M:W-P"+ str(math.floor(move_scale*abs(delta_x0)*m)) + "+P"+ str(math.floor(move_scale*abs(delta_y0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def lowerRight(self, delta_x0, delta_y0, m): 
        ser.write(("M:W+P"+ str(math.floor(move_scale*abs(delta_x0)*m)) + "-P"+ str(math.floor(move_scale*abs(delta_y0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def lowerLeft(self, delta_x0, delta_y0, m):
        ser.write(("M:W-P"+ str(math.floor(move_scale*abs(delta_x0)*m)) + "-P"+ str(math.floor(move_scale*abs(delta_y0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def Up(self, delta_y0, m): 
        ser.write(("M:2+P"+ str(math.floor(move_scale*abs(delta_y0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def Down(self, delta_y0, m):
        ser.write(("M:2-P"+ str(math.floor(move_scale*abs(delta_y0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def Right(self, delta_x0, m): 
        ser.write(("M:1+P"+ str(math.floor(move_scale*abs(delta_x0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    def Left(self, delta_x0, m):
        ser.write(("M:1-P"+ str(math.floor(move_scale*abs(delta_x0)*m)) + "\r\n").encode())
        ser.write(b"G\r\n")
    

    def track(self):
        try:
            os.mkdir(os.getcwd()+ "/save")
        except FileExistsError:
            pass

        global flag,writer,writer_detect,track_flag
        flag = True
        track_flag = True
        
        self.sp6.setEnabled(False)
        self.sp7.setEnabled(False)
        self.sp8.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.pushButton_5.setStyleSheet("background-color: red")
         
        
    def trackoff(self):

        global flag,log_file,log_name,writer,writer_detect,track_flag,save_flag
        flag = False
        track_flag = False
        save_flag = True
        
        print("log saved")

        self.sp6.setEnabled(True)
        self.sp7.setEnabled(True)
        self.sp8.setEnabled(True)
        self.pushButton_8.setEnabled(True)
        self.pushButton_9.setEnabled(True)
        self.pushButton_5.setStyleSheet("background-color: none")

      
    def quit(self):
        log_file.close()
        writer.release()
        """files = glob.glob(os.getcwd() + "/save/*")
        for file in files:
            if(os.path.getsize(file) < 6000):#6000とは何か
                os.remove(file)"""

        if(flag == True):
            self.trackoff()
        else:
            pass
        app.quit()


    def manual(self):
        self.pushButton_8.setStyleSheet("background-color: red")
        ser.write(b"C:W0\r\n")
    def manualoff(self):
        self.pushButton_8.setStyleSheet("background-color: none")
        ser.write(b"C:W1\r\n")
    
    def eventFilter(self, source, event):
        global detect_flag,reset_flag
        global click_x,click_y
        
        if event.type() == QtCore.QEvent.MouseButtonPress and source is self.graphicsView and self.graphicsView.itemAt(event.pos()):
            if event.button() == Qt.LeftButton:                
                detect_flag = True
                pos = self.graphicsView.mapToScene(event.pos())
                click_x = pos.x()
                click_y = pos.y()
            if event.button() == Qt.RightButton: 
                reset_flag = True
        return QWidget.eventFilter(self, source, event)


    def valuechange(self):
        global centerx
        centerx = self.sp.value() + 320
    def valuechange2(self):
        global centery
        centery = self.sp2.value() + 240
    def valuechange3(self):
        global radius
        radius = self.sp3.value()
    def valuechange4(self):
        global asobi
        asobi = self.sp4.value()
    def valuechange5(self):
        global move_scale
        move_scale = int(self.sp5.value()) 
    def valuechange6(self):
        self.low_speed = self.sp6.value()
        ser.write(("D:WS" + str(low_speed) + "F" + str(top_speed)+ "R" + str(acce_time) + "S" + str(low_speed) + "F" + str(top_speed) + "R" + str(acce_time) + "\r\n").encode())
    def valuechange7(self):
        self.top_speed = self.sp7.value()
        ser.write(("D:WS" + str(low_speed) + "F" + str(top_speed)+ "R" + str(acce_time) + "S" + str(low_speed) + "F" + str(top_speed) + "R" + str(acce_time) + "\r\n").encode())
    def valuechange8(self):
        self.acce_time = self.sp8.value()
        ser.write(("D:WS" + str(low_speed) + "F" + str(top_speed)+ "R" + str(acce_time) + "S" + str(low_speed) + "F" + str(top_speed) + "R" + str(acce_time) + "\r\n").encode())
    def valuechange9(self):
        global mark
        mark = self.sp9.value()
    
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
    
    
class CapThread(QtCore.QThread):
    mainFrameGrabbed = QtCore.pyqtSignal(QtGui.QImage)

    rec_x = 0
    rec_y = 0
    srec_x = 0
    srec_y = 0

    time10 = 0
    time11 = 0
        
    def run(self):

        

        #track開始時に保存先ディレクトリをつくる
        try:
            os.mkdir(os.getcwd()+ "/log")
        except FileExistsError:
            None

        global detect_flag,reset_flag,writer,writer_detect,detect_flag,detect_up,detect_down,detect_left,detect_right,stage_flag
        global flag,log_file,log_name,track_flag,save_flag,radius,asobi
        #previous_approxCurves = None #消して不具合がないか確認する
        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None

        self.frame_num = 0
    
        end_flag, self.frame = True, getBasler()

        self.gray_prev = cv2.resize(self.frame , (640,480))
        self.gray_prev = cv2.GaussianBlur(self.gray_prev, (kernel, kernel), sigma)
        self.gray_prev = cv2.cvtColor(self.gray_prev, cv2.COLOR_BGR2GRAY)
        
        detection_x = 0
        detection_y = 0
        previous_x = 0
        previous_y = 0
        
        dete_time = 0
        prev_time = 0

        self.m = m
        self.now = now
        self.filename = filename
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')


        while end_flag:
             
            time11 = time.time()
            ser.write(b"Q:\r\n")
            line = ser.readline().decode("utf-8")
            sdetect_x1 = (line[0:1].replace(" ", ""))
            sdetection_x = int(line[1:10].replace(" ", ""))
            sdetect_y1 = (line[11:12].replace(" ", ""))
            sdetection_y = int(line[12:21].replace(" ", ""))
            

            
            if sdetect_x1 == "-":
                sdetection_x = -sdetection_x
            else:
                pass
            
            if sdetect_y1 == "-":
                sdetection_y = -sdetection_y
            else:
                pass

            time12 = time.time()
            #print(time12-time11)
            #print(sdetection_x,sdetection_y)

            dete_time = time.time()
            tn = datetime.datetime.now()
            time_delta = datetime.timedelta(hours=tn.hour,minutes=tn.minute,seconds=tn.second,microseconds=tn.microsecond)
        
            self.frame = cv2.resize(self.frame , (640,480))
            self.gray_next = cv2.GaussianBlur(self.frame, (kernel,kernel), sigma)
            self.gray_next = cv2.cvtColor(self.gray_next, cv2.COLOR_BGR2GRAY)

            time1 = time.time()

            #現在の画像を動画に書き込み
            if (flag == True) :
                try:
                    writer.write(self.frame) 
                except:
                    pass

            time2 = time.time()

            #print(time2-time1)
            
            # 探索半径（pixel）
            self.radius = radius
            
            #特徴点を指定する
            if(detect_flag == True):
                image_operation2.onMouse(self,click_x,click_y)  
                detect_flag = False
                self.rec_x = 0
                self.rec_y = 0
                self.srec_x = 0
                self.srec_y = 0

            #右クリックで特徴点リセット
            if(reset_flag == True):
                self.features = None
                reset_flag = False
                detection_x = None
                detection_y = None
                previous_x = None
                previous_y = None
                delta_x = 0
                delta_y = 0

            time3 = time.time()
            
            if(detect_up == True):
                for feature in self.features:
                        feature[0][1] -= mark
                detect_up = False
            elif(detect_down == True):
                for feature in self.features:
                        feature[0][1] += mark
                detect_down = False
            elif(detect_left == True):
                for feature in self.features:
                        feature[0][0] -= mark
                detect_left = False
            elif(detect_right == True):
                for feature in self.features:
                        feature[0][0] += mark
                detect_right = False
            else:
               pass

            #if(detect_up == True):
                # for feature in self.features:
                #         feature[0][1] -= mark
            #    ser.write(("D:WS10000F10001R20S10000F10001R20" + "\r\n").encode())
            #    detect_up = False
            #elif(detect_down == True):
                # for feature in self.features:
                #         feature[0][1] += mark
            #    ser.write(("D:WS30000F30001R20S30000F30001R20"+ "\r\n").encode())
            #    detect_down = False
            #elif(detect_left == True):
                # for feature in self.features:
                #         feature[0][0] -= mark
            #    ser.write(("D:WS50000F50001R20S50000F50001R20"+ "\r\n").encode())
            #    detect_left = False
            #elif(detect_right == True):
                # for feature in self.features:
                #         feature[0][0] += mark
            #    ser.write(("D:WS60000F60001R20S60000F60001R20"+"\r\n").encode())
            #    detect_right = False
            #else:
            #    pass

            time4 = time.time()
                
            # 特徴点が登録されている場合にOpticalFlowを計算する
            if self.features is not None or (detection_x is not None and detection_y is not None):
                if  len(self.features) != 0:
                    features_prev = self.features
                    self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
                                                        self.gray_prev, \
                                                        self.gray_next, \
                                                        features_prev, \
                                                        None, \
                                                        winSize = (self.radius, self.radius), \
                                                        maxLevel = 3, \
                                                        criteria = CRITERIA, \
                                                        flags = 0)
                    
                    # 有効な特徴点のみ残す
                    image_operation2.refreshFeatures(self)
            
                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        detection_x = int(feature[0][0])
                        detection_y = int(feature[0][1])
                
                    time5 = time.time()
                        
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
                
                time6 = time.time()

                if detection_x is not None and detection_y is not None and previous_x is not None and previous_y is not None:
                    if(track_flag == True):

                        self.delta_d_flag = math.sqrt(pow((detection_x - previous_x),2)+pow((detection_y - previous_y),2))
                
                        if self.delta_d_flag < asobi:
                            self.m = 1#10#10
                        elif self.delta_d_flag >= asobi and self.delta_d_flag < 5:
                            self.m = 1#20#50
                        elif self.delta_d_flag >= 5 and self.delta_d_flag < 10:
                            self.m = 1#30#80
                        else:
                            self.m = 1#40#90

                        gap_x = centerx-detection_x
                        gap_y = centery-detection_y
                        #print(gap_x, centerx, detection_x)

                    
                        # if ((abs(gap_x) <= asobi and abs(gap_y) <= asobi)) and (flag == True):
                        #     stage_flag = False      
                        # elif (gap_x) < -asobi and (gap_y) > asobi and (flag == True):
                        #     MainWindow.upperRight(self, gap_x, gap_y, self.m)
                        # elif (gap_x) < -asobi and (gap_y) < -asobi and (flag == True):
                        #     MainWindow.lowerRight(self, gap_x, gap_y, self.m)                   
                        # elif (gap_x) > asobi and (gap_y) > asobi and (flag == True):
                        #     MainWindow.upperLeft(self, gap_x, gap_y, self.m)
                        # elif (gap_x) > asobi and (gap_y) < -asobi and (flag == True):
                        #     MainWindow.lowerLeft(self, gap_x, gap_y, self.m)
                        # elif abs(gap_x) <= asobi and (gap_y) > asobi and (flag == True):
                        #     MainWindow.Up(self, gap_y, self.m)    
                        # elif abs(gap_x) <= asobi and (gap_y) < -asobi and (flag == True):
                        #     MainWindow.Down(self, gap_y, self.m)      
                        # elif (gap_x) < -asobi and abs(gap_y) <= asobi and (flag == True):
                        #     MainWindow.Right(self, gap_x, self.m)             
                        # elif (gap_x) > asobi and abs(gap_y) <= asobi and (flag == True):
                        #     MainWindow.Left(self, gap_x, self.m)
                        # else:
                        #     print("err\n")
                        
                    
                        time7 = time.time()
                        
                                                       
                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), self.radius*2, (255, 0, 255))
                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), asobi*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (detection_x, detection_y), (0, 255, 125))

                    time8 = time.time()

            else:
                pass
                #print("No detected")
            
            # 表示
            self.frame = cv2.drawMarker(self.frame, position=(centerx, centery), color=(255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
            mainImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            time9 = time.time()
            if (flag == True) or (track_flag == True) :
                self.frame_num = self.frame_num +1
                #ループ時間
                t = dete_time - prev_time
                #print(t)
                #フレーム間の移動距離
                d = math.sqrt(pow((detection_x-previous_x)*pi_m,2)+pow((detection_y-previous_y)*pi_m,2))
                #マーカーの移動速度
                track_v = d / float(t)
                #ステージの移動速度
                if stage_flag == True:
                    if abs(gap_x) >= abs(gap_y):
                        stage_t = math.sqrt(float((4*abs(gap_x)*self.m*pu_m))/acce)
                        stage_v = (low_speed*pu_m + acce*stage_t/4)
                    elif abs(gap_x) < abs(gap_y):
                        stage_t = math.sqrt(float((4*abs(gap_y)*self.m*pu_m))/acce)
                        stage_v = (low_speed*pu_m + acce*stage_t/4)
                else:
                    stage_t = 0
                    stage_v = 0
                    stage_flag = True
                #線虫の絶対速度
                v = abs(track_v-stage_v)
                #中心とのずれ
                Err = math.sqrt(pow((gap_x)*pi_m,2)+pow((gap_y)*pi_m,2))

                #log_file.write("frame,time,x(pixel),y(pixel),radius(pixel),asobi(pixel),Δt(s),Δd(μm),track_v(μm/s),err(μm),sum_x(pixel),sum_y(pixel)\n")
                log_file.write(str(self.frame_num)+","+str(time_delta.total_seconds())+","+str(detection_x)+","+str(detection_y)+","
                    +str(sdetection_x)+","+str(sdetection_y)+","+str(t)+","+str(d)+","+str(track_v)+","+str(Err)+","
                    +str(self.rec_x)+","+str(self.rec_y)+"\n")   
                #","+str(stage_v)+","+str(v)+""""""","+str(stage_t)+","+str(self.m)","+str(gap_x)+","+str(gap_y)+
                """log_file.write(str(dete_time)+","+str(time1)+","+str(time2)+","+str(time3)+","
                    +str(time4)+","+str(time5)+","+str(time6)+","+str(time7)+","+str(time8)+","+str(time9)+","
                    +str(time10)+","
                    +str(time11)+") """

                self.rec_x = self.rec_x + delta_x
                self.rec_y = self.rec_y + delta_y   

            # 次のループ処理の準備
            previous_x = detection_x
            previous_y = detection_y
            prev_time = dete_time
            self.gray_prev = self.gray_next
            time10 = time.time()

            end_flag, self.frame = True, getBasler()

            time11 = time.time()

            # 追跡終了後の処理（データ保存など）
            """if end_flag:
                pass
            else:
                pass"""
            if save_flag ==True:

                self.frame_num = 0
                save_flag =False
                log_file.close()
                writer.release()
                
                video_path = os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")+"/"+self.filename

                csv_file = open(log_name,"r") 
                csvReader = csv.reader(csv_file)
                csvHeader = next(csvReader)

                s_time = open(r"F:\tracking stage\log\start_time.txt","r") 
                start_time = s_time.read()
                s_path = open(r"F:\tracking stage\log\save_path.txt","r")
                save_path = s_path.read()  #os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")

                parameter_name = save_path+"/"+"experimental condition.txt"
                parameter_file = open(parameter_name,"a")
                parameter_file.write("fps:\t"+str(fps)+"\npixsel:\t"+str(size)+"\nlow_speed:\t"+str(low_speed)+"\ntop_speed:\t"+str(top_speed)+"\nacce_time:\t"+str(acce_time)+"\n 5.4%"+"\n")
                parameter_file.close()
                #write(オリジナル編集画像)の保存先を作成
                filename_edit = "rec-edit.avi"
                writer_edit = cv2.VideoWriter(save_path+"/"+filename_edit, fourcc, fps, size)
                #writer_edit = cv2.VideoWriter(os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")+"/"+filename_edit, fourcc, fps, size)
                #write_detect(検出編集画像)の保存先を作成
                filename_dete_edit = "rec-detect.avi"
                #writer_detect = cv2.VideoWriter(save_path+"/"+filename_dete_edit, fourcc, fps, size)
                writer_detect = cv2.VideoWriter(os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")+"/"+filename_dete_edit, fourcc, fps, size)

                for row in csvReader:
                    time_csv = float(row[1])-0.15
                    time_csv = str(time_csv)
                    if start_time < time_csv:
                        frame_num = row[0]
                        break
                    else:
                        frame_num = 0
                        pass

                #parameter情報の保存
                num_file = open(r"F:\tracking stage\log\num.txt","w")
                num_file.write(str(frame_num))
                num_file.close()

                frame_num = int(frame_num)
                csv_file.close()

                

                # 動画ファイル・座標ファイルの読み込み
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("動画の読み込み失敗")
                    sys.exit()

                csv_fileR = open(log_name,"r") 
                csvR = pandas.read_csv(csv_fileR,usecols=[0,2,3,4,5])

                n = 0
                while True:
                    is_image,frame_img = cap.read()
                    if n == frame_num:
                        while True:
                            if is_image:
                                # 動画に現在の画像を書き込む-ver.normal
                                try:
                                    writer_edit.write(frame_img) 
                                except:
                                    pass

                                """detection_x_ed = csvR.iat[n,1]
                                detection_y_ed = csvR.iat[n,2]
                                #radius_ed = int(csvR.iat[n,3])
                                #asobi_ed = int(csvR.iat[n,4])
                                

                                frame_img = cv2.circle(frame_img, (detection_x_ed, detection_y_ed), radius_ed*2, (140, 0, 230))
                                frame_img = cv2.circle(frame_img, (detection_x_ed, detection_y_ed), asobi_ed*2, (0, 255, 255))
                                frame_img = cv2.drawMarker(frame_img, (detection_x_ed, detection_y_ed), (0, 255, 0))

                                if n>=1:
                                    previous_x_ed = int(csvR.iat[n-1,2])
                                    previous_y_ed = int(csvR.iat[n-1,3])
                                    radius_ed = int(csvR.iat[n-1,4])
                                    asobi_ed = int(csvR.iat[n-1,5])
                                    frame_img = cv2.circle(frame_img, (previous_x_ed, previous_y_ed), radius_ed*2, (0, 0, 255))
                                    frame_img = cv2.circle(frame_img, (previous_x_ed, previous_y_ed), asobi_ed*2, (0, 255, 255))
                                    frame_img = cv2.drawMarker(frame_img, (previous_x_ed, previous_y_ed), (0, 255, 0))
                                else:
                                    pass"""

                                # 動画に現在の画像を書き込む-ver.detect
                                """try:
                                    writer_detect.write(frame_img)
                                except:
                                    pass
                                n += 1"""
                            else:
                                cap.release()
                                writer_edit.release()
                                #writer_detect.release()
                                csv_fileR.close()
                                break
                            is_image,frame_img = cap.read()
                            
                    else:
                        pass
                    if is_image:
                        pass
                    else:
                        break
                    n += 1

                """csv_file = open(log_name,"r") 
                csv_df = pandas.read_csv(csv_file,header=100,encoding="shift-jis")
                plt.rcParams["figure.figsize"]=(12,6)
                fig = plt.figure()
                ax1=fig.subplots()
                ax2 = ax1.twinx()
                x = csv_df[csv_df.columns[0]]
                data_y1 = csv_df[csv_df.columns[8]]
                data_y2 = csv_df[csv_df.columns[9]]
                plt.title("Title",fontsize=10)
                ax1.set_xlabel("frame",fontsize=10)
                ax1.set_ylabel("velocity",fontsize=10)
                ax2.set_ylabel("err")
                ax1.tick_params(labelsize=10)
                ax2.tick_params(labelsize=10)
                ax1.plot(x, data_y1, linestyle='dashed',color="blue",label="v",linewidth=2)
                ax2.plot(x, data_y2, linestyle='solid',color="cyan",label="err",alpha=0.4)
                h1,l1=ax1.get_legend_handles_labels()
                h2,l2=ax2.get_legend_handles_labels()
                ax1.legend(h1+h2,l1+l2)
                plt.grid()
                savename = "graph.jpg"
                fig.savefig(os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")+"/"+savename,dpi=300)
                csv_file.close()
                plt.clf()
                plt.close()

                data_set = np.loadtxt(fname=log_name,dtype="float",delimiter=",",skiprows=100)
                for data in data_set:
                    plt.scatter(data[9],data[8])
                plt.title("correlation")
                plt.xlabel("err")
                plt.ylabel("velocity")
                plt.grid()
                savename = "correlation.jpg"
                plt.savefig(os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")+"/"+savename,dpi=300)
                plt.clf()
                plt.close()"""

                print("edit finished")

                self.now = datetime.datetime.now()
                try:
                    os.mkdir(os.getcwd()+ "/log/"+self.now.strftime("%Y%m%d%H%M%S"))
                except FileExistsError:
                    pass
                #logの保存先を作成
                log_name = self.now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"log.csv"
                log_file = open(log_name,"a")
                log_file.write("frame,time,x(pixel),y(pixel),radius(pixel),asobi(pixel),Δt(s),Δd(μm),track_v(μm/s),err(μm),sum_x(pixel),sum_y(pixel)\n")
                #parameter情報の保存
                parameter_name = self.now.strftime(os.getcwd()+"/log/"+"%Y%m%d%H%M%S/")+"parameter.txt"
                parameter_file = open(parameter_name,"a")
                parameter_file.write("fps:\t"+str(fps)+"\npixsel:\t"+str(size)+"\nlow_speed:\t"+str(low_speed)+"\ntop_speed:\t"+str(top_speed)+"\nacce_time:\t"+str(acce_time)+"\n")
                parameter_file.close()
                #write(オリジナル画像)の保存先を作成
                self.filename = "rec.avi"
                writer = cv2.VideoWriter(os.getcwd()+"/log/"+self.now.strftime("%Y%m%d%H%M%S")+"/"+self.filename, fourcc, fps, size)

                print("ready\n")
                
            else:
                pass
                    
             
            # numpy 配列を QImage に変換する。
            mainQImage = QtGui.QImage(mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            # シグナルを送出する。
            self.mainFrameGrabbed.emit(mainQImage)
        

def getBasler():
    grabResult = basler.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = basler.converter.Convert(grabResult)
        img = image.GetArray()
    return img


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())



"""
[理想]
振動が無視可能な範囲に収まり、相対速度vが300µm/s以内に収まり、線虫の追跡が可能なステージ速度の最適なパラメータ設定
---------------------------------------------------------------------------------------------------------------------

2022/07/21追記

 ・振動条件まとめる
 ・速度最適化
 ・Gaussian Filter条件まとめる
 ・kernel & sigma 最適化
 ・move_scale & m (magnification) 最適化
 ・T'の検討
 ・相対速度精度の評価
 ・ステージ移動の時間（どれが正しいのか）
 ・時間計測（どこに時間がかかっているのか、ステージがR状態になるのにかかる時間）
 ・ループの高速化（可能な限り速く）
 ・エラーの大きさを15-20ミクロン以内に
 ・実験の始めに静止状態でキャリブレーション（蛍光ビーズを用いて）

2022/08/17追記

 ・画像のコントラスト設定の追加
 ・画像の拡大表示の設定
 ・→キーを用いたmarkerの移動

2022/08/18追記

 ・動画保存を別のスレッドで保存（マルチスレッド処理）
 ・保存した動画にマーカーをつける（後付け）
 ・速度表示の部分を正確にする（マーカー計測or移動量計算））
 ・detectionのマーカー付け

2022/08/31追記

 ・速度表示
 ・追跡可能な速度の検証
 ・エラー率の表示
 ・及びこれらのデータ取り

2022/09/05追記
 試すこと
 ・閾値処理
 ・ぼかしと平滑化
 ・動画ファイルの編集
 ・動画ファイル2スレッド処理
 ・テンプレートマッチング
 ・opticalflowの数値変え

2022/09/16追記

 ・解析するための情報の取得
  ・速度（ステージ速度との組み合わせ）
  ・エラー（中心との差）
  ・上記二個の相関図（散布図）
  ・実験条件（パラメータ）
  ・フレーム数
  ・フレーム時刻

 ・今日撮影した動画から速度を計算出るかどうかやってみる

 ・多点計測での線中頭部のオプティカルフロー計算ベクトル量
 ・所定感覚で計測するハチの巣状
 ・テキストデータから時刻を読み取り、フレーム取得し編集。
 ・線虫の中心線
 ・線虫の移動マップ


ライトシートのxy範囲
x
y
0.455*300

線虫パラメータの記載
GUIの調整
トラッキング位置の推敲


線虫の蛍光たんぱくの種類
ライトシートの撮像範囲（大きさ）

線虫のスクリーニング（明るさ・大きさ（卵・子持ち）の観点から難しい）

"""