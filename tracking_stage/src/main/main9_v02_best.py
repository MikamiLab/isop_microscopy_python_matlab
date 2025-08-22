# -*- coding: utf-8 -*-
#from fileinput import filename
from re import S
import typing

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow, QDialog, QMessageBox,QGraphicsPixmapItem
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2, time ,datetime ,os ,math ,glob ,csv ,pandas ,sys, shutil, queue
import numpy as np
import matplotlib.pyplot as plt

from pypylon import pylon
import common.basler as basler

import common.image_operation3 as image_operation3

from multiprocessing import Process, Queue

import ctypes
import Pxep


CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

MAINPC_PATH = r"F:\tracking stage\log\save_path.txt"
START_TIME = r"F:\tracking stage\log\start_time.txt"


class variable():
    # profile
    fps = 115
    size = (640, 480)
    PulsetoMicro = 0.01590033
    PixeltoMicro = 0.616291862

    now = None
    today = None
    click_x = None
    click_y = None

    stagex = 0
    stagey = 0
    kernel = 1 #7
    sigma = 1 #5

    # path
    file_path = None
    csv_path = None
    param_path = None
    video_path = None
    videoD_path = None

    # log
    log_file = None
    param_file = None
    writer = None
    writerD =None

    # parameter
    ## 励磁
    wait_SetServo = 100
    ## 明視野パラメータ
    center_x = 345
    center_y = 224
    radius = 15#16
    asobi = 1
    marker = 2
    scale = 35 #38.85873036
    ## 速度パラメータ
    StartVelocity = 3000#600
    HighVelocity = 20000#6000
    AccelerationTime = 50# 100
    DecelerationTime = 50#80
    AccelerationS = 50
    OverVelocity = 10
    ## ソフトリミット
    Mode = 0
    Event = 0
    SoftLimit_upper = 0
    SoftLimit_lower = 0
    ## ドライブ
    pulse = 5000

    # flag
    flag = False
    save_flag = False
    detect_flag = False
    reset_flag = True
    marker_flag = None

    fov1_flag = False
    fov2_flag = False

    cap_flag = False


class KalmanFilter():
    kf = cv2.KalmanFilter(4,2)
    kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

class HomeStart_Window(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi(r"Gui\Home_Dialog.ui", self)

class Help_Window(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi(r"Gui\Help_Dialog.ui", self)

class Save_Window(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi(r"Gui\Save_Dialog.ui", self)

class Analysis_Window(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi(r"Gui\Analysis_Dialog.ui", self)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi(r"F:\tracking stage\Gui\MainWindow.ui",self)

        self.WinHome = HomeStart_Window()
        self.WinHelp = Help_Window()
        self.WinAnalysis = Analysis_Window()

        scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.graphicsView.installEventFilter(self)

        self.spinBox_1.setValue(variable.center_x)
        self.spinBox_2.setValue(variable.center_y)
        self.spinBox_3.setValue(variable.radius)
        self.spinBox_4.setValue(variable.asobi)
        self.spinBox_5.setValue(variable.marker)
        self.spinBox_6.setValue(variable.StartVelocity)
        self.spinBox_7.setValue(variable.HighVelocity)
        self.spinBox_8.setValue(variable.AccelerationTime)
        self.spinBox_9.setValue(variable.DecelerationTime)
        self.spinBox_10.setValue(variable.AccelerationS)
        self.spinBox_11.setValue(variable.OverVelocity)
        self.spinBox_12.setValue(variable.pulse)
        self.spinBox_13.setValue(variable.SoftLimit_upper)
        self.spinBox_14.setValue(variable.SoftLimit_lower)

        self.doubleSpinBox.setValue(variable.scale)
        self.doubleSpinBox_2.setValue(variable.wait_SetServo)

        self.spinBox_1.valueChanged.connect(self.valuechange_1)
        self.spinBox_2.valueChanged.connect(self.valuechange_2)
        self.spinBox_3.valueChanged.connect(self.valuechange_3)
        self.spinBox_4.valueChanged.connect(self.valuechange_4)
        self.spinBox_5.valueChanged.connect(self.valuechange_5)
        self.spinBox_6.valueChanged.connect(self.valuechange_6)
        self.spinBox_7.valueChanged.connect(self.valuechange_7)
        self.spinBox_8.valueChanged.connect(self.valuechange_8)
        self.spinBox_9.valueChanged.connect(self.valuechange_9)
        self.spinBox_10.valueChanged.connect(self.valuechange_10)
        self.spinBox_11.valueChanged.connect(self.valuechange_11)
        self.spinBox_12.valueChanged.connect(self.valuechange_12)
        self.spinBox_13.valueChanged.connect(self.valuechange_13)
        self.spinBox_14.valueChanged.connect(self.valuechange_14)

        self.doubleSpinBox.valueChanged.connect(self.valuechange2_1)
        self.doubleSpinBox_2.valueChanged.connect(self.valuechange2_2)

        self.pushButton.pressed.connect(self.CloseMainWindow)
        self.pushButton_1.pressed.connect(self.Ready)
        self.pushButton_2.pressed.connect(self.TrackON)
        self.pushButton_3.pressed.connect(self.TrackOFF)
        self.pushButton_4.pressed.connect(self.ServoON)
        self.pushButton_5.pressed.connect(self.ServoOFF)
        self.pushButton_6.pressed.connect(self.SetVelocity)
        self.pushButton_7.pressed.connect(self.GetVelocity)
        self.pushButton_8.pressed.connect(self.Up_button)
        self.pushButton_9.pressed.connect(self.Down_button)
        self.pushButton_10.pressed.connect(self.Left_button)
        self.pushButton_11.pressed.connect(self.Right_button)
        self.pushButton_12.pressed.connect(self.UpperLeft_button)
        self.pushButton_13.pressed.connect(self.UpperRight_button)
        self.pushButton_14.pressed.connect(self.LowerLeft_button)
        self.pushButton_15.pressed.connect(self.LowerRight_button)
        self.pushButton_16.pressed.connect(self.Stop)
        self.pushButton_17.pressed.connect(self.SetSoftLimit)
        self.pushButton_18.pressed.connect(self.GetSoftLimit)
        self.pushButton_19.pressed.connect(self.CameraON)
        self.pushButton_20.pressed.connect(self.CameraOFF)

        self.radioButton_7.toggled.connect(self.FOV)
        self.radioButton_8.toggled.connect(self.FOV)

        self.commandLinkButton.pressed.connect(self.Open_WinHome)
        self.commandLinkButton_2.pressed.connect(self.Open_WinHelp)

        self.wBSN = 0
        self.wIDx = 2
        self.wIDy = 1

        # ネットワーク初期化
        pbID = (ctypes.c_uint8*32)()
        ret = Pxep.PxepwAPI.RcmcwOpen(self.wBSN,1,pbID)

        # アラームクリア
        ret = Pxep.PxepwAPI.RcmcwALMCLR(self.wBSN,self.wIDx)
        ret = Pxep.PxepwAPI.RcmcwALMCLR(self.wBSN,self.wIDy)
        time.sleep(1)

        # ソフトリミット無効設定
        ret= Pxep.PxepwAPI.RcmcwSetSoftLimit(self.wBSN,self.wIDx,500,0,3,0)
        ret= Pxep.PxepwAPI.RcmcwSetSoftLimit(self.wBSN,self.wIDy,500,0,3,0)

        # ハードリミットアクティブレベル設定,尚、デフォルト値が0のため、アクティブレベルLの場合は、設定する必要なし
        objNo=0x2000   # wID=1で1軸目の場合は、0x2000設定
        setdata = (ctypes.c_uint32*1)()
        setdata[0] = 0
        ret = Pxep.PxepwAPI.RcmcwObjectControl(self.wBSN,self.wIDx,0,objNo,0,2,setdata)
        objNo=0x2400   # wID=2で2軸目の場合は、0x2400設定
        setdata = (ctypes.c_uint32*1)()
        setdata[0] = 0
        ret = Pxep.PxepwAPI.RcmcwObjectControl(self.wBSN,self.wIDy,0,objNo,0,2,setdata)

        # サーボON
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDx,1,0)
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDy,1,0)

        # 原点復帰現在位置 = 0 設定
        HomeSt = Pxep.HOMEPARAM(35,35,1,0,0,0,0,0)
        ret = Pxep.PxepwAPI.RcmcwHomeStart(self.wBSN,self.wIDx,HomeSt)
        ret = Pxep.PxepwAPI.RcmcwHomeStart(self.wBSN,self.wIDy,HomeSt)

        # 速度設定
        AxisSpeed = Pxep.SPDPARAM(1,variable.StartVelocity,variable.StartVelocity,40000,50,50,0,0,0,0,0,0)
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(self.wBSN,self.wIDx,AxisSpeed)
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(self.wBSN,self.wIDy,AxisSpeed)
        time.sleep(0.1)

        # ドライブ
        ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,100)
        ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,100)

        # サーボOFF
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDx,0,0)
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDy,0,0)

        self._item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._item)
        thread = CapThread(self)
        thread.image_update.connect(self.update_mainimage)
        thread.start()

    @pyqtSlot(QtGui.QImage)
    def update_mainimage(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self._item.setPixmap(pixmap)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseButtonPress and source is self.graphicsView and self.graphicsView.itemAt(event.pos()):
            if event.button() == Qt.LeftButton:
                variable.detect_flag = True
                pos = self.graphicsView.mapToScene(event.pos())
                variable.click_x = pos.x()
                variable.click_y = pos.y()
            if event.button() == Qt.RightButton:
                variable.reset_flag = True
        return QWidget.eventFilter(self, source, event)

    # 関数実行結果を16進変換して表示
    def ShowResult(self,Message,rel):
        if rel >= 0:    # 正数の場合
            print(Message,'Result=0x{:08X}'.format(rel))
        else:           # 負数の場合
            print(Message,'Result=0x{:08X}'.format(0xFFFFFFFF + rel + 0x01))

    def valuechange_1(self):
        variable.center_x           = self.spinBox_1.value()
    def valuechange_2(self):
        variable.center_y           = self.spinBox_2.value()
    def valuechange_3(self):
        variable.radius             = self.spinBox_3.value()
    def valuechange_4(self):
        variable.asobi              = self.spinBox_4.value()
    def valuechange_5(self):
        variable.marker             = self.spinBox_5.value()
    def valuechange_6(self):
        variable.StartVelocity      = self.spinBox_6.value()
    def valuechange_7(self):
        variable.HighVelocity       = self.spinBox_7.value()
    def valuechange_8(self):
        variable.AccelerationTime   = self.spinBox_8.value()
    def valuechange_9(self):
        variable.DecelerationTime   = self.spinBox_9.value()
    def valuechange_10(self):
        variable.AccelerationS      = self.spinBox_10.value()
    def valuechange_11(self):
        variable.OverVelocity       = self.spinBox_11.value()
    def valuechange_12(self):
        variable.pulse              = self.spinBox_12.value()
    def valuechange_13(self):
        variable.SoftLimit_upper    = self.spinBox_13.value()
    def valuechange_14(self):
        variable.SoftLimit_lower    = self.spinBox_14.value()

    def valuechange2_1(self):
        variable.scale              = self.doubleSpinBox.value()
    def valuechange2_2(self):
        variable.wait_SetServo      = self.doubleSpinBox_2.value()

    def Ready(self):
        now = datetime.datetime.now()
        today = datetime.date.today()
        variable.now = now.strftime("%Y%m%d%H%M%S")
        variable.file_path = os.getcwd()+ "/log/"+str(today)+"/"+variable.now

        try:
            os.mkdir(os.getcwd()+ "/log")
        except FileExistsError:
            pass
        try:
            os.mkdir(os.getcwd()+ "/log/"+str(today))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.getcwd()+ "/log/"+str(today)+"/"+variable.now)
        except FileExistsError:
            pass

        variable.csv_path = variable.file_path+"/log.csv"
        variable.log_file = open(variable.csv_path,"a")
        variable.log_file.write("frame"+","+
                                "timestamp"+","+
                                "stageX[pulse]"+","+
                                "stageY[pulse]"+","+
                                "Δt[s]"+","+
                                "Δd[μm]"+","+
                                "velocity[μm/s]"+","+
                                "detectionX[pixel]"+","+
                                "detectionY[pixel]"+","+
                                "gap_x[pixel]"+","+
                                "gap_y[pixel]"+","+
                                "gap_x[μm]"+","+
                                "gap_y[μm]"+","+
                                "gap_x[pulse]"+","+
                                "gap_y[pulse]"+","+
                                "error[μm]"+","+
                                "kal_x[pixel]"+","+
                                "kal_y[pixel]"+","+
                                "radius[pixel]"+","+
                                "asobi[pixel]"+"\n"
                                )

        variable.param_path = variable.file_path+"/parameter.txt"
        variable.param_file = open(variable.param_path,"a")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        variable.video_path = variable.file_path+"/rec.avi"
        variable.videoD_path = variable.file_path+"/rec-detect.avi"
        variable.writer = cv2.VideoWriter(variable.video_path, fourcc, variable.fps, variable.size)
        variable.writerD = cv2.VideoWriter(variable.videoD_path, fourcc, variable.fps, variable.size)

        try:
            os.mkdir(variable.file_path+"/rec-detect.avi")
        except FileExistsError:
            print("ready\n")

        self.pushButton_1.setEnabled(False)
        self.pushButton_2.setEnabled(True)

        self.pushButton_1.setStyleSheet("background-color: red")


    def TrackON(self):
        variable.flag = True
        variable.save_flag = False

        self.ServoON()

        # self.spinBox_2.setEnabled(True)
        # self.spinBox_6.setEnabled(False)
        # self.spinBox_7.setEnabled(False)
        # self.spinBox_8.setEnabled(False)

        self.radioButton_5.setChecked(True)

        self.pushButton_1.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(True)
        # self.pushButton_4.setEnabled(False)
        # self.pushButton_5.setEnabled(False)
        # self.pushButton_6.setEnabled(False)
        # self.pushButton_7.setEnabled(False)
        # self.pushButton_16.setEnabled(False)
        # self.pushButton_17.setEnabled(False)
        # self.pushButton_18.setEnabled(False)
        # self.pushButton_19.setEnabled(False)
        # self.pushButton_20.setEnabled(False)

        self.pushButton_2.setStyleSheet("background-color: red")

    def TrackOFF(self):
        variable.flag = False
        variable.save_flag = True

        self.ServoOFF()

        # self.spinBox_2.setEnabled(True)
        # self.spinBox_6.setEnabled(True)
        # self.spinBox_7.setEnabled(True)
        # self.spinBox_8.setEnabled(True)

        self.radioButton_5.setChecked(False)

        self.pushButton_1.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        # self.pushButton_4.setEnabled(True)
        # self.pushButton_5.setEnabled(True)
        # self.pushButton_6.setEnabled(True)
        # self.pushButton_7.setEnabled(True)
        # self.pushButton_16.setEnabled(True)
        # self.pushButton_17.setEnabled(True)
        # self.pushButton_18.setEnabled(True)
        # self.pushButton_19.setEnabled(True)
        # self.pushButton_20.setEnabled(True)

        self.pushButton_1.setStyleSheet("background-color: white")
        self.pushButton_2.setStyleSheet("background-color: white")

    def CloseMainWindow(self):
        """# files = glob.glob(os.getcwd() + "/save/*")
        # for file in files:
        #     if(os.path.getsize(file) < 6000):
        #         os.remove(file)"""
        if(variable.flag == True):
            self.TrackOFF()
        else:
            pass
        ret = Pxep.PxepwAPI.RcmcwClose(self.wBSN)
        app.quit()

    def ServoON(self):
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(True)
        self.pushButton_4.setStyleSheet("background-color: red")
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDx,1,0)
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDy,1,0)
        # ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,10)
        # ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,10)

    def ServoOFF(self):
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(False)
        self.pushButton_4.setStyleSheet("background-color: white")
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDx,0,0)
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDy,0,0)

    def SetVelocity(self):
        Mode = 1
        SPEEDst = (Mode,
                   variable.StartVelocity,
                   variable.StartVelocity,
                   variable.HighVelocity,
                   variable.AccelerationTime,
                   variable.AccelerationS,
                   variable.DecelerationTime,
                   variable.DecelerationS,
                   variable.OverRide_Vel,
                   variable.TriangularDrive,
                   variable.AccelerationPLS,
                   variable.DecelerationPLS
                   )
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameterF(self.wBSN,self.wIDy,SPEEDst)
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameterF(self.wBSN,self.wIDy,SPEEDst)

    def GetVelocity(self):
        print("GetSpeedParameter")

    def Up_button(self):
        if self.radioButton_5.isChecked():
            variable.marker_flag = 0
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,-variable.pulse)
    def Down_button(self):
        if self.radioButton_5.isChecked():
            variable.marker_flag = 1
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,variable.pulse)
    def Right_button(self):
        if self.radioButton_5.isChecked():
            variable.marker_flag = 3
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,-variable.pulse)
    def Left_button(self):
        if self.radioButton_5.isChecked():
            variable.marker_flag = 2
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,variable.pulse)
    def UpperRight_button(self):
        if self.radioButton_5.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,-variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,-variable.pulse)
    def UpperLeft_button(self):
        if self.radioButton_5.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,-variable.pulse)
    def LowerRight_button(self):
        if self.radioButton_5.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,-variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,variable.pulse)
    def LowerLeft_button(self):
        if self.radioButton_5.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,variable.pulse)

    def Stop(self):
        ret = Pxep.PxepwAPI.RcmcwDriveStop(self.wBSN,self.wIDx,0)
        ret = Pxep.PxepwAPI.RcmcwDriveStop(self.wBSN,self.wIDy,0)

    # def DrivePLS(self,xPulse,yPulse):
    #     ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,xPulse)
    #     ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,yPulse)

    def OverRidePLS(self,xPulse,yPulse):
        ret = Pxep.PxepwAPI.RcmcwPulseOverride(self.wBSN,self.wIDx,xPulse)
        ret = Pxep.PxepwAPI.RcmcwPulseOverride(self.wBSN,self.wIDy,yPulse)
    def OverRideVel(self):
        HighSpeed = 1000
        ret = Pxep.PxepwAPI.RcmcwSpeedOverride(self.wBSN,self.wIDx,HighSpeed)
        ret = Pxep.PxepwAPI.RcmcwSpeedOverride(self.wBSN,self.wIDy,HighSpeed)

    def SetSoftLimit(self):
        print("set")
    def GetSoftLimit(self):
        print("get")

    def GetTimeOutInfo(self):
        pdwDataInfo = (ctypes.c_int32*10)()
        ret = Pxep.PxepwAPI.RcmcwGetTimeOutInfo(self.wBSN,pdwDataInfo)
        # for i in range(3):
        #     print('pdwMasterInfo[{0:d}]={1:08X} '.format(i,pdwDataInfo[i]))
        print(pdwDataInfo[0], pdwDataInfo[1], pdwDataInfo[2])
        #[0]:通信状況,[1]:CPU使用率,[2]:指令更新周期ns

    def GetCounter(self):
        pdwDatax = (ctypes.c_int32*7)()
        pdwDatay = (ctypes.c_int32*7)()
        ret = Pxep.PxepwAPI.RcmcwGetCounter(self.wBSN,self.wIDx,pdwDatax)
        ret = Pxep.PxepwAPI.RcmcwGetCounter(self.wBSN,self.wIDy,pdwDatay)
        # for i in range(3):
        #     print('pdwMasterInfo[{0:d}]={1:08X} '.format(i,pdwData[i]))
        print(pdwDatax[1], pdwDatay[1])

    def CameraON(self):
        self.radioButton_7.setEnabled(False)
        self.radioButton_8.setEnabled(False)

    def CameraOFF(self):
        self.radioButton_7.setEnabled(True)
        self.radioButton_8.setEnabled(True)

    def FOV(self):
        radioBtn = self.sender()
        if radioBtn == self.radioButton_7:
            variable.fov2_flag = False
            variable.fov1_flag = True
        elif radioBtn == self.radioButton_8:
            variable.fov1_flag = False
            variable.fov2_flag = True


    def Open_WinHome(self):
        self.WinHome.show()
    def Close_WinHome(self):
        self.WinHome.close()

    def Open_WinHelp(self):
        self.WinHelp.show()
    def Close_WinHelp(self):
        self.WinHelp.close()


class CapThread(QThread):
    image_update = pyqtSignal(QtGui.QImage)

    def __init__(self,parent=None):
        QThread.__init__(self,parent)

        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None

        self.frame_num = 0

        self.wBSN = 0
        self.wIDx = 2
        self.wIDy = 1

    def run(self):

        self.kf = KalmanFilter()

        # BaslerCamera
        end_flag, self.frame = True, getBasler()
        
        time.sleep(3)


        while end_flag:

            tn = datetime.datetime.now()
            time_delta = datetime.timedelta(hours=tn.hour,minutes=tn.minute,seconds=tn.second,microseconds=tn.microsecond)

            self.frame = cv2.resize(self.frame , (640,480))
            #self.gray_next = cv2.GaussianBlur(self.frame, (variable.kernel, variable.kernel), variable.sigma)
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            #現在の画像を動画に書き込み
            if (variable.flag == True) :
                variable.writer.write(self.frame)
            else:
                pass

            #特徴点を指定する
            if variable.detect_flag == True:
                image_operation3.onMouse(self,variable.click_x,variable.click_y,variable.radius)
                variable.detect_flag = False

            #右クリックで特徴点リセット
            if variable.flag == True:
                pass
            elif variable.reset_flag == True:
                variable.reset_flag = False
                self.features = None
                detection_x = None
                detection_y = None
                previous_x = None
                previous_y = None

            if variable.marker_flag != None:
                if detection_x or detection_y != None:
                    if(variable.marker_flag == 0):
                        for feature in self.features:
                                feature[0][1] -= variable.marker
                    elif(variable.marker_flag == 1):
                        for feature in self.features:
                                feature[0][1] += variable.marker
                    elif(variable.marker_flag == 2):
                        for feature in self.features:
                                feature[0][0] -= variable.marker
                    elif(variable.marker_flag == 3):
                        for feature in self.features:
                                feature[0][0] += variable.marker
                    else:
                        pass
                    variable.marker_flag = None
                else:
                    variable.marker_flag = None
            else:
                pass

            # ステージ座標取得
            pdwDatax = (ctypes.c_int32*7)()
            pdwDatay = (ctypes.c_int32*7)()
            ret = Pxep.PxepwAPI.RcmcwGetCounter(self.wBSN,self.wIDx,pdwDatax)
            ret = Pxep.PxepwAPI.RcmcwGetCounter(self.wBSN,self.wIDy,pdwDatay)
            stageX = pdwDatax[1]
            stageY = pdwDatay[1]

            dete_time = time.time()

            # 特徴点が登録されている場合にOpticalFlowを計算する
            if self.features is not None or (detection_x is not None and detection_y is not None):
                if  len(self.features) != 0:
                    features_prev = self.features
                    self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
                                                        self.gray_prev, \
                                                        self.gray_next, \
                                                        features_prev, \
                                                        None, \
                                                        winSize = (variable.radius, variable.radius), \
                                                        maxLevel = 3, \
                                                        criteria = CRITERIA, \
                                                        flags = 0)

                    # 有効な特徴点のみ残す
                    image_operation3.refreshFeatures(self)

                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        detection_x = int(feature[0][0])
                        detection_y = int(feature[0][1])

                    # KalmanFiter
                    predictd = self.kf.predict(detection_x, detection_y)
                    #cv2.circle(self.frame, (detection_x, detection_y), 10, (0, 255, 255), -1)
                    #cv2.circle(self.frame, (predictd[0], predictd[1]), 10, (255, 0, 0), 4)

                else:
                    image_operation3.onMouse(self,detection_x,detection_y,variable.radius)
                    #print("No Detection")

                if detection_x is not None and detection_y is not None and previous_x is not None and previous_y is not None:
                    if(variable.flag == True):

                        # 距離に応じた移動倍率の変更
                        # self.delta_d_flag = math.sqrt(pow((detection_x - previous_x),2)+pow((detection_y - previous_y),2))
                        # if self.delta_d_flag < variable.asobi:
                        #     self.m = 0.7
                        # elif self.delta_d_flag >= variable.asobi and self.delta_d_flag < 10:
                        #     self.m = 0.9
                        # elif self.delta_d_flag >= 10 and self.delta_d_flag < 20:
                        #     self.m = 1.1
                        # else:
                        #     self.m = 1.2

                        gap_x = variable.center_x-detection_x
                        gap_y = variable.center_y-detection_y

                        gap_x_Micro = gap_x*variable.PixeltoMicro
                        gap_y_Micro = gap_y*variable.PixeltoMicro

                        gap_x_Pulse = gap_x*variable.PixeltoMicro/variable.PulsetoMicro
                        gap_y_Pulse = gap_y*variable.PixeltoMicro/variable.PulsetoMicro

                        xPulse = int(float(stageX) + gap_x_Pulse*0.8)
                        yPulse = int(float(stageY) - gap_y_Pulse*0.8)

                        if ((abs(gap_x) <= variable.asobi and abs(gap_y) <= variable.asobi)):
                            pass
                        else:
                            ret = Pxep.PxepwAPI.RcmcwPulseOverride(self.wBSN,self.wIDx,xPulse)
                            ret = Pxep.PxepwAPI.RcmcwPulseOverride(self.wBSN,self.wIDy,yPulse)


                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), variable.radius*2, (255, 0, 255))
                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), variable.asobi*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (detection_x, detection_y), (0, 255, 125))

            # 表示
            self.frame = cv2.drawMarker(self.frame, position=(variable.center_x, variable.center_y), color=(255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
            mainImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            if (variable.flag == True):
                self.frame_num = self.frame_num +1

                delta_t = dete_time - prev_time
                delta_d = math.sqrt(pow((stageX-previous_stageX)*variable.PulsetoMicro,2)+pow((stageY-previous_stageY)*variable.PulsetoMicro,2))
                velocity = delta_d / delta_t
                Err = math.sqrt(pow(gap_x_Micro,2)+pow(gap_y_Micro,2))

                variable.log_file.write(str(self.frame_num)+","+
                                        str(time_delta.total_seconds())+","+
                                        str(stageX)+","+
                                        str(stageY)+","+
                                        str(delta_t)+","+
                                        str(delta_d)+","+
                                        str(velocity)+","+
                                        str(detection_x)+","+
                                        str(detection_y)+","+
                                        str(gap_x)+","+
                                        str(gap_y)+","+
                                        str(gap_x_Micro)+","+
                                        str(gap_y_Micro)+","+
                                        str(gap_x_Pulse)+","+
                                        str(gap_y_Pulse)+","+
                                        str(Err)+","+
                                        str(predictd[0])+","+
                                        str(predictd[1])+","+
                                        str(variable.radius)+","+
                                        str(variable.asobi)+"\n"
                                        )
            else:
                pass

            # 次のループ処理の準備
            previous_stageX = stageX
            previous_stageY = stageY
            previous_x = detection_x
            previous_y = detection_y
            prev_time = dete_time
            self.gray_prev = self.gray_next

            # BaslerCamera
            end_flag, self.frame = True, getBasler()


            # Track終了後 保存処理
            if variable.save_flag == True:
                print("Start Editing")

                variable.log_file.close()
                variable.writer.release()

                # Initialization
                self.frame_num = 0

                # Save
                s_time = open(START_TIME,"r")
                start_time = s_time.read()

                variable.param_file.write("datetime:\t"+str(variable.now)+"\n"+
                                          "fps:\t"+str(variable.fps)+"\n"+
                                          "size:\t"+str(variable.size)+"\n"+
                                          "center:\t("+str(variable.center_x)+","+str(variable.center_y)+")\n"
                                          "noudo:\t%\n"
                                          )

                if start_time != str(0):

                    CSV_file = open(variable.csv_path,"r")
                    csvReader = csv.reader(CSV_file)
                    csvHeader = next(csvReader)

                    for row in csvReader:
                        time_csv = float(row[1])
                        time_csv = str(time_csv)
                        if start_time < time_csv:
                            start_frame_num = row[0]
                            break
                        else:
                            start_frame_num = 0

                    variable.param_file.write("start_time:\t"+str(start_time)+"\n"+
                                              "start_frame_num:\t"+str(start_frame_num)+"\n"
                                              )

                    CSV_file.close()
                    s_time.close()
                    s_time = open(START_TIME,"w")
                    s_time.write(str(0))
                    s_time.close()

                    """csvReader = pandas.read_csv(variable.csv_path,encoding="shift-jis")
                    edit_cap = cv2.VideoCapture(variable.video_path)
                    if not edit_cap.isOpened():
                        print("読み込み失敗")
                        sys.exit()

                    n = 0
                    while(edit_cap.isOpened()):
                        is_image, edit_frame = edit_cap.read()

                        if is_image:
                            edit_detection_x = int(csvReader.iat[n,2])
                            edit_detection_y = int(csvReader.iat[n,3])
                            edit_radius = int(csvReader.iat[n,4])
                            edit_asobi = int(csvReader.iat[n,5])

                            edit_frame = cv2.circle(edit_frame, (edit_detection_x, edit_detection_y), edit_radius*2, (255, 0, 255))
                            edit_frame = cv2.circle(edit_frame, (edit_detection_x, edit_detection_y), edit_asobi*2, (0, 255, 255))
                            edit_frame = cv2.drawMarker(edit_frame, (edit_detection_x, edit_detection_y), (0, 255, 125))
                            edit_frame = cv2.drawMarker(edit_frame, position=(variable.center_x, variable.center_y), color=(255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)

                            try:
                                variable.writerD.write(edit_frame)
                            except:
                                pass
                        else:
                            edit_cap.release()
                            variable.writerD.release()
                            break
                        n += 1
                        """
                else:
                    s_time.close()
                    #variable.writerD.release()

                variable.writerD.release()
                variable.param_file.close()

                # Transfer
                MPC_path = open(MAINPC_PATH,"r")
                mainPC_path = MPC_path.read()
                if mainPC_path != str(0):
                    shutil.move(variable.file_path, mainPC_path)
                    MPC_path.close()
                    MPC_path = open(MAINPC_PATH,"w")
                    MPC_path.write(str(0))
                    MPC_path.close()
                else:
                    MPC_path.close()

                print("Editing Completed")
                variable.save_flag = False

            else:
                pass

            # numpy 配列を QImage に変換する。
            mainQImage = QtGui.QImage(mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            # シグナルを送出する。
            self.image_update.emit(mainQImage)



def getBasler():
    grabResult = basler.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = basler.converter.Convert(grabResult)
        img = image.GetArray()
    return img


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



# 条件記入用.txtファイル書き込みと保存
# 測定時間（経過時間）表示
# カメラ視野変更

# 英語化
