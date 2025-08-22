from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow, QDialog, QMessageBox,QGraphicsPixmapItem
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen

import cv2, time, datetime, os, math, csv, sys, shutil, queue, ctypes
import numpy as np

from pypylon import pylon
from multiprocessing import Process, Queue

from util import basler, Pxep
from util.utils import OpticalFlow, KalmanFilter, getBasler, setPxep, write_log


##################################################
## ==> START: shared variable
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
    center_x = 320
    center_y = 240
    radius = 25#16
    asobi = 1
    marker = 5
    scale = 38 #38.85873036
    ## 速度パラメータ
    StartVelocity = 3000#600
    HighVelocity = 15000#6000
    AccelerationTime = 30# 100
    DecelerationTime = 30#80
    AccelerationS = 40
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
## ==> END



##################################################
## ==> START: Main Window
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi(r"F:\tracking stage\Gui\MainWindow_2.ui",self)
        

        scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.graphicsView.installEventFilter(self)

        self.spinBox_X.setValue(variable.center_x)
        self.spinBOx_Y.setValue(variable.center_y)
        self.spinBox_Radius.setValue(variable.radius)
        self.spinBox_Play.setValue(variable.asobi)
        self.doubleSpinBox_PulseRatio.setValue(variable.scale)

        self.spinBox_Pulse.setValue(variable.pulse)
        self.spinBox_Speed.setValue(variable.StartVelocity)
        self.spinBox_HighSpeed.setValue(variable.HighVelocity)
        self.spinBox_Utime.setValue(variable.AccelerationTime)
        self.spinBox_Dtime.setValue(variable.DecelerationTime)
        self.spinBox_SshapeRatio.setValue(variable.AccelerationS)
        self.spinBox_Override.setValue(variable.OverVelocity)
        
        self.spinBox_Mmodify.setValue(variable.marker)
        

        self.spinBox_X.valueChanged.connect(self.valuechange_1)
        self.spinBOx_Y.valueChanged.connect(self.valuechange_2)
        self.spinBox_Radius.valueChanged.connect(self.valuechange_3)
        self.spinBox_Play.valueChanged.connect(self.valuechange_4)
        self.doubleSpinBox_PulseRatio.valueChanged.connect(self.valuechange2_1)

        self.spinBox_Pulse.valueChanged.connect(self.valuechange_12)
        self.spinBox_Speed.valueChanged.connect(self.valuechange_6)
        self.spinBox_HighSpeed.valueChanged.connect(self.valuechange_7)
        self.spinBox_Utime.valueChanged.connect(self.valuechange_8)
        self.spinBox_Dtime.valueChanged.connect(self.valuechange_9)
        self.spinBox_SshapeRatio.valueChanged.connect(self.valuechange_10)
        self.spinBox_Override.valueChanged.connect(self.valuechange_11)
        
        self.spinBox_Mmodify.valueChanged.connect(self.valuechange_5)

        

        self.pushButton_Close.pressed.connect(self.CloseMainWindow)
        self.pushButton_Ready.pressed.connect(self.Ready)
        self.pushButton_TrackON.pressed.connect(self.TrackON)
        self.pushButton_TrackOFF.pressed.connect(self.TrackOFF)
        self.pushButton_ServoON.pressed.connect(self.ServoON)
        self.pushButton_ServoOFF.pressed.connect(self.ServoOFF)
        self.pushButton_SetParam.pressed.connect(self.SetVelocity)
        self.pushButton_Up.pressed.connect(self.Up_button)
        self.pushButton_Down.pressed.connect(self.Down_button)
        self.pushButton_Left.pressed.connect(self.Left_button)
        self.pushButton_Right.pressed.connect(self.Right_button)
        self.pushButton_UpLeft.pressed.connect(self.UpperLeft_button)
        self.pushButton_UpRight.pressed.connect(self.UpperRight_button)
        self.pushButton_LowLeft.pressed.connect(self.LowerLeft_button)
        self.pushButton_LowRight.pressed.connect(self.LowerRight_button)
        self.pushButton_Stop.pressed.connect(self.Stop)

        self.pushButton_SaveTxt.pressed.connect()
        self.pushButton_PixelCount.pressed.connect()
        self.pushButton_ResetTimer.pressed.connect()

        

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
        variable.center_x           = self.spinBox_X.value()
    def valuechange_2(self):
        variable.center_y           = self.spinBOx_Y.value()
    def valuechange_3(self):
        variable.radius             = self.spinBox_Radius.value()
    def valuechange_4(self):
        variable.asobi              = self.spinBox_Play.value()
    def valuechange_5(self):
        variable.marker             = self.spinBox_Mmodify.value()
    def valuechange_6(self):
        variable.StartVelocity      = self.spinBox_Speed.value()
    def valuechange_7(self):
        variable.HighVelocity       = self.spinBox_HighSpeed.value()
    def valuechange_8(self):
        variable.AccelerationTime   = self.spinBox_Utime.value()
    def valuechange_9(self):
        variable.DecelerationTime   = self.spinBox_Dtime.value()
    def valuechange_10(self):
        variable.AccelerationS      = self.spinBox_SshapeRatio.value()
    def valuechange_11(self):
        variable.OverVelocity       = self.spinBox_Override.value()
    def valuechange_12(self):
        variable.pulse              = self.spinBox_Pulse.value()

    def valuechange2_1(self):
        variable.scale              = self.doubleSpinBox_PulseRatio.value()


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
        write_log(variable.log_file,init=True)

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

        self.pushButton_Ready.setEnabled(False)
        self.pushButton_TrackON.setEnabled(True)

        self.pushButton_Ready.setStyleSheet("background-color: red")


    def TrackON(self):
        variable.flag = True
        variable.save_flag = False

        self.ServoON()

        # self.spinBOx_Y.setEnabled(True)
        # self.spinBox_Speed.setEnabled(False)
        # self.spinBox_HighSpeed.setEnabled(False)
        # self.spinBox_Utime.setEnabled(False)

        self.pushButton_Marker.setChecked(True)

        self.pushButton_Ready.setEnabled(False)
        self.pushButton_TrackON.setEnabled(False)
        self.pushButton_TrackOFF.setEnabled(True)
        # self.pushButton_ServoON.setEnabled(False)
        # self.pushButton_ServoOFF.setEnabled(False)
        # self.pushButton_SetParam.setEnabled(False)
        # self.pushButton_7.setEnabled(False)
        # self.pushButton_Stop.setEnabled(False)
        # self.pushButton_Ready7.setEnabled(False)
        # self.pushButton_Ready8.setEnabled(False)
        # self.pushButton_Ready9.setEnabled(False)
        # self.pushButton_TrackON0.setEnabled(False)

        self.pushButton_TrackON.setStyleSheet("background-color: red")

    def TrackOFF(self):
        variable.flag = False
        variable.save_flag = True

        # self.spinBOx_Y.setEnabled(True)
        # self.spinBox_Speed.setEnabled(True)
        # self.spinBox_HighSpeed.setEnabled(True)
        # self.spinBox_Utime.setEnabled(True)

        self.pushButton_Marker.setChecked(False)

        self.pushButton_Ready.setEnabled(True)
        self.pushButton_TrackON.setEnabled(False)
        self.pushButton_TrackOFF.setEnabled(False)
        # self.pushButton_ServoON.setEnabled(True)
        # self.pushButton_ServoOFF.setEnabled(True)
        # self.pushButton_SetParam.setEnabled(True)
        # self.pushButton_7.setEnabled(True)
        # self.pushButton_Stop.setEnabled(True)
        # self.pushButton_Ready7.setEnabled(True)
        # self.pushButton_Ready8.setEnabled(True)
        # self.pushButton_Ready9.setEnabled(True)
        # self.pushButton_TrackON0.setEnabled(True)

        self.pushButton_Ready.setStyleSheet("background-color: white")
        self.pushButton_TrackON.setStyleSheet("background-color: white")

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
        self.pushButton_ServoON.setEnabled(False)
        self.pushButton_ServoOFF.setEnabled(True)
        self.pushButton_ServoON.setStyleSheet("background-color: red")
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDx,1,0)
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDy,1,0)

    def ServoOFF(self):
        self.pushButton_ServoON.setEnabled(True)
        self.pushButton_ServoOFF.setEnabled(False)
        self.pushButton_ServoON.setStyleSheet("background-color: white")
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
        if self.pushButton_Marker.isChecked():
            variable.marker_flag = 0
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,-variable.pulse)
    def Down_button(self):
        if self.pushButton_Marker.isChecked():
            variable.marker_flag = 1
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,variable.pulse)
    def Right_button(self):
        if self.pushButton_Marker.isChecked():
            variable.marker_flag = 3
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,-variable.pulse)
    def Left_button(self):
        if self.pushButton_Marker.isChecked():
            variable.marker_flag = 2
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,variable.pulse)
    def UpperRight_button(self):
        if self.pushButton_Marker.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,-variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,-variable.pulse)
    def UpperLeft_button(self):
        if self.pushButton_Marker.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,-variable.pulse)
    def LowerRight_button(self):
        if self.pushButton_Marker.isChecked():
            pass
        else:
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,-variable.pulse)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,variable.pulse)
    def LowerLeft_button(self):
        if self.pushButton_Marker.isChecked():
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

## ==> END


##################################################
## ==> START: CapThread class
class CapThread(QThread):
    image_update = pyqtSignal(QtGui.QImage)

    def __init__(self,parent=None):
        QThread.__init__(self,parent)

        self.opt = OpticalFlow()
        self.kf = KalmanFilter()

        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

        self.MAINPC_PATH = 'log/save_path.txt'
        self.START_TIME = 'log/start_time.txt'

        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None

        self.wBSN = 0
        self.wIDx = 2
        self.wIDy = 1
        setPxep(self.wBSN,self.wIDx,self.wIDy)

    def run(self):
        # BaslerCamera
        end_flag, self.frame = True, getBasler()        
        time.sleep(1)

        frame_num = 0
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
                self.opt.onMouse(self,variable.click_x,variable.click_y,variable.radius)
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
                                                        criteria = self.CRITERIA, \
                                                        flags = 0)

                    # 有効な特徴点のみ残す
                    self.opt.refreshFeatures(self)

                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        detection_x = int(feature[0][0])
                        detection_y = int(feature[0][1])

                    # KalmanFiter
                    predictd = self.kf.predict(detection_x, detection_y)
                    #cv2.circle(self.frame, (detection_x, detection_y), 10, (0, 255, 255), -1)
                    #cv2.circle(self.frame, (predictd[0], predictd[1]), 10, (255, 0, 0), 4)

                else:
                    self.opt.onMouse(self,detection_x,detection_y,variable.radius)
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

            if variable.flag == True:
                frame_num = frame_num +1

                delta_t = dete_time - prev_time
                delta_d = math.sqrt(pow((stageX-previous_stageX)*variable.PulsetoMicro,2)+pow((stageY-previous_stageY)*variable.PulsetoMicro,2))
                velocity = delta_d / delta_t
                Err = math.sqrt(pow(gap_x_Micro,2)+pow(gap_y_Micro,2))

                write_log(variable.log_file,frame_num,time_delta,stageX,stageY,delta_t,delta_d, velocity, detection_x, detection_y,gap_x, gap_y, gap_x_Micro, gap_y_Micro, gap_x_Pulse, gap_y_Pulse, Err, predictd, variable.radius, variable.asobi, init=False)


            # 次のループ処理の準備
            previous_stageX = stageX
            previous_stageY = stageY
            previous_x = detection_x
            previous_y = detection_y
            prev_time = dete_time
            self.gray_prev = self.gray_next

            # BaslerCamera
            end_flag, self.frame = True, getBasler()
            
            # numpy 配列を QImage に変換する。
            mainQImage = QtGui.QImage(mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            # シグナルを送出する。
            self.image_update.emit(mainQImage)


            # Track終了後 保存処理
            if variable.save_flag == True:
                print("Start Editing")

                variable.log_file.close()
                variable.writer.release()

                # Initialization
                frame_num = 0

                # Save
                s_time = open(self.START_TIME,"r")
                start_time = s_time.read()

                variable.param_file.write("datetime:\t"+str(variable.now)+"\n"+
                                          "fps:\t"+str(variable.fps)+"\n"+
                                          "size:\t"+str(variable.size)+"\n"+
                                          "center:\t("+str(variable.center_x)+","+str(variable.center_y)+")\n"
                                          "noudo:\t3.0%\n"
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
                    s_time = open(self.START_TIME,"w")
                    s_time.write(str(0))
                    s_time.close()

                else:
                    s_time.close()

                variable.writerD.release()
                variable.param_file.close()

                # Transfer
                MPC_path = open(self.MAINPC_PATH,"r")
                mainPC_path = MPC_path.read()
                if mainPC_path != str(0):
                    shutil.move(variable.file_path, mainPC_path)
                    MPC_path.close()
                    MPC_path = open(self.MAINPC_PATH,"w")
                    MPC_path.write(str(0))
                    MPC_path.close()
                else:
                    MPC_path.close()

                print("Editing Completed")
                variable.save_flag = False
## ==> END



##################################################
## ==> START: define(for getting image from basler camera)
def gui():
    None
## ==> END

##################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

