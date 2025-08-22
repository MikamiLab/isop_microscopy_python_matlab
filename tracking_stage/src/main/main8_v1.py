# -*- coding: utf-8 -*-
#from fileinput import filename
from re import S
from pypylon import pylon
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow, QDialog, QMessageBox,QGraphicsPixmapItem
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2, time ,datetime ,os ,math ,glob ,csv ,pandas ,sys, shutil
import serial, serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt

import common.basler as basler
import common.image_operation3 as image_operation3

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
VIDEO = r"rec.avi"

MAINPC_PATH = r"F:\tracking stage\log\save_path.txt"
START_TIME = r"F:\tracking stage\log\start_time.txt"

delta_x0 = 0
delta_y0 = 0


class variable():
    # profile
    fps = 100
    size = (640, 480)
    pu_m = 0.01590033
    pi_m = 0.616291862

    now = None
    today = None

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

    sdetx = 0
    sdety = 0

    # Date
    center_x = 320
    center_y = 240
    radius = 16
    asobi = 2
    scale = 38 #38.85873036
    Min = 1
    Max = 30000
    acce = 10
    pulse = 5000 #分割数250,2040x1530,15000 / 250,1020x765,5000
    marker = 5

    kernel = 1 #7
    sigma = 1 #5

    click_x = None
    click_y = None

    # flag
    flag = False
    flag_detect = False
    flag_VH = True
    flag_dia = False
    track_flag = False
    save_flag = False

    marker_flag = None

    detect_flag = False
    reset_flag = True


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


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui = uic.loadUi(r"F:\tracking stage\Gui\prototype.ui",self)

        scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(scene)
        self.graphicsView.installEventFilter(self)

        self.spinBox_1.setValue(variable.center_x)
        self.spinBox_2.setValue(variable.center_y)
        self.spinBox_3.setValue(variable.radius)
        self.spinBox_4.setValue(variable.asobi)
        self.spinBox_5.setValue(variable.scale)
        self.spinBox_6.setValue(variable.Min)
        self.spinBox_7.setValue(variable.Max)
        self.spinBox_8.setValue(variable.acce)
        self.spinBox_9.setValue(variable.pulse)
        self.spinBox_10.setValue(variable.marker)

        self.spinBox_1.valueChanged.connect(self.valuechange)
        self.spinBox_2.valueChanged.connect(self.valuechange_2)
        self.spinBox_3.valueChanged.connect(self.valuechange_3)
        self.spinBox_4.valueChanged.connect(self.valuechange_4)
        self.spinBox_5.valueChanged.connect(self.valuechange_5)
        self.spinBox_6.valueChanged.connect(self.valuechange_6)
        self.spinBox_7.valueChanged.connect(self.valuechange_7)
        self.spinBox_8.valueChanged.connect(self.valuechange_8)
        self.spinBox_9.valueChanged.connect(self.valuechange_9)
        self.spinBox_10.valueChanged.connect(self.valuechange_10)

        self.pushButton.pressed.connect(self.Ready)
        self.pushButton_2.pressed.connect(self.track)
        self.pushButton_3.pressed.connect(self.trackoff)
        self.pushButton_4.clicked.connect(self.manual)
        self.pushButton_5.clicked.connect(self.manualoff)
        self.pushButton_6.pressed.connect(self.close)
        self.pushButton_7.pressed.connect(self.set)
        self.pushButton_8.pressed.connect(self.up)
        self.pushButton_9.pressed.connect(self.down)
        self.pushButton_10.pressed.connect(self.left)
        self.pushButton_11.pressed.connect(self.right)
        self.pushButton_12.pressed.connect(self.m_up)
        self.pushButton_13.pressed.connect(self.m_down)
        self.pushButton_14.pressed.connect(self.m_left)
        self.pushButton_15.pressed.connect(self.m_right)

        try:
            self._item = QtWidgets.QGraphicsPixmapItem()
            scene.addItem(self._item)
            thread = CapThread(self)
            thread.image_update.connect(self.update_mainimage)
            thread.start()
        except Exception as error :
            print("err")


        global ser
        try:
            ser = serial.Serial(port=list(serial.tools.list_ports.grep("0403:6001"))[0][0], baudrate=38400, timeout=0.05)
        except IndexError:
            QMessageBox.information(self,"Message","No Motor Connection")
            sys.exit()
        ser.write(("D:WS"+str(variable.Min)+"F"+str(variable.Max)+"R"+str(variable.acce)+"S"+str(variable.Min)+"F"+str(variable.Max)+"R"+str(variable.acce)+"\r\n").encode())
        ser.write(b"S:1250\r\n")
        ser.write(b"S:2250\r\n")
        self.pushButton_4.setStyleSheet("background-color: red")
        ser.write(b"C:W0\r\n")

    def valuechange(self):
        variable.center_x = self.spinBox_1.value()
    def valuechange_2(self):
        variable.center_y = self.spinBox_2.value()
    def valuechange_3(self):
        variable.radius = self.spinBox_3.value()
    def valuechange_4(self):
        variable.asobi = self.spinBox_4.value()
    def valuechange_5(self):
        variable.scale = self.spinBox_5.value()
    def valuechange_6(self):
        variable.Min = self.spinBox_6.value()
    def valuechange_7(self):
        variable.Max = self.spinBox_7.value()
    def valuechange_8(self):
        variable.acce = self.spinBox_8.value()
    def valuechange_9(self):
        variable.pulse = self.spinBox_9.value()
    def valuechange_10(self):
        variable.marker = self.spinBox_10.value()

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

    def Ready(self):
        now = datetime.datetime.now()
        today = datetime.date.today()
        variable.now = now.strftime("%Y%m%d%H%M%S")
        variable.today = str(today)
        variable.file_path = os.getcwd()+ "/log/"+variable.today+"/"+variable.now

        try:
            os.mkdir(os.getcwd()+ "/log")
        except FileExistsError:
            pass
        try:
            os.mkdir(os.getcwd()+ "/log/"+variable.today)
        except FileExistsError:
            pass
        try:
            os.mkdir(os.getcwd()+ "/log/"+variable.today+"/"+variable.now)
        except FileExistsError:
            pass

        variable.csv_path = variable.file_path+"/log.csv"
        variable.log_file = open(variable.csv_path,"a")
        variable.log_file.write("frame,time,x(pixel),y(pixel),radius(pixel),asobi(pixel),Δt(s),Δd(μm),track_v(μm/s),err(μm),m,gap_x(pixel),gap_y(pixel),kal_x,kal_y\n")

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
            pass

        self.pushButton_2.setEnabled(True)


    def track(self):
        variable.flag = True
        variable.track_flag = True
        variable.save_flag = False

        self.spinBox_1.setEnabled(True)
        self.spinBox_2.setEnabled(True)
        self.spinBox_6.setEnabled(False)
        self.spinBox_7.setEnabled(False)
        self.spinBox_8.setEnabled(False)

        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setStyleSheet("background-color: red")
        self.pushButton_3.setEnabled(True)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.pushButton_10.setEnabled(False)
        self.pushButton_11.setEnabled(False)


    def trackoff(self):
        variable.flag = False
        variable.track_flag = False
        variable.save_flag = True

        self.spinBox_1.setEnabled(True)
        self.spinBox_2.setEnabled(True)
        self.spinBox_6.setEnabled(True)
        self.spinBox_7.setEnabled(True)
        self.spinBox_8.setEnabled(True)

        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setStyleSheet("background-color: none")
        self.pushButton_3.setEnabled(False)
        self.pushButton_6.setEnabled(True)
        self.pushButton_7.setEnabled(True)
        self.pushButton_8.setEnabled(True)
        self.pushButton_9.setEnabled(True)
        self.pushButton_10.setEnabled(True)
        self.pushButton_11.setEnabled(True)



    def quit(self):

        """# files = glob.glob(os.getcwd() + "/save/*")
        # for file in files:
        #     if(os.path.getsize(file) < 6000):
        #         os.remove(file)"""
        if(variable.flag == True):
            self.trackoff()
        else:
            pass
        app.quit()

    def manual(self):
        self.pushButton_4.setStyleSheet("background-color: red")
        ser.write(b"C:W0\r\n")
    def manualoff(self):
        self.pushButton_4.setStyleSheet("background-color: none")
        ser.write(b"C:W1\r\n")

    def set(self):
        print("set")
        ser.write(("D:WS"+str(variable.Min)+"F"+str(variable.Max)+"R"+str(variable.acce)+"S"+str(variable.Min)+"F"+str(variable.Max)+"R"+str(variable.acce)+"\r\n").encode())


    def up(self):
        ser.write(("M:2+P"+ str(variable.pulse) + "\r\n").encode())
        ser.write(b"G\r\n")
    def down(self):
        ser.write(("M:2-P"+ str(variable.pulse) + "\r\n").encode())
        ser.write(b"G\r\n")
    def right(self):
        ser.write(("M:1+P"+ str(variable.pulse) + "\r\n").encode())
        ser.write(b"G\r\n")
    def left(self):
        ser.write(("M:1-P"+ str(variable.pulse) + "\r\n").encode())
        ser.write(b"G\r\n")

    def m_up(self):
        variable.marker_flag = 0
    def m_down(self):
        variable.marker_flag = 1
    def m_left(self):
        variable.marker_flag = 2
    def m_right(self):
        variable.marker_flag = 3

    def upperRight(self, delta_x0, delta_y0, m):
        ser.write(("M:W+P"+str(math.floor(variable.scale*abs(delta_x0)*m))+"+P"+str(math.floor(variable.scale*abs(delta_y0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def upperLeft(self, delta_x0, delta_y0, m):
        ser.write(("M:W-P"+str(math.floor(variable.scale*abs(delta_x0)*m))+"+P"+str(math.floor(variable.scale*abs(delta_y0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def lowerRight(self, delta_x0, delta_y0, m):
        ser.write(("M:W+P"+str(math.floor(variable.scale*abs(delta_x0)*m))+"-P"+str(math.floor(variable.scale*abs(delta_y0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def lowerLeft(self, delta_x0, delta_y0, m):
        ser.write(("M:W-P"+str(math.floor(variable.scale*abs(delta_x0)*m))+"-P"+str(math.floor(variable.scale*abs(delta_y0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def Up(self, delta_y0, m):
        ser.write(("M:2+P"+str(math.floor(variable.scale*abs(delta_y0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def Down(self, delta_y0, m):
        ser.write(("M:2-P"+str(math.floor(variable.scale*abs(delta_y0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def Right(self, delta_x0, m):
        ser.write(("M:1+P"+str(math.floor(variable.scale*abs(delta_x0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")
    def Left(self, delta_x0, m):
        ser.write(("M:1-P"+str(math.floor(variable.scale*abs(delta_x0)*m))+"\r\n").encode())
        ser.write(b"G\r\n")


class CapThread(QThread):
    image_update = pyqtSignal(QtGui.QImage)

    rec_x = 0
    rec_y = 0
    srec_x = 0
    srec_y = 0

    def run(self):

        #track開始時に保存先ディレクトリをつくる
        try:
            os.mkdir(os.getcwd()+ "/log")
        except FileExistsError:
            None

        self.kf = KalmanFilter()

        #previous_approxCurves = None #消して不具合がないか確認する
        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None

        self.frame_num = 0

        # BaslerCamera
        end_flag, self.frame = True, getBasler()

        # # VIDEO
        # cap = cv2.VideoCapture(0)
        # end_flag, self.frame = cap.read()
        # time.sleep(3)

        self.gray_prev = cv2.resize(self.frame , (640,480))
        self.gray_prev = cv2.GaussianBlur(self.gray_prev, (variable.kernel, variable.kernel), variable.sigma)
        self.gray_prev = cv2.cvtColor(self.gray_prev, cv2.COLOR_BGR2GRAY)

        detection_x = 0
        detection_y = 0
        previous_x = 0
        previous_y = 0

        dete_time = 0
        prev_time = 0

        while end_flag:

            dete_time = time.time()
            tn = datetime.datetime.now()
            time_delta = datetime.timedelta(hours=tn.hour,minutes=tn.minute,seconds=tn.second,microseconds=tn.microsecond)

            self.frame = cv2.resize(self.frame , (640,480))
            self.gray_next = cv2.GaussianBlur(self.frame, (variable.kernel, variable.kernel), variable.sigma)
            self.gray_next = cv2.cvtColor(self.gray_next, cv2.COLOR_BGR2GRAY)

            time1 = time.time()
            #現在の画像を動画に書き込み
            if (variable.flag == True) :
                try:
                    variable.writer.write(self.frame)
                except:
                    pass
            time2 = time.time()
            print(time2-time1)

            """# ステージ座標取得
            # ser.write(b"Q:\r\n")
            # line = ser.readline().decode("utf-8")
            # sdetect_x1 = (line[0:1].replace(" ", ""))
            # sdetection_x = int(line[1:10].replace(" ", ""))
            # sdetect_y1 = (line[11:12].replace(" ", ""))
            # sdetection_y = int(line[12:21].replace(" ", ""))
            # if sdetect_x1 == "-":
            #     sdetection_x = -sdetection_x
            # else:
            #     pass
            # if sdetect_y1 == "-":
            #     sdetection_y = -sdetection_y
            # else:
            #     pass
            # variable.sdetx = sdetection_x
            # variable.sdety = sdetection_y
            """

            #特徴点を指定する
            if(variable.detect_flag == True):
                image_operation3.onMouse(self,variable.click_x,variable.click_y,variable.radius)
                variable.detect_flag = False
                self.rec_x = 0
                self.rec_y = 0
                self.srec_x = 0
                self.srec_y = 0

            #右クリックで特徴点リセット
            if(variable.reset_flag == True):
                self.features = None
                variable.reset_flag = False
                detection_x = None
                detection_y = None
                previous_x = None
                previous_y = None
                delta_x = 0
                delta_y = 0

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
                    cv2.circle(self.frame, (predictd[0], predictd[1]), 10, (255, 0, 0), 4)

                else:
                    image_operation3.onMouse(self,detection_x,detection_y,variable.radius)
                    #print("No Detection")

                if detection_x == None or previous_x == None:
                    delta_x = 0
                else:
                    delta_x = detection_x - previous_x

                if detection_y == None or previous_y == None:
                    delta_y = 0
                else:
                    delta_y = detection_y - previous_y

                if detection_x is not None and detection_y is not None and previous_x is not None and previous_y is not None:
                    if(variable.track_flag == True):

                        self.delta_d_flag = math.sqrt(pow((detection_x - previous_x),2)+pow((detection_y - previous_y),2))

                        if self.delta_d_flag < variable.asobi:
                            self.m = 0.8
                        elif self.delta_d_flag >= variable.asobi and self.delta_d_flag < 10:
                            self.m = 1
                        elif self.delta_d_flag >= 10 and self.delta_d_flag < 20:
                            self.m = 1.2
                        else:
                            self.m = 1.4

                        gap_x = variable.center_x-detection_x
                        gap_y = variable.center_y-detection_y

                        if ((abs(gap_x) <= variable.asobi and abs(gap_y) <= variable.asobi)) and (variable.flag == True):
                            pass
                        elif (gap_x) < -variable.asobi and (gap_y) > variable.asobi and (variable.flag == True):
                            MainWindow.upperRight(self, gap_x, gap_y, self.m)
                        elif (gap_x) < -variable.asobi and (gap_y) < -variable.asobi and (variable.flag == True):
                            MainWindow.lowerRight(self, gap_x, gap_y, self.m)
                        elif (gap_x) > variable.asobi and (gap_y) > variable.asobi and (variable.flag == True):
                            MainWindow.upperLeft(self, gap_x, gap_y, self.m)
                        elif (gap_x) > variable.asobi and (gap_y) < -variable.asobi and (variable.flag == True):
                            MainWindow.lowerLeft(self, gap_x, gap_y, self.m)
                        elif abs(gap_x) <= variable.asobi and (gap_y) > variable.asobi and (variable.flag == True):
                            MainWindow.Up(self, gap_y, self.m)
                        elif abs(gap_x) <= variable.asobi and (gap_y) < -variable.asobi and (variable.flag == True):
                            MainWindow.Down(self, gap_y, self.m)
                        elif (gap_x) < -variable.asobi and abs(gap_y) <= variable.asobi and (variable.flag == True):
                            MainWindow.Right(self, gap_x, self.m)
                        elif (gap_x) > variable.asobi and abs(gap_y) <= variable.asobi and (variable.flag == True):
                            MainWindow.Left(self, gap_x, self.m)
                        else:
                            print("err\n")


                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), variable.radius*2, (255, 0, 255))
                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), variable.asobi*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (detection_x, detection_y), (0, 255, 125))


            # 表示
            self.frame = cv2.drawMarker(self.frame, position=(variable.center_x, variable.center_y), color=(255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
            mainImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


            if (variable.flag == True) or (variable.track_flag == True) :
                self.frame_num = self.frame_num +1
                #ループ時間
                t = dete_time - prev_time
                #フレーム間の移動距離
                d = math.sqrt(pow((detection_x-previous_x)*variable.pi_m,2)+pow((detection_y-previous_y)*variable.pi_m,2))
                #マーカーの移動速度
                if float(t) != 0:
                    track_v = d / float(t)
                else:
                    track_v = 0

                """# ステージの移動速度
                # if stage_flag == True:
                #     if abs(gap_x) >= abs(gap_y):
                #         stage_t = math.sqrt(float((4*abs(gap_x)*self.m*pu_m))/acce)
                #         stage_v = (low_speed*pu_m + acce*stage_t/4)
                #     elif abs(gap_x) < abs(gap_y):
                #         stage_t = math.sqrt(float((4*abs(gap_y)*self.m*pu_m))/acce)
                #         stage_v = (low_speed*pu_m + acce*stage_t/4)
                # else:
                #     stage_t = 0
                #     stage_v = 0
                #     stage_flag = True
                # # 線虫の絶対速度
                # v = abs(track_v-stage_v)"""

                #中心とのずれ
                Err = math.sqrt(pow((gap_x)*variable.pi_m,2)+pow((gap_y)*variable.pi_m,2))

                # log_file.write
                variable.log_file.write(str(self.frame_num)+","+str(time_delta.total_seconds())+","+str(detection_x)+","+str(detection_y)+","
                    +str(variable.radius)+","+str(variable.asobi)+","+str(t)+","+str(d)+","+str(track_v)+","+str(Err)+","+str(self.m)+","
                    +str(gap_x)+","+str(gap_y)+","+str(predictd[0])+","+str(predictd[1])+"\n")

                self.rec_x = self.rec_x + delta_x
                self.rec_y = self.rec_y + delta_y
            else:
                pass

            # 次のループ処理の準備
            previous_x = detection_x
            previous_y = detection_y
            prev_time = dete_time
            self.gray_prev = self.gray_next

            # BaslerCamera
            end_flag, self.frame = True, getBasler()

            # # VIDEO
            # end_flag, self.frame = cap.read()


            # VIDEO終了処理
            if end_flag:
                pass
            else:
                variable.save_flag == True

            # Track終了後 保存処理
            if variable.save_flag == True:
                print("Start Editing")
                variable.save_flag = False

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
                                    "center:\t("+str(variable.center_x)+","+str(variable.center_y)+")\n"+
                                    "low_speed:\t"+str(variable.Min)+"\n"+
                                    "top_speed:\t"+str(variable.Max)+"\n"+
                                    "acce_time:\t"+str(variable.acce)+"\n")

                if start_time != str(0):

                    CSV_file = open(variable.csv_path,"r")
                    csvReader = csv.reader(CSV_file)
                    csvHeader = next(csvReader)

                    for row in csvReader:
                        time_csv = float(row[1])#-0.15
                        time_csv = str(time_csv)
                        if start_time < time_csv:
                            start_frame_num = row[0]
                            break
                        else:
                            start_frame_num = 0

                    variable.param_file.write("start_time:\t"+str(start_time)+"\n"+
                                                "start_frame_num:\t"+str(start_frame_num)+"\n")

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

                # Edit

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



"""# 速度グラフ 生成
csv_file = open(log_name,"r")
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
"""

"""# 散布図 生成
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
plt.close()
"""