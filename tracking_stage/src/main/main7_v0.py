from re import S
from PyQt5 import QtWidgets, uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox,QGraphicsPixmapItem
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from pypylon import pylon
from matplotlib import pyplot as plt
import serial,serial.tools.list_ports
import os,cv2,time,datetime,sys,csv,math,pandas
import numpy as np

import common.basler as basler
import common.image_operation2 as optical
#from common import basler
#from tracking import template,optical

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


class variable():

    #//////////////////////////
    # variables
    #//////////////////////////
    click_x = None
    click_y = None

    temp = None

    centerx,centery = 320,640
    markersize = 8
    mergin = 0
    magnification = 1
    min, max = 1000, 2000
    acce_time = 10

    stagemove = 1000
    markermove = 2

    roi_x, roi_y, roi_w, roi_h = 20,20,10,10
    sp, ep = [0,0], [0,0]

    marker_ctl = 0

    fps = 100
    size = (640,480)

    #//////////////////////////
    # flags
    #//////////////////////////
    detect_flag = False
    reset_flag = True

    #getimg_method = None
    basler_flag = False
    webcam_flag = False
    vidcap_flag = False

    #track_method = None
    template_flag = False
    optical_flag = False
    ESM_flag = False
    KCF_flag = False

    track_flag = False
    rec_flag = False


    #//////////////////////////
    # path
    #//////////////////////////
    now = datetime.datetime.now()
    csv_path = None
    video_path = None

    
#--------------------------------------------------------
# LogThread
#--------------------------------------------------------
class LogThread(QThread):
    try:
       os.mkdir(os.getcwd()+ "/log/"+variable.now.strftime("%Y%m%d%H%M%S"))
    except FileExistsError:
       pass
    
    log_csv_name = variable.now.strftime(os.getcwd()+"/log/"+"%Y%m%d%H%M%S/")+"log.csv"
    variable.csv_path = log_csv_name
    log_file = open(log_csv_name,"a")
    log_file.write("0,1,2,3,4,5,6,7,8,9,10\n")

    log_path_name = variable.now.strftime(os.getcwd()+"/log/"+"%Y%m%d%H%M%S/")+"log_path.txt"
    log_path = open(log_path_name,"a")
    log_path.write(log_csv_name)
    log_path.close()

    parameter_name = variable.now.strftime(os.getcwd()+"/log/"+"%Y%m%d%H%M%S/")+"parameter.txt"
    parameter_file = open(parameter_name,"a")
    parameter_file.write("fps:\t"+str(variable.fps)+"\n"+
                         "pixsel:\t"+str(variable.size)+"\n"+
                         "low_speed:\t"+str(variable.min)+"\n"+
                         "top_speed:\t"+str(variable.max)+"\n"+
                         "acce_time:\t"+str(variable.acce_time)+"\n")
    parameter_file.close()


#--------------------------------------------------------
# RecordThread
#--------------------------------------------------------
class RecThread(QThread):
    try:
       os.mkdir(os.getcwd()+ "/log/"+variable.now.strftime("%Y%m%d%H%M%S"))
    except FileExistsError:
       pass

    filename = variable.now.strftime(os.getcwd()+"/log/"+"%Y%m%d%H%M%S/")+"rec.avi"
    variable.video_path = filename
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    writer = cv2.VideoWriter(filename, fourcc, variable.fps, variable.size)


#--------------------------------------------------------
# error window
#--------------------------------------------------------
class Window_ErrorAlarm(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui\Error.ui", self)

 
#--------------------------------------------------------
# CaptureThread
#--------------------------------------------------------
class CapThread(QThread):
    image_update = pyqtSignal(np.ndarray)

    roi_x, roi_y, roi_w, roi_h = 20,20,10,10
    sp, ep = [0,0], [0,0]
    msp, mep = [0,0], [0,0]

    fps = 100
    size = (640,480)

    pu_m = 0.01590033
    pi_m = 0.616291862

    frame_num = 0

    t0 = None
    t1 = None
    t2 = None
    t3= None
    t4= None
    t5= None
    t6= None
    t7= None
    t8= None
    t9= None
    t10 = None
    t11 = None

    features = None
    gray_next = None
    gray_prev = None

    detection_x = None
    detection_y = None
    previous_x = None
    previous_y = None
    delta_x = 0
    delta_y = 0

    now = datetime.datetime.now()
    try:
       os.mkdir(os.getcwd()+ "/log/"+now.strftime("%Y%m%d%H%M%S"))
    except FileExistsError:
       pass
    # logの保存先を作成
    log_csv_name = now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"log.csv"
    log_file = open(log_csv_name,"a")
    log_file.write("0,1,2,3,4,5,6,7,8,9,10\n")
    # logの保存先urlを記録
    log_path_name = now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"log_path.txt"
    log_path = open(log_path_name,"a")
    log_path.write(log_csv_name)
    log_path.close()
    # rec動画の保存先を作成
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    filename = "rec.avi"
    writer = cv2.VideoWriter(os.getcwd()+"/log/"+now.strftime("%Y%m%d%H%M%S")+"/"+filename, fourcc, fps, size)
    
    def run(self):
        
        self.ThreadActive, self.frame = True, self.getbasler()

        while self.ThreadActive: 

            self.t0 = time.time()
            self.t1 = time.time()
            dete_time = time.time()

            self.frame = cv2.resize(self.frame , (640,480))
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            if variable.detect_flag == True:
                variable.detect_flag = False
                print(variable.click_x)
                optical.onMouse(self,float(variable.click_x),float(variable.click_y))
                self.rec_x = 0
                self.rec_y = 0
            else:
                pass
                
            if variable.reset_flag == True:
                variable.reset_flag=False
                self.features = None
                self.delta_x = 0
                self.delta_y = 0

            
            if variable.marker_ctl == 0:
                pass
            elif variable.marker_ctl == 1:
                variable.marker_ctl = 0
                for feature in self.features:
                    feature[0][1] -= variable.markermove
            elif variable.marker_ctl == 2:
                variable.marker_ctl = 0
                for feature in self.features:
                        feature[0][1] += variable.markermove
            elif variable.marker_ctl == 3:
                variable.marker_ctl = 0
                for feature in self.features:
                        feature[0][0] -= variable.markermove
            elif variable.marker_ctl == 4:
                variable.marker_ctl = 0
                for feature in self.features:
                        feature[0][0] += variable.markermove
            else:
                pass


            ### Optical flow --------------------------------------------------------
            if self.features is not None or (self.detection_x is not None and self.detection_y is not None):
                
                #if  len(self.features) != 0:
                try:
                    features_prev = self.features
                    self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
                                                        self.gray_prev, \
                                                        self.gray_next, \
                                                        features_prev, \
                                                        None, \
                                                        winSize = (variable.markersize, variable.markersize), \
                                                        maxLevel = 3, \
                                                        criteria = CRITERIA, \
                                                        flags = 0)
                    
                    # 有効な特徴点のみ残す
                    optical.refreshFeatures(self)
            
                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        self.detection_x = int(feature[0][0])
                        self.detection_y = int(feature[0][1])
                        
                #else:
                except:
                    print("再検出")
                    optical.onMouse(self,self.detection_x,self.detection_y)
                
                if self.detection_x == None or self.previous_x == None:
                    self.delta_x = 0
                else:
                    self.delta_x = self.detection_x - self.previous_x

                if self.detection_y == None or self.previous_y == None:
                    self.delta_y = 0
                else:
                    self.delta_y = self.detection_y - self.previous_y

                if self.detection_x is not None and self.detection_y is not None and self.previous_x is not None and self.previous_y is not None:
                    if variable.track_flag == True:

                        self.delta_d_flag = math.sqrt(pow((self.detection_x - self.previous_x),2)+pow((self.detection_y - self.previous_y),2))
                
                        if self.delta_d_flag < variable.mergin:
                            self.m = 1#10
                        elif self.delta_d_flag >= variable.mergin and self.delta_d_flag < 5:
                            self.m = 1#50
                        elif self.delta_d_flag >= 5 and self.delta_d_flag < 10:
                            self.m = 1#80
                        else:
                            self.m = 1#90

                        gap_x = variable.centerx-self.detection_x
                        gap_y = variable.centery-self.detection_y
                    
                        if ((abs(gap_x) <= variable.mergin and abs(gap_y) <= variable.mergin)) and (variable.track_flag == True):
                            pass      
                        elif (gap_x) < -variable.mergin and (gap_y) > variable.mergin and (variable.track_flag == True):
                            MainWindow.upperRight(self, gap_x, gap_y, self.m)
                        elif (gap_x) < -variable.mergin and (gap_y) < -variable.mergin and (variable.track_flag == True):
                            MainWindow.lowerRight(self, gap_x, gap_y, self.m)                   
                        elif (gap_x) > variable.mergin and (gap_y) > variable.mergin and (variable.track_flag == True):
                            MainWindow.upperLeft(self, gap_x, gap_y, self.m)
                        elif (gap_x) > variable.mergin and (gap_y) < -variable.mergin and (variable.track_flag == True):
                            MainWindow.lowerLeft(self, gap_x, gap_y, self.m)
                        elif abs(gap_x) <= variable.mergin and (gap_y) > variable.mergin and (variable.track_flag == True):
                            MainWindow.UP(self, gap_y, self.m)    
                        elif abs(gap_x) <= variable.mergin and (gap_y) < -variable.mergin and (variable.track_flag == True):
                            MainWindow.DOWN(self, gap_y, self.m)      
                        elif (gap_x) < -variable.mergin and abs(gap_y) <= variable.mergin and (variable.track_flag == True):
                            MainWindow.RIGHT(self, gap_x, self.m)             
                        elif (gap_x) > variable.mergin and abs(gap_y) <= variable.mergin and (variable.track_flag == True):
                            MainWindow.LEFT(self, gap_x, self.m)
                        else:
                            print("err\n")
                                                    
                    self.frame = cv2.circle(self.frame, (self.detection_x, self.detection_y), variable.markersize*2, (255, 0, 255))
                    self.frame = cv2.circle(self.frame, (self.detection_x, self.detection_y), variable.mergin*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (self.detection_x, self.detection_y), (0, 255, 125))
            else:
                pass
            
            
            self.frame = cv2.drawMarker(self.frame, position=(variable.centerx, variable.centery), color=(255, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
            #mainImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if variable.track_flag == True:
                self.frame_num = self.frame_num +1
                
                t = dete_time - prev_time
                
                d = math.sqrt(pow((self.detection_x-self.previous_x)*self.pi_m,2)+pow((self.detection_y-self.previous_y)*self.pi_m,2))
                
                track_v = d / float(t)
                
                if abs(gap_x) >= abs(gap_y):
                    stage_t = math.sqrt(float((4*abs(gap_x)*self.m*self.pu_m))/variable.acce_time)
                    stage_v = (variable.min*self.pu_m + variable.acce_time*stage_t/4)
                elif abs(gap_x) < abs(gap_y):                   
                    stage_t = math.sqrt(float((4*abs(gap_y)*self.m*self.pu_m))/variable.acce_time)
                    stage_v = (variable.min*self.pu_m + variable.acce_time*stage_t/4)
                
                v = abs(track_v-stage_v)
                
                Err = math.sqrt(pow((gap_x)*self.pi_m,2)+pow((gap_y)*self.pi_m,2))

                

                self.rec_x = self.rec_x + self.delta_x
                self.rec_y = self.rec_y + self.delta_y   

            # 次のループ処理の準備
            self.previous_x = self.detection_x
            self.previous_y = self.detection_y
            prev_time = dete_time
            self.gray_prev = self.gray_next


        


            # Log save --------------------------------------------------------
            self.log_file.write(str(self.t0)+","+str(self.t1)+","+str(self.t2)+","+str(self.t3)+","+str(self.t4)+","+str(self.t5)+","+
                                str(self.t6)+","+str(self.t7)+","+str(self.t8)+","+str(self.t9)+"\n")
            """log_file.write("frame,time,x(pixel),y(pixel),radius(pixel),asobi(pixel),Δt(s),Δd(μm),track_v(μm/s),err(μm),sum_x(pixel),sum_y(pixel)\n")"""
            
            # Frame save --------------------------------------------------------
            if variable.rec_flag == True:
                self.writer.write(self.frame)

            ### Capture read --------------------------------------------------------
            self.ThreadActive, self.frame = True, self.getbasler()
            self.image_update.emit(self.frame)

    def getbasler(self):
        grabResult = basler.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = basler.converter.Convert(grabResult)
            img = image.GetArray()
        return img


    def stop(self):
        self.ThreadActive = False
        self.quit()

    
    def cv_img(self,img):
        image = QImage(img.data,img.shape[1],img.shape[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap

    def subcv_img(self,img):
        image = QImage(img.data,img.shape[1],img.shape[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap

#--------------------------------------------------------
# main window
#--------------------------------------------------------
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui\prototype.ui",self)

        self.spinBox_13.setValue(variable.centerx)
        self.spinBox_10.setValue(variable.centery)
        self.spinBox_3.setValue(variable.markersize)
        self.spinBox_4.setValue(variable.mergin)
        self.spinBox_5.setValue(variable.magnification)
        self.spinBox_6.setValue(variable.min)
        self.spinBox_7.setValue(variable.max)
        self.spinBox_8.setValue(variable.acce_time)
        self.spinBox_9.setValue(variable.stagemove)
        self.spinBox_12.setValue(variable.markermove)

        self.pushButton.pressed.connect(self.UP)
        self.pushButton_2.pressed.connect(self.RIGHT)
        self.pushButton_3.pressed.connect(self.DOWN)
        self.pushButton_4.pressed.connect(self.LEFT)
        self.pushButton_5.pressed.connect(self.mUP)
        self.pushButton_6.pressed.connect(self.mRIGHT)
        self.pushButton_7.pressed.connect(self.mDOWN)
        self.pushButton_8.pressed.connect(self.mLEFT)
        self.pushButton_9.pressed.connect(self.trackOFF)
        self.pushButton_10.pressed.connect(self.trackON)
        self.pushButton_11.clicked.connect(self.motor)
        self.pushButton_12.clicked.connect(self.monitor)
        self.pushButton_13.pressed.connect(self.close)
        self.pushButton_14.pressed.connect(self.set)

        self.spinBox_13.valueChanged.connect(self.valuechange)
        self.spinBox_10.valueChanged.connect(self.valuechange_2)
        self.spinBox_3.valueChanged.connect(self.valuechange_3)
        self.spinBox_4.valueChanged.connect(self.valuechange_4)
        self.spinBox_5.valueChanged.connect(self.valuechange_5)
        self.spinBox_6.valueChanged.connect(self.valuechange_6)
        self.spinBox_7.valueChanged.connect(self.valuechange_7)
        self.spinBox_8.valueChanged.connect(self.valuechange_8)
        self.spinBox_9.valueChanged.connect(self.valuechange_9)
        self.spinBox_12.valueChanged.connect(self.valuechange_10)

        global ser

        try:
            ser = serial.Serial(port=list(serial.tools.list_ports.grep("0403:6001"))[0][0], baudrate=38400, timeout=0.5)
        except IndexError:
            QMessageBox.information(self,"Message","No Motor Connection")
            sys.exit()
        #ser.write(("D:WS" + str(low_speed) + "F" + str(top_speed)+ "R" + str(acce_time) + "S" + str(low_speed) + "F" + str(top_speed) + "R" + str(acce_time) + "\r\n").encode())
        ser.write(b"S:1250\r\n")
        ser.write(b"S:2250\r\n")
        ser.write(b"C:W1\r\n")
    
    # Image output --------------------------------------------------------
    @pyqtSlot(np.ndarray)
    def update_mainimage(self,frame_cap):

        #frame_gray = cv2.cvtColor(frame_cap, cv2.COLOR_BGR2RGB)
        #original = self.cv_img(frame_gray)
        original = self.cv_img(frame_cap)
        self.MainImage.setPixmap(original)
        self.MainImage.setScaledContents(True)
        self.MainImage.mousePressEvent = self.getPos
        
        if variable.temp is not None:
            self.SubImage.setPixmap(variable.temp)
            self.SubImage.setScaledContents(True)
    
    # get position --------------------------------------------------------
    def getPos(self,event):

        if event.button() == Qt.LeftButton:
            variable.click_x = event.pos().x()
            variable.click_y = event.pos().y()
            variable.detect_flag = True
            
        if event.button() == Qt.RightButton: 
            variable.reset_flag = True
    
    def GetObject(self):
        try:
            self.SubImage.setpixmap()
            self.MainImage.setScaledContents(True)
        except:
            self.Win_error.show()
            self.Win_error.Qlabel_error.setText("Please start camera before set !")
    
    def cv_img(self,img):
        image = QImage(img.data,img.shape[1],img.shape[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap

    def subcv_img(self,img):
        image = QImage(img.data,img.shape[1],img.shape[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        return pixmap
    
    def monitor(self):
        # ON
        if self.pushButton_12.isChecked():
            # variable.basler_flag = True
            # variable.optical_flag = True
            
            try:
                self.pushButton_12.setText("MonitorON")
                self.pushButton_12.setStyleSheet("background-color: red")
                #LogThread().start()
                self.videothread = CapThread()
                self.videothread.image_update.connect(self.update_mainimage)
                self.videothread.start()
            except Exception as error :
                print("err")
        
        # OFF
        else:
            self.pushButton_12.setText("MonitorOFF")
            self.pushButton_11.setStyleSheet("background-color: none")
            # LogThread().log_file.close()
            # LogThread().stop()
            self.videothread.stop()
            variable.basler_flag,variable.webcam_flag,variable.vidcap_flag = False,False,False
            variable.template_flag,variable.optical_flag = False,False
    
    def motor(self):
        # ON
        if self.pushButton_11.isChecked():
            variable
            self.pushButton_11.setText("MotorON")
            self.pushButton_11.setStyleSheet("background-color: red")
            ser.write(b"C:W0\r\n")
        # OFF
        else:
            self.pushButton_11.setText("MotorOFF")
            self.pushButton_11.setStyleSheet("background-color: none")
            ser.write(b"C:W1\r\n")

    def trackON(self):
        variable.track_flag = True
        variable.rec_flag = True
        pass

    def trackOFF(self):
        variable.track_flag = False
        variable.rec_flag = False
        pass
    
    def close(self):
        app.quit()

    def set(self):
        ser.write(("D:WS"+str(variable.min)+"F"+str(variable.max)+"R"+str(variable.acce_time)
                     +"S"+str(variable.min)+"F"+str(variable.max)+"R"+str(variable.acce_time)+"\r\n").encode())
    
    def UP(self,pulse):
        ser.write(("M:2+P"+str(pulse)+"\r\n").encode())
        ser.write(b"G\r\n")
    def DOWN(self,pulse):
        ser.write(("M:2-P"+str(pulse)+"\r\n").encode())
        ser.write(b"G\r\n")
    def RIGHT(self,pulse):
        ser.write(("M:1+P"+str(pulse)+"\r\n").encode())
        ser.write(b"G\r\n")
    def LEFT(self,pulse):
        ser.write(("M:1-P"+str(pulse)+"\r\n").encode())
        ser.write(b"G\r\n")

    def mUP(self):
        variable.marker_ctl = 1
    def mDOWN(self):
        variable.marker_ctl = 2
    def mRIGHT(self):
        variable.marker_ctl = 4
    def mLEFT(self):
        variable.marker_ctl = 3

    def upperRight(self, delta_x0, delta_y0): 
        ser.write(("M:W+P"+str(math.floor(variable.magnification*abs(delta_x0)))+"+P"+str(math.floor(variable.magnification*abs(delta_y0)))+"\r\n").encode())
        ser.write(b"G\r\n")

    def upperLeft(self, delta_x0, delta_y0):
        ser.write(("M:W-P"+str(math.floor(variable.magnification*abs(delta_x0)))+"+P"+str(math.floor(variable.magnification*abs(delta_y0)))+"\r\n").encode())
        ser.write(b"G\r\n")
    def lowerRight(self, delta_x0, delta_y0): 
        ser.write(("M:W+P"+str(math.floor(variable.magnification*abs(delta_x0)))+"-P"+str(math.floor(variable.magnification*abs(delta_y0)))+"\r\n").encode())
        ser.write(b"G\r\n")
    def lowerLeft(self, delta_x0, delta_y0):
        ser.write(("M:W-P"+str(math.floor(variable.magnification*abs(delta_x0)))+"-P"+str(math.floor(variable.magnification*abs(delta_y0)))+"\r\n").encode())
        ser.write(b"G\r\n")
    def up(self, delta_y0): 
        ser.write(("M:2+P"+str(math.floor(variable.magnification*abs(delta_y0)))+"\r\n").encode())
        ser.write(b"G\r\n")
    def down(self, delta_y0):
        ser.write(("M:2-P"+str(math.floor(variable.magnification*abs(delta_y0)))+"\r\n").encode())
        ser.write(b"G\r\n")
    def right(self, delta_x0): 
        ser.write(("M:1+P"+str(math.floor(variable.magnification*abs(delta_x0)))+"\r\n").encode())
        ser.write(b"G\r\n")
    def left(self, delta_x0):
        ser.write(("M:1-P"+str(math.floor(variable.magnification*abs(delta_x0)))+"\r\n").encode())
        ser.write(b"G\r\n")

    def valuechange(self):
        variable.centerx = self.spinBox_13.value() + 320
    def valuechange_2(self):
        variable.centery = self.spinBox_10.value() + 240
    def valuechange_3(self):
        variable.markersize = self.spinBox_3.value()
    def valuechange_4(self):
        variable.mergin = self.spinBox_4.value()
    def valuechange_5(self):
        variable.magnification = int(self.spinBox_5.value()) 
    def valuechange_6(self):
        variable.min = self.spinBox_6.value()
    def valuechange_7(self):
        variable.max = self.spinBox_7.value()
    def valuechange_8(self):
        variable.acce_time = self.spinBox_8.value()
    def valuechange_9(self):
        variable.stagemove = self.spinBox_9.value()
    def valuechange_10(self):
        variable.markermove = self.spinBox_12.value()

    

#--------------------------------------------------------
# start
#--------------------------------------------------------
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

