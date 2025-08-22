'''
previous:

progress:
    /multiprocessingで動かすために、class variableを撤廃
        self.に移行
    /終了処理を追加
    /複数フォルダに分割
    /multiprocess 保存処理の追加

tasks:
'''
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow, QDialog, QMessageBox,QGraphicsPixmapItem
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QTimer, QTime, QDateTime, Qt, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen

import cv2, time, datetime, os, math, csv, sys, shutil, queue
from multiprocessing import Process, Queue
from util.utils import OpticalFlow, KalmanFilter, StageController, Pxep, write_log_init, write_log
from app_module import *


class MainWindow(QMainWindow):
    
    def __init__(self, que):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.que = que
        self.scl = StageController()
        self.scl.initStage()
        self.cthread = CapThread(self.scl, que)
        UIFunctions.removeTitleBar(True)
        
        ## ==> MOVE WINDOW 
        def moveWindow(event):
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()
        
        # WIDGET TO MOVE
        self.ui.frame_label_top_btns.mouseMoveEvent = moveWindow
        
        self.click_x = None
        self.click_y = None
        self.marker_direction = None
        self.centerX = 320
        self.centerY = 240
        self.radius = 15
        self.play = 1
        self.marker_move = 2
        
        self.pulse = 5000
        self.StartSpeed = 600
        self.HighSpeed = 35000
        self.Uptime = 50
        self.Downtime = 50
        self.SshapeRatio = 40
        self.OverSpeed = 3
        
        self.track_flag = False
        self.save_flag = False
        self.detect_flag = False
        self.reset_flag = True
        self.simulation_flag = False
        self.do_in_laptop = False
        
        self.file_path = None
        self.csv_path = None
        self.log_file = None
        self.param_file = None
        self.writer = None
        
        ## ==> LOAD DEFINITIONS#
        UIFunctions.uiDefinitions(self)
        
        self.cthread.start()
        self.show()
    
    @pyqtSlot(QtGui.QImage)
    def update_mainimage(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.ui._item.setPixmap(pixmap)
    
    @pyqtSlot(bool, bool, bool, bool, bool, bool)
    def flag_update_from_thread(self, track, save, detect, reset, simulate, do_in_laptop):
        self.track_flag = track
        self.save_flag = save
        self.detect_flag = detect
        self.reset_flag = reset
        self.simulation_flag = simulate
        self.do_in_laptop = do_in_laptop
        
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseButtonPress and source is self.ui.graphicsView and self.ui.graphicsView.itemAt(event.pos()):
            if event.button() == Qt.LeftButton:
                self.detect_flag = True
                pos = self.ui.graphicsView.mapToScene(event.pos())
                self.click_x = pos.x()
                self.click_y = pos.y()
            if event.button() == Qt.RightButton:
                self.reset_flag = True
                self.click_x = None
                self.click_y = None
            UIFunctions.flag_update_to_thread(self)
            UIFunctions.features_update_to_thread(self)
        return QWidget.eventFilter(self, source, event)


## ==> END


##################################################
## ==> START: CapThread class
class CapThread(QThread):
    image_update = pyqtSignal(QtGui.QImage)
    flag_update = pyqtSignal(bool, bool, bool, bool, bool, bool)
    
    def __init__(self, scl, que, parent=None):
        super(CapThread, self).__init__(parent)

        self.que = que
        self.opt = OpticalFlow()
        self.kf = KalmanFilter()
        self.scl = scl
        
        import util.basler as basler
        self.bas = basler

        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

        self.MAINPC_PATH = 'log/save_path.txt'
        self.START_TIME = 'log/start_time.txt'

        self.features = None
        self.frame = None
        self.gray_next = None
        self.gray_prev = None

        self.video_path = 'img/rec.avi'

    def run(self):
        '''
        '''
        
        PulsetoMicro = 0.01590033
        PixeltoMicro = 0.616291862
        gap_x_Micro = None
        gap_y_Micro = None
        
        # BaslerCamera
        if self.do_in_laptop:
            cap = cv2.VideoCapture(self.video_path)
            ret, self.frame_origin = cap.read()
        else:
            ret, self.frame_origin = True, self.bas.getBasler()
        time.sleep(1)
        
        frame_num = 0
        self.is_Running = True
        self.lefttoright = True
        while self.is_Running:

            dete_time = time.time()
            tn = datetime.datetime.now()
            time_delta = datetime.timedelta(hours=tn.hour,minutes=tn.minute,seconds=tn.second,microseconds=tn.microsecond)

            self.frame = cv2.resize(self.frame_origin , (640,480))
            if self.track_flag == True:
                self.que.put(self.frame.copy())
            #self.gray_next = cv2.GaussianBlur(self.frame, (variable.kernel, variable.kernel), variable.sigma)
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            

            #特徴点を指定する
            if self.detect_flag == True:
                self.detect_flag = False
                self.flag_update_to_main()
                image_operation3.onMouse(self,self.click_x,self.click_y,self.radius)              

            #右クリックで特徴点リセット
            if self.reset_flag == True:
                self.reset_flag = False
                self.flag_update_to_main()
                self.features = None
                detection_x = None
                detection_y = None
                previous_x = None
                previous_y = None

            if self.marker_direction != None:
                if detection_x and detection_y != None:
                    if(self.marker_direction == 0):
                        for feature in self.features:
                                feature[0][1] -= self.marker_dis
                    elif(self.marker_direction == 1):
                        for feature in self.features:
                                feature[0][1] += self.marker_dis
                    elif(self.marker_direction == 2):
                        for feature in self.features:
                                feature[0][0] -= self.marker_dis
                    elif(self.marker_direction == 3):
                        for feature in self.features:
                                feature[0][0] += self.marker_dis
                    self.marker_direction = None
                else:
                    self.marker_direction = None
            
            if self.simulation_flag == True:
                # statusX, statusY = self.scl.wGetDriveStatus()
                # statusX = 0 if statusX == 65536 else statusX
                # statusY = 0 if statusY == 65536 else statusY
                # if statusX == 0 and statusY == 0:
                if self.lefttoright == True:
                    ret, _ = self.scl.wDriveStart(self.pulse, 0)#2)
                    if ret == 1:
                        self.lefttoright = False
                else:
                    ret, _ = self.scl.wDriveStart(self.pulse, 1)#3)
                    if ret == 1:
                        self.lefttoright = True

            # ステージ座標取得
            stageX, stageY = self.scl.wGetCounter()

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
                                                        criteria = self.CRITERIA, \
                                                        flags = 0)

                    # 有効な特徴点のみ残す
                    image_operation3.refreshFeatures(self)

                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        detection_x = max(0, min(int(feature[0][0]), 639))
                        detection_y = max(0, min(int(feature[0][1]), 479))

                    # KalmanFiter
                    predictd = self.kf.predict(detection_x, detection_y)
                
                else:
                    image_operation3.onMouse(self, detection_x, detection_y, self.radius)

                if detection_x and detection_y and previous_x and previous_y != None:
                    if self.track_flag == True:

                        # 距離に応じた移動倍率の変更
                        # self.delta_d_flag = math.sqrt(pow((detection_x - previous_x),2)+pow((detection_y - previous_y),2))
                        # if self.delta_d_flag < variable.play:
                        #     self.m = 0.7
                        # elif self.delta_d_flag >= variable.play and self.delta_d_flag < 10:
                        #     self.m = 0.9
                        # elif self.delta_d_flag >= 10 and self.delta_d_flag < 20:
                        #     self.m = 1.1
                        # else:
                        #     self.m = 1.2

                        gap_x = self.centerX-detection_x
                        gap_y = self.centerY-detection_y

                        gap_x_Micro = gap_x*PixeltoMicro
                        gap_y_Micro = gap_y*PixeltoMicro

                        gap_x_Pulse = gap_x*PixeltoMicro/PulsetoMicro
                        gap_y_Pulse = gap_y*PixeltoMicro/PulsetoMicro

                        xPulse = int(float(stageX) + gap_x_Pulse * 0.8)
                        yPulse = int(float(stageY) + gap_y_Pulse * 0.8)

                        if abs(gap_x) < self.play and abs(gap_y) < self.play:
                            pass
                        else:
                            print(f"xPulse: {xPulse}, yPulse: {yPulse}")
                            self.scl.wPulseOverride(xPulse, yPulse)
  
                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), self.radius*2, (255, 0, 255))
                    self.frame = cv2.circle(self.frame, (detection_x, detection_y), self.play*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (detection_x, detection_y), color=(0, 255, 125), markerSize=10)

            # 表示
            self.frame = cv2.drawMarker(self.frame, position=(self.centerX, self.centerY), color=(255, 150, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
            mainImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # numpy 配列を QImage に変換する。
            mainQImage = QtGui.QImage(mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            # シグナルを送出する。
            self.image_update.emit(mainQImage)

            if self.track_flag == True:
                frame_num = frame_num +1

                delta_t = dete_time - prev_time
                delta_d = math.sqrt(pow((stageX-previous_stageX)*PulsetoMicro,2)+pow((stageY-previous_stageY)*PulsetoMicro,2))
                if delta_t == 0:
                    velocity = None
                else:
                    velocity = delta_d / delta_t
                Err = None if gap_x_Micro == None else math.sqrt(pow(gap_x_Micro,2)+pow(gap_y_Micro,2))

                write_log(self.log_file,frame_num,time_delta,stageX,stageY,delta_t,delta_d, velocity, detection_x, detection_y,gap_x, gap_y, gap_x_Micro, gap_y_Micro, gap_x_Pulse, gap_y_Pulse, Err, predictd, self.radius, self.play)

            # 次のループ処理の準備
            previous_stageX = stageX
            previous_stageY = stageY
            previous_x = detection_x
            previous_y = detection_y
            prev_time = dete_time
            self.gray_prev = self.gray_next
            
            # BaslerCamera
            if self.do_in_laptop:
                ret, self.frame_origin = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, self.frame_origin = cap.read()
            else:
                ret, self.frame_origin = True, self.bas.getBasler()
            
            if not ret:
                print("No Frame")
            

            # Track終了後 保存処理
            if self.save_flag == True:
                print("===== Start Editing =====")
                
                self.que.put(None)

                self.log_file.close()

                # Initialization
                frame_num = 0

                # Save
                s_time = open(self.START_TIME,"r")
                start_time = s_time.read()

                self.param_file.write("center:\t("+str(self.centerX)+","+str(self.centerY)+")\n")

                if start_time != str(0):
                    CSV_file = open(self.csv_path,"r")
                    csvReader = csv.reader(CSV_file)
                    csvHeader = next(csvReader)
                    MPC_path = open(self.MAINPC_PATH,"r")
                    mainPC_path = MPC_path.read()
                    corrected_path = mainPC_path.replace("169.254.104.216", "Desktop-vungl90")

                    for row in csvReader:
                        time_csv = float(row[1])
                        time_csv = str(time_csv)
                        if start_time < time_csv:
                            start_frame_num = row[0]
                            break
                        else:
                            start_frame_num = 0

                    self.param_file.write(
                        "path:\t"+str(corrected_path)+"\n"+
                        "start_time:\t"+str(start_time)+"\n"+
                        "start_frame_num:\t"+str(start_frame_num)+"\n"
                    )

                    CSV_file.close()
                    MPC_path.close()
                    s_time.close()
                    s_time = open(self.START_TIME,"w")
                    s_time.write(str(0))
                    s_time.close()

                else:
                    s_time.close()

                self.param_file.close()

                # Transfer
                MPC_path = open(self.MAINPC_PATH,"r")
                mainPC_path = MPC_path.read()
                corrected_path = mainPC_path.replace("169.254.104.216", "Desktop-vungl90")
                if corrected_path != str(0):
                    shutil.move(self.file_path, corrected_path)
                    MPC_path.close()
                    MPC_path = open(self.MAINPC_PATH,"w")
                    MPC_path.write(str(0))
                    MPC_path.close()
                else:
                    MPC_path.close()
                
                self.save_flag = False
                self.flag_update_to_main()
                
                print("===== Editing Completed =====")

    def update_param(self, x, y, r, play, marker, pulse):
        self.centerX = x
        self.centerY = y
        self.radius = r
        self.play = play
        self.marker_dis = marker
        self.pulse = pulse

    def update_features(self, x, y):
        self.click_x = x
        self.click_y = y

    def flag_update_to_main(self):
        self.flag_update.emit(self.track_flag, self.save_flag, self.detect_flag, self.reset_flag, self.simulation_flag, self.do_in_laptop)

    def flag_update_from_main(self, track, save, detect, reset, simulate, do_in_laptop):
        self.track_flag = track
        self.save_flag = save
        self.detect_flag = detect
        self.reset_flag = reset
        self.simulation_flag = simulate
        self.do_in_laptop = do_in_laptop

    def update_marker_direction(self, dir):
        self.marker_direction = dir

    def update_filepath(self, file_path, csv_path, param_file, writer, log_file):
        self.log_file = log_file
        self.param_file = param_file
        self.writer = writer
        self.file_path = file_path
        self.csv_path = csv_path

    def stop(self):
        # self.bas.closeBasler()
        self.is_Running = False

## ==> END


##################################################
## ==> START: define(for getting image from basler camera)
def SaveImage_process(que):
    '''
    Save the image
    '''
    init = False
    path = None
    while True:
        try:
            image = que.get_nowait()
            if image is not None:
                
                if init == True:
                    init = False
                    print("create rec.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    path = image
                    fps = 115
                    size = (640, 480)
                    size_origin = (1020, 764)
                    writer = cv2.VideoWriter(path+"/rec.avi", fourcc, fps, size)
                else:
                    if image is True:
                        init = True
                        print("init save process")
                    elif image is False:
                        break
                    else:
                        writer.write(image)
            else:
                writer.release()
                print("release rec.avi")

        except queue.Empty:
            pass

## ==> END


##################################################
## ==> START: define(for getting image from basler camera)
def Gui_process(que):
    app = QApplication(sys.argv)
    window = MainWindow(que)
    sys.exit(app.exec_())

## ==> END


##################################################
## ==> START: define(for getting image from basler camera)
if __name__ == "__main__":
    ### Set flags
    do_profile = False

    ### Set queues
    save_queue = Queue()

    ### Set processes
    process_SaveImage = Process(target=SaveImage_process, args=(save_queue,))
    process_Gui = Process(target=Gui_process, args=(save_queue,))

    ### Start processes
    process_SaveImage.start()
    process_Gui.start()
