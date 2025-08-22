# -*- coding: utf-8 -*-
from re import S
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtCore import Qt
import cv2
# import serial
# import serial.tools.list_ports
import time
import datetime
import os

import glob

import csv
import math
import image_operation3 as image_operation3
import numpy as np

#以下二行Baslerカメラのためのモジュール読み込み
#import basler
#from pypylon import pylon

CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)



#動画のパスを入力
VIDEO_DATA = r"VIDEO_DATA\0809\20230809172434_toofasttrack\rec-10StoF.avi"

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

delta_x0 = 0
delta_y0 = 0

fps = 38

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

# 以下225行までGUI関係の記述。ボタンやスピンボックスの設定です
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

    #追跡ボタンの開始設定とRecThreadの開始
    def track(self):
        global flag
        flag = True
        self.pushButton_3.setStyleSheet("background-color: red")
        RecThread(self).start()
        RecThreadDetect(self).start()

    #追跡ボタンの終了設定、ログの保存
    def trackoff(self):
        global flag,log_file,log_name
        flag = False
        self.pushButton_3.setStyleSheet("background-color: none")
        RecThread(self).video.release()
        RecThreadDetect(self).video.release()
        RecThread(self).quit()
        RecThreadDetect(self).quit()
        print("log saved")
        log_file.close()
        
        # log_name = datetime.datetime.now().strftime(os.getcwd()+ "OFFLINE/log/" + "%Y%m%d%H%M%S/")+"log.csv"
        # log_file = open(log_name,"a")
        # log_file.write("frame,rec_x,rec_y,detect_x,detect_y,delta_x,delta_y,delta\n")
        
    #プログラム終了処理
    def quit(self):
        files = glob.glob(os.getcwd() + "/save/*")
        for file in files:
            if(os.path.getsize(file) < 6000):
                RecThread(self).video.release()
                RecThreadDetect(self).video.release()
                RecThread(self).quit()
                RecThreadDetect(self).quit()
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

    #探索範囲
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
    

#動画録画スレッド
class RecThread(QtCore.QThread):
        #now = datetime.datetime.now()
        # try:
        #     os.mkdir(os.getcwd()+ "/save")
        # except FileExistsError:
        #     None
        filename = now.strftime(os.getcwd()+ "/module/VideoTracking_module/log/" + "%Y%m%d%H%M%S/")+"rec.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        video = cv2.VideoWriter(filename, fourcc, 38, (640,480))

#検出動画録画スレッド
class RecThreadDetect(QtCore.QThread):
        #now = datetime.datetime.now()
        filename = now.strftime(os.getcwd()+ "/module/VideoTracking_module/log/" + "%Y%m%d%H%M%S/")+"rec-detect.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        video = cv2.VideoWriter(filename, fourcc, 38, (640,480))

    
    

#メインスレッド
class CapThread(QtCore.QThread):
    mainFrameGrabbed = QtCore.pyqtSignal(QtGui.QImage)


    rec_x = 0
    rec_y = 0
        
    def run(self):
        self.kf = KalmanFilter()
        
        #2021.12.17日追記(冨樫)
        #track開始時に保存先ディレクトリをつくる
        # try:
        #     os.mkdir(os.getcwd()+ "/log")
        # except FileExistsError:
        #     None

        global detect_flag,reset_flag,detect_up,detect_down,detect_left,detect_right
        #previous_approxCurves = None #消して不具合がないか確認する
        # 各変数を初期化
        # 特徴点
        self.features = None
        # 現在のフレーム（カラー）
        self.frame = None
        # 現在のフレーム（グレー）
        self.gray_next = None
        # 前回のフレーム（グレー）
        self.gray_prev = None
        # 検出されたX,Y座標
        detection_x = 0
        detection_y = 0
        delta = 0
        
        # フレーム数
        self.frame_num = 0


        # 映像を選択する。いずれかのコメントを外す。
        #self.video = cv2.VideoCapture(VIDEO_DATA)#どのビデオを見ているか　適当な動画をみたいときはVIDEO DATA
        # カメラではなく、ビデオを使用する場合は、コメントを外して待ち時間を設ける。
        time.sleep(1)
        
        # 元ビデオファイル読み込み
        org = cv2.VideoCapture(VIDEO_DATA)
        time.sleep(1)
        time.sleep(1)
        # 保存ビデオファイルの準備
        end_flag, self.frame = org.read()

        # 1フレーム目の画像をグレースケールに変換する
        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        # 動画が終了するまでループする。
        while end_flag:
            time1 = time.time()
            # 表示画面のサイズを、640✖︎480に変換する。
            self.frame = cv2.resize(self.frame , (640,480))
            # オリジナル画像を動画として保存するため、コピーをとる。
            image_org = self.frame.copy()
            # 動画保存フラグがTrueの場合、画像をビデオとして書き込む
            if (flag == True) :
                RecThread().video.write(image_org)

            # 探索半径（pixel）を設定する。
            self.radius = radius
            
            # 特徴点を検出するため、グレースケールに変換
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            #クリックした画像内の座標を取得する。特徴点を指定する。
            if(detect_flag == True):
                image_operation3.onMouse(self,click_x,click_y) 
                detect_flag = False
                # 記録座標を初期化する
                self.rec_x = 0
                self.rec_y = 0

            #右クリックで特徴点を初期化する
            if(reset_flag == True):
                # 特徴点を初期化する。
                self.features = None
                # リセットフラグをFalseに戻す。
                reset_flag = False
                # 特徴点の座標を初期化する。
                detection_x = None
                detection_y = None
                # 検出していた前フレームの特徴点の座標を初期化する。
                previous_x = None
                previous_y = None
                # 特徴点と前フレームの特徴点の座標の差を初期化する。
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

            #print(self.features)
            


            # 特徴点が登録されている場合にOpticalFlowを計算する
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
                    
                    # 有効な特徴点のみ残す
                    image_operation3.refreshFeatures(self)

                    # フレームに有効な特徴点を描画
                    for feature in self.features:
                        # X座標、Y座標をそれぞれ取得する。
                        detection_x = int(feature[0][0])
                        detection_y = int(feature[0][1])
                    
                    predictd = self.kf.predict(detection_x, detection_y)
                    cv2.circle(self.frame, (detection_x, detection_y), 10, (0, 255, 255), -1)
                    cv2.circle(self.frame, (predictd[0], predictd[1]), 10, (255, 0, 0), 4)
                    
                else: # 特徴点が登録されている場合にOpticalFlowを計算する
                    print("再検出")
                    image_operation3.onMouse(self,detection_x,detection_y)
                
                
                if detection_x == None or previous_x == None: # 前フレームあるいは現在のフレームのX座標が存在しない場合
                    delta_x = 0
                else: # 前フレームと現在のフレームのX座標が存在する場合
                    # 前フレームと現在のフレームのX座標を比較して、差を計算する。
                    delta_x = detection_x - previous_x

                if detection_y == None or previous_y == None:# 前フレームあるいは現在のフレームのY座標が存在しない場合
                    delta_y = 0
                else:# 前フレームと現在のフレームのY座標が存在する場合
                    # 前フレームと現在のフレームのY座標を比較して、差を計算する。
                    delta_y = detection_y - previous_y 
                
                delta = math.sqrt(pow(delta_x,2)+pow(delta_y,2))

                # 画像上に、検出した縁を描く
                
                # cv2.circle(画像,(縁の中心のX座標,縁の中心のY座標),直径,(B,G,R))
                self.frame = cv2.circle(self.frame, (detection_x, detection_y), self.radius*2, (255, 0, 255))
                self.frame = cv2.circle(self.frame, (detection_x, detection_y), asobi*2, (0, 255, 255))
                # 画像上に、マーカーを描く
                # cv2.drawMarker(画像,(縁の中心のX座標,縁の中心のY座標),(B,G,R))
                

                # 前フレームで検出した座標に、円とマーカを描く
                if (previous_y is not None) and (previous_x is not None):
                    self.frame = cv2.circle(self.frame, (previous_x, previous_y), self.radius*2, (0, 0, 255))
                    self.frame = cv2.circle(self.frame, (previous_x, previous_y), asobi*2, (0, 255, 255))
                    self.frame = cv2.drawMarker(self.frame, (previous_x, previous_y), (0, 255, 0))
                    

            else:
                pass
                #print("No detected")
            time2 = time.time()
            
            # ログ保存フラグがTrueの場合
            if (flag == True) :
                
                # 動画に現在の画像を書き込む
                RecThreadDetect().video.write(self.frame)
                # CSVに線虫の移動量を書き込む
                self.rec_x = self.rec_x + delta_x
                self.rec_y = self.rec_y + delta_y
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
            time.sleep(1/fps)
            # 次のループ処理のため、前フレームの値に代入する。
            previous_x = detection_x
            previous_y = detection_y
            self.gray_prev = self.gray_next

            # 次の動画のフレームを取得する。        
            end_flag, self.frame = org.read()

            if end_flag: # 動画が続いていたら、グレースケールに変換する。
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            else: # 動画が終了している場合は各スレッドを解放する。
                RecThread(self).video.release()
                RecThreadDetect(self).video.release()
                log_file.close()
                # RecThread(self).quit()
                # RecThreadDetect(self).quit()
            time1 = time.time()
            # numpy 配列を QImage に変換する。
            mainQImage = QtGui.QImage(
                mainImage.data, mainImage.shape[1], mainImage.shape[0], QtGui.QImage.Format_RGB888)
            # シグナルを送出する。
            self.mainFrameGrabbed.emit(mainQImage)
            time2 = time.time()
            print(time2-time1)
            
            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())