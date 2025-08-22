from main10 import *

GLOBAL_STATE = 0
GLOBAL_TITLE_BAR = True

class UIFunctions(MainWindow):

    ## GUI DEFINITIONS
    ########################################################################

    ## ==> UI DEFINITIONS
    def uiDefinitions(self):
        '''
        '''
        ## REMOVE ==> Standard Title Bar
        if GLOBAL_TITLE_BAR:
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        else:
            self.ui.horizontalLayout.setContentsMargins(0, 0, 0, 0)
            self.ui.frame_label_top_btns.setContentsMargins(8, 0, 0, 5)
            self.ui.frame_label_top_btns.setMinimumHeight(42)
            self.ui.frame_icon_top_bar.hide()
            self.ui.frame_btns_right.hide()
            self.ui.frame_size_grip.hide()
    
        ## SET ==> Graphics View
        scene = QtWidgets.QGraphicsScene()
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.installEventFilter(self)
        self.ui._item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self.ui._item)
        
        ## SET ==> Timer
        self.timer = QTimer()
        self.time = QTime(0, 0, 0)
        self.timer.timeout.connect(lambda: UIFunctions.updateTimer(self))
        self.ui.pushButton_ResetTimer.pressed.connect(lambda: UIFunctions.resetTimer(self))

        ## SET ==> Parameters
        self.ui.spinBox_X.setValue(self.centerX)
        self.ui.spinBox_Y.setValue(self.centerY)
        self.ui.spinBox_Radius.setValue(self.radius)
        self.ui.spinBox_Play.setValue(self.play)
        self.ui.spinBox_Mmodify.setValue(self.marker_move)
        self.ui.spinBox_Pulse.setValue(self.pulse)
        self.ui.spinBox_Speed.setValue(self.StartSpeed)
        self.ui.spinBox_HighSpeed.setValue(self.HighSpeed)
        self.ui.spinBox_Utime.setValue(self.Uptime)
        self.ui.spinBox_Dtime.setValue(self.Downtime)
        self.ui.spinBox_SshapeRatio.setValue(self.SshapeRatio)
        self.ui.spinBox_Override.setValue(self.OverSpeed)
        
        self.ui.comboBox.setCurrentIndex(1)
        
        ## SET ==> Signals when value changed
        self.ui.spinBox_X.valueChanged.connect(lambda: UIFunctions.valuechange_operation(self))
        self.ui.spinBox_Y.valueChanged.connect(lambda: UIFunctions.valuechange_operation(self))
        self.ui.spinBox_Radius.valueChanged.connect(lambda: UIFunctions.valuechange_operation(self))
        self.ui.spinBox_Play.valueChanged.connect(lambda: UIFunctions.valuechange_operation(self))
        self.ui.spinBox_Mmodify.valueChanged.connect(lambda: UIFunctions.valuechange_operation(self))
        self.ui.spinBox_Pulse.valueChanged.connect(lambda: UIFunctions.valuechange_operation(self))
        self.ui.spinBox_Speed.valueChanged.connect(lambda: UIFunctions.valuechange_stage(self))
        self.ui.spinBox_HighSpeed.valueChanged.connect(lambda: UIFunctions.valuechange_stage(self))
        self.ui.spinBox_Utime.valueChanged.connect(lambda: UIFunctions.valuechange_stage(self))
        self.ui.spinBox_Dtime.valueChanged.connect(lambda: UIFunctions.valuechange_stage(self))
        self.ui.spinBox_SshapeRatio.valueChanged.connect(lambda: UIFunctions.valuechange_stage(self))
        self.ui.spinBox_Override.valueChanged.connect(lambda: UIFunctions.valuechange_stage(self))
        
        ## SET ==> Signals when button pressed
        self.ui.pushButton_Close.pressed.connect(lambda: UIFunctions.CloseMainWindow(self))
        self.ui.pushButton_Ready.pressed.connect(lambda: UIFunctions.Ready(self))
        self.ui.pushButton_TrackON.pressed.connect(lambda: UIFunctions.TrackON(self))
        self.ui.pushButton_TrackOFF.pressed.connect(lambda: UIFunctions.TrackOFF(self))
        self.ui.pushButton_ServoON.pressed.connect(lambda: UIFunctions.ServoON(self))
        self.ui.pushButton_ServoOFF.pressed.connect(lambda: UIFunctions.ServoOFF(self))
        self.ui.pushButton_MotorSimulate.pressed.connect(lambda: UIFunctions.MotorServo(self)) 
        self.ui.pushButton_SetParam.pressed.connect(lambda: UIFunctions.SetVelocity(self))
        self.ui.pushButton_Stop.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_Up.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_Down.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_Left.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_Right.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_UpLeft.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_UpRight.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_LowLeft.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        self.ui.pushButton_LowRight.pressed.connect(lambda: UIFunctions.Arrow_keys(self))
        # self.ui.pushButton_PixelCount.pressed.connect(lambda: UIFunctions.setPixelCount(self))
        # self.ui.pushButton_SaveTxt.pressed.connect(lambda: UIFunctions.SaveTxt(self))
        
        ## SET ==> Threads
        self.cthread.image_update.connect(self.update_mainimage)
        self.cthread.flag_update.connect(self.flag_update_from_thread)
        
        ## SET ==> Initial state
        UIFunctions.valuechange_operation(self)
        UIFunctions.flag_update_to_thread(self)
        UIFunctions.features_update_to_thread(self)
        UIFunctions.marker_update_to_thread(self)
    
    
    
    ## GUI FUNCTIONS
    ########################################################################

    ## ==> SET TITLE BAR
    def removeTitleBar(status):
        global GLOBAL_TITLE_BAR
        GLOBAL_TITLE_BAR = status
    
        
    def CloseMainWindow(self):
        '''
        要修正
        '''
        if self.track_flag == True:
            UIFunctions.TrackOFF(self)
            while self.save_flag == True:
                continue
 
        # break save_process
        self.que.put(False)
        # break capthread
        self.cthread.stop()
        self.cthread.wait()
        # close stage
        self.scl.wClose()
        # app.quit()
        self.close()
        print('=========== Closed ===========')
    
    def startTimer(self):
        '''
        '''
        self.timer.start(1000)
    
    def resetTimer(self):
        '''
        '''
        self.timer.stop()
        self.time = QTime(0, 0, 0)
        self.ui.label_Timer.setText(self.time.toString('mm:ss'))
        if self.sender() == self.ui.pushButton_ResetTimer:
            UIFunctions.startTimer(self)
    
    def updateTimer(self):
        '''
        '''
        self.time = self.time.addSecs(1)
        self.ui.label_Timer.setText(self.time.toString('mm:ss'))
    
    def valuechange_operation(self):
        '''
        '''
        self.centerX = self.ui.spinBox_X.value()
        self.centerY = self.ui.spinBox_Y.value()
        self.radius = self.ui.spinBox_Radius.value()
        self.play = self.ui.spinBox_Play.value()
        self.marker_move = self.ui.spinBox_Mmodify.value()
        self.pulse = self.ui.spinBox_Pulse.value()
        self.cthread.update_param(self.centerX,self.centerY,self.radius,self.play,self.marker_move,self.pulse)
    
    def valuechange_stage(self):
        '''
        '''
        self.StartSpeed = self.ui.spinBox_Speed.value()
        self.HighSpeed = self.ui.spinBox_HighSpeed.value()
        self.Uptime = self.ui.spinBox_Utime.value()
        self.Downtime = self.ui.spinBox_Dtime.value()
        self.SshapeRatio = self.ui.spinBox_SshapeRatio.value()
        self.OverSpeed = self.ui.spinBox_Override.value()
    
    def marker_update_to_thread(self):
        '''
        '''
        self.cthread.update_marker_direction(self.marker_direction)
        
    def features_update_to_thread(self):
        '''
        '''
        self.cthread.update_features(self.click_x, self.click_y)

    def flag_update_to_thread(self):
        '''
        '''
        self.cthread.flag_update_from_main(self.track_flag, self.save_flag, self.detect_flag, self.reset_flag, self.simulation_flag, self.do_in_laptop)
    
    def Ready(self):
        '''
        '''
        today = datetime.date.today()
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.file_path = os.getcwd()+ "/log/"+str(today)+"/"+now

        directories = ["/log", "/log/" + str(today), "/log/" + str(today) + "/" + now]
        for directory in directories:
            try:
                os.mkdir(os.getcwd()+ directory)
            except FileExistsError:
                pass

        self.csv_path = self.file_path+"/log.csv"
        self.log_file = open(self.csv_path,"a")
        write_log_init(self.log_file)

        self.param_file = open(self.file_path+"/parameter.txt","a")
        
        try:
            os.mkdir(self.file_path+"/parameter.txt")
        except FileExistsError:
            self.cthread.update_filepath(self.file_path, self.csv_path, self.param_file, self.writer, self.log_file)
            self.que.put(True)
            self.que.put(self.file_path)
            print("ready\n")

        self.ui.pushButton_Ready.setEnabled(False)
        self.ui.pushButton_TrackON.setEnabled(True)
        self.ui.pushButton_Ready.setStyleSheet("background-color: red")

    def TrackON(self):
        '''
        '''
        if self.click_x and self.click_y != None:
            self.track_flag = True
            self.save_flag = False
            UIFunctions.flag_update_to_thread(self)
            # UIFunctions.ServoON(self)
            UIFunctions.startTimer(self)
            self.ui.radioButton_Marker.setChecked(True)
            self.ui.radioButton_Stage.setChecked(False)
            self.ui.pushButton_ServoOFF.setEnabled(False)
            self.ui.pushButton_TrackON.setEnabled(False)
            self.ui.pushButton_TrackOFF.setEnabled(True)
            self.ui.pushButton_TrackON.setStyleSheet("background-color: red")
        else:
            msg = QMessageBox()
            msg.setText("Please set the feature point")
            msg.setWindowTitle("Warning")
            msg.exec_()

    def TrackOFF(self):
        '''
        '''
        self.track_flag = False
        self.save_flag = True
        UIFunctions.flag_update_to_thread(self)
        UIFunctions.ServoOFF(self)
        UIFunctions.resetTimer(self)
        self.ui.radioButton_Marker.setChecked(False)
        self.ui.radioButton_Stage.setChecked(True)
        self.ui.pushButton_Ready.setEnabled(True)
        self.ui.pushButton_TrackOFF.setEnabled(False)
        self.ui.pushButton_Ready.setStyleSheet("background-color: rgb(52, 59, 72)")
        self.ui.pushButton_TrackON.setStyleSheet("background-color: rgb(52, 59, 72)")

    def ServoON(self):
        '''
        '''
        self.scl.wServoONOFF(flag=True)
        self.ui.pushButton_ServoON.setEnabled(False)
        self.ui.pushButton_ServoOFF.setEnabled(True)
        self.ui.pushButton_MotorSimulate.setEnabled(False)
        self.ui.pushButton_ServoON.setStyleSheet("background-color: red")

    def ServoOFF(self):
        '''
        '''
        self.simulation_flag = False
        UIFunctions.flag_update_to_thread(self)
        self.scl.wServoONOFF(flag=False)
        self.ui.pushButton_ServoON.setEnabled(True)
        self.ui.pushButton_ServoOFF.setEnabled(False)
        self.ui.pushButton_MotorSimulate.setEnabled(True)
        self.ui.pushButton_ServoON.setStyleSheet("background-color: rgb(52, 59, 72)")
        self.ui.pushButton_MotorSimulate.setStyleSheet("background-color: rgb(52, 59, 72)")
    
    def MotorServo(self):
        '''
        '''
        self.simulation_flag = True
        UIFunctions.flag_update_to_thread(self)
        self.scl.wServoONOFF(flag=True)
        self.ui.pushButton_ServoON.setEnabled(False)
        self.ui.pushButton_ServoOFF.setEnabled(True)
        self.ui.pushButton_MotorSimulate.setEnabled(False)
        self.ui.pushButton_MotorSimulate.setStyleSheet("background-color: red")
            
    def SetVelocity(self):
        '''
        '''
        Mode = self.ui.comboBox.currentIndex()
        SPEEDst = Pxep.SPDPARAM(
            Mode,
            self.StartSpeed,
            self.StartSpeed,
            self.HighSpeed,
            self.Uptime,
            self.SshapeRatio,
            self.Downtime,
            self.SshapeRatio,
            self.OverSpeed,
            0, 0, 0,
        )
        self.scl.wSetSpeedParameter(SPEEDst)

    def Arrow_keys(self):
        '''
        '''
        sender = self.sender()
        if self.ui.radioButton_Marker.isChecked():
            if sender == self.ui.pushButton_Up:
                self.marker_direction = 0
            elif sender == self.ui.pushButton_Down:
                self.marker_direction = 1
            elif sender == self.ui.pushButton_Left:
                self.marker_direction = 2
            elif sender == self.ui.pushButton_Right:
                self.marker_direction = 3
            UIFunctions.marker_update_to_thread(self)
            
        elif self.ui.radioButton_Stage.isChecked():
            if sender == self.ui.pushButton_Up:
                _, _ = self.scl.wDriveStart(self.pulse, 3) #0
            elif sender == self.ui.pushButton_Down:
                _, _ = self.scl.wDriveStart(self.pulse, 2) #1
            elif sender == self.ui.pushButton_Left:
                _, _ = self.scl.wDriveStart(self.pulse, 1) #2
            elif sender == self.ui.pushButton_Right:
                _, _ = self.scl.wDriveStart(self.pulse, 0) #3
            elif sender == self.ui.pushButton_UpLeft:
                _, _ = self.scl.wDriveStart(self.pulse, 7) #4
            elif sender == self.ui.pushButton_UpRight:
                _, _ = self.scl.wDriveStart(self.pulse, 5) #5
            elif sender == self.ui.pushButton_LowLeft:
                _, _ = self.scl.wDriveStart(self.pulse, 6) #6
            elif sender == self.ui.pushButton_LowRight:
                _, _ = self.scl.wDriveStart(self.pulse, 4) #7
            elif sender == self.ui.pushButton_Stop:
                self.scl.wDriveStop()
        
    def setPixelCount(self):
        '''
        '''
        pass

    def SaveTxt(self):
        '''
        '''
        pass