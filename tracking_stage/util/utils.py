import cv2
import numpy as np
import time
import ctypes
from util import Pxep
import csv


def read_image(image_path):
    '''
    Read an image from the specified path
    '''
    return cv2.imread(image_path)

def write_log_init(log_path):
    '''
    Save the log to the specified path
    '''
    log_path.write(
        "frame"+","+
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

def write_log(
        log_path, 
        frame_num, 
        time_delta, 
        stageX, 
        stageY, 
        delta_t, 
        delta_d, 
        velocity, 
        detection_x, 
        detection_y, 
        gap_x, 
        gap_y, 
        gap_x_Micro, 
        gap_y_Micro, 
        gap_x_Pulse, 
        gap_y_Pulse, 
        Err, 
        predictd, 
        radius,
        asobi,
    ):
    '''
    Save the log to the specified path
    '''
    log_path.write(
        str(frame_num)+","+
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
        str(radius)+","+
        str(asobi)+"\n"
    )


class StageController():
    '''
    Initialize the stage controller
    '''
    def __init__(self):
        self.wBSN = 0
        self.wIDx = 2
        self.wIDy = 1
        self.wID_map = {0: self.wIDy, 1: self.wIDy, 2: self.wIDx, 3: self.wIDx}
        self.pdwDataInfo = (ctypes.c_int32*10)()
        self.pdwDatax = (ctypes.c_int32*7)()
        self.pdwDatay = (ctypes.c_int32*7)()
        self.pdwNowSpeedx = (ctypes.c_int32*4)()
        self.pdwNowSpeedy = (ctypes.c_int32*4)()
        self.pdwStatusx = (ctypes.c_int32*1)()
        self.pdwStatusy = (ctypes.c_int32*1)()
    
    def initStage(self):
        # Initialize network of stage controller
        pbID = (ctypes.c_uint8 * 32)()
        ret = Pxep.PxepwAPI.RcmcwOpen(self.wBSN, 1, pbID)
        time.sleep(1)

        # Clear alarms
        ret = Pxep.PxepwAPI.RcmcwALMCLR(self.wBSN, self.wIDx)
        ret = Pxep.PxepwAPI.RcmcwALMCLR(self.wBSN, self.wIDy)

        # Disable software limits
        ret = Pxep.PxepwAPI.RcmcwSetSoftLimit(self.wBSN, self.wIDx, 500, 0, 3, 0)
        ret = Pxep.PxepwAPI.RcmcwSetSoftLimit(self.wBSN, self.wIDy, 500, 0, 3, 0)

        # Set hardware limit active level, if default value is 0, no need to set when active level is L
        objNo = 0x2000   # Set to 0x2000 for axis 1 when wID=1
        setdata = (ctypes.c_uint32 * 1)()
        setdata[0] = 0
        ret = Pxep.PxepwAPI.RcmcwObjectControl(self.wBSN, self.wIDy, 0, objNo, 0, 2, setdata)
        objNo = 0x2400   # Set to 0x2400 for axis 2 when wID=2
        setdata = (ctypes.c_uint32 * 1)()
        setdata[0] = 0
        ret = Pxep.PxepwAPI.RcmcwObjectControl(self.wBSN, self.wIDx, 0, objNo, 0, 2, setdata)
        
        # Turn servo ON
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN, self.wIDx, 1, 0)
        ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN, self.wIDy, 1, 0)
        
        # Set current position as home position = 0
        HomeSt = Pxep.HOMEPARAM(35, 35, 1, 0, 0, 0, 0, 0)
        ret = Pxep.PxepwAPI.RcmcwHomeStart(self.wBSN, self.wIDx, HomeSt)
        ret = Pxep.PxepwAPI.RcmcwHomeStart(self.wBSN, self.wIDy, HomeSt)
        
        # Set speed
        # AxisSpeed = Pxep.SPDPARAM(1, 100, 100, 101, 100, 40, 100, 40, 3, 0, 0, 0)
        AxisSpeed = Pxep.SPDPARAM(1, 600, 600, 35000, 100, 40, 100, 40, 3, 0, 0, 0)
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(self.wBSN, self.wIDx, AxisSpeed)
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(self.wBSN, self.wIDy, AxisSpeed)
        time.sleep(0.1)

        # Start drive
        ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN, self.wIDx, 1, 0, 5000)
        ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN, self.wIDy, 1, 0, 5000)

        # Turn servo OFF
        # ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN, self.wIDx, 0, 0)
        # ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN, self.wIDy, 0, 0)
        # print(ret)


    #### => START: Initializing function
    def wClose(self):
        ret = Pxep.PxepwAPI.RcmcwClose(self.wBSN)
    
    def wServoONOFF(self, flag):
        if flag == True:
            ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDx,1,0)
            ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN,self.wIDy,1,0)
        else:
            ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN, self.wIDx, 0, 0)
            ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(self.wBSN, self.wIDy, 0, 0)
        self.ShowResult("RcmcwSetServoONOFF",ret)
    #### <= END: Initializing function


    #### => START: Status and Information Retrieval Function
    def wGetTimeOutInfo(self):
        '''
        0:通信状況, 1:CPU使用率, 2:指令更新周期ns, else:無効
        '''
        ret = Pxep.PxepwAPI.RcmcwGetTimeOutInfo(self.wBSN,self.pdwDataInfo)
        # for i in range(3):
        #     print('pdwMasterInfo[{0:d}]={1:08X} '.format(i,pdwDataInfo[i]))
        return self.pdwDataInfo[0], self.pdwDataInfo[1], self.pdwDataInfo[2]

    def wGetCounter(self):
        '''
        1:実位置_NET, else:無効
        '''
        ret = Pxep.PxepwAPI.RcmcwGetCounter(self.wBSN,self.wIDx,self.pdwDatax)
        ret = Pxep.PxepwAPI.RcmcwGetCounter(self.wBSN,self.wIDy,self.pdwDatay)
        # for i in range(3):
        #     print('pdwMasterInfo[{0:d}]={1:08X} '.format(i,pdwData[i]))
        return self.pdwDatax[1], self.pdwDatay[1]
    #### <= END: Status and Information Retrieval Function


    #### => START: Speed function
    def wSetSpeedParameter(self,SPEEDst):
        '''
        SPEEDst = (
            dwMode,                  # 加減速モード
            dwStartSpeed,            # 開始速度(PPS)
            dwStopSpeed,             # 停止速度(PPS)
            dwHighSpeed,             # 目標速度(PPS)
            dwUpTime1,               # 加速時間(ms)
            dwUpSRate,               # 加速S字比率(0-100%)
            dwDownTime1,             # 減速時間(ms)
            dwDownSRate,             # 減速S字比率(0-100%)
            bOverONOFF,              # 速度オーバーライド
            dtriangleconst,          # 三角駆動時間設定
            dwUpPls,                 # 必要な加速パルス数(Pulse)
            dwDownPls,               # 必要な減速パルス数(Pulse)(Mode,
        )
        '''
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(self.wBSN,self.wIDx,SPEEDst)
        self.ShowResult("RcmcwSetSpeedParameter",ret)
        ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(self.wBSN,self.wIDy,SPEEDst)
        self.ShowResult("RcmcwSetSpeedParameter",ret)
        
    def wSpeedOverride(self, dwHighSpeed):
        ret = Pxep.PxepwAPI.RcmcwSpeedOverride(self.wBSN,self.wIDx,dwHighSpeed)
        ret = Pxep.PxepwAPI.RcmcwSpeedOverride(self.wBSN,self.wIDy,dwHighSpeed)
    
    def wGetNowSpeed(self):
        ret = Pxep.PxepwAPI.RcmcwGetNowSpeed(self.wBSN,self.wIDx,self.pdwNowSpeedx)
        ret = Pxep.PxepwAPI.RcmcwGetNowSpeed(self.wBSN,self.wIDy,self.pdwNowSpeedy)
        return self.pdwNowSpeedx[2], self.pdwNowSpeedy[2]
    #### <= END: Speed function


    #### => START: 5.5.x Drive function
    def wDriveStart(self, pulse, mode):
        '''
        mode: 0=Up, 1=Down, 2=Left, 3=Right, 4=UpperLeft, 5=UpperRight, 6=LowerLeft, 7=LowerRight
        '''
        if mode in {0, 1, 2, 3}:
            wID = self.wID_map[mode]
            Pulse = -pulse if mode in {0, 3} else pulse
            
            print(self.wBSN, wID, Pulse, mode)
            ret = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,wID,1,0,Pulse)
            return ret, None
        else:
            xPulse = -pulse if mode in {5, 7} else pulse
            yPulse = -pulse if mode in {4, 5} else pulse

            retx = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDx,1,0,xPulse)
            rety = Pxep.PxepwAPI.RcmcwDriveStart(self.wBSN,self.wIDy,1,0,yPulse)
            return retx, rety
    
    def wDriveStop(self):
        ret = Pxep.PxepwAPI.RcmcwDriveStop(self.wBSN,self.wIDx,0)
        ret = Pxep.PxepwAPI.RcmcwDriveStop(self.wBSN,self.wIDy,0)
    
    def wPulseOverride(self, xPulse, yPulse):
        ret = Pxep.PxepwAPI.RcmcwPulseOverride(self.wBSN,self.wIDx,xPulse)
        ret = Pxep.PxepwAPI.RcmcwPulseOverride(self.wBSN,self.wIDy,yPulse)
    
    def wGetDriveStatus(self):
        ret = Pxep.PxepwAPI.RcmcwGetDriveStatus(self.wBSN,self.wIDx,self.pdwStatusx)
        ret = Pxep.PxepwAPI.RcmcwGetDriveStatus(self.wBSN,self.wIDy,self.pdwStatusy)
        return self.pdwStatusx[0], self.pdwStatusy[0]
    #### <= END: Drive function
    
    def ShowResult(self,Message,rel):
        print(rel)
        if rel >= 0:    # 正数の場合
            print(Message,'Result=0x{:08X}'.format(rel))
        else:           # 負数の場合
            print(Message,'Result=0x{:08X}'.format(0xFFFFFFFF + rel + 0x01))



class OpticalFlow():
    '''
    Optical flow class
    '''
    def __init__(self):
        '''
        Initialize the optical flow class
        '''
        # Maximum number of feature points
        self.MAX_FEATURE_NUM = 1
        # Termination criteria for the iterative algorithm
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    def onMouse(self, previous_x, previous_y, radius):
        '''
        Specify feature points with mouse clicks
        Toggle feature point: remove if near, add if not
        '''
        # Add the first feature point
        if self.features is None:
            self.addFeature(self, previous_x, previous_y)
            return

        # Search for an existing feature point near the clicked area
        index = self.getFeatureIndex(self, previous_x, previous_y, radius)

        # If there is an existing feature point near the clicked area, remove the existing feature point
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)

        # If there is no existing feature point near the clicked area, add a new feature point
        else:
            self.addFeature(self, previous_x, previous_y)

    def getFeatureIndex(self, x, y, radius):
        '''
        Get the index of an existing feature point within the specified radius
        If there is no feature point within the specified radius, return index = -1
        '''
        index = -1

        # No feature points are registered
        if self.features is None:
            return index

        max_r2 = radius ** 2
        index = 0

        for point in self.features:
            dx = x - point[0][0]
            dy = y - point[0][1]
            r2 = dx ** 2 + dy ** 2
            if r2 <= max_r2:
                # This feature point is within the specified radius
                return index
            else:
                # This feature point is outside the specified radius
                index += 1

        # All feature points are outside the specified radius
        return -1

    def addFeature(self, x, y):
        '''
        Add a new feature point
        '''
        # Create ndarray and register the coordinates of the feature point
        self.features = np.array([[[x, y]]], np.float32)
        self.status = np.array([1])

        # Refine the feature point
        cv2.cornerSubPix(self.gray_next, self.features, (40, 40), (-1, -1), self.CRITERIA)

        
        # # If the number of feature points exceeds the maximum limit
        # elif len(self.features) >= self.MAX_FEATURE_NUM:
        #     print("max feature num over: " + str(self.MAX_FEATURE_NUM))

        # # Add a new feature point
        # else:
        #     # Append the coordinates of the feature point to the existing ndarray
        #     self.features = np.append(self.features, [[[x, y]]], axis = 0).astype(np.float32)
        #     self.status = np.append(self.status, 1)
        #     # Refine the feature point
        #     cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), self.CRITERIA)
        
    def refreshFeatures(self):
        '''
        Keep only valid feature points
        '''
        # No feature points are registered
        if self.features is None:
            return

        # Check all status
        i = 0
        while i < len(self.features):
            # Feature point is not recognized
            if self.status[i] == 0:
                # Remove from the existing ndarray
                self.features = np.delete(self.features, i, 0)
                self.status = np.delete(self.status, i, 0)
                i -= 1

            i += 1


class KalmanFilter():
    '''
    Kalman filter class
    '''
    def __init__(self):
        '''
        Initialize the Kalman filter class
        '''
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)

    def predict(self, coordX, coordY):
        '''
        Predict the next position of the object
        '''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
    