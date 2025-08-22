import cv2
import sys, os, datetime
import numpy as np
import pandas as pd
import re

now = datetime.datetime.now()

#videoの読み込み
video_path = r"C:\Users\Hikaru Shishido\Desktop\20240208_cam2_004\MAX_20240208_cam2_004.avi"
path = r"C:\Users\Hikaru Shishido\Desktop\20240208_cam2_004"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("読み込み失敗")
    sys.exit()

# try:
#     os.mkdir(os.getcwd()+ "/log/"+now.strftime("%Y%m%d%H%M%S"))
# except FileExistsError:
#     pass

#video保存先の作成
# filename = now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"video_LocalOptical.avi"
filename = path+"/video_LocalOptical.avi"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20
size = (512, 512)
writer = cv2.VideoWriter(filename, fourcc, fps, size)

# feature_log_name = now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"feature.csv"
feature_log_name = path+"/feature.csv"
feature_log = open(feature_log_name,"a")
# status_log_name = now.strftime(os.getcwd()+ "/log/" + "%Y%m%d%H%M%S/")+"status.csv"
status_log_name = path+"/status.csv"
status_log = open(status_log_name,"a")
#log.write("frame,time,x(pixel),y(pixel),radius(pixel),asobi(pixel),Δt(s),Δd(μm),track_v(μm/s),err(μm),sum_x(pixel),sum_y(pixel)\n")


ST_param = dict( maxCorners   = 50,   #特徴点の最大数
                        qualityLevel = 0.2,  #特徴点を選択する閾値
                        minDistance  = 7,    #特徴点間の最小距離
                        blockSize    = 7)    #特徴点の計算に使う周辺領域のサイズ


LK_param = dict( winSize  = (50, 50),  #オプティカルフローの推定の計算に使う周辺領域サイズ（小：敏感、大：鈍感）
                maxLevel = 3,         #ピラミッド数
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 0.03))  #繰り返しの終了条件

#color = np.random.randint(0,255,(100,3))


# First frame
ret, frame = cap.read()
# mask_black = np.zeros(frame.shape, dtype=np.uint8)

print(frame.shape)
# print(mask_black.shape)

# cv2.rectangle(mask_black, (139, 151), (470, 349), (255, 255, 255), -1)
# frame = cv2.bitwise_and(frame, mask_black)

old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ft1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **ST_param )

# 軌跡描画用のマスクを生成
# mask = np.zeros_like(frame)

# 撮像範囲マスクを生成


while True:
    ret, frame = cap.read()

    if ret:
        # mask_black = np.zeros(frame.shape, dtype=np.uint8)
        # frame = cv2.bitwise_and(frame, mask_black)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow
        ft2, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, ft1, None, **LK_param )

        #オプティカルフローが計算できた場合
        if status is not None:
            #オプティカルフローを検出した特徴点を取得
            g_old = ft1[status == 1]
            g_new = ft2[status == 1]

            #特徴点とオプティカルフローをマスクに描画する
            for i, (new, old) in enumerate(zip(g_new, g_old)):
                #1フレーム目の特徴点座標
                x1, y1 = old.ravel()
                #2フレーム目の特徴点座標
                x2, y2 = new.ravel()

                x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)

                #軌跡をマスクに描画
                #mask = cv2.line(mask, (int(x2), int(y2)), (int(x1), int(y1)), [0, 0, 200], 2)

                #現フレームにオプティカルフローを描画
                frame = cv2.circle(frame, (int(x2), int(y2)), 5,  [0, 0, 200], -1)

            #フレームとマスクを合成
            #img = cv2.add(frame, mask)
            writer.write(frame)
            feature_log.write(str(ft2)+"\n"+"N"+"\n")
            status_log.write(str(status)+"\n"+"N"+"\n")


            old_gray = frame_gray.copy()
            ft1 = g_new.reshape(-1,1,2)

    else:
        break

cap.release()
writer.release()
feature_log.close()
status_log.close()

# def transform_cell(cell):
#     cleaned = re.sub(r'N', '', cell)
#     return cleaned

# df = pd.read_csv(feature_log_name)
# df = df.dropna()
# df = df.applymap(transform_cell)
# df.to_csv(feature_log_name, index=False)



# df2 = pd.read_csv(status_log_name)
# df2 = df2.applymap(transform_cell)
# df2.to_csv(status_log_name, index=False)

print("Finished")
