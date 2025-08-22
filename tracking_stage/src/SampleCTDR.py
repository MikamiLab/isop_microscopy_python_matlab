# ###########################################################################
#
# Pxep.pyとpxep.dllを本モジュールと同じフォルダに入れてください。
#
# ###########################################################################

import ctypes
import time
import Pxep

# 関数実行結果を16進変換して表示
def ShowResult(Message,rel):
    if rel >= 0:    # 正数の場合
        print(Message,'  Result=0x{:08X}'.format(rel))
    else:           # 負数の場合
        print(Message,'Result=0x{:08X}'.format(0xFFFFFFFF + rel + 0x01))

wBSN = 0
wID = 1 # SyCON.NETでアドレスを1設定した場合、1軸目のwID=1,2軸目のwID=2です。

# ネットワーク初期化（成功）
pbID = (ctypes.c_uint8*32)()
ret = Pxep.PxepwAPI.RcmcwOpen(wBSN,1,pbID)
ShowResult("RcmcwOpen",ret)
for i in range(len(pbID)):
    if (pbID[i] != 0xFF):
        print('pbID[{0:d}]={1:02X} '.format(i,pbID[i]))

# バージョン情報読み出し（成功）
pdwMasterInfo = (ctypes.c_int32*10)()
ret = Pxep.PxepwAPI.RcmcwGetVersion(wBSN,pdwMasterInfo)
ShowResult("RcmcwGetVersion",ret)
for i in range(4):
    print('pdwMasterInfo[{0:d}]={1:08X} '.format(i,pdwMasterInfo[i]))

# アラームクリア（成功）
ret = Pxep.PxepwAPI.RcmcwALMCLR(wBSN,wID)
ShowResult("RcmcwALMCLR",ret)

# アラームクリアまで1s待機→実際のスレーブに合わせて待機時間調整
time.sleep(1)

# ソフトリミット無効設定（成功）
ret= Pxep.PxepwAPI.RcmcwSetSoftLimit(wBSN,wID,500,0,3,0)
ShowResult("RcmcwSetSoftLimit",ret)

# ハードリミットアクティブレベル設定,尚、デフォルト値が0のため、アクティブレベルLの場合は、設定する必要なし 
objNo=0x2000   # wID=2で2軸目の場合は、0x2400設定
setdata = (ctypes.c_uint32*1)()
setdata[0] = 0
ret = Pxep.PxepwAPI.RcmcwObjectControl(wBSN,wID,0,objNo,0,2,setdata)

# サーボON  
ret = Pxep.PxepwAPI.RcmcwSetServoONOFF(wBSN,wID,1,0)
ShowResult("RcmcwSetServoONOFF",ret)

# 1s待機→実際のスレーブに合わせて待機時間調整
time.sleep(1)

# 原点復帰　現在位置 = 0 設定   
HomeSt = Pxep.HOMEPARAM(35,35,1,0,0,0,0,0)
ret = Pxep.PxepwAPI.RcmcwHomeStart(wBSN,wID,HomeSt)
ShowResult("RcmcwHomeStart",ret)

# 速度設定（成功）
AxisSpeed = Pxep.SPDPARAM(0,2000,2000,30000,100,0,0,0,0,0,0,0)
ret = Pxep.PxepwAPI.RcmcwSetSpeedParameter(wBSN,wID,AxisSpeed)
ShowResult("RcmcwSetSpeedParameter",ret)

# ドライブスタート  
ret = Pxep.PxepwAPI.RcmcwDriveStart(wBSN,wID,1,0,50000)
ShowResult("RcmcwDriveStart",ret)

# ドライブステータス取得（成功）
pdwStatus = (ctypes.c_uint32*1)()
ret = Pxep.PxepwAPI.RcmcwGetDriveStatus(wBSN,wID,pdwStatus)
ShowResult("RcmcwGetDriveStatus",ret)
print('DriveStatus =0x{:08X} '.format(pdwStatus[0]))


