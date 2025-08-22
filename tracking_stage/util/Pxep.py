
# ############################################################################
#
#  PXEP-CN  API.
#
#  Copyright CosmoTechs Co.,Ltd.
#
#  更新履歴：
#  2023.09.21   初版
#
# ############################################################################

import ctypes

# 速度パラメータ
class SPDPARAM(ctypes.Structure):
    _fields_ = [("dwMode", ctypes.c_int),                   # 加減速モード
                ("dwStartSpeed", ctypes.c_int),             # 開始速度(PPS)
                ("dwStopSpeed", ctypes.c_int),              # 停止速度(PPS)
                ("dwHighSpeed", ctypes.c_int),              # 目標速度(PPS)
                ("dwUpTime1", ctypes.c_int),                # 加速時間(ms)
                ("dwUpSRate", ctypes.c_int),                # 加速S字比率(0-100%)
                ("dwDownTime1", ctypes.c_int),              # 減速時間(ms)
                ("dwDownSRate", ctypes.c_int),              # 減速S字比率(0-100%)
                ("bOverONOFF", ctypes.c_int),               # 速度オーバーライド
                ("dtriangleconst", ctypes.c_int),           # 三角駆動時間設定
                ("dwUpPls", ctypes.c_int),                  # 必要な加速パルス数(Pulse)
                ("dwDownPls", ctypes.c_int)]                # 必要な減速パルス数(Pulse)

# 補間パラメータ
class IPTBLDATA(ctypes.Structure):
    _fields_ = [("IpObjectPosi_Main", ctypes.c_int),        # 主軸の目標位置
                ("AssistData_Main", ctypes.c_int),          # 主軸の補助データ
                ("Parameter_Main", ctypes.c_int),           # 主軸のパラメータ
                ("IpObjectPosi_A", ctypes.c_int),           # 従軸Aの目標位置
                ("AssistData_A", ctypes.c_int),             # 従軸Aの補助データ
                ("Parameter_A", ctypes.c_int),              # 従軸Aのパラメータ
                ("IpObjectPosi_B", ctypes.c_int),           # 従軸Bの目標位置
                ("AssistData_B", ctypes.c_int),             # 従軸Bの補助データ
                ("Parameter_B", ctypes.c_int),              # 従軸Bのパラメータ
                ("IpObjectPosi_C", ctypes.c_int),           # 従軸Cの目標位置
                ("AssistData_C", ctypes.c_int),             # 従軸Cの補助データ
                ("Parameter_C", ctypes.c_int),              # 従軸Cのパラメータ
                ("IpObjectPosi_D", ctypes.c_int),           # 従軸Dの目標位置
                ("AssistData_D", ctypes.c_int),             # 従軸Dの補助データ
                ("Parameter_D", ctypes.c_int),              # 従軸Dのパラメータ
                ("IpObjectPosi_E", ctypes.c_int),           # 従軸Eの目標位置
                ("AssistData_E", ctypes.c_int),             # 従軸Eの補助データ
                ("Parameter_E", ctypes.c_int),              # 従軸Eのパラメータ
                ("IpObjectPosi_F", ctypes.c_int),           # 従軸Fの目標位置
                ("AssistData_F", ctypes.c_int),             # 従軸Fの補助データ
                ("Parameter_F", ctypes.c_int),              # 従軸Fのパラメータ
                ("IpObjectPosi_G", ctypes.c_int),           # 従軸Gの目標位置
                ("AssistData_G", ctypes.c_int),             # 従軸Gの補助データ
                ("Parameter_G", ctypes.c_int),              # 従軸Gのパラメータ
                ("IpObjectPosi_H", ctypes.c_int),           # 従軸Hの目標位置
                ("AssistData_H", ctypes.c_int),             # 従軸Hの補助データ
                ("Parameter_H", ctypes.c_int),              # 従軸Hのパラメータ
                ("WaitTimeMS", ctypes.c_int),               # 待ち時間＿連続補間では無い時、次の補間が開始するまでの待ち時間
                ("CTO_ID", ctypes.c_int),                   # CTO出力_対象MAC-ID        設定範囲：0~ 31
                ("CTO_Mask", ctypes.c_int),                 # CTO出力_マスクデータ      設定範囲：0：マスクする    1：マスクしない
                ("CTO_Data", ctypes.c_int),                 # CTO出力_出力データ        設定範囲：0～FFFFFFFFｈ
                ("MasterOUT_Mask", ctypes.c_int),           # マスタ汎用OUT出力_マスクデータ  設定範囲：0：マスクする    1：マスクしない
                ("MasterOUT_Data", ctypes.c_int),           # マスタ汎用OUT出力_出力データ    設定範囲：0～FFFｈ
                ("DA_Write", ctypes.c_int)]                 # DAボード出力

# 機能拡張補間パラメータ
class IPTBLDATAHI(ctypes.Structure):
    _fields_ = [("IpObjectPosi_Main", ctypes.c_int),        # 主軸目標位置
                ("AssistData_Main", ctypes.c_int),          # 主軸補助データ：目標速度
                ("Parameter_Main", ctypes.c_int),           # 主軸パラメータ
                ("IpObjectPosi_A", ctypes.c_int),           # 従軸Aの目標位置
                ("AssistData_A", ctypes.c_int),             # 従軸Aの補助データ
                ("Reserve01", ctypes.c_int),                # 予約
                ("IpObjectPosi_B", ctypes.c_int),           # 従軸Bの目標位置
                ("AssistData_B", ctypes.c_int),             # 従軸Bの補助データ
                ("Reserve02", ctypes.c_int),                # 予約
                ("IpObjectPosi_C", ctypes.c_int),           # 従軸Cの目標位置
                ("AssistData_C", ctypes.c_int),             # 従軸Cの補助データ
                ("Reserve03", ctypes.c_int),                # 予約
                ("IpObjectPosi_D", ctypes.c_int),           # 従軸Dの目標位置
                ("AssistData_D", ctypes.c_int),             # 従軸Dの補助データ
                ("Reserve04", ctypes.c_int),                # 予約
                ("Reserve05", ctypes.c_int),                # 予約
                ("Reserve06", ctypes.c_int),                # 予約
                ("Reserve07", ctypes.c_int),                # 予約
                ("Reserve08", ctypes.c_int),                # 予約
                ("Reserve09", ctypes.c_int),                # 予約
                ("Reserve10", ctypes.c_int),                # 予約
                ("Reserve11", ctypes.c_int),                # 予約
                ("Reserve12", ctypes.c_int),                # 予約
                ("Reserve13", ctypes.c_int),                # 予約
                ("AutoCornerSpd", ctypes.c_int),            # 自動コーナー減速機能が有効の場合、減速速度
                ("AutoCornerArcS", ctypes.c_int),           # 自動コーナー減速機能が有効の場合、減速開始角度
                ("AutoCornerArcE", ctypes.c_int),           # 自動コーナー減速機能が有効の場合、減速終了角度
                ("WaitTimeMS", ctypes.c_int),               # 待ち時間      設定範囲：0~ 10,000,000（単位：ms)
                ("Reserve14", ctypes.c_int),                # 予約
                ("GOut_Mask", ctypes.c_int),                # 汎用出力_マスクデータ   設定範囲：0~FFFFh
                ("GOut_Data", ctypes.c_int),                # 汎用出力_出力データ     設定範囲：0~FFFFh
                ("GOut_DelayTs", ctypes.c_int),             # 汎用出力遅延時間
                ("GOut_DelayTe", ctypes.c_int),             # 汎用出力維持時間
                ("DA_Write", ctypes.c_int)]                 # DAボード出力

# 補間バッファ設定関連パラメータ
class BUFDATA(ctypes.Structure):
    _fields_ = [("StartTblArea", ctypes.c_int),             # 補間テーブル範囲指定＿開始ステップ（設定範囲0~5000）
                ("EndTblArea", ctypes.c_int),               # 補間テーブル範囲指定＿終了ステップ（設定範囲0~5000）
                ("System", ctypes.c_int),                   # システム設定
                ("PosiKind", ctypes.c_int),                 # 補間テーブル登録する位置データの座標の種別設定
                ("OptionSet", ctypes.c_int)]                # オプション機能の有効・無効を設定

# 機能拡張補間 汎用出力の各ビットの割当情報
class OUTASSIGN(ctypes.Structure):
    _fields_ = [("Type", ctypes.c_int),                     # 割り当てる出力信号のタイプを設定
                ("MacID", ctypes.c_int),                    # Type=１~2の時、割り当てる対象アドレスを設定（設定範囲0~31）
                ("No", ctypes.c_int)]                       # 割り当てる出力信号を設定

# 原点復帰パラメータ
class HOMEPARAM(ctypes.Structure):
    _fields_ = [("dwHomeMode", ctypes.c_int),               # 原点復帰方法
                ("dwHomeOffsetMode", ctypes.c_int),         # 原点オフセット方法
                ("dwHomeOffsetType", ctypes.c_int),         # 原点オフセットタイプ
                ("dwHomeOffsetValue", ctypes.c_int),        # 原点オフセット値  (単位：Pulse)
                ("dwHighSpeed", ctypes.c_int),              # 高速検知速度      (単位：PPS)
                ("dwLowSpeed", ctypes.c_int),               # 低速検知速度      (単位：PPS)
                ("dwUpTime", ctypes.c_int),                 # 加速時間          (単位：ms)
                ("dwDownTime", ctypes.c_int)]               # 減速時間          (単位：ms)

# 軸連動パラメータ
class LINKPARAM(ctypes.Structure):
    _fields_ = [("LinkID", ctypes.c_int),                   # 連動軸のMAC-ID選択       設定範囲：0~31
                ("Mode", ctypes.c_int),                     # 連動軸の位置情報を設定   0:指令位置  1:実位置
                ("Dir", ctypes.c_int),                      # 連動方向                 0:+方向  1:-方向
                ("Division_Numerator", ctypes.c_int),       # 連動分周比分子           設定範囲：0~4294967295
                ("Division_Denominator", ctypes.c_int)]     # 連動分周比分母           設定範囲：1~4294967295

# トリガー設定パラメータ
class TRIGGERPARAM(ctypes.Structure):
    _fields_ = [("dwTrgSel", ctypes.c_int),                 # トリガー信号の選択
                ("dwMInSig", ctypes.c_int),                 # MasterIn信号の選択
                ("dwMInEdge", ctypes.c_int),                # MasterIn信号のエッジレベルを選択
                ("dwSInID", ctypes.c_int),                  # SlaveIn信号に割り当てるMAC-ID
                ("dwSInSig", ctypes.c_int),                 # SlaveIn信号の選択
                ("dwSInLogic", ctypes.c_int),               # SlaveIn信号の論理
                ("dwAction", ctypes.c_int),                 # トリガー発生時のアクション設定
                ("dwMOutSig", ctypes.c_int),                # 出力したいMasterOut信号選択
                ("dwMOutdata", ctypes.c_int),               # MasterOut信号の出力レベル設定
                ("dwEvent", ctypes.c_int),                  # 実行したい関数登録
                ("dwdata0", ctypes.c_int),                  # dwEventの補助データ0
                ("dwdata1", ctypes.c_int),                  # dwEventの補助データ1
                ("dwdata2", ctypes.c_int),                  # dwEventの補助データ2
                ("dwdata3", ctypes.c_int),                  # dwEventの補助データ3
                ("dwdata4", ctypes.c_int)]                  # dwEventの補助データ4

# CIA402製品の0x60FDでの各センサーのBitNo設定
class SENSBITNOPARAM(ctypes.Structure):
    _fields_ = [("BitNo_INP", ctypes.c_int),                # 0x60FDでのINP信号のBit番号
                ("BitNo_EMG", ctypes.c_int),                # 0x60FDでのEMG信号のBit番号
                ("BitNo_Z", ctypes.c_int),                  # 0x60FDでのZ信号のBit番号
                ("BitNo_HOME", ctypes.c_int),               # 0x60FDでのHOME信号のBit番号
                ("BitNo_POT", ctypes.c_int),                # 0x60FDでのPOT信号のBit番号
                ("BitNo_NOT", ctypes.c_int)]                # 0x60FDでのNOT信号のBit番号

PxepwAPI=ctypes.cdll.LoadLibrary('./util/pxep')