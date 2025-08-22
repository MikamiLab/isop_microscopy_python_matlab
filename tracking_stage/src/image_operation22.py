import cv2
import numpy as np
    

# 特徴点の最大数
MAX_FEATURE_NUM = 1
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

# マウスクリックで特徴点を指定する
#     クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
#     クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
def onMouse(previous_x,previous_y, radius, features, status, gray_next):
    # 最初の特徴点追加
    if features is None:
        addFeature(previous_x, previous_y, gray_next)
        return

    # 既存の特徴点が近傍にあるか探索
    index = getFeatureIndex(previous_x, previous_y, radius, features)

    # クリックされた近傍に既存の特徴点があるので既存の特徴点を削除する
    if index >= 0:
        features = np.delete(features, index, 0)
        status = np.delete(status, index, 0)

    # クリックされた近傍に既存の特徴点がないので新規に特徴点を追加する
    else:
        addFeature(previous_x, previous_y, gray_next)



# 指定した半径内にある既存の特徴点のインデックスを１つ取得する
#     指定した半径内に特徴点がない場合 index = -1 を応答
def getFeatureIndex(x, y, radius, features):
    index = -1

    # 特徴点が１つも登録されていない
    if features is None:
        return index

    max_r2 = radius ** 2
    index = 0
    
    for point in features:
        dx = x - point[0][0]
        dy = y - point[0][1]
        r2 = dx ** 2 + dy ** 2
        if r2 <= max_r2:
            # この特徴点は指定された半径内
            return index
        else:
            # この特徴点は指定された半径外
            index += 1

    # 全ての特徴点が指定された半径の外側にある
    return -1


# 特徴点を新規に追加する
def addFeature(x, y, gray_next):
    # ndarrayの作成し特徴点の座標を登録
    features = np.array([[[x, y]]], np.float32)
    status = np.array([1])

    # 特徴点を高精度化
    cv2.cornerSubPix(gray_next, features, (40, 40), (-1, -1), CRITERIA)

    """
    # 特徴点の最大登録個数をオーバー
    elif len(self.features) >= MAX_FEATURE_NUM:
        print("max feature num over: " + str(MAX_FEATURE_NUM))

    # 特徴点を追加登録
    else:
        # 既存のndarrayの最後に特徴点の座標を追加
        self.features = np.append(self.features, [[[x, y]]], axis = 0).astype(np.float32)
        self.status = np.append(self.status, 1)
        # 特徴点を高精度化
        cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)
    """


# 有効な特徴点のみ残す
def refreshFeatures(features, status):
    # 特徴点が未登録
    if features is None:
        return

    # 全statusをチェックする
    i = 0
    while i < len(features):

        # 特徴点として認識できず
        if status[i] == 0:
            # 既存のndarrayから削除
            features = np.delete(features, i, 0)
            status = np.delete(status, i, 0)
            i -= 1

        i += 1