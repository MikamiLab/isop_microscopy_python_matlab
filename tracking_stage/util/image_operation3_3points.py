import cv2
import numpy as np

# 特徴点の最大数
MAX_FEATURE_NUM = 1
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

# マウスクリックで特徴点を指定する
#     クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
#     クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
def onMouse(self,previous_x,previous_y,radius):
    # 最初の特徴点追加
    if self.features is None:
        addFeature(self,previous_x, previous_y)
        return

    # 既存の特徴点が近傍にあるか探索
    index = getFeatureIndex(self,previous_x, previous_y, radius)

    # クリックされた近傍に既存の特徴点があるので既存の特徴点を削除する
    if index >= 0:
        self.features = np.delete(self.features, index, 0)
        self.status = np.delete(self.status, index, 0)

    # クリックされた近傍に既存の特徴点がないので新規に特徴点を追加する
    else:
        addFeature(self,previous_x, previous_y)

# 指定した半径内にある既存の特徴点のインデックスを１つ取得する
#     指定した半径内に特徴点がない場合 index = -1 を応答
def getFeatureIndex(self, x, y, radius):
    index = -1

    # 特徴点が１つも登録されていない
    if self.features is None:
        return index

    max_r2 = radius ** 2
    index = 0

    for point in self.features:
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
def addFeature(self, x, y):
    # ndarrayの作成し特徴点の座標を登録
    self.features = np.array([[[x, y]]], np.float32)
    self.status = np.array([1])

    # 特徴点を高精度化
    cv2.cornerSubPix(self.gray_next, self.features, (40, 40), (-1, -1), CRITERIA)

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
def refreshFeatures(self):
    # 特徴点が未登録
    if self.features is None:
        return

    # 全statusをチェックする
    i = 0
    while i < len(self.features):

        # 特徴点として認識できず
        if self.status[i] == 0:
            # 既存のndarrayから削除
            self.features = np.delete(self.features, i, 0)
            self.status = np.delete(self.status, i, 0)
            i -= 1

        i += 1