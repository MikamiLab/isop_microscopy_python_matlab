#%%
import tifffile, struct, os, cv2, datetime, time
from scipy.signal import butter, filtfilt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from util.utils_tracking_v3 import Detect2Tracking

from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage import measure
from scipy import ndimage
from tqdm import tqdm

def make_tiff_volume_from_3DU16(folder, binary_path, show_img=False, save_tiff=False):
    
    if save_tiff:
        os.makedirs(folder + "/images/ch_1", exist_ok=True)
        os.makedirs(folder + "/images/ch_2", exist_ok=True)

    with open(binary_path, 'rb') as f:
    
        data1 = f.read(4)
        height = struct.unpack('>I', data1)[0]
        print("Number of elements:", height)

        title_voxel_size = ["Voxel Size Z:", "Voxel Size Y:", "Voxel Size X:"]
        for i in range(3):
            data = f.read(8)
            voxel_size = struct.unpack('>d', data)[0]
            print(title_voxel_size[i], voxel_size)
        print()
        
        title_values = ["Number of elements:", "Volumes:", "Channels:", "Voxels Z:", "Voxels Y:", "Voxels X:"]
        data = f.read(4)
        value = struct.unpack('>I', data)[0]
        print(title_values[0], value)
        data = f.read(4)
        num_volume = struct.unpack('>I', data)[0]
        num_volume = 10000
        print(title_values[1], num_volume)
        for i in range(2,6):
            data = f.read(4)
            value = struct.unpack('>I', data)[0]
            print(title_values[i], value)
        
        for t in range(num_volume):
            print(f'===== Volume {t+1} =====')
            data = f.read(4)
            channel = struct.unpack('>I', data)[0]
            data = f.read(4)
            depth = struct.unpack('>I', data)[0]
            data = f.read(4)
            height = struct.unpack('>I', data)[0]
            data = f.read(4)
            width = struct.unpack('>I', data)[0]
            # print("Channels:", channel)
            # print("Voxels Z:", depth)
            # print("Voxels Y:", height)
            # print("Voxels X:", width)

            for c in range(2):
                for d in range(depth):
                    data = f.read(2 * width * height)
                    img_array = np.frombuffer(data, dtype='>u2').reshape((height, width))  # '>u2' is big-endian uint16
                    if d == 0:
                        ls_image_stack = np.empty((0, height, width), dtype=np.uint16)  # Initialize as an empty 3D array
                        # Append the new image to the NumPy stack
                        ls_image_stack = np.append(ls_image_stack, img_array[np.newaxis, :, :], axis=0)
                    else:
                        # Append the new image to the NumPy stack
                        ls_image_stack = np.append(ls_image_stack, img_array[np.newaxis, :, :], axis=0)

                if save_tiff:
                    tiff_file_path = f'{folder}/images/ch_{c+1}/volume_t{t}.tiff'
                    with tifffile.TiffWriter(tiff_file_path) as tiff:
                        tiff.write(ls_image_stack)
                
                if show_img:
                    plt.imshow(np.max(ls_image_stack,axis=0))
                    plt.show()

def make_montage_img(img, gap=10):
    """
    Create a montage image by combining multiple images.

    Parameters:
    img (numpy.ndarray): The input image array with shape (nz, ny, nx).
    gap (int, optional): The gap size between images in the montage. Default is 10.

    Returns:
    numpy.ndarray: The montage image with shape (ny+nz+10, nx+nz+10).
    """
    if len(img.shape)==4:
        nz, ny, nx = img[1].shape
        montage = np.zeros((ny+nz+10, nx+nz+10), dtype=img.dtype)
        montage[:ny, :nx] = np.max(img[1], axis=0)
        montage[ny+gap:, :nx] = np.max(img[1], axis=1)
        montage[:ny, nx+gap:] = np.max(img[1], axis=2).transpose()
        return montage
    else:
        nz, ny, nx = img.shape
        montage = np.zeros((ny+nz+10, nx+nz+10), dtype=img.dtype)
        montage[:ny, :nx] = np.max(img, axis=0)
        montage[ny+gap:, :nx] = np.max(img, axis=1)
        montage[:ny, nx+gap:] = np.max(img, axis=2).transpose()
        return montage

def make_track_frame(path, img, nz, ny, nx, points, cmap, num_volume, i, trajectory):
    plt.figure()  
    plt.imshow(img, cmap="gray")
    
    if trajectory:
        # 軌跡（線）
        for frame_num in range(1, len(points)):
            previous_coords = points[frame_num - 1]
            current_coords = points[frame_num]
            
            color_val = frame_num / num_volume  # 正規化された時系列の値
            color = cmap(color_val)
            
            for obj_idx in range(previous_coords.shape[0]):
                # XY平面での追跡点の軌跡
                plt.plot([previous_coords[obj_idx, 2], current_coords[obj_idx, 2]],  # X 座標
                            [previous_coords[obj_idx, 1], current_coords[obj_idx, 1]],  # Y 座標
                            color=color)
            
                # YZ平面での追跡点の軌跡
                plt.plot([previous_coords[obj_idx, 0] + nx + 10, current_coords[obj_idx, 0] + nx + 10],  # Z 座標
                        [previous_coords[obj_idx, 1], current_coords[obj_idx, 1]],  # Y 座標
                        color=color)

                # XZ平面での追跡点の軌跡
                plt.plot([previous_coords[obj_idx, 2], current_coords[obj_idx, 2]],  # X 座標
                        [previous_coords[obj_idx, 0] + ny + 10, current_coords[obj_idx, 0] + ny + 10],  # Z 座標
                        color=color)


        # 現在時刻の追跡点をカラーマップに基づく色の点で描画
        current_coords = points[-1]
        for obj_idx in range(current_coords.shape[0]):
            color_val = (len(points) - 1) / num_volume  # 現在の時刻に基づく正規化された値
            color = cmap(color_val)
            
            # XY平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2], current_coords[obj_idx, 1], s=20, color=color)

            # YZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 0] + nx + 10, current_coords[obj_idx, 1], s=20, color=color)

            # XZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2], current_coords[obj_idx, 0] + ny + 10, s=20, color=color)

        # カラーバーの追加
        norm = plt.Normalize(vmin=0, vmax=num_volume*0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 空のデータをセット
        cbar = plt.colorbar(sm)
        cbar.set_label("Time [sec]")
        
    else:
        current_coords = points[-1]
        for obj_idx in range(current_coords.shape[0]):
            color = 'blue'
            # XY平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2], current_coords[obj_idx, 1], s=10, color=color)

            # YZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 0] + nx + 10, current_coords[obj_idx, 1], s=10, color=color)

            # XZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2], current_coords[obj_idx, 0] + ny + 10, s=10, color=color)

    
    # スケールバーの設定
    scale_bar_length = 49  # スケールバーの長さ（ピクセル単位）
    # scale_bar_text = "50 μm"  # スケールバーのテキスト
    scale_bar_color = 'white'  # スケールバーの色
    scale_bar_thickness = 3  # スケールバーの太さ

    # スケールバーの位置
    x_position = 10  # スケールバーを表示するX座標
    y_position = ny - 10  # スケールバーを表示するY座標

    # スケールバーの描画
    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
            color=scale_bar_color, linewidth=scale_bar_thickness)

    plt.axis('off')
    plt.savefig(path + f"/temp_frame_t{i}.png", dpi=600, bbox_inches='tight',pad_inches=0)
    plt.close()

    frame = cv2.imread(path + f"/temp_frame_t{i}.png")
    
    if not i == 0:
        os.remove(path + f"/temp_frame_t{i}.png")
    return frame

def make_montage_img_v2(img, gap=10):
    """
    Create a montage image by combining multiple images.

    Parameters:
    img (numpy.ndarray): The input image array with shape (nz, ny, nx).
    gap (int, optional): The gap size between images in the montage. Default is 10.

    Returns:
    numpy.ndarray: The montage image with shape (ny+nz+10, nx+nz+10).
    """
    border_value = 255
    border_thickness = 2
    
    nz, ny, nx = img[1].shape if len(img.shape)==4 else img.shape
    img = img[1] if len(img.shape)==4 else img
    
    img = cp.asarray(img)
    montage = cp.zeros((ny+nz+6, nx+nz+6), dtype=img.dtype)
    
    # xy平面
    montage[
        border_thickness:border_thickness + ny, 
        border_thickness:border_thickness + nx
    ] = cp.max(img, axis=0)
    
    # xz平面
    montage[
        border_thickness * 2 + ny:border_thickness * 2 + ny + nz, 
        border_thickness:nx + border_thickness
    ] = cp.max(img, axis=1)
    
    # yz平面
    montage[
        border_thickness:border_thickness + ny, 
        border_thickness * 2 + nx:border_thickness * 2 + nx + nz
    ] = cp.max(img, axis=2).transpose()
    
    # 境界の追加
    montage[:border_thickness, :] = border_value
    montage[border_thickness + ny:border_thickness * 2 + ny, :] = border_value
    montage[border_thickness * 2 + ny + nz:, :] = border_value
    montage[:, :border_thickness] = border_value
    montage[:, border_thickness + nx:border_thickness * 2 + nx] = border_value
    montage[:, border_thickness * 2 + nx + nz:] = border_value
    
    # 空白の穴埋め
    montage[
        border_thickness * 2 + ny:border_thickness * 2 + ny + nz, 
        border_thickness * 2 + nx:border_thickness * 2 + nx + nz
    ] = border_value
    
    return cp.asnumpy(montage)

def make_track_frame_v2(path, img, nz, ny, nx, points, cmap, num_volume, i, trajectory, track_num=0):
    plt.figure()  
    plt.imshow(img)
    
    border_thickness = 2
    
    if trajectory:
        # 軌跡（線）
        start_frame = max(1, len(points) - track_num + 1)  # 描画の開始フレーム
        
        for frame_num in range(start_frame, len(points)):
            # coordsの取得
            previous_coords = points[frame_num - 1]
            current_coords = points[frame_num]
            # 全フレームに基づいてフレーム番号を正規化
            color_val = (len(points) - 1) / num_volume
            color = cmap(color_val)

            for obj_idx in range(previous_coords.shape[0]):
                # XY平面での追跡点の軌跡
                plt.plot([previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X 座標
                            [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y 座標
                            color=color)
            
                # YZ平面での追跡点の軌跡
                plt.plot([previous_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 0] + nx + border_thickness * 2],  # Z 座標
                        [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y 座標
                        color=color)

                # XZ平面での追跡点の軌跡
                plt.plot([previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X 座標
                        [previous_coords[obj_idx, 0] + ny + border_thickness * 2, current_coords[obj_idx, 0] + ny + border_thickness * 2],  # Z 座標
                        color=color)


        # 現在時刻の追跡点をカラーマップに基づく色の点で描画
        current_coords = points[-1]
        for obj_idx in range(current_coords.shape[0]):
            color_val = (len(points) - 1) / num_volume  # 現在の時刻に基づく正規化された値
            color = cmap(color_val)
            
            # XY平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, s=20, color=color)

            # YZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness * 2, s=20, color=color)

            # XZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, s=20, color=color)

        # # カラーバーの追加
        # norm = plt.Normalize(vmin=0, vmax=num_volume*0.1)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])  # 空のデータをセット
        # cbar = plt.colorbar(sm)
        # cbar.set_label("Time [sec]")
        
    else:
        current_coords = points[-1]
        for obj_idx in range(current_coords.shape[0]):
            color = 'blue'
            # XY平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)

            # YZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)

            # XZ平面に追跡点を描画
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, s=10, color=color)

    
    # スケールバーの設定
    scale_bar_length = 20  # スケールバーの長さ（ピクセル単位）
    # scale_bar_text = "50 μm"  # スケールバーのテキスト
    scale_bar_color = 'white'  # スケールバーの色
    scale_bar_thickness = 2  # スケールバーの太さ

    # スケールバーの位置
    x_position = 125  # スケールバーを表示するX座標
    y_position = ny - 5  # スケールバーを表示するY座標

    # スケールバーの描画
    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
            color=scale_bar_color, linewidth=scale_bar_thickness)
    
    # 平面名の位置と文字
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    
    # Timer 現在フレーム時刻の表示
    current_time = f"T = {len(points) - 1} ms"  # 現在フレームの時刻情報（例: フレーム番号と総フレーム数）
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11, fontname="Segoe UI", va="top", ha="left")

    plt.axis('off')
    plt.savefig(path + f"/temp_frame_t{i}.png",  dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    frame = cv2.imread(path + f"/temp_frame_t{i}.png")
    
    if not i == 0:
        os.remove(path + f"/temp_frame_t{i}.png")
    return frame

def make_track_frame_v2_np(path, img, nz, ny, nx, points_np, cmap, num_volume, i, trajectory, track_num=0):
    plt.figure()
    plt.imshow(img)

    border_thickness = 2

    if trajectory:
        # 軌跡（線）
        start_frame = max(1, len(points_np) - track_num + 1)  # 描画の開始フレーム

        for frame_num in range(start_frame, len(points_np)):
            # 現在のフレームと前フレームの座標を取得
            previous_coords = points_np[frame_num - 1]  # Shape: (num_objects, 3)
            current_coords = points_np[frame_num]       # Shape: (num_objects, 3)

            # 全フレームに基づいてフレーム番号を正規化
            color_val = (frame_num - 1) / num_volume
            color = cmap(color_val)

            # 各オブジェクトの描画
            for obj_idx in range(previous_coords.shape[0]):
                # XY平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X座標
                    [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y座標
                    color=color,
                    linewidth=0.5
                )

                # YZ平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 0] + nx + border_thickness * 2],  # Z座標
                    [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y座標
                    color=color,
                    linewidth=0.5
                )

                # XZ平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X座標
                    [previous_coords[obj_idx, 0] + ny + border_thickness * 2, current_coords[obj_idx, 0] + ny + border_thickness * 2],  # Z座標
                    color=color,
                    linewidth=0.5
                )

        # 現在時刻の追跡点をカラーマップに基づく色の点で描画
        current_coords = points_np[-1]  # 最後のフレームの座標
        for obj_idx in range(current_coords.shape[0]):
            color_val = (len(points_np) - 1) / num_volume
            color = cmap(color_val)

            # XY平面に追跡点を描画
            plt.scatter(
                current_coords[obj_idx, 2] + border_thickness,
                current_coords[obj_idx, 1] + border_thickness,
                s=20, color=color
            )

            # YZ平面に追跡点を描画
            plt.scatter(
                current_coords[obj_idx, 0] + nx + border_thickness * 2,
                current_coords[obj_idx, 1] + border_thickness * 2,
                s=20, color=color
            )

            # XZ平面に追跡点を描画
            plt.scatter(
                current_coords[obj_idx, 2] + border_thickness,
                current_coords[obj_idx, 0] + ny + border_thickness * 2,
                s=20, color=color
            )

    else:
        # 軌跡なしの場合: 現在時刻の追跡点のみ描画
        current_coords = points_np[-1]
        for obj_idx in range(current_coords.shape[0]):
            color = 'blue'
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)
            plt.scatter(current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, s=10, color=color)

    # スケールバー
    scale_bar_length = 20
    scale_bar_color = 'white'
    scale_bar_thickness = 2
    x_position = 125
    y_position = ny - 5

    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
             color=scale_bar_color, linewidth=scale_bar_thickness)

    # 平面名
    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position], color=scale_bar_color, linewidth=scale_bar_thickness)
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    # 時間の表示
    current_time = f"T = {len(points) - 1} ms"
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11, fontname="Segoe UI", va="top", ha="left")

    plt.axis('off')
    plt.savefig(path + f"/temp_frame_t{i}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    frame = cv2.imread(path + f"/temp_frame_t{i}.png")
    if i != 0:
        os.remove(path + f"/temp_frame_t{i}.png")
    return frame

def make_track_frame_v2_fast(path, img, nz, ny, nx, points, cmap, num_volume, i, trajectory, track_num=0):
    plt.figure()
    plt.imshow(img)
    
    border_thickness = 2
    lines = []
    scatters = []

    if trajectory:
        start_frame = max(1, len(points) - track_num + 1)
        for frame_num in range(start_frame, len(points)):
            previous_coords = points[frame_num - 1]
            current_coords = points[frame_num]
            color_val = (len(points) - 1) / num_volume
            color = cmap(color_val)

            for obj_idx in range(previous_coords.shape[0]):
                lines.append(([previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],
                              [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness], color))
                lines.append(([previous_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 0] + nx + border_thickness * 2],
                              [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness], color))
                lines.append(([previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],
                              [previous_coords[obj_idx, 0] + ny + border_thickness * 2, current_coords[obj_idx, 0] + ny + border_thickness * 2], color))

        current_coords = points[-1]
        for obj_idx in range(current_coords.shape[0]):
            color_val = (len(points) - 1) / num_volume
            color = cmap(color_val)
            scatters.append((current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, 20, color))
            scatters.append((current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness * 2, 20, color))
            scatters.append((current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, 20, color))
    else:
        current_coords = points[-1]
        for obj_idx in range(current_coords.shape[0]):
            color = 'blue'
            scatters.append((current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, 10, color))
            scatters.append((current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness, 10, color))
            scatters.append((current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, 10, color))

    for line in lines:
        plt.plot(line[0], line[1], color=line[2])
    for scatter in scatters:
        plt.scatter(scatter[0], scatter[1], s=scatter[2], color=scatter[3])

    scale_bar_length = 20
    scale_bar_color = 'white'
    scale_bar_thickness = 2
    x_position = 125
    y_position = ny - 5
    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position], color=scale_bar_color, linewidth=scale_bar_thickness)
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    current_time = f"T = {len(points) - 1} ms"
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11, fontname="Segoe UI", va="top", ha="left")

    plt.axis('off')
    plt.savefig(path + f"/temp_frame_t{i}.png", dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    frame = cv2.imread(path + f"/temp_frame_t{i}.png")
    if not i == 0:
        os.remove(path + f"/temp_frame_t{i}.png")
    return frame

def make_track_frame_v2_np_save(path, img, nz, ny, nx, points_np, cmap, num_volume, trajectory=True):
    plt.figure()
    plt.imshow(img)

    border_thickness = 2

    if trajectory:
        # 軌跡（線）
        for frame_num in range(1, len(points_np)):
            # 現在のフレームと前フレームの座標を取得
            previous_coords = points_np[frame_num - 1]  # Shape: (num_objects, 3)
            current_coords = points_np[frame_num]       # Shape: (num_objects, 3)

            # 全フレームに基づいてフレーム番号を正規化
            color_val = (frame_num - 1) / num_volume
            color = cmap(color_val)

            # 各オブジェクトの描画
            for obj_idx in range(previous_coords.shape[0]):
                # XY平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X座標
                    [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y座標
                    color=color,
                    linewidth=0.5
                )

                # YZ平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 0] + nx + border_thickness * 2],  # Z座標
                    [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y座標
                    color=color,
                    linewidth=0.5
                )

                # XZ平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X座標
                    [previous_coords[obj_idx, 0] + ny + border_thickness * 2, current_coords[obj_idx, 0] + ny + border_thickness * 2],  # Z座標
                    color=color,
                    linewidth=0.5
                )

        # 現在時刻の追跡点をカラーマップに基づく色の点で描画
        #current_coords = points_np[-1]  # 最後のフレームの座標
        #for obj_idx in range(current_coords.shape[0]):
         #   color_val = (len(points_np) - 1) / num_volume
          #  color = cmap(color_val)

            # XY平面に追跡点を描画
           # plt.scatter(
            #    current_coords[obj_idx, 2] + border_thickness,
             #   current_coords[obj_idx, 1] + border_thickness,
              #  s=15, color=color
            #)

            # YZ平面に追跡点を描画
            #plt.scatter(
             #   current_coords[obj_idx, 0] + nx + border_thickness * 2,
              #  current_coords[obj_idx, 1] + border_thickness * 2,
               # s=15, color=color
            #)

            # XZ平面に追跡点を描画
            #plt.scatter(
             #   current_coords[obj_idx, 2] + border_thickness,
              #  current_coords[obj_idx, 0] + ny + border_thickness * 2,
               # s=15, color=color
           # )

    else:
        # 軌跡なしの場合: 現在時刻の追跡点のみ描画
        current_coords = points_np[-1]
        for obj_idx in range(current_coords.shape[0]):
            color = 'blue'
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)
            plt.scatter(current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, s=10, color=color)

    # スケールバー
    scale_bar_length = 20
    scale_bar_color = 'white'
    scale_bar_thickness = 2
    x_position = 125
    y_position = ny - 5

    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
             color=scale_bar_color, linewidth=scale_bar_thickness)

    # 平面名
    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position], color=scale_bar_color, linewidth=scale_bar_thickness)
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    # 時間の表示
    current_time = f"T = {len(points) - 1} ms"
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11, fontname="Segoe UI", va="top", ha="left")

    #norm = plt.Normalize(vmin=0, vmax=len(points_np) - 1)
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])  # 空のデータをセット
    #cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)  # サイズを小さくする
    #cbar.set_label("Time [ms]")
    #cbar.set_ticks([0,2000,4000,6000,8000,9999])  # 0と9999を表示
    #cbar.set_ticklabels(['0','2000','4000','6000','8000','9999'])  # ラベルを設定
    
    norm = plt.Normalize(vmin=0, vmax=7001)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label("Time [ms]")
    cbar.set_ticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7001])
    cbar.set_ticklabels(['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000'])
    plt.axis('off')
    plt.savefig(path + f"/trajectory.png", dpi=300, bbox_inches='tight', pad_inches=0)
    

    plt.show()
    plt.close()

def make_track_speed_frame_v2_np_save(path, img, nz, ny, nx, points_np, speed_np, cmap, num_volume, trajectory=True):
    plt.figure()
    plt.imshow(img)

    border_thickness = 2

    if trajectory:
        # 軌跡（線）
        for frame_num in range(1, len(points_np)):
            # 現在のフレームと前フレームの座標を取得
            previous_coords = points_np[frame_num - 1]  # Shape: (num_objects, 3)
            current_coords = points_np[frame_num]       # Shape: (num_objects, 3)

            # 各オブジェクトの描画
            for obj_idx in range(previous_coords.shape[0]):
                # 現在の速度を取得
                speed_val = speed_np[frame_num - 1, obj_idx]  # 各オブジェクトの速度
                # print(f"Object {obj_idx}, Speed: {speed_val}")
                # print(np.max(speed_np))
                
                # 速度を0～1に正規化
                norm_speed_val = speed_val / np.max(speed_np)
                # print(f"Normalized Speed: {norm_speed_val}")
                
                # 速度に基づいてカラーマップを適用
                color = cmap(norm_speed_val)  # カラーマップを使用
                
                # XY平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X座標
                    [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y座標
                    color=color,
                    linewidth=0.5
                )

                # YZ平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 0] + nx + border_thickness * 2],  # Z座標
                    [previous_coords[obj_idx, 1] + border_thickness, current_coords[obj_idx, 1] + border_thickness],  # Y座標
                    color=color,
                    linewidth=0.5
                )

                # XZ平面での追跡点の軌跡
                plt.plot(
                    [previous_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 2] + border_thickness],  # X座標
                    [previous_coords[obj_idx, 0] + ny + border_thickness * 2, current_coords[obj_idx, 0] + ny + border_thickness * 2],  # Z座標
                    color=color,
                    linewidth=0.5
                )

        # # 現在時刻の追跡点をカラーマップに基づく色の点で描画
        # current_coords = points_np[-1]  # 最後のフレームの座標
        # for obj_idx in range(current_coords.shape[0]):
        #     speed_val = speed_np[-1]  # 最後の速度
        #     norm_speed_val = (speed_val - speed_np.min()) / (speed_np.max() - speed_np.min())
        #     color = cmap(norm_speed_val)
            
        #     # 現在の速度を取得
        #     speed_val = speed_np[frame_num - 1, obj_idx]  # 各オブジェクトの速度
        #     # print(f"Object {obj_idx}, Speed: {speed_val}")
        #     # print(np.max(speed_np))
            
        #     # 速度を0～1に正規化
        #     norm_speed_val = speed_val / np.max(speed_np)
        #     # print(f"Normalized Speed: {norm_speed_val}")
            
        #     # 速度に基づいてカラーマップを適用
        #     color = cmap(norm_speed_val)  # カラーマップを使用

        #     # XY平面に追跡点を描画
        #     plt.scatter(
        #         current_coords[obj_idx, 2] + border_thickness,
        #         current_coords[obj_idx, 1] + border_thickness,
        #         s=15, color=color
        #     )

        #     # YZ平面に追跡点を描画
        #     plt.scatter(
        #         current_coords[obj_idx, 0] + nx + border_thickness * 2,
        #         current_coords[obj_idx, 1] + border_thickness * 2,
        #         s=15, color=color
        #     )

        #     # XZ平面に追跡点を描画
        #     plt.scatter(
        #         current_coords[obj_idx, 2] + border_thickness,
        #         current_coords[obj_idx, 0] + ny + border_thickness * 2,
        #         s=15, color=color
        #     )

    else:
        # 軌跡なしの場合: 現在時刻の追跡点のみ描画
        current_coords = points_np[-1]
        for obj_idx in range(current_coords.shape[0]):
            color = 'blue'
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)
            plt.scatter(current_coords[obj_idx, 0] + nx + border_thickness * 2, current_coords[obj_idx, 1] + border_thickness, s=10, color=color)
            plt.scatter(current_coords[obj_idx, 2] + border_thickness, current_coords[obj_idx, 0] + ny + border_thickness * 2, s=10, color=color)

    # スケールバー
    scale_bar_length = 20
    scale_bar_color = 'white'
    scale_bar_thickness = 2
    x_position = 125
    y_position = ny - 5

    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
             color=scale_bar_color, linewidth=scale_bar_thickness)

    # 平面名
    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position], color=scale_bar_color, linewidth=scale_bar_thickness)
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10, fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    # 時間の表示
    current_time = f"T = {len(points) - 1} ms"
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11, fontname="Segoe UI", va="top", ha="left")

    norm = plt.Normalize(vmin=speed_np.min(), vmax=speed_np.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 空のデータをセット
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)  # サイズを小さくする
    cbar.set_label("Speed [µm/s]")
    # cbar.set_ticks([0,2000,4000,6000,8000,9999])  # 0と9999を表示
    # cbar.set_ticklabels(['0','2000','4000','6000','8000','9999'])  # ラベルを設定
    plt.axis('off')
    plt.savefig(path + f"/speed.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.show()
    plt.close()

import matplotlib.pyplot as plt
import numpy as np

def make_track_speed_frame_v2_np_save_scatter(path, img, nz, ny, nx, points_np, speed_np, cmap, num_volume, trajectory=True):
    plt.figure()
    plt.imshow(img)

    border_thickness = 2

    if trajectory:
        for frame_num in range(1, len(points_np)):
            previous_coords = points_np[frame_num - 1]
            current_coords = points_np[frame_num]

            for obj_idx in range(previous_coords.shape[0]):
                # === 各座標成分 ===
                z0 = previous_coords[obj_idx, 0]
                y0 = previous_coords[obj_idx, 1]
                x0 = previous_coords[obj_idx, 2]

                z1 = current_coords[obj_idx, 0]
                y1 = current_coords[obj_idx, 1]
                x1 = current_coords[obj_idx, 2]

                # === スピード正規化とカラーマップ適用 ===
                speed_val = speed_np[frame_num - 1, obj_idx]
                norm_speed_val = speed_val / np.max(speed_np)
                color = cmap(norm_speed_val)

                # === XY (左上) ===
                plt.scatter([x0 + border_thickness, x1 + border_thickness],
                            [y0 + border_thickness, y1 + border_thickness],
                            color=color, s=1)

                # === XZ (左下) ===
                plt.scatter([x0 + border_thickness, x1 + border_thickness],
                            [z0 + ny + border_thickness * 2, z1 + ny + border_thickness * 2],
                            color=color, s=1)

                # === YZ (右上) – 左右反転対応 (Z軸flip) ===
                z0_flipped = nz - z0
                z1_flipped = nz - z1
                plt.scatter([z0_flipped + nx + border_thickness * 2, z1_flipped + nx + border_thickness * 2],
                            [y0 + border_thickness, y1 + border_thickness],
                            color=color, s=1)
    else:
        # 最終時点のプロットのみ（軌跡なしモード）
        current_coords = points_np[-1]
        for obj_idx in range(current_coords.shape[0]):
            z = current_coords[obj_idx, 0]
            y = current_coords[obj_idx, 1]
            x = current_coords[obj_idx, 2]
            z_flipped = nz - z

            color = 'blue'
            plt.scatter(x + border_thickness, y + border_thickness, s=10, color=color)
            plt.scatter(x + border_thickness, z + ny + border_thickness * 2, s=10, color=color)
            plt.scatter(z_flipped + nx + border_thickness * 2, y + border_thickness, s=10, color=color)

    # === スケールバー ===
    scale_bar_length = 20
    scale_bar_color = 'white'
    scale_bar_thickness = 2
    x_position = 125
    y_position = ny - 5

    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
             color=scale_bar_color, linewidth=scale_bar_thickness)

    # === 面ラベル ===
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="top", ha="left")

    # === 時間表示 ===
    current_time = f"T = {len(points_np) - 1} ms"
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11,
             fontname="Segoe UI", va="top", ha="left")

    # === カラーバー ===
    norm = plt.Normalize(vmin=speed_np.min(), vmax=speed_np.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label("Speed [µm/s]")

    plt.axis('off')
    plt.savefig(path + f"/speed_scatter.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

def make_track_speed_frame_v2_np_degree_save(path, img, nz, ny, nx, points_np, speed_np, cmap, num_volume, trajectory=True):
    plt.figure()
    plt.imshow(img)

    border_thickness = 2

    if trajectory:
        for frame_num in range(1, len(points_np)):
            previous_coords = points_np[frame_num - 1]
            current_coords = points_np[frame_num]

            for obj_idx in range(previous_coords.shape[0]):
                speed_val = speed_np[frame_num - 1, obj_idx]
                #norm_speed_val = speed_val / 360  # normalize degree
                #color = cmap(norm_speed_val)

                # speed_val は角度なので 0〜360 を 0〜1 に正規化
                norm_speed_val = np.clip(speed_val / 360.0, 0.0, 1.0)
                color = colormap(norm_speed_val)


                # === 各座標
                z0 = previous_coords[obj_idx, 0]
                y0 = previous_coords[obj_idx, 1]
                x0 = previous_coords[obj_idx, 2]

                z1 = current_coords[obj_idx, 0]
                y1 = current_coords[obj_idx, 1]
                x1 = current_coords[obj_idx, 2]

                # === XY平面（左上）
                plt.plot([x0 + border_thickness, x1 + border_thickness],
                         [y0 + border_thickness, y1 + border_thickness],
                         color=color, linewidth=0.5)

                # === XZ平面（左下）
                plt.plot([x0 + border_thickness, x1 + border_thickness],
                         [z0 + ny + border_thickness * 2, z1 + ny + border_thickness * 2],
                         color=color, linewidth=0.5)

               
                # === YZ平面（右上、Z軸そのまま）
                plt.plot([z0 + nx + border_thickness * 2, z1 + nx + border_thickness * 2],  # Z軸
                         [y0 + border_thickness, y1 + border_thickness],  # Y軸
                          color=color, linewidth=0.5)

                # === YZ平面（右上・Z軸左右反転）
                #z0_flipped = nz - 1 - z0
                #z1_flipped = nz - 1 - z1
                #plt.plot([z0_flipped + nx + border_thickness * 2, z1_flipped + nx + border_thickness * 2],
                #         [y0 + border_thickness, y1 + border_thickness],
                #         color=color, linewidth=0.5)

    else:
        current_coords = points_np[-1]
        for obj_idx in range(current_coords.shape[0]):
            z = current_coords[obj_idx, 0]
            y = current_coords[obj_idx, 1]
            x = current_coords[obj_idx, 2]
            z_flip = nz - 1 - z

            color = 'blue'
            plt.scatter(x + border_thickness, y + border_thickness, s=10, color=color)
            plt.scatter(x + border_thickness, z + ny + border_thickness * 2, s=10, color=color)
            plt.scatter(z_flip + nx + border_thickness * 2, y + border_thickness, s=10, color=color)

    # スケールバー
    scale_bar_length = 20
    scale_bar_color = 'white'
    scale_bar_thickness = 2
    x_position = 125
    y_position = ny - 5

    plt.plot([x_position, x_position + scale_bar_length], [y_position, y_position],
             color=scale_bar_color, linewidth=scale_bar_thickness)

    # 面ラベル
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="top", ha="left")

    # 時間表示
    current_time = f"T = {len(points_np) - 1} ms"
    plt.text(border_thickness+3, border_thickness+3, current_time, color="white", fontsize=11,
             fontname="Segoe UI", va="top", ha="left")

    # カラーバー（角度）
    norm = plt.Normalize(vmin=0, vmax=360)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label("Degree")
    cbar.set_ticks([0, 60, 120, 180, 240, 300, 360])
    cbar.set_ticklabels(['0', '60', '120', '180', '240', '300', '360'])

    # === 終着点の中抜き丸をプロット ===
    # 軌跡の最後の座標（z, y, x）を取得
    #last_point = points[-1, 0]
    #z, y, x = last_point
    #angle = speeds[-1, 0, 0] % 360
    #color = colormap(angle / 360.0)

    #marker_size = 10  # 調整済みマーカーサイズ

    # xy面
    #plt.scatter(x, y, s=marker_size, facecolors='none', edgecolors=[color], linewidths=1.5)

    # yz面（x軸分だけ右にずらす）
    #plt.scatter(z + nx +5, y, s=marker_size, facecolors='none', edgecolors=[color], linewidths=1.5)

    # xz面（y軸分だけ下にずらす）
    #plt.scatter(x, z + ny+5, s=marker_size, facecolors='none', edgecolors=[color], linewidths=1.5)


    plt.axis('off')
    plt.savefig(path + f"/degree.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()



def water(image):
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage import measure

    # ガウスフィルタで平滑化
    smoothed_image = ndimage.gaussian_filter(image, sigma=2)

    # 負の勾配を計算（Watershedでは"山"を"谷"に変換）
    gradient = -smoothed_image

    # 局所最大値のマーカを計算
    distance = ndimage.distance_transform_edt(smoothed_image > smoothed_image.mean())
    local_max = peak_local_max(
        image=distance,         # 距離画像
        min_distance=4,        # 最小距離
        threshold_abs=0.1,      # 絶対値しきい値
        exclude_border=False,   # 境界を除外しない
        labels=smoothed_image > smoothed_image.mean()  # ラベルを使用
    )
    
    # ラベリング
    markers, _ = ndimage.label(local_max)

    # Watershedの適用
    labels = watershed(gradient, markers, mask=smoothed_image > smoothed_image.mean())

    # セグメントのプロパティ計算
    regions = measure.regionprops(labels)
    
    return labels

def adjust_contrast(image, v_min, v_max):
    """
    コントラストを調整する関数。
    v_min: コントラストの下限値
    v_max: コントラストの上限値
    """
    adjusted = np.clip((image - v_min) / (v_max - v_min), 0, 1)
    return adjusted


def extract_rows_from_csv(input_csv_path, output_csv_path, n):
    """
    Extract every nth row from the input CSV file and save to a new CSV file.

    Parameters:
    input_csv_path (str): Path to the input CSV file.
    output_csv_path (str): Path to the output CSV file.
    n (int): Interval for row extraction (e.g., every nth row).
    """
    with open(input_csv_path, 'r') as infile, open(output_csv_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for i, row in enumerate(reader):
            if i % n == 0:
                writer.writerow(row)


#%%
# フォルダとバイナリファイルのパス指定
# -----------------------------------------------------------------------
folder = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood"
binary_path = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood\20241016-161347chlamyCYTO16verygood.3DU16"

# folder = r"C:\Users\mlab-\Desktop\20241016-160147chlamyCYTO16verygood"
# binary_path = r"C:\Users\mlab-\Desktop\20241016-160147chlamyCYTO16verygood\20241016-160147chlamyCYTO16verygood.3DU16"


#%%
# channelごとにtiff画像を作成
# -----------------------------------------------------------------------
make_tiff_volume_from_3DU16(folder, binary_path, show_img=False, save_tiff=True)




#%% # watershed画像の確認
# -----------------------------------------------------------------------

# 3次元画像の読み込み
for i in range(7):
    print(f" ================== volume {6000+i*500} ================== ")
    file_path = folder + f"/images/ch_2/volume_t{(6000+i*500)}.tiff"
    data = tifffile.imread(file_path)

    # print(data.shape)

    thre = 155
    # mon = make_montage_img(data>200)
    # plt.imshow(mon)
    # plt.show()

    # Step 1: スムージング（オプション、ノイズ除去）
    # smoothed = gaussian_filter(data, sigma=0.5)

    # Step 2: 距離変換
    distance = distance_transform_edt(data > thre)

    # Step 3: ラベル付け（局所最大をマーカーとして使用）
    local_maxi = measure.label(distance > 1) #2
    # print(local_maxi.shape)

    # Step 4: Watershed分割
    labels = watershed(-distance, markers=local_maxi, mask=data > thre)

    # Step 5: 結果の可視化
    # plt.figure(figsize=(12, 12))
    # plt.subplot(2, 2, 1)
    # plt.title("Original")
    # plt.imshow(make_montage_img(data))
    # plt.subplot(2, 2, 2)
    # plt.title("Smoothed")
    # plt.imshow(make_montage_img(data))
    # plt.subplot(2, 2, 3)
    # plt.title("Distance")
    # plt.imshow(make_montage_img(distance))
    # plt.subplot(2, 2, 4)
    # plt.title("Labels")
    # plt.imshow(make_montage_img(labels))
    
    plt.imshow(make_montage_img(data>thre))
    plt.show()
    
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.title("Distance")
    plt.imshow(make_montage_img(distance))
    plt.subplot(1, 2, 2)
    plt.title("Labels")
    plt.imshow(make_montage_img(labels))

    plt.tight_layout()
    plt.show()






# %% Tracking and Calculation of Centroid
use_centroid = True  # 重心を使用するかどうか
use_watershed = True  # Watershedを使用するかどうか
use_kalman = False  # カルマンフィルタを使用するかどうか
image_display = False  # 画像を表示するかどうか
do_profile = True  # プロファイリングを行うかどうか

# Detection and Tracking
d2t = Detect2Tracking()

# Make log folder
today = datetime.date.today()
now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
log_path = folder+"/log/"+str(today)+"/"+now
os.makedirs(log_path, exist_ok=True)
os.makedirs(folder + "/videos", exist_ok=True)

if do_profile == True:
    csv_file_path = f"{log_path}/tracking.csv"
    tracking_csv_file = open(csv_file_path, "a")

initial = True
for i in tqdm(range(10000), desc="Processing", ncols=100):
    # print(f"------- volume {i} -------")
    file_path = folder + f"/images/ch_2/volume_t{(i)}.tiff"
    img = tifffile.imread(file_path)
    
    time1 = time.time()
    img = gaussian_filter(img, sigma=0.5)
    # img = gaussian_laplace(img, sigma=0.7)
    
    time2 = time.time()
    ## ==> START 'Detect ROI'
    if initial == True:
        next2, peak2 = d2t.custom_detect_roi_sort(
            image=img, 
            t=0, 
            min_dist=4, 
            detect_num= 5,#np.inf, 
            footprint=np.ones((15,15,15), dtype='bool'), 
            use_gpu=True,
            do_profile=False,
        )   

        d2t.plot_coordinate_on_mip_with_num(np.max(img, axis=0), np.array(next2),True ,f"{folder}/videos")
        # キーボードからの入力を待つ
        user_input = input("追跡する細胞を選択してください(ex. 2,5,7)、全てを追跡する場合は'Enter'を押してください: ")
        if not user_input == "":
        # 選択したインデックスのリストを作成
            selected_indices = [int(inp) for inp in user_input.split(',')]
            # next2 と peak2 から選択されていないインデックスを削除
            next2 = np.delete(next2, np.where(~np.isin(np.arange(next2.shape[0]), selected_indices)), axis=0)
            peak2 = np.delete(peak2, np.where(~np.isin(np.arange(peak2.shape[1]), selected_indices)), axis=1)
    ## ==> END
    
    ## ==> START 'Tracking'
    else:
        if use_kalman:
            d2t.initialize_kalman_filters(next2)
            next2, peak2 = d2t.track_and_update(
                img=img,
                z_around=2,
                y_around=5,
                x_around=5,
                peak_record_with_intensity=peak2,
                threshold=0.5,
                sigma = 1,
                do_profile=False
            )
        else:
            next2, peak2 = d2t.tracking_using_intensity(
                img=img,
                z_around=3, 
                y_around=5, 
                x_around=5,
                pre_coordinate_with_intensity=next2, 
                peak_record_with_intensity=peak2, 
                threshold=0.5,
                sigma = 1,
                do_profile=False
            )
    ## ==> END
    
    time3 = time.time()
    
    if use_centroid:
        iteration = 0
        ditance_num = 1 #2
        threshold = 155 #175
        while iteration < 2:
            ## ==> START centroid
            if use_watershed:
                # Step 1: スムージング
                # smoothed = gaussian_filter(img, sigma=0.7)

                # Step 2: 距離変換
                distance = distance_transform_edt(img > threshold)

                # Step 3: 局所最大をラベルとして使用
                markers = measure.label(distance > ditance_num)

                # Step 4: Watershed
                labeled_image = watershed(-distance, markers=markers, mask=img > threshold)

                # 重心計算
                properties = regionprops(labeled_image, intensity_image=img)
                # print(properties)
                # print(len(properties))
                
            else:
                # 1. 二値化処理：輝度のしきい値を設定し、しきい値以上を1、それ以外を0にする
                binary_image = img > threshold

                # 2. ラベリング処理：二値化した画像内の塊（連結成分）を識別
                labeled_image, num_features = ndimage.label(binary_image)

                # 3. 重心を計算：各塊の重心を計算
                properties = measure.regionprops(labeled_image, intensity_image=img)
                # 結果の出力
                for _, prop in enumerate(properties):
                    # 重心の座標
                    centroid = prop.weighted_centroid
                    # 輝度範囲
                    min_intensity, max_intensity = prop.intensity_min, prop.intensity_max
            ## ==> END

            # ==> START 重心取得と重複確認
            updated_centroids = [] # 重心結果を格納するリスト

            # 近傍探索で得られた座標に対して、ラベル付き画像内の塊をチェック
            for coord in next2:
                z, y, x, intensity_value = coord

                # 二値化画像で該当座標が含まれるラベルを確認
                label = labeled_image[z, y, x]

                # print(label)
                if label > 0: # ラベルが存在する場合、その塊の重心を取得
                    prop = properties[label - 1]
                    centroid = prop.weighted_centroid
                    centroid_int = tuple(map(int, centroid))
                    # 重心の座標に対応する輝度値を取得
                    centroid_intensity = img[centroid_int[0], centroid_int[1], centroid_int[2]]
                    
                    updated_centroids.append((centroid_int[0], centroid_int[1], centroid_int[2], centroid_intensity))
                        
                else: # ラベルが存在しない場合
                    # print("not found label")
                    updated_centroids.append((z, y, x, intensity_value))
            
            if len(updated_centroids) != len(set(updated_centroids)):
                # print("re-calculate")
                iteration += 1 
                ditance_num += 1
            else:
                # next2の置換
                next2 = np.array(updated_centroids)

                # peaksの置換
                for peak_num in range(len(updated_centroids)):
                    centroid = updated_centroids[peak_num]
                    # ラベルから輝度情報を取得
                    z_centoroid, y_centoroid, x_centoroid, centroid_intensity = centroid
                    # 重心を近傍探索結果の参照ピクセルとして置き換える
                    peak2[-1][peak_num] = [0., float(centroid[0]), float(centroid[1]), float(centroid[2]), float(centroid_intensity)]
                break
        # ==> END
        
    time4 = time.time()
        
    # print(f"Gaussian: {time2 - time1}")
    # print(f"Detection or Tracking: {time3 - time2}")
    # print(f"Centroid: {time4 - time3}")
    # print(f"all: {time4 - time1}")

    # print(next2)
    if do_profile:
        flattened_array = np.concatenate(next2)
        output = ', '.join(map(str, flattened_array))
        tracking_csv_file.write(f"{output}\n")

    if image_display:
        d2t.plot_coordinate_on_mip(np.max(img, axis=0), np.array(next2))


    n_points = len(next2)
    colors = plt.cm.jet(np.linspace(0, 1, n_points))  # jetカラーマップを使用

    plt.figure()
    nz, ny, nx = img.shape
    fig = make_montage_img(img)
    plt.imshow(fig, cmap="gray")
    coordinate = np.array(next2)
    for l, color in enumerate(colors):
        plt.scatter(coordinate[l, 2], coordinate[l, 1], s=10, color=color) #xy
        plt.scatter(coordinate[l, 0]+nx+10, coordinate[l, 1], s=10, color=color) #yz
        plt.scatter(coordinate[l, 2], coordinate[l, 0]+ny+10, s=10, color=color) #xz
    plt.axis('off')
    plt.savefig("temp_frame.png", bbox_inches='tight')
    plt.close()
    if initial == True:
        initial = False
        video_filename = f"{log_path}/{today}_{now}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')#(*'mp4v')
        fps = 100
        frame_size = (nx+10+nz, ny+10+nz)
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
    frame = cv2.imread("temp_frame.png")
    frame = cv2.resize(frame, frame_size)
    video_writer.write(frame)

if do_profile == True:
    tracking_csv_file.close()


# os.remove("temp_frame.png")
video_writer.release()




#%% ----------------------------------------------------------
# >>>> START 'カラー調整（マジェンタ&シアン ver.）'
# ------------------------------------------------------------
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

folder = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood"


def adjust_contrast(image, v_min, v_max):
    """
    コントラストを調整する関数。
    v_min: コントラストの下限値
    v_max: コントラストの上限値
    """
    adjusted = np.clip((image - v_min) / (v_max - v_min), 0, 1)
    return adjusted

# チャネル1（赤）とチャネル2（緑）の読み込み
ch1 = tifffile.imread(folder + f"/images/ch_1/volume_t{7000}.tiff")
ch2 = tifffile.imread(folder + f"/images/ch_2/volume_t{7000}.tiff")

# チャネルの正規化（uint16の場合を想定）
normalize_value = 450
ch1_norm = ch1 / normalize_value * 255
ch2_norm = ch2 / normalize_value * 255

# 最大強度投影（z方向）
ch1_mip = np.max(ch1_norm, axis=0)
ch2_mip = np.max(ch2_norm, axis=0)

# 変数をグローバルに宣言して保存する
v_min1_saved = 0
v_max1_saved = 255
v_min2_saved = 0
v_max2_saved = 255

# コントラスト調整(GUI
def display_rgb(v_min1=0, v_max1=255, v_min2=0, v_max2=255):
    global v_min1_saved, v_max1_saved, v_min2_saved, v_max2_saved
    
    ch1_adjusted = adjust_contrast(ch1_mip, v_min1, v_max1)
    ch2_adjusted = adjust_contrast(ch2_mip, v_min2, v_max2)

    # RGB画像の作成（ch1はマジェンタ: 赤と青、ch2は緑）
    rgb_image = np.zeros((*ch1_mip.shape, 3), dtype=np.float32)
    rgb_image[..., 0] = ch1_adjusted  # 赤チャネル（ch1から）
    rgb_image[..., 2] = ch1_adjusted  # 青チャネル（ch1から）
    rgb_image[..., 1] = ch2_adjusted  # 緑チャネル（ch2から）

    # プロット
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title(f"Contrast Adjusted (Magenta: {v_min1}-{v_max1}, Green: {v_min2}-{v_max2})")
    plt.show()
    
    # 調整されたmin/maxを保存
    v_min1_saved, v_max1_saved = v_min1, v_max1
    v_min2_saved, v_max2_saved = v_min2, v_max2

# スライダーを設定してインタラクティブ表示
interact(
    display_rgb,
    v_min1=FloatSlider(value=0, min=0, max=255, step=1, description='Magenta Min'),
    v_max1=FloatSlider(value=255, min=0, max=255, step=1, description='Magenta Max'),
    v_min2=FloatSlider(value=0, min=0, max=255, step=1, description='Green Min'),
    v_max2=FloatSlider(value=255, min=0, max=255, step=1, description='Green Max')
)





#%% ----------------------------------------------------------
# >>>> START '追跡動画の生成'
# ------------------------------------------------------------
use_trajectory = True

import csv
import pandas as pd

output_csv = r"C:\Users\mlab-\Desktop\Chlamydomonas\log\2024-11-21\20241121161907 - complete\tracking.csv"
df = pd.read_csv(output_csv, header=None)
colormap = plt.colormaps['winter']

# pointsに追加する際もnumpy配列として操作
points = np.empty((0, 4, 3), dtype=float)  # shape: (0, 4, 3)

# Make log folder
today = datetime.date.today()
now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
log_path = folder+"/log/"+str(today)+"/"+now
os.makedirs(log_path, exist_ok=True)

# 行数をカウント
with open(output_csv, 'r') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader)

print(f"行数: {row_count}")

# for i, row in tqdm(df.iterrows(), total=row_count, desc="Processing", ncols=100):
for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"Processing volume {i}")
    time_s = time.time()
    # print(f"------- volume {i} -------")
    
    ### MIP
    # チャネル1（赤）とチャネル2（緑）の読み込み
    ch1 = tifffile.imread(folder + f"/images/ch_1/volume_t{i}.tiff")
    ch2 = tifffile.imread(folder + f"/images/ch_2/volume_t{i}.tiff")

    # チャネルの正規化（uint16の場合を想定）
    normalize_value = 450
    ch1_norm = (ch1 / normalize_value * 255).astype(np.float32)
    ch2_norm = (ch2 / normalize_value * 255).astype(np.float32)

    # 最大強度投影（z方向）
    ch1_mip = make_montage_img_v2(ch1_norm)
    ch2_mip = make_montage_img_v2(ch2_norm)
    
    ch1_adjusted = adjust_contrast(ch1_mip, v_min1_saved, v_max1_saved)
    ch2_adjusted = adjust_contrast(ch2_mip, v_min2_saved, v_max2_saved)

    # RGB画像の作成（ch1はマジェンタ: 赤と青、ch2は緑）
    mip_image = np.zeros((*ch1_mip.shape, 3), dtype=np.float32)
    mip_image[..., 0] = ch1_adjusted  # 赤チャネル（ch1から）
    mip_image[..., 2] = ch1_adjusted  # 青チャネル（ch1から）
    mip_image[..., 1] = ch2_adjusted  # 緑チャネル（ch2から）
    
    ### image shape
    nz, ny, nx = ch1.shape
    
    ### coordinate
    # next2を一度にnumpy配列として作成
    point = np.array([
        [row[0], row[1], row[2]],
        [row[4], row[5], row[6]],
        [row[8], row[9], row[10]],
        [row[12], row[13], row[14]],
    ], dtype=float)


    # pointの形状を(1, 4, 3)に変換
    point = np.expand_dims(point, axis=0)
    # pointsに追加
    points = np.append(points, point, axis=0)
    
    ### plot
    frame = make_track_frame_v2_np(
        log_path, 
        mip_image,
        nz,
        ny,
        nx,
        points, 
        colormap, 
        row_count, 
        i, 
        use_trajectory,
        track_num=200
    )

    if i == 0:
        # print("Creating a video...")
        frame_size = (frame.shape[1], frame.shape[0])
        # video_filename = log_path + '/plot_video.avi'
        video_filename = folder + f"/videos/{today}_{now}.mp4"
        fps = 100
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')#(*'MJPG')
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            
    video_writer.write(frame)
    
    # print(f"volume {i}")
    # print(f"volume {i} - {time.time() - time_s} sec")

# os.remove("temp_frame.png")
video_writer.release()




#%%
video_writer.release()






#%% ----------------------------------------------------------
# >>>> START '追跡軌跡画像の生成'
# ------------------------------------------------------------
#%% ----------------------------------------------------------
# >>>> START '追跡軌跡画像の生成'
# ------------------------------------------------------------
use_trajectory = True

import csv
import pandas as pd

output_csv = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood\figs\filtered_coordinates_voxel_and_angle_v2.csv"
df = pd.read_csv(output_csv, header=None)
colormap = plt.colormaps['winter']

max_frame = 7000

# 利用可能な最大フレーム数と比較して安全に制限
max_frame = min(max_frame, len(df))
df = df.iloc[:max_frame]

print(f"処理対象フレーム数: {max_frame}")




# 小数点を丸める（必要であれば）
round_coords = True
if round_coords:
    df.iloc[:, 0:6] = df.iloc[:, 0:6].round().astype(int)


# pointsに追加する際もnumpy配列として操作
#points = np.empty((0, 3, 3), dtype=float)  # shape: (0, 4, 3)

# === 軌跡データ格納 ===
points = np.empty((0, 1, 3), dtype=float)  # shape: (0, 1, 3)
speeds = np.empty((0, 1, 1), dtype=float)  # shape: (0, 1, 1)


# Make log folder
#today = datetime.date.today()
#now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
#log_path = folder+"/log/"+str(today)+"/"+now
#os.makedirs(log_path, exist_ok=True)

# === ログフォルダ作成 ===
today = datetime.date.today()
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
log_path = os.path.join(folder, "log", str(today), now)
os.makedirs(log_path, exist_ok=True)


# 行数をカウント
with open(output_csv, 'r') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader)

print(f"行数: {row_count}")


# === 軌跡抽出ループ ===
for i, row in df.iterrows():
    # ch1の座標 (z, y, x)
    point = np.array([[row[0], row[1], row[2]]], dtype=float)
    point = np.expand_dims(point, axis=0)  # (1, 1, 3)
    points = np.append(points, point, axis=0)

    # 速度データは使わないので 0 を代入
    speed = np.array([[0.0]], dtype=float)
    speed = np.expand_dims(speed, axis=0)  # (1, 1, 1)
    speeds = np.append(speeds, speed, axis=0)


#for i, row in df.iterrows():
    ### coordinate
    # next2を一度にnumpy配列として作成
 #   point = np.array([
   #     [row[0], row[1], row[2]],
  #      [row[3], row[4], row[5]],
      # [row[8], row[9], row[10]],
        # [row[12], row[13], row[14]],
    #], dtype=float)

    # pointの形状を(1, 4, 3)に変換
    #point = np.expand_dims(point, axis=0)
    # pointsに追加
    #points = np.append(points, point, axis=0)


### MIP
# チャネル1（赤）とチャネル2（緑）の読み込み
#ch1 = tifffile.imread(folder + f"/images/ch_1/volume_t{7000}.tiff")
#ch2 = tifffile.imread(folder + f"/images/ch_2/volume_t{7000}.tiff")

# === MIP画像の作成 ===
ch1 = tifffile.imread(os.path.join(folder, "images/ch_1/volume_t7001.tiff"))
ch2 = tifffile.imread(os.path.join(folder, "images/ch_2/volume_t7001.tiff"))


# チャネルの正規化（uint16の場合を想定）
normalize_value = 450
ch1_norm = ch1 / normalize_value * 255
ch2_norm = ch2 / normalize_value * 255

# 最大強度投影（z方向）
ch1_mip = make_montage_img_v2(ch1_norm)
ch2_mip = make_montage_img_v2(ch2_norm)

ch1_adjusted = adjust_contrast(ch1_mip, 60, 0)
ch2_adjusted = adjust_contrast(ch2_mip, 60, 185)

# RGB画像の作成（ch1はマジェンタ: 赤と青、ch2は緑）
mip_image = np.zeros((*ch1_mip.shape, 3), dtype=np.float32)
mip_image[..., 0] = ch1_adjusted  # 赤チャネル（ch1から）
mip_image[..., 2] = ch1_adjusted  # 青チャネル（ch1から）
mip_image[..., 1] = ch2_adjusted  # 緑チャネル（ch2から）

### image shape
nz, ny, nx = ch1.shape
row_count = len(df)


### plot
make_track_frame_v2_np_save(
    log_path, 
    mip_image,
    nz,
    ny,
    nx,
    points, 
    colormap, 
    row_count, 
    use_trajectory,
)

# <<<< END
#%% ----------------------------------------------------------
# >>>> START '追跡軌跡画像の生成'
# ------------------------------------------------------------
use_trajectory = True

import csv
import pandas as pd

# folder = r"C:\Users\mlab-\Desktop\20241016-161347chlamyCYTO16verygood"
folder = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood"
output_csv = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood\figs\filtered_coordinates_voxel_and_angle_v2.csv"
df = pd.read_csv(output_csv, header=None)

colormap = plt.colormaps['winter']

max_frame = 7001

# 利用可能な最大フレーム数と比較して安全に制限
max_frame = min(max_frame, len(df))
df = df.iloc[:max_frame]

print(f"処理対象フレーム数: {max_frame}")




# 小数点を丸める（必要であれば）
#round_coords = True
#if round_coords:
 #   df.iloc[:, 0:6] = df.iloc[:, 0:6].round().astype(int)


# pointsに追加する際もnumpy配列として操作
#points = np.empty((0, 3, 3), dtype=float)  # shape: (0, 4, 3)

# === 軌跡データ格納 ===
points = np.empty((0, 1, 3), dtype=float)  # shape: (0, 1, 3)
speeds = np.empty((0, 1, 1), dtype=float)  # shape: (0, 1, 1)


# Make log folder
#today = datetime.date.today()
#now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
#log_path = folder+"/log/"+str(today)+"/"+now
#os.makedirs(log_path, exist_ok=True)

# === ログフォルダ作成 ===
today = datetime.date.today()
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
log_path = os.path.join(folder, "log", str(today), now)
os.makedirs(log_path, exist_ok=True)


# 行数をカウント
with open(output_csv, 'r') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader)

print(f"行数: {row_count}")


# === 軌跡抽出ループ ===
for i, row in df.iterrows():
    # ch1の座標 (z, y, x)
    point = np.array([[row[0], row[1], row[2]]], dtype=float)
    point = np.expand_dims(point, axis=0)  # (1, 1, 3)
    points = np.append(points, point, axis=0)

    # 速度データは使わないので 0 を代入
    speed = np.array([[0.0]], dtype=float)
    speed = np.expand_dims(speed, axis=0)  # (1, 1, 1)
    speeds = np.append(speeds, speed, axis=0)


#for i, row in df.iterrows():
    ### coordinate
    # next2を一度にnumpy配列として作成
 #   point = np.array([
   #     [row[0], row[1], row[2]],
  #      [row[3], row[4], row[5]],
      # [row[8], row[9], row[10]],
        # [row[12], row[13], row[14]],
    #], dtype=float)

    # pointの形状を(1, 4, 3)に変換
    #point = np.expand_dims(point, axis=0)
    # pointsに追加
    #points = np.append(points, point, axis=0)


### MIP
# チャネル1（赤）とチャネル2（緑）の読み込み
#ch1 = tifffile.imread(folder + f"/images/ch_1/volume_t{7000}.tiff")
#ch2 = tifffile.imread(folder + f"/images/ch_2/volume_t{7000}.tiff")

# === MIP画像の作成 ===
ch1 = tifffile.imread(os.path.join(folder, "images/ch_1/volume_t7000.tiff"))
ch2 = tifffile.imread(os.path.join(folder, "images/ch_2/volume_t7000.tiff"))


# チャネルの正規化（uint16の場合を想定）
normalize_value = 450
ch1_norm = ch1 / normalize_value * 255
ch2_norm = ch2 / normalize_value * 255

# 最大強度投影（z方向）
ch1_mip = make_montage_img_v2(ch1_norm)
ch2_mip = make_montage_img_v2(ch2_norm)

ch1_adjusted = adjust_contrast(ch1_mip, 60, 200)
ch2_adjusted = adjust_contrast(ch2_mip, 60, 190)

# RGB画像の作成（ch1はマジェンタ: 赤と青、ch2は緑）
mip_image = np.zeros((*ch1_mip.shape, 3), dtype=np.float32)
mip_image[..., 0] = ch1_adjusted  # 赤チャネル（ch1から）
mip_image[..., 2] = ch1_adjusted  # 青チャネル（ch1から）
mip_image[..., 1] = ch2_adjusted  # 緑チャネル（ch2から）

### image shape
nz, ny, nx = ch1.shape
row_count = len(df)


### plot
make_track_frame_v2_np_save(
    log_path, 
    mip_image,
    nz,
    ny,
    nx,
    points, 
    colormap, 
    row_count, 
    use_trajectory,
)

# <<<< END


#%% ----------------------------------------------------------
# >>>> START '追跡軌跡画像（角度）の生成'
# ------------------------------------------------------------

#%% ----------------------------------------------------------
# >>>> START '追跡軌跡画像（角度）の生成'
# ------------------------------------------------------------
import os
import csv
import datetime
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

# 事前に make_montage_img_v2 と adjust_contrast を定義済みであることを前提

# === 設定 ===
use_trajectory = True
folder = r"H:\Chlamydomonas_Analysis\20241016-161347chlamyCYTO16verygood"
output_csv = os.path.join(folder, "figs", "filtered_coordinates_voxel_and_angle_v2.csv")
df = pd.read_csv(output_csv, header=None)

# 角度データはそのまま（マイナスを維持）
df = df.iloc[:7001]
angle_min = df.iloc[:, 6].min()
angle_max = df.iloc[:, 6].max()
print(f"角度の範囲: {angle_min:.2f}° ~ {angle_max:.2f}°")

colormap = plt.colormaps['hsv']
points = np.empty((0, 1, 3), dtype=float)
speeds = np.empty((0, 1, 1), dtype=float)

today = datetime.date.today()
now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
log_path = os.path.join(folder, "log", str(today), now)
os.makedirs(log_path, exist_ok=True)

for _, row in df.iterrows():
    point = np.array([[row[0], row[1], row[2]]], dtype=float)
    point = np.expand_dims(point, axis=0)
    points = np.append(points, point, axis=0)

    speed = np.array([[row[6]]], dtype=float)
    speed = np.expand_dims(speed, axis=0)
    speeds = np.append(speeds, speed, axis=0)

# 画像読み込みとMIP処理
ch1 = tifffile.imread(folder + f"/images/ch_1/volume_t{7000}.tiff")
ch2 = tifffile.imread(folder + f"/images/ch_2/volume_t{7000}.tiff")

normalize_value = 450
ch1_mip = make_montage_img_v2(ch1 / normalize_value * 255)
ch2_mip = make_montage_img_v2(ch2 / normalize_value * 255)

ch1_adjusted = adjust_contrast(ch1_mip, 60, 200)
ch2_adjusted = adjust_contrast(ch2_mip, 60, 190)

mip_image = np.zeros((*ch1_mip.shape, 3), dtype=np.float32)
mip_image[..., 0] = ch1_adjusted
mip_image[..., 2] = ch1_adjusted
mip_image[..., 1] = ch2_adjusted

nz, ny, nx = ch1.shape

# === 可視化関数 ===
def make_track_speed_frame_v2_np_degree_save(
    path, img, nz, ny, nx, points_np, speed_np, cmap,
    num_volume, trajectory=True
):
    plt.figure()
    plt.imshow(img)
    border_thickness = 2

    if trajectory:
        for frame_num in range(1, len(points_np)):
            previous_coords = points_np[frame_num - 1]
            current_coords = points_np[frame_num]

            for obj_idx in range(previous_coords.shape[0]):
                speed_val = speed_np[frame_num - 1, obj_idx]
                # ラップしてカラー値に変換（-180°と180°を同色に）
                wrapped_angle = float(speed_val) % 360
                norm_speed_val = wrapped_angle / 360.0
                color = cmap(norm_speed_val)

                z0, y0, x0 = previous_coords[obj_idx]
                z1, y1, x1 = current_coords[obj_idx]

                plt.plot([x0 + border_thickness, x1 + border_thickness],
                         [y0 + border_thickness, y1 + border_thickness],
                         color=color, linewidth=0.5)

                plt.plot([x0 + border_thickness, x1 + border_thickness],
                         [z0 + ny + border_thickness * 2, z1 + ny + border_thickness * 2],
                         color=color, linewidth=0.5)

                plt.plot([z0 + nx + border_thickness * 2, z1 + nx + border_thickness * 2],
                         [y0 + border_thickness, y1 + border_thickness],
                         color=color, linewidth=0.5)

    # スケールバー
    plt.plot([125, 145], [ny - 5, ny - 5], color='white', linewidth=2)

    # ラベル
    plt.text(border_thickness+2, ny + border_thickness-2, "xy", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="bottom", ha="left")
    plt.text(border_thickness+2, ny + border_thickness * 2+2, "xz", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="top", ha="left")
    plt.text(nx + border_thickness * 2+2, border_thickness+2, "yz", color="white", fontsize=10,
             fontname="Segoe UI", fontstyle="italic", va="top", ha="left")

    # 時間表示
    plt.text(border_thickness+3, border_thickness+3,
             f"T = {len(points_np) - 1} ms", color="white", fontsize=11,
             fontname="Segoe UI", va="top", ha="left")

    # === カラーバー（-360〜0 の範囲で表示）===
    norm = plt.Normalize(vmin=-360, vmax=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.set_label("Degree")

    tick_values = np.linspace(-360, 0, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.0f}" for v in tick_values])

    plt.axis('off')
    plt.savefig(os.path.join(path, "degree.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

# === 実行 ===
make_track_speed_frame_v2_np_degree_save(
    log_path,
    mip_image,
    nz,
    ny,
    nx,
    points,
    speeds,
    colormap,
    len(points),
    use_trajectory
)

# 必要なら保存（これは make_ 関数内で保存済なので基本不要）
save_file = os.path.join(folder, "degree.png")
plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0)
print(f"保存完了: {save_file}")



#%% ----------------------------------------------------------
# >>>> START '速度csvの間引き'
# ------------------------------------------------------------
input_csv_path = r"c:\Users\mlab-\Desktop\20241016-161347chlamyCYTO16verygood\figs\20241121161907 - ch1 complete\tracking_speed.csv"
output_csv_path = r"c:\Users\mlab-\Desktop\20241016-161347chlamyCYTO16verygood\figs\20241121161907 - ch1 complete\tracking_speed_2.csv"
extract_rows_from_csv(
    input_csv_path=input_csv_path,
    output_csv_path=output_csv_path,
    n=2,
)

#%% ----------------------------------------------------------
# >>>> START '追跡軌跡画像（速度）の生成'
# ------------------------------------------------------------
use_trajectory = True

import csv
import pandas as pd
folder = r"C:\Users\mlab-\Desktop\20241016-161347chlamyCYTO16verygood"
output_csv = r"C:\Users\mlab-\Desktop\20241016-161347chlamyCYTO16verygood\figs\20241121161907 - ch1 complete\tracking_speed_10.csv"
df = pd.read_csv(output_csv, header=None)
colormap = plt.colormaps['rainbow']

# pointsに追加する際もnumpy配列として操作
points = np.empty((0, 1, 3), dtype=float)  # shape: (0, 4, 3)
speeds = np.empty((0, 1, 1), dtype=float)  # shape: (0, 4)

# Make log folder
today = datetime.date.today()
now = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
log_path = folder+"/log/"+str(today)+"/"+now
os.makedirs(log_path, exist_ok=True)

# 行数をカウント
with open(output_csv, 'r') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader)

print(f"行数: {row_count}")

df = df.iloc[:7001]
for i, row in df.iterrows():
    ### coordinate
    # next2を一度にnumpy配列として作成
    point = np.array([
        # [row[0], row[1], row[2]],
        [row[4], row[5], row[6]],
        # [row[8], row[9], row[10]],
        # [row[12], row[13], row[14]],
    ], dtype=float)

    # pointの形状を(1, 4, 3)に変換
    point = np.expand_dims(point, axis=0)
    # pointsに追加
    points = np.append(points, point, axis=0)
    
    speed = np.array([
        # [row[3]],
        [row[7]],
        # [row[11]],
        # [row[15]],
    ], dtype=float)
    
    # speedの形状を(1, 4, 1)に変換
    speed = np.expand_dims(speed, axis=0)
    # speedsに追加
    speeds = np.append(speeds, speed, axis=0)
    
# def lowpass_filter(data, cutoff, fs, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data, axis=0)
#     return y

# # Apply lowpass filter to speeds
# cutoff_frequency = 10  # Adjust the cutoff frequency as needed
# sampling_rate = 1000  # Assuming the data is sampled at 1 Hz, adjust as needed
# filtered_speeds = lowpass_filter(speeds, cutoff_frequency, sampling_rate)
# # Save speeds to CSV
# speeds_csv_path = folder + "/speeds.csv"
# with open(speeds_csv_path, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     for speed in speeds:
#         csvwriter.writerow(speed.flatten())

### MIP
# チャネル1（赤）とチャネル2（緑）の読み込み
ch1 = tifffile.imread(folder + f"/images/ch_1/volume_t{0}.tiff")
ch2 = tifffile.imread(folder + f"/images/ch_2/volume_t{0}.tiff")

# チャネルの正規化（uint16の場合を想定）
normalize_value = 450
ch1_norm = ch1 / normalize_value * 255
ch2_norm = ch2 / normalize_value * 255

# 最大強度投影（z方向）
ch1_mip = make_montage_img_v2(ch1_norm)
ch2_mip = make_montage_img_v2(ch2_norm)

ch1_adjusted = adjust_contrast(ch1_mip, 60, 255)
ch2_adjusted = adjust_contrast(ch2_mip, 60, 255)

# RGB画像の作成（ch1はマジェンタ: 赤と青、ch2は緑）
mip_image = np.zeros((*ch1_mip.shape, 3), dtype=np.float32)
mip_image[..., 0] = ch1_adjusted  # 赤チャネル（ch1から）
mip_image[..., 2] = ch1_adjusted  # 青チャネル（ch1から）
mip_image[..., 1] = ch2_adjusted  # 緑チャネル（ch2から）

### image shape
nz, ny, nx = ch1.shape

### plot
make_track_speed_frame_v2_np_save_scatter(
    log_path, 
    mip_image,
    nz,
    ny,
    nx,
    points, 
    speeds,
    colormap, 
    row_count, 
    use_trajectory,
)

# <<<< END





#%% ----------------------------------------------------------
# >>>> START '移動速度確率分布図'
# ------------------------------------------------------------

csv_paths = r"C:\Users\mlab-\Desktop\Chlamydomonas\log\2024-11-21\20241121161907 - ch1 complete\tracking_speed.csv"
csv_paths = r"C:\Users\mlab-\Desktop\20241016-160147chlamyCYTO16verygood\log\2024-12-10\20241210191216 - ch1 complete\tracking_speed.csv"
import winsound
import matplotlib.pyplot as plt
import pandas as pd

csv_paths = csv_paths.replace("\\", "/")
print(csv_paths)

# bright image csv file's path
df = pd.read_csv(csv_paths, header=None)

list_speeds = []
for i, row in tqdm(df.iterrows(), desc="Creating Video"):
    next2 = []
    points_per_row = len(row) // 4  # 各行に含まれるポイント数を計算
    for j in range(points_per_row):
        idx = j * 4 + 3
        next2.append([row[idx]])

    speeds = np.array(next2, float)
    list_speeds.append(speeds)
    
list_speeds = np.array(list_speeds)
transposed_speeds = list_speeds.T
    
# velocity 列の範囲ごとのカウント数を計算
bins = list(range(0, 2000, 1)) #+ [float('inf')]  # -5から505までの10刻みとそれ以上
# labels = [f'{i+5}' for i in range(-5, 500, 10)] #+ ['500+']  # ラベルの設定
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]  # ビンの中心を計算


# 各点ごとの速度分布プロット
for point_idx in range(len(list_speeds[0])):  # 点ごとにループ
    all_speeds = [speeds[point_idx] for speeds in list_speeds]  # 各時点での特定点の速度を収集
    all_speeds = np.array(all_speeds)

    # ヒストグラムで頻度をカウントし、確率（発生率）に変換
    counts, _ = np.histogram(all_speeds, bins=bins)
    total_count = counts.sum()
    probability = counts / total_count  # 発生率を計算

    # 発生率のプロット
    plt.rcParams['xtick.direction'] = 'in'  # メモリの向き in：内向き、out：外向き
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(5, 3))
    plt.plot(bin_centers, probability, lw=1, marker='^', markersize=4, label=f'Point {point_idx + 1}')
    # marker=['o'点, 's'四角, 'x'バツ, 'D'ひし形, 'v'下向き三角形, '^'上向き三角形, '<'左向き三角形, '>'右向き三角形, 'p'五角形, 'h'六角形, '+'プラス, '8'八角形, 'd'ひし形,  '.'点,]
    plt.xlabel('Speed [μm/s]', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.title(f'Speed Probability for Point {point_idx + 1}')
    plt.grid(True)

    # グラフの枠線の設定
    plt.xlim(left=0)  # X軸の左端を0に設定
    plt.ylim(bottom=0)  # Y軸の下端を0に設定
    plt.gca().spines['right'].set_visible(False)    # グラフの枠 False：非表示
    plt.gca().spines['top'].set_visible(False)
    plt.grid(False)

    # X軸のラベルをカスタマイズ（0.00を0に変更）
    yticks = plt.gca().get_yticks()
    ytick_labels = [f'{int(x)}' if x == 0 else f'{x:.2f}' for x in yticks if x >= 0]
    plt.yticks(yticks[yticks >= 0], ytick_labels)

    # # 表示するY軸のラベルを選択して設定
    # selected_ticks = list(range(0, len(labels), 5))  # 5刻みでラベルを選択
    # selected_labels = [labels[i] for i in selected_ticks]  # 選択したラベル
    # plt.xticks(ticks=selected_ticks, labels=selected_labels)  # ラベルを設定

    # グラフを保存
    output_path = csv_paths.replace("/tracking_speed.csv", f"/Probability_Point_{point_idx + 1}.png")
    plt.savefig(output_path, pad_inches=0, bbox_inches='tight', dpi=300)  # ここでファイル名と形式を指定

plt.show()
plt.close()
print("All process is finished")



#%%
import winsound
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

# 色とマーカーのリストを定義
colors = ['green','red', 'blue', 'orange', 'magenta', 'purple',    'darkorange','cyan','darkgreen',]
markers = ['^', 'o', 's', 'D', 'x', 'v', 'p', '*', '+']

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(figsize=(5, 3))

for h in range(len(csv_paths)):
    csv_paths = csv_paths.replace("\\", "/")
    print(csv_paths)

    # bright image csv file's path
    file_path = csv_paths

    df = pd.read_csv(file_path, encoding='shift_jis')
    
    if len(df) > 1000:
        df = df.iloc[1000:]  # 最初の1000行を除外

    # velocity 列の範囲ごとのカウント数を計算
    bins = list(range(-5, 510, 10))  # -5から505までの10刻みとそれ以上
    labels = [f'{i+5}' for i in range(-5, 500, 10)]  # ラベルの設定

    df['velocity_bin'] = pd.cut(df['velocity[μm/s]'], bins=bins, labels=labels, right=False)

    # 範囲ごとのカウント数を計算
    counts = df['velocity_bin'].value_counts().sort_index()

    # 全体に対する割合を計算
    total_count = len(df)
    probability = counts / total_count

    # 個体ごとに異なる色とマーカーを使用してプロット
    plt.plot(probability.index, probability.values, lw=1, marker=markers[h % len(markers)], 
             markersize=4, color=colors[h % len(colors)], label=f'Worm {h+1}')

# グラフの設定
plt.ylabel('Probability')#, fontsize=15)
plt.xlabel('Speed [μm/s]')#, fontsize=15)
# plt.title('Speed vs Probability')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(False)

# X軸のラベルをカスタマイズ（0.00を0に変更）
yticks = plt.gca().get_yticks()
ytick_labels = [f'{int(y)}' if y == 0 else f'{y:.2f}' for y in yticks if y >= 0]
plt.yticks(yticks[yticks >= 0], ytick_labels)

# 表示するY軸のラベルを選択して設定
selected_ticks = list(range(0, len(labels), 5))  # 5刻みでラベルを選択
selected_labels = [labels[i] for i in selected_ticks]  # 選択したラベル
plt.xticks(ticks=selected_ticks, labels=selected_labels)

# 凡例を追加
plt.legend()

# グラフを保存
new_path = csv_paths.replace("/tracking_speed.csv", "")  # 同じ保存先を使用
plt.savefig(new_path, pad_inches=0, bbox_inches='tight', dpi=300)

# グラフを表示
plt.show()

print("All process is finished")
winsound.Beep(1000, 200)
