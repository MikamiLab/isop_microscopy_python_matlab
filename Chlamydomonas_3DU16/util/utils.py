import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from typing import Callable, Iterable
import datetime, csv

from skimage.feature import peak_local_max
from cupyx.scipy.ndimage import affine_transform as cupy_affine_transform


def np2cp(array_np):
    """
    Convert a NumPy array to a CuPy array.

    Args:
        array_np (numpy.ndarray): The NumPy array to convert.

    Returns:
        cupy.ndarray: The converted CuPy array.
    """
    # Convert the NumPy array to a CuPy array and return it.
    return cp.asarray(array_np)

def cp2np(array_cp):
    """
    Convert a CuPy array to a NumPy array.

    Args:
        array_cp (cupy.ndarray): The CuPy array to convert.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    # Convert the CuPy array to a NumPy array and return it.
    return cp.asnumpy(array_cp)

def check_image_information(img):
    """
    Print basic information about an image array.

    Args:
        img (numpy.ndarray or cupy.ndarray): The image array to check.
    """
    # Print the type, shape, data type, and value range of the image.
    print("type: {}".format(type(img)))
    print("img_shape: {}".format(img.shape))
    print("img_data_shape: {}".format(img.dtype))
    print("img_data_range: ({}, {})".format(img.min(), img.max()))
    print("\n")

def do_sub_background(img, background, t, draw=False):
    """
    Subtract background from an image, setting pixels below the background level to 0.

    Args:
        img (numpy.ndarray): The input image.
        background (int): The background pixel value.
        t (int): The time point to process, used only when drawing.
        draw (bool, optional): If True, visualize the original and processed images.

    Returns:
        numpy.ndarray: The image with background subtracted.
    """
    # Subtract the background from the image, setting values below the background to 0.
    sub_img = np.where(img < background, 0, img)

    # Optionally draw the original and background-subtracted images.
    if draw:
        if img.ndim == 3:
            fig, axes = plt.subplots(1, 2, tight_layout=True)
            axes[0].imshow(img[t,:,:], cmap='hsv')
            axes[1].imshow(sub_img[t,:,:], cmap='hsv')
            plt.show()
        elif img.ndim == 2:
            fig, axes = plt.subplots(1, 2, tight_layout=True)
            axes[0].imshow(img, cmap='hsv')
            axes[1].imshow(sub_img, cmap='hsv')
            plt.show()
    else:
        return sub_img

def transform_for_napari(peaks_record):
    """
    Transform the peaks record array for visualization in napari.

    Args:
        peaks_record (cupy.ndarray): The peaks record array to transform.

    Returns:
        numpy.ndarray: The transformed array suitable for napari.
    """
    # Convert the CuPy array to a NumPy array for processing.
    peaks_record_cpu = cp2np(peaks_record)

    # Initialize a list to hold the transformed data.
    new_array_list = []

    # Reorganize the data for each peak across all time points.
    for peak_id in range(peaks_record_cpu.shape[1]):
        for t in range(peaks_record_cpu.shape[0]):
            new_array_list.append(peaks_record_cpu[t][peak_id])

    # Convert the list to a NumPy array.
    new_array = np.asarray(new_array_list)

    # Initialize a new array to include an additional column for time.
    new_array_2 = np.zeros((new_array.shape[0], peaks_record_cpu.shape[2] + 1))
    new_array_2[:, 1:] = new_array

    # Add the time dimension to the array.
    for i in range(0, new_array_2.shape[0], peaks_record_cpu.shape[0]):
        for j in range(peaks_record_cpu.shape[0]):
            new_array_2[i+j][1] = j

    # Add an identifier for each peak.
    id = 0
    for i in range(new_array_2.shape[0]):
        new_array_2[i, 0] = id
        if i % peaks_record_cpu.shape[0] == peaks_record_cpu.shape[0] - 1:
            id += 1

    # Return the transformed array suitable for visualization in napari.
    return new_array_2


#================================================================================================
#================================================================================================
# 以下 originalに追加した関数

# cropcalb.csvに項目を追加（事前計算）
def caluculate_y_shit_crop_parameter(cropcalib_path):

    with open(cropcalib_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 5:  # 5列目がある場合
                flag = False
            else:
                flag = True
                
    if flag == True:
        # CSVファイルを開いて処理
        with open(cropcalib_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # 計算結果を追加して新しい行を作成
        for m in range(len(rows) // 16):  # 行数の16分割のループ
            for i in range(16):
                row = rows[m*16 + i]
                # 計算
                y = round(2 * (float(row[1]) - float(rows[m*16][1])) / 2304)
                # 5列目に計算結果を追加
                row.append(y)

        # 同じファイルに書き込み
        with open(cropcalib_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
    else:
        print("The calculation results have already been written to the CSV file.")


# ライトシート画像を読み込む関数
def read_img(img_path):
    try:
        img = io.imread(img_path)
        # print("Image loading is successful")
        # img_height, img_width = img.shape[0], img.shape[1]
        # print(f"Image shape:{img.shape}")
        return img
    except:
        print("Image not found. Use dummy img!!")
        img = np.random.random((256, 256, 3))
        # img_height, img_width = img.shape[0], img.shape[1]
        # print(f"Dummy image shape:{img.shape}")
        return img
    
# 表示用
def plot_projection(imglist: Iterable[np.ndarray], backend: str='matplotlib'):
    if backend == 'matplotlib':
        for img in imglist:
            plt.imshow(img, cmap='gray')
            plt.show()
    else:
        Warning.warn(f"Backend {backend} is not supported")

def get_projection_montage(vol: np.ndarray, gap: int=10, proj_function: Callable=np.max) -> np.ndarray:
    assert len(vol.shape) == 3, "Input volume must be 3D"
    nz, ny, nx = vol.shape
    montage = np.zeros((ny+nz+gap, nx+nz+gap), dtype=vol.dtype)
    montage[:ny, :nx] = proj_function(vol, axis=0)
    montage[ny+gap:, :nx] = np.max(vol, axis=1)
    montage[:ny, nx+gap:] = np.max(vol, axis=2).transpose()
    
    return montage

def save_array_for_napari(peak_array, image_array):
    save_path = 'C:/Users/mikami/Desktop/Observer_Closed-loop-optogenetics/numpy_array/'
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    peak_file_name = f'peaks/peaks_{time_str}.npy'
    image_file_name = f'images/images_{time_str}.npy'
    np.save(save_path + peak_file_name, peak_array)
    np.save(save_path + image_file_name, image_array)


def draw(img, t=None, z=None, color="bwr"):
    """(t, y, x)の画像を描画

    Args:
        img (numpy.ndarray): 表示する画像 shape(t, y, x)
        t (int): 表示したい時刻
        color (str): 表示したいcolormap defalut:bwr
    """
    if img.ndim==3:
        plt.imshow(img[t, :, :], cmap=color)
        plt.show()

    elif img.ndim==4:
        plt.imshow(img[t, z, :, :], cmap=color)
        plt.show()


def draw_one_vol(img, z, color='bwr'):
    """任意の時刻のvolを描画

    Args:
        img (numpy.ndaary): (z, y, x)
        z (int, optional): どのzスライスか. Defaults to None.
        color (str, optional): matplotlibのcmap. Defaults to 'bwr'.
    """
    plt.imshow(img[z,:,:], cmap=color)
    plt.show()

def draw_2_img(img1, img2, row=1, col=2, t=0, z=44, color='bwr'):

    fig, axes = plt.subplots(row, col, tight_layout=True)
    print("img.shape: {}".format(img1.shape))

    if len(img1.shape)==4:
        axes[0].imshow(img1[t, z, :, :], cmap=color)
        axes[1].imshow(img2[t, z, :, :], cmap=color)
        plt.show()

    elif img1.ndim==3:
        axes[0].imshow(img1[z, :, :], cmap=color)
        axes[1].imshow(img2[z, :, :], cmap=color)

    elif len(img1.shape)==2:
        axes[0].imshow(img1, cmap=color)
        axes[1].imshow(img2, cmap=color)
        plt.show()


def draw_three_image(img1, img2, img3, t=1, z=1, row=1, col=3):
    fig, axes = plt.subplots(row, col, tight_layout=True)
    if len(img1.shape) == 4:
        print("img.shape: {}".format(img1.shape))
        axes[0].imshow(img1[t, z, :, :], cmap='gray')
        axes[1].imshow(img2[t, z, :, :], cmap='gray')
        axes[2].imshow(img3[t, z, :, :], cmap='gray')
        plt.show()

    elif len(img1.shape) == 3:
        print("img.shape: {}".format(img1.shape))
        axes[0].imshow(img1[t, :, :], cmap='gray')
        axes[1].imshow(img2[t, :, :], cmap='gray')
        axes[2].imshow(img3[t, :, :], cmap='gray')
        plt.show()

    elif len(img1.shape) == 2:
        print("img.shape: {}".format(img1.shape))
        axes[0].imshow(img1[:,:], cmap='gray')
        axes[1].imshow(img2[:,:], cmap='gray')
        axes[2].imshow(img3[:,:], cmap='gray')
        plt.show()

    else:
        print("Not supported shape. \n Please chcek")
        print("Your img shape: {}".format(img1.shape))

def do_binarize(img, threshold, max_value=65535):
    img_bool = img >threshold
    binary_img = img_bool * max_value
    plt.imshow(binary_img, cmap='gray')
    plt.show()


def do_binarize_zslice(img, t, threshold, max_value=65535):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img[t, :, :], cmap='gray')
    img_bool = img[t, :, :] > threshold
    binary_img = img_bool * max_value
    axes[1].imshow(binary_img, cmap='gray')
    plt.show()


def do_fft_zslice(img, t):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img[t, :, :], cmap='gray')    # 元画像表示
    # ここから高速フーリエ変換
    img_fq = np.fft.fft2(img[t, :, :])
    img_fq_shifted = np.fft.fftshift(img_fq)    # 低周波数成分を中央に寄せる
    magnitude_spectrum = 20 * np.log(np.abs(img_fq_shifted))
    axes[1].imshow(magnitude_spectrum, cmap='gray')
    plt.show()


def check_image_information(img):
    print("type: {}".format(type(img)))
    print("img_shape: {}".format(img.shape))
    print("img_data_shape: {}".format(img.dtype))
    print("img_data_range: ({}, {})".format(img.min(), img.max()))
    print("\n")


def do_sub_background(img, background, t, draw=False):
    """バックグラウンド減算
    バックグラウンドよりも小さい値は0、それ以上はそのまま

    Args:
        img (numpy.ndarray): 顕微鏡画像
        background (int): バックグラウンドの画素値

    Returns:
        numpy.ndarry: バックグラウンドが除去された画像
    """
    sub_img = np.where(img < background, 0, img)

    if draw:
        if img.ndim==3:
            fig, axes = plt.subplots(1, 2, tight_layout=True)
            axes[0].imshow(img[t,:,:], cmap='hsv')
            axes[1].imshow(sub_img[t,:,:], cmap='hsv')
            plt.show()


        elif img.ndim==2:
            fig, axes = plt.subplots(1, 2, tight_layout=True)
            axes[0].imshow(img, cmap='hsv')
            axes[1].imshow(sub_img, cmap='hsv')
            plt.show()

    else:
        return sub_img


def find_local_maxima(img_nom, min_distance, draw=False):
    coorinates = peak_local_max(img_nom, min_distance=min_distance)

    if draw:
        plt_local_maxima_2(img_nom, img_nom, coorinates)

    else:
        return coorinates


def plt_local_maxima(img1, img2, coordinates):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img1, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(img2, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')

    ax[2].imshow(img1, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')

    fig.tight_layout()

    plt.show()

def plt_local_maxima_2(img1, img2, coordinates):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img1, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(img1, cmap=plt.cm.gray)
    ax[1].autoscale(False)
    ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('Peak local max')

    fig.tight_layout()

    plt.show()

def custom_cmap(minval, maxval, cmap_name='custom_cmap'):
    colors = plt.cm.viridis(np.linspace(minval/maxval, 1, maxval-minval))
    new_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(cmap_name, colors)
    return new_cmap

class DoingFullyReconst:
    """
    Class for performing fully reconstruction operations on an input image.

    Attributes:
    - deskew_instance: Instance of the outer class (Deskew).

    Methods:
    - get_transform_corners: Calculate the transformed corner points of the input volume.
    - ceil_to_multiple: Rounds up the given number to the nearest multiple of the specified base.
    - calc_output_dimensions: Calculate the output dimensions based on the affine transformation matrix and the volume or shape.
    - shift_center: Shifts the center of the matrix by a specified direction.
    - unshift_center: Shifts the center of the matrix back to the original position.
    - scale_pixel_z: Scale the pixel values of the input reconstructed image along the z-axis.
    - rotation_around_y: Returns the rotation matrix for rotation around the y-axis by the specified angle.
    - rotate: Rotate the input image around the y-axis.
    - trim_data: Trims the data by removing the empty slices from the start and end.
    - zscale_rotate_trim: Applies z-scaling, rotation, and trimming operations to the input image.
    - reconst: Performs reconstruction operations on the input image.
    """

    def __init__(self, shear_factor, sheet_angle):
        self.shear_factor = shear_factor
        self.sheet_angle = sheet_angle
        

    def get_transform_corners(self, aff, vol_or_shape, zeroindex=True):
        """
        Calculate the transformed corner points of the input volume.

        Parameters:
        - aff: Transformation matrix.
        - vol_or_shape: Input volume or shape of the volume.
        - zeroindex: Boolean value indicating whether the corner points should be calculated for zero-indexed arrays (numpy) or one-indexed arrays (Matlab-style).

        Returns:
        - corner_array: Array containing the transformed corner points.
        """
        if cp.array(vol_or_shape).ndim == 3:
            d0, d1, d2 = cp.array(vol_or_shape).shape
        elif cp.array(vol_or_shape).ndim == 1:
            d0, d1, d2 = vol_or_shape
        else:
            raise ValueError

        if zeroindex:
            d0 -= 1
            d1 -= 1
            d2 -= 1

        corners_in = [
            (0, 0, 0, 1),
            (d0, 0, 0, 1),
            (0, d1, 0, 1),
            (0, 0, d2, 1),
            (d0, d1, 0, 1),
            (d0, 0, d2, 1),
            (0, d1, d2, 1),
            (d0, d1, d2, 1),
        ]

        corners_out = list(map(lambda c: aff @ cp.array(c), corners_in))
        corner_array = cp.concatenate(corners_out).reshape((-1, 4))

        return corner_array

    def ceil_to_multiple(self, x, base=4):
        """
        Rounds up the given number to the nearest multiple of the specified base.

        Parameters:
        - x: The number to be rounded up.
        - base: The base to which the number should be rounded up. Default is 4.

        Returns:
        - The rounded up number.
        """
        return (cp.int32(base) * cp.ceil(cp.array(x) / base)).astype(cp.int32)

    def calc_output_dimensions(self, aff, vol_or_shape):
        """
        Calculate the output dimensions based on the affine transformation matrix and the volume or shape.

        Parameters:
        - aff (array-like): The affine transformation matrix.
        - vol_or_shape (array-like): The volume or shape.

        Returns:
        array-like: The calculated output dimensions as an array of integers.
        """
        corners = self.get_transform_corners(aff, vol_or_shape)
        dims = cp.max(corners, axis=0) - cp.min(corners, axis=0) + 1
        dims = self.ceil_to_multiple(dims, base=2)
        return dims[:3].astype(cp.int32)

    def shift_center(self, matrix_shape, direction=-1.0):
        """
        Shifts the center of the matrix by a specified direction.

        Parameters:
        - matrix_shape: The shape of the matrix.
        - direction: The direction of the shift. Default is -1.0.

        Returns:
        - The shifted matrix.
        """
        center = cp.array(matrix_shape)/2
        shift = cp.eye(4)
        shift[0:3, 3] = direction * center
        return shift

    def unshift_center(self, matrix_shape):
        """
        Shifts the center of the matrix back to the original position.

        Parameters:
        - matrix_shape: The shape of the matrix.

        Returns:
        - The unshifted matrix.
        """
        return self.shift_center(matrix_shape, direction=1.0)

    def scale_pixel_z(self, reconst_img):
        """
        Scale the pixel values of the input reconstructed image along the z-axis.

        Args:
            reconst_img (ndarray): The input reconstructed image.

        Returns:
            ndarray: The scaled reconstructed image.
        """
        scale = cp.eye(4)
        scale[0,0] = self.shear_factor    # Actually scale[0,0]=np.sin(angle*np.pi/180.0)*dz_stage/xypixelsize, but if light-sheet_angle=45, it is equal to self.shear_factor
        _output_shape = self.calc_output_dimensions(scale, reconst_img)
        output_shape = (int(_output_shape[0]), int(_output_shape[1]), int(_output_shape[2]))
        # print(output_shape)
        # print(f"map type of output is {type(tuple(output_shape))}")
        # print(f"type(reconst_img) is {type(reconst_img)}")
        # print(f"reconst_img.dtype is {reconst_img.dtype}")
        reconst_img = cp.asarray(reconst_img).astype(cp.float32)
        # print(f"reconst_img.dtype is {reconst_img.dtype}")
        scale = scale.astype(cp.float32)
        # print(f"scale matrix: scale")
        scaled_vol = cupy_affine_transform(reconst_img.astype(cp.float32), cp.linalg.inv(scale.astype(cp.float32)), output_shape=output_shape, order=1, mode='constant')
        return scaled_vol
    
    def scale_pixel_z_return_all(self, reconst_img):
        """
        Scale the pixel values of the input reconstructed image along the z-axis.

        Args:
            reconst_img (ndarray): The input reconstructed image.

        Returns:
            ndarray: The scaled reconstructed image.
        """
        scale = cp.eye(4)
        scale[0,0] = self.shear_factor    # Actually scale[0,0]=np.sin(angle*np.pi/180.0)*dz_stage/xypixelsize, but if light-sheet_angle=45, it is equal to self.shear_factor
        input_shape = reconst_img.shape
        _output_shape = self.calc_output_dimensions(scale, reconst_img)
        output_shape = (int(_output_shape[0]), int(_output_shape[1]), int(_output_shape[2]))
        # print(output_shape)
        # print(f"map type of output is {type(tuple(output_shape))}")
        # print(f"type(reconst_img) is {type(reconst_img)}")
        # print(f"reconst_img.dtype is {reconst_img.dtype}")
        reconst_img = reconst_img.astype(cp.float32)
        # print(f"reconst_img.dtype is {reconst_img.dtype}")
        scale = scale.astype(cp.float32)
        # print(f"scale matrix: scale")
        scaled_vol = cupy_affine_transform(reconst_img.astype(cp.float32), cp.linalg.inv(scale.astype(cp.float32)), output_shape=output_shape, order=1, mode='constant')
        return scaled_vol, input_shape, output_shape, scale


    def rotation_around_y(self):
        """
        Returns the rotation matrix for rotation around the y-axis by the specified angle.

        Parameters:
        - self.sheet_angle: The angle of rotation in degrees.

        Returns:
        - rotation_mat: The rotation matrix for rotation around the y-axis.
        """
        angle_rad = cp.deg2rad(-self.sheet_angle)
        rotation_mat = cp.eye(4)
        rotation_mat[0,0] = cp.cos(angle_rad)
        rotation_mat[0,2] = cp.sin(angle_rad)
        rotation_mat[2,0] = -cp.sin(angle_rad)
        rotation_mat[2,2] = cp.cos(angle_rad)
        return rotation_mat

    def rotate(self, z_scaled_img):
        """
        Rotate the input image around the y-axis.

        Parameters:
        - z_scaled_img: numpy.ndarray
            The input image to be rotated.

        Returns:
        - numpy.ndarray
            The rotated image.
        """
        rotation_mat = self.rotation_around_y()
        shift_mat = self.shift_center(z_scaled_img.shape)
        _output_shape = self.calc_output_dimensions(rotation_mat @ shift_mat, z_scaled_img.shape)
        output_shape = (int(_output_shape[0]), int(_output_shape[1]), int(_output_shape[2]))
        unshift_mat = self.unshift_center(output_shape)
        # print(f"")
        rotated = cupy_affine_transform(z_scaled_img, cp.linalg.inv(unshift_mat @ rotation_mat @ shift_mat), output_shape=output_shape, order=1, mode='constant')
        return rotated
    
    def rotate_return_all(self, z_scaled_img):
        """
        Rotate the input image around the y-axis.

        Parameters:
        - z_scaled_img: numpy.ndarray
            The input image to be rotated.

        Returns:
        - numpy.ndarray
            The rotated image.
        """
        rotation_mat = self.rotation_around_y()
        shift_mat = self.shift_center(z_scaled_img.shape)
        input_shape = z_scaled_img.shape
        _output_shape = self.calc_output_dimensions(rotation_mat @ shift_mat, z_scaled_img.shape)
        output_shape = (int(_output_shape[0]), int(_output_shape[1]), int(_output_shape[2]))
        unshift_mat = self.unshift_center(output_shape)
        # print(f"")
        rotated = cupy_affine_transform(z_scaled_img, cp.linalg.inv(unshift_mat @ rotation_mat @ shift_mat), output_shape=output_shape, order=1, mode='constant')
        return rotated, input_shape, output_shape, rotation_mat, shift_mat, unshift_mat

    def trim_data(self, rotated_img):
        """
        Trims the data by removing the empty slices from the start and end.

        Parameters:
        - rotated_img (ndarray): The input image data.

        Returns:
        - ndarray: The trimmed image data.
        """
        data = rotated_img
        # Get the maximum value for each z-slice
        max_values = data.max(axis=(1, 2))
        # Find the index where the maximum value is non-zero for the first time
        start_index = (max_values > 0).tolist().index(True)
        # Find the index whose maximum value was last non-zero
        end_index = len(max_values) - (max_values[::-1] > 0).tolist().index(True)
        # print(f"Start index: {start_index}, End index: {end_index}")
        return data[start_index:end_index,:,:]

    def zscale_rotate_trim(self, deskewd_img):
        """
        Applies z-scaling, rotation, and trimming operations to the input image.

        Parameters:
        - deskewd_img: The input image to be processed.

        Returns:
        - fully_reconsted_img: The fully reconstructed image after applying z-scaling, rotation, and trimming.
        """
        scaled_img = self.scale_pixel_z(deskewd_img)
        rotated_img = self.rotate(scaled_img)
        fully_reconsted_img = self.trim_data(rotated_img)
        return cp.asnumpy(fully_reconsted_img)
    
    def only_zscale(self, deskewd_img):
        scaled_img = self.scale_pixel_z(deskewd_img)
        return cp.asnumpy(scaled_img)

    def trim_data_get_index(self, rotated_img):
        """
        Trims the data by removing the empty slices from the start and end.

        Parameters:
        - rotated_img (ndarray): The input image data.

        Returns:
        - ndarray: The trimmed image data.
        """
        data = rotated_img
        # Get the maximum value for each z-slice
        max_values = data.max(axis=(1, 2))
        # Find the index where the maximum value is non-zero for the first time
        start_index = (max_values > 0).tolist().index(True)
        # Find the index whose maximum value was last non-zero
        end_index = len(max_values) - (max_values[::-1] > 0).tolist().index(True)
        # print(f"Start index: {start_index}, End index: {end_index}")
        return data[start_index:end_index,:,:], start_index, end_index

    def zscale_rotate_trim_get_index(self, deskewd_img):
        """
        Applies z-scaling, rotation, and trimming operations to the input image.

        Parameters:
        - deskewd_img: The input image to be processed.

        Returns:
        - fully_reconsted_img: The fully reconstructed image after applying z-scaling, rotation, and trimming.
        """
        scaled_img = self.scale_pixel_z(deskewd_img)
        rotated_img = self.rotate(scaled_img)
        fully_reconsted_img, start_index, end_index = self.trim_data_get_index(rotated_img)
        return cp.asnumpy(fully_reconsted_img), start_index, end_index
    
    def zscale_rotate_trim_index(self, deskewd_img, start_index, end_index):
        """
        Applies z-scaling, rotation, and trimming operations to the input image.

        Parameters:
        - deskewd_img: The input image to be processed.

        Returns:
        - fully_reconsted_img: The fully reconstructed image after applying z-scaling, rotation, and trimming.
        """
        scaled_img = self.scale_pixel_z(deskewd_img)
        rotated_img = self.rotate(scaled_img)
        fully_reconsted_img = rotated_img[start_index:end_index,:,:]
        return cp.asnumpy(fully_reconsted_img)
    
    def zscale_rotate_trim_return_all(self, deskewd_img):
        scaled_img, input_shape_s, output_shape_s, scale = self.scale_pixel_z_return_all(deskewd_img)
        rotated_img, input_shape_r, output_shape_r, rotation_mat, shift_mat, unshift_mat = self.rotate_return_all(scaled_img)
        fully_reconsted_img, start_index, end_index = self.trim_data_get_index(rotated_img)
        return cp.asnumpy(fully_reconsted_img), input_shape_s, output_shape_s, scale, input_shape_r, output_shape_r, rotation_mat, shift_mat, unshift_mat, start_index, end_index

    def reconst(self, deskew_img):
        """
        Performs reconstruction operations on the input image.

        Parameters:
        - deskew_img: The input image to be reconstructed.

        Returns:
        - reconsted_img: The reconstructed image.
        """
        rotation_mat = self.rotation_around_y()
        # print(f"rotation_mat is {rotation_mat}")
        scale_mat = cp.eye(4)
        scale_mat[0,0] = self.shear_factor
        # print(f"scale_mat is {scale_mat}")
        shift_mat = self.shift_center(self.raw_img[1].shape)
        # print(f"shift_mat is {shift_mat}")
        combined_mat = rotation_mat @ scale_mat @ shift_mat
        output_shape = self.calc_output_dimensions(combined_mat, self.raw_img[1])
        unshift_mat = self.unshift_center(output_shape)
        all_in_one = unshift_mat @ rotation_mat @ scale_mat @ shift_mat
        # print(output_shape)
        output_shape = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
        reconsted_img = cupy_affine_transform(deskew_img, cp.linalg.inv(all_in_one), output_shape=output_shape, order=1, mode='constant')
        return reconsted_img
    
    def get_combined_transformation_matrix(self):
        # transformation matrix for z_scaling
        s_z = cp.eye(4)
        s_z[0,0] = self.shear_factor

        # transformation matrix for rotation around y-axis
        angle_rad = cp.deg2rad(-self.sheet_angle)
        r_y = cp.eye(4)
        r_y[0,0] = r_y[2,2] = cp.cos(angle_rad)
        r_y[0,2] = cp.sin(angle_rad)
        r_y[2,0] = -cp.sin(angle_rad)

        # get combined transformation matrix
        combined_matrix = r_y @ s_z    # Note the order in which the conversion matrices are multiplied
        return combined_matrix
    
    def get_transform_corners_matrixcalc(self, aff, vol_shape):
        # 変換後のボリュームのコーナーポイントを計算
        d0, d1, d2 = vol_shape
        corners = np.array([[0, 0, 0, 1], [d0, 0, 0, 1], [0, d1, 0, 1], [0, 0, d2, 1],
                            [d0, d1, 0, 1], [d0, 0, d2, 1], [0, d1, d2, 1], [d0, d1, d2, 1]])
        transformed_corners = np.dot(aff, corners.T).T
        return transformed_corners[:, :3]

    def calc_output_dimensions_matrixcalc(self, aff, vol_shape):
        transformed_corners = self.get_transform_corners_matrixcalc(aff, vol_shape)
        min_corner = np.min(transformed_corners, axis=0)
        max_corner = np.max(transformed_corners, axis=0)
        # output_shapeを計算する際に、結果を整数にキャストする
        output_shape = np.ceil(max_corner - min_corner).astype(np.int32)
        return tuple(output_shape.tolist())  # ndarrayをリストに変換し、それをタプルに変換


    def shift_center_matrixcalc(self, matrix_shape, direction=-1.0):
        # 中心移動の行列を作成
        center = np.array(matrix_shape) / 2
        center_shift = np.eye(4)
        center_shift[0:3, 3] = direction * center
        return center_shift

    def unshift_center_matrixcalc(self, matrix_shape):
        # 中心移動を元に戻す行列を作成
        return self.shift_center_matrixcalc(matrix_shape, direction=1.0)

    def create_composite_matrix_matrixcalc(self, vol_shape, z_scale_factor, y_rotate_angle):
        # 合成変換行列を作成
        shift_to_center = self.shift_center_matrixcalc(vol_shape)
        shift_back = self.unshift_center_matrixcalc(vol_shape)
        
        scale = np.eye(4)
        scale[0, 0] = z_scale_factor
        
        angle_rad = np.deg2rad(-y_rotate_angle)
        rotation = np.eye(4)
        rotation[0, 0] = rotation[2, 2] = np.cos(angle_rad)
        rotation[0, 2] = np.sin(angle_rad)
        rotation[2, 0] = -np.sin(angle_rad)
        
        composite_matrix = shift_back @ rotation @ scale @ shift_to_center
        return composite_matrix

    