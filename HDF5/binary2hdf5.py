import struct
import sys
import os
from pathlib import Path

import numpy as np
import h5py
import time


def transform_to_hdf5(binary_file_path: str, compression: str = "gzip"):
    assert compression in ("none", "gzip", "lzf"), "'compression' should be 'none', 'gzip', or 'lzf'!"
    t0 = time.time()
    with open(binary_file_path, 'rb') as f:
        raw_dim_img = f.read(4)
        dim_img, = struct.unpack(">i", raw_dim_img)
        print("Dim_img：", dim_img)
        assert dim_img == 3, "dim_img should be 3!"

        raw_resolutions = f.read(dim_img * 8)
        size_z, size_y, size_x = struct.unpack('>ddd', raw_resolutions)
        print("size_z:", size_z, "um")
        print("size_y:", size_y, "um")
        print("size_x:", size_x, "um")

        raw_dim_data = f.read(4)
        dim_data, = struct.unpack(">i", raw_dim_data)
        print("Dim_data：", dim_data)
        assert dim_data == 5, "dim_data should be 5!"

        raw_shape = f.read(dim_data * 4)
        num_vol, num_channel, num_z, num_y, num_x = struct.unpack('>iiiii', raw_shape)
        print("num_vol:", num_vol)
        print("num_channel:", num_channel)
        print("num_x:", num_x)
        print("num_y:", num_y)
        print("num_z:", num_z)

        dirname = os.path.splitext(binary_file_path)[0]

        hdf5_path = Path(os.path.join(dirname+"_"+f"raw_{compression}.h5"))

        with h5py.File(hdf5_path, 'w') as f_h5:
            if compression == "none":
                dset = f_h5.create_dataset('default', shape=(num_vol, num_channel, num_z, num_y, num_x),
                                           chunks=(1, num_channel, num_z, num_y, num_x), dtype="uint16")
            elif compression == "gzip":
                dset = f_h5.create_dataset('default', shape=(num_vol, num_channel, num_z, num_y, num_x),
                                           chunks=(1, num_channel, num_z, num_y, num_x), dtype="uint16",
                                           compression="gzip",  compression_opts=1)
            elif compression == "lzf":
                dset = f_h5.create_dataset('default', shape=(num_vol, num_channel, num_z, num_y, num_x),
                                           chunks=(1, num_channel, num_z, num_y, num_x), dtype="uint16",
                                           compression="lzf")
            else:
                raise ValueError("compression should be either none, gzip or lzf")

            count = num_channel * num_z * num_y * num_x
            for t in range(num_vol):
                f.seek(16, 1)
                dset[t, ...] = np.fromfile(f, dtype='>u2', count=count).reshape(num_channel, num_z, num_y, num_x)
                simple_progress_bar(t + 1, num_vol)

            dset.attrs["size_x"] = size_x
            dset.attrs["size_y"] = size_y
            dset.attrs["size_z"] = size_z
            dset.attrs["shape"] = ("n_vol", "n_channel", "n_z", "n_y", "n_x")

    print(f"\nCreate hdf5 with compression={compression} takes {time.time() - t0:.3f}s")
    total_size = hdf5_path.stat().st_size / 1024 ** 3
    print(f"Total size of all hdf5 files in './{hdf5_path}': {total_size:.4f} GB")



def transform_hdf5(binary_file_path: str, hdf5_file_path: str, compression: str = "gzip"):
    t0 = time.time()
    with open(binary_file_path, 'rb') as f:
        dim_img=3
            
        print("Dim_img：", dim_img)

        size_z = 0.4095
        size_y = 0.4095
        size_x = 0.4095
        print("size_z:", size_z, "um")
        print("size_y:", size_y, "um")
        print("size_x:", size_x, "um")

        num_vol = 9995
        num_channel = 2
        num_z = 97
        num_y = 300
        num_x = 498
        print("num_vol:", num_vol)
        print("num_channel:", num_channel)
        print("num_x:", num_x)
        print("num_y:", num_y)
        print("num_z:", num_z)

        hdf5_path = Path(f"raw_{compression}.h5")

        with h5py.File(f"raw_{compression}.h5", 'w') as f_h5:
            if compression == "none":
                dset = f_h5.create_dataset('default', shape=(num_vol, num_channel, num_z, num_y, num_x),
                                           chunks=(1, num_channel, num_z, num_y, num_x), dtype="uint16")
            elif compression == "gzip":
                dset = f_h5.create_dataset('default', shape=(num_vol, num_channel, num_z, num_y, num_x),
                                           chunks=(1, num_channel, num_z, num_y, num_x), dtype="uint16",
                                           compression="gzip",  compression_opts=1)
            elif compression == "lzf":
                dset = f_h5.create_dataset('default', shape=(num_vol, num_channel, num_z, num_y, num_x),
                                           chunks=(1, num_channel, num_z, num_y, num_x), dtype="uint16",
                                           compression="lzf")
            else:
                raise ValueError("compression should be either none, gzip or lzf")

            with h5py.File(hdf5_file_path,"r") as h5:
                for t in range(num_vol):
                    dset[t, ...] = h5['virtual_array'][t]
                    simple_progress_bar(t + 1, num_vol)

                dset.attrs["size_x"] = size_x
                dset.attrs["size_y"] = size_y
                dset.attrs["size_z"] = size_z
                dset.attrs["shape"] = ("n_vol", "n_channel", "n_z", "n_y", "n_x")

    print(f"\nCreate hdf5 with compression={compression} takes {time.time() - t0:.3f}s")
    total_size = hdf5_path.stat().st_size / 1024 ** 3
    print(f"Total size of all hdf5 files in './{hdf5_path}': {total_size:.4f} GB")




def simple_progress_bar(current, total, bar_length=50):
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rProgress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def read_hdf5_at_t(hdf5_path: str, t: int) -> np.ndarray:
    assert t >= 1
    with h5py.File(hdf5_path, 'r') as file:
        t0 = time.time()
        array = file['default']
        array_t = array[t - 1, :, :, :, :]
        print(f"Read images at t={t} takes {time.time() - t0:.3f}s")
        # print(f"Voxel size (x, y, z): ({array.attrs['size_x']}, {array.attrs['size_y']}, {array.attrs['size_z']}) um")
        print(f"Image shape (vols, channels, z, y, x): {array.shape}")
    return array_t

def read_virtual_hdf5(hdf5_path: str, t: int) -> np.ndarray:
    assert t >= 1
    with h5py.File(str(hdf5_path), 'r') as f_virtual:
        t0 = time.time()
        array = f_virtual['default']
        array_t = array[t - 1, :, :, :]
        # print(f"Read images at t={t} takes {time.time() - t0:.3f}s")
        # print(f"Voxel size (x, y, z): ({array.attrs['size_x']}, {array.attrs['size_y']}, {array.attrs['size_z']}) um")
        # print(f"Image shape (vols, channels, z, y, x): {array.shape}")
    return array_t