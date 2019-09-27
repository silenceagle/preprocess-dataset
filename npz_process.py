"""
    npz_process.py
    20190902
"""
import numpy as np
import os


def npyz_print_data_shape(npyz_file_path):
    """
    show data shape stored in given npy or npz file.
    :param npyz_file_path: the npy or npz file's full path
    :return: True if success.
    """
    if not os.path.exists(npyz_file_path):
        raise ValueError(f"path not found: {npyz_file_path}")
    extension = os.path.splitext(npyz_file_path)[1][1:]
    if extension == 'npy':
        npy_data = np.load(npyz_file_path)
        print(f'data shape of {npyz_file_path}:')
        print(npy_data.shape)
        return True
    elif extension == 'npz':
        npz_data = np.load(npyz_file_path)
        print(f'information of {npyz_file_path}')
        for file in npz_data.files:
            npy_data = npz_data[file]
            print(f'''{file}'s shape: {npy_data.shape}''')
        print('Done!')
        return True
    else:
        raise ValueError('only support .npy or .npz file!')


if __name__ == "__main__":
    pass