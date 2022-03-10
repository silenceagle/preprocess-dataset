"""
    untarFiles.py
    20220309
"""
import tarfile
import os
import scipy.io as sio
from tqdm import tqdm


def un_tar(file_name, output_root_path='../train', new_folder_name=None):
    """
        extract the files in the file_name in to output_root_path/new_folder_name, and new_folder_name will be
        the name of 'tar' file if new_folder_name is None.
        So the output is organized in output_root_path/new_folder_name/image files.
        :param file_name: str, tar filename, end with '.tar'.
        :param output_root_path: str, the upper folder to save the extracted folder.
        :param new_folder_name: str or None, None means the extracted files will be saved in a folder whose name is same
            as the tar file's name, or the folder will be renamed to new_folder_name.
        :return: True
    """
    assert os.path.exists(file_name)
    assert file_name[-4:] == '.tar'
    # untar zip file to folder whose name is same as tar file
    tar = tarfile.open(file_name)
    names = tar.getnames()

    file_name = os.path.basename(file_name)
    if new_folder_name is None:
        extract_dir = os.path.join(output_root_path, file_name.split('.')[0])
    else:
        extract_dir = os.path.join(output_root_path, new_folder_name)

    os.makedirs(extract_dir, exist_ok=True)
    for name in names:
        tar.extract(name, extract_dir)
    tar.close()
    return True


def untar_train_images(train_path, output_root_path='../train', meta_data_for_name_mapping=None):
    """
    untar all the .tar files in train_path to output_root_path. If meta data is given, the tar files' destination
        folder name will be renamed to the 'words' corresponding to the tar filename('WNID').
        So the output is organized in output_root_path/'WIND' or 'words'/image files.
    :param train_path: str, the path that contains the train 'tar' files named by 'WNID'.
    :param output_root_path: str, the upper folder to save the extracted folder.
    :param meta_data_for_name_mapping: str or None, fullpath to 'meta.mat', this mat contains a array named with
        'synsets', and each item ia a cell with fields:
            "ILSVRC2012_ID, WNID, words, gloss, num_children, children, wordnet_height, num_train_images"
        if this parameter is given, the files in each tar file under the train_path will be extracted to a folder
        named by its corresponding 'words'.
    :return: True
    """
    assert os.path.exists(train_path)
    assert os.path.isdir(train_path)

    # read meta data
    if meta_data_for_name_mapping is not None:
        assert meta_data_for_name_mapping[-4:] == '.mat'
        meta_data_dict = dict()
        meta_data = sio.loadmat(meta_data_for_name_mapping)['synsets']
        for i in range(len(meta_data)):
            meta_data_dict[meta_data[i][0][1].item()] = meta_data[i][0][2].item()
    pbar = tqdm(os.scandir(train_path))
    for tar_file in pbar:
        if tar_file.is_file() and tar_file.name[-4:] == '.tar':
            WNID = tar_file.name.split('.')[0]
            new_folder_name = meta_data_dict[WNID] if meta_data is not None else WNID
            os.makedirs(os.path.join(output_root_path, new_folder_name), exist_ok=True)
            pbar.set_description(f'Extracting {tar_file.path} to {os.path.join(output_root_path, new_folder_name)}')
            un_tar(
                file_name=tar_file.path,
                output_root_path=output_root_path,
                new_folder_name=new_folder_name
            )
    return True


if __name__ == '__main__':
    untar_train_images(
        train_path=r'D:\tmp_dataset',
        output_root_path=r'D:\tmp_dataset\imagenet_train',
        meta_data_for_name_mapping=r'E:\datasets\ImageNet\ILSVRC2012_devkit_t12\data\meta.mat')



