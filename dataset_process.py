"""
    dataset_process.py
    process dataset
    20181108
"""


import tensorflow as tf
import numpy as np
import time
import os
from libtiff import TIFF
import shutil
from tqdm import tqdm
import logging
import cv2


def generate_train_validation_dataset_multisize(
        source_path, validation_proportion):
    """
    load multisize source images and generate train and validation datasets
    :param source_path: the store path of source images
    :param validation_proportion: the validation dataset proportion of total
            dataset
    :return: train_x, train_y, train_x_shape,
               validation_x, validation_y, validation_x_shape,
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    number_of_categories = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            number_of_categories += 1
    number_of_image_per_category = np.zeros(number_of_categories, dtype=int)
    category_name = []
    dataset_x = []
    dataset_x_shap = []
    index_category = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            index_category += 1
            number_of_images = 0
            category_name.append(category.name)
            for img_file in os.scandir(category.path):
                extension = os.path.splitext(img_file.path)[1][1:]
                if extension == 'tif':
                    number_of_images += 1
                    tif = TIFF.open(img_file.path, mode='r')
                    image = tif.read_image()
                    dataset_x_shap.append([image.shape[0], image.shape[1]])
                    dataset_x.append(np.reshape(
                        np.sqrt(np.power(image[:, :, 0], 2) +
                                np.power(image[:, :, 1], 2)), (1, -1),
                        order='C'))
            number_of_image_per_category[index_category-1] = number_of_images
    # print(number_of_image_per_category)
    dataset_y = np.zeros(
        [sum(number_of_image_per_category), number_of_categories],
        dtype=np.int32)
    for index_category in range(number_of_categories):
        dataset_y[sum(number_of_image_per_category[0:index_category]):
                  sum(number_of_image_per_category[0:index_category+1]),
                  index_category] = 1
    # print(len(dataset_x))
    number_total_images = len(dataset_x)
    number_validation_images = int(np.floor(number_total_images *
                                            validation_proportion))
    validation_indices = np.random.choice(
        number_total_images, number_validation_images, replace=False)
    validation_x = []
    validation_x_shape = []
    for indices in validation_indices:  # 'List' object, more complicated !!!
        validation_x.append(dataset_x[indices])
        validation_x_shape.append(dataset_x_shap[indices])
    validation_y = dataset_y[validation_indices]
    train_indices = np.array(
        list(set(range(number_total_images)) - set(validation_indices)))
    train_x = []
    train_x_shape = []
    for indices in train_indices:
        train_x.append(dataset_x[indices])
        train_x_shape.append(dataset_x_shap[indices])
    train_y = dataset_y[train_indices]
    return train_x, train_y, train_x_shape,\
        validation_x, validation_y, validation_x_shape


def generate_train_validation_dataset_singlesize(
        source_path, save_path, npz_file_name, validation_proportion,
        file_extension='png'):
    """
    load single size source images and generate train and validation datasets
    (the struct of source path must be :source_path/category_name/image files)
    and save them in a .npz file, with variable named 'train_x', 'train_y',
    'validation_x', 'validation_y'
    :param source_path: the store path of source images
    :param save_path: the path to save .npz file
    :param validation_proportion: the validation dataset proportion of total
            dataset
    :param file_extension: the image files' extension, default to 'png'
    :return: train_x, train_y, validation_x, validation_y
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    number_of_categories = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            number_of_categories += 1
    number_of_image_per_category = np.zeros(number_of_categories, dtype=int)
    category_name = []
    index_category = 0
    is_first = True
    is_first_total = True
    for category in os.scandir(source_path):
        if category.is_dir():
            index_category += 1
            number_of_images = 0
            category_name.append(category.name)
            for img_file in os.scandir(category.path):
                extension = os.path.splitext(img_file.path)[1][1:]
                if extension == file_extension:
                    number_of_images += 1
                    temp_img = cv2.imread(img_file.path, -1)
                    if is_first:
                        dataset_x = np.reshape(temp_img, [1, -1])
                        is_first = False
                    else:
                        dataset_x = np.append(dataset_x,
                                              np.reshape(temp_img, [1, -1]),
                                              axis=0)
            is_first = True
            dataset_y = np.zeros([number_of_images, number_of_categories],
                                 dtype=np.int32)
            dataset_y[:, index_category-1] = 1
            number_of_image_per_category[index_category-1] = number_of_images
            this_validation_size = int(np.floor(number_of_images *
                                                validation_proportion))
            this_validation_indices = np.random.choice(
                number_of_images, this_validation_size, replace=False)
            this_train_indices = np.array(
                list(set(range(number_of_images)) -
                     set(this_validation_indices)))
            if is_first_total:
                train_x = dataset_x[this_train_indices]
                train_y = dataset_y[this_train_indices]
                validation_x = dataset_x[this_validation_indices]
                validation_y = dataset_y[this_validation_indices]
                is_first_total = False
            else:
                train_x = np.append(train_x, dataset_x[this_train_indices],
                                    axis=0)
                train_y = np.append(train_y, dataset_y[this_train_indices],
                                    axis=0)
                validation_x = np.append(validation_x,
                                         dataset_x[this_validation_indices],
                                         axis=0)
                validation_y = np.append(validation_y,
                                         dataset_y[this_validation_indices],
                                         axis=0)
    # print(number_of_image_per_category)
    np.savez(os.path.join(save_path, npz_file_name), train_x=train_x,
             train_y=train_y, validation_x=validation_x,
             validation_y=validation_y)
    return train_x, train_y, validation_x, validation_y


def generate_dataset_singlesize_with_category(
        source_path, save_path, npz_file_name, file_extension='png'):
    """
    load single size source images and generate  datasets
    (the struct of source path must be :source_path/category_name/image files)
    and save them in a .npz file, with variable named 'x', 'y'
    :param source_path: the store path of source images
    :param save_path: the path to save .npz file
    :param file_extension: the image files' extension, default to 'png'
    :return: dataset_x, dataset_y, category_name
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    number_of_categories = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            number_of_categories += 1
    number_of_image_per_category = np.zeros(number_of_categories, dtype=int)
    category_name = []
    dataset_x = []
    dataset_y = []
    index_category = 0
    is_first = True
    is_first_total = True
    for category in os.scandir(source_path):
        if category.is_dir():
            index_category += 1
            number_of_images = 0
            category_name.append(category.name)
            for img_file in os.scandir(category.path):
                extension = os.path.splitext(img_file.path)[1][1:]
                if extension == file_extension:
                    number_of_images += 1
                    if extension == 'npy':
                        temp_img = np.load(img_file.path)
                    else:
                        temp_img = cv2.imread(img_file.path, -1)
                    if is_first:
                        this_dataset_x = np.reshape(temp_img, [1, -1])
                        is_first = False
                    else:
                        this_dataset_x = np.append(this_dataset_x,
                                                   np.reshape(temp_img, [1, -1]),
                                                   axis=0)
            is_first = True
            this_dataset_y = np.zeros([number_of_images, number_of_categories],
                                      dtype=np.int32)
            this_dataset_y[:, index_category-1] = 1
            number_of_image_per_category[index_category-1] = number_of_images
            if is_first_total:
                dataset_x = this_dataset_x
                dataset_y = this_dataset_y
                is_first_total = False
            else:
                dataset_x = np.append(dataset_x, this_dataset_x, axis=0)
                dataset_y = np.append(dataset_y, this_dataset_y, axis=0)
    np.savez(os.path.join(save_path, npz_file_name), x=dataset_x, y=dataset_y)
    return dataset_x, dataset_y, category_name


def generate_train_test_validation_dataset_multisize(
        source_path, test_proportion, validation_proportion):
    """
    load source images and generate train, validation and test datasets
    :param source_path: the store path of source images
    :param test_proportion: the test dataset proportion of total dataset
    :param validation_proportion: the validation dataset proportion of total
            dataset
    :return: train_x, train_y, train_x_shape,
               validation_x, validation_y, validation_x_shape,
               test_x, test_y, test_x_shape
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    number_of_categories = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            number_of_categories += 1
    number_of_image_per_category = np.zeros(number_of_categories, dtype=int)
    category_name = []
    dataset_x = []
    dataset_x_shap = []
    index_category = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            index_category += 1
            number_of_images = 0
            category_name.append(category.name)
            for img_file in os.scandir(category.path):
                extension = os.path.splitext(img_file.path)[1][1:]
                if extension == 'tif':
                    number_of_images += 1
                    tif = TIFF.open(img_file.path, mode='r')
                    image = tif.read_image()
                    dataset_x_shap.append([image.shape[0], image.shape[1]])
                    dataset_x.append(np.reshape(
                        np.sqrt(np.power(image[:, :, 0], 2) +
                                np.power(image[:, :, 1], 2)), (1, -1),
                        order='C'))
            number_of_image_per_category[index_category-1] = number_of_images
    # print(number_of_image_per_category)
    dataset_y = np.zeros(
        [sum(number_of_image_per_category), number_of_categories],
        dtype=np.int32)
    for index_category in range(number_of_categories):
        dataset_y[sum(number_of_image_per_category[0:index_category]):
                  sum(number_of_image_per_category[0:index_category+1]),
                  index_category] = 1
    # print(len(dataset_x))
    number_total_images = len(dataset_x)
    number_test_images = int(np.floor(number_total_images * test_proportion))
    number_validation_images = int(np.floor(number_total_images *
                                            validation_proportion))
    test_indices = np.random.choice(
        number_total_images, number_test_images, replace=False)  # 不放回抽样
    test_x = []
    test_x_shape = []
    for indices in test_indices: # 'List' object, more complicated !!!
        test_x.append(dataset_x[indices])
        test_x_shape.append(dataset_x_shap[indices])
    test_y = dataset_y[test_indices]
    train_validation_indices = np.array(
        list(set(range(number_total_images)) - set(test_indices)))
    validation_indices = train_validation_indices[np.random.choice(
        number_total_images-number_test_images, number_validation_images,
        replace=False)]
    train_indices = list(set(train_validation_indices) -
                         set(validation_indices))
    train_x = []
    train_x_shape = []
    for indices in train_indices:
        train_x.append(dataset_x[indices])
        train_x_shape.append(dataset_x_shap[indices])
    train_y = dataset_y[train_indices]
    validation_x = []
    validation_x_shape = []
    for indices in validation_indices:
        validation_x.append(dataset_x[indices])
        validation_x_shape.append(dataset_x_shap[indices])
    validation_y = dataset_y[validation_indices]
    return train_x, train_y, train_x_shape,\
        validation_x, validation_y, validation_x_shape, \
        test_x, test_y, test_x_shape


def _random_choose_fix_proportion_train_test_files(
        source_path, train_save_path, test_save_path, train_files_proportion):
    """
    random pick fix proportion of files from source path and copy it to train
    folder, the rest copy to test folder
    :param source_path: the source images' store folder, struct should be source_path/image_files
    :param train_save_path: the folder to store picked train image files
    :param test_save_path: the folder to store picked test image files
    :param train_files_proportion: the train files' number to be picked
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    file_name_list = os.listdir(source_path)
    total_files_number = len(file_name_list)
    train_files_number = int(
        np.ceil(total_files_number * train_files_proportion))
    train_files_indices = np.random.choice(
        total_files_number, train_files_number, replace=False)
    test_files_indices = np.array(
        list(set(range(total_files_number)) - set(train_files_indices)))
    for file_index in train_files_indices:
        shutil.copyfile(os.path.join(source_path, file_name_list[file_index]),
                        os.path.join(train_save_path,
                                     file_name_list[file_index]))
    for file_index in test_files_indices:
        shutil.copyfile(os.path.join(source_path, file_name_list[file_index]),
                        os.path.join(test_save_path,
                                     file_name_list[file_index]))
    return True


def split_dataset_to_train_test_with_category(
        source_path, save_path, train_file_proportion=0.5):
    """
    split the total dataset to two part: train and test
    :param source_path: dataset path, struct should be source_path/category/image_files
    :param save_path: train and test folders' root path
    :param train_file_proportion: train proportion of total dataset
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    pbar = tqdm(os.scandir(source_path))
    for sub_folder in pbar:
        pbar.set_description("Processing %s" % sub_folder.name)
        if sub_folder.is_dir():
            train_save_path = os.path.join(save_path, 'train', sub_folder.name)
            test_save_path = os.path.join(save_path, 'test', sub_folder.name)
            _random_choose_fix_proportion_train_test_files(
                sub_folder.path, train_save_path, test_save_path,
                train_file_proportion)
    return True


def random_choose_fix_number_image_files_with_category(source_path, save_path, pick_file_number):
    """
    random pick fix number of files from source path and copy them to save_path/category/images
    :param source_path: the source images' store folder, struct should be source_path/category/image_files
    :param save_path: the folder to store picked image files
    :param pick_file_number: the number of image files to pick
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    this_pbar = tqdm(os.scandir(source_path))
    for category in this_pbar:
        if category.is_dir():
            this_pbar.set_description("Processing %s" % category.name)
            os.makedirs(os.path.join(save_path, category.name), exist_ok=True)
            file_name_list = os.listdir(os.path.join(source_path, category.name))
            total_files_number = len(file_name_list)
            pick_files_indices = np.random.choice(total_files_number, pick_file_number, replace=False)
            for file_index in pick_files_indices:
                shutil.copyfile(os.path.join(source_path, category.name, file_name_list[file_index]),
                                os.path.join(save_path, category.name, file_name_list[file_index]))
    return True


def random_choose_fix_number_image_files(source_path, save_path, pick_file_number):
    """
    random pick fix number of files from source path and copy them to save_path/images
    :param source_path: the source images' store folder, struct should be source_path/image_files
    :param save_path: the folder to store picked image files
    :param pick_file_number: the number of image files to pick
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    file_name_list = os.listdir(source_path)
    total_files_number = len(file_name_list)
    pick_files_indices = np.random.choice(total_files_number, pick_file_number, replace=False)
    for file_index in pick_files_indices:
        shutil.copyfile(os.path.join(source_path, file_name_list[file_index]),
                        os.path.join(save_path, file_name_list[file_index]))
    return True


def data_norm_zscore(src_data):
    """
        normalization function : norm = (true - mean) / std
        only can be used to 2D matrix, each row represents a data point, such as an image
    """
    if not len(src_data.shape) == 2:
        raise ValueError("normalization only support 2D data, but get %d data" % len(src_data.shape))
    norm_data = np.zeros(src_data.shape, dtype=np.float32)
    for ind_r in tqdm(range(src_data.shape[0])):
            this_mean = np.mean(src_data[ind_r])
            this_std = np.std(src_data[ind_r])
            # print(this_mean)
            norm_data[ind_r] = (src_data[ind_r]*1.0 - this_mean) / this_std
    # print("normalization data size : %d × %d" % (norm_data.shape[0], norm_data.shape[1]))
    return norm_data


def data_norm_minmax(src_data):
    """
        normalization function : norm = (true - min) / (max - min)
        only can be used to 2D matrix, each row represents a data point, such as an image
    """
    if not len(src_data.shape) == 2:
        raise ValueError("normalization only support 2D data, but get %d data" % len(src_data.shape))
    norm_data = np.zeros(src_data.shape, dtype=np.float32)
    for ind_r in tqdm(range(src_data.shape[0])):
            this_min = np.min(src_data[ind_r])
            this_max = np.max(src_data[ind_r])
            # print(this_mean)
            norm_data[ind_r] = (src_data[ind_r]*1.0 - this_min) / (this_max - this_min)
    # print("normalization data size : %d × %d" % (norm_data.shape[0], norm_data.shape[1]))
    return norm_data


def copy_npy_files_with_category(source_path, save_path):
    """
    copy npy files from source path to save path
    :param source_path: struct: source_path/category/npy files
    :param save_path: save path
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            pbar = tqdm(os.scandir(category.path))
            for npy_files in pbar:
                if npy_files.is_file():
                    extension = os.path.splitext(npy_files.path)[1][1:]
                    if extension == 'npy':
                        pbar.set_description("Processing %s" % npy_files.name)
                        os.makedirs(os.path.join(save_path, category.name), exist_ok=True)
                        shutil.copyfile(npy_files.path,
                                        os.path.join(save_path, category.name, npy_files.name))
    return True


# multisize
def random_perm3(data_x, data_y, data_z):
    """
        do random perm on x, y and z, x and z are list object, y is ndarray object
        :param data_x: x data, list object
        :param data_y: label data, ndarray object
        :param data_z: x data size, list object
    """
    data_size = data_y.shape[0]
    rand_perm = np.arange(data_size)
    np.random.shuffle(rand_perm)
    random_data_x = []
    random_data_z = []
    for indices in rand_perm:  # 'List' object, more complicated !!!
        random_data_x.append(data_x[indices])
        random_data_z.append(data_z[indices])
    # random_data_x = data_x[rand_perm]
    random_data_y = data_y[rand_perm]
    # random_data_z = data_z[rand_perm]
    return random_data_x, random_data_y, random_data_z


def generate_dataset_multisize_with_cagtegory(source_path):
    """
    load multisize source images and generate datasets
    :param source_path: the store path of source images, source_path/category/image files
    :return: x, y, x_shape,
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    number_of_categories = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            number_of_categories += 1
    number_of_image_per_category = np.zeros(number_of_categories, dtype=np.int32)
    category_name = []
    dataset_x = []
    dataset_x_shape = []
    index_category = 0
    for category in os.scandir(source_path):
        if category.is_dir():
            index_category += 1
            number_of_images = 0
            category_name.append(category.name)
            for img_file in os.scandir(category.path):
                extension = os.path.splitext(img_file.path)[1][1:]
                if extension == 'tif':
                    number_of_images += 1
                    tif = TIFF.open(img_file.path, mode='r')
                    image = tif.read_image()
                    dataset_x_shape.append([image.shape[0], image.shape[1]])
                    this_x = np.reshape(np.sqrt(np.power(image[:, :, 0], 2) + np.power(image[:, :, 1], 2)), (1, -1),
                                        order='C')
                    this_x_norml2 = (this_x * 1.0) / np.sqrt(np.sum(np.square(this_x)))
                    dataset_x.append(this_x_norml2)
            number_of_image_per_category[index_category-1] = number_of_images
    # print(number_of_image_per_category)
    dataset_y = np.zeros(
        [sum(number_of_image_per_category), number_of_categories],
        dtype=np.int32)
    for index_category in range(number_of_categories):
        dataset_y[sum(number_of_image_per_category[0:index_category]):
                  sum(number_of_image_per_category[0:index_category+1]),
                  index_category] = 1
    # print(len(dataset_x))
    return dataset_x, dataset_y, dataset_x_shape


def generate_dataset_multisize(source_path, number_category, this_category_index):
    """
    load multisize source images and generate datasets
    :param source_path: the store path of source images, source_path/image files
    :param number_category: for generating one hot label y
    :param this_category_index: for generating one hot label y
    :return: x, y, x_shape,
    """
    if not os.path.exists(source_path):
        raise FileExistsError('file not found! : %s' % source_path)
    dataset_x = []
    dataset_x_shape = []
    for img_file in os.scandir(source_path):
        extension = os.path.splitext(img_file.path)[1][1:]
        if extension == 'tif':
            tif = TIFF.open(img_file.path, mode='r')
            image = tif.read_image()
            dataset_x_shape.append([image.shape[0], image.shape[1]])
            this_x = np.reshape(np.sqrt(np.power(image[:, :, 0], 2) + np.power(image[:, :, 1], 2)), (1, -1),
                                order='C')
            this_x_norml2 = (this_x * 1.0) / np.sqrt(np.sum(np.square(this_x)))  # norm with L2
            dataset_x.append(this_x_norml2)
    # print(number_of_image_per_category)
    dataset_y = np.zeros([len(dataset_x), number_category], dtype=np.int32)
    dataset_y[:, this_category_index] = 1
    # print(len(dataset_x))
    return dataset_x, dataset_y, dataset_x_shape


def random_choose_fix_number_pattern_multise(x, y, shape_matrix, pick_number):
    """
    random pick fix number of patterns form x, y and shape_matrix
    :param x: a list object, per item stores a ndarray, which store a image data
    :param y: a ndarray object, is the one hot labels
    :param shape_matrix: a list object, each object is also a list object : [image_height, img_width], respect to x
    :param pick_number: the number of pattern to pick
    :return: picked_x, picked_y, picked_shape
    """
    total_files_number = len(x)
    pick_indices = np.random.choice(total_files_number, pick_number, replace=False)
    picked_x = []
    picked_shape = []
    for indices in pick_indices:  # 'List' object, more complicated !!!
        random_data_x.append(data_x[indices])
        random_data_z.append(data_z[indices])
    return True


if __name__ == '__main__':
    pass

