## data_prepare.py

import os
# from libtiff import TIFF
import numpy as np
import cv2


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


# def generate_dataset_multisize(source_path, file_extension='tif'):
#     """
#     load multisize source images and generate datasets
#     :param source_path: the store path of source images, source_path/category/image files
#     :param file_extension: image files' extension, default to 'tif', support tif, png, npy
#     :return: x, y, x_shape
#     """
#     if not os.path.exists(source_path):
#         raise FileExistsError('file not found! : %s' % source_path)
#     number_of_categories = 0
#     for category in os.scandir(source_path):
#         if category.is_dir():
#             number_of_categories += 1
#     number_of_image_per_category = np.zeros(number_of_categories, dtype=np.int32)
#     category_name = []
#     dataset_x = []
#     dataset_x_shape = []
#     index_category = 0
#     for category in os.scandir(source_path):
#         if category.is_dir():
#             index_category += 1
#             number_of_images = 0
#             category_name.append(category.name)
#             image = []
#             for img_file in os.scandir(category.path):
#                 extension = os.path.splitext(img_file.path)[1][1:]
#                 if file_extension == extension:
#                     number_of_images += 1
#                     if extension == 'tif':
#                         tif = TIFF.open(img_file.path, mode='r')
#                         image = tif.read_image()
#                         this_x = np.reshape(np.sqrt(np.power(image[:, :, 0], 2) + np.power(image[:, :, 1], 2)), (1, -1),
#                                             order='C')
#                     elif extension == 'png':
#                         image = cv2.imread(img_file.path, -1)
#                         this_x = image
#                     elif extension == 'npy':
#                         image = np.load(img_file.path)
#                         this_x = image
#                     else:
#                         raise ValueError('''unsupported image file's extension: %s''' % file_extension)
#                     dataset_x_shape.append([image.shape[0], image.shape[1]])
#                     this_x_norml2 = (this_x * 1.0) / np.sqrt(np.sum(np.square(this_x)))
#                     dataset_x.append(this_x_norml2)
#             number_of_image_per_category[index_category-1] = number_of_images
#     # print(number_of_image_per_category)
#     dataset_y = np.zeros(
#         [sum(number_of_image_per_category), number_of_categories],
#         dtype=np.int32)
#     for index_category in range(number_of_categories):
#         dataset_y[sum(number_of_image_per_category[0:index_category]):
#                   sum(number_of_image_per_category[0:index_category+1]),
#                   index_category] = 1
#     # print(len(dataset_x))
#     return dataset_x, dataset_y, dataset_x_shape


def norm_with_l2(original_mat):
    """
    devided by original mat's L2 norm to got identity length mat
    each row is a datapoint
    :param original_mat:
    :return: normed mat
    """
    normed_mat = np.zeros(original_mat.shape, dtype=np.float32)
    if len(original_mat.shape) == 2:
        for ind_r in range(original_mat.shape[0]):
            a = np.square(original_mat[ind_r]*1.0)
            b = np.sum(a)
            c = np.sqrt(b)
            normed_mat[ind_r] = (original_mat[ind_r] * 1.0) / c
            # normed_mat[ind_r] = (original_mat[ind_r] * 1.0) / np.sqrt(np.sum(np.square(original_mat[ind_r])*1.0))
    return normed_mat

