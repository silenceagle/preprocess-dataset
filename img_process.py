# preprocess oriage img
# 20180916

import os
import cv2
import numpy as np
from tqdm import tqdm
from libtiff import TIFF
from scipy import misc


def gen_destfolder(src_basepath: str, dst_basepath=' '):
    src_category = []
    for entry in os.scandir(src_basepath):
        if entry.is_dir() and entry.name != 'out_dataset':
            src_category.append(entry.name)
    if dst_basepath == ' ':
        dst_basepath = os.path.join(src_basepath, 'out_dataset')
    if not os.path.exists(dst_basepath):
        os.makedirs(dst_basepath)
    return src_category, dst_basepath


def img_gray(src_basepath: str, dst_basepath, src_category: list, min_Width=11, min_Heigh=11, ext=['png']):
    """
        do graying to RGB image and save
        :param src_basepath: folder's path that stores original RGB images files
        :param src_category: the names of imgs' categories, which are also the sub folders' name of src_basepath
        :param dst_basepath: gray imgs' storaged path
        :param min_Heigh: the minimal images' height that will be grayed
        :param min_Width: the maximal images' width that will be grayed
        :param ext: the extension of the images to be grayed
        :return
    """
    for this_cate in src_category:
        this_cate_srcpath = os.path.join(src_basepath, this_cate)
        this_cate_dstpath = os.path.join(dst_basepath, 'gray', this_cate)
        if not os.path.exists(this_cate_dstpath):
            os.makedirs(this_cate_dstpath)
        pbar = tqdm(os.scandir(this_cate_srcpath))
        for entry in pbar:
            pbar.set_description("Processing %s" % entry.name)
            if entry.is_file():
                # get the file's extension
                extension = os.path.splitext(entry.path)[1][1:]
                if extension in ext:
                    # print("find png file: %s" % entry.name)
                    # gray img
                    tmp_img = cv2.imread(entry.path)
                    # print(entry.path)
                    img_h, img_w, img_c = tmp_img.shape
                    if img_h>=min_Heigh and img_w>=min_Width:
                        if img_c==3: # RGB图像
                            tmp_gray_img = np.zeros([img_h, img_w], dtype=np.int)
                            for ind_h in range(img_h):
                                for ind_w in range(img_w):
                                    tmp_gray_img[ind_h][ind_w] = np.int(0.39 * tmp_img[ind_h][ind_w][0] +
                                                                        0.5 * tmp_img[ind_h][ind_w][1] +
                                                                        0.11 * tmp_img[ind_h][ind_w][2])
                                    # gray = 0.39 * R + 0.5 * G + 0.11 * B
                            # save gray img
                            # print("\tsaving gray img: %s ..." % entry.name)
                            cv2.imwrite(os.path.join(this_cate_dstpath, entry.name), tmp_gray_img)
    return


def get_and_save_amplitude_image_slc(source_path, save_path, source_extension='tif'):
    """
    get the amplitude images of OpenSARShip SLC mode images and save .png images and the .npy files
    :param source_path: source images' root ptah
    :param save_path: the path to save processed images
    :param source_extension: source image files' extension, default to 'tif'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            os.makedirs(os.path.join(save_path, category.name), exist_ok=True)
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                extension = os.path.splitext(img_files.path)[1][1:]
                if extension == source_extension:
                    pbar.set_description("Processing %s" % img_files.name)
                    tif = TIFF.open(img_files.path, mode='r')
                    source_img = tif.read_image()
                    # 1st channel is the real part for VH
                    # 2st channel is the imaginary part for VH
                    # to get the amplitude value of VH
                    img_amplitude = np.sqrt(
                        np.square(source_img[:, :, 0]) +
                        np.square(source_img[:, :, 1]))
                    img_amplitude = np.reshape(img_amplitude, [img_amplitude.shape[0], img_amplitude.shape[1]])
                    filename_no_extension, _ = os.path.splitext(
                        img_files.name)
                    cv2.imwrite(os.path.join(save_path, category.name, filename_no_extension+'.png'), img_amplitude)
                    np.save(os.path.join(save_path, category.name, filename_no_extension+'.npy'), img_amplitude)
    return True


def gen_npy_file(source_path, save_path, npy_file_name, img_extension='png'):
    """
    concat images data with only one channel in soruce path and generate npy file
    :param source_path: images folder
    :param save_path: npy file's save path
    :param npy_file_name:
    :param img_extension: source images' file extension
    :return: Ture
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    pbar = tqdm(os.scandir(source_path))
    is_first = True
    total_data = []
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == img_extension:
                pbar.set_description("Processing %s" % img_files.name)
                source_img = cv2.imread(img_files.path, -1)
                if is_first:
                    total_data = np.reshape(source_img, [1, -1])
                    is_first = False
                else:
                    total_data = np.concatenate([total_data, np.reshape(source_img, [1, -1])], axis=0)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, npy_file_name), total_data)
    return True


# crop

def img_crop_from_center(
        original_img, crop_width, crop_height):
    """
        split a image with fixed size centrally
        :param original_img: the original img data matrix
        :param  crop_width: the sample image's width
        :param  crop_height: the sample image's height
        :return if successful, return splited image data, else False
    """
    if len(original_img.shape) == 3:
        ori_height, ori_width, chanels = original_img.shape
        if (ori_width >= crop_width) and (ori_height >= crop_width):
            up_left_raw = int((ori_height - crop_height) / 2)
            up_left_col = int((ori_width - crop_width) / 2)
            crop_img = original_img[
                           up_left_raw:up_left_raw+crop_height,
                           up_left_col:up_left_col+crop_width,
                           :]
            return crop_img
    else:
        ori_height, ori_width= original_img.shape
        if (ori_width >= crop_width) and (ori_height >= crop_width):
            up_left_raw = int((ori_height - crop_height) / 2)
            up_left_col = int((ori_width - crop_width) / 2)
            crop_img = original_img[
                           up_left_raw:up_left_raw+crop_height,
                           up_left_col:up_left_col+crop_width]
            return crop_img
    return False


def crop_imgs_and_save_smaller(source_path, save_path, crop_height: int, crop_width: int, extension='png'):
    """
    crop fix sized images and save them, also save the pictures whose size is smaller than crop window's size
    :param source_path: source images path: the struct should be source_path/category_folder/images files
    :param save_path: the processed images' save path
    :param crop_height: crop window's height
    :param crop_width: crop window's width
    :param extension: source images' file extension
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            os.makedirs(os.path.join(save_path, 'h'+str(crop_height)+'w'+str(crop_width), category.name), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'smaller_than_crop_window', 'h'+str(crop_height)+'w'+str(crop_width),
                category.name), exist_ok=True)
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                extension = os.path.splitext(img_files.path)[1][1:]
                if extension == extension:
                    pbar.set_description("Processing %s" % img_files.name)
                    img_data = cv2.imread(img_files.path, -1)
                    img_height = img_data.shape[0]
                    img_width = img_data.shape[1]
                    filename_no_extension, _ = os.path.splitext(img_files.name)
                    if img_height >= crop_height and img_width >= crop_width:
                        crop_img_data = img_crop_from_center(img_data, crop_width, crop_height)
                        cv2.imwrite(os.path.join(save_path, 'h'+str(crop_height)+'w'+str(crop_width), category.name,
                                                 filename_no_extension+'.png'), crop_img_data)
                    else:
                        cv2.imwrite(
                            os.path.join(save_path, 'smaller_than_crop_window',
                                         'h'+str(crop_height)+'w'+str(crop_width), category.name,
                                         filename_no_extension + '.png'), img_data)
    return True

# padding


def img_padding(src_basepath: str,  src_category=None, dst_basepath=None, out_size=[[64, 64],[128, 128]], ext=['png'],
                is_save_img = False):
    """
        padding zero to src imgs to get fix sized imgs, and save as npy files
        :param src_basepath: folder's path that stores original images files
        :param src_category: the names of imgs' categories, which are also the sub folders' name of src_basepath
        :param dst_basepath: processed imgs' storaged path, default to src_basepath/out_dataset
        :param out_size: every row represent one of the images' out size :[height, width],
                         and the row's index bigger, the size bigger.
        :param ext: the extension of the images to be processed
        :param is_save_img: whether to save processed images, default to False
        :return
    """
    # if source images' stored folder not exist, raise error
    if not os.path.exists(src_basepath):
        raise FileExistsError('path not found! : %s' % src_basepath)
    # if the source images' category information is not provided, then generate it.
    if src_category is None:
        src_category = []
        for entry in os.scandir(src_basepath):
            if entry.is_dir() and entry.name != 'out_dataset':
                src_category.append(entry.name)
    # if the after-processed images' stored folder is not provided, then set it to src_basepath\out_dataset
    if dst_basepath is None:
        dst_basepath = os.path.join(src_basepath, 'out_dataset')
    # generate folder to save images with size of outsize1 or outsize2
    num_outsize = np.array(out_size).shape[0]
    path_size = []
    for ind_size in range(num_outsize):
        path_size.append(os.path.join(dst_basepath, ('%d_%d' % (out_size[ind_size][0], out_size[ind_size][1]))))
        os.makedirs(path_size[ind_size], exist_ok=True)  # if exist, don't raise exception
    # do padding
    for this_cate in src_category:
        this_cate_srcpath = os.path.join(src_basepath, this_cate)
        pbar = tqdm(os.scandir(this_cate_srcpath))
        for entry in pbar:
            pbar.set_description("Processing %s" % entry.name)
            if entry.is_file():
                # get the file's extension
                extension = os.path.splitext(entry.path)[1][1:]
                if extension in ext:
                    # print('find image : %s' % entry.path)
                    # if extension == 'tiff':
                    #     tif = TIFF.open(entry.path, mode='r')
                    #     tmp_img = tif.read_image()
                    #     tif.close
                    # else:
                    #     tmp_img = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
                    tmp_img = cv2.imread(entry.path, -1)  # <0 returns the image as is
                    # print(tmp_img[85][108])
                    # print("image's shape:")
                    # print(tmp_img.shape)
                    # print(len(tmp_img.shape))
                    # img_h, img_w = tmp_img.shape
                    # this_size = 0
                    # for ind_size in range(num_outsize):
                    #     if img_h <= out_size[ind_size][0] and img_w <= out_size[ind_size][1]:
                    #         break
                    #     this_size += 1
                    # # print('out image size index : %d' % this_size)
                    # if this_size < num_outsize:  # image with valid size
                    #     # print('gray image')
                    #     save_img = np.zeros([out_size[this_size][0], out_size[this_size][1]], dtype=np.uint16)
                    #     start_r = int(np.floor((out_size[this_size][0]-img_h)/2.0))
                    #     start_c = int(np.floor((out_size[this_size][1]-img_w)/2.0))
                    #     for ind_r in range(img_h):
                    #         for ind_c in range(img_w):
                    #             save_img[start_r+ind_r, start_c+ind_c] = tmp_img[ind_r, ind_c]
                    is_valid, save_img, this_size= matrix_padding(tmp_img, out_size=out_size, dtype=np.uint16)
                    if is_valid:
                        if is_save_img:
                            os.makedirs(os.path.join(path_size[this_size], this_cate), exist_ok=True)
                            cv2.imwrite(os.path.join(path_size[this_size], this_cate, entry.name), save_img)
    return


def matrix_padding_multi_size_soft(src_mat, out_size=[[64, 64], [128, 128]], dtype=np.float32):
    """
        padding zero to src matrix to get larger size matrix
        if and only if source image's width and height are both small than out's, do padding
        :param src_mat: source matrix
        :param out_size: every row represent one of the images' out size :[height, width],
                         and the row's index bigger, the size bigger.
        :param dtype: the src matrix's date type, default to np.float32
        :return is_valid: if src_mat's size is smaller than out's , is_valid set to be Ture, else Flase
        :return out_mat: if is_valid is true, then out_mat be useful, else will be useless.
    """
    num_outsize = np.array(out_size).shape[0]
    img_h, img_w = src_mat.shape[0:2]
    this_size = 0
    for ind_size in range(num_outsize):
        if img_h <= out_size[ind_size][0] and img_w <= out_size[ind_size][1]:
            break
        this_size += 1
    # print('out image size index : %d' % this_size)

    is_valid = False
    if this_size < num_outsize:  # image with valid size
        out_mat = np.zeros([out_size[this_size][0], out_size[this_size][1]], dtype=dtype)
        # print('gray image')
        start_r = int(np.floor((out_size[this_size][0] - img_h) / 2.0))
        start_c = int(np.floor((out_size[this_size][1] - img_w) / 2.0))
        for ind_r in range(img_h):
            for ind_c in range(img_w):
                out_mat[start_r + ind_r, start_c + ind_c] = src_mat[ind_r, ind_c]
        is_valid = True
        return is_valid, out_mat, this_size
    else:
        return is_valid, 0, 0


def matrix_padding_force_to_get_fix_sized_matrix(src_mat, padding_height, padding_width):
    """
    force padding zero to original matrix
        ori width <= padding width and ori height <= padding height: padding around
        ori width <= padding width and ori height > padding height: padding on width direction, crop on height direction
        ori width > padding width and ori height <= padding height: crop on W directory, padding on H directory
        ori width > padding width and ori height > padding height: crop around
    :param src_mat: source matrix
    :param padding_height: out matrix's height
    :param padding_width: out matrix's width
    :return: padding_mat
    """
    if len(src_mat.shape) == 2:
        src_height, src_width = src_mat.shape
        if src_width <= padding_width and src_height <= padding_height:
            padding_mat = np.zeros([padding_height, padding_width])
            start_h = int(np.floor((padding_height - src_height) / 2.0))
            start_w = int(np.floor((padding_width - src_width) / 2.0))
            padding_mat[start_h:start_h+src_height, start_w:start_w+src_width] = src_mat
        elif src_width <= padding_width and src_height > padding_height:
            padding_mat = np.zeros([padding_height, padding_width])
            start_h = int(np.floor((src_height - padding_height) / 2.0))
            start_w = int(np.floor((padding_width - src_width) / 2.0))
            padding_mat[:, start_w:start_w+src_width] = src_mat[start_h:start_h+padding_height, :]
        elif src_width > padding_width and src_height <= padding_height:
            padding_mat = np.zeros([padding_height, padding_width])
            start_h = int(np.floor((padding_height - src_height) / 2.0))
            start_w = int(np.floor((src_width - padding_width) / 2.0))
            padding_mat[start_h:start_h+src_height, :] = src_mat[:, start_w:start_w+padding_width]
        else:
            start_h = int(np.floor((src_height - padding_height) / 2.0))
            start_w = int(np.floor((src_width - padding_width) / 2.0))
            padding_mat = src_mat[start_h:start_h+padding_height, start_w:start_w+padding_width]
    else:
        src_height, src_width, channels = src_mat.shape
        if src_width <= padding_width and src_height <= padding_height:
            padding_mat = np.zeros([padding_height, padding_width, channels])
            start_h = int(np.floor((padding_height - src_height) / 2.0))
            start_w = int(np.floor((padding_width - src_width) / 2.0))
            padding_mat[start_h:start_h+src_height, start_w:start_w+src_width, :] = src_mat
        elif src_width <= padding_width and src_height > padding_height:
            padding_mat = np.zeros([padding_height, padding_width, channels])
            start_h = int(np.floor((src_height - padding_height) / 2.0))
            start_w = int(np.floor((padding_width - src_width) / 2.0))
            padding_mat[:, start_w:start_w+src_width] = src_mat[start_h:start_h+padding_height, :]
        elif src_width > padding_width and src_height <= padding_height:
            padding_mat = np.zeros([padding_height, padding_width, channels])
            start_h = int(np.floor((padding_height - src_height) / 2.0))
            start_w = int(np.floor((src_width - padding_width) / 2.0))
            padding_mat[start_h:start_h+src_height, :] = src_mat[:, start_w:start_w+padding_width]
        else:
            start_h = int(np.floor((src_height - padding_height) / 2.0))
            start_w = int(np.floor((src_width - padding_width) / 2.0))
            padding_mat = src_mat[start_h:start_h + padding_height, start_w:start_w+padding_width]
    return padding_mat


def dataset_padding(src_mat, ori_size, out_size=[[128, 128]], dtype=np.float32):
    """
            padding zero to src matrix to get larger size matrix
            :param src_mat: source matrix, every line represent a matrix which has been reshaped to a
                             vector with lenth ori_size[0]*ori_size[1]
            :param ori_size: original matrix size [height, width]
            :param out_size: every row represent one of the images' out size :[height, width],
                             and the row's index bigger, the size bigger.
            :param dtype: the src matrix's date type, default to np.float32
            :return out_mat: if is_valid is true, then out_mat be useful, else will be useless.
    """
    if not len(np.array(src_mat).shape) == 2:
        raise ValueError('input must be 2D matrix!')
    out_mat = np.zeros((src_mat.shape[0], out_size[0][0]*out_size[0][1]), dtype=dtype)
    for ind_mat in range(src_mat.shape[0]):
        _, tmp_mat, _ = matrix_padding_multi_size_soft(np.reshape(src_mat[ind_mat], (ori_size[0], ori_size[1])),
                                    out_size=out_size, dtype=dtype)
        out_mat[ind_mat] = np.reshape(tmp_mat, out_size[0][0]*out_size[0][1])
    return out_mat


def padding_to_fix_sized_and_save_imgs(source_path, save_path, padding_height, padding_width, source_extension='png'):
    """
    padding zeros to images data to get fix sized images and save them. But if the source images's size is larger than
    the required size, do croping on them
    :param source_path: source images path: the struct should be source_path/category_folder/images files
    :param save_path: the processed images' save path
    :param padding_height: output image's height
    :param padding_width: output image's width
    :param extension: source images' file extension
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            os.makedirs(os.path.join(save_path, category.name), exist_ok=True)
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                extension = os.path.splitext(img_files.path)[1][1:]
                if extension == source_extension:
                    pbar.set_description("Processing %s" % img_files.name)
                    img_data = cv2.imread(img_files.path, -1)
                    filename_no_extension, _ = os.path.splitext(img_files.name)
                    padding_img = matrix_padding_force_to_get_fix_sized_matrix(img_data, padding_height, padding_width)
                    cv2.imwrite(os.path.join(save_path, category.name, filename_no_extension+'.png'), padding_img)
    return True


# slide sample


def sample_img_with_slide_windows(img, sample_width, sample_height, save_path=None, img_name=None):
    """
        sample a image with fixed size slide windows to get several fixed
        sized small images
        :param img: the original img data matrix
        :param sample_width: the sample image's width
        :param sample_height: the sample image's height
        :param save_path: sample images' save_path
        :param img_name: image files' save name, without extension !
        :return if successful, return the sample images data saved in a 2D
        matrix, which each line represents a sample image with size
        (1, sample_width*sample_height, channels); else return False
    """
    isfirst = True
    if len(img.shape) == 3:
        ori_height, ori_width, chanels = img.shape
        if ori_height >= sample_height and ori_width >= sample_width:
            sample_per_row = ori_width - sample_width + 1
            sample_per_column = ori_height - sample_height + 1
            for ind_row in range(sample_per_column):
                for ind_col in range(sample_per_row):
                    if save_path is not None:
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path, img_name+'_'+str(int(
                            ind_row*sample_per_row+ind_col+1))+'.png'),
                                    img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width, :])
                    if isfirst:
                        expand_img_data = np.reshape(
                            img[ind_row:ind_row+sample_height,
                                ind_col:ind_col+sample_width,
                                :],
                            (1, sample_width*sample_height, chanels),
                            order='C')
                        isfirst = False
                    else:
                        expand_img_data = np.append(
                            expand_img_data,
                            np.reshape(img[ind_row:ind_row+sample_height,
                                       ind_col:ind_col+sample_width,
                                       :],
                                       (1, sample_width*sample_height, chanels),
                                       order='C'),
                            axis=0)
        return expand_img_data
    else:
        ori_height, ori_width = img.shape
        if ori_height >= sample_height and ori_width >= sample_width:
            sample_per_row = ori_width - sample_width + 1
            sample_per_column = ori_height - sample_height + 1
            for ind_row in range(sample_per_column):
                for ind_col in range(sample_per_row):
                    if save_path is not None:
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path, img_name+'_'+str(int(ind_row*sample_per_row+ind_col+1))
                                                 + '.png'),
                                    img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width])
                    if isfirst:
                        expand_img_data = np.reshape(
                            img[ind_row:ind_row + sample_height,
                                ind_col:ind_col + sample_width],
                            (1, sample_width * sample_height),
                            order='C')
                        isfirst = False
                    else:
                        expand_img_data = np.append(
                            expand_img_data,
                            np.reshape(img[ind_row:ind_row + sample_height,
                                       ind_col:ind_col + sample_width],
                                       (1, sample_width * sample_height),
                                       order='C'),
                            axis=0)
        return expand_img_data


def sample_with_slide_window_and_save_npy(source_path, slide_height, slide_width, source_extension='png'):
    """
    sample with slide window on images to get fix sized images, and save as .npy files
    :param source_path: source images' path
    :param slide_height: slide window's height
    :param slide_width: slide window's width
    :param source_extension: source images' file extension
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            is_save_npy = False
            pbar = tqdm(os.scandir(category.path))
            is_first = True
            total_dataset = []
            for img_files in pbar:
                if img_files.is_file():
                    extension = os.path.splitext(img_files.path)[1][1:]
                    filename_no_extension, _ = os.path.splitext(img_files.name)
                    if extension == source_extension:
                        is_save_npy = True
                        pbar.set_description("Processing %s" % img_files.name)
                        img_data = cv2.imread(img_files.path, -1)
                        if is_first:
                            total_dataset = sample_img_with_slide_windows(
                                img_data, slide_width, slide_height,
                                save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width),
                                                       category.name),
                                img_name=filename_no_extension)
                            is_first = False
                        else:
                            total_dataset = np.append(total_dataset, sample_img_with_slide_windows(
                                img_data, slide_width, slide_height,
                                save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width),
                                                       category.name),
                                img_name=filename_no_extension), axis=0)
            if is_save_npy:
                np.save(os.path.join(source_path, category.name+'.npy'), total_dataset)
    return True


# resize


def resize_img_and_save_to_folder_opensarship_slc(
    source_path, save_path, source_extension='tif', new_size=[88, 88]):
    """
    resize SLC mode OpenSARShip images to fixed size and save them
    the source path's struct : root/category_folders/image files
    :param source_path: source images' root ptah
    :param save_path: the path to save resized images
    :param source_extension: source image files' extension, default to 'tif'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                extension = os.path.splitext(img_files.path)[1][1:]
                if extension == source_extension:
                    pbar.set_description("Processing %s" % img_files.name)
                    tif = TIFF.open(img_files.path, mode='r')
                    source_img = tif.read_image()
                    # 1st channel is the real part for VH
                    # 2st channel is the imaginary part for VH
                    # to get the amplitude value of VH
                    img_amplitude = np.sqrt(
                        np.square(source_img[:, :, 0]) +
                        np.square(source_img[:, :, 1]))
                    image = misc.toimage(img_amplitude)
                    im_resize = misc.imresize(image, (new_size[0], new_size[
                        1]))
                    filename_no_extension, extension = os.path.splitext(
                        img_files.name)
                    os.makedirs(os.path.join(save_path, category.name),
                                exist_ok=True)
                    os.chdir(os.path.join(save_path, category.name))
                    misc.imsave(filename_no_extension+'.png', im_resize)
    return True


def resize_img_and_save_to_folder_with_gray_rgb(
    source_path, save_path, source_extension='png', new_size=[88, 88]):
    """
    resize images to fixed size and save them
    the source path's struct : root/image files
    :param source_path: source images' root ptah
    :param save_path: the path to save resized images
    :param source_extension: source image files' extension, default to 'png'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == source_extension:
                pbar.set_description("Processing %s" % img_files.name)
                source_img = cv2.imread(img_files.path, -1)
                if source_img.shape[0] < 20 or source_img.shape[1] < 20:
                    continue
                # 3 channels :RGB
                # gray = 0.39 * R + 0.5 * G + 0.11 * B
                img_amplitude = 0.39 * source_img[:, :, 0] + \
                                       0.5 * source_img[:, :, 1] + \
                                       0.11 * source_img[:, :, 2]
                image = misc.toimage(img_amplitude)
                im_resize = misc.imresize(image, (new_size[0], new_size[
                    1]))
                filename_no_extension, extension = os.path.splitext(
                    img_files.name)
                os.makedirs(save_path, exist_ok=True)
                os.chdir(save_path)
                misc.imsave(filename_no_extension+'.png', im_resize)
    return True


def resize_img_and_save_to_folder_with_category(
    source_path, save_path, source_extension='png', new_size=[88, 88]):
    """
    resize images to fixed size and save them
    the source path's struct : root/category/image files
    :param source_path: source images' root ptah
    :param save_path: the path to save resized images
    :param source_extension: source image files' extension, default to 'png'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                if img_files.is_file():
                    extension = os.path.splitext(img_files.path)[1][1:]
                    if extension == source_extension:
                        pbar.set_description("Processing %s" % img_files.name)
                        source_img = cv2.imread(img_files.path, -1)
                        image = misc.toimage(source_img)
                        im_resize = misc.imresize(image, (new_size[0], new_size[
                            1]))
                        filename_no_extension, extension = os.path.splitext(
                            img_files.name)
                        os.makedirs(os.path.join(save_path, category.name), exist_ok=True)
                        os.chdir(os.path.join(save_path, category.name))
                        misc.imsave(filename_no_extension+'.png', im_resize)
    return True


if __name__ == '__main__':
    # src_basepath = r'F:\dataset_se\DOTA_wuhanU\DOTA1.0_wuhanU\train\out_dataset'
    # src_basepath = r'F:\dataset_se\DOTA_wuhanU\DOTA1.0_wuhanU\val\out_dataset\out_dataset\gray'
    # src_category, dst_basepath = gen_destfolder(src_basepath)
    # img_gray(src_basepath, dst_basepath, src_category)
    # img_padding(src_basepath)
    # img_padding(r'F:\dataset_se', src_category=['mship'], out_size=[[128, 128]], ext='tiff')
    # cargo = np.load(r'I:\dataset_se\OpenSARShip\noexpand\GRDH_Cargo.npy')
    # tanker = np.load(r'I:\dataset_se\OpenSARShip\noexpand\GRDH_Tanker.npy')
    # cargo_128_128 = dataset_padding(cargo[:, :, 0], [28, 28])
    # tanker_128_128 = dataset_padding(tanker[:, :, 0], [28, 28])
    # np.save(r'I:\dataset_se\OpenSARShip\noexpand\GRDH_Cargo_128_128.npy', cargo_128_128)
    # np.save(r'I:\dataset_se\OpenSARShip\noexpand\GRDH_Tanker_128_128.npy', tanker_128_128)
    # source_path = r'/media/se/document/dataset_se/OpenSARShip/OpenSARShip_img_class_data/Patch/SLC'
    # save_path = r'/media/se/document/dataset_se/OpenSARShip/88_88_3class'
    # resize_img_and_save_to_folder_OpenSARShip_SLC(source_path, save_path)
    # source_path = r'/media/se/document/dataset_se/DOTA_wuhanU/DOTA1.0_wuhanU/train/out_dataset/ship'
    # save_path = r'/media/se/document/dataset_se/DOTA_wuhanU/DOTA1.0_wuhanU/train/out_dataset/ship_88_88'
    # resize_img_and_save_to_folder(source_path, save_path)
    # source_path = r'/media/se/document/dataset_se/DOTA_wuhanU/DOTA1.0_wuhanU/train/out_dataset/ship_88_88'
    # save_path = r'/media/se/document/dataset_se/DOTA_wuhanU/DOTA1.0_wuhanU/train/out_dataset'
    # npy_file_name = 'ship_88_88_DOTA.npy'
    # gen_npy_file(source_path, save_path, npy_file_name)
    # a = np.load(r'/media/se/document/dataset_se/DOTA_wuhanU/DOTA1.0_wuhanU/train/out_dataset/ship_88_88_DOTA.npy')
    source_path = r'/media/se/document/dataset_se/OpenSARShip/OpenSARShip_img_class_data/Patch/SLC'
    save_path = r'/media/se/document/dataset_se/OpenSARShip/OpenSARShip_img_class_data/Patch/SLC_amplitude'
    get_and_save_amplitude_image_slc(source_path, save_path)
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude'
    # save_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_crop'
    # crop_imgs_and_save_smaller(source_path, save_path, 88, 88)
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\temp'
    # save_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_force_padding_h94w90'
    # padding_to_fix_sized_and_save_imgs(source_path, save_path, padding_height=94, padding_width=90, source_extension='png')
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_force_padding_h94w90'
    # sample_with_slide_window_and_save_npy(source_path, slide_height=88, slide_width=88)
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\temp2'
    # save_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_force_padding_h94w90'
    # padding_to_fix_sized_and_save_imgs(source_path, save_path, padding_height=88, padding_width=88)
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\temp'
    # save_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_resize_h94w90'
    # resize_img_and_save_to_folder_with_category(source_path, save_path, source_extension='png', new_size=[94, 90])
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\temp2'
    # save_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_resize_h94w90'
    # resize_img_and_save_to_folder_with_category(source_path, save_path, source_extension='png', new_size=[88, 88])
    # source_path = r'F:\dataset_se\OpenSARShip\OpenSARShip_img_class_data\Patch\SLC_amplitude_resize_h94w90'
    # sample_with_slide_window_and_save_npy(source_path, slide_height=88, slide_width=88)
    pass


