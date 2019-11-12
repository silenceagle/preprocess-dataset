# preprocess img
# 20180916

import os
import cv2
import numpy as np
from tqdm import tqdm
from libtiff import TIFF
from scipy import misc
import shutil


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


def img_gray_with_category(src_basepath: str, dst_basepath, src_category='all', min_Width=1, min_Heigh=1, ext=['png']):
    """
        do graying on RGB images and save them
        :param src_basepath: source_path/category/image files
        :param src_category: list of the names of imgs' categories, which are also the sub folders' name of src_basepath
                             Eg. ['a', 'b'], default to 'all', which means all sub folders will be processed
        :param dst_basepath: gray imgs' storaged path
        :param min_Heigh: the minimal images' height that will be grayed
        :param min_Width: the maximal images' width that will be grayed
        :param ext: the extension of the images to be grayed
        :return True
    """
    if not os.path.exists(src_basepath):
        raise FileExistsError('path not found! : %s' % src_basepath)
    os.makedirs(dst_basepath, exist_ok=True)
    for this_cate in os.scandir(src_basepath):
        if this_cate.is_dir():
            if src_category == 'all' or this_cate.name in [src_category]:
                this_cate_srcpath = os.path.join(src_basepath, this_cate)
                this_cate_dstpath = os.path.join(dst_basepath, this_cate.name+'_gray')
                os.makedirs(this_cate_dstpath, exist_ok=True)
                pbar = tqdm(os.scandir(this_cate_srcpath))
                for entry in pbar:
                    pbar.set_description("Processing %s" % entry.path)
                    if entry.is_file():
                        # get the file's extension
                        extension = os.path.splitext(entry.path)[1][1:]
                        if extension in ext:
                            # print("find png file: %s" % entry.name)
                            # gray img
                            tmp_img = cv2.imread(entry.path)
                            # print(entry.path)
                            if len(tmp_img.shape) == 3:
                                img_h, img_w, img_c = tmp_img.shape
                                if img_h>=min_Heigh and img_w>=min_Width:
                                    if img_c==3: # RGB图像
                                        tmp_gray_img = np.zeros([img_h, img_w], dtype=np.float32)
                                        for ind_h in range(img_h):
                                            for ind_w in range(img_w):
                                                tmp_gray_img[ind_h][ind_w] = np.int(0.39 * tmp_img[ind_h][ind_w][0] +
                                                                                    0.5 * tmp_img[ind_h][ind_w][1] +
                                                                                    0.11 * tmp_img[ind_h][ind_w][2])
                                                # gray = 0.39 * R + 0.5 * G + 0.11 * B
                                        # save gray img
                                        # print("\tsaving gray img: %s ..." % entry.name)
                                        cv2.imwrite(os.path.join(this_cate_dstpath, entry.name), tmp_gray_img)
    return True


def img_gray(source_path: str, save_path, extensions=['png']):
    """
        do graying on RGB image and save them
        :param source_path: folder's path that stores original RGB images files, struction source_path/images
        :param save_path: gray images save path
        :param extensions: the extension of the images to be grayed
        :return True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_file in pbar:
        if img_file.is_file():
            extension = os.path.splitext(img_file.path)[1][1:]
            if extension in extensions:
                pbar.set_description("Processing %s" % img_file.name)
                # gray img
                tmp_img = cv2.imread(img_file.path, -1)
                # print(entry.path)
                if len(tmp_img.shape) == 3:
                    img_h, img_w, img_c = tmp_img.shape
                    if img_c == 3:
                        tmp_gray_img = np.zeros([img_h, img_w], dtype=np.float32)
                        for ind_h in range(img_h):
                            for ind_w in range(img_w):
                                tmp_gray_img[ind_h][ind_w] = np.int(0.39 * tmp_img[ind_h][ind_w][0] +
                                                                    0.5 * tmp_img[ind_h][ind_w][1] +
                                                                    0.11 * tmp_img[ind_h][ind_w][2])
                                # gray = 0.39 * R + 0.5 * G + 0.11 * B
                        # save gray img
                        # print("\tsaving gray img: %s ..." % entry.name)
                        cv2.imwrite(os.path.join(save_path, img_file.name), tmp_gray_img)
                        is_successful = True
    return True


def get_and_save_amplitude_image_slc_with_category(source_path, save_path, source_extension='tif'):
    """
    get the amplitude images of OpenSARShip SLC mode images and save .png images and the .npy files
    :param source_path: source images' root ptah, source_path/category/image files
    :param save_path: the path to save processed images
    :param source_extension: source image files' extension, default to 'tif'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            os.makedirs(os.path.join(save_path, 'VH', category.name), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'VV', category.name), exist_ok=True)
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                extension = os.path.splitext(img_files.path)[1][1:]
                if extension == source_extension:
                    pbar.set_description("Processing %s" % img_files.name)
                    tif = TIFF.open(img_files.path, mode='r')
                    source_img = tif.read_image()
                    if source_img.shape[-1] != 4:
                        print (f"{img_files.path} has {source_img.shape[-1]} channels ! (not 4)")
                        continue
                        # raise ValueError(f"{img_files.path} has {source_img.shape[-1]} channels ! (not 4)")
                    # 1st channel is the real part for VH
                    # 2st channel is the imaginary part for VH
                    # 3st channel is the real part for VV
                    # 4st channel is the imaginary part for VV
                    # to get the amplitude value of VH and VV
                    # VH
                    img_amplitude_VH = np.sqrt(np.square(source_img[:, :, 0]) + np.square(source_img[:, :, 1]))
                    img_amplitude_VH = np.reshape(img_amplitude_VH, [img_amplitude_VH.shape[0],
                                                                     img_amplitude_VH.shape[1]])
                    filename_no_extension, _ = os.path.splitext(img_files.name)
                    cv2.imwrite(os.path.join(save_path, 'VH', category.name, filename_no_extension+'_VH.png'),
                                img_amplitude_VH)
                    np.save(os.path.join(save_path, 'VH', category.name, filename_no_extension+'_VH.npy'),
                            img_amplitude_VH)
                    # VV
                    img_amplitude_VV = np.sqrt(np.square(source_img[:, :, 2]) + np.square(source_img[:, :, 3]))
                    img_amplitude_VV = np.reshape(img_amplitude_VV, [img_amplitude_VV.shape[0],
                                                                     img_amplitude_VV.shape[1]])
                    cv2.imwrite(os.path.join(save_path, 'VV', category.name, filename_no_extension + '_VV.png'),
                                img_amplitude_VV)
                    np.save(os.path.join(save_path, 'VV', category.name, filename_no_extension + '_VV.npy'),
                            img_amplitude_VV)
    return True


def get_and_save_amplitude_image_grd_with_category(source_path, save_path, source_extension='tif'):
    """
    get the amplitude images of OpenSARShip GRD mode images and save .png images and the .npy files
    :param source_path: source images' root ptah, source_path/category/image files
    :param save_path: the path to save processed images
    :param source_extension: source image files' extension, default to 'tif'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            os.makedirs(os.path.join(save_path, 'VH', category.name), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'VV', category.name), exist_ok=True)
            pbar = tqdm(os.scandir(category.path))
            for img_files in pbar:
                extension = os.path.splitext(img_files.path)[1][1:]
                if extension == source_extension:
                    pbar.set_description("Processing %s" % img_files.name)
                    tif = TIFF.open(img_files.path, mode='r')
                    source_img = tif.read_image()
                    if source_img.shape[-1] != 2:
                        print (f"{img_files.path} has {source_img.shape[-1]} channels ! (not 2)")
                        continue
                        # raise ValueError(f"{img_files.path} has {source_img.shape[-1]} channels ! (not 4)")
                    # 1st channel is the amplitude value for VH
                    # 2st channel is the amplitude value for VV
                    # VH
                    img_amplitude_VH = np.reshape(source_img[:, :, 0], [source_img.shape[0],
                                                                        source_img.shape[1]])
                    filename_no_extension, _ = os.path.splitext(img_files.name)
                    cv2.imwrite(os.path.join(save_path, 'VH', category.name, filename_no_extension+'_VH.png'),
                                img_amplitude_VH)
                    np.save(os.path.join(save_path, 'VH', category.name, filename_no_extension+'_VH.npy'),
                            img_amplitude_VH)
                    # VV
                    img_amplitude_VV = np.reshape(source_img[:, :, 1], [source_img.shape[0],
                                                                        source_img.shape[1]])
                    cv2.imwrite(os.path.join(save_path, 'VV', category.name, filename_no_extension + '_VV.png'),
                                img_amplitude_VV)
                    np.save(os.path.join(save_path, 'VV', category.name, filename_no_extension + '_VV.npy'),
                            img_amplitude_VV)
    return True


def gen_npy_file(source_path, save_path, npy_file_name, img_extension='png'):
    """
    concat images data with only one channel in soruce path and generate npy file
    :param source_path: source_path/image files
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
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
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

def __img_crop_from_center(original_img, crop_width, crop_height):
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
                        crop_img_data = __img_crop_from_center(img_data, crop_width, crop_height)
                        cv2.imwrite(os.path.join(save_path, 'h'+str(crop_height)+'w'+str(crop_width), category.name,
                                                 filename_no_extension+'.png'), crop_img_data)
                    else:
                        cv2.imwrite(
                            os.path.join(save_path, 'smaller_than_crop_window',
                                         'h'+str(crop_height)+'w'+str(crop_width), category.name,
                                         filename_no_extension + '.png'), img_data)
    return True

# padding


# def img_padding_with_category(src_basepath: str,  src_category=None, dst_basepath=None, out_size=[[64, 64],[128, 128]],
#                               ext=['png'], is_save_img = False):
#     """
#         padding zero to src imgs to get fix sized imgs, and save as npy files
#         :param src_basepath: folder's path that stores original images files, src_basepath/category/image files
#         :param src_category: the names of imgs' categories, which are also the sub folders' name of src_basepath
#         :param dst_basepath: processed imgs' storaged path, default to src_basepath/out_dataset
#         :param out_size: every row represent one of the images' out size :[height, width],
#                          and the row's index bigger, the size bigger.
#         :param ext: the extension of the images to be processed
#         :param is_save_img: whether to save processed images, default to False
#         :return
#     """
#     # if source images' stored folder not exist, raise error
#     if not os.path.exists(src_basepath):
#         raise FileExistsError('path not found! : %s' % src_basepath)
#     # if the source images' category information is not provided, then generate it.
#     if src_category is None:
#         src_category = []
#         for entry in os.scandir(src_basepath):
#             if entry.is_dir() and entry.name != 'out_dataset':
#                 src_category.append(entry.name)
#     # if the after-processed images' stored folder is not provided, then set it to src_basepath\out_dataset
#     if dst_basepath is None:
#         dst_basepath = os.path.join(src_basepath, 'out_dataset')
#     # generate folder to save images with size of outsize1 or outsize2
#     num_outsize = np.array(out_size).shape[0]
#     path_size = []
#     for ind_size in range(num_outsize):
#         path_size.append(os.path.join(dst_basepath, ('%d_%d' % (out_size[ind_size][0], out_size[ind_size][1]))))
#         os.makedirs(path_size[ind_size], exist_ok=True)  # if exist, don't raise exception
#     # do padding
#     for this_cate in src_category:
#         this_cate_srcpath = os.path.join(src_basepath, this_cate)
#         pbar = tqdm(os.scandir(this_cate_srcpath))
#         for entry in pbar:
#             pbar.set_description("Processing %s" % entry.name)
#             if entry.is_file():
#                 # get the file's extension
#                 extension = os.path.splitext(entry.path)[1][1:]
#                 if extension in ext:
#                     # print('find image : %s' % entry.path)
#                     # if extension == 'tiff':
#                     #     tif = TIFF.open(entry.path, mode='r')
#                     #     tmp_img = tif.read_image()
#                     #     tif.close
#                     # else:
#                     #     tmp_img = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
#                     tmp_img = cv2.imread(entry.path, -1)  # <0 returns the image as is
#                     # print(tmp_img[85][108])
#                     # print("image's shape:")
#                     # print(tmp_img.shape)
#                     # print(len(tmp_img.shape))
#                     # img_h, img_w = tmp_img.shape
#                     # this_size = 0
#                     # for ind_size in range(num_outsize):
#                     #     if img_h <= out_size[ind_size][0] and img_w <= out_size[ind_size][1]:
#                     #         break
#                     #     this_size += 1
#                     # # print('out image size index : %d' % this_size)
#                     # if this_size < num_outsize:  # image with valid size
#                     #     # print('gray image')
#                     #     save_img = np.zeros([out_size[this_size][0], out_size[this_size][1]], dtype=np.uint16)
#                     #     start_r = int(np.floor((out_size[this_size][0]-img_h)/2.0))
#                     #     start_c = int(np.floor((out_size[this_size][1]-img_w)/2.0))
#                     #     for ind_r in range(img_h):
#                     #         for ind_c in range(img_w):
#                     #             save_img[start_r+ind_r, start_c+ind_c] = tmp_img[ind_r, ind_c]
#                     is_valid, save_img, this_size = __matrix_padding(tmp_img, out_size=out_size, dtype=np.uint16)
#                     if is_valid:
#                         if is_save_img:
#                             os.makedirs(os.path.join(path_size[this_size], this_cate), exist_ok=True)
#                             cv2.imwrite(os.path.join(path_size[this_size], this_cate, entry.name), save_img)
#     return


def padding_images_with_zero(
        source_path, save_path, out_size=[[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        image_extension='png', is_save_npy=False):
    """
    padding images with zeros to get required size images, the final image's size is determined by it original size.
    Eg. when the out_size is [[32, 32], [64, 64]]
        if the original size is [h<=32, w<=32], then final size is [32, 32]
                        [h<=64, 32<w<=64] or [32<h<=64, w<=64]   -->  [64, 64]
                            [h>64, w] or [h, w>64]               -->  will not be processed sand saved
    :param source_path: the original images' path, source path/image files
    :param save_path: the output images' save path
    :param out_size: the output image size, Eg. [[32, 32], [64, 64], ...]
    :param image_extension: image file's extension, default to 'png', also support 'npy'
    :param is_save_npy: bool. if to save npy files, default to false
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == image_extension:
                pbar.set_description("Processing %s" % img_files.path)
                if extension == 'npy':
                    img_data = np.load(img_files.path)
                else:
                    img_data = cv2.imread(img_files.path, -1)
                is_valid, out_img, _ = __matrix_padding_multi_size_soft(img_data, out_size=out_size)
                if is_valid:
                    cv2.imwrite(os.path.join(save_path, img_files.name.split('.')[0]+'.png'), out_img)
                    if is_save_npy:
                        np.save(os.path.join(save_path, img_files.name.split('.')[0]+'.npy'), out_img)


def padding_images_with_zero_with_category(
        source_path, save_path, out_size=[[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        image_extension='png', is_save_npy=False):
    """
    padding images with zeros to get required size images, the final image's size is determined by it original size.
    Eg. when the out_size is [[32, 32], [64, 64]]
        if the original size is [h<=32, w<=32], then final size is [32, 32]
                        [h<=64, 32<w<=64] or [32<h<=64, w<=64]   -->  [64, 64]
                            [h>64, w] or [h, w>64]               -->  will not be processed sand saved
    :param source_path: the original images' path, source path/category/image files
    :param save_path: the output images' save path
    :param out_size: the output image size, Eg. [[32, 32], [64, 64], ...]
    :param image_extension: image file's extension, default to 'png', also support 'npy'
    :param is_save_npy: bool. if to save npy files, default to false
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            padding_images_with_zero(category.path, os.path.join(save_path, category.name), out_size, image_extension,
                                     is_save_npy)

    return True


def padding_images_with_zero_to_square_size(source_path, save_path, image_extension='png'):
    """
    padding images with zeros to get squared images, the output image's size is determined by it's larger side:
    [out_h, out_w] = [max(w, h), max(w, h)]
    :param source_path: the original images' path, source path/image files
    :param save_path: the output images' save path
    :param image_extension: image file's extension, default to 'png', also support 'npy'
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == image_extension:
                pbar.set_description("Processing %s" % img_files.path)
                if extension == 'npy':
                    img_data = np.load(img_files.path)
                else:
                    img_data = cv2.imread(img_files.path, -1)
                is_valid, out_img = __matrix_padding(img_data, out_size=[np.max(img_data.shape),
                                                                         np.max(img_data.shape)])  # suppose c<= w or h
                if is_valid:
                    cv2.imwrite(os.path.join(save_path, img_files.name), out_img)


def __matrix_padding_multi_size_soft(src_mat, out_size=[[64, 64], [128, 128]], dtype=np.float32):
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
    img_c = None
    if len(src_mat.shape) == 2:
        img_h, img_w = src_mat.shape
    else:
        img_h, img_w, img_c = src_mat.shape
    this_size = 0
    for ind_size in range(num_outsize):
        if img_h <= out_size[ind_size][0] and img_w <= out_size[ind_size][1]:
            break
        this_size += 1
    # print('out image size index : %d' % this_size)

    is_valid = False
    if this_size < num_outsize:  # image with valid size
        if img_c is None:
            out_mat = np.zeros([out_size[this_size][0], out_size[this_size][1]], dtype=dtype)
        else:
            out_mat = np.zeros([out_size[this_size][0], out_size[this_size][1], img_c], dtype=dtype)
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


def __matrix_padding_force_to_get_fix_sized_matrix(src_mat, padding_height, padding_width):
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


def __matrix_padding(src_mat, out_size=[64, 64], dtype=np.float32):
    """
        padding zero to src matrix to get larger size matrix
        if and only if source image's width and height are both not bigger than out's, do padding
        :param src_mat: source matrix
        :param out_size: list of int:[height, width]
        :param dtype: the src matrix's date type, default to np.float32
        :return is_valid: if src_mat's size is smaller than out's , is_valid set to be Ture, else Flase
        :return out_mat: if is_valid is true, then out_mat be useful, else will be useless.
    """
    img_c = None
    if len(src_mat.shape) == 2:
        img_h, img_w = src_mat.shape
    else:
        img_h, img_w, img_c = src_mat.shape
    is_valid = False
    if img_h <= out_size[0] and img_w <= out_size[1]:  # image with valid size
        if img_c is None:
            out_mat = np.zeros([out_size[0], out_size[1]], dtype=dtype)
        else:
            out_mat = np.zeros([out_size[0], out_size[1], img_c], dtype=dtype)
        # print('gray image')
        start_r = int(np.floor((out_size[0] - img_h) / 2.0))
        start_c = int(np.floor((out_size[1] - img_w) / 2.0))
        for ind_r in range(img_h):
            for ind_c in range(img_w):
                out_mat[start_r + ind_r, start_c + ind_c] = src_mat[ind_r, ind_c]
        is_valid = True
        return is_valid, out_mat
    else:
        return is_valid, 0


def __dataset_padding(src_mat, ori_size, out_size=[[128, 128]], dtype=np.float32):
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
        _, tmp_mat, _ = __matrix_padding_multi_size_soft(np.reshape(src_mat[ind_mat], (ori_size[0], ori_size[1])),
                                    out_size=out_size, dtype=dtype)
        out_mat[ind_mat] = np.reshape(tmp_mat, out_size[0][0]*out_size[0][1])
    return out_mat


def padding_to_fix_sized_and_save_imgs_with_category(source_path, save_path, padding_height, padding_width,
                                                     source_extension='png'):
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
                    padding_img = __matrix_padding_force_to_get_fix_sized_matrix(img_data, padding_height, padding_width)
                    cv2.imwrite(os.path.join(save_path, category.name, filename_no_extension+'.png'), padding_img)
    return True


# slide sample


# def __sample_img_with_slide_windows(img, sample_width, sample_height, save_path=None, img_name=None):
#     """
#         sample a image with fixed size slide windows to get several fixed
#         sized small images
#         :param img: the original img data matrix
#         :param sample_width: the sample image's width
#         :param sample_height: the sample image's height
#         :param save_path: sample images' save_path
#         :param img_name: image files' save name, without extension !
#         :return if successful, return the sample images data saved in a 2D
#         matrix, which each line represents a sample image with size
#         (1, sample_width*sample_height, channels); else return False
#     """
#     isfirst = True
#     if len(img.shape) == 3:
#         ori_height, ori_width, chanels = img.shape
#         if ori_height >= sample_height and ori_width >= sample_width:
#             sample_per_row = ori_width - sample_width + 1
#             sample_per_column = ori_height - sample_height + 1
#             for ind_row in range(sample_per_column):
#                 for ind_col in range(sample_per_row):
#                     if save_path is not None:
#                         os.makedirs(save_path, exist_ok=True)
#                         cv2.imwrite(os.path.join(save_path, img_name+'_'+str(int(
#                             ind_row*sample_per_row+ind_col+1))+'.png'),
#                                     img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width, :])
#                     if isfirst:
#                         expand_img_data = np.reshape(
#                             img[ind_row:ind_row+sample_height,
#                                 ind_col:ind_col+sample_width,
#                                 :],
#                             (1, sample_width*sample_height, chanels),
#                             order='C')
#                         isfirst = False
#                     else:
#                         expand_img_data = np.append(
#                             expand_img_data,
#                             np.reshape(img[ind_row:ind_row+sample_height,
#                                        ind_col:ind_col+sample_width,
#                                        :],
#                                        (1, sample_width*sample_height, chanels),
#                                        order='C'),
#                             axis=0)
#         return expand_img_data
#     else:
#         ori_height, ori_width = img.shape
#         if ori_height >= sample_height and ori_width >= sample_width:
#             sample_per_row = ori_width - sample_width + 1
#             sample_per_column = ori_height - sample_height + 1
#             for ind_row in range(sample_per_column):
#                 for ind_col in range(sample_per_row):
#                     if save_path is not None:
#                         os.makedirs(save_path, exist_ok=True)
#                         cv2.imwrite(os.path.join(save_path, img_name+'_'+str(int(ind_row*sample_per_row+ind_col+1))
#                                                  + '.png'),
#                                     img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width])
#                         np.save(os.path.join(save_path, img_name+'_'+str(int(ind_row*sample_per_row+ind_col+1))
#                                                  + '.npy'),
#                                     img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width])
#                     if isfirst:
#                         expand_img_data = np.reshape(
#                             img[ind_row:ind_row + sample_height,
#                                 ind_col:ind_col + sample_width],
#                             (1, sample_width * sample_height),
#                             order='C')
#                         isfirst = False
#                     else:
#                         expand_img_data = np.append(
#                             expand_img_data,
#                             np.reshape(img[ind_row:ind_row + sample_height,
#                                        ind_col:ind_col + sample_width],
#                                        (1, sample_width * sample_height),
#                                        order='C'),
#                             axis=0)
#         return expand_img_data


def __sample_img_with_slide_windows(img, sample_width, sample_height, save_path=None, img_name=None, stride=[1, 1],
                                    is_save_npy=False):
    """
        sample a image with fixed size slide windows to get several fixed
        sized small images
        :param img: the original img data matrix
        :param sample_width: the sample image's width
        :param sample_height: the sample image's height
        :param save_path: sample images' save_path, default to None, which means not to save sample image files
        :param img_name: image files' save name, without extension !
        :param stride: stride, a 2 length list, [h_stride, w_stride]
        :param is_save_npy: if to save npy files, default to False
        :return if successful, return the sample images data saved in a 2D
        matrix, which each line represents a sample image with size
        (1, sample_width*sample_height, channels); else return False
    """
    isfirst = True
    h_stride = np.int(stride[0])
    w_stride = np.int(stride[1])
    if len(img.shape) == 3:
        ori_height, ori_width, chanels = img.shape
        if ori_height >= sample_height and ori_width >= sample_width:
            sample_per_row = np.int(np.ceil((ori_width - sample_width + 1) / w_stride))
            sample_per_column = np.int(np.ceil((ori_height - sample_height + 1) / h_stride))
            for _ind_row in range(sample_per_column):
                ind_row = _ind_row * h_stride
                print('processed %.4f %%' % ((_ind_row*1.0+1.0)/sample_per_column*100))
                for _ind_col in range(sample_per_row):
                    ind_col = _ind_col * w_stride
                    if save_path is not None:
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path, img_name+'_'+str(int(
                            ind_row*sample_per_row+ind_col+1))+'.png'),
                                    img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width, :])
                        if is_save_npy:
                            np.save(os.path.join(save_path, img_name+'_'+str(int(ind_row*sample_per_row+ind_col+1))
                                                 + '.npy'),
                                    img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width])
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
            sample_per_row = np.int(np.ceil((ori_width - sample_width + 1) / w_stride))
            sample_per_column = np.int(np.ceil((ori_height - sample_height + 1) / h_stride))
            for ind_row in range(sample_per_column):
                ind_row = ind_row * h_stride
                for ind_col in range(sample_per_row):
                    ind_col = ind_col * w_stride
                    if save_path is not None:
                        os.makedirs(save_path, exist_ok=True)
                        cv2.imwrite(os.path.join(save_path, img_name+'_'+str(int(ind_row*sample_per_row+ind_col+1))
                                                 + '.png'),
                                    img[ind_row:ind_row + sample_height, ind_col:ind_col + sample_width])
                        if is_save_npy:
                            np.save(os.path.join(save_path, img_name+'_'+str(int(ind_row*sample_per_row+ind_col+1))
                                                 + '.npy'),
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


def sample_img_with_slide_window_and_save_npy_with_category(source_path, slide_height, slide_width, source_extension='png',
                                                        stride=[1, 1]):
    """
    sample with slide window on images to get fix sized images, and save as .npy files
        out images will be saved in source_path/h**w**/category/images,
        npy file will be saved in source_path/category/npyfile
    :param source_path: source images' path, source path/category/files
    :param slide_height: slide window's height
    :param slide_width: slide window's width
    :param source_extension: source images' file extension
    :param stride: stride, a 2 length list, [h_stride, w_stride]
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
                        if extension == 'npy':
                            img_data = np.load(img_files.path)
                        else:
                            img_data = cv2.imread(img_files.path, -1)
                        if is_first:
                            total_dataset = __sample_img_with_slide_windows(
                                img_data, slide_width, slide_height,
                                save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width),
                                                       category.name),
                                img_name=filename_no_extension, stride=stride)
                            is_first = False
                        else:
                            total_dataset = np.append(total_dataset, __sample_img_with_slide_windows(
                                img_data, slide_width, slide_height,
                                save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width),
                                                       category.name),
                                img_name=filename_no_extension, stride=stride), axis=0)
            if is_save_npy:
                np.save(os.path.join(source_path, category.name+'.npy'), total_dataset)
    return True


# def sample_with_slide_window_and_save_npy(source_path, slide_height, slide_width, npy_save_name, source_extension='png'):
#     """
#     sample with slide window on images to get fix sized images, and save as .npy files
#     :param source_path: source images' path, source path/files
#     :param slide_height: slide window's height
#     :param slide_width: slide window's width
#     :param npy_save_name: the npy file's save name
#     :param source_extension: source images' file extension
#     :return: True
#     """
#     if not os.path.exists(source_path):
#         raise FileExistsError('path not found! : %s' % source_path)
#     pbar = tqdm(os.scandir(source_path))
#     is_first = True
#     for img_files in pbar:
#         if img_files.is_file():
#             extension = os.path.splitext(img_files.path)[1][1:]
#             filename_no_extension, _ = os.path.splitext(img_files.name)
#             if extension == source_extension:
#                 pbar.set_description("Processing %s" % img_files.name)
#                 if extension == 'npy':
#                     img_data = np.load(img_files.path)
#                 else:
#                     img_data = cv2.imread(img_files.path, -1)
#                 if is_first:
#                     total_dataset = __sample_img_with_slide_windows(
#                         img_data, slide_width, slide_height,
#                         save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width)),
#                         img_name=filename_no_extension)
#                     is_first = False
#                 else:
#                     total_dataset = np.append(total_dataset, __sample_img_with_slide_windows(
#                         img_data, slide_width, slide_height,
#                         save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width)),
#                         img_name=filename_no_extension), axis=0)
#     if npy_save_name is not None:
#         np.save(os.path.join(source_path, npy_save_name), total_dataset)
#     return True


def sample_with_slide_window_and_save_npy_stride(source_path, slide_height, slide_width, npy_save_name,
                                                 source_extension='png', stride=[1, 1]):
    """
    sample with slide window on images to get fix sized images, and save as .npy files
    :param source_path: source images' path, source path/files
    :param slide_height: slide window's height
    :param slide_width: slide window's width
    :param npy_save_name: the npy file's save name
    :param source_extension: source images' file extension
    :param stride: stride, a 2 length list, [h_stride, w_stride]
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    pbar = tqdm(os.scandir(source_path))
    is_first = True
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            filename_no_extension, _ = os.path.splitext(img_files.name)
            if extension == source_extension:
                pbar.set_description("Processing %s" % img_files.name)
                if extension == 'npy':
                    img_data = np.load(img_files.path)
                else:
                    img_data = cv2.imread(img_files.path, -1)
                if is_first:
                    total_dataset = __sample_img_with_slide_windows(
                        img_data, slide_width, slide_height,
                        save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width)),
                        img_name=filename_no_extension, stride=stride)
                    is_first = False
                else:
                    total_dataset = np.append(total_dataset, __sample_img_with_slide_windows(
                        img_data, slide_width, slide_height,
                        save_path=os.path.join(source_path, 'h'+str(slide_height)+'w'+str(slide_width)),
                        img_name=filename_no_extension, stride=stride), axis=0)
    if npy_save_name is not None:
        np.save(os.path.join(source_path, npy_save_name), total_dataset)
    return True


# resize


def resize_img_and_save_to_folder_opensarship_slc_with_category(
    source_path, save_path, source_extension='tif', new_size=[88, 88], is_save_npy=False, is_save_img=True):
    """
    resize SLC mode OpenSARShip images to fixed size and save them
    the source path's struct : root/category_folders/image files
    :param source_path: source images' root ptah
    :param save_path: the path to save resized images
    :param source_extension: source image files' extension, default to 'tif'
    :param is_save_npy : if to save npy files whose name is same to original image file, npy files contain float values
    :param is_save_img: if to save image files, default to True
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
                    # image = misc.toimage(img_amplitude)
                    # im_resize = misc.imresize(image, (new_size[0], new_size[
                    #     1]))
                    im_resize = cv2.resize(img_amplitude, (new_size[1], new_size[0]))
                    filename_no_extension, extension = os.path.splitext(
                        img_files.name)
                    os.makedirs(os.path.join(save_path, category.name),
                                exist_ok=True)
                    os.chdir(os.path.join(save_path, category.name))
                    # misc.imsave(filename_no_extension+'.png', im_resize)
                    if is_save_img:
                        cv2.imwrite(filename_no_extension + '.png', im_resize)
                    if is_save_npy:
                        np.save(filename_no_extension + '.npy', im_resize)

    return True


def resize_img_and_save_to_folder(
    source_path, save_path, source_extension='png', new_size=[88, 88], is_do_gray=False, img_smallest=1,
        is_save_npy=False, is_save_img=True):
    """
    resize images to fixed size and save them
    the source path's struct : root/image files
    and save npy files contain float values
    :param source_path: source images' root ptah
    :param save_path: the path to save resized images
    :param source_extension: source image files' extension, default to 'png', also support 'npy
    :param is_do_gray: if to do gray on RGB image, default to False
    :param img_smallest: the image's smallest size
    :param is_save_npy : if to save npy files whose name is same to original image file, npy files contain float values
    :param is_save_img: if to save image files, default to True
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
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
                    source_img = cv2.imread(img_files.path, -1)
                if source_img.shape[0] < img_smallest or source_img.shape[1] < img_smallest:
                    continue
                if len(source_img.shape) == 3:
                    # 3 channels :RGB
                    if is_do_gray:
                        # gray = 0.39 * R + 0.5 * G + 0.11 * B
                        img_gray = 0.39 * source_img[:, :, 0] + 0.5 * source_img[:, :, 1] + 0.11 * source_img[:, :, 2]
                        # image = misc.toimage(img_gray)
                        # im_resize = misc.imresize(image, (new_size[0], new_size[1]))
                        im_resize = cv2.resize(img_gray, (new_size[1], new_size[0]))
                    else:
                        # image = misc.toimage(source_img)
                        # im_resize = misc.imresize(image, (new_size[0], new_size[1], source_img.shape[2]))
                        im_resize = cv2.resize(np.float32(source_img), (new_size[1], new_size[0]))
                elif len(source_img.shape) == 2:
                    # image = misc.toimage(source_img)
                    # im_resize = misc.imresize(image, (new_size[0], new_size[1]))
                    im_resize = cv2.resize(np.float32(source_img), (new_size[1], new_size[0]))
                filename_no_extension, extension = os.path.splitext(img_files.name)
                os.makedirs(save_path, exist_ok=True)
                os.chdir(save_path)
                # misc.imsave(filename_no_extension+'.png', im_resize)
                if is_save_img:
                    cv2.imwrite(filename_no_extension + '.png', im_resize)
                if is_save_npy:
                    np.save(filename_no_extension + '.npy', im_resize)
    return True


def resize_img_and_save_to_folder_with_category(
    source_path, save_path, source_extension='png', new_size=[88, 88], is_do_gray=False, img_smallest=1,
    is_save_npy=False, is_save_img=True):
    """
    resize images to fixed size and save them
    the source path's struct : root/category/image files
    also save the npy files contain float value
    :param source_path: source images' root ptah
    :param save_path: the path to save resized images, save path/category/images
    :param source_extension: source image files' extension, default to 'png', also support 'npy'
    :param new_size: image's new size: [height, width]
    :param is_do_gray: if to do gray on image, default to False
    :param img_smallest: the smallest size of image which is to be processed
    :param is_save_npy : if to save npy files whose name is same to original image file, npy files contain float values
    :param is_save_img: if to save image files, default to True
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
                        if extension == 'npy':
                            source_img = np.load(img_files.path)
                        else:
                            source_img = cv2.imread(img_files.path, -1)
                        if source_img.shape[0] < img_smallest or source_img.shape[1] < img_smallest:
                            continue
                        if len(source_img.shape) == 3:
                            # 3 channels :RGB
                            if is_do_gray:
                                # gray = 0.39 * R + 0.5 * G + 0.11 * B
                                img_gray = 0.39 * source_img[:, :, 0] + 0.5 * source_img[:, :, 1] + 0.11 * source_img[:,
                                                                                                           :, 2]
                                # image = misc.toimage(img_gray)
                                # im_resize = misc.imresize(image, (new_size[0], new_size[1]))
                                im_resize = cv2.resize(img_gray, (new_size[1], new_size[0]))
                            else:
                                # image = misc.toimage(source_img)
                                # im_resize = misc.imresize(image, (new_size[0], new_size[1], source_img.shape[2]))
                                im_resize = cv2.resize(np.float32(source_img), (new_size[1], new_size[0]))
                        elif len(source_img.shape) == 2:
                            # image = misc.toimage(source_img)
                            # im_resize = misc.imresize(image, (new_size[0], new_size[1]))
                            im_resize = cv2.resize(np.float32(source_img), (new_size[1], new_size[0]))
                        filename_no_extension, extension = os.path.splitext(img_files.name)
                        os.makedirs(os.path.join(save_path, category.name), exist_ok=True)
                        os.chdir(os.path.join(save_path, category.name))
                        # misc.imsave(filename_no_extension+'.png', im_resize)
                        if is_save_img:
                            cv2.imwrite(filename_no_extension + '.png', im_resize)
                        if is_save_npy:
                            np.save(filename_no_extension + '.npy', im_resize)
    return True


# rotate


def __rotate_img_90_degree(img):
    """
    rotate image 90 degree clockwise
    :param img: image data
    :return: rotated image data
    """
    img_width = None
    if len(img.shape) == 3:
        img_height, img_width, img_channels = img.shape
        rotated_img = np.zeros([img_width, img_height, img_channels], dtype=np.float32)
    elif len(img.shape) == 2:
        img_height, img_width = img.shape
        rotated_img = np.zeros([img_width, img_height], dtype=np.float32)
    if img_width is None:
        raise ValueError('''input image's shape is not valid !''')
    for ind_row in range(img_height):
        rotated_img[:, img_height - ind_row - 1] = img[ind_row]
    return rotated_img


def rotate_img_90degree_and_save_to_folder(source_path, save_path, source_extension='png',  save_extension='png',
                                           is_save_npy=False, is_save_img=True):
    """
    rotate image 90 degree clockwise and save to folder
    :param source_path: source_path/image files
    :param save_path: save_path/rotated image files
    :param source_extension: image files' extension, default to 'png'
    :param save_extension: rotated image's save extension, default to 'png'
    :param is_save_npy: if to save npy files, default to False
    :param is_save_img: if to save image files, default to True.
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == source_extension:
                pbar.set_description("Processing %s" % img_files.name)
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
                    source_img = cv2.imread(img_files.path, -1)
                rotated_img = __rotate_img_90_degree(source_img)
                filename_no_extension, extension = os.path.splitext(img_files.name)
                os.chdir(save_path)
                if is_save_img:
                    cv2.imwrite(filename_no_extension + '_rotate90c.' + save_extension, rotated_img)
                if is_save_npy:
                    np.save(filename_no_extension + '_rotate90c.npy', rotated_img)
    return True


def rotate_img_90degree_and_save_to_folder_with_category(
        source_path, save_path, source_extension='png',  save_extension='png', is_save_npy=False, is_save_img=True):
    """
        rotate image 90 degree clockwise and save to folder
        :param source_path: source_path/category/image files
        :param save_path: save_path/rotated image files
        :param source_extension: image files' extension, default to 'png'
        :param save_extension: rotated image's save extension, default to 'png'
        :param is_save_npy: if to save npy files, default to False
        :param is_save_img: if to save image files, default to True.
        :return: True
        """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            rotate_img_90degree_and_save_to_folder(
                category.path, os.path.join(save_path, category.name), source_extension, save_extension,
                is_save_npy=is_save_npy, is_save_img=is_save_img)
    return True


def rotate_img_180degree_and_save_to_folder(source_path, save_path, source_extension='png', save_extension='png',
                                            is_save_npy=False, is_save_img=True):
    """
    rotate image 180 degree clockwise and save to folder
    :param source_path: source_path/image files
    :param save_path: save_path/rotated image files
    :param source_extension: image files' extension, default to 'png'
    :param save_extension: rotated image's save extension, default to 'png'
    :param is_save_npy: if to save npy files, default to False.
    :param is_save_img: if to save image files, default to True.
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == source_extension:
                pbar.set_description("Processing %s" % img_files.name)
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
                    source_img = cv2.imread(img_files.path, -1)
                rotated_img = __rotate_img_90_degree(source_img)
                rotated_img = __rotate_img_90_degree(rotated_img)
                filename_no_extension, extension = os.path.splitext(img_files.name)
                os.chdir(save_path)
                if is_save_img:
                    cv2.imwrite(filename_no_extension + '_rotate180c.' + save_extension, rotated_img)
                if is_save_npy:
                    np.save(filename_no_extension + '_rotate180c.npy', rotated_img)
    return True


def rotate_img_180degree_and_save_to_folder_with_category(
        source_path, save_path, source_extension='png',  save_extension='png', is_save_npy=False, is_save_img=True):
    """
        rotate image 180 degree clockwise and save to folder
        :param source_path: source_path/category/image files
        :param save_path: save_path/rotated image files
        :param source_extension: image files' extension, default to 'png'
        :param save_extension: rotated image's save extension, default to 'png'
        :param is_save_npy: if to save npy files, default to False
        :param is_save_img: if to save image files, default to True.
        :return: True
        """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            rotate_img_180degree_and_save_to_folder(
                category.path, os.path.join(save_path, category.name), source_extension, save_extension,
                is_save_npy=is_save_npy, is_save_img=is_save_img)
    return True


def rotate_img_270degree_and_save_to_folder(source_path, save_path, source_extension='png',  save_extension='png',
                                            is_save_npy=False, is_save_img=True):
    """
    rotate image 270 degree clockwise and save to folder
    :param source_path: source_path/image files
    :param save_path: save_path/rotated image files
    :param source_extension: image files' extension, default to 'png'
    :param save_extension: rotated image's save extension, default to 'png'
    :param is_save_npy: if to save npy files, default to False
    :param is_save_img: if to save image files, default to True.
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == source_extension:
                pbar.set_description("Processing %s" % img_files.name)
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
                    source_img = cv2.imread(img_files.path, -1)
                rotated_img = __rotate_img_90_degree(source_img)
                rotated_img = __rotate_img_90_degree(rotated_img)
                rotated_img = __rotate_img_90_degree(rotated_img)
                filename_no_extension, extension = os.path.splitext(img_files.name)
                os.chdir(save_path)
                if is_save_img:
                    cv2.imwrite(filename_no_extension + '_rotate270c.' + save_extension, rotated_img)
                if is_save_npy:
                    np.save(filename_no_extension + '_rotate270c.npy', rotated_img)
    return True


def rotate_img_270degree_and_save_to_folder_with_category(
        source_path, save_path, source_extension='png',  save_extension='png', is_save_npy=False, is_save_img=True):
    """
        rotate image 270 degree clockwise and save to folder
        :param source_path: source_path/category/image files
        :param save_path: save_path/rotated image files
        :param source_extension: image files' extension, default to 'png'
        :param save_extension: rotated image's save extension, default to 'png'
        :param is_save_npy: if to save npy files, default to False.
        :param is_save_img: if to save image files, default to False.
        :return: True
        """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            rotate_img_270degree_and_save_to_folder(
                category.path, os.path.join(save_path, category.name), source_extension, save_extension, is_save_npy,
            is_save_img)
    return True

# flip
def __flip_img(img, mode=0):
    """
    flip image
    :param img: image data
    :param mode: int
                 0: flip horizontally
                 1: flip vertically
    :return: flip image data
    """
    img_width = None
    img_height, img_width = img.shape[0:2]
    flip_img = np.zeros_like(img)
    if img_width is None:
        raise ValueError('''input image's shape is not valid !''')
    if 0 == mode:
        for ind_col in range(img_width):
            flip_img[:, img_width - ind_col - 1] = img[:, ind_col]
    elif 1 == mode:
        for ind_row in range(img_height):
            flip_img[img_height - ind_row - 1] = img[ind_row]
    else:
        raise ValueError('invalid input mode: %d, must be 0(Horizontally) or 1(Vertically)' % mode)
    return flip_img


def flip_img_and_save_to_folder(source_path, save_path, mode=0, source_extension='png', is_save_npy=False):
    """
    flip image and save to folder
    :param source_path: source_path/image files
    :param save_path: save_path/rotated image files
    :param mode: int 0: flip horizontally    1: flip vertically
    :param source_extension: image files' extension, default to 'png'
    :param is_save_npy: if to save npy files, default to False
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == source_extension:
                pbar.set_description("Processing %s" % img_files.name)
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
                    source_img = cv2.imread(img_files.path, -1)
                flip_img = __flip_img(source_img, mode)
                filename_no_extension, extension = os.path.splitext(img_files.name)
                os.chdir(save_path)
                cv2.imwrite(filename_no_extension + '_flip.png', flip_img)
                if is_save_npy:
                    np.save(filename_no_extension + '_flip.npy', flip_img)
    return True


def flip_img_and_save_to_folder_with_category(
        source_path, save_path, mode=0, source_extension='png', is_save_npy=False):
    """
        flip img and save to folder
        :param source_path: source_path/category/image files
        :param save_path: save_path/rotated image files
        :param mode: int 0: flip horizontally    1: flip vertically
        :param source_extension: image files' extension, default to 'png'
        :param is_save_npy: if to save npy files, default to False
        :return: True
        """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    for category in os.scandir(source_path):
        if category.is_dir():
            flip_img_and_save_to_folder(
                category.path, os.path.join(save_path, category.name), mode, source_extension, is_save_npy)
    return True


def gen_dataset_img_size_statistic(source_path, source_extension):
    """
    generate dataset image's size statistic data for a particular category
    :param source_path: the image files path , source_path/image files
    :return: two matrix, size_statistics and size_list
            size_statistics:
                      width1 width2 width3 width4 ...
             height1  number number number number ...
             height2  number ...
             height3  number .   .   .
             height4  number  .        .        .
             .         .
             .                     .
             .                                   .
             size_list:
             [[height1, width1, number]
              [height2, width2, number]
              ...]
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    img_width_list = []
    img_height_list = []
    for img_file in os.scandir(source_path):
        if img_file.is_file():
            extension = os.path.splitext(img_file.path)[1][1:]
            if extension == source_extension:
                if extension == 'npy':
                    source_img = np.load(img_file.path)
                elif extension in ['tif', 'tiff']:
                    tif = TIFF.open(img_file.path, mode='r')
                    source_img = tif.read_image()
                else:
                    source_img = cv2.imread(img_file.path, -1)
                img_height = source_img.shape[0]
                img_width = source_img.shape[1]
                if img_height not in img_height_list:
                    img_height_list.append(img_height)
                if img_width not in img_width_list:
                    img_width_list.append(img_width)
    img_width_list.sort()  # ascending
    img_height_list.sort()
    size_statistic = np.zeros([len(img_height_list)+1, len(img_width_list)+1], dtype=np.int32)
    size_statistic[0, 1:] = img_width_list
    size_statistic[1:, 0] = img_height_list
    pbar = tqdm(os.scandir(source_path))
    for img_file in pbar:
        if img_file.is_file():
            pbar.set_description("Processing %s" % img_file.name)
            extension = os.path.splitext(img_file.path)[1][1:]
            if extension == source_extension:
                if extension == 'npy':
                    source_img = np.load(img_file.path)
                elif extension in ['tif', 'tiff']:
                    tif = TIFF.open(img_file.path, mode='r')
                    source_img = tif.read_image()
                else:
                    source_img = cv2.imread(img_file.path, -1)
                img_height = source_img.shape[0]
                img_width = source_img.shape[1]
                for ind_w in range(len(img_width_list)):
                    if img_width == img_width_list[ind_w]:
                        break
                for ind_h in range(len(img_height_list)):
                    if img_height == img_height_list[ind_h]:
                        break
                size_statistic[ind_h+1, ind_w+1] += 1
    is_first = True
    for ind_h in range(1, size_statistic.shape[0]):
        for ind_w in range(1, size_statistic.shape[1]):
            if size_statistic[ind_h, ind_w]:
                if is_first:
                    size_list = np.array([[size_statistic[ind_h, 0],
                                           size_statistic[0, ind_w],
                                           size_statistic[ind_h, ind_w]]])
                    is_first = False
                else:
                    size_list = np.append(size_list,
                                          np.array(
                                              [[size_statistic[ind_h, 0],
                                               size_statistic[0, ind_w],
                                               size_statistic[ind_h, ind_w]]]),
                                          axis=0)
    # print(img_height_list)
    # print(img_width_list)
    print(size_statistic)
    print('\n')
    print(size_list)
    return size_statistic, size_list


def change_img_file_extension(source_path, save_path, ori_extension, new_extension):
    """
    change image files' extension, source_path/image files
    :param source_path: original images' path
    :param save_path: new images' save path
    :param ori_extension: original images' file extension
    :param new_extension: new extension
    :return: True
    """
    if not os.path.exists(source_path):
        raise FileExistsError('path not found! : %s' % source_path)
    os.makedirs(save_path, exist_ok=True)
    pbar = tqdm(os.scandir(source_path))
    for img_files in pbar:
        if img_files.is_file():
            extension = os.path.splitext(img_files.path)[1][1:]
            if extension == ori_extension:
                pbar.set_description("Processing %s" % img_files.name)
                if extension == 'npy':
                    source_img = np.load(img_files.path)
                else:
                    source_img = cv2.imread(img_files.path, -1)
                filename_no_extension, extension = os.path.splitext(img_files.name)
                os.chdir(save_path)
                cv2.imwrite(filename_no_extension+'.'+new_extension, source_img)




if __name__ == '__main__':
    pass


