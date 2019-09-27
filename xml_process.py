"""
    xml_process.py
    20190408
"""

import xml.dom.minidom
import os
from tqdm import tqdm
import cv2
import numpy as np


def pnpoly(test_point, polygon):
    """
        Point Inclusion in Polygon Test
        https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
        :param test_point: the point to test , e[x, y]
        :param polygon: the polygon , [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        :return is_inside : if in, return True
    """
    is_inside = False
    minX = polygon[0][0]
    maxX = polygon[0][0]
    minY = polygon[0][1]
    maxY = polygon[0][1]
    for p in polygon:
        minX = min(p[0], minX)
        maxX = max(p[0], maxX)
        minY = min(p[1], minY)
        maxY = max(p[1], maxY)
    if test_point[0] < minX or test_point[0] > maxX or test_point[1] < minY or test_point[1] > maxY:
        return False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        if ((polygon[i][1] > test_point[1]) != (polygon[j][1] > test_point[1]) and (
                test_point[0] < (polygon[j][0] - polygon[i][0]) * (test_point[1] - polygon[i][1]) / (
                polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            is_inside = not is_inside
        j = i
    return is_inside


def gen_dataset_ssdd(xml_path, source_img_path, save_img_path):
    """
        pick, crop and save target images
        :param xml_path: str. The folder path save xml files
        :param source_img_path: str. The source image's path
        :param save_img_path: str. The path to save croped images
        :return
    """
    if not os.path.exists(xml_path):
        raise FileExistsError('path not found! : %s' % xml_path)
    if not os.path.exists(source_img_path):
        raise FileExistsError('path not found! : %s' % source_img_path)
    os.makedirs(save_img_path, exist_ok=True)
    pbar = tqdm(os.scandir(xml_path))
    for xml_file in pbar:
        if xml_file.is_file():
            extension = os.path.splitext(xml_file.path)[1][1:]
            if 'xml' == extension:
                pbar.set_description("Processing %s" % xml_file.path)
                dom = xml.dom.minidom.parse(xml_file.path)
                root = dom.documentElement
                img_name = root.getElementsByTagName('filename')[0].firstChild.data
                my_object_list = root.getElementsByTagName('object')
                for my_object in my_object_list:
                    object_type = my_object.getElementsByTagName('name')[0].firstChild.data
                    if object_type == 'ship':
                        bndbox = my_object.getElementsByTagName('bndbox')[0]
                        xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
                        ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
                        xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
                        ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
                        a = os.path.join(source_img_path, img_name+'.jpg')
                        ori_image = cv2.imread(os.path.join(source_img_path, img_name+'.jpg'), -1)
                        box = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
                        if len(ori_image.shape) == 3:
                            _, _, image_channels = ori_image.shape
                            sub_image = np.zeros([ymax - ymin + 1, xmax - xmin + 1, image_channels], dtype=np.int)
                        else:
                            sub_image = np.zeros([ymax - ymin + 1, xmax - xmin + 1], dtype=np.int)
                        for y in range(sub_image.shape[0]): #row
                            for x in range(sub_image.shape[1]): #col
                                sub_image[y,x] = ori_image[ymin+y-1, xmin+x-1]
                        sub_imagename = img_name+'_'+str(xmin)+'_'+str(ymin)+'_'+str(xmax)+'_'+str(ymax)+'.png'
                        cv2.imwrite(os.path.join(save_img_path, sub_imagename), sub_image[:, :, 0])


if __name__ == '__main__':
    gen_dataset_ssdd(xml_path=r'F:\dataset_se\SSDD\Annotations',
                     source_img_path=r'F:\dataset_se\SSDD\JPEGImages',
                     save_img_path=r'F:\dataset_se\SSDD\crop_img')
    pass

