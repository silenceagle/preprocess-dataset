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

                        
def gen_dataset_hrsc2016(xml_path, source_img_path, save_img_path):
    """
    crop ship images from original bigger images and save them to folders named with their categories and keep the train
    and test set unchanged and save them with name format :
    name_{category}_ort_{}_rsl_{}_x_{}_y{}.bmp
    name category oirentation(rad) resoluton(m) center x coordinate  center y coordinate
    :param xml_path: annotation files path, structure should be xml_path/xml files
    :param source_img_path: original images path, structure should be source_img_path/Train/AllImages/train images(.bmp)
                                                                      source_img_path/Test/AllImages/test images(.bmp)
    :param save_img_path: str. The path to save croped images, structure should be:
                    save_img_path/train/aircraft carrier/ship type folders/images
                                                        /images just be 'aircraft carrier'
                                       /warcraft/ images/images
                                       /merchant ship/images
                                       /images just be 'ship'
                                /test/...
    :return:
    """
    if not os.path.exists(xml_path):
        raise FileExistsError('path not found! : %s' % xml_path)
    if not os.path.exists(source_img_path):
        raise FileExistsError('path not found! : %s' % source_img_path)
    train_img_path = os.path.join(source_img_path, 'Train', 'AllImages')
    test_img_path = os.path.join(source_img_path, 'Test', 'AllImages')
    categories_dict = {}
    with xml.dom.minidom.parse(os.path.join(source_img_path, 'FullDataSet', 'sysdata.xml')) as category_document:
        categories = category_document.getElementsByTagName('HRSC_Classes')[0].getElementsByTagName('HRSC_Class')
        for category in categories:
            category_id = category.getElementsByTagName('Class_ID')[0].firstChild.data
            category_layer = category.getElementsByTagName('Class_Layer')[0].firstChild.data
            category_engname = category.getElementsByTagName('Class_EngName')[0].firstChild.data.split('(')[0].replace(
                ' ', '-').replace('|--)', '')
            category_name = category.getElementsByTagName('Class_Name')[0].firstChild.data.split('(')[0]
            if '0' != category_layer:  # there is specific ship category
                category_class_id = category.getElementsByTagName('HRS_Class_ID')[0].firstChild.data
                categories_dict[category_id] = {
                    'category_id': category_id,
                    'category_layer':category_layer,
                    'category_engname':category_engname,
                    'category_name':category_name,
                    'category_class_id': category_class_id
                }
            else:  # label is just 'ship'
                categories_dict[category_id] = {
                    'category_id': category_id,
                    'category_layer': category_layer,
                    'category_engname': category_engname,
                    'category_name': category_name,
                    'category_class_id': '100000001'
                }
    # train files
    train_pbar = tqdm(os.scandir(train_img_path))
    for train_img in train_pbar:
        if train_img.is_file():
            extension = os.path.splitext(train_img.path)[1][1:]
            train_img_name = train_img.name.split('.')[0]
            if 'bmp' == extension:  # bmp images
                train_pbar.set_description("Processing %s" % train_img.path)
                try:
                    document = xml.dom.minidom.parse(os.path.join(xml_path, train_img_name+'.xml'))
                    is_annotated = document.getElementsByTagName('Annotated')[0].firstChild.data
                    if '0' == is_annotated:  # without annotations
                        continue
                    # img_id = document.getElementsByTagName('Img_ID')[0].firstChild.data
                    img_resolution = document.getElementsByTagName('Img_Resolution')[0].firstChild.data
                    ships = document.getElementsByTagName('HRSC_Objects')[0].getElementsByTagName('HRSC_Object')
                    for ship in ships:
                        ship_category_id = ship.getElementsByTagName('Class_ID')[0].firstChild.data
                        ship_category_dict = categories_dict[ship_category_id]

                        # get four corner points' coordinates of the rotated bounding box
                        box_cx = float(ship.getElementsByTagName('mbox_cx')[0].firstChild.data)
                        box_cy = float(ship.getElementsByTagName('mbox_cy')[0].firstChild.data)
                        box_w = float(ship.getElementsByTagName('mbox_w')[0].firstChild.data)
                        box_h = float(ship.getElementsByTagName('mbox_h')[0].firstChild.data)
                        box_angle = float(ship.getElementsByTagName('mbox_ang')[0].firstChild.data)  # rad
                        box_x1 = int(box_cx + box_h * 0.5 * np.sin(box_angle) - box_w * 0.5 * np.cos(box_angle))
                        box_y1 = int(box_cy - box_h * 0.5 * np.cos(box_angle) - box_w * 0.5 * np.sin(box_angle))
                        box_x2 = int(box_cx + box_h * 0.5 * np.sin(box_angle) + box_w * 0.5 * np.cos(box_angle))
                        box_y2 = int(box_cy - box_h * 0.5 * np.cos(box_angle) + box_w * 0.5 * np.sin(box_angle))
                        box_x3 = int(box_cx - box_h * 0.5 * np.sin(box_angle) + box_w * 0.5 * np.cos(box_angle))
                        box_y3 = int(box_cy + box_h * 0.5 * np.cos(box_angle) + box_w * 0.5 * np.sin(box_angle))
                        box_x4 = int(box_cx - box_h * 0.5 * np.sin(box_angle) - box_w * 0.5 * np.cos(box_angle))
                        box_y4 = int(box_cy + box_h * 0.5 * np.cos(box_angle) - box_w * 0.5 * np.sin(box_angle))

                        # get ship orientation, define as the clockwise angle from ship head to North (Up)
                        try:
                            ship_head_x = int(ship.getElementsByTagName('header_x')[0].firstChild.data)
                            ship_head_y = int(ship.getElementsByTagName('header_y')[0].firstChild.data)
                            if box_w < box_h:
                                if ship_head_y > box_cy:
                                    ship_orientation = np.pi - box_angle
                                elif box_angle < 0:
                                    ship_orientation = -box_angle
                                else:
                                    ship_orientation = 2.0 * np.pi - box_angle
                            else:
                                if ship_head_x < box_cx:
                                    ship_orientation = np.pi * 0.5 - box_angle
                                else:
                                    ship_orientation = 1.5 * np.pi - box_angle
                        except:  # ship head coordinates is not given
                            if box_w < box_h:  # heads up
                                if box_angle < 0:
                                    ship_orientation = -box_angle
                                else:
                                    ship_orientation = 2.0 * np.pi - box_angle
                            else:  # heads right
                                ship_orientation = 1.5 * np.pi - box_angle

                        # crop ship images
                        ori_image = cv2.imread(train_img.path, -1)
                        box = [(box_x1, box_y1), (box_x2, box_y2), (box_x3, box_y3), (box_x4, box_y4)]
                        xmin = min(box_x1, box_x2, box_x3, box_x4)
                        xmax = max(box_x1, box_x2, box_x3, box_x4)
                        ymin = min(box_y1, box_y2, box_y3, box_y4)
                        ymax = max(box_y1, box_y2, box_y3, box_y4)
                        if len(ori_image.shape) == 3:
                            ori_h, ori_w, image_channels = ori_image.shape
                            sub_image = np.zeros([ymax - ymin + 1, xmax - xmin + 1, image_channels], dtype=np.int)
                        else:
                            oir_h, ori_w = ori_image.shape
                            sub_image = np.zeros([ymax - ymin + 1, xmax - xmin + 1], dtype=np.int)
                        for y in range(sub_image.shape[0]):  # row
                            for x in range(sub_image.shape[1]):  # col
                                if pnpoly([xmin + x, ymin + y], box):
                                    sub_image[y, x] = ori_image[min(ymin + y - 1, ori_h-1), min(xmin + x - 1, ori_w-1)]
                        sub_imagename = f'''{train_img_name}_{ship_category_dict['category_engname']}''' + \
                            f'''_ort_{ship_orientation:.3f}_rsl_{img_resolution}_x_{int(box_cx)}_y_{int(box_cy)}.bmp'''

                        if '0' == ship_category_dict['category_layer']:  # just be 'ship'
                            ship_save_folder = os.path.join(save_img_path, 'train', 'ship')
                        elif '1' == ship_category_dict['category_layer']:  # ship class
                            ship_save_folder = os.path.join(save_img_path,'train', 'ship',
                                                            ship_category_dict['category_engname'])
                        else:  # '2' == ship_category_dict['category_layer']:  # ship type
                            ship_class_name = categories_dict[ship_category_dict['category_class_id']][
                                'category_engname']
                            ship_save_folder = os.path.join(save_img_path, 'train', 'ship', ship_class_name,
                                                            ship_category_dict['category_engname'])
                        os.makedirs(ship_save_folder, exist_ok=True)
                        cv2.imwrite(os.path.join(ship_save_folder, sub_imagename), sub_image)
                except:  #
                    print(f'''could not find {os.path.join(xml_path, train_img_name+'.xml')}''')

                pass

    # test files
    test_pbar = tqdm(os.scandir(test_img_path))
    for test_img in test_pbar:
        if test_img.is_file():
            extension = os.path.splitext(test_img.path)[1][1:]
            test_img_name = test_img.name.split('.')[0]
            if 'bmp' == extension:  # bmp images
                test_pbar.set_description("Processing %s" % test_img.path)
                try:
                    document = xml.dom.minidom.parse(os.path.join(xml_path, test_img_name + '.xml'))
                    is_annotated = document.getElementsByTagName('Annotated')[0].firstChild.data
                    if '0' == is_annotated:  # without annotations
                        continue
                    # img_id = document.getElementsByTagName('Img_ID')[0].firstChild.data
                    img_resolution = document.getElementsByTagName('Img_Resolution')[0].firstChild.data
                    ships = document.getElementsByTagName('HRSC_Objects')[0].getElementsByTagName('HRSC_Object')
                    for ship in ships:
                        ship_category_id = ship.getElementsByTagName('Class_ID')[0].firstChild.data
                        ship_category_dict = categories_dict[ship_category_id]

                        # get four corner points' coordinates of the rotated bounding box
                        box_cx = float(ship.getElementsByTagName('mbox_cx')[0].firstChild.data)
                        box_cy = float(ship.getElementsByTagName('mbox_cy')[0].firstChild.data)
                        box_w = float(ship.getElementsByTagName('mbox_w')[0].firstChild.data)
                        box_h = float(ship.getElementsByTagName('mbox_h')[0].firstChild.data)
                        box_angle = float(ship.getElementsByTagName('mbox_ang')[0].firstChild.data)  # rad
                        box_x1 = int(box_cx + box_h * 0.5 * np.sin(box_angle) - box_w * 0.5 * np.cos(box_angle))
                        box_y1 = int(box_cy - box_h * 0.5 * np.cos(box_angle) - box_w * 0.5 * np.sin(box_angle))
                        box_x2 = int(box_cx + box_h * 0.5 * np.sin(box_angle) + box_w * 0.5 * np.cos(box_angle))
                        box_y2 = int(box_cy - box_h * 0.5 * np.cos(box_angle) + box_w * 0.5 * np.sin(box_angle))
                        box_x3 = int(box_cx - box_h * 0.5 * np.sin(box_angle) + box_w * 0.5 * np.cos(box_angle))
                        box_y3 = int(box_cy + box_h * 0.5 * np.cos(box_angle) + box_w * 0.5 * np.sin(box_angle))
                        box_x4 = int(box_cx - box_h * 0.5 * np.sin(box_angle) - box_w * 0.5 * np.cos(box_angle))
                        box_y4 = int(box_cy + box_h * 0.5 * np.cos(box_angle) - box_w * 0.5 * np.sin(box_angle))

                        # get ship orientation, define as the clockwise angle from ship head to North (Up)
                        try:
                            ship_head_x = int(ship.getElementsByTagName('header_x')[0].firstChild.data)
                            ship_head_y = int(ship.getElementsByTagName('header_y')[0].firstChild.data)
                            if box_w < box_h:
                                if ship_head_y > box_cy:
                                    ship_orientation = np.pi - box_angle
                                elif box_angle < 0:
                                    ship_orientation = -box_angle
                                else:
                                    ship_orientation = 2.0 * np.pi - box_angle
                            else:
                                if ship_head_x < box_cx:
                                    ship_orientation = np.pi * 0.5 - box_angle
                                else:
                                    ship_orientation = 1.5 * np.pi - box_angle
                        except:  # ship head coordinates is not given
                            if box_w < box_h:  # heads up
                                if box_angle < 0:
                                    ship_orientation = -box_angle
                                else:
                                    ship_orientation = 2.0 * np.pi - box_angle
                            else:  # heads right
                                ship_orientation = 1.5 * np.pi - box_angle

                        # crop ship images
                        ori_image = cv2.imread(test_img.path, -1)
                        box = [(box_x1, box_y1), (box_x2, box_y2), (box_x3, box_y3), (box_x4, box_y4)]
                        xmin = min(box_x1, box_x2, box_x3, box_x4)
                        xmax = max(box_x1, box_x2, box_x3, box_x4)
                        ymin = min(box_y1, box_y2, box_y3, box_y4)
                        ymax = max(box_y1, box_y2, box_y3, box_y4)
                        if len(ori_image.shape) == 3:
                            ori_h, ori_w, image_channels = ori_image.shape
                            sub_image = np.zeros([ymax - ymin + 1, xmax - xmin + 1, image_channels], dtype=np.int)
                        else:
                            oir_h, ori_w = ori_image.shape
                            sub_image = np.zeros([ymax - ymin + 1, xmax - xmin + 1], dtype=np.int)
                        for y in range(sub_image.shape[0]):  # row
                            for x in range(sub_image.shape[1]):  # col
                                if pnpoly([xmin + x, ymin + y], box):
                                    sub_image[y, x] = ori_image[
                                        min(ymin + y - 1, ori_h - 1), min(xmin + x - 1, ori_w - 1)]
                        sub_imagename = f'''{test_img_name}_{ship_category_dict['category_engname']}''' + \
                                        f'''_ort_{ship_orientation:.3f}_rsl_{img_resolution}''' + \
                                        f'''_x_{int(box_cx)}_y_{int(box_cy)}.bmp'''

                        if '0' == ship_category_dict['category_layer']:  # just be 'ship'
                            ship_save_folder = os.path.join(save_img_path, 'test', 'ship')
                        elif '1' == ship_category_dict['category_layer']:  # ship class
                            ship_save_folder = os.path.join(save_img_path, 'test', 'ship',
                                                            ship_category_dict['category_engname'])
                        else:  # '2' == ship_category_dict['category_layer']:  # ship type
                            ship_class_name = categories_dict[ship_category_dict['category_class_id']][
                                'category_engname']
                            ship_save_folder = os.path.join(save_img_path, 'test', 'ship', ship_class_name,
                                                            ship_category_dict['category_engname'])
                        os.makedirs(ship_save_folder, exist_ok=True)
                        cv2.imwrite(os.path.join(ship_save_folder, sub_imagename), sub_image)
                except:  #
                    print(f'''could not find {os.path.join(xml_path, test_img_name + '.xml')}''')


if __name__ == '__main__':
    gen_dataset_ssdd(xml_path=r'F:\dataset_se\SSDD\Annotations',
                     source_img_path=r'F:\dataset_se\SSDD\JPEGImages',
                     save_img_path=r'F:\dataset_se\SSDD\crop_img')
    pass

