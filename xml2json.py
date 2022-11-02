import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import time
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_dir, json_file, nums, dataset):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    # print(categories)
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}

    pbar = tqdm(total=nums, desc="%s-porcess" % dataset, unit="xml")  # 开启进度条

    index = -1
    '''读xml'''
    for line in glob.glob(xml_dir + "*.xml"):
        index = index + 1
        # print("Processing %s"%(line))

        pbar.update(1)

        xml_f = open(line, "r", encoding="utf-8")
        tree = ET.parse(xml_f)
        root = tree.getroot()

        filename = os.path.basename(line)[:-4] + ".jpg"
        image_id = 20210700001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            # print("all_categories",all_categories)#统计各类别数目
            if category not in categories:
                if only_care_pre_define_categories:
                    # print("continue????")
                    continue
                new_id = len(categories) + 1
                print(
                    "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                        category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]  # 0
            bndbox = get_and_check(obj, 'bndbox', 1)
            # print(bndbox)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            # print(xmin,ymin,xmax,ymax)
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            # 存入字典
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=2)  # 缩进2格
    json_fp.write(json_str)
    json_fp.close()

    pbar.close()
    # print("-----create {} done-----".format(json_file))
    # print("find {} categories: {} \n-->>> your pre_define_categories {}: {}".format(len(all_categories), all_categories.keys(), len(pre_define_categories), pre_define_categories.keys()))
    # print("category: id \n--> {}".format(categories))
    # print(categories.keys())
    # print(categories.values())


if __name__ == '__main__':
    classes = ['fuwo', 'cewo', 'zhanli']  # xml上保存的类别名称
    pre_define_categories = {}
    for i, clss in enumerate(classes):
        pre_define_categories[str(i)] = i + 1
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False

    xml_train_dir = "E:/DL/detectron2/SwinT_detectron2/datasets/new/train/annotations/"
    save_json_train = 'E:/DL/detectron2/SwinT_detectron2/datasets/new/pig_annotation_train.json'

    # xml_test_dir = "E:/DL/detectron2/SwinT_detectron2/datasets/test/annotations/"
    # save_json_test = 'E:/DL/detectron2/SwinT_detectron2/datasets/pig_annotation_test.json'

    train_num = len(os.listdir(xml_train_dir))
    # test_num = len(os.listdir(xml_test_dir))

    convert(xml_train_dir, save_json_train, train_num, "train")
    print("\n")
    # convert(xml_test_dir, save_json_test, test_num, "test")
    print("\n------------done!-----------")