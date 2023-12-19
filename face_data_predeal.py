import os
import sys
import numpy as np
import cv2

# 将图片大小设置为64*64
IMAGE_SIZE = 64

# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape

    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多少像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

        # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 将图像设置为灰度图
    constant = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)

    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))


# 读取训练数据
images = []
labels = []

def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # 从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)

                images.append(image)
                labels.append(path_name)

    return images, labels


# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)

    # 将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    # 尺寸为 200*5* 64 * 64 * 3
    # 5个人 每个人200张 图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    print(images.shape)

    # # 标注数据（采用onehot编码）
    temp = 0
    for label in labels:
        if label.endswith('user1'):
            labels[temp] = 0
        elif label.endswith('user2'):
            labels[temp] = 1
        elif label.endswith('user3'):
            labels[temp] = 2
        elif label.endswith('user4'):
            labels[temp] = 3
        elif label.endswith('user5'):
            labels[temp] = 4
        elif label.endswith('user6'):
            labels[temp] = 5
        elif label.endswith('user7'):
            labels[temp] = 6
        elif label.endswith('user8'):
            labels[temp] = 7
        elif label.endswith('user9'):
            labels[temp] = 8
        elif label.endswith('user10'):
            labels[temp] = 9
        temp = temp + 1



    # labels = []
    # dataset_folders = [folder for folder in os.listdir(path_name) if os.path.isdir(os.path.join(path_name, folder))]
    # dataset_folders.sort()  # 确保按顺序处理文件夹
    # for folder_name in dataset_folders:
    #     label = int(folder_name.replace('user', ''))
    #     labels.append(label)
    return images, labels


if __name__ == '__main__':
    images, labels = load_dataset("./face_data")
    print(labels)
