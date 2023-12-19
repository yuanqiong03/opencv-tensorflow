from __future__ import absolute_import, division, print_function, unicode_literals

import subprocess
import tkinter
from tkinter import messagebox
from tkinter.messagebox import showinfo, showerror
import tkinter.simpledialog as sd
from PIL.Image import fromarray, Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from PIL.ImageTk import PhotoImage
import numpy as np
import os
from face_data_train import CNN
import cv2
from PIL import Image, ImageDraw, ImageFont



def create_new_dataset(path_name):
    i = 1
    while True:
        new_path = os.path.join(path_name, f'user{i}')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            break
        i += 1
    return new_path


def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name, dir_name):
    cv2.namedWindow(window_name)

    # 确保所有的输出目录都存在
    for dir_name in path_name:
        if not os.path.exists(path_name):
            os.makedirs(path_name)

    # 视频来源，可以选择摄像头或者视频
    cap = cv2.VideoCapture(camera_idx)

    # OpenCV人脸识别分类器
    classfier = cv2.CascadeClassifier(
        "./haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 将当前帧保存为图片
                img_name = '%s/%d.jpg' % (path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)

                num += 1
                if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                    break

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (30, 30), font, 1, (255, 0, 0), 1)

                # 左上角显示当前捕捉到了多少人脸图片了
                # 超过指定最大保存数量结束程序
        if num > (catch_pic_num): break

        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


class GUI:
    def __init__(self):

        self.num = None
        self.face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
        self.captured_faces = []


        self.labels = self.load_labels_from_file('labels.txt')  # 默认的标签列表
        self.video = None
        self.after = None

        self.root = tkinter.Tk()
        self.root.geometry('1000x550')
        self.root.title('人脸识别')
        self.root.resizable(width=False, height=False)

        self.Img1label = tkinter.Label(self.root, text='', bg='green', bd=10)
        self.Img1label.place(x=200 + 40 + 100, y=20, width=640, height=480)

        self.btn1 = tkinter.Button(self.root, text='测试摄像头', command=self.video_change1, font=('宋体', 20))
        self.btn1.place(x=50, y=20, width=200, height=50)

        self.btn2 = tkinter.Button(self.root, text='收集人脸', command=self.face_catch, font=('宋体', 20))
        self.btn2.place(x=50, y=20 + 50 + 20, width=200, height=50)

        self.btn3 = tkinter.Button(self.root, text='模型训练', command=self.face_train, font=('宋体', 20))
        self.btn3.place(x=50, y=20 + 50 + 20 + 50 + 20, width=200, height=50)

        self.btn4 = tkinter.Button(self.root, text='开始识别', command=self.video_change2, font=('宋体', 20))
        self.btn4.place(x=50, y=20 + 50 + 20 + 50 + 20 + 50 + 20, width=200, height=50)

        self.btn4 = tkinter.Button(self.root, text='关闭摄像头', command=self.close_video, font=('宋体', 20))
        self.btn4.place(x=50, y=20 + 50 + 20 + 50 + 20 + 50 + 20 + 50 + 20, width=200, height=50)

    def video_change1(self):
        if self.video:
            self.video.release()
        self.video = cv2.VideoCapture(0)
        if self.after:
            self.root.after_cancel(self.after)
        self.video_open()

    def video_change2(self):
        if self.video:
            self.video.release()
        self.video = cv2.VideoCapture(0)
        if self.after:
            self.root.after_cancel(self.after)
        self.face_apply()

    def face_train(self):
        if self.video:
            self.video.release()
        self.video = cv2.VideoCapture(0)

        # 执行你的项目文件
        project_file = 'face_data_train.py'
        subprocess.run(['python', project_file])

        # 弹出训练完成的文本弹窗
        showinfo("训练完成", message='训练已完成')

    # def face_apply(self):
    #     ret, frame = self.video.read()
    #     if ret and np.any(frame):
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #         for (x, y, w, h) in faces:
    #             cv2.rectangle(frame, (x-10, y-10), (x + w+10, y + h+10), (0, 255, 0), 2)
    #         self.img1_show(frame)
    #     self.root.after(10, self.face_apply)
    def face_apply(self):
        model = CNN()
        model.load_weights('./model/face1')  # 读取模型权重参数

        ret, frame = self.video.read()
        if ret and np.any(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceRects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), thickness=2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pilimg = Image.fromarray(frame)

                    draw = ImageDraw.Draw(pilimg)
                    font = ImageFont.truetype("simkai.ttf", 20, encoding="utf-8")

                    face_probe = model.face_predict(image)

                    max_prob = max(face_probe)
                    max_label_index = np.argmax(face_probe)

                    labels = self.labels  # 使用更新后的标签数组

                    if max_prob >= 0.75:
                        label = labels[max_label_index]
                        draw.text((x + 25, y - 45), '{}:{:.2%}'.format(label, max_prob), (255, 0, 0), font=font)
                        frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
                    else:
                        draw.text((x + 25, y - 45), '未识别', (255, 0, 0), font=font)
                        frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

                    self.img1_show(frame)

        self.root.after(10, self.face_apply)

    def video_open(self):
        ret, frame = self.video.read()
        if ret and np.any(frame):
            self.img1_show(frame)
        self.after = self.root.after(10, self.video_open)

    def close_video(self):
        if self.video:
            self.video.release()
            self.video = None
            self.Img1label.configure(text='摄像头已关闭')
            self.Img1label.image = None  # 清空图像
            self.num = ''  # 重置手势识别结果为初始状态

    def img1_show(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = fromarray(img)
        img = PhotoImage(img)
        self.Img1label.image = img
        self.Img1label['image'] = img

    # 读取文本文件内容并将其存储到标签数组中
    def load_labels_from_file(self, filename):
        labels = []
        with open(filename, 'r') as file:
            for line in file:
                label = line.strip()
                labels.append(label)
        return labels


    # 当用户添加新的标签时，将其保存到txt文件中。
    def save_labels_to_file(self, filename, labels):
        with open(filename, 'w') as file:
            for label in labels:
                file.write(label + '\n')

    def face_catch(self):
        if self.video:
            self.video.release()
        if self.after:
            self.root.after_cancel(self.after)

        new_label = sd.askstring("添加用户", "请输入名字")
        if new_label:
            self.labels.append(new_label)
            self.save_labels_to_file('labels.txt', self.labels)
            messagebox.showinfo("添加成功", "已成功添加，请等待捕获人脸进程")
        path_name = './face_data'
        new_dataset_path = create_new_dataset(path_name)
        CatchPICFromVideo("catch_face_data", 0, 200 - 1, new_dataset_path, path_name)

        # 调用人脸捕获函数

    def open(self):
        self.root.mainloop()

    def close(self):
        if self.video:
            self.video.release()


GUI().open()
GUI().close()
