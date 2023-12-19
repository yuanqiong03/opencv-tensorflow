import cv2
import sys
import os

from PIL import Image



def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name,dir_name):
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


if __name__ == '__main__':
    path_name = './face_data'
    CatchPICFromVideo("catch_face_data", 0, 200 - 1, './face_data/uesr3', path_name)