from __future__ import absolute_import, division, print_function, unicode_literals
import random
import numpy as np
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from keras import backend as K

from face_data_predeal import load_dataset, resize_image, IMAGE_SIZE
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras import Model



class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None
        self.n_classes = None

        # n_classes = len(np.unique(y_train))
        # n_classes

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=1):  # 灰度图 所以通道数为1

        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 100))
        # 将总数据按0.3比重随机分配给训练集和测试集

        # 由于TensorFlow需要通道数，我们上一步设置为灰度图，所以这里为1，否则彩色图为3
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)

        self.input_shape = (img_rows, img_cols, img_channels)

        # 输出训练集、测试集的数量

        print(train_images.shape[0], 'train samples')
        print(test_images.shape[0], 'test samples')
        print(len(np.unique(train_labels)), 'train_labels')
        # 像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        # 将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.n_classes = len(np.unique(self.train_labels))



# 建立CNN模型
class CNN(tf.keras.Model):
    # 模型初始化
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu,  # 激活函数
        )
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])
        self.dropout2 = tf.keras.layers.Dropout(rate=0.5)

        self.flaten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.dropout3 = tf.keras.layers.Dropout(rate=0.25)
        self.dense2 = tf.keras.layers.Dense(units=10)



    # 模型输出
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        # x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        # x = self.dropout2(x)

        x = self.flaten(x)
        x = self.dense1(x)
        # x = self.dropout3(x)
        x = self.dense2(x)
        output = tf.nn.softmax(x)
        # print('conv1:', x.shape)
        # print('conv2:', x.shape)
        # print('pool1:', x.shape)
        # print('dropout1:', x.shape)
        # print('conv3:', x.shape)
        # print('conv4:', x.shape)
        # print('pool2:', x.shape)
        # print('dropout2:', x.shape)
        # print('flaten:', x.shape)
        # print('dense1:', x.shape)
        # print('dropout3:', x.shape)
        # print('dense2:', x.shape)
        return output

        # 识别人脸

    def face_predict(self, image):
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))

        # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率
        result = self.predict(image)
        # print('result:', result[0])
        # 返回类别预测结果
        return result[0]


if __name__ == '__main__':
    learning_rate = 0.001  # 学习率
    batch = 32  # batch数
    EPOCHS = 20  # 学习轮数

    # 在实例化 Dataset 对象时传入输出单元的数量 ,数据都保存在这个文件夹下
    dataset = Dataset('./face_data/')
    dataset.load()
    # n_classes = dataset.n_classes

    # 实例化 CNN 对象并传入输出单元的数量
    model = CNN()


    # model.build(input_shape=(None, 64, 64, 3))  # 指定输入的形状
    # model.summary()  # 打印模型的概要信息
    #
    # # 创建一个临时的输入张量，用于调用模型的call方法并打印每一层的输出形状
    # temp_inputs = tf.keras.Input(shape=(64, 64, 3))
    # _ = model(temp_inputs)


    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 选择优化器

    sgd = SGD(learning_rate=learning_rate, decay=1e-6,
              momentum=0.9, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象

    datagen = ImageDataGenerator(
        featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
        samplewise_center=False,  # 是否使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
        samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
        zca_whitening=False,  # 是否对输入数据施以ZCA白化
        rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
        width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
        height_shift_range=0.2,  # 同上，只不过这里是垂直
        horizontal_flip=True,  # 是否进行随机水平翻转
        vertical_flip=True)  # 是否进行随机垂直翻转



    augmented_train_images = datagen.flow(dataset.train_images, dataset.train_labels, shuffle=300,batch_size=batch)
    augmented_test_images = datagen.flow(dataset.test_images, dataset.test_labels, shuffle=300,batch_size=batch)

    # train_ds = tf.data.Dataset.from_tensor_slices((dataset.train_images, dataset.train_labels)).shuffle(300).batch(
    #     batch)
    # test_ds = tf.data.Dataset.from_tensor_slices((dataset.test_images, dataset.test_labels)).shuffle(300).batch(
    #     batch)


    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    history = model.fit(augmented_train_images, epochs=20, validation_data=augmented_test_images,callbacks=[tensorboard_callback])

    model.save_weights('./model/face1')  # 保存权重模型 命名为face1


