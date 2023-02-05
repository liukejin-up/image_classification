# 图像分类：水果分类
# 数据集说明：
# 总共1036水果图片，共5个类别（苹果288张、香蕉275张、葡萄216张、橙子276张、梨251张）

# 1.数据预处理
import os
import json

name_dict = {"apple":0, "banana":1, "grape":2, "orange":3, "pear":4}   # 名称对应分类字典
data_root_path = "data/fruits/"    # 数据集目录
test_file_path = data_root_path + "test.list"    # 测试集文件路径
train_file_path = data_root_path + "train.list"   # 训练集文件路径
readme_file = data_root_path + "readme.json"      # 样本数据汇总文件
name_data_list = {}      # 用来记录每个类别多少张训练图片、测试图片

# 遍历目录、将图片路径存入字典，再由字典写入文件
def save_train_test_file(path, name):
    if name not in name_data_list:
        img_list = []
        img_list.append(path)                # 将图片添加到列表
        name_data_list[name] = img_list      # 将图片列表存入字典
    else:                                    # 某类水果已经在字典中
        name_data_list[name].append(path)    # 直接加入
dirs = os.listdir(data_root_path)       # 列出data/fruits/目录下的所有内容
for d in dirs:
    full_path = data_root_path + d       # 拼出完整的路径
    if os.path.isdir(full_path):         # 如果是目录，遍历目录中的图片
        imgs = os.listdir(full_path)
        for img in imgs:
            save_train_test_file(full_path + "/" + img, d)
    else:                                 # 如果是文件，就不做处理
        pass
# 划分训练集和测试集
with open(test_file_path, "w") as f:
    pass
with open(train_file_path, "w") as f:
    pass
# 遍历字典， 每10笔数据分出1笔到测试集中
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)
    print("%s: %d张" % (name, num))                     # 打印每类图片的张数
    for img in img_list:
        if i % 10 == 0:                                # 放测试集
            with open(test_file_path, "a") as f:
                line = "%s\t%d" % (img, name_dict[name])      # 拼出一行
                f.write(line)                                 # 写入 test.list中（格式： 图片路径       类别）
        else:
            with open(train_file_path, "a") as f:
                line = "%s\t%d" % (img, name_dict[name])
                f.write(line)
        i += 1
# 2.搭建神经网络、模型的训练和保存
import paddle
import paddle.fluid as fluid
import numpy
import sys
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

def train_mapper(sample):
    img, label = sample               # 一行样本由图片的路径和标记组成
    if not os.path.exists(img):
        print("图片不存在：", img)
    else:
        # 读取图片，并对图片做维度变化
        img = paddle.dataset.image.load_image(img)      # 读取图像
        # 对图片进行变换，修剪，输出（3，100,100）的矩阵
        img = paddle.dataset.image.simple_transform(im=img, resize_size=100, crop_size=100, is_color=True, is_train=True)
        img = img.flatten().astype("float32") / 255.0       # 图像归一化处理，将值压缩到0到1之间
        return img, label
# 自定义reader，从训练集读取数据，并交给train_mapper处理
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]          # strip函数去空格
            for line in lines:
                img_path, lab = line.strip().split("\t")
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), buffered_size) # mapper函数 reader 线程数 缓冲区大小
# 搭建神经网络
# 输入层 ——》 卷积-池化层-dropout ——》 卷积-池化层-dropout ——》 卷积-池化层-dropout ——》全连接层 ——》dropout ——》 全连接层
def convolution_neural_network(image, type_size):
    # 第一个卷积-池化层
    # 卷积核大小3*3， 数量为32（与输出通道数相同），池化层大小2*2，步长为2，激活函数为relu
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image, filter_size=3, num_filters=32, pool_size=2, pool_stride=2, act="relu")
    # dropout：丢弃学习，丢弃一些神经元的输出，防止过拟合（丢弃率0.5）
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)
    # 第二层卷积
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop, filter_size=3, num_filters=64, pool_size=2, pool_stride=2, act="relu")
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)
    # 第三层卷积
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop, filter_size=3, num_filters=64, pool_size=2, pool_stride=2, act="relu")
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)
    # 全连接层
    fc = fluid.layers.fc(input=drop, size=512, act="relu")
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    predict = fluid.layers.fc(input=drop, size=type_size, act="softmax")      # softmax得到样本在每个类别概率，总和为1，取得概率最大的类别

    return predict

# 准备数据执行训练
BATCH_SIZE = 32
trainer_reader = train_r(train_list=train_file_path)
train_reader = paddle.batch(paddle.reader.shuffle(reader=trainer_reader, buf_size=1200), batch_size=BATCH_SIZE)

# 训练时的输入数据（RGB三通道彩色图像）
image = fluid.layers.data(name="image", shape=[3, 100, 100], dtype="float32")
# 训练时期望的输出值（真实类别）
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
# 调用函数，创建卷积神经网络（类别数量为5）
predict = convolution_neural_network(image=image, type_size=5)
# 计算交叉熵
cost = fluid.layers.cross_entropy(input=predict, label=label)
# 计算损失值得平均值
avg_cost = fluid.layers.mean(cost)
# 计算预测准确率
accuracy = fluid.layers.accuracy(input=predict, label=label)
# 优化器：自适应梯度下降优化器（自动调整学习率，离目标点越近学习率越小）
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)
# 执行器
place = fluid.CUDAPlace(0)  # GPU上执行
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())        # 初始化执行参数
feeder = fluid.DataFeeder(feed_list=[image, label], place=place) # 喂入数据

for pass_id in range(10):
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        # 获取结果
        train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_cost, accuracy])
        if batch_id % 20 == 0:        # 每训练20次打印一次（%.f输出的不是0就是1）
            print("pass:%d, batch:%d, cost:%f, acc:%f" % (pass_id, batch_id, train_cost[0], train_acc[0]))
print("训练完成！")

# 保存模型
model_save_dir = "model/fruits/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
fluid.io.save_inference_model(dirname=model_save_dir, feeded_var_names=["image"], target_vars=[predict], executor=exe)
print("保存模型完成！")

# 3.模型加载、执行预测

from PIL import Image
# 预测在CPU上执行
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)

def load_image(path):
     # 读取图片，调整尺寸，做归一化处理
    img = paddle.dataset.image.load_and_transform(path, 100, 100, False).astype("float32")
    img = img / 255.0
    return img

infer_imgs = [] # 图像数据列表
test_img = "apple.png"  # 预测图像路径

infer_imgs.append(load_image(test_img))    # 加载图像数据，添加到列表
infer_imgs = numpy.array(infer_imgs)
# 加载模型
[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)

# 先显示原始图片
img = Image.open(test_img)
plt.imshow(img)
plt.show()

# 执行预测
# result为数组，包含每个类别的概率
results = infer_exe.run(infer_program, feed={feed_target_names[0]:infer_imgs}, fetch_list=fetch_targets)
print(results)
result = numpy.argmax(results[0]) # 获取最大值得索引
# 根据字典获取预测结果名称
for k, v in name_dict.items():
    if result == v:
        print("预测结果：", k)








