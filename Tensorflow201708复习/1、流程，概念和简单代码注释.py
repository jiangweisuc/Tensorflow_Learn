# -*- coding: utf-8 -*-
"""
=================================================
File Name   ： 1、流程，概念和简单代码注释.py
Author      :  jiang
Create Date ： 17-7-20 下午5:02
Description :  
-------------------------------------------------
Change Activity:
               17-7-20 下午5:02
=================================================
"""
# 1.tensorflow的运行流程

# tensorflow的运行流程主要有2步，分别是构造模型和训练。

########################################################################################################################
# Tensor的意思是张量，不过按我的理解，其实就是指矩阵。也可以理解为tensorflow中矩阵的表示形式。
# Tensor的生成方式有很多种，最简单的就如

import tensorflow as tf # 在下面所有代码中，都去掉了这一行，默认已经导入
a = tf.zeros(shape=[1,2])

# 不过要注意，因为在训练开始前，所有的数据都是抽象的概念，也就是说，此时a只是表示这应该是个1*5的零矩阵，
# 而没有实际赋值，也没有分配空间，所以如果此时print,就会出现如下情况:
print(a)
#===>Tensor("zeros:0", shape=(1, 2), dtype=float32)

# 只有在训练过程开始后，才能获得a的实际值
sess = tf.InteractiveSession()
print(sess.run(a))
#===>[[ 0.  0.]]


########################################################################################################################
# Variable

# 故名思议，是变量的意思。一般用来表示图中的各计算参数，包括矩阵，向量等。例如，y=Relu(Wx+b)
# 这里W和b是我要用来训练的参数，那么此时这两个值就可以用Variable来表示。

# W = tf.Variable(tf.zeros(shape=[1,2]))
# 注意，此时W一样是一个抽象的概念，而且与Tensor不同，Variable必须初始化以后才有具体的值。

tensor = tf.zeros(shape=[1,2])
variable = tf.Variable(tensor) #Variable必须初始化
sess = tf.InteractiveSession()
# print(sess.run(variable))  # 会报错： Attempting to use uninitialized value Variable
sess.run(tf.initialize_all_variables()) # 对variable进行初始化
print(sess.run(variable))
#===>[[ 0.  0.]]

########################################################################################################################
# placeholder
# 又叫占位符，同样是一个抽象的概念。用于表示输入输出数据的格式。告诉系统：这里有一个值/向量/矩阵，现在我没法给你具体数值，
# 不过我正式运行的时候会补上的！例如上式中的x和y。因为没有具体数值，所以只要指定尺寸即可

x = tf.placeholder(tf.float32, shape=[1, 5], name='input')
y = tf.placeholder(tf.float32, shape=[None, 5], name='input')
# 上面有两种形式，第一种x，表示输入是一个[1,5]的横向量。
# 而第二种形式，表示输入是一个[?,5]的矩阵。

########################################################################################################################
# Session

# session，也就是会话。我的理解是，session是抽象模型的实现者。为什么之前的代码多处要用到session？
# 因为模型是抽象的嘛，只有实现了模型以后，才能够得到具体的值。同样，具体的参数训练，预测，甚至变量的实际值查询，
# 都要用到session,看后面就知道了



#########################################################################################################################
# 1.2 模型构建

# 这里我们使用官方tutorial中的mnist数据集的分类代码，公式可以写作
# z=Wx+b
# a=softmax(z)



"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.flags #flags里面已经出初始化了 FLAGS = _FlagValues()
flags.DEFINE_string('data_dir', '/home/jiang/PycharmProjects/TensorflowLearn/MNIST_data', 'Directory for storing data') # 把数据放在/tmp/data文件夹中
FLAGS = flags.FLAGS

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集


# 建立抽象模型
x = tf.placeholder(tf.float32, [None, 784]) # 输入占位符
y = tf.placeholder(tf.float32, [None, 10])  # 输出占位符（预期输出）
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
a = tf.nn.softmax(tf.matmul(x, W) + b)      # a表示模型的实际输出

# 定义损失函数和训练方法
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))  # 损失函数为交叉熵
optimizer = tf.train.GradientDescentOptimizer(0.5) # 优化器：梯度下降法，学习速率为0.5
train = optimizer.minimize(cross_entropy) # 训练目标：最小化损失函数

# 可以看到这样以来，模型中的所有元素(图结构，损失函数，下降方法和训练目标)都已经包括在train里面。
# 我们可以把train叫做训练模型。那么我们还需要模型评价模块
# Test trained model
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 上述两行代码，tf.argmax表示找到最大值的位置(也就是预测的分类和实际的分类)，然后看看他们是否一致，
# 是就返回true,不是就返回false,这样得到一个boolean数组。tf.cast将boolean数组转成int数组，最后求平均值，
# 得到分类的准确率(怎么样，是不是很巧妙)

# Train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
        if(i%100==0):
            print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))