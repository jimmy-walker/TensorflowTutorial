# Convolutional Neural Networks

## Convolutional Layer
**CNN中最重要的构件就是卷积层，特点是非全连接，而只是前一层的一小矩形区域的连接**。
each neuron in the second convolutional layer is connected only to neurons located within a small rectangle in the first layer.

![](picture/convolutional layer.png)

### zero padding（零填充）

![](picture/zero padding.png)

![](picture/padding.png)

###stride（步幅）

**<u>注意到卷积层的计算，采用same方式的话，那其实就是从原图的顶点开始，以该点为中心点产生一个包围着的矩阵，生成一个新图的点，然后再按方向移动，除非中心点不在原图了，就不再移动。因此当stride=1时，往往原图和新图的形状是一样的，但是如果stride=2，则往往会缩小一倍尺寸。</u>**

![](picture/stride.png)

### 权重组就是filter（过滤器），又称为convolution kernels（卷积核），最终得到feature map（特征图）

![](picture/feature map.png)

### 卷积层实际由多组特征图组成，用3D表示（卷积层是卷积核在上一级输入层上通过逐一滑动窗口计算而得，准备说卷积核生成了卷积层）

**<u>每一层卷积层，由多组特征图构成。其中每一组特征图的输出值都是由该组特征图的计算权重与前一层的多组特征图的输出值相乘，再累加而得。</u>**

![](picture/equation.png)

![](picture/feature maps.png)

###具体卷积层计算过程

如下图，假设现有一个为 6×6×3的图片样本，使用 3×3×3的卷积核（filter）进行卷积操作。此时输入图片的 `channels` 为 3 ，而**卷积核中**的 `in_channels` 与 需要进行卷积操作的数据的 `channels` 一致（这里就是图片样本，为3）。

![cnn](picture/conv1.png)

接下来，进行卷积操作，卷积核中的27个数字与分别与样本对应相乘后，再进行求和，得到第一个结果。依次进行，最终得到 4×4的结果。

![单个卷积核](picture/conv2.png)

上面步骤完成后，由于只有一个卷积核，所以最终得到的结果为 4×4×1， `out_channels` 为 1 。

在实际应用中，都会使用多个卷积核。这里如果再加一个卷积核，就会得到 4×4×2的结果。

![多个卷积核](picture/conv3.png)

### 卷积层的尺寸计算

####**卷积层尺寸的计算原理**

- **输入矩阵**格式：四个维度，依次为：样本数、图像高度、图像宽度、图像通道数

- **输出矩阵**格式：与输出矩阵的维度顺序和含义相同，但是后三个维度（图像高度、图像宽度、图像通道数）的尺寸发生变化。

- **权重矩阵**（卷积核）格式：同样是四个维度，但维度的含义与上面两者都不同，为：卷积核高度、卷积核宽度、输入通道数、输出通道数（卷积核个数）

- **输入矩阵、权重矩阵、输出矩阵这三者之间的相互决定关系**

- - 卷积核的输入通道数（in depth）由输入矩阵的通道数所决定。（红色标注）
  - 输出矩阵的通道数（out depth）由卷积核的输出通道数所决定。（绿色标注）
  - 输出矩阵的高度和宽度（height, width）这两个维度的尺寸由输入矩阵、卷积核、扫描方式所共同决定。计算公式如下。（蓝色标注）

![ \begin{cases} height_{out} &= (height_{in} - height_{kernel} + 2 * padding) ~ / ~ stride + 1\\[2ex] width_{out} &= (width_{in} - width_{kernel} + 2 * padding) ~ / ~ stride + 1 \end{cases}](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bcases%7D+height_%7Bout%7D+%26%3D+%28height_%7Bin%7D+-+height_%7Bkernel%7D+%2B+2+%2A+padding%29+%7E+%2F+%7E+stride+%2B+1%5C%5C%5B2ex%5D+width_%7Bout%7D+%26%3D+%28width_%7Bin%7D+-+width_%7Bkernel%7D+%2B+2+%2A+padding%29+%7E+%2F+%7E+stride+%2B+1+%5Cend%7Bcases%7D)

> \* 注：以下计算演示均省略掉了 Bias ，严格来说其实每个卷积核都还有一个 Bias 参数。

#### **标准卷积计算举例**

> 以 AlexNet 模型的第一个卷积层为例，
> \- 输入图片的尺寸统一为 227 x 227 x 3 （高度 x 宽度 x 颜色通道数），
> \- 本层一共具有96个卷积核，
> \- 每个卷积核的尺寸都是 11 x 11 x 3。
> \- 已知 stride = 4， padding = 0，
> \- 假设 batch_size = 256，
> \- 则输出矩阵的高度/宽度为 (227 - 11) / 4 + 1 = 55

![ \begin{matrix} & \mathbf{Batch} & \mathbf{Height} && \mathbf{Width} && \mathbf{In~Depth} && \mathbf{Out~Depth}\\[2ex] \mathbf{Input} & \quad\quad 256 \quad\quad \times & \color{blue}{227} & \times & \color{blue}{227} & \times & \color{red}{3} \\[2ex] \mathbf{Kernel} &\quad\quad\quad\quad\quad & \color{blue}{11} & \times & \color{blue}{11} & \times & \color{red}{3} & \times & \color{green}{96} \\[2ex] \mathbf{Output} & \quad\quad 256 \quad\quad \times & \color{blue}{55} & \times & \color{blue}{55} &&& \times & \color{green}{96} \end{matrix}](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bmatrix%7D+%26+%5Cmathbf%7BBatch%7D+%26+%5Cmathbf%7BHeight%7D+%26%26+%5Cmathbf%7BWidth%7D+%26%26+%5Cmathbf%7BIn%7EDepth%7D+%26%26+%5Cmathbf%7BOut%7EDepth%7D%5C%5C%5B2ex%5D+%5Cmathbf%7BInput%7D+%26+%5Cquad%5Cquad+256+%5Cquad%5Cquad+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B227%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B227%7D+%26+%5Ctimes+%26+%5Ccolor%7Bred%7D%7B3%7D+%5C%5C%5B2ex%5D+%5Cmathbf%7BKernel%7D+%26%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad+%26+%5Ccolor%7Bblue%7D%7B11%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B11%7D+%26+%5Ctimes+%26+%5Ccolor%7Bred%7D%7B3%7D+%26+%5Ctimes+%26+%5Ccolor%7Bgreen%7D%7B96%7D+%5C%5C%5B2ex%5D+%5Cmathbf%7BOutput%7D+%26+%5Cquad%5Cquad+256+%5Cquad%5Cquad+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B55%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B55%7D+%26%26%26+%5Ctimes+%26+%5Ccolor%7Bgreen%7D%7B96%7D+%5Cend%7Bmatrix%7D)

#### **1 x 1 卷积计算举例**

> 后期 GoogLeNet、ResNet 等经典模型中普遍使用一个像素大小的卷积核作为降低参数复杂度的手段。
> 从下面的运算可以看到，其实 1 x 1 卷积没有什么神秘的，其作用就是将输入矩阵的通道数量缩减后输出（512 降为 32），并保持它在宽度和高度维度上的尺寸（227 x 227）。

![ \begin{matrix} & \mathbf{Batch} & \mathbf{Height} && \mathbf{Width} && \mathbf{In~Depth} && \mathbf{Out~Depth}\\[2ex] \mathbf{Input} & \quad\quad 256 \quad\quad \times & \color{blue}{227} & \times & \color{blue}{227} & \times & \color{red}{512} \\[2ex] \mathbf{Kernel} &\quad\quad\quad\quad\quad & \color{blue}{1} & \times & \color{blue}{1} & \times & \color{red}{512} & \times & \color{green}{32} \\[2ex] \mathbf{Output} & \quad\quad 256 \quad\quad \times & \color{blue}{227} & \times & \color{blue}{227} &&& \times & \color{green}{32} \end{matrix}](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bmatrix%7D+%26+%5Cmathbf%7BBatch%7D+%26+%5Cmathbf%7BHeight%7D+%26%26+%5Cmathbf%7BWidth%7D+%26%26+%5Cmathbf%7BIn%7EDepth%7D+%26%26+%5Cmathbf%7BOut%7EDepth%7D%5C%5C%5B2ex%5D+%5Cmathbf%7BInput%7D+%26+%5Cquad%5Cquad+256+%5Cquad%5Cquad+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B227%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B227%7D+%26+%5Ctimes+%26+%5Ccolor%7Bred%7D%7B512%7D+%5C%5C%5B2ex%5D+%5Cmathbf%7BKernel%7D+%26%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad+%26+%5Ccolor%7Bblue%7D%7B1%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B1%7D+%26+%5Ctimes+%26+%5Ccolor%7Bred%7D%7B512%7D+%26+%5Ctimes+%26+%5Ccolor%7Bgreen%7D%7B32%7D+%5C%5C%5B2ex%5D+%5Cmathbf%7BOutput%7D+%26+%5Cquad%5Cquad+256+%5Cquad%5Cquad+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B227%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B227%7D+%26%26%26+%5Ctimes+%26+%5Ccolor%7Bgreen%7D%7B32%7D+%5Cend%7Bmatrix%7D)

#### **全连接层计算举例**

> 实际上，全连接层也可以被视为是一种极端情况的卷积层，其卷积核尺寸就是输入矩阵尺寸，因此输出矩阵的高度和宽度尺寸都是1。

![ \begin{matrix} & \mathbf{Batch} & \mathbf{Height} && \mathbf{Width} && \mathbf{In~Depth} && \mathbf{Out~Depth}\\[2ex] \mathbf{Input} & \quad \quad 256 \quad \quad \times & \color{blue}{32} & \times & \color{blue}{32} & \times & \color{red}{512} \\[2ex] \mathbf{Kernel} &\quad\quad\quad\quad\quad & \color{blue}{32} & \times & \color{blue}{32} & \times & \color{red}{512} & \times & \color{green}{4096} \\[2ex] \mathbf{Output} & \quad \quad 256 \quad \quad \times & \color{blue}{1} & \times & \color{blue}{1} &&& \times & \color{green}{4096} \end{matrix}](https://www.zhihu.com/equation?tex=+%5Cbegin%7Bmatrix%7D+%26+%5Cmathbf%7BBatch%7D+%26+%5Cmathbf%7BHeight%7D+%26%26+%5Cmathbf%7BWidth%7D+%26%26+%5Cmathbf%7BIn%7EDepth%7D+%26%26+%5Cmathbf%7BOut%7EDepth%7D%5C%5C%5B2ex%5D+%5Cmathbf%7BInput%7D+%26+%5Cquad+%5Cquad+256+%5Cquad+%5Cquad+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B32%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B32%7D+%26+%5Ctimes+%26+%5Ccolor%7Bred%7D%7B512%7D+%5C%5C%5B2ex%5D+%5Cmathbf%7BKernel%7D+%26%5Cquad%5Cquad%5Cquad%5Cquad%5Cquad+%26+%5Ccolor%7Bblue%7D%7B32%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B32%7D+%26+%5Ctimes+%26+%5Ccolor%7Bred%7D%7B512%7D+%26+%5Ctimes+%26+%5Ccolor%7Bgreen%7D%7B4096%7D+%5C%5C%5B2ex%5D+%5Cmathbf%7BOutput%7D+%26+%5Cquad+%5Cquad+256+%5Cquad+%5Cquad+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B1%7D+%26+%5Ctimes+%26+%5Ccolor%7Bblue%7D%7B1%7D+%26%26%26+%5Ctimes+%26+%5Ccolor%7Bgreen%7D%7B4096%7D+%5Cend%7Bmatrix%7D)

总结下来，其实只需要认识到，虽然输入的每一张图像本身具有三个维度，但是对于卷积核来讲依然只是一个一维向量。卷积核做的，其实就是与感受野范围内的像素点进行点积（而不是矩阵乘法）。

#### **附：TensorFlow 中卷积层的简单实现**

```
def conv_layer(x, out_channel, k_size, stride, padding):
    in_channel = x.shape[3].value
    w = tf.Variable(tf.truncated_normal([k_size, k_size, in_channel, out_channel], mean=0, stddev=stddev))
    b = tf.Variable(tf.zeros(out_channel))
    y = tf.nn.conv2d(x, filter=w, strides=[1, stride, stride, 1], padding=padding)
    y = tf.nn.bias_add(y, b)
    y = tf.nn.relu(y)
    return x
```

- 输入 x：[batch, height, width, in_channel]
- 权重 w：[height, width, in_channel, out_channel]
- 输出 y：[batch, height, width, out_channel]

### 代码

注意参数的类型是`numpy.ndarray。`

**<u>参数的维度，其实类似索引方式，比如多加一个括号，`dataset = np.array([china, flower]`，那么在形状上多一个维度`(2, 427, 640, 3)`</u>**。

<u>Each input image is typically represented as a 3D tensor of shape [height, width, channels]</u>. 

<u>A mini-batch is represented as a 4D tensor of shape [mini-batch size, height, width, channels].</u> 

<u>The weights of a convolutional layer are represented as a 4D tensor of shape [fh, fw, fn, fn′]</u>.

<u>这里的fn表示的是当前层的特征图数；fn'表示的是前一层的特征图数</u>。

<u>而经过`tf.nn.conv2d`返回的张量与输入有同样的类型` [mini-batch size, height, width, channels]`，但是channel变为filter中当前层的特征图数。</u>

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32) #这里与公式中的w的顺序有些不同，其中channel是上一层的特征图数，而第四个参数是这一层的特征图数
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line
# channel一共有三个值，等于就是在0,1,2三个通道上，把固定位置变成1，就是变亮。
# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels)) #batch size仍是none
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")
#conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding="SAME")
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
    #output = sess.run(conv, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1], cmap="gray") # plot 1st image's 2nd feature map
#而经过tf.nn.conv2d返回的张量与输入有同样的类型[mini-batch size, height, width, channels]，但是channel变为filter中当前层的特征图数，即为2。
plt.show()
```

```python
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

plot_image(filters[:, :, 0, 0]) #可以看到filter0对于channel为0时的2D权值图
```

### 内存需求

如上所述，**<u>每一层卷积层，由多组特征图构成。其中每一组特征图的输出值都是由该组特征图的计算权重与前一层的多组特征图的输出值相乘，再累加而得。</u>**

因此存在几处需要的内存消耗：filter中的参数及bias参数；计算每层特征图输出值的乘法运算以及输出值的所占空间；多组instances造成的内存消耗；训练时需要保存所有的数据，以方便反向传播时调整。

## Pooling Layer

**CNN中第二个常见的构件就是池化层，与卷积层类似，区别在于并不采用过滤器的权值去乘积累积，只是对于过滤器范围内进行聚合，比如max或mean**。

**<u>注意到池化的计算，采用valid方式的话，那其实就是从原图的顶点开始，生成一个新图的点，然后再按方向移动，当没有数据了，就不再移动，导致新图的尺寸比较小。</u>**

![](picture/pooling layer.png)

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_sample_image

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
# filters[:, 3, :, 0] = 1  # vertical line
# filters[3, :, :, 1] = 1  # horizontal line

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")
# The ksize argument contains the kernel shape along all four dimensions of the input tensor: [batch size, height, width, channels]. 
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

# plt.imshow(output[0].astype(np.uint8))  # plot the output for the 1st image
# plt.show()


plot_color_image(dataset[0])
# save_fig("china_original")
plt.show()
    
plot_color_image(output[0])
# save_fig("china_max_pool")
plt.show()
```

## CNN Architectures

**<u>典型的cnn结构如下图所示</u>**：J<u>猜测a few convolutional layers表示的就是一组特征层；a pooling layer就是对每一个特征层，生成一个池化后的结果，前后数量并不变</u>。

Typical CNN architectures stack <u>a few convolutional layers</u> (each one generally followed by a ReLU layer), then <u>a pooling layer</u>, then another few convolutional layers (+ReLU), then another pooling layer, and so on. The image gets smaller and smaller as it progresses through the network, but it also typically gets deeper and deeper (i.e., with more feature maps) thanks to the convolutional layers. At the top of the stack, a regular feedforward neural network is added, composed of a few fully connected layers (+ReLUs), and the final layer outputs the prediction (e.g., a softmax layer that outputs estimated class probabilities).

![](picture/cnn architecture.png)

##CNN结构的变体

LeNet-5、AlexNet、GoogleNet、ResNet

###关于MNIST的代码（这是作者自己搭的一个神经网络）

CPU可以运行

**<u>minibatch输入到神经网络的张量形状为：[mini-batch size, height, width, channels]</u>**，一起输入的，而不是一个个样本分开输入，所以需要进行reshape下。

**<u>全连接层的输入的张量形状为:[mini-batch size, input]</u>**，所以需要进行reshape下。

```python
import tensorflow as tf
height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X") #none表示mini batch中的样本个数
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels]) #最外围的-1就表示样本个数
    y = tf.placeholder(tf.int32, shape=[None], name="y")
#两个卷积层+Relu激活函数
#因为要求输入的数据格式为： [mini-batch size, height, width, channels]
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")
#一个池化层
with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7]) #为了便于下一步的全连接，需要将其flat变平成为1行，多少行代表多少样本个数。把第二个卷积层中的64个特征图，经过池化后得到64个池化层，而7*7则是每个池化层的尺寸。
#经过第一层的卷积层，还是28*28，经过第二层的卷积层，由于stride=2，same，所以变成了14*14，经过池化层，因为stide=2，valid，所以变成了7*7。
#全连接网络
with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
#这里的n_fc1与之前的64个特征图数量没有关系
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
#把输出的logits进行softmax得到概率值，或者直接像下面一样直接用函数计算与y的交叉熵的值
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy) #计算该批次样本的损失值
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss) #进行自动微分进行更新参数

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)#把输出的值与输入的值对比，是否最大值的序号等于y。比如[0.1,0.6,0.4]与[1]
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #将所有样本的结果进行平均，这里是将true, false变成1,0

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./my_mnist_model")
```

```linux
0 Train accuracy: 0.98 Test accuracy: 0.9751
1 Train accuracy: 0.96 Test accuracy: 0.9837
2 Train accuracy: 0.99 Test accuracy: 0.984
3 Train accuracy: 1.0 Test accuracy: 0.9864
4 Train accuracy: 1.0 Test accuracy: 0.9881
5 Train accuracy: 0.98 Test accuracy: 0.9884
6 Train accuracy: 1.0 Test accuracy: 0.9893
7 Train accuracy: 0.99 Test accuracy: 0.9885
8 Train accuracy: 1.0 Test accuracy: 0.9881
9 Train accuracy: 1.0 Test accuracy: 0.9879
```

###VGC16
```
下面算一下每一层的像素值计算： 
输入：224*224*3 
1. conv3 - 64（卷积核的数量）：kernel size:3 stride:1 pad:1 
像素：（224-3+2*1）/1+1=224 224*224*64 
参数： （3*3*3）*64 =1728 
2. conv3 - 64：kernel size:3 stride:1 pad:1 
像素： （224-3+1*2）/1+1=224 224*224*64 
参数： （3*3*64）*64 =36864 
3. pool2 kernel size:2 stride:2 pad:0 
像素： （224-2）/2 = 112 112*112*64 
参数： 0 
4.conv3-128:kernel size:3 stride:1 pad:1 
像素： （112-3+2*1）/1+1 = 112 112*112*128 
参数： （3*3*64）*128 =73728 
5.conv3-128:kernel size:3 stride:1 pad:1 
像素： （112-3+2*1）/1+1 = 112 112*112*128 
参数： （3*3*128）*128 =147456 
6.pool2: kernel size:2 stride:2 pad:0 
像素： （112-2）/2+1=56 56*56*128 
参数：0 
7.conv3-256: kernel size:3 stride:1 pad:1 
像素： （56-3+2*1）/1+1=56 56*56*256 
参数：（3*3*128）*256=294912 
8.conv3-256: kernel size:3 stride:1 pad:1 
像素： （56-3+2*1）/1+1=56 56*56*256 
参数：（3*3*256）*256=589824 
9.conv3-256: kernel size:3 stride:1 pad:1 
像素： （56-3+2*1）/1+1=56 56*56*256 
参数：（3*3*256）*256=589824 
10.pool2: kernel size:2 stride:2 pad:0 
像素：（56 - 2）/2+1=28 28*28*256 
参数：0 
11. conv3-512:kernel size:3 stride:1 pad:1 
像素：（28-3+2*1）/1+1=28 28*28*512 
参数：（3*3*256）*512 = 1179648 
12. conv3-512:kernel size:3 stride:1 pad:1 
像素：（28-3+2*1）/1+1=28 28*28*512 
参数：（3*3*512）*512 = 2359296 
13. conv3-512:kernel size:3 stride:1 pad:1 
像素：（28-3+2*1）/1+1=28 28*28*512 
参数：（3*3*512）*512 = 2359296 
14.pool2: kernel size:2 stride:2 pad:0 
像素：（28-2）/2+1=14 14*14*512 
参数： 0 
15. conv3-512:kernel size:3 stride:1 pad:1 
像素：（14-3+2*1）/1+1=14 14*14*512 
参数：（3*3*512）*512 = 2359296 
16. conv3-512:kernel size:3 stride:1 pad:1 
像素：（14-3+2*1）/1+1=14 14*14*512 
参数：（3*3*512）*512 = 2359296 
17. conv3-512:kernel size:3 stride:1 pad:1 
像素：（14-3+2*1）/1+1=14 14*14*512 
参数：（3*3*512）*512 = 2359296 
18.pool2:kernel size:2 stride:2 pad:0 
像素：（14-2）/2+1=7 7*7*512 
参数：0 
19.FC: 4096 neurons 
像素：1*1*4096 
参数：7*7*512*4096 = 102760448 
20.FC: 4096 neurons 
像素：1*1*4096 
参数：4096*4096 = 16777216 
21.FC：1000 neurons 
像素：1*1*1000 
参数：4096*1000=4096000
```

## Reference

- [CNN计算](https://blog.csdn.net/sscc_learning/article/details/79814146)
- [CNN尺寸](https://zhuanlan.zhihu.com/p/29119239)
- [VGC16尺寸计算](https://blog.csdn.net/zhangwei15hh/article/details/78417789)

