# Tensorflow

**<u>所有的数学计算</u>**：先使用得到损失值$C$对输入的导数，这里的输入可以看做是下面的$y_4$，其实下面可以当做是计算$\frac{\partial C}{\partial y_4}$，涉及到交叉熵损失和softmax进行求导，只不过举了逻辑回归的例子。

$\frac{\partial}{\partial{\theta_j}}l(\theta)=\sum_{i=1}^{m}(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}_j$

然后逐步返回计算网络中C对各个参数的导数，<u>**注意，只对一个参数，就看这一路，不用管其他支路**</u>。
$$
\begin{align}
&\frac{\partial C}{\partial b_1}=\frac{\partial C}{\partial y_4}\frac{\partial y_4}{\partial z_4}\frac{\partial z_4}{\partial x_4}\frac{\partial x_4}{\partial z_3}\frac{\partial z_3}{\partial x_3}\frac{\partial x_3}{\partial z_2}\frac{\partial z_2}{\partial x_2}\frac{\partial x_2}{\partial z_1}\frac{\partial z_1}{\partial b_1}\\
&=\frac{\partial C}{\partial y_4}\sigma'\left(z_4\right)w_4\sigma'\left(z_3\right)w_3\sigma'\left(z_2\right)w_2\sigma'\left(z_1\right)
\end{align}
$$
再进行移动，这就是反向传播的精髓，这里是举了逻辑回归的例子，**注意最后的公式中代入的值，其实就是训练数据输入和输出的值，如下面的$x^{(i)}$和$y^{(i)}$。**

$\theta_j=\theta_j+\alpha\sum_{i=1}^{m}(y^{(i)}-h_\theta(x^{(i)}))x^{(i)}_j=\theta_j+\alpha\sum_{i=1}^{m}(y^{(i)}-\frac{1}{1+e^{-{\theta^T}{x^{(i)}}}})x^{(i)}_j$



##安装
```linux
conda create -n tf35 python=3.5
source activate tf35
pip install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
pip install --upgrade --ignore-installed tensorflow
```

##概念

**<u>在图的构建阶段，设置初始值，然后设置梯度更新方法，最后设置`training_op`。</u>**
**<u>然后在执行阶段，利用循环执行`sess.run(training_op)`就可以不断更新迭代了，此时不用考虑图中的初始值。</u>**
**<u>最后用`best_theta.eval`把参数取出来即可。此外还有一个measure model的指标比如mse，因为这个指标是对所有样本而言的，常和tf.reduce_mean搭配使用</u>**

### 构造阶段和执行阶段
A TensorFlow program is typically split into two parts: the first part builds a computation graph (this is called the construction phase), and the second part runs it (this is the execution phase).

`Tensor`和`Operation`都是`Graph`中的对象。`Operation`是图的节点，`Tensor`是图的边上流动的数据。 

可以通过 graph.get_operation_by_name，或者 x.op 的方式获得该 OP 的详细信息

`tf.get_default_graph().get_operations()`可以得到图内所有的op。

```python
tf.reset_default_graph()
A = tf.placeholder(tf.float32, shape=(None, 3))
B = 5
# B = tf.constant(5) #注意此时表示标量5，而上面那种加法会让数组每个量都加上5
C = A + B
tf.get_default_graph().get_operations()
output:
[<tf.Operation 'Placeholder' type=Placeholder>,
 <tf.Operation 'add/y' type=Const>,
 <tf.Operation 'add' type=Add>]
```

### 构造阶段：常量，变量，计算都会生成op，也就是图中的node。
**<u>所有的常量、变量和计算的操作都是  OP；变量包含的 OP 更加复杂。</u>**

An `Operation` is a node in a TensorFlow `Graph` that takes zero or more `Tensor` objects as input, and produces zero or more `Tensor` objects as output. Objects of type `Operation` are created by calling a Python op constructor (such as `tf.matmul`) or `tf.Graph.create_op`.

For example `c = tf.matmul(a, b)` creates an `Operation` of type "MatMul" that takes tensors `a`and `b` as input, and produces `c` as output.

After the graph has been launched in a session, an `Operation` can be executed by passing it to`tf.Session.run`. `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.

###张量就是多维数组
每一个op使用0个或多个Tensor, 执行一些计算，并生成0个或多个Tensor. 一个tensor就是一种多维数组。

###placeholder node占位符节点也是一种op，<u>先占位后赋值</u>
`tf.placeholder()` 操作(operation)允许你定义一种必须提供值的 tensor。

<u>To create a placeholder node, you must call the placeholder() function and specify the output tensor’s data type. Optionally, you can also specify its shape, if you want to enforce it. If you specify None for a dimension, it means “any size.”</u>

```linux
>>> A = tf.placeholder(tf.float32, shape=(None, 3))
>>> B = A + 5
>>> with tf.Session() as sess:
... B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
... B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
...
>>> print(B_val_1)
[[ 6. 7. 8.]]
>>> print(B_val_2)
[[ 9. 10. 11.]
[ 12. 13. 14.]]
```

```python
X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
sess.run(training_op,feed_dict={X:X_batch,y:y_batch}) #每次的训练数据X和y都不一样，所以需要用placeholder进行先占位，后赋值
```

###执行阶段

`f.eval()` is equivalent to calling `tf.get_default_session().run(f)`或`sess.run(f)`。

##常用函数

### `functools.partial`

当函数的参数个数太多，需要简化时，使用`functools.partial`可以创建一个新的函数，这个新函数可以固定住原函数的部分参数，从而在调用时更简单。

Python的`functools`模块提供了很多有用的功能，其中一个就是偏函数（Partial function）。要注意，这里的偏函数和数学意义上的偏函数不一样。

在介绍函数参数的时候，我们讲到，通过设定参数的默认值，可以降低函数调用的难度。而偏函数也可以做到这一点。举例如下：

`int()`函数可以把字符串转换为整数，当仅传入字符串时，`int()`函数默认按十进制转换：

```
>>> int('12345')
12345

```

但`int()`函数还提供额外的`base`参数，默认值为`10`。如果传入`base`参数，就可以做N进制的转换：

```
>>> int('12345', base=8)
5349
>>> int('12345', 16)
74565

```

假设要转换大量的二进制字符串，每次都传入`int(x, base=2)`非常麻烦，于是，我们想到，可以定义一个`int2()`的函数，默认把`base=2`传进去：

```
def int2(x, base=2):
    return int(x, base)

```

这样，我们转换二进制就非常方便了：

```
>>> int2('1000000')
64
>>> int2('1010101')
85

```

`functools.partial`就是帮助我们创建一个偏函数的，不需要我们自己定义`int2()`，可以直接使用下面的代码创建一个新的函数`int2`：

```
>>> import functools
>>> int2 = functools.partial(int, base=2)
>>> int2('1000000')
64
>>> int2('1010101')
85

```

所以，简单总结`functools.partial`的作用就是，把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。

### `np.random.randint`
low、high、size三个参数。默认high是None,如果只有low，那范围就是[0,low)。如果有high，范围就是[low,high)。
```linux
>>> np.random.randint(2, size=10)
array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
```

### `np.ceil`

ceil [siːl] 向正无穷取整 朝正无穷大方向取整
```linux
>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.ceil(a)
array([-1., -1., -0.,  1.,  2.,  2.,  2.])
```

### `np.c_`
将切片对象沿第二个轴（按列）转换为连接。
```python
np.c_[np.array([1,2,3]), np.array([4,5,6])]  
Out[96]:   
array([[1, 4],  
       [2, 5],  
       [3, 6]])  
  
np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]  
Out[97]: array([[1, 2, 3, 0, 0, 4, 5, 6]])  
```

### `np.reshape`
新数组的shape属性应该要与原来数组的一致，即新数组元素数量与原数组元素数量要相等。**<u>一个参数为-1时，那么reshape函数会根据另一个参数的维度计算出数组的另外一个shape属性值。</u>**
```python
>>> z = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]])
>>> print(z)
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]
>>> print(z.shape)
(4, 4)
>>> print(z.reshape(-1)) #因为另一个参数为0，所以等于就是这个参数代表了全部数量
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
>>> print(z.reshape(-1,1))  #我们不知道z的shape属性是多少，
                            #但是想让z变成只有一列，行数不知道多少，
                            #通过`z.reshape(-1,1)`，Numpy自动计算出有16行，
                            #新的数组shape属性为(16, 1)，与原来的(4, 4)配套。
[[ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]
 [12]
 [13]
 [14]
 [15]
 [16]]
>>> print(z.reshape(2,-1))
[[ 1  2  3  4  5  6  7  8]
 [ 9 10 11 12 13 14 15 16]]
```

### `np.argmax`
Returns the indices of the maximum values along an axis.

注意axis是与平时的0,1对应的x，y轴垂直的

```
>>> a = np.arange(6).reshape(2,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])
```

###`np.random.permutation`

Randomly permute a sequence, or return a permuted range.

```
>>> np.random.permutation(10)
array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

>>> np.random.permutation([1, 4, 9, 12, 15])
array([15,  1,  9,  4, 12])
```

###`np.array_split`
Split an array into multiple sub-arrays.
```python
>>> x = np.arange(8.0)
>>> np.array_split(x, 3)
    [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]
```
### `tf.matmul`
Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

### `tf.matrix_inverse`
Computes the inverse of one or more square invertible matrices or their

### `tf.reduce_mean`
Computes the mean of elements across dimensions of a tensor.

### `tf.assign`
变量更新，将值赋给变量。
```python
training_op = tf.assign(theta, theta - learning_rate * gradients)

import tensorflow as tf

# We define a Variable
x = tf.Variable(0, dtype=tf.int32)

# We use a simple assign operation
assign_op = tf.assign(x, x + 1)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for i in range(5):
    print('x:', sess.run(x))
    sess.run(assign_op)
    
# outputs:
# x: 0
# x: 1
# x: 2
# x: 3
# x: 4
```

###`tf.maximum`



### `tf.add_n`

### `tf.truncated_normal`

###`tf.nn.sparse_softmax_cross_entropy_with_logits`
该函数是将softmax和cross_entropy放在一起计算，对于分类问题而言，最后一般都是一个单层全连接神经网络，比如softmax分类器居多，对这个函数而言，tensorflow神经网络中是没有softmax层，而是在这个函数中进行softmax函数的计算。这里的logits通常是最后的全连接层的输出结果，labels是具体哪一类的标签，这个函数是直接使用标签数据的，而不是采用one-hot编码形式。

![](picture/tf.nn.sparse_softmax_cross_entropy_with_logits.png)

###`tf.nn.in_top_k`
tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。target就是实际样本类别的标签，大小就是样本数量的个数。K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。

例如：
```python
import tensorflow as tf;  

A = [[0.8,0.6,0.3], [0.1,0.6,0.4]]  
B = [1, 1]  
out = tf.nn.in_top_k(A, B, 1)  
with tf.Session() as sess:  
	sess.run(tf.initialize_all_variables())  
	print sess.run(out)  
```

输出：
[False  True]

解释：因为A张量里面的第一个元素的最大值的标签是0，第二个元素的最大值的标签是1.。但是实际的确是1和1.所以输出就是False 和True。如果把K改成2，那么第一个元素的前面2个最大的元素的位置是0，1，第二个的就是1，2。实际结果是1和1。包含在里面，所以输出结果就是True 和True.如果K的值大于张量A的列，那就表示输出结果都是true

###`tf.layers.dense`

构建一个密集的图层。以神经元数量和激活函数作为参数。

This layer implements the operation: `outputs = activation(inputs.kernel + bias)` Where `activation` is the activation function passed as the `activation` argument (if not `None`), `kernel`is a weights matrix created by the layer, and `bias` is a bias vector created by the layer (only if `use_bias` is `True`).

### `tf.placeholder`和`tf.placeholder_with_default`

`tf.placeholder`插入一个坐等 **被feed_dict** 的 `tensor占位符` 。

> tf.placeholder (dtype, shape=None, name=None)

```
import tensorflow as tf

x = tf.placeholder(dtype=tf.int32)
print x
with tf.Session() as sess:
    print sess.run(x, feed_dict={x:[10, 20]})123456
```

```
Tensor("Placeholder:0", dtype=int32)

[10 20]
```

`tf.placeholder_with_default`与 `tf.placeholder` 不同的是，这里如果 未 被feed_dict，并不会 打印报错，而是打印出 默认数据。

> tf.placeholder_with_default (input, shape, name=None)

```
import tensorflow as tf

x = tf.placeholder_with_default(input=[3, 3], shape=(2,))
print x
with tf.Session() as sess:
    print sess.run(x)
    print sess.run(x, feed_dict={x:[10, 20]})
# 占位符 未 被feed_dict，打印默认输出
[3 3]

# 占位符 被feed_dict 了，打印 所被feed_dict 的内容
[10 20]
```

### `tf.layers.batch_normalization`

在网络中加入batch normalization的功能。

注意在tf.layers.dense之后使用，因为需要使用`tf.layers.batch_normalization`函数归一化层的输出，传递归一化后的值给激活函数。

### `tf.nn.elu`

Computes exponential linear: `exp(features) - 1` if < 0,`features` otherwise.

### `tf.control_dependencies`

**<u>执行某些`op,tensor`之前，某些`op,tensor`得首先被运行。</u>**

```python
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss)
sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})
```

**Note:** when training, the moving_mean and moving_variance need to be updated. By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they need to be added as a dependency to the `train_op`. For example:

```python
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```

###`tf.get_collection`
从一个集合中取出全部变量，返回的是一个列表。
The list of values in the collection with the given name, or an empty list if no value has been added to that collection. The list contains the values in the order under which they were collected.

### `tf.clip_by_value`

tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。

###`tf.train.exponential_decay`

Applies exponential decay to the learning rate.

When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies an exponential decay function to a provided initial learning rate. It requires a `global_step` value to compute the decayed learning rate. You can just pass a TensorFlow variable that you increment at each training step.

接着调用minimize函数，则会将global_step自动加一: Optional `Variable` to increment by one after the variables have been updated.

###`tf.add_n`

Adds all input tensors element-wise. 就是将tensor进行一一相加。

```python
import tensorflow as tf;  
import numpy as np;  
  
input1 = tf.constant([1.0, 2.0, 3.0])  
input2 = tf.Variable(tf.random_uniform([3]))  
output = tf.add_n([input1, input2])  
  
with tf.Session() as sess:  
    sess.run(tf.initialize_all_variables())  
    print sess.run(input1 + input2)  
    print sess.run(output)  

输出：[ 1.68921876  2.73008633  3.04061747]
[ 1.68921876  2.73008633  3.04061747]
```

###`tf.layers.dropout`

**<u>J在构建阶段来看，就像是多加了一层，套在需要dropout的层上。</u>**

Applies Dropout to the input.

Dropout consists in randomly setting a fraction `rate` of input units to 0 at each update during training time, which helps prevent overfitting. The units that are kept are scaled by `1 / (1 - rate)`, so that their sum is unchanged at training time and inference time.

## 数据标准化

### Standard Scaler

The `StandardScaler` assumes your data is normally distributed within each feature and will scale them such that the distribution is now centred around 0, with a standard deviation of 1.

The mean and standard deviation are calculated for the feature and then the feature is scaled based on:

$\dfrac{x_i – mean(x)}{stdev(x)}$

**<u>If data is not normally distributed, this is not the best scaler to use.</u>**

但是不是根据值得到百分位数，所以可以转换。

**<u>1) Just keep in mind that the distribution of z-scores will mirror the original distribution. Since your distribution is skewed left, the distribution of z-scores will be skewed left.</u>**

**<u> 2)Regardless of whether it's normal or not.The issue of normality or not is what it will be used for. If you're going to compute their income percentile off a Gaussian distribution table than it's invalid. If you're going to do some other transformation with the z-score, then it's fine.</u>**

Let’s take a look at it in action:

In [1]:

```
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
matplotlib.style.use('ggplot')
```

In [2]:

```
np.random.seed(1)
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 10000),
    'x2': np.random.normal(5, 3, 10000),
    'x3': np.random.normal(-5, 5, 10000)
})

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['x1', 'x2', 'x3'])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(df['x1'], ax=ax1)
sns.kdeplot(df['x2'], ax=ax1)
sns.kdeplot(df['x3'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['x1'], ax=ax2)
sns.kdeplot(scaled_df['x2'], ax=ax2)
sns.kdeplot(scaled_df['x3'], ax=ax2)
plt.show()
```

![](picture/标准化.png)

All features are now on the same scale relative to one another.

### fit_transform区别：对训练数据fit得到参数均值和方差，从而对测试数据transform

To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation.

$x' = \frac{x-\mu}{\sigma}$

You do that on the training set of data. But then you have to apply the same transformation to your testing set (e.g. in cross-validation), or to newly obtained examples before forecast. But you have to use the same two parameters $\mu$ and $\sigma$ (values) that you used for centering the training set.

Hence, every sklearn's transform's `fit()` just calculates the parameters (e.g. $\mu$ and $\sigma$ in case of StandardScaler and saves them as an internal objects state. Afterwards, you can call its `transform()` method to apply the transformation to a particular set of examples.

`fit_transform()` joins these two steps and is used for the initial fitting of parameters on the training set xx, but it also returns a transformed x′x′. Internally, it just calls first `fit()` and then `transform()`on the same data.

## 自动微分

`tf.gradients`：`gradients = tf.gradients(mse, [theta])[0] `

The gradients() function takes an op (in this case mse) and a list of variables (in this case just theta), and it creates a list of ops (one per variable) to compute the gradients of the op with regards to each variable. So the gradients node will compute the gradient vector of the MSE with regards to theta.

```python
#Manually Computing the Gradients
n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error) #calculate the gradient
#注意这里实际上的结果应该是error*X。但是由于我们要得到的gradients是n+1行1列，因此需要根据结果来调整相乘的方法，这就是加入转置的原因！！！！
training_op = tf.assign(theta, theta - learning_rate * gradients) #assign the new value to variable
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs): #一共1000次迭代，每一次都要全部计算所有样本的误差从而更新梯度
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
#autodiff
gradients = tf.gradients(mse, [theta])[0] #calculate the gradient
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #calculate the gradient
training_op = optimizer.minimize(mse)  #assign the new value to variable
```

## 优化算法

深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为**Batch gradient descent，批梯度下降**。

另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为**随机梯度下降，stochastic gradient descent**。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。

为了克服两种方法的缺点，现在一般采用的是一种折中手段，**mini-batch gradient decent，小批的梯度下降**，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

tf.reset_default_graph()
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = StandardScaler().fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias) #standard the data, m * n+1
# 而y是m行1列
""" mini-batch GD """
n_epochs = 1000
learning_rate = 0.01
X = tf.placeholder(dtype=tf.float32, shape=(None,n +1), name="X") #因为不知道多少行，多少个样本，所以设置为None
y = tf.placeholder(dtype=tf.float32, shape=(None,1), name ="y")

theta = tf.Variable(tf.random_uniform([n+1 ,1],-1, 1,seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

"""重写了n_epochs的大小，因为批量大小100，所以10轮就可以训练完成"""
n_epochs = 10 #一共迭代10次，每次都进行n_batches批次的更新
batch_size = 100 #即不进行全量计算梯度了，只对该batch进行计算
"""n_batches数量的小批量样本"""
n_batches = int(np.ceil(m/batch_size)) #得到batch的总批次
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index) #batch_index是记录批次的，也用来记录random的seed
    indices = np.random.randint(m,size=batch_size) #从[0, m)中取batch_size个序号
    X_batch = scaled_housing_data_plus_bias[indices] #根据indices取数据
    y_batch = housing.target.reshape(-1,1)[indices]
    return X_batch, y_batch

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    """一共10轮"""
    for epoch in range(n_epochs):
        """每轮需要n_batches才能完成"""
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch}) #每次的训练数据X和y都不一样，所以需要用placeholder进行先占位，后赋值
    best_theta = theta.eval()
    save_path = saver.save(sess,"/home/jimmy/Desktop/tf/my_model_final.ckpt")
print("best_theta : \n", best_theta)
```

## 模型保存和恢复
### 模型保存
**<u>等于是先在构建阶段的最后定义一个saver保存恢复器，然后在需要保存的地方调用保存器保存或恢复。</u>**
Just create a Saver node at the end of the construction phase (after all variable nodes are created); then, in the execution phase, just call its save() method whenever you want to save the model, passing it the session and path of the checkpoint file
```python
#常放在init = tf.global_variables_initializer()之后
saver = tf.train.Saver()
#可以放在任何地方，单没有必要每一次更新就保存，比如print打印状态的位置
save_path = saver.save(sess,"/home/jimmy/Desktop/tf/my_model_final.ckpt")
```

可以存储指定的变量并赋予不同的名字: 

```python
saver = tf.train.Saver({"weights": theta})
```

### 模型恢复

如果是想继续运行模型，那么就把saver也添加进去，然后再如下恢复模型，继续进行迭代更新。

```python
with tf.Session() as sess:
    saver.restore(sess,"/home/jimmy/Desktop/tf/my_model_final.ckpt") #这里只是恢复，并没有运行该模型
    best_theta_restored = theta.eval()
print(best_theta_restored)
```

![](picture/保存.png)

##可视化
见tensorboard即可

在定义计算图的时候，在适当的位置加上一些summary 操作 。
```python
mse_summary = tf.summary.scalar('MSE', mse)
```

在没有运行的时候这些操作是不会执行任何东西的，仅仅是定义了一下而已。在运行（开始训练）的时候，我们需要通过 tf.summary.FileWriter() 指定一个目录来告诉程序把产生的文件放到哪。然后在运行的时候使用 add_summary() 来将某一步的 summary 数据记录到文件中。

```python
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```

```python
for batch_index in range(n_batches):
    X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
    if batch_index % 10 == 0:
        summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
        step = epoch * n_batches + batch_index
        file_writer.add_summary(summary_str, step)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
```

最后关闭filewriter

```python
file_writer.close()
```

打开可视化界面。
```linux
tensorboard --logdir tf_logs/
```


**<u>J若在程序最后只有这两句，那么就是会只保存图。J我认为这是一个挺好的方法</u>**
```python
file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close()
```
```linux
tensorboard --logdir logs/
```

## 命名空间
```python
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error),name="mse")
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
```

## 模块化

**<u>减少代码，用函数（没啥特别的，函数模块化而已）。可以利用namescope让图更简洁。</u>**

```python
"""比较重复的写法"""
reset_graph()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.add(tf.matmul(X, w1), b1, name="relu1")
relu2 = tf.add(tf.matmul(X, w2), b2, name="relu2")

output = tf.add(relu1,relu2, name="output")

"""用函数定义"""
reset_graph()

def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())

"""用namescope是更好的方式"""
#reset_graph()
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name ="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
"""这里tensorflow会自动的检测节点名是否存在，已经存在的话，创捷节点名的时候会自动加上后缀_1（第一个重复节点名）"""
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus,name="output")
file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())
```

##共享变量

**variable_scope用于变量，name_scope用于op。**

Variables created using get_variable() are always named using the name of their variable_scope as a prefix (e.g., "relu/thres
hold"), but for all other nodes (including variables created with tf.Variable()) **<u>the variable scope acts like a new name scope.</u>**

**<u>也就是说variable_scope在图上也像name_scope一样显示为一个命名空间图标，里面的名字都是以variable_scope开头。</u>**

**<u>get_variable生成一个共享变量。在需要使用的地方，加上reuse或reuse_variables进行使用。</u>**注意一旦reuse设置为true了，那就不能更改回来了，子空间也会继承。

```python
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),initializer=tf.constant_initializer(0.0))
```

```python
#两种方式都可以
with tf.variable_scope("relu", reuse=True):
    threshold = tf.get_variable("threshold")
with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")
```



```python
#将共享变量作为函数的参数
def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (X.get_shape()[1],1)
        w = tf.Variable(w_shape, name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w),b, name="z")
        return  tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0 , name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")

#将共享变量作为函数的属性
def relu(X):
    with tf.name_scope("relu"):
        """把threshold作为了relu函数的属性"""
        if not hasattr(relu, "threshold"):#在第一次时候进行判断
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1                       
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  
        b = tf.Variable(0.0, name="bias")                           
        z = tf.add(tf.matmul(X, w), b, name="z")                  
        return tf.maximum(z, relu.threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

#在python中一切皆对象。函数属性以字典的形式存储的，键为属性名，值为属性内容。
#函数的属性可以在定义函数时同时定义函数属性, 也可以在函数声明外定义函数属性。
#可以通过句点标识符和直接赋值的方法为一个函数添加属性。
#函数对象的 __dict__ 特殊属性包含了函数对象的属性和属性值。
#print_header.category = 1和print_header.text = "some info"定义了两个函数属性。
#通过print_header.__dict__可以查看函数属性。{'category': 1, 'text': 'some info'}

#使用get_variable创建共享变量
#只使用tf.summary.filewriter则可以使得只画图graph
#此段程序threshold在外面，不在那五个relu里面，因为是在relu外面创建的共享变量。
n_features = 3
def relu(X):
    """重用后"""
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1                         
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  
        b = tf.Variable(0.0, name="bias")                          
        z = tf.add(tf.matmul(X, w), b, name="z")                    
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close()

"""把threshold定义在relu函数中。"""
#其实就是用了一个for循环的小技巧，更改了之前get_variable的参数无法修改的问题。
reset_graph()
def relu(X):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  
    b = tf.Variable(0.0, name="bias")                           
    z = tf.add(tf.matmul(X, w), b, name="z")                    
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    """使用reuse的条件让命名作用域第一次创建时不重用，之后的relu创建时得到重用，这使得所有的共享变量都在第一个relu内"""
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu9", tf.get_default_graph())
file_writer.close()
```

## 优化器(optimizer)

###Adam Optimization

加快训练的技巧之一是使用更快的优化器。**<u>建议采用Adam Optimization</u>**。

```python
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
```

除此之外，还有四种方式：

So far we have seen four ways to speed up training (and reach a better solution):

- applying a good initialization strategy for the connection weights

- using a good activation function

- using Batch Normalization

- reusing parts of a pretrained network

### Learning Rate Scheduling

####意义
However, you can do better than a constant learning rate: if you start with a high learning rate and then reduce it once it stops making fast progress, you can reach a good solution faster than with the optimal constant learning rate.

![](picture/learning rate.png)

####注意事项

- 推荐使用Exponential scheduling，因为易于实现：`decayed_learning_rate = learning_rate *                        decay_rate ^ (global_step / decay_steps)`
- 对于一些优化器，本身已经自带衰减学习速率的效果。Since AdaGrad, RMSProp, and Adam optimization automatically reduce the learning rate during training, it is not necessary to add an extra learning schedule. For other optimization algorithms, using exponential decay or performance scheduling can
  considerably speed up convergence.
  
####代码

核心代码如下：

```python
with tf.name_scope("train"):       # not shown in the book
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)
```

```python
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    
with tf.name_scope("train"):       # not shown in the book
    initial_learning_rate = 0.1
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               decay_steps, decay_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss, global_step=global_step)
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 5
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

## Avoiding Overfitting Through Regularization

一共介绍了五种方法。

### Early Stopping

One way to implement this with TensorFlow is to evaluate the model on a validation set at regular intervals (e.g., every 50 steps), and save a “winner” snapshot if it outperforms previous “winner” snapshots. Count the number of steps since the last “winner” snapshot was saved, and interrupt training when this number reaches some limit (e.g., 2,000 steps). Then restore the last “winner” snapshot.

### ℓ1 and ℓ2 Regularization

####原理

原理方面具体可参加machine learning中的关于L1，L2范数的笔记。

**解决过拟合可以从两个方面入手，一是减少模型复杂度，一是增加训练集个数**。而正则化就是减少模型复杂度的一个方法。

**由于模型的参数个数一般是由人为指定和调节的，所以正则化常常是用来限制模型参数值不要过大，也被称为惩罚项。一般是在目标函数(经验风险)中加上一个正则化项**。**而这个正则化项一般会采用L1范数或者L2范数。**

$J(\theta) = \frac{1}{2m} \left( \sum_{i=1}^{m}  (h(x^{(i)}) – y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right)$

**注意这里并没有指定是逻辑回归的正则化情况，给出的是一个一般情况下的正则化**，根据惯例，**我们不对**$\theta_0$**进行惩罚,所以最后一项$j$从1开始。**其中$\lambda$叫正则化参数（Regularization Parameter）。如果正则化参数太大，则会把所有参数最小化，导致近似直线，造成欠拟合。

**L1会趋向于产生少量的特征，而其他的特征都是0，而L2会选择更多的特征，这些特征都会接近于0。L1在特征选择时候非常有用，而L2就只是一种规则化而已**。

**<u>L1范数：表示为向量中各个元素绝对值之和。</u>**

**<u>L2范数：表示为向量中各个元素的平方和然后求平方根。</u>**

#### 代码

Tensorflow提供了代码，帮助计算正则化，此外需要将正则化的loss与原本的loss进行汇总，才能进一步更新，核心代码如下：

```python
my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("loss"):                                     # not shown in the book
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
        labels=y, logits=logits)                                # not shown
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")   # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")
```

```python
from functools import partial
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

scale = 0.001

my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("dnn"):
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, activation=None,
                            name="outputs")

with tf.name_scope("loss"):                                     # not shown in the book
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
        labels=y, logits=logits)                                # not shown
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")   # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images,
                                                y: mnist.test.labels})
        print(epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

```

### Dropout

####原理

除了输出层的神经元，其他神经元都有50%的机会被丢弃。

It is a fairly simple algorithm: at every training step, **<u>every neuron (including the input neurons but excluding the output neurons) has a probability p of being temporarily “dropped out,”</u>** meaning it will be entirely ignored during this training step, but it may be active during the next step. **<u>The hyperparameter p is called the dropout rate, and it is typically set to 50%.</u>** After training, neurons don’t get dropped anymore. 

涉及到权值的修正：<u>More generally, we need to multiply each input connection weight by the keep probability (1 – p) after training. Alternatively, we can divide each neuron’s output by the keep probability during training (these alternatives are not perfectly equivalent, but they work equally well).</u>

####代码

Tensorflow实现的是后一种就是对每次的训练进行除以保存率，放大每个神经元的输出。<u>就像是在需要dropout的层上，外加一层，然后设置训练时是training，而测试时会关闭为false。</u>核心代码如下：

```python
training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")
```

```python

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")
    
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)    

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")
```

### Max-Norm Regularization

####原理

**在每一次的training step之后，计算权值，使得每个神经元的权值满足$\Vert w \Vert_2 < r$。**

因为书中写的有点复杂，节约时间就不看代码了。需要用到的时候，再返回去细看。

### Data Augmentation

通过现有的数据，修改生成新的数据。

![](picture/data augmentation.png)

## 推荐的常用配置

**<u>大多数情况下，该优化配置都表现良好。</u>**

![](picture/practical guideline.png)

##教程

https://zhuanlan.zhihu.com/p/35515805

# References

- [Feature Scaling with scikit-learn](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
- [Can z-scores be used for data that is not Normally distributed](https://www.reddit.com/r/math/comments/zqcei/can_zscores_be_used_for_data_that_is_not_normally/)
- [配套代码](https://github.com/ageron/handson-ml)
