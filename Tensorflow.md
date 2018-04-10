# Tensorflow

##安装
```linux
conda create -n tf35 python=3.5
source activate tf35
pip install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
pip install --upgrade --ignore-installed tensorflow
```

##概念
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

###placeholder node占位符节点也是一种op
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


###执行阶段

`f.eval()` is equivalent to calling `tf.get_default_session().run(f)`或`sess.run(f)`。

##常用函数
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
training_op = tf.assign(theta, theta - learning_rate * gradients) #assign the new value to variable
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
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

# References

- [Feature Scaling with scikit-learn](http://benalexkeen.com/feature-scaling-with-scikit-learn/)
- [Can z-scores be used for data that is not Normally distributed](https://www.reddit.com/r/math/comments/zqcei/can_zscores_be_used_for_data_that_is_not_normally/)
