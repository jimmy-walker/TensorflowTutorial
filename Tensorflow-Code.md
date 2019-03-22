# Tensorflow Code

##tf.flags 

用于帮助我们添加命令行的可选参数。  也就是说利用该函数我们可以实现在命令行中选择需要设定的参数来运行程序，  可以不用反复修改源代码中的参数，直接在命令行中进行参数的设定。 

```python
import tensorflow as tf

flags = tf.flags #生成一个flags对象
FLAGS=flags.FLAGS #用其FLAGS变量来存参数值
flags.DEFINE_integer('data_num', 100, """Flag of type integer""") #调用方法进行赋值，参数名，参数值，参数说明
flags.DEFINE_string('img_path', './img', """Flag of type string""")
FLAGS.flag_values_dict() #Returns a dictionary that maps flag names to flag values.

def main():
    print(FLAGS.data_num, FLAGS.img_path) #调用参数

if __name__ == '__main__':
    tf.app.run() #Runs the program with an optional 'main' function and 'argv' list.
```

主函数中的tf.app.run()的源码显示：首先加载`flags`的参数项，其中参数是使用`tf.app.flags.FLAGS`定义的，然后执行`main`函数。因此必须在main函数中设置一个参数位置。如果要更换main名字，只需要在tf.app.run()中传入一个指定的函数名即可。

```python
def test(args):
    # test
    ...
if __name__ == '__main__':
    tf.app.run(test)
```

##tf.logging.info

```python
tf.logging.set_verbosity (tf.logging.INFO) 
```

作用：将 TensorFlow 日志信息输出到屏幕

TensorFlow有五个不同级别的日志信息。其严重性为调试DEBUG<信息INFO<警告WARN<错误ERROR<致命FATAL。当你配置日志记录在任何级别，TensorFlow将输出与该级别相对应的所有日志消息以及更高程度严重性的所有级别的日志信息。例如，如果设置错误的日志记录级别，将得到包含错误和致命消息的日志输出，并且如果设置了调试级别，则将从所有五个级别获取日志消息。

默认情况下，TENSFlow在WARN的日志记录级别进行配置，但是在跟踪模型训练时，需要将级别调整为INFO。

## tf.gfile.MakeDirs

Creates a directory and all parent/intermediate directories.

It succeeds if dirname already exists and is writable.如果存在就重写

## tf.nn.embedding_lookup 

tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引，其他的参数不介绍。

```python
import tensorflow as tf;
import numpy as np;

c = np.random.random([10,1])
b = tf.nn.embedding_lookup(c, [1, 3])

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print sess.run(b)
	print c
```
```
输出：
[[ 0.77505197]
 [ 0.20635818]]
[[ 0.23976515]
 [ 0.77505197]
 [ 0.08798201]
 [ 0.20635818]
 [ 0.37183035]
 [ 0.24753178]
 [ 0.17718483]
 [ 0.38533808]
 [ 0.93345168]
 [ 0.02634772]]
```
分析：输出为张量的第一和第三个元素。

## tf.contrib.layers.layer_norm

Adds a Layer Normalization layer. 