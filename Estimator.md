# Estimator

**J粗略理解Estimator就是利用回调函数的方式就能调用其进行训练评估预测等操作。可以利用已有的Estimator（就是其中已经有定义好的神经网络模型了即模型函数model_fn），也可以自己定义新的神经网络模型。**

##Brief

[**Estimator**](https://www.tensorflow.org/api_docs/python/tf/estimator?hl=zh-cn) 是一种可极大地简化机器学习编程的高阶 TensorFlow API。Estimator 会封装下列操作：

- 训练
- 评估
- 预测
- 导出以供使用

可以使用提供的**预创建的 Estimator，也可以编写自定义 Estimator**。所有 Estimator（无论是预创建的还是自定义）都是基于 [`tf.estimator.Estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator?hl=zh-cn) 类的类。 

优点是：

- **可以在本地主机上或分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。此外，可以在 CPU、GPU 或 TPU 上运行基于 Estimator 的模型，而无需重新编码模型**。
- Estimator 简化了在模型开发者之间共享实现的过程

**使用 Estimator 编写应用时，必须将数据输入管道从模型中分离出来**。这种分离简化了不同数据集的实验流程。  

## Pre-made Estimators

依赖预创建的 Estimator 的 TensorFlow 程序通常包含下列四个步骤：

1. **编写一个或多个数据集导入函数。** 例如，您可以创建一个函数来导入训练集，并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象：

   - 一个字典，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
   - 一个包含一个或多个标签的张量

   例如，以下代码展示了输入函数的基本框架：

```python
def input_fn(dataset):   ...  # manipulate dataset, extracting the feature dict and the label   
   return feature_dict, label
```

2. **定义特征列。** 每个 [`tf.feature_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column?hl=zh-cn) 都标识了特征名称、特征类型和任何输入预处理操作。例如，以下代码段创建了三个存储整数或浮点数据的特征列。前两个特征列仅标识了特征的名称和类型。第三个特征列还指定了一个 lambda，该程序将调用此 lambda 来调节原始数据：

```python
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                   normalizer_fn=lambda x: x - global_education_mean)
```

3. **实例化相关的预创建的 Estimator。** 例如，下面是对名为 `LinearClassifier` 的预创建 Estimator 进行实例化的示例代码：

```python
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
   feature_columns=[population, crime_rate, median_education],
   )
```

4. **调用训练、评估或推理方法。**例如，所有 Estimator 都提供训练模型的 `train` 方法。

```python
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000) #input_fn就是回调函数
```

```
拓展: 编程分为两大类:系统编程和应用编程,所谓的系统编程简单来说就是编写库,而应用编程就是利用写好的各种库来编写具体的某种功能的程序,也就是应用. 系统程序员会在自己写的库中留下API(应用编程接口),以供应用程序员使用.
当程序跑起来,一般情况下,应用程序会时常通过API调用库中所预备好的函数,但是有些库函数却要求应用先给它传递一个函数,好在合适的时候调用,以完成目标,这个被传入的后来又被调用的函数成为回调函数
```

```python
def wake_call(time):
   #第一种叫醒服务
   print(time,"使用夺命电话连环call叫醒主人")

def wake_water(time):
   #第二种叫醒服务
   print(time,"使用早起泼冷水的方式叫醒主人")

def call_wake(time, func_name):
   # 这个很重要,这个就是酒店服务业务的系统业务
   #这是实现回调函数的核心
   # time :预约时间
   # func_time:回调函数名
   # return :调用的函数的结果
   return func_name(time)

# 调用函数
# wake_call 被调用的函数就是回调函数
call_wake("凌晨7点", wake_call)
```

##Custom Estimators

模型函数（即 `model_fn`）会实现机器学习算法。采用预创建的 Estimator 和自定义 Estimator 的唯一区别是：

- 如果采用预创建的 Estimator，则有人已为您编写了模型函数。
- 如果采用自定义 Estimator，则您必须自行编写模型函数。

这里引入官网的教程来解释。

其中模型函数必须提供代码来处理全部三个 mode 值。对于每个 mode 值，您的代码都必须返回 [`tf.estimator.EstimatorSpec`](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec?hl=zh_cn) 的一个实例，其中包含调用程序所需的信息 。`EstimatorSpec`表示Ops and objects returned from a `model_fn` and passed to an `Estimator`.

`EstimatorSpec` fully defines the model to be run by an `Estimator`.

![](picture/estimator.png)

##TPUEstimator

[Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator?hl=zh_cn) 是 TensorFlow 的模型级抽象层。 标准 `Estimators` 可以在 CPU 和 GPU 上运行模型。您必须使用 [`tf.contrib.tpu.TPUEstimator`](https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimator?hl=zh_cn) 才能在 TPU 上运行模型。

`TPUEstimator` 类与 `Estimator` 类有所不同。

要维护可在 CPU/GPU 或 Cloud TPU 上运行的模型，**最简单的方式是将模型的推理阶段（从输入到预测）定义在 `model_fn` 之外。然后，确保 `Estimator` 设置和 `model_fn` 的单独实现**，二者均包含此推理步骤。

### Config

**Estimator的配置，如果不指定的话，则默认代表本地执行的配置**。

The `config` argument can be passed [`tf.estimator.RunConfig`](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) object containing information about the execution environment. It is passed on to the `model_fn`, if the `model_fn` has a parameter named "config" (and input functions in the same manner). If the `config` parameter is not passed, it is instantiated by the `Estimator`. **Not passing config means that defaults useful for local execution are used.** `Estimator` makes config available to the model (for instance, to allow specialization based on the number of workers available), and also uses some of its fields to control internals, especially regarding checkpointing. 



## Reference

- [python基础之回调函数以及返回函数与闭包](https://blog.csdn.net/qq_29074261/article/details/80016788)
- [Estimator](https://www.tensorflow.org/guide/estimators?hl=zh-cn#pre-made_estimators)
- [创建自定义 Estimator](https://www.tensorflow.org/guide/custom_estimators?hl=zh-cn)
```

```