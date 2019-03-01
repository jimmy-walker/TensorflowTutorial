# Keras

- TensorFlow和theano以及Keras都是深度学习框架，TensorFlow和theano比较灵活，也比较难学，它们其实就是一个微分器。

- Keras其实就是TensorFlow和Keras的接口（Keras作为前端，TensorFlow或theano作为后端），它也很灵活，且比较容易学。可以把keras看作为tensorflow封装后的一个API。

  ![](picture/Keras.png)

##资料
参考keras官方文档

##在 Keras 中有两类主要的模型：[Sequential 顺序模型](https://keras.io/zh/models/sequential) 和 [使用函数式 API 的 Model 类模型](https://keras.io/zh/models/model)。 

##最基础的Sequential顺序模型搭建的流程

### 基于多层感知器 (MLP) 的 softmax 多分类：

```
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 生成虚拟数据
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

### 基于多层感知器的二分类：

```
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 生成虚拟数据
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

### 类似 VGG 的卷积神经网络：

```
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 生成虚拟数据
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器。
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)
```

### 基于 LSTM 的序列分类：

```
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 基于 1D 卷积的序列分类：

```
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

## 使用函数式 API 的 Model 类模型搭建过程

###核心是callable

在函数式 API 中，给定一些输入张量和输出张量，可以通过以下方式实例化一个 `Model`：

```
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

这个模型将包含从 `a` 到 `b` 的计算的所有网络层。

在多输入或多输出模型的情况下，你也可以使用列表：

```
model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])
```

### 典型例子

#### 建立Model

用输入和输出的向量定义一个Model

```python
model = Model(img_input, output)
```

#### 配置compile

**`metrics=['accuracy']` means that we print the accuracy during training.**

**acc是accuracy的缩写。**

**而loss则表示我们要在训练中打印出loss。**

the training loss is the average of the losses over each batch of training data. 应该acc也是每个批次的acc的平均值。

```python
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
```

#### 输出模型构建

```python
model.summary()
```

#### 开始训练

fit_generator函数使用见下文。

```python
model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                        batch_size),
                    steps_per_epoch=len(train_faces) / batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data)
```



```python
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from models.cnn import mini_XCEPTION
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = '../trained_models/emotion_models/'

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)
```





## 结合tfrecord

https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/

https://github.com/keras-team/keras/blob/master/examples/mnist_tfrecord.py

在`keras`中有`keras.preprocessing.image.ImageDataGenerator()`类和`.flow_from_directory()`函数可以很容易将保存在 **文件夹** 下面的数据进行读取；也可以用`.flow()`函数将数据直接从np.array中读取后输入网络进行训练（具体可以查看[官方文档](https://keras.io/preprocessing/image/)）。在使用图片并以文件夹名作为分类名的训练任务时这个方案是十分简单有效的，但是Tensorflow官方推荐的数据保存格式是 **TFRecords**，而keras官方不支持直接从tfrecords文件中读取数据（`tf.keras`也不行，但是这个[issue](https://github.com/tensorflow/tensorflow/issues/8787)中提供了一些PR是可以的，keras作者不太推荐就是了），所以这里就可以用`data`类来处理从TFRecords中的数据（也可以用之前常用的`tf.train.batch()`或`tf.train.shuffle_batch()`来处理训练数据）。 

https://www.cnblogs.com/arkenstone/p/8448208.html

**所以还是用回tf.data来读数据，在tf.data里面做增强。**

### 另一种声音

[此文](https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36)作者提出可以在keras中使用tfrecord，待使用查看效果。

```python
import tensorflow as tf
from tensorflow.python import keras as keras

STEPS_PER_EPOCH= SUM_OF_ALL_DATASAMPLES / BATCHSIZE
#Get your datatensors
image, label = create_dataset(filenames_train)

#Combine it with keras
model_input = keras.layers.Input(tensor=image)

#Build your network
model_output = keras.layers.Flatten(input_shape=(-1, 255, 255, 1))(model_input)
model_output = keras.layers.Dense(1000, activation='relu')(model_output)

#Create your model
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

#Compile your model
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    loss='mean_squared_error',
                    metrics=[soft_acc],
                    target_tensors=[label])

#Train the model
train_model.fit(epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOC)

#More Kerasstuff here
```



## 回调函数

回调函数是一个函数的合集，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。你可以传递一个列表的回调函数（作为 `callbacks` 关键字参数）到 `Sequential` 或 `Model` 类型的 `.fit()` 方法。在训练时，相应的回调函数的方法就会被在各自的阶段被调用。 

```python
log_file_path = base_path + dataset_name + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
data_loader = DataManager(dataset_name, image_size=input_shape[:2])
faces, emotions = data_loader.get_data()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data
model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                        batch_size),
                    steps_per_epoch=len(train_faces) / batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data)
```

### CSVLogger

Callback that streams epoch results to a csv file.

```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

### EarlyStopping

当被监测的指标不再提升，则停止训练。

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
```

- **monitor**: 被监测的数据。
- **min_delta**: 在被监测的数据中被认为是提升的最小变化， 例如，小于 min_delta 的绝对变化会被认为没有提升。
- **patience**: 没有进步的训练轮数，在这之后训练就会被停止。
- **verbose**: 详细信息模式。

```python
early_stop = EarlyStopping('val_loss', patience=patience)
```

### ReduceLROnPlateau

Reduce learning rate when a metric has stopped improving.

当学习停止时，模型总是会受益于降低 2-10 倍的学习速率。 这个回调函数监测一个数据并且当这个数据在一定「有耐心」的训练轮之后还没有进步， 那么学习速率就会被降低。 

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

- **monitor**: 被监测的数据。
- **factor**: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
- **patience**: 没有进步的训练轮数，在这之后训练速率会被降低。
- **verbose**: 整数。0：安静，1：更新信息。
- **mode**: {auto, min, max} 其中之一。如果是 `min` 模式，学习速率会被降低如果被监测的数据已经停止下降； 在 `max` 模式，学习塑料会被降低如果被监测的数据已经停止上升； 在 `auto` 模式，方向会被从被监测的数据中自动推断出来。
- **epsilon**: 对于测量新的最优化的阀值，只关注巨大的改变。
- **cooldown**: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
- **min_lr**: 学习速率的下边界。

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

### ModelCheckpoint

```
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

Save the model after every epoch.

- **filepath**: 字符串，保存模型的路径。

- **monitor**: 被监测的数据。

- **verbose**: 详细信息模式，0 或者 1 。

- **save_best_only**: 如果 `save_best_only=True`， the latest best model according to the quantity monitored will not be overwritten.J也就是说之前的最佳模型不会被覆盖。

- **mode**: {auto, min, max} 的其中之一。 如果 `save_best_only=True`，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。 对于 `val_acc`，模式就会是 `max`，而对于 `val_loss`，模式就需要是 `min`，等等。 在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来。

- **save_weights_only**: 如果 True，那么只有模型的权重会被保存 (`model.save_weights(filepath)`)， 否则的话，整个模型会被保存 (`model.save(filepath)`)。

- **period**: 每个检查点之间的间隔（训练轮数）。

```python
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)
```

##设置loss和metrics
###自定义的方法定义metrics

具体代码可见官网的[loss代码](https://github.com/keras-team/keras/blob/master/keras/losses.py)和[metric代码](https://github.com/keras-team/keras/blob/master/keras/metrics.py)，从而学习如何撰写。其中的函数可参考官网的[说明](https://keras.io/backend/)。

而`y_true`和`y_pred`的具体的shape分析，可见[StackOverflow](https://stackoverflow.com/a/46667294/8355906)回答。

The tensor `y_true` is the true data (or target, ground truth) you pass to the fit method.
It's a conversion of the numpy array `y_train` into a tensor.

The tensor `y_pred` is the data predicted (calculated, output) by your model.

Both `y_true` and `y_pred` have exactly the same shape, always.

------

It contains an entire batch. Its first dimension is always the batch size, and it must exist, even if the batch has only one element.

Two very easy ways to find the shape of `y_true` are:

- check your true/target data: `print(Y_train.shape)`
- check your `model.summary()` and see the last output

But its first dimension will be the batch size.

So, if your last layer outputs `(None, 1)`, the shape of `y_true` is `(batch, 1)`. If the last layer outputs `(None, 200,200, 3)`, then `y_true` will be `(batch, 200,200,3)`.

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```



###回归和分类常用网络设计及其loss，metrics

####最后一层网络的设计

网络最后一层的[设计建议](https://stats.stackexchange.com/a/218589)：

- Regression: linear (because values are unbounded)
- Classification: softmax (simple sigmoid works too but softmax works better)

####loss及metrics

具体回归实例可见官网[例子](https://www.tensorflow.org/tutorials/keras/basic_regression)，分类实例也可见官网[例子](https://www.tensorflow.org/tutorials/keras/basic_classification)。

- 对于回归任务：
- - Mean Squared Error (MSE) is a common loss function used for regression problems (different than classification problems).
- - Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).

```python
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
model.summary()
```

- 对于分类任务：
- - 常使用交叉损失熵作为loss
- - 常使用accuracy作为metrics
  - 如果使用`categorical_crossentropy`，需要事先转成稀疏形式表达的向量。
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

**Note**: when using the `categorical_crossentropy` loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample). In order to convert *integer targets* into *categorical targets*, you can use the Keras utility `to_categorical`:

```
from keras.utils.np_utils import to_categorical
categorical_labels = to_categorical(int_labels, num_classes=None)
```


## 常用函数

###`ImageDataGenerator`

对图像进行增强，解决样本数较少的问题。

https://github.com/JustinhoCHN/keras-image-data-augmentation

#### `flow`

Takes data & label arrays, generates batches of augmented data. 

采集数据和标签数组，生成批量增强数据。

返回一个生成元组 `(x, y)` 的 Iterator，其中 `x` 是图像数据的 Numpy 数组（在单张图像输入时），或 Numpy 数组列表（在额外多个输入时），`y` 是对应的标签的 Numpy 数组。如果 'sample_weight' 不是 None，生成的元组形式为 `(x, y, sample_weight)`。如果 `y` 是 None, 只有 Numpy 数组 `x` 被返回。 

**`ImageDataGenerator`对象的flow方法，对输入数据`（imgs,ylabel）`打乱（默认参数，可设置）后，依次取batch_size的图片并逐一进行变换。取完后再循环（J就是再打乱，再取batch size）。** 

```python
import time
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

imgs=np.random.randint(0,10,size=(7,100,100,3))

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

f=datagen.flow(imgs,[0,1,2,3,4,5,6],batch_size=3,save_to_dir="output/")

# print(f.next()[1])
# time.sleep(2)
# print(f.next()[1])
# time.sleep(2)
# print(f.next()[1])

for index,(x,y) in enumerate(f):
    if index==10:
        break
    time.sleep(1)
    print(x.shape,y)
```

```
(3, 100, 100, 3) [5 0 6]
(3, 100, 100, 3) [3 2 1]
(1, 100, 100, 3) [4]
(3, 100, 100, 3) [6 0 4]
(3, 100, 100, 3) [1 2 3]
(1, 100, 100, 3) [5]
(3, 100, 100, 3) [6 1 0]
(3, 100, 100, 3) [4 5 2]
(1, 100, 100, 3) [3]
(3, 100, 100, 3) [3 2 1]
```

```python
if shuffle==True:
    shuffle(x,y)#打乱
while(True):
    for i in range(0,len(x),batch_size):
        x_batch=x[i:i+batch_size]
        y_batch=y[i:i+batch_size]
        ImagePro(x_batch)#数据增强
        saveToFile()#保存提升后的图片
        yield (x_batch,y_batch)
```



####`flow_from_directory`

Takes the path to a directory & generates batches of augmented data. 

输入图片的目录地址，然后会生成增强后图像的各批次数据。

返回一个生成 `(x, y)` 元组的 `DirectoryIterator`，其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)`的图像的 Numpy 数组，`y` 是对应标签的 Numpy 数组。 

```python
path = '/home/ubuntu/dataset/dogs_cats_sample/'
gen_path = '/home/ubuntu/dataset/dogs_cats_gen/'
datagen = image.ImageDataGenerator()
gen_data = datagen.flow_from_directory(path, batch_size=1, shuffle=False, 
                                       save_to_dir=gen_path+'18',
                                  save_prefix='gen', target_size=(224, 224))
for i in range(9):
    gen_data.next() #此处并不使用其返回值(x,y)元祖，只是将其保存图片。
fig = print_result(gen_path+'18/*')
```

###`Conv2D`

```
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 卷积层 (例如对图像的空间卷积)。 该层创建了一个卷积核， 该卷积核对层输入进行卷积， 以生成输出张量。 

当使用该层作为模型第一层时，需要提供 `input_shape` 参数 （tuple of integers, does not include the sample axis ），例如，`input_shape=(128, 128, 3)` 表示 128x128 RGB 图像。

**即在定义时不包含sample的维度，只需三维，而在具体输入时则包含sample，需要四维。**

**filter就是代表卷积核数。因为卷积核的in_channels是由输入决定的，因此定义kernel_size时只需要定义两维就行了。**

**输入尺寸**

- 如果 data_format='channels_first'， 输入 4D 张量，尺寸为 `(samples, channels, rows, cols)`。
- 如果 data_format='channels_last'， 输入 4D 张量，尺寸为 `(samples, rows, cols, channels)`。

**输出尺寸**

- 如果 data_format='channels_first'， 输出 4D 张量，尺寸为 `(samples, filters, new_rows, new_cols)`。

- 如果 data_format='channels_last'， 输出 4D 张量，尺寸为 `(samples, new_rows, new_cols, filters)`。

```python
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
```

###`summary`
`model.summary()` prints a summary representation of your model. Shortcut for [utils.print_summary](https://keras.io/utils/#print_summary)

 J打印出网络的相关信息

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 64, 64, 1)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 62, 62, 8)    72          input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 62, 62, 8)    32          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 62, 62, 8)    0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 60, 60, 8)    576         activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 60, 60, 8)    32          conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 60, 60, 8)    0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
separable_conv2d_1 (SeparableCo (None, 60, 60, 16)   200         activation_2[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 60, 60, 16)   64          separable_conv2d_1[0][0]         
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 60, 60, 16)   0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
separable_conv2d_2 (SeparableCo (None, 60, 60, 16)   400         activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 60, 60, 16)   64          separable_conv2d_2[0][0]         
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 30, 30, 16)   128         activation_2[0][0]               
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 30, 30, 16)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 30, 30, 16)   64          conv2d_3[0][0]                   
__________________________________________________________________________________________________
add_1 (Add)                     (None, 30, 30, 16)   0           max_pooling2d_1[0][0]            
                                                                 batch_normalization_3[0][0]      
__________________________________________________________________________________________________
separable_conv2d_3 (SeparableCo (None, 30, 30, 32)   656         add_1[0][0]                      
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 30, 30, 32)   128         separable_conv2d_3[0][0]         
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 30, 30, 32)   0           batch_normalization_7[0][0]      
__________________________________________________________________________________________________
separable_conv2d_4 (SeparableCo (None, 30, 30, 32)   1312        activation_4[0][0]               
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 30, 30, 32)   128         separable_conv2d_4[0][0]         
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 15, 15, 32)   512         add_1[0][0]                      
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 15, 15, 32)   0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 15, 15, 32)   128         conv2d_4[0][0]                   
__________________________________________________________________________________________________
add_2 (Add)                     (None, 15, 15, 32)   0           max_pooling2d_2[0][0]            
                                                                 batch_normalization_6[0][0]      
__________________________________________________________________________________________________
separable_conv2d_5 (SeparableCo (None, 15, 15, 64)   2336        add_2[0][0]                      
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 15, 15, 64)   256         separable_conv2d_5[0][0]         
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 15, 15, 64)   0           batch_normalization_10[0][0]     
__________________________________________________________________________________________________
separable_conv2d_6 (SeparableCo (None, 15, 15, 64)   4672        activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 15, 15, 64)   256         separable_conv2d_6[0][0]         
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 8, 8, 64)     2048        add_2[0][0]                      
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 8, 8, 64)     0           batch_normalization_11[0][0]     
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 8, 8, 64)     256         conv2d_5[0][0]                   
__________________________________________________________________________________________________
add_3 (Add)                     (None, 8, 8, 64)     0           max_pooling2d_3[0][0]            
                                                                 batch_normalization_9[0][0]      
__________________________________________________________________________________________________
separable_conv2d_7 (SeparableCo (None, 8, 8, 128)    8768        add_3[0][0]                      
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 8, 8, 128)    512         separable_conv2d_7[0][0]         
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 8, 8, 128)    0           batch_normalization_13[0][0]     
__________________________________________________________________________________________________
separable_conv2d_8 (SeparableCo (None, 8, 8, 128)    17536       activation_6[0][0]               
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 8, 8, 128)    512         separable_conv2d_8[0][0]         
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 4, 4, 128)    8192        add_3[0][0]                      
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 4, 4, 128)    0           batch_normalization_14[0][0]     
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 4, 4, 128)    512         conv2d_6[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 4, 4, 128)    0           max_pooling2d_4[0][0]            
                                                                 batch_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 4, 2)      2306        add_4[0][0]                      
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 2)            0           conv2d_7[0][0]                   
__________________________________________________________________________________________________
predictions (Activation)        (None, 2)            0           global_average_pooling2d_1[0][0] 
==================================================================================================
Total params: 52,658
Trainable params: 51,186
Non-trainable params: 1,472
```

### `fit_generator`

```
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```

**即每次epoch，就用generator产生多个batch的数据（这些batch数据对应整个训练集），训练完后，再用validation_data测试下。其中steps_per_epoch表示的是从generator要产生的batch数，一般等于len(train_faces) / batch_size，剩下不足batch_size的部分就不使用了。**

- **generator**: 一个生成器，或者一个 `Sequence` (`keras.utils.Sequence`) 对象的实例， 以在使用多进程时避免数据的重复。 生成器的输出应该为以下之一：
- 一个 `(inputs, targets)` 元组
- 一个 `(inputs, targets, sample_weights)` 元组。 这个元组（生成器的单个输出）组成了单个的 batch。 因此，这个元组中的所有数组长度必须相同（与这一个 batch 的大小相等）。 不同的 batch 可能大小不同。 例如，一个 epoch 的最后一个 batch 往往比其他 batch 要小， 如果数据集的尺寸不能被 batch size 整除。 生成器将无限地在数据集上循环。当运行到第 `steps_per_epoch` 时，记一个 epoch 结束。
- **steps_per_epoch**: 在声明一个 epoch 完成并开始下一个 epoch 之前从 `generator` 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。 对于 `Sequence`，它是可选的：如果未指定，将使用`len(generator)` 作为步数。
- **epochs**: 整数，数据的迭代总轮数。
- **verbose**: 日志显示模式。0，1 或 2。
- **callbacks**: 在训练时调用的一系列回调函数。
- **validation_data**: 它可以是以下之一：
- 验证数据的生成器或 `Sequence` 实例
- 一个 `(inputs, targets)` 元组
- 一个 `(inputs, targets, sample_weights)` 元组。
- **validation_steps**: 仅当 `validation_data` 是一个生成器时才可用。 在停止前 `generator` 生成的总步数（样本批数）。 对于 `Sequence`，它是可选的：如果未指定，将使用 `len(generator)` 作为步数。
- **class_weight**: 将类别索引映射为权重的字典。
- **max_queue_size**: 整数。生成器队列的最大尺寸。 如未指定，`max_queue_size` 将默认为 10。
- **workers**: 整数。使用的最大进程数量，如果使用基于进程的多线程。 如未指定，`workers` 将默认为 1。如果为 0，将在主线程上执行生成器。
- **use_multiprocessing**: 布尔值。如果 True，则使用基于进程的多线程。 如未指定， `use_multiprocessing` 将默认为 False。 请注意，由于此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- **shuffle**: 是否在每轮迭代之前打乱 batch 的顺序。 只能与 `Sequence` (keras.utils.Sequence) 实例同用。
- **initial_epoch**: 开始训练的轮次（有助于恢复之前的训练）。

## Reference
- [Keras官方介绍](https://www.tensorflow.org/guide/keras)
- [keras：ImageDataGenerator的flow方法](https://blog.csdn.net/nima1994/article/details/80625938)
- [Keras Image Data Augmentation 各参数详解](https://github.com/JustinhoCHN/keras-image-data-augmentation/blob/master/data_augmentation.ipynb)
- [Keras中compile决定的acc和loss](https://github.com/keras-team/keras/issues/10426)

