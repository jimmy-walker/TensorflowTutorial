#Reading data

## 基础知识

https://zhuanlan.zhihu.com/p/27238630

很好的讲解文章，了解了文件名队列

https://hk.saowen.com/a/ba8b012d8c6505b401c36a12c9cbd6d63db4d2a7d3d97abda4ca18133a6bf72b

讲解了decode_raw和FixedLengthRecordReader搭配使用

##所有方式

第二和第三种的区别方式在于前一种需要自己每次循环中设置数据，第三种无需指定。

而第一种方式是最新的方法，代码中使用的则是第三种方式。

There are four methods of getting data into a TensorFlow program:

- [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) API: Easily construct a complex input pipeline. (preferred method)
- Feeding: Python code provides the data when running each step.
- `QueueRunner`: a queue-based input pipeline reads the data from files at the beginning of a TensorFlow graph.
- Preloaded data: a constant or variable in the TensorFlow graph holds all the data (for small data sets).

## 生成tfrecords-bert中最新的生成方式:

数据写入 .tfrecords 文件，需要将每一个样本数据封装为tf.train.Example格式，再将Example逐个写入文件。Example格式中的数据基础类型是tf.train.Feature。

```python
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
```

###基础数据格式

三种基础数据类型：bytes，float，int64

对应tf.train中三种类型：BytesList (字符串列表), FloatList (浮点数列表), Int64List (64位整数列表)，构造它们三个依照下面的方式传入相对应的value。

```
tf.train.BytesList(value=[context_idxs.tostring()]
tf.train.Int64List(value=[1,2])
tf.train.FloatList(value=[0.1,0.2])
```

###example 协议格式

**要将我们的数据写入 .tfrecords 文件，需要将每一个样本数据封装为tf.train.Example格式，再将Example逐个写入文件。Example格式中的数据基础类型是tf.train.Feature**

tf.train.Feature(): 它的参数是BytesList, FloatList, Int64List三种

```
tf.train.Feature(bytes_list=tf.train.BytesList(value=)
tf.train.Feature(int64_list=tf.train.Int64List(value=)
tf.train.Feature(float_list=tf.train.FloatList(value=)
```

tf.train.Features: 它的参数是一个字典，k-v对中 v 的类型是Feature，对应每一个字段。

```
tf.train.Features(feature={
            "k1": tf.train.Feature(bytes_list=tf.train.BytesList(value=])),
            "k2": tf.train.Feature(bytes_list=tf.train.BytesList(value=)),
            "k3": tf.train.Feature(float_list=tf.train.FloatList(value=)),
        })
```

tf.train.FeatureList: 它的参数是一个Feature的list, [Feature1, Feature2,...]

```
"context_idxs": tf.train.FeatureList(feature=
                                   [tf.train.Feature(int64_list=tf.train.Int64List(value=[])])
```

tf.train.FeatureLists: 它的参数是一个字典，k-v对中 v 的类型是FeatureList。

```
feature_lists=tf.train.FeatureLists(feature_list={
                "k1": tf.train.FeatureList(feature=
                                          [tf.train.Feature(int64_list=tf.train.Int64List(value=[])]),
                "k2": tf.train.FeatureList(feature=
                                          [tf.train.Feature(int64_list=tf.train.Int64List(value=v))])
            })
```

------

我们需要根据我们的数据，找到每一个字段应该映射为 Feature或FeatureList, 多个Feature组成Features，多个FeatureList组成FeatureLists, 然后我们就定义了我们的一个训练数据对应的 Features，FeatureLists, 再将其封装为 tf.train.Example 就可以写入 tfrecords二进制文件了。

```
tf.train.Example(features=):   传入的features对应一个 tf.train.Features
tf.train.SequenceExample(context=, featurelists=): 传入的context对应一个 tf.train.Features, features_lists对应一个tf.train.FeatureLists
```

这样就需要选择使用Example还是SequenceExample, SequenceExample多了一个featurelists, 也就是说如果数据中存在字段，我们把它映射为了FeatureList而不是Feature, 那么就要用SequenceExample, 否则用Example。

那么什么样的数据需要映射为FeatureList或Feature？

我的理解是对于长度固定的字段类型，映射为Feature， 比如分类问题中的类别这个字段一般用一个数字表示，二分类就是0或1，那么就class=0映射为tf.train.Feature(tf.train.Int64List(value=[0])), 只要这个字段包含的数据维度是固定的，就可以封装为 Feature。

对于长度不固定的字段类型，映射为FeatureList。比如NLP样本有一个特征是一句话，那么一句话的长度是不固定的，NLP中一般是先分词，然后把每个词对应为该词在字典中的索引，一句话就用一个一维整形数组来表示 [2, 3, 5, 20, ...]，这个数组的长度是不固定的，我们就映射为

```
tf.train.FeatureList(feature=[tf.train.Feature(value=[v]) for v in [2, 3, 5, 20,...] ] )
```

## 读取tfrecords-bert中最新的读取方式：

- 读入所有的tfrecord文件，组成列表传入dataset = tf.data.TFRecordDataset(filename)
- 利用函数的方式进行处理dataset = dataset.map(decode)，对其中每一个图片文件名进行处理，返回的也只是一个样本。注意这里虽然没有对所有样本循环，我猜测是流式隐循环；
  然后设置shuffle（不用考虑min_after_dequeue和num_threads），repeat和batch或apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
- 定义解析函数：
- parse_single_example将之前的A scalar string Tensor, a single serialized Example转变成张量。返回：A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.
- 其中FixedLenFeature表示Configuration for parsing a fixed-length input feature，J理解为怎么从protocol buffer中以怎样的格式结束数据。



```python
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn
```



## `QueueRunner`

**J总结一下该方式的整体思路：**

```
先用string_input_producer生成一个文件名队列，注意这里虽然传入的是tfrecord格式的地址，但我猜测其会解析成具体图片，那么就是说根据epoch数，生成epoch遍的图片文件名；
然后采用read和parse_single_example对其中每一个图片文件名进行处理，返回的也只是一个样本。注意这里虽然没有对所有样本循环，我猜测是流式隐循环；
最后将单一个样本符号送入shuffle_batch，他会根据batch_size等到输入足够的样本数，就返回一个batch，然后不断这么返回，注意这个函数返回的是batch数据！！！见下面演示代码，调用一次run后的数据看效果。
```

###查看具体数据的演示代码

如果需要查看具体的数据可使用下列代码

```python
import tensorflow as tf

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['./data/train/train-001.tfrecords'], num_epochs=None)

# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# get feature from serialized example
features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'age': tf.FixedLenFeature([], tf.int64),
        'gender': tf.FixedLenFeature([], tf.int64),
        'file_name': tf.FixedLenFeature([], tf.string)
    })

#preprocessing
image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([160 * 160 * 3])
image = tf.reshape(image, [160, 160, 3])
image = tf.reverse_v2(image, [-1])
image = tf.image.per_image_standardization(image)

age = features['age']
gender = features['gender']
file_path = features['file_name']

print (image)
print (age)
print (gender)
print (file_path)
# Tensor("div:0", shape=(160, 160, 3), dtype=float32)
# Tensor("ParseSingleExample/Squeeze_age:0", shape=(), dtype=int64)
# Tensor("ParseSingleExample/Squeeze_gender:0", shape=(), dtype=int64)
# Tensor("ParseSingleExample/Squeeze_file_name:0", shape=(), dtype=string)

#batch
images_batch, sparse_labels_batch, genders_batch, file_paths_batch = tf.train.shuffle_batch([image, age, gender, file_path], batch_size=3, 
	capacity=200, min_after_dequeue=100, num_threads=2)

#excuate
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
images_val, sparse_labels_val, genders_val = sess.run([images_batch, sparse_labels_batch, genders_batch])
# print(a_val, b_val, c_val)
print ('first batch:')
print ('  image_val:',images_val)
print ('  label_val:',sparse_labels_val)
print ('  gender_val:',genders_val)
# label_val: [51 31 17]
# gender_val: [1 0 0]
images_val, sparse_labels_val, genders_val = sess.run([images_batch, sparse_labels_batch, genders_batch])
print ('second batch:')
print ('  image_val:',images_val)
print ('  label_val:',sparse_labels_val)
print ('  gender_val:',genders_val)
# label_val: [35 69 32]
# gender_val: [0 1 1]
```



**Warning:** This section discusses implementing input pipelines using the queue-based APIs which can be cleanly replaced by the [`tf.data` API](https://www.tensorflow.org/guide/datasets).

A typical queue-based pipeline for reading records from files has the following stages:

1. The list of filenames
2. *Optional* filename shuffling
3. *Optional* epoch limit
4. Filename queue
5. A Reader for the file format
6. A decoder for a record read by the reader
7. *Optional* preprocessing
8. Example queue

### Filenames, shuffling, and epoch limits

For the list of filenames, use either a constant string Tensor (like `["file0", "file1"]` or `[("file%d" % i) for i in range(2)]`).

```python
def get_files_name(path):
    list = os.listdir(path)
    result = []
    for line in list:
        file_path = os.path.join(path, line)
        if os.path.isfile(file_path):
            result.append(file_path)
    return result
```

Pass the list of filenames to the [`tf.train.string_input_producer`](https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer) function.`string_input_producer` creates a FIFO queue for holding the filenames until the reader needs them.

```python
def inputs(path, batch_size, num_epochs, allow_smaller_final_batch=False):
    """Reads input data num_epochs times.
    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    # filename = os.path.join(FLAGS.train_dir,
    #                       TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            path, num_epochs=num_epochs)
```

### Reader

Select the reader that matches your input file format and pass the filename queue to the reader's read method. The read method outputs a key identifying the file and record (useful for debugging if you have some weird records), and a scalar string value. Use one (or more) of the decoder and conversion ops to decode this string into the tensors that make up an example.

<u>J即返回(key, value)，其中value就是serialized_example，因为是scalar string（字符串），所以需要进一步解析成为tensor。</u>

#### Standard TensorFlow format

<u>J对于tfrecord格式，单独一个则用tf.data.TFRecordDataset，多个文件的话，使用队列，则用tf.TFRecordReader。</u>**J我认为这里我搞错了，其实tfrecorddataset也可以，而且输入的也不是多文件（不是多个tfrecord文件）。**

Another approach is to convert whatever data you have into a supported format. This approach makes it easier to mix and match data sets and network architectures. The recommended format for TensorFlow is a [TFRecords file](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) containing [`tf.train.Example`](https://www.tensorflow.org/api_docs/python/tf/train/Example) protocol buffers (which contain[`Features`](https://www.github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/core/example/feature.proto) as a field). You write a little program that gets your data, stuffs it in an `Example` protocol buffer, serializes the protocol buffer to a string, and then writes the string to a TFRecords file using the [`tf.python_io.TFRecordWriter`](https://www.tensorflow.org/api_docs/python/tf/python_io/TFRecordWriter). For example,[`tensorflow/examples/how_tos/reading_data/convert_to_records.py`](https://www.github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/how_tos/reading_data/convert_to_records.py) converts MNIST data to this format.

The recommended way to read a TFRecord file is with a [`tf.data.TFRecordDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset), [as in this example](https://www.github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py):

```python
dataset = tf.data.TFRecordDataset(filename)
dataset = dataset.repeat(num_epochs)

# map takes a python function and applies it to every sample
dataset = dataset.map(decode)
```

To accomplish the same task with a queue based input pipeline requires the following code (using the same `decode` function from the above example):

其中的reader方法：Returns the next record (key, value) pair produced by a reader.

Will dequeue a work unit from queue if necessary (e.g. when the Reader needs to start reading from a new file since it has finished with the previous file).

<u>J也就是意思说read每次产生一个新的样本，而非新的tfrecord file文件。见原文The read method outputs a key identifying the file and record (useful for debugging if you have some weird records), and a scalar string value.  </u>

<u>这里最大的误会可能就在于没有循环，因而以为是读取了整个tfrecord的file文件，J实际上是隐循环，因为下面的decode肯定是对每一个图像进行解析的，因而读取的是样本级，而非文件级。</u>

```python
filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
image,label = decode(serialized_example)
```
```python
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
```

### Decode

<u>Jparse_single_example将之前的A scalar string Tensor, a single serialized Example转变成张量。返回：A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.</u>

<u>其中FixedLenFeature表示Configuration for parsing a fixed-length input feature，J理解为怎么从protocol buffer中以怎样的格式结束数据。</u>

<u>而decode_raw表示Reinterpret the bytes of a string as a vector of numbers，就是将字符串解析为数值，J猜测上一步只是按照什么格式去读，但是读出来还是字符串，需要转换。</u>

To read a file of TFRecords, use [`tf.TFRecordReader`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/TFRecordReader) with the [`tf.parse_single_example`](https://www.tensorflow.org/versions/r1.0/api_docs/python/tf/parse_single_example) decoder. The `parse_single_example` op decodes the example protocol buffers into tensors. 

```python
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'age': tf.FixedLenFeature([], tf.int64),
            'gender': tf.FixedLenFeature([], tf.int64),
            'file_name': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #<tf.Tensor 'ParseSingleExample/Squeeze_image_raw:0' shape=() dtype=string>
    image.set_shape([160 * 160 * 3]) #人脸时设置的像素值
    image = tf.reshape(image, [160, 160, 3]) #将其展开
    image = tf.reverse_v2(image, [-1]) #opencv默认bgr，转换成rgb
    image = tf.image.per_image_standardization(image)
    # <tf.Tensor 'div:0' shape=(160, 160, 3) dtype=float32>
    # 因为需要转成0，255范围，而对于dense tensor的可转类型只有tfloat32, int64, string，所以无法在上一步直接转，所以需要先转成string，然后调用decode_raw转成uint8
    # 这里是因为age和gender可以直接int64位，所以无需调用decode_raw转换。
    age = features['age']
    #<tf.Tensor 'ParseSingleExample/Squeeze_age:0' shape=() dtype=int64>
    gender = features['gender']
    file_path = features['file_name']
    return image, age, gender, file_path
```

### Preprocessing

tf.image.per_image_standardization：Linearly scales `image` to have zero mean and unit norm.

### Batching

At the end of the pipeline we use another queue to batch together examples for training, evaluation, or inference. For this we use a queue that randomizes the order of examples, using the[`tf.train.shuffle_batch`](https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch).

<u>J这里有等待前面样本凑够了一批的含义。之前read是每个样本读出来后操作，虽然这里batch传入的参数还是单个样本的符号，比如下面的[example, label]，但是这个函数会负责后面生成batch的操作，所以不用担心。</u>

tf.train.shuffle_batch是将队列中数据打乱后，再读取出来，因此队列中剩下的数据也是乱序的，队头也是一直在补充（我猜也是按顺序补充），比如batch_size=5,capacity=10,min_after_dequeue=5,

初始是有序的0,1，..,9(10条记录)，

然后打乱8,2,6,4,3,7,9,2,0,1(10条记录),

队尾取出5条，剩下7,9,2,0,1(5条记录),

然后又按顺序补充进来，变成7,9,2,0,1,10,11,12,13,14(10条记录)，

再打乱13,10,2,7,0,12...1(10条记录)，

再出队...

capacity可以看成是局部数据的范围，读取的数据是基于这个范围的，

在这个范围内，min_after_dequeue越大，数据越乱。

这样按batch读取的话，最后会自动在前面添加一个维度，比如数据的维度是[1],batch_size是10，那么读取出来的shape就是[10,1] 。J就是第一个是batch_size的维度。

```python
def read_my_file_format(filename_queue):
  reader = tf.SomeReader()
  key, record_string = reader.read(filename_queue)
  example, label = tf.some_decoder(record_string)
  processed_example = some_processing(example)
  return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch
```

###Creating threads to prefetch using `QueueRunner` objects

上面的`tf.train`的函数都会产生`QueueRunner`对象，其存储了对于队列的相关操作。Holds a list of enqueue operations for a queue, each to be run in a thread.需要调用`start_queue_runners`启动这些`QueueRunner`对象才能操作。而`QueueRunner`需要配合`Coordinator`处理其队列可能出现的error。

 示例代码为：

```python
# Create the graph, etc.
init_op = tf.global_variables_initializer()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    while not coord.should_stop():
        # Run training steps or whatever
        sess.run(train_op)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
```

实例代码为：

```python
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    step = sess.run(global_step) #初始化一次step值方便后面的第一次summary
    start_time = time.time()
    while not coord.should_stop(): #在此循环，等待是否有任何队列所关联的线程发出停止的消息，因为需要不断对批量数据进行迭代，之前设置的allow_smaller_final_batch=False也是这个道理保证只有最后批次才不满足batch，直到末尾触发了outofrange的error，才会退出while
        _, summary = sess.run([train_op, merged], {train_mode: True})
        train_writer.add_summary(summary, step)
        if step % 100 == 0: #因为一开始就是0，所以会对第一次进行统计
            duration = time.time() - start_time
            print('%.3f sec' % duration)
            start_time = time.time()
        if step % 1000 == 0:
            save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=global_step)
            print("Model saved in file: %s" % save_path)
        step = sess.run(global_step)
except tf.errors.OutOfRangeError:#到了末尾后，就会到这里
    print('Done training for %d epochs, %d steps.' % (epoch, step))
finally:#最后退出try模块，就会执行此步骤
    # When done, ask the threads to stop.
    save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=global_step)
    print("Model saved in file: %s" % save_path)
    coord.request_stop() #要求停止所有队列
# Wait for threads to finish.
coord.join(threads) #等待队列全部停止
sess.close() #关闭sess即可
```

## `tf.data`

```
在一个input函数中：
先读入所有的tfrecord文件，组成列表传入dataset = tf.data.TFRecordDataset(filename)
然后利用函数的方式进行处理dataset = dataset.map(decode)，对其中每一个图片文件名进行处理，返回的也只是一个样本。注意这里虽然没有对所有样本循环，我猜测是流式隐循环；
然后设置shuffle（不用考虑min_after_dequeue和num_threads），repeat和batch或apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
最后生成迭代器返回iterator = dataset.make_one_shot_iterator()，get_next()

而在input函数外
用try进行训练sess.run，捕获tf.errors.OutOfRangeError，然后关闭sess.close即可
```

### 查看具体数据的演示代码

可以考虑像上一个一样，设置较小的batch，然后sess.run就可以打印出。

### 代码

```python
def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'age': tf.FixedLenFeature([], tf.int64),
          'gender': tf.FixedLenFeature([], tf.int64),
          'file_name': tf.FixedLenFeature([], tf.string),
      })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([160 * 160 * 3])
    image = tf.reshape(image, [160, 160, 3])
    image = tf.reverse_v2(image, [-1])
    image = tf.image.per_image_standardization(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    age = features['age']
    gender = features['gender']
    file_path = features['file_name']

    return image, age, gender, file_path

def inputs(path, batch_size, num_epochs):
    """Reads input data

    Args:
    path: the path where tfrecord file is stored

    Returns:
    A tuple (images, age_labels, gender_labels, file_paths), where:

    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
    """
    filename = glob.glob(path + "/*.tfrecords")

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(filename)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        #dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        #the final batch contain smaller tensors with shape N % batch_size in the batch dimension. 
        #If your program depends on the batches having the same shape, 
        #consider using the tf.contrib.data.batch_and_drop_remainder transformation instead.

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def test_once(image_path, batch_size, model_checkpoint_path):
    with tf.Graph().as_default():
        sess = tf.Session()
        images, age_labels, gender_labels, file_paths = inputs(
            path=image_path,
            batch_size=batch_size,
            num_epochs=1)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)
        age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
        age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)

        gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                              logits=gender_logits)
        gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)
        total_loss = tf.add_n(
            [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
            name="total_loss")

        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        prob_age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        abs_age_error = tf.losses.absolute_difference(prob_age, age_labels)

        prob_gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)

        mean_error_age, mean_gender_acc, mean_loss = [], [], []

        try:
            while True:  # Train until OutOfRangeError
                prob_gender_val, real_gender, prob_age_val, real_age, image_val, gender_acc_val, abs_age_error_val, cross_entropy_mean_val, file_names = sess.run(
                    [prob_gender, gender_labels, prob_age, age_labels, images, gender_acc, abs_age_error, total_loss,
                     file_paths], {train_mode: False})
                mean_error_age.append(abs_age_error_val)
                mean_gender_acc.append(gender_acc_val)
                mean_loss.append(cross_entropy_mean_val)
                print("Age_MAE:%.2f,Gender_Acc:%.2f%%,Loss:%.2f" % (
                    abs_age_error_val, gender_acc_val * 100, cross_entropy_mean_val))
        except tf.errors.OutOfRangeError:
            print('!!!TESTING DONE!!!')

        sess.close()
        return prob_age_val, real_age, prob_gender_val, real_gender, image_val, np.mean(
            mean_error_age), np.mean(mean_gender_acc), np.mean(mean_loss), file_names
```

## 数据增强

###基本代码
```python
def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'age': tf.FixedLenFeature([], tf.int64),
          'gender': tf.FixedLenFeature([], tf.int64),
          'file_name': tf.FixedLenFeature([], tf.string),
      })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([160 * 160 * 3])
    image = tf.reshape(image, [160, 160, 3])
    image = tf.reverse_v2(image, [-1])
    # image = tf.image.per_image_standardization(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    age = features['age']
    gender = features['gender']
    file_path = features['file_name']

    return image, age, gender, file_path

def augment(image, age, gender, file_path):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image, age, gender, file_path

def normalize(image, age, gender, file_path):
    image = tf.image.per_image_standardization(image)
    return image, age, gender, file_path

def inputs(path, batch_size, num_epochs, allow_smaller_final_batch=False):
    """Reads input data num_epochs times.
    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    # filename = os.path.join(FLAGS.train_dir,
    #                       TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(path)

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode)
        dataset = dataset.map(augment)
        dataset = dataset.map(normalize)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        #the final batch contain smaller tensors with shape N % batch_size in the batch dimension. 
        #If your program depends on the batches having the same shape, 
        #consider using the tf.contrib.data.batch_and_drop_remainder transformation instead.

        iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
```

###基础知识

#### 对比度，饱和度，色相，亮度

**J因而根据分析，一般色相不去动（颜色就变了），明度和饱和度去动（因为可能环境会有灯光影响），对比度就不动了（默认最亮颜色和最暗间的阶层是不变的）。**

对比度，则表示指图像的最亮部分与最暗部分亮度的比值。

色相：色彩的相貌。我们常说的 红色黄色橙色绿色蓝色。。这些就是对色相的一个描述。
![](picture/hue.png)

明度：色彩的明亮程度。我们可以通俗的理解成，这个颜色加了多少白颜料，加了多少黑颜料。
![](picture/brightness.png)

纯度：色彩的饱和度。从左至右，纯度由高变低。纯度越高，颜色越正，越容易分辨。所以我们在日常生活中看到的那些 一眼看不出是什么颜色的颜色大致就是纯度低的颜色。
![](picture/saturation.png)

#### 参数设置

```python
#!/usr/bin/python
# coding:utf-8

# 图像预处理
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 调整亮度.对比度.饱和度.色相的顺序可以得到不同的结果
# 预处理时随机选择的一种,降低无关因素对模型的影响
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering ==1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==2:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==3:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==4:
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering ==5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)

# 给定解码后的图像.目标图像的尺寸以及图像上的标注框
def preprocess(image, height, width, bbox):
    # 若没有提供标注框则默认为关注区域为整个图像
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像数据类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 随机截取图像减小识别物体大小对模型的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 随机调整图像的大小
    distorted_image = tf.image.resize_images(distorted_image, (height, width), method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image

# 获取图像
image_raw_data = tf.gfile.FastGFile('daibola.jpg', 'r').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.1, 0.32, 0.8, 0.7]]])
    # 获得9种不同的图像并显示结果
    for i in range(9):
        # 将图像大小调整为200*200
        result = preprocess(img_data, 200, 200, boxes)
        plt.subplot(331+i), plt.imshow(result.eval()), plt.title(str(i+1))
    plt.show()
```

#### 测试代码

```python
#coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
 
image_raw_data = tf.gfile.FastGFile('cat.jpg','rb').read()
 
with tf.Session() as sess:
     img_data = tf.image.decode_jpeg(image_raw_data)
     plt.subplot(331)
     plt.title("Original")
 
     plt.imshow(img_data.eval())
     #plt.show()
 
     resized = tf.image.resize_images(img_data, [100, 100], method=0)
     # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
     print("Digital type: ", resized.dtype)
     resized = np.asarray(resized.eval(), dtype='uint8')
     # tf.image.convert_image_dtype(rgb_image, tf.float32)
     plt.subplot(332)
     plt.title("100*100")
     plt.imshow(resized)
 
     #plt.show()
 
     croped = tf.image.resize_image_with_crop_or_pad(img_data, 500, 500)
     padded = tf.image.resize_image_with_crop_or_pad(img_data, 1500, 1500)
     plt.subplot(333)
     plt.title("500*500")
     plt.imshow(croped.eval())
    # plt.show()
     plt.subplot(334)
     plt.title("1500*1500")
     plt.imshow(padded.eval())
     #plt.show()
 
     central_cropped = tf.image.central_crop(img_data, 0.5)
     plt.subplot(335)
     plt.title("*0.5")
     plt.imshow(central_cropped.eval())
#     plt.show()
 
     # 上下翻转
     flipped1 = tf.image.flip_up_down(img_data)
     plt.subplot(336)
     plt.title("up-down")
     plt.imshow(flipped1.eval())
     #plt.show()
     # 左右翻转
     flipped2 = tf.image.flip_left_right(img_data)
     plt.subplot(337)
     plt.title("left-right")
     plt.imshow(flipped2.eval())
     #plt.show()
     # 对角线翻转
     transposed = tf.image.transpose_image(img_data)
     plt.subplot(338)
     plt.title("transpose")
     plt.imshow(transposed.eval())
    # plt.show()
 
     flipped3 = tf.image.random_flip_up_down(img_data)
     plt.subplot(339)
     plt.title("flip-up-down")
     plt.imshow(flipped3.eval())
     plt.show()
#————————————————————————————————————————————#
     # 将图片的亮度-0.5。
     adjusted = tf.image.adjust_brightness(img_data, -0.5)
     plt.subplot(331)
     plt.imshow(adjusted.eval())
 
     plt.title("bright-0.5")
     #plt.show()
 
     # 将图片的亮度0.5
     adjusted = tf.image.adjust_brightness(img_data, 0.5)
     plt.subplot(332)
     plt.imshow(adjusted.eval())
 
     plt.title("bright+0.5")
     #plt.show()
     # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
     adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
     plt.subplot(333)
     plt.imshow(adjusted.eval())
 
     plt.title("bright-random")
     #plt.show()
     # 将图片的对比度-5
     adjusted = tf.image.adjust_contrast(img_data, -5)
     plt.subplot(334)
     plt.imshow(adjusted.eval())
     plt.title("contrast-5")
     #plt.show()
     # 将图片的对比度+5
     adjusted = tf.image.adjust_contrast(img_data, 5)
     plt.subplot(335)
     plt.imshow(adjusted.eval())
 
     plt.title("contrast+5")
     #plt.show()
     # 在[lower, upper]的范围随机调整图的对比度。
     adjusted = tf.image.random_contrast(img_data, 0.1, 0.6)
     plt.subplot(336)
     plt.imshow(adjusted.eval())
     plt.title("contrast-random")
     #plt.show()
 
     # 调整图片的色相
     adjusted = tf.image.adjust_hue(img_data, 0.1)
     plt.subplot(337)
     plt.imshow(adjusted.eval())
     plt.title("hue_0.1")
     #plt.show()
 
     # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
     adjusted = tf.image.random_hue(img_data, 0.5)
     plt.subplot(338)
     plt.imshow(adjusted.eval())
     plt.title("hue-random_0.5")
     #plt.show()
 
     # 将图片的饱和度-5。
     adjusted = tf.image.adjust_saturation(img_data, -2)
     plt.subplot(339)
     plt.title("saturation-2")
     plt.imshow(adjusted.eval())
     plt.show()
 
     # 在[lower, upper]的范围随机调整图的饱和度。
     #adjusted = tf.image.random_saturation(img_data, 0, 5)
 
     # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
     #adjusted = tf.image.per_image_standardization(img_data)
```



## Reference

- [Reading data](https://www.tensorflow.org/api_guides/python/reading_data)
- [Importing Data](https://www.tensorflow.org/guide/datasets)
- [Building an image data pipeline](https://cs230-stanford.github.io/tensorflow-input-data.html#building-an-image-data-pipeline)