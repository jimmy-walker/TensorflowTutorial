# Bert Code

阅读bert代码

##run_classifier.py

主要过程就是：先定义好参数，然后利用数据类读取微调的数据文件，将其每一行转成InputExample对象，然后利用`file_based_convert_examples_to_features`保存到TFrecord中。然后定义好input_fn和model_fn给Estimator，进行推断。

### 定义参数

其中`data_dir`即本次需要微调的新数据；`vocab_file`为bert模型预训练时候用的词典；`uncased`表示不保留大小写，都变成小写；`cased`表示保留大小写。

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
```

### main函数初始化

数据文件读取的类DataProcessor, 官方自带了4个不同数据集(Xnli, Mnli, Mrpc和Cola)的子类。

从json文件中读取bert配置。

```
{
"attention_probs_dropout_prob": 0.1, #乘法attention时，softmax后dropout概率 
"hidden_act": "gelu", #激活函数 
"hidden_dropout_prob": 0.1, #隐藏层dropout概率 
"hidden_size": 768, #隐藏单元数 
"initializer_range": 0.02, #初始化范围 
"intermediate_size": 3072, #升维维度
"max_position_embeddings": 512, #一个大于seq_length的参数，用于生成position_embedding 
"num_attention_heads": 12, #每个隐藏层中的attention head数 
"num_hidden_layers": 12, #隐藏层数 
"type_vocab_size": 2, #segment_ids类别 [0,1] 
"vocab_size": 30522 #词典中词数
}
```

新建分词器。

```python
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  #从json文件中读取bert配置
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()
  #分词
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir) 
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs) #多次迭代的总步数，这里前面表示一行中的两句除以每次的batch值，就能得到一次迭代内的步数
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion) 
```



#### 数据文件读取

定义了DataProcessor这个抽象基类，定义了get_train_examples、get_dev_examples、get_test_examples和get_labels等4个需要子类实现的方法，另外提供了一个_read_tsv函数用于读取tsv文件。 

```python
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines
```

####子类（文本中每行变成一个InputExample对象）

对于MRPC任务，这里定义了MrpcProcessor来基础DataProcessor。我们来看其中的get_labels和get_train_examples，其余两个抽象方法是类似的。首先是get_labels，它非常简单，这任务只有两个label。

```python
def get_labels(self): 
  return ["0", "1"]
```

接下来是get_train_examples：

```python
def get_train_examples(self, data_dir):
  return self._create_examples(
		  self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
```

这个函数首先使用\_read\_tsv读入训练文件train.tsv，然后使用\_create\_examples函数把每一行变成一个InputExample对象。

```python
def _create_examples(self, lines, set_type):
  examples = [] #生成一个列表，里面存放的都是InputExample对象
  for (i, line) in enumerate(lines):
	  if i == 0:
		  continue
	  guid = "%s-%s" % (set_type, i)
	  text_a = tokenization.convert_to_unicode(line[3])
	  text_b = tokenization.convert_to_unicode(line[4])
	  if set_type == "test": #对测试集来说，就不关心其label了，随便给个0
		  label = "0"
	  else:
		  label = tokenization.convert_to_unicode(line[0])
	  examples.append(
		  InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
	  return examples
```

代码非常简单，line是一个list，line[3]和line[4]分别代表两个句子，如果是训练集合和验证集合，那么第一列line[0]就是真正的label，而如果是测试集合，label就没有意义，随便赋值成”0”。然后对于所有的字符串都使用tokenization.convert_to_unicode把字符串变成unicode的字符串。这是为了兼容Python2和Python3，因为Python3的str就是unicode，而Python2的str其实是bytearray，Python2却有一个专门的unicode类型。感兴趣的读者可以参考其实现，不感兴趣的可以忽略。

最终构造出一个InputExample对象来，它有4个属性：guid、text_a、text_b和label，guid只是个唯一的id而已。text_a代表第一个句子，text_b代表第二个句子，第二个句子可以为None，label代表分类标签。

```python
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples
```

#### FullTokenizer（初始化分词器，将用于后续的train/dev/test环节，对中文没啥用）

分词是我们需要重点关注的代码，**因为如果想要把BERT产品化，我们需要使用Tensorflow Serving，Tensorflow Serving的输入是Tensor，把原始输入变成Tensor一般需要在Client端完成**。BERT的分词是Python的代码，如果我们使用其它语言的gRPC Client，那么需要用其它语言实现同样的分词算法，否则预测时会出现问题。 

代码在tokenization.py文件中，目的将词变成更细粒度的词，减少词表数量，loves, loved, loving => lov, ed, ing, es.

```python
class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_tokens_to_ids(self.vocab, tokens)
```



### 建立模型函数（用于构造Estimator的model_fn）

```
若在一个函数内部定义了另一个函数,外部的我们暂且称之为外函数,内部的称之为内函数

闭包:在一个外函数中定义了一个内函数,内函数里运用了外函数的临时变量,并且外函数的返回值是内函数的引用,这样就构成了一个闭包

一般情况下,如果一个函数结束,函数内部所有的东西都会被释放掉,还给内存,局部变量也会消失,但是闭包是一种特殊的情况,如果外函数在结束的时候发现有自己的临时变量将来还会在内部变量中用到,就把这个临时变量绑定给了内函数,然后再自己结束.
```

闭包就是回调函数中的内部函数用到了外部函数的参数。好处在于，**这里固定了model_fn的参数个数，那么我们要传入额外的变量，要么用全局变量访问不安全，要么就可以用此闭包的形式**。

####model_fn_bulider（返回Estimator的闭包model_fn）

```python
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)
```

```python
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
				num_train_steps, num_warmup_steps, use_tpu,
				use_one_hot_embeddings): 
	# 注意：在model_fn的设计里，features表示输入(特征)，而labels表示输出
	# 但是这里的实现有点不好，把label也放到了features里。
	def model_fn(features, labels, mode, params): 
		input_ids = features["input_ids"]
		input_mask = features["input_mask"]
		segment_ids = features["segment_ids"]
		label_ids = features["label_ids"]
		
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		
		# 创建Transformer模型，这是最主要的代码。
		(total_loss, per_example_loss, logits, probabilities) = create_model(
			bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
			num_labels, use_one_hot_embeddings)
		
		tvars = tf.trainable_variables()
		
		# 从checkpoint恢复参数
		if init_checkpoint: 
			(assignment_map, initialized_variable_names) = 	
				modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			
			tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
		 
		
		output_spec = None
		# 构造训练的spec
		if mode == tf.estimator.ModeKeys.TRAIN:
			train_op = optimization.create_optimizer(total_loss, learning_rate, 
							num_train_steps, num_warmup_steps, use_tpu)
			
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
					mode=mode,
					loss=total_loss,
					train_op=train_op,
					scaffold_fn=scaffold_fn)
					
		# 构造eval的spec
		elif mode == tf.estimator.ModeKeys.EVAL:	
			def metric_fn(per_example_loss, label_ids, logits):
				predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
				accuracy = tf.metrics.accuracy(label_ids, predictions)
				loss = tf.metrics.mean(per_example_loss)
				return {
					"eval_accuracy": accuracy,
					"eval_loss": loss,
				}
			
			eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				loss=total_loss,
				eval_metrics=eval_metrics,
				scaffold_fn=scaffold_fn)
		
		# 预测的spec
		else:
			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
				mode=mode,
				predictions=probabilities,
				scaffold_fn=scaffold_fn)
		return output_spec
	
	return model_fn
```

####create_model()建立真正的transformer模型

调用modeling.BertModel得到BERT模型，然后使用它的get_pooled_output方法得到[CLS]最后一层的输出，这是一个768(默认参数下)的向量，然后就是常规的接一个全连接层得到logits，然后softmax得到概率，之后就可以根据真实的分类标签计算loss。这时候发现关键的代码是modeling.BertModel。 

```python
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
					labels, num_labels, use_one_hot_embeddings): 
	model = modeling.BertModel(
			config=bert_config,
			is_training=is_training,
			input_ids=input_ids,
			input_mask=input_mask,
			token_type_ids=segment_ids,
			use_one_hot_embeddings=use_one_hot_embeddings)
	
	# 在这里，我们是用来做分类，因此我们只需要得到[CLS]最后一层的输出。
	# 如果需要做序列标注，那么可以使用model.get_sequence_output()
	# 默认参数下它返回的output_layer是[8, 768]
	output_layer = model.get_pooled_output()
	
	# 默认是768
	hidden_size = output_layer.shape[-1].value
	
	
	output_weights = tf.get_variable(
		"output_weights", [num_labels, hidden_size],
		initializer=tf.truncated_normal_initializer(stddev=0.02))
	
	output_bias = tf.get_variable(
		"output_bias", [num_labels], initializer=tf.zeros_initializer())
	
	with tf.variable_scope("loss"):
		if is_training:
			# 0.1的概率会dropout
			output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
			
		# 对[CLS]输出的768的向量再做一个线性变换，输出为label的个数。得到logits
		logits = tf.matmul(output_layer, output_weights, transpose_b=True)
		logits = tf.nn.bias_add(logits, output_bias)
		probabilities = tf.nn.softmax(logits, axis=-1)
		log_probs = tf.nn.log_softmax(logits, axis=-1)
		
		one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
		
		per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
		loss = tf.reduce_mean(per_example_loss)
	
	return (loss, per_example_loss, logits, probabilities)
```

### 利用model_fn和配置config建立Estimator对象

```python
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
```

### 利用Estimator对象进行train/dev/test

通过file_based_convert_examples_to_features函数把输入的tsv文件变成TFRecord文件，便于Tensorflow处理。

```python
  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                 FLAGS.max_seq_length, tokenizer, predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    if FLAGS.use_tpu:
      # Warning: According to tpu_estimator.py Prediction on TPU is an
      # experimental feature and hence not supported here
      raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for prediction in result:
        output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
        writer.write(output_line)

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

```

####将微调数据变成TFRecord文件，file_based_convert_examples_to_features，其中传入分词器，InputExample->InputFeature->tf.train.Feature->tf.train.Example->TFRecord

file_based_convert_examples_to_features函数遍历每一个example(InputExample类的对象)。然后使用convert_single_example函数把每个InputExample对象变成InputFeature。InputFeature就是一个存放特征的对象，它包括input_ids、input_mask、segment_ids和label_ids，这4个属性除了label_ids是一个int之外，其它都是int的列表，因此使用create_int_feature函数把它变成tf.train.Feature，最后构造tf.train.Example对象，然后写到TFRecord文件里。后面Estimator的input_fn会用到它。

**注意file_based_convert_examples_to_features中会传入分词器，这样就意味着，微调数据，基本就是按照bert预训练的词典，进行转换。**

```python
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
		    train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

def file_based_convert_examples_to_features(
				examples, label_list, max_seq_length, tokenizer, output_file):

	writer = tf.python_io.TFRecordWriter(output_file)
	
	for (ex_index, example) in enumerate(examples):
	 
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

##### 将输入InputExample对象变成向量InputFeature对象:convert_single_example

```python
def convert_single_example(ex_index, example, label_list, max_seq_length,
				tokenizer):
	"""把一个`InputExample`对象变成`InputFeatures`."""
	# label_map把label变成id，这个函数每个example都需要执行一次，其实是优化的。
	# 只需要在可以再外面执行一次传入即可。
	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i
	
	tokens_a = tokenizer.tokenize(example.text_a)
	tokens_b = None
	if example.text_b:
		tokens_b = tokenizer.tokenize(example.text_b)
	
	if tokens_b:
		# 如果有b，那么需要保留3个特殊Token[CLS], [SEP]和[SEP]
		# 如果两个序列加起来太长，就需要去掉一些。
		_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
	else:
		# 没有b则只需要保留[CLS]和[SEP]两个特殊字符
		# 如果Token太多，就直接截取掉后面的部分。
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[0:(max_seq_length - 2)]
	
	# BERT的约定是：
	# (a) 对于两个序列：
	#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
	#  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
	# (b) 对于一个序列：
	#  tokens:   [CLS] the dog is hairy . [SEP]
	#  type_ids: 0     0   0   0  0     0 0
	#
	# 这里"type_ids"用于区分一个Token是来自第一个还是第二个序列
	# 对于type=0和type=1，模型会学习出两个Embedding向量。
	# 虽然理论上这是不必要的，因为[SEP]隐式的确定了它们的边界。
	# 但是实际加上type后，模型能够更加容易的知道这个词属于那个序列。
	#
	# 对于分类任务，[CLS]对应的向量可以被看成 "sentence vector"
	# 注意：一定需要Fine-Tuning之后才有意义
	tokens = []
	segment_ids = []
	tokens.append("[CLS]")
	segment_ids.append(0)
	for token in tokens_a:
		tokens.append(token)
		segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)
	
	if tokens_b:
		for token in tokens_b:
			tokens.append(token)
			segment_ids.append(1)
		tokens.append("[SEP]")
		segment_ids.append(1)
	
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	
	# mask是1表示是"真正"的Token，0则是Padding出来的。在后面的Attention时会通过tricky的技巧让
	# 模型不能attend to这些padding出来的Token上。
	input_mask = [1] * len(input_ids)
	
	# padding使得序列长度正好等于max_seq_length
	while len(input_ids) < max_seq_length:
		input_ids.append(0)
		input_mask.append(0)
		segment_ids.append(0)
 
	label_id = label_map[example.label]
	#返回InputFeatures对象
	feature = InputFeatures(
		input_ids=input_ids,
		input_mask=input_mask,
		segment_ids=segment_ids,
		label_id=label_id)
	return feature
```

如果两个Token序列的长度太长，那么需要去掉一些，这会用到_truncate_seq_pair函数：

```
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()
```

这个函数很简单，如果两个序列的长度小鱼max_length，那么不用truncate，否则在tokens_a和tokens_b中选择长的那个序列来pop掉最后面的那个Token，这样的结果是使得两个Token序列一样长(或者最多a比b多一个Token)。

#### 定义Estimator的数据读取函数input_fn，读取保存的TFRecord：file_based_input_fn_builder

input_fn，它是由file_based_input_fn_builder构造出来的，闭包。

用于让Estimator读取保存好的TFRecord对象，在train_file文件夹中。

```python
def file_based_input_fn_builder(input_file, seq_length, is_training,
			drop_remainder):
 
	name_to_features = {
		"input_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"input_mask": tf.FixedLenFeature([seq_length], tf.int64),
		"segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
		"label_ids": tf.FixedLenFeature([], tf.int64),
	}
	
	def _decode_record(record, name_to_features):
		# 把record decode成TensorFlow example.
		example = tf.parse_single_example(record, name_to_features)
		
		# tf.Example只支持tf.int64，但是TPU只支持tf.int32.
		# 因此我们把所有的int64变成int32.
		for name in list(example.keys()):
			t = example[name]
			if t.dtype == tf.int64:
				t = tf.to_int32(t)
			example[name] = t
		
		return example
	
	def input_fn(params): 
		batch_size = params["batch_size"]
		
		# 对于训练来说，我们会重复的读取和shuffling 
		# 对于验证和测试，我们不需要shuffling和并行读取。
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

这个函数返回一个函数input_fn。这个input_fn函数首先从文件得到TFRecordDataset，然后根据是否训练来shuffle和重复读取。然后用apply函数对每一个TFRecord进行map_and_batch，调用_decode_record函数对record进行parsing。从而把TFRecord的一条Record变成tf.Example对象，这个对象包括了input_ids等4个用于训练的Tensor。 

#### 执行操作

```python
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
```

##modeling.py

###Bert构造函数

输入向量的说明：**对于一个输入向量，如果是两个句子的任务，则是拼接在一个向量中，如下所示，然后如果有位置不足的，就补充pad0假数据，因此首先需要判断哪些是假数据，然后再对真数据判断，哪两句话。而batch_size就是表示有多少个输入向量一起进入训练，表示并行训练，一个batch内的输入向量不会互相干扰，只是最后算loss更新时候会用到对所有输入向量的结果进行汇总。**

```
1. tokens = ["[CLS], "it", "is" "a", "[MASK]", "day", "[SEP]", "I", "apple", "to", "go", "out", "[SEP]"]
而input_ids就是把上述变成id的形式，方便输入网络训练。
而input_mask中mask是1表示是"真正"的Token，0则是Padding出来的。
2. segment_ids=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
其表示两个不同句子
3. is_random_next=False
4. masked_lm_positions=[4, 8, 9] 
   表示Mask后为["[CLS], "it", "is" "a", "[MASK]", "day", "[SEP]", "I", "[MASK]", "to", "go", "out", "[SEP]"]
masked_lm_positions记录哪些位置被Mask了。
5. masked_lm_labels=["good", "want", "to"]
而masked_lm_labels记录被Mask之前的词。
```

其中`all_encoder_layers`就是把每个transformer block的输出值保存下来。

`BertModel.sequence_output` 是取最后attenion层的输出。` BertModel.pooled_output` 取`sequence_output`的第一个token“CLS”的emb，然后加个连接层。 

```python
def __init__(self,
		  config,
		  is_training,
		  input_ids,
		  input_mask=None,
		  token_type_ids=None,
		  use_one_hot_embeddings=True,
		  scope=None): 

  # Args:
  #       config: `BertConfig` 对象
  #       is_training: bool 表示训练还是eval，是会影响dropout
  #	  input_ids: int32 Tensor  shape是[batch_size, seq_length]
  #	  input_mask: (可选) int32 Tensor shape是[batch_size, seq_length]
  #	  token_type_ids: (可选) int32 Tensor shape是[batch_size, seq_length]
  #	  use_one_hot_embeddings: (可选) bool
  #		  如果True，使用矩阵乘法实现提取词的Embedding；否则用tf.embedding_lookup()
  #		  对于TPU，使用前者更快，对于GPU和CPU，后者更快。
  #	  scope: (可选) 变量的scope。默认是"bert"
  
  # Raises:
  #	  ValueError: 如果config或者输入tensor的shape有问题就会抛出这个异常

  config = copy.deepcopy(config) #对config(BertConfig对象)深度拷贝一份
  if not is_training: #如果不是训练，那么把dropout都置为零
	  config.hidden_dropout_prob = 0.0
	  config.attention_probs_dropout_prob = 0.0
  
  input_shape = get_shape_list(input_ids, expected_rank=2) #get_shape_list就是得到该输入向量的具体维度，以便下面用到
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  #如果输入的input_mask为None，那么构造一个shape合适值全为1的input_mask，这表示输入都是”真实”的输入，没有padding的内容。
  if input_mask is None:
	  input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
  #如果token_type_ids为None，那么构造一个shape合适并且值全为0的tensor，表示所有Token都属于第一个句子。
  if token_type_ids is None:
	  token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
  
  with tf.variable_scope(scope, default_name="bert"):
	  with tf.variable_scope("embeddings"):
		  # 词的Embedding lookup 
		  (self.embedding_output, self.embedding_table) = embedding_lookup(
				  input_ids=input_ids,
				  vocab_size=config.vocab_size,
				  embedding_size=config.hidden_size,
				  initializer_range=config.initializer_range,
				  word_embedding_name="word_embeddings",
				  use_one_hot_embeddings=use_one_hot_embeddings)
		  
		  # 增加位置embeddings和token type的embeddings，然后是
		  # layer normalize和dropout。
		  self.embedding_output = embedding_postprocessor(
				  input_tensor=self.embedding_output,
				  use_token_type=True,
				  token_type_ids=token_type_ids,
				  token_type_vocab_size=config.type_vocab_size,
				  token_type_embedding_name="token_type_embeddings",
				  use_position_embeddings=True,
				  position_embedding_name="position_embeddings",
				  initializer_range=config.initializer_range,
				  max_position_embeddings=config.max_position_embeddings,
				  dropout_prob=config.hidden_dropout_prob)
	  
	  with tf.variable_scope("encoder"):
		  # 把shape为[batch_size, seq_length]的2D mask变成
		  # shape为[batch_size, seq_length, seq_length]的3D mask
		  # 以便后向的attention计算，读者可以对比之前的Transformer的代码。
		  attention_mask = create_attention_mask_from_input_mask(
				  input_ids, input_mask)
		  
		  # 多个Transformer模型stack起来。
		  # all_encoder_layers是一个list，长度为num_hidden_layers（默认12），每一层对应一个值。
		  # 每一个值都是一个shape为[batch_size, seq_length, hidden_size]的tensor。
		  
		  self.all_encoder_layers = transformer_model(
			  input_tensor=self.embedding_output,
			  attention_mask=attention_mask, #传入到transformer中
			  hidden_size=config.hidden_size,
			  num_hidden_layers=config.num_hidden_layers,
			  num_attention_heads=config.num_attention_heads,
			  intermediate_size=config.intermediate_size,
			  intermediate_act_fn=get_activation(config.hidden_act),
			  hidden_dropout_prob=config.hidden_dropout_prob,
			  attention_probs_dropout_prob=config.attention_probs_dropout_prob,
			  initializer_range=config.initializer_range,
			  do_return_all_layers=True)
	  
	  # `sequence_output` 是最后一层的输出，shape是[batch_size, seq_length, hidden_size]
	  self.sequence_output = self.all_encoder_layers[-1]

	  with tf.variable_scope("pooler"):
		  # 取最后一层的第一个时刻[CLS]对应的tensor
		  # 从[batch_size, seq_length, hidden_size]变成[batch_size, hidden_size]
		  # sequence_output[:, 0:1, :]得到的是[batch_size, 1, hidden_size]
		  # 我们需要用squeeze把第二维去掉。
		  first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
		  # 然后再加一个全连接层，输出仍然是[batch_size, hidden_size]
		  self.pooled_output = tf.layers.dense(
				  first_token_tensor,
				  config.hidden_size,
				  activation=tf.tanh,
				  kernel_initializer=create_initializer(config.initializer_range))
```

#### embedding_lookup函数用于实现词的Embedding，即从词变成id 

Embedding本来很简单，使用tf.nn.embedding_lookup就行了。但是为了优化TPU，它还支持使用矩阵乘法来提取词向量。另外为了提高效率，输入的shape除了[batch_size, seq_length]外，它还增加了一个维度变成[batch_size, seq_length, num_inputs]。**如果不关心细节，我们把这个函数当成黑盒，那么我们只需要知道它的输入input_ids(可能)是[8, 128]，输出是[8, 128, 768]就可以了，此外这里还返回了随机初始化的embedding_table**。 

```python
def embedding_lookup(input_ids,
			vocab_size,
			embedding_size=128,
			initializer_range=0.02,
			word_embedding_name="word_embeddings",
			use_one_hot_embeddings=False):
	"""word embedding
	
	Args:
		input_ids: int32 Tensor shape为[batch_size, seq_length]，表示WordPiece的id
		vocab_size: int 词典大小，需要与vocab.txt一致 
		embedding_size: int embedding后向量的大小 
		initializer_range: float 随机初始化的范围 
		word_embedding_name: string 名字，默认是"word_embeddings"
		use_one_hot_embeddings: bool 如果True，使用one-hot方法实现embedding；否则使用 		
			`tf.nn.embedding_lookup()`. TPU适合用One hot方法。
	
	Returns:
		float Tensor shape为[batch_size, seq_length, embedding_size]
	"""
	# 这个函数假设输入的shape是[batch_size, seq_length, num_inputs]
	# 普通的Embeding一般假设输入是[batch_size, seq_length]，
	# 增加num_inputs这一维度的目的是为了一次计算更多的Embedding
	# 但目前的代码并没有用到，传入的input_ids都是2D的，这增加了代码的阅读难度。
	
	# 如果输入是[batch_size, seq_length]，
	# 那么我们把它 reshape成[batch_size, seq_length, 1]
	if input_ids.shape.ndims == 2:
		input_ids = tf.expand_dims(input_ids, axis=[-1])
	
	# 构造Embedding矩阵，shape是[vocab_size, embedding_size]，随机初始化一个张量embedding_table
	embedding_table = tf.get_variable(
		name=word_embedding_name,
		shape=[vocab_size, embedding_size],
		initializer=create_initializer(initializer_range))
	
	if use_one_hot_embeddings:
		flat_input_ids = tf.reshape(input_ids, [-1])
		one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
		output = tf.matmul(one_hot_input_ids, embedding_table)
	else:
		output = tf.nn.embedding_lookup(embedding_table, input_ids) #tf.nn.embedding_lookup选取一个张量里面索引对应的元素
	
	input_shape = get_shape_list(input_ids)
	# 把输出从[batch_size, seq_length, num_inputs(这里总是1), embedding_size]
	# 变成[batch_size, seq_length, num_inputs*embedding_size]
	output = tf.reshape(output,
				input_shape[0:-1] + [input_shape[-1] * embedding_size])
	return (output, embedding_table)
```



#### create_attention_mask_from_input_mask函数用于构造Mask矩阵，解决对pad的非真实词进行忽略

用途：在计算Self-Attention的时候每一个样本都需要一个Attention Mask矩阵，表示每一个时刻可以attend to的范围，1表示可以attend，0表示是padding的。

	Args:
		from_tensor: 2D or 3D Tensor，shape为[batch_size, from_seq_length, ...].
		to_mask: int32 Tensor， shape为[batch_size, to_seq_length].
	
	Returns:
		float Tensor，shape为[batch_size, from_seq_length, to_seq_length]
比如调用它时的两个参数是是：

```
input_ids=[
	[1,2,3,0,0],
	[1,3,5,6,1]
]
input_mask=[
	[1,1,1,0,0],
	[1,1,1,1,1]
]
```

表示这个batch有两个样本，第一个样本长度为3(padding了2个0)，第二个样本长度为5。**在计算Self-Attention的时候每一个样本都需要一个Attention Mask矩阵，表示每一个时刻可以attend to的范围，1表示可以attend，0表示是padding的**(或者在机器翻译的Decoder中不能attend to未来的词)。对于上面的输入，这个函数返回一个shape是[2, 5, 5]的tensor，分别代表两个Attention Mask矩阵。

```
[
	[1, 1, 1, 0, 0], #它表示第1个词可以attend to 3个词
	[1, 1, 1, 0, 0], #它表示第2个词可以attend to 3个词
	[1, 1, 1, 0, 0], #它表示第3个词可以attend to 3个词
	[1, 1, 1, 0, 0], #无意义，因为输入第4个词是padding的0
	[1, 1, 1, 0, 0]  #无意义，因为输入第5个词是padding的0
]

[
	[1, 1, 1, 1, 1], # 它表示第1个词可以attend to 5个词
	[1, 1, 1, 1, 1], # 它表示第2个词可以attend to 5个词
	[1, 1, 1, 1, 1], # 它表示第3个词可以attend to 5个词
	[1, 1, 1, 1, 1], # 它表示第4个词可以attend to 5个词
	[1, 1, 1, 1, 1]	 # 它表示第5个词可以attend to 5个词
]
```

比如前面举的例子，broadcast_ones的shape是[2, 5, 1]，值全是1，而to_mask是

```
to_mask=[
[1,1,1,0,0],
[1,1,1,1,1]
]
```

shape是[2, 5]，reshape为[2, 1, 5]。然后broadcast_ones * to_mask就得到[2, 5, 5]，正是我们需要的两个Mask矩阵，读者可以验证。注意**[batch, A, B]*[batch, B, C]=[batch, A, C]**，我们可以认为是batch个[A, B]的矩阵乘以batch个[B, C]的矩阵。

```python
def create_attention_mask_from_input_mask(from_tensor, to_mask):
	"""Create 3D attention mask from a 2D tensor mask.
	
	Args:
		from_tensor: 2D or 3D Tensor，shape为[batch_size, from_seq_length, ...].
		to_mask: int32 Tensor， shape为[batch_size, to_seq_length].
	
	Returns:
		float Tensor，shape为[batch_size, from_seq_length, to_seq_length].
	"""
	from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
	batch_size = from_shape[0]
	from_seq_length = from_shape[1]
	
	to_shape = get_shape_list(to_mask, expected_rank=2)
	to_seq_length = to_shape[1]
	
	to_mask = tf.cast(
		tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
	
	# `broadcast_ones` = [batch_size, from_seq_length, 1]
	broadcast_ones = tf.ones(
		shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
	
	# Here we broadcast along two dimensions to create the mask.
	mask = broadcast_ones * to_mask
	
	return mask
```

##run_pretraining.py

get_masked_lm_output函数用于计算语言模型的Loss(Mask位置预测的词（即最后一层的输出向量model.get_sequence_output() ）和真实的词是否相同)

`model.get_sequence_output()`的shape是：[batch_size, seq_length, hidden_size]。

**其中表示为：batch_size是一个批次内的样本数（向量数），比如8。**

**而hidden_size则是每一个token被表示成的向量维度（就是一个transformer encoder中一层的维度），比如768。**

**而seq_length则表示有多少个transformer encoder在一个transformer block中。**

`model.get_embedding_table()`的shape是：[vocab_size, embedding_size]。

表示为这些词典中的词随机初始化的向量，embedding_size表示为768。

masked_lm_positions表示mask词的位置，shape是[batch_size, masked_length]

masked_lm_ids表示mask词的id，shape是[batch_size, masked_length]

masked_lm_weights表示哪些是真mask，哪些是没有mask，shape是[batch_size, masked_length]

```python
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
			label_ids, label_weights):
	"""得到masked LM的loss和log概率"""
	# 只需要Mask位置的Token的输出。
	input_tensor = gather_indexes(input_tensor, positions) #返回的格式应该为[batch_size * masked_length, hidden_size]
	
	with tf.variable_scope("cls/predictions"):
		# 在输出之前再加一个非线性变换，这些参数只是用于训练，在Fine-Tuning的时候就不用了。
		with tf.variable_scope("transform"):#貌似论文里这块没有讲!!!!!!!!!!!!!!!!!!!!
			input_tensor = tf.layers.dense( #全连接层，常用于最后限定输出维度
					input_tensor,
					units=bert_config.hidden_size, #表示输出的张量中的最后一个维度的大小
					activation=modeling.get_activation(bert_config.hidden_act),
					kernel_initializer=modeling.create_initializer(
						bert_config.initializer_range))
			input_tensor = modeling.layer_norm(input_tensor) #调用tf.contrib.layers.layer_norm，实现多加一层Layer Normalization 
            #针对 batch normalization 存在的问题 提出了 Layer Normalization 进行改进的。
		
		# output_weights是复用输入的word Embedding，所以是传入的，J就是用之前初始化的词向量
		# 这里再多加一个bias。
		output_bias = tf.get_variable(
				"output_bias",
				shape=[bert_config.vocab_size], #因为是加到行上，所以是列的维度
				initializer=tf.zeros_initializer())
		logits = tf.matmul(input_tensor, output_weights, transpose_b=True) #[batch_size*masked_length,vocab_size] #余弦相似度等于是用相乘来判断与哪些词汇的相似概率最大。每一个预测到的mask的词汇都得到与所有vocab的之间的乘积结果。当角度为0时，二者重合，最相近，此时其余弦值也最大为1。
        #注意在训练阶段，因为model.get_embedding_table也是变量会不断变化的，所以意味着预测的越来越准！！！！！！！！
        ##J注意而在predict中这里是因为已经导入了bert训练好的模型的embedding_table，所以loss值直接计算出来，另外output_bias也是，注意这两者都是共享变量的方式导入。
		logits = tf.nn.bias_add(logits, output_bias) #再给他们加一个向量，每行加一个偏置项
		log_probs = tf.nn.log_softmax(logits, axis=-1) #返回仍是[batch_size*masked_length,vocab_size]
        #就是用logsoftmax变成损失值
		#J注意即masked_length表示是最大的mask的token数，不一定全部mask了，所以要配合label_weights一起计算，这里label_weights has a value of 1.0 for every real prediction and 0.0 for the padding predictions.。即对label_ids进行进一步过滤。
		# label_ids的长度是20，表示最大的MASK的Token数
		# label_ids里存放的是MASK过的Token的id
		label_ids = tf.reshape(label_ids, [-1]) #把这些token id进行平铺成一维[batch_size*masked_length]
		label_weights = tf.reshape(label_weights, [-1]) #把这些token 真假的标志进行平铺成一维 [batch_size*masked_length]
		
		one_hot_labels = tf.one_hot(
			label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
		#返回为[batch_size*masked_length , vocab_size] 即不用实际向量表示了，用独热编码表示了
		# 但是由于实际MASK的可能不到20，比如只MASK18，那么label_ids有2个0(padding)
		# 而label_weights=[1, 1, ...., 0, 0]，说明后面两个label_id是padding的，计算loss要去掉。
		per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1]) #只去记录真实词对应的loss损失，然后reduce就会把其他0*loss+真实词*loss汇总起来，最后取负数，越大说明损失越大。返回：[batch_size*masked_length]
		numerator = tf.reduce_sum(label_weights * per_example_loss) #利用label_weights进行区分哪些是真实的，返回一个数
		denominator = tf.reduce_sum(label_weights) + 1e-5 #表示真实词的数目，其中1e-5表示不能使得概率为1
		loss = numerator / denominator
	
	return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0] #将model.get_sequence_output()的shape是：[batch_size, seq_length, hidden_size]分别提取到，8
  seq_length = sequence_shape[1] #128
  width = sequence_shape[2] #768

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1]) #变成[[0], [128] ..., [896]]形式，这样就知道batch中每个样本（句子向量）的开始位置，每个都是128
  flat_positions = tf.reshape(positions + flat_offsets, [-1]) #假设positions[4,6,9]变成[  4   6   9 132 134 137 260 262 265 388 390 393 516 518 521 644 646 649 772 774 777 900 902 905] 这样就知道了具体句子中哪些是mask的词，所有句子都是一样的
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width]) #等于是把model.get_sequence_output()中batch的样本，按照上述样式，拼接起来，就可以知道mask的词语的被预测的向量值了。
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor #返回的格式应该为： [batch_size * masked_length, hidden_size]
```



## Reference

- [BERT代码阅读](http://fancyerii.github.io/2019/03/09/bert-codes/)

## Code

### run_classifier.py

```python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature


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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    if FLAGS.use_tpu:
      # Warning: According to tpu_estimator.py Prediction on TPU is an
      # experimental feature and hence not supported here
      raise ValueError("Prediction in TPU not supported")

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      tf.logging.info("***** Predict results *****")
      for prediction in result:
        output_line = "\t".join(
            str(class_probability) for class_probability in prediction) + "\n"
        writer.write(output_line)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

```


