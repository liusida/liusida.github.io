---
layout: post
title: 学习Tensorflow的LSTM的RNN例子
---

前几天写了[学习Embeddings的例子][embeddings]，因为琢磨了各个细节，自己也觉得受益匪浅。于是，开始写下一个LSTM的教程吧。

还是[Udacity上那个课程][udacity]。

[源码也在Github上][github-source]。

RNN是一个**非常棒**的技术，可能它已经向我们揭示了“活”的意义。RNN我已经尝试学习了几次，包括前面我[这篇笔记][blog-rnn]，所以就直接进入代码阅读吧。

## 读例子程序：

### 1. 引入库文件

```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
```

### 2. 下载数据

然后下载数据，如果[前面][blog-rnn]已经下载过，那直接把text8.zip拷过来就可以用。

```python
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)
```

### 3. 读入文本

读文件稍微有些不一样，不是处理成list，而是直接读成一个字符串，因为后面用到的就是串数据。

```python
def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()

text = read_data(filename)
print('Data size %d' % len(text))
```

### 4. 生成训练数据集函数

切割一下，留1000个字符做检验，其他99999000个字符拿来训练。

```python
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])
```

### 5. 两个工具函数

建立两个函数`char2id`和`id2char`，用来把字符对应成数字。

本程序只考虑26个字母外加1个空格字符，其他字符都当做空格来对待。所以可以用两个函数，通过ascii码加减，直接算出对应的数值或字符。

```python
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))
```

### 6. 生成训练数据集函数

这次`BatchGenerator`做的比前两天的那个要认真了，用了成员变量来记录位置，而不是用全局变量。

用`BatchGenerator.next()`方法，可以获取一批子字符串用于训练。

`batch_size`是每批几串字符串，`num_unrollings`是每串子字符串的长度（实际上字符串开头还加了上一次获取的最后一个字符，所以实际上字符串长度要比`num_unrollings`多一个）。

```python
batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches
```

真不愧是优秀程序员写的代码，这个函数写的又让我学习了！

它在初始化的时候先根据`batch_size`把段分好，然后设立一组游标`_cursor`，是一组哦，不是一个哦！然后定义好`_last_batch`看或许到哪了。

然后获取需要的字符串的时候，是一批一批的获取各个字符。

这样做，就可以针对整段字符串均匀的取样，从而避免某些地方学的太细，某些地方又没有学到。

值得注意的是，在RNN准备数据的时候，所喂数据的结构是很容易搞错的。在前面博客中，也有很多同学对于他使用`transpose`的意义没法理解。这里需要详细记录一下。

`BatchGenerator.next()`返回的数据格式，是一个list，list的长度是`num_unrollings+1`，每一个元素，都是一个(`batch_size`,27)的array，27是`vocabulary_size`，一个27维向量代表一个字符，是one-hot encoding的格式。

所以，**喂这一批数据进神经网络的时候，理论上是先进去一批的首字符，然后再进去同一批的第二个字符，然后再进去同一批的第三个字符...**

也就是说，下图才是真正的RNN的结构，我们要做的，是按照顺序一个一个的按顺序把东西喂进去。这个图，我看到名字叫`RNN-rolled`：

![RNN-rolled](/images/2016-11-16-study-lstm/RNN-rolled.png)

我们平时看到的向右一路展开的RNN其实向右方向（我用了虚线）是代表先后顺序（同时也带记忆数据流），跟上下方向意义是不一样的。有没有同学误解那么一排东西是可以同时喂进去的？这个图，我看到名字叫`RNN-unrolled`。

![RNN-unrolled](/images/2016-11-16-study-lstm/RNN-unrolled.png)


### 7. 另外两个工具函数

再定义两个用来把训练数据转换成可展现字符串的函数。

`characters`先从one-hot encoding变回数字，再用id2char变成字符。

`batches2string`则将训练数据变成可以展现的字符串。高手这么一批一批的处理数据逻辑还这么绕，而不是按凡人逻辑一个一个的处理让我觉得有点窒息的感觉，自感智商捉急了。

```python
def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))
```

### 8. 另外四个工具函数

四个函数，用途未知

```python
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]
```

### 9. 定义Tensorflow模型

分为几个部分：定义变量，定义LSTM Cell，定义输入接口，循环执行LSTM Cell，定义loss，定义优化，定义预测。

num_nodes 是代表这个神经网络中LSTM Cell层的Cell个数。

```python
num_nodes = 64

graph = tf.Graph()
with graph.as_default():
```

#### 1) 定义变量

```python

  # Parameters:
  # Input gate: input, previous output, and bias.
  ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
```

LSTM Cell 首先有三个门，input output forget三门。

Memory cell 暂时不知道是个什么。

saved_output 是向上的产出，saved_state 是自己的状态记忆。

w 和 b 是最后用来做一个 full connection 的标准神经网络层，把结果变为 vocabulary_size 个之一。

#### 2) 定义LSTM Cell

```python
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    return output_gate * tf.tanh(state), state
```

这里定义的 LSTM Cell 似乎并不是我们平时熟悉的那种，而是如下图（http://arxiv.org/pdf/1402.1128v1.pdf）：

![lstm for chars](/images/2016-11-16-study-lstm/lstm-for-chars.png)

初看这个图可能不是很能理解，于是我重新画了一下：

![lstm model](/images/2016-11-16-study-lstm/lstm-model.png)

我手画的图例解释：

(1) \\(\otimes\\)代表两个数据源乘上参数后相加。\\(\oplus\\)代表两个数据源相加。

(2) \\(\otimes\\)外面再加花边的，代表两个数据源相乘后再取 `sigmoid` 。

(3) 圆圈里是 \\(g\\) 的，代表取 `tanh` 。

(4) \\(State_{-1}\\) 下标-1代表这是上一次迭代时的结果。

> 回想一下，`sigmoid` 函数产生一个(0,1)的数，`tanh` 函数产生一个(-1,1)的数。

作为对比，我再引用一个我认为画的最完美的标准 LSTM Cell 图，来自 [Colah 的博客][colah]：

![LSTM3-chain](/images/2016-11-16-study-lstm/LSTM3-chain.png)

Colah 图例解释：

(1) 方形中带 \\(\sigma\\) ，代表两个数据源连接在一起后乘参数，再取 `sigmoid` 。（**嗯，这里有不同**：Colah 博客中标准的 LSTM Cell 中，这里的操作是先接在一起，再乘参数，而我们这里是先各自乘参数，再相加。）

(2) 方形中带 \\(tanh\\) ，代表两个数据源连接在一起后乘参数，再取 `tanh` 。（**这里也是**）

(3) 椭圆形中带 \\(tanh\\)， 代表直接取 `tanh` 。

(4) \\(\otimes\\)代表两个数据源相乘。\\(\oplus\\)代表两个数据源相加。

(5) 两条从过去\\(-1\\)到当前 Cell 再到未来\\(+1\\)的横向黑色线条箭头，上方代表 `state`，下方代表 `output`。

所以像论文里指出的，这里实现的 LSTM Cell 含有更多参数，效果更好？这种比较目前超出我的认知范围，以后再细看。

#### 3) 定义输入接口

```python
  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  # labels are inputs shifted by one time step.
```

#### 4) 循环执行LSTM Cell

```python
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)
```

#### 5) 定义loss

```python
  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output),
                                saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))
```

#### 6) 定义优化

```python
  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)
```

#### 7) 定义预测

```python
  # Predictions.
  train_prediction = tf.nn.softmax(logits)

  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  sample_output, sample_state = lstm_cell(
    sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
```



#未完待续...

[embeddings]:https://liusida.github.io/2016/11/14/study-embeddings/
[udacity]:https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/63770919610923
[github-source]:https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/6_lstm.ipynb
[blog-rnn]:https://liusida.github.io/2016/11/04/rnn-implementation/
[colah]:http://colah.github.io/posts/2015-08-Understanding-LSTMs/
