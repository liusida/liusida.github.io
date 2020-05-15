---
layout: post
title: Cross Entropy 的通俗意义
---

cross_entropy 公式如下：

$$ CrossEntropy = - \sum_i( L_i \cdot \log( S_i ) ) $$

它描述的是**可能性 S 到 L 的距离**，也可以说是描述**用 S 来描述 L 还需要多少信息**（如果是以2为底的log，则代表还需要多少bit的信息；如果是以10为底的log，则代表还需要多少位十进制数的信息）。

当年 香农 Shannon 创立信息论的时候，考虑的是每一次都是扔硬币，结果只有2个可能，所以用的是以2为底，发明了bit计量单位。

而软件实现，例如 Tensorflow 里的实现，则是使用以 e 为底的log。

Tensorflow 中有个经常用到的函数叫 `tf.nn.softmax_cross_entropy_with_logits` 。这个函数的实现并不在 Python 中，所以我用 Numpy 实现一个同样功能的函数进行比对，确认它使用的是以 e 为底的log。理由很简单，因为 Softmax 函数里使用了 e 的指数，所以当 Cross Entropy 也使用以 e 的log，然后这两个函数放到一起实现，可以进行很好的性能优化。

其中对于 logits 这个称呼，我仍然没有明白是为什么。

```python
import tensorflow as tf
import numpy as np


# Make up some testing data, need to be rank 2

x = np.array([
		[0.,2.,1.],
		[0.,0.,2.]
		])
label = np.array([
		[0.,0.,1.],
		[0.,0.,1.]
		])


# Numpy part #

def softmax(logits):
    sf = np.exp(logits)
    sf = sf/np.sum(sf, axis=1).reshape(-1,1)
    return sf

def cross_entropy(softmax, labels):
	return -np.sum( labels * np.log(softmax), axis=1 )

def loss(cross_entropy):
	return np.mean( cross_entropy )

numpy_result = loss(cross_entropy( softmax(x), label ))

print(numpy_result)

# Tensorflow part #

g = tf.Graph()
with g.as_default():
	tf_x = tf.constant(x)
	tf_label = tf.constant(label)
	tf_ret = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(tf_x,tf_label) )

with tf.Session(graph=g) as ss:
	tensorflow_result = ss.run([tf_ret])

print(tensorflow_result)

```

## 附各公式
### 1. Softmax
$$ S_i = { e^{X_i} \over \sum_j( e^{X_j})} $$
这里的 X 就是 logits，S 表示一次判断，Si 表示一次判断中的第i个选项。
### 2. Cross Entropy
$$ D = - \sum_i( L_i \cdot \log( S_i ) ) $$
这里 D 表示一次判断，Li 是一次判断中一个 label 的第 i 个选项。log 是以 e 为底。
### 3. loss
$$ loss = {1\over N} \sum_k(Dk) $$
这里的 Dk 表示第 k 次判断，N 表示总次数，也就是取平均值。
