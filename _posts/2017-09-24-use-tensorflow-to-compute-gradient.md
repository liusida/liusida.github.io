---
layout: post
title: Use Tensorflow to Compute Gradient
---
In most of Tensorflow tutorials, we use minimize(loss) to automatically update parameters of the model.

In fact, minimize() is an integration of two steps: computing gradients, and applying the gradients to update parameters.

Let's take a look at an example:

$$
Y = (100 - 3W - B)^2
$$

What is the gradient of W and B **when W=1.0, B=1.0**?

We can calculate them by hand:

let $$N = 100 - 3W - B$$, so that $$Y = N^2$$


$$
\frac{\partial{Y}}{\partial{W}} = 
\frac{\partial{Y}}{\partial{N}} * \frac{\partial{N}}{\partial{W}} = 
2N * 3 = 600 - 18W - 6B = 576
$$


$$
\frac{\partial{Y}}{\partial{B}} = 
\frac{\partial{Y}}{\partial{N}} * \frac{\partial{N}}{\partial{B}} = 
2N * 1 = 200 - 3W - B = 196
$$

ok, now let use tensorflow to compute that:

```python
import tensorflow as tf

# make an example:
# Y = (100 - W X - B)^2
X = tf.constant(3.)
W = tf.Variable(1.)
B = tf.Variable(1.)
Y = tf.square(100 - W*X - B)

#the lr here is not about gradient computing. it only effect when appling
Ops = tf.train.GradientDescentOptimizer(learning_rate=0.001)
grads_and_vars = Ops.compute_gradients(Y)
# we can modify the gradient here and then:
# Op_update = Ops.apply_gradients(grads_and_vars)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print(sess.run(grads_and_vars))
```

run it, and we get:
```
[(-576.0, 1.0), (-192.0, 1.0)]
```

So next time your professor ask you to implement a back-propagation for some complex networks by your self, maybe this trick can help you double-check your implementation. Hooray!
