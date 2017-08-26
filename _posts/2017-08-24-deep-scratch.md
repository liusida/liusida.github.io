---
layout: post
title: Implement a Deep Neural Network using Python and Numpy
---

I have just finished [Andrew Ng's new Coursera courses of Deep Learning (1-3)][1]. They are one part of his new project [DeepLearning.ai][2].

In those courses, there is a series of interview of [Heroes of Deep Learning][3], which is very helpful for a newbie to enter this field. I heard several times those masters require a newbie to build a whole deep network from scratch, maybe just use Python and Numpy, to understand things better. So, after the courses, I decided to build one on my own.

## Day 1

I create a new repository named [Deep Scratch][4] in github, and a [main.ipynb][5] file, to do this job.

First of all, I made a todo list, those are functions or algorithms metioned in the courses. I planed to implement most of them.

Second, I decided to begin with MNIST dataset. It is the Hello, World dataset. But I will switch to other dataset to test my model. In my mind, maybe too ambitious, I want to build a transferable model, I think that's the correct direction to general thinking.

After these, I can now start this project.

## Day 2

I implemented a basic model, including those functions:

```python
ReLU(X)
softmax(X)
forward_propagation_each_layer(W, A_prev, b, activation_function=ReLU)
loss(Y_hat, Y)
cost(loss)
predict(Y_hat)
accuracy(Y_predict, Y)
backpropagate_cost(Y, AL)
backpropagate_softmax(AL, dAL, Y=None, ZL=None)
backpropagate_linear(dZ, W, A_prev)
backpropagate_ReLU(dA, Z)

forwardpropagation_all(X)
backpropagate_all(X, Y)
update_parameters()

model(X, Y, learning_rate=0.01, print_every=100, iteration=500, hidden_layers=[100], batch_size=128)
```

I had to say, the math is complex for me. When I implemented first time, I almost have 10 bugs in calculation!

There are a few unconcrete concepts, such as what loss function should I use for multi-class classification? What is the derivative of softmax? When should I divide the result by m (the number of examples)?

After maybe 10 hours of debug, I even implement a bunch of tensorflow alternative functions, finally, the model work out!

[Day 2's notebook][day2] <- Here is the code and formula. (I found that Jupyter Notebook is great to comment codes!)

## Day 3

The training set accuracy was already 1.0, so I looked at the dev set accuracy: 0.6? Oh, that's bad. So I had a variance problem.

I tried to implement regularization, but seems had little help to this variance problem.

Then suddenly I figured out why: because I use random batch to train, the distribution of random batch can not cover all training examples, so I wasted a lot of training examples.

So I changed to mini-batch, which define a mini-batch size, every time use a segment of training data, so it can sure every single example was used for training.

And I also realize a very interesting aspect of mini-batch, it has a very good effect to variance problem, especially when the network is relatively shallow. I think it acts just like dropout! The network can not depend on any single data!

Thanks to mini-batch, my dev accuracy jumped to 0.98, and test set accuracy was also 0.98. Not bad for today's work!

[Day 3's notebook][day3] <- Here is mini-batch, regularization, train/dev/test accuracy.

## Day 4

Since the code was work and was ugly... I decided to refactory the code.

[Day 4's notebook][day4] <- I have refactoried half of the code.

  

[1]:https://www.coursera.org/specializations/deep-learning
[2]:https://www.deeplearning.ai/
[3]:https://youtu.be/-eyhCTvrEtE?list=PLfsVAYSMwsksjfpy8P2t_I52mugGeA5gR
[4]:https://github.com/liusida/DeepScratch
[5]:https://github.com/liusida/DeepScratch/blob/master/main.ipynb
[day2]:https://github.com/liusida/DeepScratch/blob/day2/main.ipynb
[day3]:https://github.com/liusida/DeepScratch/blob/day3/main.ipynb
[day4]:https://github.com/liusida/DeepScratch/blob/day4/main.ipynb
