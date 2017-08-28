---
layout: post
title: Implement a Deep Neural Network using Python and Numpy
---

I have just finished [Andrew Ng's new Coursera courses of Deep Learning (1-3)][1]. They are one part of his new project [DeepLearning.ai][2].

In those courses, there is a series of interview of [Heroes of Deep Learning][3], which is very helpful for a newbie to enter this field. I heard several times those masters require a newbie to build a whole deep network from scratch, maybe just use Python and Numpy, to understand things better. So, after the courses, I decided to build one on my own.

## Day 1

I created a new repository named [Deep Scratch][4] in github, and a [main.ipynb][5] file, to do this job.

First of all, I made a todo list, those are functions or algorithms mentioned in the courses. I planed to implement most of them.

Second, I decided to begin with MNIST dataset. It is the Hello, World dataset. But I will switch to other datasets to test my model. In my mind, maybe too ambitious, I want to build a transferable model, I think that's the correct direction to general thinking.

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

I had to say, the math is complex for me. When I implemented the first time, I almost have 10 bugs in the calculation!

There were a few un-concrete concepts, such as what loss function should I use for multi-class classification? What is the derivative of softmax? When should I divide the result by m (the number of examples)?

After maybe 10 hours of debugging, I even implemented a bunch of tensorflow alternative functions, finally, the model worked out!

[Day 2's notebook][day2] <- Here were the code and formula. (I found that Jupyter Notebook is great to comment codes!)

## Day 3

The training set accuracy was already 1.0, so I looked at the dev set accuracy: 0.6? Oh, that's bad. So I had a variance problem.

I tried to implement regularization, but seems had little help to this variance problem.

Then suddenly I figured out why: because I use random batch to train, the distribution of random batch can not cover all training examples, so I wasted a lot of training examples.

So I changed to mini-batch, which define a mini-batch size, every time use a segment of training data, so it can sure every single example was used for training.

And I also realize a very interesting aspect of mini-batch, it has a very good effect to variance problem, especially when the network is relatively shallow. I think it acts just like dropout! The network can not depend on any single data!

Thanks to mini-batch, my dev accuracy jumped to 0.98, and test set accuracy was also 0.98. Not bad for today's work!

[Day 3's notebook][day3] <- Here is mini-batch, regularization, train/dev/test accuracy.

## Day 4

Since the code was work and was ugly... I decided to refactor the code.

[Day 4's notebook][day4] <- I had refactored half of the code.

## Day 5

Refactor done. I was happy that the code looks clean now.

During refactoring, I attended to calculate the derivative of **Z=WA+b**, the **dL/dA**, but since **Z,W,A** are all matrices, I failed to understand the derivative of Matrix-by-Matrix. According to the Wikipedia, it seems results a four-rank tensor. So, it was lucky that in the neural network, the final $L$ is a scalar, so derivative of Scalar-by-Martix is much easy to understand.

I also noticed that in Tensorflow, the final loss function is a Vector! So, they must understand what is the derivative of Matrix-by-Matrix!

[Day 5's notebook][day5] <- Now the code is runable and clean.


## Day 6

As there was still a variance problem, and L2 regularization seems not help much, I decided to implement Dropout.

I chose the "inverted dropout", which introduced by Andrew Ng in the course.

I just watched a video comparing algebra and geometrics, it says that calculus and algebra can give you great power of solving a problem by just computing, but geometrics sometimes has its own beauty--it sometimes can solve a problem in a very simple way. Today, I felt like that the dropout technic is an analogy to geometrics, simple, effective, and beautiful.

Now after 100 iterations learning from training set, the dev accuracy raised to 98.34%.

There's another lovely feature I added into the code: during training, I can just press the stop button of notebook, and change some of the hyperparameters (only except the arcitecture--the hidden layers), and Ctrl+Enter run the cell, the parameters W and b are kept, not re-initialized, so the training can go on without restart from beginning.

But till now, I spent more and more time running the program by CPU--actually I am lucky that I have MKL for numpy, so I can use all of my CPU--I felt a little wasting of time. Maybe I will implement those in Tensorflow, and use my GPU to save time. And Tensorflow has auto-gradient computation ...

[Day 6's notebook][day6] <- Dropout version
  

[1]:https://www.coursera.org/specializations/deep-learning
[2]:https://www.deeplearning.ai/
[3]:https://youtu.be/-eyhCTvrEtE?list=PLfsVAYSMwsksjfpy8P2t_I52mugGeA5gR
[4]:https://github.com/liusida/DeepScratch
[5]:https://github.com/liusida/DeepScratch/blob/master/main.ipynb
[day2]:https://github.com/liusida/DeepScratch/blob/day2/main.ipynb
[day3]:https://github.com/liusida/DeepScratch/blob/day3/main.ipynb
[day4]:https://github.com/liusida/DeepScratch/blob/day4/main.ipynb
[day5]:https://github.com/liusida/DeepScratch/blob/day5/main.ipynb
[day6]:https://github.com/liusida/DeepScratch/blob/day6/main.ipynb