---
layout: post
title: Learning Rate is Too Large
---

What if I see a training accuracy scalar graphic like this:

![Accuracy](/images/2017-09-09-learning-rate-too-large/accuracy-1.png)

The accuracy curve of training mini-batch is going down a little bit over time after reached a relative high point. That might tell me the learning rate is too large.

When the learning rate is too large, the optimizer function can not converge the loss by adding derivative to variables--every step is too large, and the loss will become biger and biger.
