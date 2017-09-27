---
layout: post
title: Why does the person with highest IQ not become the most successful one?
---
I heard about the "[Study of Mathematically Precocious Youth After 35 Years][1]" years ago, but after studying machine learning, especially the generalization problem, I guess I have glanced some possible reasons.

While a human is learning, the process is more or less like the machine learning. The talent, the IQ testing result, can somehow prove the human has a more complex brain, just like the neural networks have more complex architectures. Unfortunately, overfitting often occurs when a model with more complex architecture learning. When a model stuck at overfitting, the training error will go down steadily while the validation error will become larger. This problem is called generalization problem. In the context of machine learning, we have several tricks to partially solve it. Here is a list of methods and their analogies of human learning.

* More data. This is the best method both for machine learning and human learning. If we are smart youths, just keep learning new stuff, and we can avoid overfitting to the knowledge we learn, and generalize better.

* Dropout. This is my favorite method. Don't study all the time. Do something else, or just do nothing, maybe sleep. And then we will find our ability of generalization improved. Sounds nice!

* Adding noise to input. This is also a practical method. If we need study something several times to master a perticular idea, maybe after we feel confident enough, we can add some noise, e.g. maybe use an alternative material, or maybe focusing on different details.

* L2 regularization. This is pushing all the unrelated weights to be near zero. When we study, maybe after several rounds of learning, we ask ourselves to not doubt what we learned. If we are not 100% sure, then don't trust it.







[1]:https://euler.epfl.ch/files/content/sites/euler/files/users/144617/public/LubinskiPersson.pdf