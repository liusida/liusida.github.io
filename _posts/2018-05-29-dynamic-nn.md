---
layout: post
title: Dynamic NN Allowing Additional Evidence?
---
We have traditional Neural Network (NN), with static structure like this:

signal -> input -> hidden layer -> prediction =?= truth

 ->  means to propagate forward
=?= means to minimize difference


What if we already trained one model like that and there is another evidence (signal) coming in front of us? Can we add the signal into the model dynamically without abandon what has been trained already?

I think we should use Bayesian Theory, but I havn't figure out how yet.
