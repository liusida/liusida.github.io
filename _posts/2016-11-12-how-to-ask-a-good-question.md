---
layout: post
title: 如何问一个好问题？
---
[本文翻译自StackOverflow.com，原文链接（英文）][1]

首先，我们很愿意帮助你。为了提高你获得回答的机会，这里有几个小提示：

## 先搜索，再搜索。

把你找到的东西记下来。就算你在搜索的时候找不到其他有用的答案，把你搜到的跟你问题有关的链接放进来并说明为什么没有回答你的问题，这样能够让来回答的人更明白你问题的意思。

## 写一个能够准确概括你问题的**标题**。

标题是潜在回答者首先看到的东西，如果你的标题没有引起他们的兴趣，他们是不会点进去读到你其他内容的。你可以这样：

* 假装你是要问一个很忙的同事，你需要把你整个问题用一句话总结：什么细节可以让人马上分辨出你想要问的？错误提示、关键API、不寻常的情况，这些细节可以让你的问题更具识别度，如果跟其他问题回答过的没什么区别，别人可能就不会再受累回答了。
* 拼写、语法、标点，这些都很重要！请记住，标题是你给人家的第一印象。如果你不习惯使用英语，你可以找个朋友帮你过一下标题。
* 如果你觉得总结问题关键点有点难，你可以写完正文以后再写标题。有时候先把问题写出来后，就很自然能总结这个问题关键了。

举一些例子：
* **坏** ：C# Math Confusion
* **好** ：Why does using float instead of int give me different results when all of my inputs are integers?
* **坏** ：[php] session doubt
* **好** ：How can I redirect users to different pages based on session data in PHP?
* **坏** ：android if else problems
* **好** ：Why does str == "value" evaluate to false when str is set to "value"?

## 在贴代码之前，先描述问题。

在正文里，先把你标题写的问题关键点展开详细描述。讲明白你是怎么遇到这个问题的，讲明白你自己在解决这个问题时遇到了什么困难。你问题里的第一段是潜在回答者第二样看到的东西，所以尽可能写的明确一些。

## 帮别人重现这个问题

不是所有的问题都要贴代码的。但如果你遇到的问题是关于你正在写的代码的，你可以贴一部分。但是别把你整篇整篇的代码贴进来！如果你贴的代码是你工作代码的话，公司可能找你麻烦，而且在别人要跑代码重现问题时候，你贴整篇代码还包含了很多无关的东西需要他们去去掉。你可以这样：

* 精简一下代码，让你贴的代码能跑起来，而且能出现你要解决的问题。具体怎么做可以参考这里：[How to create a Minimal, Complete, and Verifiable example][2].
* 如果你的代码可以在某些平台运行，比如说 [http://sqlfiddle.com/][sqlfiddle] 或者 [http://jsbin.com/][jsbin]，那么你就贴上链接，这样人家点一下就可以重现你的问题。同时，在问题里也贴上代码，因为有时候站外链接会不好用，找不到页面或者对方宕机什么的。

## 选择一些相关的标签

选择与你的问题相关的标签，比方说是什么语言、什么库、某个特定的API等等。你在标签输入框里打字的时候，系统会提示你一些标签，请确认你仔细阅读了那些标签的描述，确保标签跟你的问题有关系。如何选择标签可以看 [What are tags, and how should I use them?][3]

## 发帖之前，再检查一遍

好了，你准备好发帖提问了，稍等，来做一个深呼吸，然后把你的帖子从头读到尾。假装你第一次看到这个帖子，能看懂帖子问的是什么吗？尝试在一个全新的环境重现这个问题，就靠你帖子里提供的信息行吗？把你漏掉的细节加一下，再读一遍。好，现在最后再来看一下你的标题，是不是还是说的是你的这个问题！

## 发帖吧，并保持反馈

发帖之后，先不要急着关掉浏览器，开一段，看看有没有人留言。如果你漏了什么明显的东西，你可以修改问题贴，把它加加进来。如果有人回答了，就抓紧试一下，然后给个反馈。

## 找 Help asking 帮忙

经过如上这些努力，你的问题贴还是没人回答的话，别失望，学会怎么问问题是很值的，而且不是一晚上就能学会的。这里还有一些材料可以继续学习：
* [Writing the perfect question][4]
* [How do I ask and answer homework questions?][5]
* [How to debug small programs][6]
* [Meta discussions on asking questions][7]
* [How to ask questions the smart way][8] — long but good advice.


## 翻译者按
*假若我们这么认真的对待问题，不仅能让回答者感动并好好回答，有时候问题都感动得自己化解掉了，根本不需要发帖就解决了。*



[1]: http://stackoverflow.com/help/how-to-ask
[2]: http://stackoverflow.com/help/mcve
[3]: http://stackoverflow.com/help/tagging
[4]: http://codeblog.jonskeet.uk/2010/08/29/writing-the-perfect-question/
[5]: http://meta.stackexchange.com/questions/10811/how-do-i-ask-and-answer-homework-questions
[6]: http://ericlippert.com/2014/03/05/how-to-debug-small-programs/
[7]: http://meta.stackexchange.com/questions/tagged/asking-questions
[8]: http://www.catb.org/~esr/faqs/smart-questions.html
[sqlfiddle]: http://sqlfiddle.com/
[jsbin]: http://jsbin.com/
