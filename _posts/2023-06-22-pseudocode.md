---
title: Talking to GPT in pseudocode
layout: post
summary: Between natural language and executable code, pseudocode is both precise and concise--clearly conveying the user's instructions while saving the precious context window.
category: AI
author: Sida Liu (with the help of ChatGPT)
---
In the fascinating world of Large Language Models (LLMs), new applications are continuously being developed. LLMs, like GPT-4, have shown incredible promise in the field of artificial intelligence. However, the question often arises: How can we improve the way we communicate with these models?

One of the main challenges is that natural language often falls short in describing instructions precisely and concisely, wasting space in the precious *context window*. This is particularly evident when giving instructions to LLMs, where exactness and precision are paramount. The use of pseudocode presents an intriguing solution to this problem.

## Pseudocode and SudoLang

In the realm of computer science, pseudocode is often used to articulate programming logic and algorithms. It offers a simplified, high-level representation, which is easy for programmers to understand.

Recently, I’ve stumbled upon [SudoLang](https://github.com/paralleldrive/sudolang-llm-support/blob/main/sudolang.sudo.md), a novel JavaScript-like pseudolanguage explicitly designed for interacting with LLMs. SudoLang aims to address the communication gap by providing a medium that both the programmer and the LLM understand. Interestingly, SudoLang is compatible with ChatGPT out of the box, creating an efficient and effective way to convey ideas to the AI.

It seems almost magical how ChatGPT can respond to SudoLang so naturally. Therefore, I held a conversation with ChatGPT about how it comprehends SudoLang code without knowing its existence beforehand. The conversation revealed that although GPT-4 had never seen this format before, it made *educated guesses* based on a series of assumptions. While the GPT-4 is fine-tuned to give definite answers, it omits mentioning these underlying assumptions. The process might seem like executing pseudocode, but it is essentially an instance of "Unplugged Coding". The conversation also revealed that this feature is very likely to be absent for LLMs without code in its training set.

## Broadening the Pseudocode Horizon

While SudoLang has its roots in JavaScript, the pseudocode concept can be extended to other popular programming languages as well. Python, known for its popularity, could potentially serve as a basis for pseudocode as well, allowing for straightforward interactions between Python programmers and LLMs like GPT-4.

Interestingly, this intriguing idea of using pseudocode as a communication tool with LLMs isn't just limited to practical applications. It's also found a place in serious computer science research. For instance, [VisProg](https://arxiv.org/abs/2211.11559), one of the best papers presented at CVPR 2023, leverages pseudocode to guide models towards desired outcomes without the need for additional training.

## The Future of Pseudocode and Programming

The future of programming could very well hinge on the use of pseudocode. For developers, provided these concepts can be efficiently communicated using pseudocode, the focus might shift from the intricacies of implementation to the clarity of ideas.

As we move into an era where AI's role is increasingly significant, the ability to communicate our thoughts to these systems accurately becomes paramount. Pseudocode, with its concise and easy-to-understand format, could complement natural language, driving a new wave of innovation in the way we interact with LLMs.

Any feedback? We can discuss it under [this Tweet. <i class="fab fa-twitter"></i>](https://twitter.com/liusida2007/status/1664470676642873344)