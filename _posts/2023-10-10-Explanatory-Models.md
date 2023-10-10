---
title: 
layout: post
summary: 
category: Reading
author: Sida Liu (with the help of ChatGPT)
---
Currently on the hunt for a prospective PhD advisor in neuroscience, which is why I stumbled upon this still-in-progress textbook: 'Explanatory Computational Models in Cognitive Neuroscience' by Andrea Stocco. Being an enthusiast for textbooks, especially those related to my research interests, I found this to be a delightful find.

What makes this book stand out to me is its initial focus on modeling cognition at a system-level, discussing concepts like the reward system, decision-making, and memory. It’s different from another book I read, 'An Introduction to Systems Biology,' which delves more into the microscopic level, exploring how epigenetics work. It was refreshing to delve into a book about high-level modeling.

One particularly enlightening section discusses the Reinforcement Learning (RL) algorithm and how the Reward Prediction Error (RPE) related to dopamine neuron activity. In simpler terms: when a reward arrives without expectation, dopamine neurons spring into action, creating an expectation for that reward. Then, they activate upon expectation, but if the reward doesn’t show up, there’s a negative response. The insightful parallel here suggests that adjusting expectations and dopamine activity are so intertwined that they might be two aspects of the same phenomenon. Furthermore, adjusting expectations with dopamine is much like updating $V(s_t)$ (the value of a state at a specific time) with RPE in basic RL. It's a psychologically plausible connection!

The book doesn’t stop at dopamine, it also compares SARSA and Q-Learning. Essentially, SARSA updates its Q-values using the next action that is actually taken, while Q-Learning employs the maximum Q-value of the next state. The realism of applying the max() function to all possible states can be contested, indicating SARSA might align more with actual animal behaviors than Q-Learning.

Further parallels are drawn, such as the Actor-Critic algorithm resembling the dorsal (Actor) vs. ventral (Critic) striatum, and the estimated transition function `\(P(s_t, a_t, s_{t+1})\)` in Model-Based RL drawing parallels with Declarative Memory. Fascinating stuff!

When it comes to decision-making, the book introduces Accumulator Models, popular in psychology, which model decision responses based on accumulated evidence. It feels like there's room for more exploration here though, as these models, being quite straightforward, don’t address complex aspects like free will or how agents seek evidence.

My favorite section? The one on long-term memory! I’ve always stressed that while Ebbinghaus's forgetting curve, which relates to memorizing non-sensical three-letter words, is valid, enriching concepts and linking them together is vital for retention. This book gave me the mathematical backing I needed! *Equation (25)* determines Activation (essentially, the 'log odds' of retrieving a memory) using two terms: the first is equavalent to the Ebbinghaus's curve, and second is the sum of influences from all connected concepts. A compelling argument for learners to weave those conceptual webs to fortify their memories.

Moving to the second part of the book, the focus shifts to artificial neural networks, likening them to the neuron level instead of the system level. A notable parallel drawn is between Convolutional Neural Networks and the Visual System, a widely recognized resemblance. The book also touches on the impracticality of backpropagation, another hot topic in the deep-learning community. I was hopeful for a section on transformers and their biological analogues, given their novelty. Delving into why transformers are so potent, it’s evident that their attention tables serve as Adjacency Matrices in Network Science, essentially implying that every substantial transformer model encompasses thousands of smaller complex networks that store knowledge. I'm curious if a similar mechanism exists in the brain.

Hebbian Learning also gets a mention, and interestingly, Contrastive Hebbian Learning is established to be equivalent to backpropagation. Since I’m not deeply acquainted with Hebbian Learning, I might delve deeper given its theoretical prowess, on par with traditional deep learning.

Sections on the Hopfield Network and Recurrent Network aren’t yet complete, so I’m eagerly awaiting more content!

You can download the book freely here: https://sites.uw.edu/stocco/textbook/. I hope you find it as engaging as I did.

Any feedback? We can discuss it under [this Tweet. <i class="fab fa-twitter"></i>](https://twitter.com/liusida2007/status/1698272066603196571)