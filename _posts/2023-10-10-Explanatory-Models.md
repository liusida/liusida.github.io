---
title: Explanatory Models in Cognitive Neuroscience
layout: post
summary: A in-progress textbook navigates through system-level cognitive modeling, paralleling computational algorithms like RL with neural phenomena.
category: Reading
author: Sida Liu (with the help of ChatGPT)
---
Currently searching for a prospective PhD advisor in neuroscience, which is why I stumbled upon this still-in-progress textbook: 'Explanatory Computational Models in Cognitive Neuroscience' by Andrea Stocco. Being an enthusiast for textbooks, especially those related to my research interests, I found this to be a delightful discovery.

What makes this book stand out to me is its initial focus on modeling cognition at a system-level, discussing concepts like the reward system, decision-making, and memory. It’s different from another book I read earlier, 'An Introduction to Systems Biology,' which delves more into the microscopic level, exploring how epigenetics work. It was refreshing to delve into a book about high-level modeling.

The first section of Part I is particularly enlightening. It discusses the relationship between the Reinforcement Learning (RL) algorithm and the reward system, more specifically, how the Reward Prediction Error (RPE) related to dopamine neuron activity. In simpler terms: when a reward arrives without expectation, dopamine neurons are activated, besides making us feel better, it creates an expectation for that reward. Then, they gradually shift their trigger from the actual reward to the expectation it created, but if the reward doesn’t show up, the dopamine neurons give negative signals, making us disappointed. The insightful parallel here suggests that adjusting expectations and dopamine activity are so intertwined that they might be two facets of a singular phenomenon. Furthermore, adjusting expectations with dopamine is much like updating `$V(s_t)$` (the value of a state at a specific time) with RPE in state-value-based RL algorithm. It's a psychologically plausible connection!

The book doesn’t stop at dopamine, it also compares SARSA and Q-Learning. Essentially, SARSA updates its Q-values using the next action that is actually taken, while Q-Learning picks the maximum Q-value of the next state. In reality, applying the max() function to all possible actions and subsequent states is obviously not feasible (thinking about the infinite parallel universes!), indicating SARSA might align more with actual animal behaviors than Q-Learning.

Further parallels are drawn, such as the Actor-Critic algorithm resembling the dorsal (Actor) vs. ventral (Critic) striatum, and the estimated Transition Function `$P(s_t, a_t, s_{t+1})$` in Model-Based RL drawing parallels with Declarative Memory. Fascinating stuff!

(Need a refresh on RL algorithm? Here's a great tutorial from David Silver 8 years ago: https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)

When it comes to decision-making, the book introduces Accumulator Models, popular in psychology, which model decision responses based on accumulated evidence. It feels like there's room for more exploration here though, as these models, being quite straightforward, don’t address complex aspects like *free will* or *how agents seek evidence*.

My favorite section? The one on long-term memory! While educators mention Ebbinghaus's forgetting curve, which was derived from the experiment of memorizing non-sensical three-letter words, I’ve always stressed that enriching the concepts and linking them together is vital for retention. This book gave me the mathematical backing I needed! *Equation (25)*, from the section talking about the Role of Context, determines Activation (essentially, the 'log odds' of retrieving a memory) using two terms: the first is equivalent to the Ebbinghaus's curve, and second is the sum of influences from all connected concepts. A compelling argument for learners to weave those conceptual webs to fortify their memories.

Moving to Part II of the book, the focus shifts to artificial neural networks, likening them to the neuron level instead of the system level. A notable parallel drawn is between Convolutional Neural Networks and the Visual System, a widely recognized resemblance. The book also touches on the impracticality of backpropagation, another hot topic in the deep-learning community. I was hopeful for a section on *Transformers* and their biological analogues. From my point of view, the potency of transformers lies in their attention tables being similar to the Adjacency Matrices from Network Science, essentially implying that every large transformer model encompasses thousands of smaller complex networks that store its knowledge. I'm curious if a similar mechanism exists in the brain.

Hebbian Learning also gets a mention, and interestingly, Contrastive Hebbian Learning is established to be equivalent to Backpropagation. Since I’m not deeply acquainted with Hebbian Learning, I might delve deeper given its potential equivalency with traditional deep learning.

The sections on the Hopfield Network and Recurrent Network aren’t yet complete, so I’m eagerly awaiting more content!

You can download the book freely here: [https://sites.uw.edu/stocco/textbook/](https://sites.uw.edu/stocco/textbook/). I hope you find it as engaging as I did.

Any feedback? We can discuss it under [this Tweet. <i class="fab fa-twitter"></i>](https://twitter.com/liusida2007/status/1711759869060190359)