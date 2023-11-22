---
title: Reading 'Game Theory'
layout: post
summary: 
category: Reading
author: Sida Liu (with the help of ChatGPT)
---
I have begun learning about game theory and recently read Ken Binmore’s “Game Theory: A Very Short Introduction.” I would like to share some of my thoughts:

## Utility

In game theory, there is a quantity called utility. It represents the payoff for a player in a game. Von Neumann proposed a practical way to estimate this quantity by measuring the size of the risk a player is willing to take to obtain it. This concept reminds me of the ‘Reward’ in reinforcement learning (RL) and ‘Fitness’ in evolutionary algorithms (EA), where the objective is also to maximize these quantities. ChatGPT comments that such parallels reflect their shared roots in the study of decision-making and optimization processes. This connection is very convenient because we can integrate game theory with RL and/or EA by sharing this quantity. A prime example is the AlphaGo algorithm, which seamlessly integrates game theory with deep neural networks and RL, demonstrating exceptional performance.

## Evolutionary Game Theory

The most intriguing concept I’ve learned from the book is evolutionary game theory. When discussing Nash equilibrium, we usually view it as a rational solution for all players, where changing one's strategy, while others remain constant, doesn't yield any additional utility. This interpretation is standard, but there is an alternative: the evolutionary interpretation. It suggests that player behavior can be shaped by an evolutionary process, implying that the players aren't necessarily intelligent. Many animals in nature seem to solve games as if they were rational players. Most importantly, we can switch between these interpretations. For instance, when considering the solution of an evolutionary algorithm to a complex problem, we can view it as a rational Nash equilibrium of a game if we reframe the problem accordingly. Conversely, if we model a complex problem as a simple game, we can analytically determine all the Nash equilibria. Thus, if we use an evolutionary algorithm to solve that problem, we can somewhat predict the outcome.

Chapter 8, ‘Evolutionary Biology,’ provides several examples of game theory explaining natural phenomena. One fascinating case is explaining why eusociality often develops among Hymenoptera (bees and ants). Assuming the goal of genes is to propagate to the next generation, we can understand why bee workers are inclined to raise their sisters—they might share the same genes. The more challenging question is why this occurs more frequently among Hymenoptera. Bees are haplodiploid, meaning females have two sets of chromosomes, while males have only one. Therefore, we can calculate the expected proportion of genes a bee worker shares with its sister, which is 3/4. This proportion is lower in diploid species, at 1/2. So, to propagate their genetic material, it is more advantageous for Hymenoptera to settle on a Nash equilibrium where individuals raise their sisters rather than their offspring. (I’ve followed the calculation on page 134 and found the reasoning to be incorrect. If you discover the same, please let me know.)

## Mixed Strategy Nash Equilibrium

The concept of a mixed Nash equilibrium is vital. For instance, choosing Rock, Paper, or Scissors in a single game represents a pure strategy, but randomizing choices across repeated games constitutes a mixed strategy. If my mixed strategy is to choose each option one-third of the time, it is guaranteed that a Nash equilibrium exists for the game. This guarantee means every game is theoretically solvable, which is reassuring.

Game theory also suggests that external enforcement isn't always necessary. Given enough time, a system can evolve to its mixed Nash equilibrium. This idea is particularly interesting because it challenges the assumption that a government is essential for societal organization. Clearly, long before the advent of government, groups of ancient humans were already cooperating.

## Emotions as Emergent Phenomena

The book also posits that emotions, often perceived as irrational, are outcomes of evolution. According to game theory's dual interpretation, the result of the evolutionary process can be considered rational. It's not that the individuals with emotions are reasoning; rather, nature is finding rational solutions for them. Therefore, we shouldn't dismiss emotions like anger; instead, we should recognize their role in achieving a society's Nash equilibrium.

## Explanation of Self-Harming Behavior

The book discusses signaling in games with imperfect information. For signals to be credible, they must incur a cost; otherwise, they are disregarded. This explains why seemingly irrational behaviors, such as self-harm to demonstrate strength in a confrontation, are observable. Anger can thus be interpreted as a costly signal to opponents.

## Explanation of Unequally Shared Housework

Another intriguing example is the use of cooperative game theory to explain why, generally, wives tend to do more housework than husbands. The model assumes that the wife deems two hours of housework per day necessary, while the husband believes one hour is sufficient. Both assign 100 utils/week when their minimum requirement for housework is met. However, the wife values an hour of housework at a cost of 5 utils, while the husband has a greater aversion, valuing it at 10 utils/hour. The Nash bargaining solution in cooperative game theory predicts that the husband will end up with more utils than the wife, leading to the wife doing significantly more housework than the husband. I followed the simple calculations in the book, and it was quite interesting to see this dynamic modeled as a bargaining process. Admittedly, I am not fond of housework either.

## Critique

Lastly, I have some critiques. While Binmore is undoubtedly an expert in game theory, he does not excel at introducing the subject to those unfamiliar with it. This concise volume might be better suited as a conversation among peers, as it contains interesting viewpoints but lacks clear explanations of key game theory concepts (although it does mention many). Having watched Justin Grana's introductory videos on game theory, I had a foundational understanding for the first few chapters. Nonetheless, I struggled with more advanced topics, such as cooperative game theory, and will need to consult additional introductory textbooks to fully comprehend these concepts."

Justin Grana's introductory videos: 
[Game Theory I • Static Games](https://gts.complexityexplorer.org/courses/69-game-theory-i-static-games)
[Game Theory II • Dynamic Games](https://gts.complexityexplorer.org/courses/78-game-theory-ii-dynamic-games)

Any feedback? We can discuss it under [this Tweet. <i class="fab fa-twitter"></i>](https://twitter.com/liusida2007/status/1711759869060190359)