---
title: A Plan to Make an AI Programmer
layout: post
summary: Make an RL environment, design the agent, and go!
category: AI
author: Sida Liu (with the help of ChatGPT)
---
What are the required skills for a being to be a programmer?

1. Coding: they must be capable of writing code, surely. This is a token-level ability.
2. Project management: project management skills are also important, which may involve organizing files into a structure that is easy to navigate later.
3. Documentation: the ability to produce documentation that clearly explains project ideas is also needed.
4. Testing: regularly testing can maintain code quality.
5. Collaboration: programmers should also be able to collaborate with other programmers, adjusting and progressing when others modify their code.
6. Philosophy: programmers should be creative, wise and kind-hearted, with goals that align with ethical standards.

Current GPT models (3.5 or 4) serve as a good starting point for all these requirements above, especially 3 and 6, since it doesn’t just read code but also tons of books, it must be wiser than single-minded coding machines. However, since its model weights are not publicly available, the first step is to create or find an alternative to GPT.

Next, we need a Reinforcement Learning environment. Similar to training a human programmer, we must design a curriculum for AI to help it develop its programming skills. The curriculum will start with the classic “hello, world!” task.

At present, it’s better for AI to learn coding within a terminal environment, given that graphics processing is more challenging for existing models. So, let’s just pick a popular Linux shell, for example, Ubuntu and bash.

The `actions` can be structured in JSON format. AI can produce actions such as bash command ‘git’ or ‘tree’. It can also read/write files. The `observations` can be the environment’s stdout/stderr.

Consider “hello, world!” as a concrete example. Here are potential actions and observations:
```
{
    Action: write-file,
    FilePath: /helloworld.py,
    Content: print(“hello, world!”)
}
{
    Observation: success
}
{
    Action: bash,
    Command: ls
}
{
    Observation: helloworld.py
}
{
    Action: bash,
    Command: python helloworld.py
}
{
    Observation: hello, world!
}
{
    Action: submit,
    EntryPoint: helloworld.py
}
{
    Observation: success,
    Reward: +1
}
```

As suggested by OpenAI’s recent concept of [Process Supervision](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision), dense rewards are likely more effective (but I guess it’ll be more rigid). While creating a curriculum with dense rewards may seem laborious, it may prove beneficial at the outset.

With our environment established, we can begin design our agent. This agent should make use of both [System 1 and System 2](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow#:~:text=Thinking%2C%20Fast%20and%20Slow%20is,more%20deliberative%2C%20and%20more%20logical). System 1 is a large neural network model, similar to GPT.

Designing System 2 is somewhat intricate. Initially, System 2 serves as the glue code between System 1 and the environment. Over time, however, System 1 should learn to bypass some steps of System 2, establishing a more direct connection to the environment. This resembles the way human programmers learn: we start by adhering to rigid rules, then internalize these rules for a more natural/creative/efficient approach. Still, System 1 should periodically consult System 2 to ensure its intuitions are accurate. For instance, System 2 could involve parsing observation JSON, formulating prompts, organizing output into action JSON. But these steps are skippable, the model might sometimes use the raw observation JSON as input without prior prompts.

Once we have our environment and agent, we can initiate our RL training process. Hopefully, the agent can successfully complete the curriculum and become a real programmer!

During RL training, System 1 continually improves, while System 2 remains constant. So perhaps, upon graduation, the agent’s first task should be to rewrite its System 2, then revisit the training process? 🙂

Any feedback? We can discuss it under [this Tweet. <i class="fab fa-twitter"></i>](https://twitter.com/liusida2007/status/1658114711605354499)