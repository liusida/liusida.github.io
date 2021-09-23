---
title: Setup Unity ML-Agents
layout: post
summary: Unity has published ML-Agents 2.x, and here is how to set things up and run the examples.
category: Simulation
---

Unity has published ML-Agents 2.x which can do reinforcement learning on Unity environment. Here is the [GitHub Repo](https://github.com/Unity-Technologies/ml-agents).

To set everything up, we first need to install Unity. I have installed 2020.3.18f1 LTS version to my Ubuntu 20.04 from Unity Hub.

Then, we need to clone the [ML-Agents 2.x GitHub Repo](https://github.com/Unity-Technologies/ml-agents), for example, to `~/code/ml-agents/`.

Create a new Unity project, drag the folder `~/code/ml-agents/Project/Assets/ML-Agents/Examples` to the Unity Assets, right beside the `Scenes` folder.

Then, we need to install some packages to Unity. Click on the menu `Window`->`Package Manager`, select `Packages: In Project`, click the `+` on its left, `Add package from disk...`, and choose the `package.json` in `~/code/ml-agents/com.unity.ml-agents/` folder, install it. Click the `+` and `Add package from disk...` again, and choose the `package.json` in `~/code/ml-agents/com.unity.ml-agents.extensions/` folder, install it as well.
![packagemanager](/images/2021-09-23/package-manager.png)

We also need another package from `Unity Registry`. Search for `Input System`, and install that.
![input-system](/images/2021-09-23/input-system.png)

We would notice that there are still errors in the console. It seems that there are some incompatible code in the example `PushBlockWithInput`. I simply delete that example in `Assets`.

Now, if we open the scene `Assets/Examples/3DBall/Scenes/3DBall.unity`, we can click play to see the pre-trained model controling the robots. (Although there are still many error messages. ;)
![3dball](/images/2021-09-23/3dball.png)
