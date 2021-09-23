---
title: A simple Android virtual environment
layout: post
summary: Inspired by DeepMind's newly release AndroidEnv.
category: Simulation
---

Creating an intelligent agent that can browse the internet like a human user does is interesting.
Many websites don't want to provide services to artificial agents, but my opinion is that, making AI that can acqure information like human will facilitate the communication between AI and human, providing a common ground for both parties.

Recently, DeepMind has open-sourced a [virtual Android environment for RL agents](https://github.com/deepmind/android_env/).
Comparing to a virtual PC, a Android emulator tends to be faster.
Moreover, some applications only provide interface on mobile devices.
So I found this project interesting.

However, I am not familiar with Android, so the whole project is a little too complicated to me.
I decide to make a toy/simplified version of this project, just to learn more about the details.

First of all, this project doesn't include the emulator of Android, rather, we need to [install the emulator from Android Studio](https://github.com/deepmind/android_env/blob/main/docs/emulator_guide.md).
Once we created an AVD (Android Virtual Device), we should be able to start the device by this command:

```bash
~/Android/Sdk/emulator/emulator -avd my_device
```
suppose the emulator was installed into `~/Android/Sdk/emulator` folder, and `my_device` is the name of the AVD I created.

However, the guide suggests that we should start the AVD from Android Studio. I am not sure why we should not start the emulator without Android Studio.
If you happen to know why, please let me know.

You can probably see a GUI like this:

![avd](/images/2021-07-03-simple-android-env/avd.png)

And you can use your mouse and keyboard to play with this virtual device.

Then, we will want to send mouse and keyboard events to the virtual device programmably.

[Here](https://developer.android.com/studio/run/emulator-console) is a documentation for how to send commands to the emulator. 
However, this is a very simple explanation of the commands, and it is not clear how to send mouse events (with the command `event`).

After reading this [source file](https://github.com/deepmind/android_env/blob/main/android_env/components/emulator_console.py#L244) from the AndroidEnv, I discovered that we can use the command `event mouse` to send mouse events to the virtual device.
This is not mentioned in the documentation, and I don't know why.

Once we are able to send events (actions) to the virtual device, we also would like to get the feedback from the device.
According to the documentation, the way we can achieve that is using the `screenrecord screenshot` command.
However, this command only takes a path to a folder as its parameter (not a path to a file) and doesn't return the filename it has created, so we need to clean the folder beforehand and probably use `glob` to get whatever created in that folder after the command returns.
Also, the command returns before the file has been created, so one need to make sure the file has been created before using `glob`.

The file created by `screenrecord` is a PNG file, and we can use, for example, Python package `imageio` to read that file as a Numpy array, and feed that to the neural network.

Now we have actions and observations, and this is basically the simplest version of an Android envrionment.
Instead of using [DeepMind/AndroidEnv](https://github.com/deepmind/android_env/), we can manipulate the virtual device directly.

I'll be happy to see some intelligent agents that can operate the virtual mobile phone and discover the Internet on their own in a more human-like way.
