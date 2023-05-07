---
title: Validate HLSL grammar in command line
layout: post
summary: HLSL is used in Unity, and one can debug HLSL code in SHADERed. However, SHADERed doesn't show the detailed compile error message as Unity does. Here we show how to use a tool called glslangValidator to validate the HLSL grammar in command line.
category: GPU
---

HLSL, a.k.a. High-level shader language, is used in Unity.

One can write the shader in Unity (with VS or VSCode) and compile it automatically in Unity. If there's any error in the shader code, the error message will show up in the Unity console.

However, since I am using a Linux, I can't debug the shader in Unity.
So I turn to a light-weight tool called SHADERed.
One can install the SHADERed plug-in in VSCode along with SHADERed, and voilà, we can debug the shader code!

So we can write HLSL in VSCode and SHADERed first, and then copy it to Unity shader and modify it to be compatible.

Here is a cool video shows how to do the modification, it even works with GLSL.
[Youtube: How to convert a shader from ShaderToy to Unity](https://www.youtube.com/watch?v=CzORVWFvZ28)

Now my problem was that SHADERed won't show the detailed compilation error if there's any error in the shader code. It just says, "Can not display preview - there are some errors you should fix." 

So I digged in a bit, and realized that both Unity and SHADERed use [glslang](https://github.com/KhronosGroup/glslang) to compile the shader code.

Luckily, glslang provide a standalone executable that can compile the shader code as a showcase of how to use the glslang API. It is called `glslangValidator`. I am on Ubuntu, so I can get this tool by simply `apt install glslang-tools`.

So for example we want to validate this piece of fragment shader named `2d_SimplePS.hlsl`:
```c
cbuffer vars : register(b0)
{
	float2 uResolution;
	float uTime;
};

float4 main(float4 fragCoord : SV_POSITION) : SV_TARGET
{
    float2 uv = fragCoord.xy/uResolution;
    uv -= .5;
    uv *= uResolution.x/uResolution.y;
    here is an intentional error;
    float4 color = float4(uv.x, uv.y, 1, 1);
    return color;
}
```
And we use the command:
```bash
# look it up using `man glslangValidator`
glslangValidator -S frag -D --client vulkan100 -e frag 2d_SimplePS.hlsl
```

and the compiler will show the error message: 
```bash
2d_SimplePS.hlsl
ERROR: 2d_SimplePS.hlsl:12: 'here' : unknown variable 
ERROR: 2d_SimplePS.hlsl:12: ';' : Expected 
2d_SimplePS.hlsl(12): error at column 11, HLSL parsing failed.
ERROR: 3 compilation errors.  No code generated.

SPIR-V is not generated for failed compile or link
```

We have the file name, line number, and the detailed error messages. Great!

