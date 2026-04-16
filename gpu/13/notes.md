# Setting Up CUDA Development (A Modern Workflow with JetBrains)

Before writing any CUDA code, there is one step you cannot skip.

You need a working env.

Not just sth that compiles code, but something you can actually work in every day without fighting the tools.

A more natural setup is built around JetBrains tools, especially CLion.

The reason for this is not about preference. It is about how modern development works.

CUDA itself has evolved. GPU architectures change faster now. Toolkits change more often. Projects are no longer tied to a single platform. You might develop on Linux, test on a remote GPU, and deploy somewhere else.

In that kind of workflow, a tightly coupled IDE like Visual Studio becomes limiting.

JetBrains tools take a different approach.

They are built around CMake. That matters more than it sounds.

Because once your project is defined through CMake, it stops being tied to one environment. You can build it on different systems, use different compilers, and still keep the same structure.

This is much closer to how real GPU systems are built.

The CUDA Toolkit is still the foundation of everything.

Nothing runs without it.

It provides the compiler, the runtime, and the libraries that actually talk to the GPU. When you install it, you are not installing an editor. You are installing the layer that makes GPU execution possible.

As of 2026, this layer has become more sensitive to the underlying hardware.

New architectures like Hopper and Blackwell introduce new instructions, new precision formats, and new execution behavior. To access these features, you need a recent CUDA version. Older versions may still work, but they will not reflect what the hardware is capable of.

So choosing the CUDA version is no longer a minor detail. It defines what your code can do.

CLion sits on top of this.

It does not try to replace the CUDA Toolkit. It does not hide it either.

Instead, it gives you a clean place to work.

You write your code there. You organize your project there. And when you build, CLion simply calls CMake, which then calls the CUDA compiler.

Nothing magical happens behind the scenes.

And that is exactly why it works well.

You always know what is happening.

There is one detail that can confuse people at first.

Even if you do not plan to use Visual Studio, you may still need parts of it installed on Windows.

This is not because you will use it directly, but because the CUDA toolchain depends on the Microsoft compiler in the background.

In practice, this means Visual Studio becomes a dependency, not your workspace.

You install it once, and then you forget about it.

All your actual work happens in CLion.

Another thing that becomes important very quickly is the driver.

CUDA does not work in isolation. It depends on the GPU driver.

If your driver is too old, you may run into problems that are hard to explain at first. Code may compile but not run correctly. Or certain features may simply not be available.

In 2026, where architectures are evolving fast, keeping the driver up to date is not optional. It is part of the setup.

Once everything is in place, the workflow becomes simple.

You open CLion. You write your code. You build with CMake. The CUDA Toolkit compiles it. The GPU runs it.

There is no friction between these steps when the setup is done right.

At this point, the important shift becomes clear.

CUDA development is not about choosing an editor anymore.

It is about understanding the toolchain.

JetBrains tools fit naturally into this because they do not try to control everything. They let each part of the system do its job.

And that is why, in a modern setup, this approach feels cleaner, more stable, and closer to how things actually work in production.

That’s exactly what I’m going to do in this repo.
