At some point, I realized that knowing the theory was not enough.

I kept seeing terms like architecture and generation, but I did not really know how to identify them when I looked at a real GPU.

So I asked myself a simple question: How do I actually figure this out in practice? 

The first thing I did was surprisingly simple.

I just searched for the GPU name.

For example, I typed: "A100 TechPowerUp" or "RTX 3090 TechPowerUp" and opened one of the results.

At first, the page looked a bit overwhelming.

There were many numbers, many specs, and a lot of details I did not fully understand yet.

So instead of trying to understand everything, I changed my approach.

I decided to focus only on a few key things.

What helped me the most was this: I started looking only at the architecture and the product category.

Everything else could wait.

For example, when I checked the RTX 3090, I saw that it was based on the Ampere architecture.

I also noticed that it belongs to the GeForce family.

That already told me something important.

It means this GPU is built for consumer use, like gaming or personal workstations.

Then I looked at something completely different: the A100.

At first, I expected it to be very different.

But when I checked the specs, I saw something interesting.

It was also based on Ampere.

So both GPUs share the same architecture.

But they clearly do not serve the same purpose.

This is where things finally clicked for me.

The architecture tells you how the GPU is built.

The category tells you where it is used.

In older materials, this category was often called "Tesla".

But in more recent Nvidia terminology (around 2026), you will mostly see names like: Data Center GPU or AI GPU.

So even though RTX 3090 and A100 are both Ampere-based, they live in completely different worlds.

One is optimized for gaming and everyday use.

The other is built for AI workloads, cloud infrastructure, and large-scale systems.

At that point, I made another mistake.

I started comparing GPUs only by looking at core counts.

Numbers like 7000 or 10000 looked very convincing.

But I quickly realized that this was misleading.

Those numbers usually refer to a specific type of core, often single-precision units.

They do not represent everything inside the GPU.

Modern GPUs include different types of compute units, especially in newer architectures like Hopper and Blackwell.

So just looking at core count does not tell the full story.

Then I noticed something else that made things easier.

Sometimes you can understand a GPU just by looking at its physical design.

When I looked at data center GPUs like A100 or H100, I realized they often do not have visible fans.

At first, this confused me.

But the reason is simple.

These GPUs are designed to run inside servers.

Cooling is handled by the system, not the GPU itself.

On the other hand, when I looked at RTX GPUs, they always had large fans and cooling systems.

That also makes sense.

They are built for desktops and workstations, so they need to manage their own heat.

This gave me a very simple mental shortcut.

If a GPU has large visible cooling, it is most likely designed for consumer use.

If it looks more like a compact module without fans, it is probably meant for data center environments.

Of course, this is not a strict rule, but it works surprisingly often.

After going through this process a few times, everything became much clearer.

I stopped trying to understand every number.

Instead, I started asking better questions.

What architecture is this GPU using?  

What category does it belong to?  

What kind of problem is it designed to solve?  

Once I had these answers, the rest of the specifications started to make more sense.

And more importantly, I stopped getting confused by random numbers.

Looking back, this was a small shift in thinking, but it made a big difference.

And if you are learning CUDA or working with GPUs, this mindset helps a lot.

Because at the end of the day, understanding the purpose of a GPU is just as important as understanding its specs.