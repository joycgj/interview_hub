We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really fast, then we set up the training run following the GPT-2 and GPT-3 paper and their hyperparameters, then we hit run, and come back the next morning to see our results, and enjoy some amusing model generations. Keep in mind that in some places this video builds on the knowledge from earlier videos in the Zero to Hero Playlist (see my channel). You could also see this video as building my nanoGPT repo, which by the end is about 90% similar.

Links:
- build-nanogpt GitHub repo, with all the changes in this video as individual commits: https://github.com/karpathy/- build-nan...
- nanoGPT repo: https://github.com/karpathy/nanoGPT
- llm.c repo: https://github.com/karpathy/llm.c
- my website: https://karpathy.ai
- my twitter:   / karpathy  
- our Discord channel:   / discord  

Supplementary links:
- Attention is All You Need paper: https://arxiv.org/abs/1706.03762
- OpenAI GPT-3 paper: https://arxiv.org/abs/2005.14165 - OpenAI GPT-2 paper: https://d4mucfpksywv.cloudfront.net/b... - The GPU I'm training the model on is from Lambda GPU Cloud, I think the best and easiest way to spin up an on-demand - GPU instance in the cloud that you can ssh to: https://lambdalabs.com 

```
Chapters:
00:00:00 intro: Let’s reproduce GPT-2 (124M)
00:03:39 exploring the GPT-2 (124M) OpenAI checkpoint
00:13:47 SECTION 1: implementing the GPT-2 nn.Module
00:28:08 loading the huggingface/GPT-2 parameters
00:31:00 implementing the forward pass to get logits
00:33:31 sampling init, prefix tokens, tokenization
00:37:02 sampling loop
00:41:47 sample, auto-detect the device
00:45:50 let’s train: data batches (B,T) → logits (B,T,C)
00:52:53 cross entropy loss
00:56:42 optimization loop: overfit a single batch
01:02:00 data loader lite
01:06:14 parameter sharing wte and lm_head
01:13:47 model initialization: std 0.02, residual init
01:22:18 SECTION 2: Let’s make it fast. GPUs, mixed precision, 1000ms
01:28:14 Tensor Cores, timing the code, TF32 precision, 333ms
01:39:38 float16, gradient scalers, bfloat16, 300ms
01:48:15 torch.compile, Python overhead, kernel fusion, 130ms
02:00:18 flash attention, 96ms
02:06:54 nice/ugly numbers. vocab size 50257 → 50304, 93ms
02:14:55 SECTION 3: hyperpamaters, AdamW, gradient clipping
02:21:06 learning rate scheduler: warmup + cosine decay
02:26:21 batch size schedule, weight decay, FusedAdamW, 90ms
02:34:09 gradient accumulation
02:46:52 distributed data parallel (DDP)
03:10:21 datasets used in GPT-2, GPT-3, FineWeb (EDU)
03:23:10 validation data split, validation loss, sampling revive
03:28:23 evaluation: HellaSwag, starting the run
03:43:05 SECTION 4: results in the morning! GPT-2, GPT-3 repro
03:56:21 shoutout to llm.c, equivalent but faster code in raw C/CUDA
03:59:39 summary, phew, build-nanogpt github repo
```

Corrections:
I will post all errata and followups to the build-nanogpt GitHub repo (link above)

SuperThanks:
I experimentally enabled them on my channel yesterday. Totally optional and only use if rich. All revenue goes to to supporting my work in AI + Education.

以下是你提供内容的中文翻译：

---

**我们从零开始复现 GPT-2（124M）模型。**
这个视频展示了完整的过程：首先我们构建 GPT-2 网络，然后优化训练流程让它变得非常快，接着根据 GPT-2 和 GPT-3 的论文及其超参数设置训练流程，点击运行按钮，第二天早上回来看看结果，并欣赏一些有趣的模型生成内容。请注意，视频中的一些内容基于“Zero to Hero”系列前面的视频（请看我的频道）。你也可以把这支视频看作是我 [nanoGPT](https://github.com/karpathy/nanoGPT) 仓库的构建过程，最终大约有 90% 的内容是一样的。

🔗 **链接**：

* build-nanogpt GitHub 仓库（本视频中的所有更改都以单独的提交保存）：
  [https://github.com/karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
* nanoGPT 仓库：
  [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
* llm.c 仓库（用 C/CUDA 写的等效但更快的代码）：
  [https://github.com/karpathy/llm.c](https://github.com/karpathy/llm.c)
* 我的个人网站：
  [https://karpathy.ai](https://karpathy.ai)
* 我的 Twitter：[@karpathy](https://twitter.com/karpathy)
* 我们的 Discord 频道：[加入](https://discord.gg/karpathy)

📚 **补充链接**：

* 《Attention is All You Need》论文：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
* OpenAI GPT-3 论文：[https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
* OpenAI GPT-2 论文：[点击查看](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* GPU 训练是在 Lambda GPU Cloud 上进行的，他们提供最便捷的按需 GPU 云实例，支持 SSH 登录：[https://lambdalabs.com](https://lambdalabs.com)

📼 **视频章节目录**：

* 00:00:00 开场：我们要复现 GPT-2（124M）
* 00:03:39 探索 OpenAI 的 GPT-2（124M）模型
* 00:13:47 第1部分：实现 GPT-2 的 `nn.Module`
* 00:28:08 加载 huggingface/GPT-2 参数
* 00:31:00 实现前向传播，得到 logits
* 00:33:31 开始采样、前缀 token、分词处理
* 00:37:02 采样循环
* 00:41:47 采样，自动检测设备
* 00:45:50 开始训练：数据批 (B,T) → logits (B,T,C)
* 00:52:53 交叉熵损失
* 00:56:42 优化循环：在一个 batch 上过拟合
* 01:02:00 精简数据加载器
* 01:06:14 参数共享：`wte` 与 `lm_head`
* 01:13:47 模型初始化：标准差 0.02、残差初始化
* 01:22:18 第2部分：提速！GPU、混合精度、1000ms
* 01:28:14 Tensor Cores、计时、TF32 精度，降到 333ms
* 01:39:38 float16、梯度缩放器、bfloat16，降到 300ms
* 01:48:15 torch.compile、Python开销、融合内核，降到 130ms
* 02:00:18 flash attention，96ms
* 02:06:54 微调 vocab size：50257 → 50304，93ms
* 02:14:55 第3部分：超参数，AdamW、梯度裁剪
* 02:21:06 学习率调度器：warmup + 余弦衰减
* 02:26:21 批量大小调度、权重衰减、FusedAdamW，90ms
* 02:34:09 梯度累积
* 02:46:52 分布式训练：DDP
* 03:10:21 GPT-2、GPT-3 所用的数据集：FineWeb（教育类）
* 03:23:10 验证集划分、验证损失、采样演示
* 03:28:23 评估：HellaSwag、启动训练
* 03:43:05 第4部分：第二天早上看结果！GPT-2、GPT-3 复现
* 03:56:21 推荐 llm.c：C/CUDA 实现的等效版本，更快
* 03:59:39 总结：呼，代码全在 build-nanogpt GitHub 仓库

🛠 **更正说明**：
所有勘误和后续补充将会发布在 [build-nanogpt GitHub 仓库](https://github.com/karpathy/build-nanogpt)

💸 **SuperThanks**：
我昨天在频道上实验性地开启了“SuperThanks”功能，完全可选，仅在你真的富有的情况下使用。所有收入都用于支持我在 AI 和教育方面的工作。

---

如果你想我进一步归纳视频重点或拆解某一部分内容（比如“参数共享”、“Flash Attention”、“分布式训练”等），也可以告诉我。

# build nanoGPT

This repo holds the from-scratch reproduction of nanoGPT. The git commits were specifically kept step by step and clean so that one can easily walk through the git commit history to see it built slowly. Additionally, there is an accompanying video lecture on YouTube where you can see me introduce each commit and explain the pieces along the way.

We basically start from an empty file and work our way to a reproduction of the GPT-2 (124M) model. If you have more patience or money, the code can also reproduce the GPT-3 models. While the GPT-2 (124M) model probably trained for quite some time back in the day (2019, ~5 years ago), today, reproducing it is a matter of ~1hr and ~$10. You'll need a cloud GPU box if you don't have enough, for that I recommend Lambda.

Note that GPT-2 and GPT-3 and both simple language models, trained on internet documents, and all they do is "dream" internet documents. So this repo/video this does not cover Chat finetuning, and you can't talk to it like you can talk to ChatGPT. The finetuning process (while quite simple conceptually - SFT is just about swapping out the dataset and continuing the training) comes after this part and will be covered at a later time. For now this is the kind of stuff that the 124M model says if you prompt it with "Hello, I'm a language model," after 10B tokens of training:

```
Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's
```

And after 40B tokens of training:

```
Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due
```

Lol. Anyway, once the video comes out, this will also be a place for FAQ, and a place for fixes and errata, of which I am sure there will be a number :)

For discussions and questions, please use Discussions tab, and for faster communication, have a look at my Zero To Hero Discord, channel #nanoGPT:

# Video

Let's reproduce GPT-2 (124M) YouTube lecture

# Errata

Minor cleanup, we forgot to delete register_buffer of the bias once we switched to flash attention, fixed with a recent PR.

Earlier version of PyTorch may have difficulty converting from uint16 to long. Inside load_tokens, we added npt = npt.astype(np.int32) to use numpy to convert uint16 to int32 before converting to torch tensor and then converting to long.

The torch.autocast function takes an arg device_type, to which I tried to stubbornly just pass device hoping it works ok, but PyTorch actually really wants just the type and creates errors in some version of PyTorch. So we want e.g. the device cuda:3 to get stripped to cuda. Currently, device mps (Apple Silicon) would become device_type CPU, I'm not 100% sure this is the intended PyTorch way.

Confusingly, model.require_backward_grad_sync is actually used by both the forward and backward pass. Moved up the line so that it also gets applied to the forward pass.

# Prod

For more production-grade runs that are very similar to nanoGPT, I recommend looking at the following repos:

- litGPT
- TinyLlama

# FAQ

# License

MIT

以下是你提供内容的中文翻译：

---

## build-nanoGPT

这个仓库是对 **nanoGPT** 从零实现的复现。Git 提交历史被精心保留为一步步、干净整洁的记录，方便大家逐步查看整个构建过程。此外，我还在 YouTube 上录制了一段视频讲解，你可以看到我依次介绍每个提交，并解释其中的每个部分。

我们基本上是从一个空文件开始，最终复现了 GPT-2（124M）模型。如果你有更多耐心或预算，也可以用这些代码复现 GPT-3 模型。虽然当年（2019 年，也就是大约 5 年前）训练 GPT-2（124M）可能需要很长时间，但如今复现它只需要大约 1 小时和约 10 美元的成本。如果你本地没有足够的 GPU，建议使用云端 GPU，比如 [Lambda](https://lambdalabs.com)。

注意：GPT-2 和 GPT-3 都是简单的语言模型，它们的训练数据来自互联网文档，而它们的作用只是“梦想”互联网文档（即模拟互联网内容的生成）。因此，这个仓库/视频并不涉及 Chat 微调，所以你无法像使用 ChatGPT 一样“与它对话”。微调流程（虽然概念上很简单——比如监督式微调 SFT 就是更换数据集并继续训练）属于后续内容，将在以后讲解。

目前，以下是我们用“Hello, I'm a language model”作为提示词，在训练了 **100 亿个 token** 后，模型的输出示例：

> Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules
> Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff.
> Hello, I'm a language model, and the question is, what should I do if I want to be a teacher?
> Hello, I'm a language model, and I'm an English person. In languages, "speak" is really speaking. Because for most people, there's

而在训练了 **400 亿个 token** 后的输出是：

> Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write
> Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet.
> Hello, I'm a language model, but I'm talking about data. You've got to create an array of data: you've got to create that.
> Hello, I'm a language model, and all of this is about modeling and learning Python. I'm very good in syntax, however I struggle with Python due

哈哈，总之，视频发布后，这里也会成为 FAQ 的整理地，以及发布修复和勘误（应该会有不少 :)）。

如有问题或讨论，请使用 GitHub 的 Discussions 页面。若想更快交流，也可以加入我在 Zero To Hero 系列中的 Discord，频道是 **#nanoGPT**。

---

### 🎬 视频

* [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=kCc8FmEb1nY) YouTube 讲解视频

---

### ⚠️ 勘误（Errata）

* 有个小问题是我们在启用 flash attention 后，忘记删除 `bias` 的 `register_buffer`，已通过近期的 PR 修复。
* 较旧版本的 PyTorch 在将 `uint16` 转为 `long` 时可能有问题，我们在 `load_tokens` 函数中增加了一行：
  `npt = npt.astype(np.int32)`
  先用 NumPy 转为 int32，再转为 torch tensor，最后转为 long。
* `torch.autocast` 函数需要一个 `device_type` 参数，我最初直接传了 `device`，结果在某些 PyTorch 版本中报错。正确方式是将 `cuda:3` 这样的设备简化为 `cuda`。
* Apple Silicon 上的 `mps` 设备会被识别为 CPU 的 `device_type`，我不太确定 PyTorch 是否就是这么设计的。
* `model.require_backward_grad_sync` 实际上在正向和反向传播中都会用到，因此我们将这行代码上移，以确保在正向传播中也生效。

---

### 🚀 生产级建议

如果你想进行更加“生产级”的训练，这些仓库与 nanoGPT 非常类似，推荐查看：

* [litGPT](https://github.com/Lightning-AI/litgpt)
* [TinyLlama](https://github.com/jzhang38/TinyLlama)

---

### 📄 License

MIT 许可证

---

如果你需要我帮你归纳重点步骤，或者翻译相关代码结构，我也可以帮忙。


