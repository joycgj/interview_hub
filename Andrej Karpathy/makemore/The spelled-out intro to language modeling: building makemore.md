- [makemore](#makemore)
    - [Usage](#usage)
    - [License](#license)
- [makemore](#makemore-1)
    - [使用方法](#使用方法)
    - [许可证](#许可证)
- [makemore.py introduction](#makemorepy-introduction)
- [The spelled-out intro to language modeling: building makemore 视频介绍](#the-spelled-out-intro-to-language-modeling-building-makemore-视频介绍)
- [intro](#intro)
- [介绍](#介绍)
- [reading and exploring the dataset](#reading-and-exploring-the-dataset)
- [读取和探索数据集](#读取和探索数据集)
- [exploring the bigrams in the dataset](#exploring-the-bigrams-in-the-dataset)
- [探索数据集中的二元组](#探索数据集中的二元组)
- [counting bigrams in a python dictionary](#counting-bigrams-in-a-python-dictionary)
- [在 Python 字典中统计二元组（bigrams）](#在-python-字典中统计二元组bigrams)
    - [一般形式：](#一般形式)
    - [举例说明：](#举例说明)
      - [✅ 示例 1：普通函数 vs lambda](#-示例-1普通函数-vs-lambda)
      - [✅ 示例 2：配合 sorted 使用](#-示例-2配合-sorted-使用)
      - [✅ 示例 3：配合 `map()` 使用](#-示例-3配合-map-使用)
    - [总结：](#总结)
- [counting bigrams in a 2D torch tensor ("training the model")](#counting-bigrams-in-a-2d-torch-tensor-training-the-model)
- [用 2D Torch 张量统计 bigram（“训练模型”）](#用-2d-torch-张量统计-bigram训练模型)
    - [✅ 步骤 1：导入 PyTorch](#-步骤-1导入-pytorch)
    - [✅ 步骤 2：创建张量示例](#-步骤-2创建张量示例)
    - [✅ 步骤 3：构建 28x28 的大张量](#-步骤-3构建-28x28-的大张量)
    - [✅ 步骤 4：字符转整数的映射（lookup 表）](#-步骤-4字符转整数的映射lookup-表)
    - [✅ 步骤 5：填充张量（即 bigram 统计）](#-步骤-5填充张量即-bigram-统计)
    - [✅ 总结：](#-总结)
    - [🔍 分步解释：](#-分步解释)
      - [第一步：`''.join(words)`](#第一步joinwords)
      - [第二步：`set(...)`](#第二步set)
    - [✅ 用法场景](#-用法场景)
    - [🧠 举个例子再总结：](#-举个例子再总结)
    - [✅ 总结一句话：](#-总结一句话)
- [visualizing the bigram tensor](#visualizing-the-bigram-tensor)
- [可视化 bigram 张量](#可视化-bigram-张量)
    - [✅ 第一步：使用 matplotlib 简单可视化](#-第一步使用-matplotlib-简单可视化)
    - [✅ 第二步：构造更美观的可视化](#-第二步构造更美观的可视化)
    - [✅ 第三步：完整可视化逻辑](#-第三步完整可视化逻辑)
    - [✅ 补充解释：](#-补充解释)
    - [✅ 总结：](#-总结-1)
    - [🔸 `import matplotlib.pyplot as plt`](#-import-matplotlibpyplot-as-plt)
    - [🔸 `%matplotlib inline`](#-matplotlib-inline)
    - [🔸 `plt.imshow(N)`](#-pltimshown)
    - [✅ 举个例子：](#-举个例子)
    - [✅ 总结：](#-总结-2)
    - [🔹 `import matplotlib.pyplot as plt`](#-import-matplotlibpyplot-as-plt-1)
    - [🔹 `%matplotlib inline`](#-matplotlib-inline-1)
    - [🔹 `plt.figure(figsize=(16, 16))`](#-pltfigurefigsize16-16)
    - [🔹 `plt.imshow(N, cmap='Blues')`](#-pltimshown-cmapblues)
    - [🔹 双层 `for` 循环：逐格标注字符和计数](#-双层-for-循环逐格标注字符和计数)
      - [🔸 `chstr = itos[i] + itos[j]`](#-chstr--itosi--itosj)
      - [🔸 `plt.text(j, i, chstr, ha='center', va='bottom', color='gray')`](#-plttextj-i-chstr-hacenter-vabottom-colorgray)
      - [🔸 `plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')`](#-plttextj-i-ni-jitem-hacenter-vatop-colorgray)
    - [🔹 `plt.axis('off')`](#-pltaxisoff)
    - [✅ 最终效果：](#-最终效果)
    - [🎯 总结一句话：](#-总结一句话-1)
- [deleting spurious (S) and (E) tokens in favor of a single . token](#deleting-spurious-s-and-e-tokens-in-favor-of-a-single--token)
- [删除多余的 (S) 和 (E) 标记，改用一个统一的 `.` 特殊符号](#删除多余的-s-和-e-标记改用一个统一的--特殊符号)
    - [✅ 我们将做出以下改变：](#-我们将做出以下改变)
    - [✅ 额外美化处理：](#-额外美化处理)
    - [✅ 更新后的行为：](#-更新后的行为)
    - [✅ 总结：](#-总结-3)
- [sampling from the model](#sampling-from-the-model)
- [从模型中进行采样（Sampling from the model）](#从模型中进行采样sampling-from-the-model)
    - [🔹 总体流程](#-总体流程)
  - [✅ 步骤详解](#-步骤详解)
    - [🔸 第一步：取起始行（即点号开头的频率分布）](#-第一步取起始行即点号开头的频率分布)
    - [🔸 第二步：将 raw count 转为概率分布](#-第二步将-raw-count-转为概率分布)
    - [🔸 第三步：使用 `torch.multinomial` 按概率采样](#-第三步使用-torchmultinomial-按概率采样)
    - [🔸 第四步：循环采样完整单词](#-第四步循环采样完整单词)
    - [🔸 示例采样代码简化版：](#-示例采样代码简化版)
    - [🔸 多次采样多个名字：](#-多次采样多个名字)
  - [🧠 模型的实际效果](#-模型的实际效果)
  - [✅ 与其他情况对比：](#-与其他情况对比)
    - [📉 使用随机均匀分布（完全未训练的模型）：](#-使用随机均匀分布完全未训练的模型)
    - [📈 使用训练过的 bigram 模型：](#-使用训练过的-bigram-模型)
  - [✅ 总结](#-总结-4)
  - [🎯 关键结论](#-关键结论)
  - [✅ 函数原型：](#-函数原型)
  - [✅ 参数解释：](#-参数解释)
  - [✅ 返回值：](#-返回值)
  - [✅ 示例 1：从概率分布中采样](#-示例-1从概率分布中采样)
  - [✅ 示例 2：多次采样 + 有放回](#-示例-2多次采样--有放回)
  - [✅ 示例 3：设定随机种子（保证可复现）](#-示例-3设定随机种子保证可复现)
  - [❗ 注意事项：](#-注意事项)
  - [✅ 应用场景（语言模型中）：](#-应用场景语言模型中)
  - [✅ 总结一句话：](#-总结一句话-2)
- [efficiency! vectorized normalization of the rows, tensor broadcasting](#efficiency-vectorized-normalization-of-the-rows-tensor-broadcasting)
- [提高效率！用向量化方法归一化 bigram 张量的每一行（Tensor Broadcasting）](#提高效率用向量化方法归一化-bigram-张量的每一行tensor-broadcasting)
  - [🎯 问题背景](#-问题背景)
  - [✅ 优化目标](#-优化目标)
  - [🔧 实现步骤](#-实现步骤)
    - [1. 转为 float 类型](#1-转为-float-类型)
    - [2. 对每一行做归一化（而不是对整个矩阵）](#2-对每一行做归一化而不是对整个矩阵)
      - [计算每一行的总和：](#计算每一行的总和)
      - [执行除法：](#执行除法)
  - [🧠 关于 Broadcasting（广播机制）](#-关于-broadcasting广播机制)
    - [✅ 示例：](#-示例)
  - [⚠️ Bug 警告：不要忘记 `keepdim=True`](#️-bug-警告不要忘记-keepdimtrue)
  - [✅ 正确 vs 错误对比：](#-正确-vs-错误对比)
  - [🧠 结论：Respect Broadcasting](#-结论respect-broadcasting)
  - [🛠 效率建议](#-效率建议)
  - [✅ 总结一句话：](#-总结一句话-3)
- [loss function (the negative log likelihood of the data under our model)](#loss-function-the-negative-log-likelihood-of-the-data-under-our-model)
    - [🎯 概述：我们已经训练了一个 Bigram 语言模型，它通过统计每对字符出现的频率来建立，然后归一化得到一个**概率矩阵** `P`，该矩阵表示每个字符后接另一个字符的概率。](#-概述我们已经训练了一个-bigram-语言模型它通过统计每对字符出现的频率来建立然后归一化得到一个概率矩阵-p该矩阵表示每个字符后接另一个字符的概率)
    - [🧠 为什么使用对数似然？](#-为什么使用对数似然)
    - [🚨 但问题来了：](#-但问题来了)
    - [❗ 损失函数的语义是：**越小越好**。](#-损失函数的语义是越小越好)
    - [📏 通常我们还会**平均化损失**，以便不同长度的句子/样本能公平比较：](#-通常我们还会平均化损失以便不同长度的句子样本能公平比较)
    - [✅ 总结逻辑链：](#-总结逻辑链)
    - [📌 举例：](#-举例)
    - [💡 延伸：](#-延伸)
- [model smoothing with fake counts](#model-smoothing-with-fake-counts)
    - [🧪 【背景问题：模型对未见过的 bigram 给出零概率】](#-背景问题模型对未见过的-bigram-给出零概率)
    - [😬 问题分析](#-问题分析)
    - [✅ 解决方案：**模型平滑（Model Smoothing）**](#-解决方案模型平滑model-smoothing)
      - [🔧 操作方法：](#-操作方法)
    - [🌊 平滑程度可调](#-平滑程度可调)
    - [🧾 效果分析：](#-效果分析)
    - [📌 总结一句话：](#-总结一句话-4)
- [PART 2: the neural network approach: intro](#part-2-the-neural-network-approach-intro)
  - [🔢 第二部分：神经网络方法简介](#-第二部分神经网络方法简介)
  - [🤖 现在，我们将采用一个**完全不同的方式** —— 用神经网络来做！](#-现在我们将采用一个完全不同的方式--用神经网络来做)
    - [🎯 新目标：把 bigram 字符级语言模型 **转化为神经网络任务**。](#-新目标把-bigram-字符级语言模型-转化为神经网络任务)
  - [🧠 训练方法：](#-训练方法)
  - [🔁 总结一下流程：](#-总结一下流程)
- [creating the bigram dataset for the neural net](#creating-the-bigram-dataset-for-the-neural-net)
  - [📊 为神经网络创建 bigram 数据集](#-为神经网络创建-bigram-数据集)
    - [🛠 步骤解析](#-步骤解析)
    - [🧩 Bigram 结构举例：](#-bigram-结构举例)
    - [🧮 代码逻辑：](#-代码逻辑)
    - [📦 示例输出（以 `"emma"` 为例）：](#-示例输出以-emma-为例)
  - [⚠️ 小心 Tensor 的构建方式！](#️-小心-tensor-的构建方式)
    - [✅ 总结建议：](#-总结建议)
- [feeding integers into neural nets? one-hot encodings](#feeding-integers-into-neural-nets-one-hot-encodings)
  - [🎯 将整数输入神经网络？使用 One-hot 编码](#-将整数输入神经网络使用-one-hot-编码)
    - [❌ 问题：整数不能直接作为神经网络输入](#-问题整数不能直接作为神经网络输入)
    - [✅ 解决方案：One-hot 编码](#-解决方案one-hot-编码)
    - [💡 PyTorch 中的 One-hot 编码](#-pytorch-中的-one-hot-编码)
    - [📊 示例结果：](#-示例结果)
    - [📈 可视化：](#-可视化)
    - [⚠️ 小心数据类型！](#️-小心数据类型)
  - [✅ 总结：](#-总结-5)
- [the "neural net": one linear layer of neurons implemented with matrix multiplication](#the-neural-net-one-linear-layer-of-neurons-implemented-with-matrix-multiplication)
  - [🧠「神经网络」的第一层：用矩阵乘法实现的线性层（Linear Layer）](#神经网络的第一层用矩阵乘法实现的线性层linear-layer)
    - [🎯 一个神经元的计算过程回顾：](#-一个神经元的计算过程回顾)
    - [🛠 第一步：定义权重 W](#-第一步定义权重-w)
    - [🧮 第二步：进行矩阵乘法](#-第二步进行矩阵乘法)
    - [🎯 拓展：用 27 个神经元代替 1 个](#-拓展用-27-个神经元代替-1-个)
    - [🧪 验证：点积确实是这么来的](#-验证点积确实是这么来的)
    - [✅ 总结](#-总结-6)
- [transforming neural net outputs into probabilities: the softmax](#transforming-neural-net-outputs-into-probabilities-the-softmax)
  - [🔁 将神经网络的输出转换为概率：Softmax 函数](#-将神经网络的输出转换为概率softmax-函数)
    - [🧠 我们想让输出代表什么？](#-我们想让输出代表什么)
    - [❓如何把这些输出变成“概率”？](#如何把这些输出变成概率)
    - [🧮 Softmax 的操作过程：](#-softmax-的操作过程)
    - [✅ 得到的结果是什么？](#-得到的结果是什么)
    - [🔄 举个例子：](#-举个例子-1)
    - [🎯 为什么要这么做？](#-为什么要这么做)
    - [🔚 总结](#-总结-7)
- [summary, preview to next steps, reference to micrograd](#summary-preview-to-next-steps-reference-to-micrograd)
    - [🧩 整体结构和流程回顾：](#-整体结构和流程回顾)
    - [📉 损失计算：Negative Log Likelihood Loss（负对数似然）](#-损失计算negative-log-likelihood-loss负对数似然)
    - [🎲 为什么 loss 可能高？](#-为什么-loss-可能高)
    - [⚠️ 这不是训练，这只是 forward 过程！](#️-这不是训练这只是-forward-过程)
    - [🔁 与 micrograd 对比：](#-与-micrograd-对比)
    - [✅ 小结：](#-小结)
- [vectorized loss](#vectorized-loss)
    - [🧠 原文翻译 + 解释：](#-原文翻译--解释)
    - [✅ PyTorch 实现向量索引：](#-pytorch-实现向量索引)
    - [🧮 然后计算损失：](#-然后计算损失)
    - [🧾 结果：](#-结果)
    - [📌 总结：](#-总结-8)
- [backward and update, in PyTorch](#backward-and-update-in-pytorch)
- [putting everything together](#putting-everything-together)
- [note 1: one-hot encoding really just selects a row of the next Linear layer's weight matrix](#note-1-one-hot-encoding-really-just-selects-a-row-of-the-next-linear-layers-weight-matrix)
- [note 2: model smoothing as regularization loss](#note-2-model-smoothing-as-regularization-loss)
- [sampling from the neural net](#sampling-from-the-neural-net)
- [conclusion](#conclusion)

# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### License

MIT

# makemore

makemore 接受一个文本文件作为输入，其中每一行被视为一个训练样本，并生成类似的内容。在后台，它是一个自回归的字符级语言模型，支持从二元组到Transformer（正如在GPT中看到的）等多种模型。例如，我们可以给它一个名字的数据库，makemore 将生成听起来像名字的酷婴儿名字建议，但这些名字并不是已存在的名字。或者，如果我们给它一个公司名称的数据库，它就能生成新的公司名称创意。或者我们可以给它有效的拼字游戏单词，makemore 将生成类似英语的胡言乱语。

这不是一个复杂的库，没有亿万个开关和按钮。它只是一个可以修改的文件，主要用于教育目的。唯一的依赖是 PyTorch。

当前的实现参考了几篇关键论文：

* Bigram（一个字符预测下一个字符，通过查找计数表）
* MLP，参考 Bengio 等人 2003
* CNN，参考 DeepMind WaveNet 2016（进行中...）
* RNN，参考 Mikolov 等人 2010
* LSTM，参考 Graves 等人 2014
* GRU，参考 Kyunghyun Cho 等人 2014
* Transformer，参考 Vaswani 等人 2017

### 使用方法

所包含的 `names.txt` 数据集作为示例，包含了来自 ssa.gov 的2018年最常见的32K个名字，格式如下：

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

我们可以这样运行脚本：

```
$ python makemore.py -i names.txt -o names
```

训练进度、日志和模型将会保存到工作目录 `names` 中。默认模型是一个非常小的200K参数的Transformer；更多的训练配置可用——请查看 argparse 并阅读代码。训练不需要任何特殊硬件，它可以在我的 Macbook Air 上运行，也可以在其他任何设备上运行，但如果有 GPU，训练会更快。随着训练的进行，脚本会定期打印一些样本。如果你想手动采样，可以使用 `--sample-only` 标志，例如，在一个单独的终端中执行：

```
$ python makemore.py -i names.txt -o names --sample-only
```

这将加载到目前为止表现最好的模型，并按需打印更多样本。以下是一些在当前默认设置下最终生成的独特婴儿名字（测试对数概率约为1.92，尽管通过调整超参数可以达到更低的对数概率）：

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

玩得开心！

### 许可证

MIT

# makemore.py introduction

you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.

你给这个脚本一些单词（每行一个），它将生成更多类似的内容。
使用最先进的 Transformer AI 技术。
这段代码旨在非常易于修改。根据你的需求进行调整。

与 minGPT 的变化：

* 我移除了 `from_pretrained` 函数，原本用于初始化 GPT2 权重。
* 我移除了 dropout 层，因为我们在这里训练的模型很小，在这个阶段和规模下不需要。
* 我移除了权重衰减以及关于哪些参数需要衰减、哪些不需要的所有复杂性。我认为在我们操作的规模下，这不会产生巨大差异。

# The spelled-out intro to language modeling: building makemore 视频介绍

We implement a bigram character-level language model, which we will further complexify in followup videos into a modern Transformer language model, like GPT. In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).

Links:
- makemore on github: https://github.com/karpathy/makemore
- jupyter notebook I built in this video: https://github.com/karpathy/nn-zero-t...
- my website: https://karpathy.ai
- my twitter:   / karpathy  
- (new) Neural Networks: Zero to Hero series Discord channel:   / discord   , for people who'd like to chat more and go beyond youtube comments

Useful links for practice:
- Python + Numpy tutorial from CS231n https://cs231n.github.io/python-numpy... . We use torch.tensor instead of numpy.- array in this video. Their design (e.g. broadcasting, data types, etc.) is so similar that practicing one is basically practicing the other, just be careful with some of the APIs - how various functions are named, what arguments they take, etc. - these details can vary.
- PyTorch tutorial on Tensor https://pytorch.org/tutorials/beginne...
- Another PyTorch intro to Tensor https://pytorch.org/tutorials/beginne...

Exercises:
- E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
- E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
- E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
- E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
- E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?
- E06: meta-exercise! Think of a fun/interesting exercise and complete it.

Chapters:
- 00:00:00 intro
- 00:03:03 reading and exploring the dataset
- 00:06:24 exploring the bigrams in the dataset
- 00:09:24 counting bigrams in a python dictionary
- 00:12:45 counting bigrams in a 2D torch tensor ("training the model")
- 00:18:19 visualizing the bigram tensor
- 00:20:54 deleting spurious (S) and (E) tokens in favor of a single . token
- 00:24:02 sampling from the model
- 00:36:17 efficiency! vectorized normalization of the rows, tensor broadcasting 
- 00:50:14 loss function (the negative log likelihood of the data under our model)
- 01:00:50 model smoothing with fake counts
- 01:02:57 PART 2: the neural network approach: intro
- 01:05:26 creating the bigram dataset for the neural net
- 01:10:01 feeding integers into neural nets? one-hot encodings
- 01:13:53 the "neural net": one linear layer of neurons implemented with matrix multiplication
- 01:18:46 transforming neural net outputs into probabilities: the softmax
- 01:26:17 summary, preview to next steps, reference to micrograd
- 01:35:49 vectorized loss
- 01:38:36 backward and update, in PyTorch
- 01:42:55 putting everything together
- 01:47:49 note 1: one-hot encoding really just selects a row of the next Linear layer's weight matrix
- 01:50:18 note 2: model smoothing as regularization loss
- 01:54:31 sampling from the neural net
- 01:56:16 conclusion

我们实现了一个二元组（bigram）字符级语言模型，之后我们将在后续视频中将其逐步复杂化，最终发展成像 GPT 那样的现代 Transformer 语言模型。
本视频的重点在于：

1. 介绍 `torch.Tensor`，包括其细节和在高效评估神经网络中的使用方式；
2. 讲解语言建模的整体框架，包括模型训练、采样，以及损失的评估（例如分类任务中的负对数似然）。

---

🔗 **相关链接：**

* **makemore 项目 GitHub**：[https://github.com/karpathy/makemore](https://github.com/karpathy/makemore)
* **本视频中构建的 Jupyter Notebook**：[https://github.com/karpathy/nn-zero-t](https://github.com/karpathy/nn-zero-t)...
* **我的网站**：[https://karpathy.ai](https://karpathy.ai)
* **我的推特**：[@karpathy](https://twitter.com/karpathy)
* **新建的神经网络「从零到精通」系列 Discord 频道**：用于更深入交流，适合不满足于只看 YouTube 评论的朋友。

---

🛠 **练习推荐：**

* **E01**：训练一个三元组（trigram）语言模型，也就是说，输入两个字符来预测第三个字符。你可以使用计数方式或神经网络。评估损失，看看是否优于 bigram 模型？
* **E02**：将数据集随机划分为 80% 训练集、10% 验证集、10% 测试集。分别在训练集上训练 bigram 和 trigram 模型，并在验证集与测试集上评估性能。你观察到了什么？
* **E03**：使用验证集调节 trigram 模型中的平滑（或正则化）强度——尝试多个设置，并观察哪个在验证集上的损失最小。在训练集与验证集损失随平滑强度变化时你观察到什么规律？用最优设置在测试集上评估一次最终损失。
* **E04**：我们看到 one-hot 向量只是用于选择矩阵 W 的某一行，因此显式生成 one-hot 向量有些浪费。你能否去掉 `F.one_hot` 的用法，改为直接索引矩阵的行？
* **E05**：查阅并使用 `F.cross_entropy`，它应该能得到相同的结果。你能想到为什么我们更愿意用 `F.cross_entropy` 吗？
* **E06**：元练习！自己设计一个有趣/有创意的练习，并完成它。

---

📚 **实用学习链接：**

* Python + Numpy 教程（来自 CS231n）：[https://cs231n.github.io/python-numpy/](https://cs231n.github.io/python-numpy/)

  > 本视频中我们使用的是 `torch.tensor` 而不是 `numpy.array`，但两者设计非常相似（如广播、数据类型等），练习一个几乎等于练习另一个。注意细节：函数命名、参数等 API 差异。

* PyTorch Tensor 教程：[https://pytorch.org/tutorials/beginner/introyt/tensors\_deeper\_tutorial.html](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)

* 另一个 PyTorch 入门 Tensor 教程：[https://pytorch.org/tutorials/beginner/basics/tensorqs\_tutorial.html](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

---

📺 **章节目录：**

```
00:00:00 介绍
00:03:03 读取并探索数据集
00:06:24 探索数据集中的 bigram
00:09:24 使用 Python 字典统计 bigram
00:12:45 使用 2D torch 张量统计 bigram（“训练模型”）
00:18:19 可视化 bigram 张量
00:20:54 用单个 “.” 替代起始 (S) 和结束 (E) 标记
00:24:02 从模型中进行采样
00:36:17 提升效率！对行进行向量化归一化，张量广播
00:50:14 损失函数（模型下数据的负对数似然）
01:00:50 使用虚假计数进行模型平滑
01:02:57 第2部分：神经网络方法介绍
01:05:26 为神经网络创建 bigram 数据集
01:10:01 将整数输入神经网络？使用 one-hot 编码
01:13:53 构建神经网络：一层线性神经元（矩阵乘法实现）
01:18:46 将神经网络输出转化为概率：softmax
01:26:17 总结 + 下一步预告 + 提到 micrograd
01:35:49 向量化损失计算
01:38:36 PyTorch 实现反向传播与更新
01:42:55 模型整合
01:47:49 附注1：one-hot 编码实质是选线性层权重矩阵的一行
01:50:18 附注2：模型平滑作为正则化损失
01:54:31 从神经网络中采样
01:56:16 总结
```


字幕翻译

# intro

hi everyone hope you're well and next up what i'd like to do is i'd like to build out make more
like micrograd before it make more is a repository that i have on my github webpage
you can look at it but just like with micrograd i'm going to build it out step by step and i'm going to spell everything out so we're
going to build it out slowly and together now what is make more make more as the name suggests
makes more of things that you give it so here's an example names.txt is an example dataset to make
more and when you look at names.txt you'll find that it's a very large data set of
names so here's lots of different types of names in fact i believe there are 32 000 names
that i've sort of found randomly on the government website and if you train make more on this data
set it will learn to make more of things like this
and in particular in this case that will mean more things that sound name-like
but are actually unique names and maybe if you have a baby and you're trying to assign name maybe you're looking for a cool new sounding unique
name make more might help you so here are some example generations from the neural network
once we train it on our data set so here's some example unique names that it will generate
dontel irot zhendi and so on and so all these are sound
name like but they're not of course names so under the hood make more is a
character level language model so what that means is that it is treating every single line here as an example and
within each example it's treating them all as sequences of individual characters so r e e s e is this example
and that's the sequence of characters and that's the level on which we are building out make more and what it means
to be a character level language model then is that it's just uh sort of modeling those sequences of characters
and it knows how to predict the next character in the sequence now we're actually going to implement a
large number of character level language models in terms of the neural networks that are involved in predicting the next
character in a sequence so very simple bi-gram and back of work models multilingual perceptrons recurrent
neural networks all the way to modern transformers in fact the transformer that we will build will be basically the
equivalent transformer to gpt2 if you have heard of gpt uh so that's kind of a big deal it's a modern network and by
the end of the series you will actually understand how that works um on the level of characters now to give you a
sense of the extensions here uh after characters we will probably spend some time on the word level so that we can
generate documents of words not just little you know segments of characters but we can generate entire large much
larger documents and then we're probably going to go into images and image text
networks such as dolly stable diffusion and so on but for now we have to start
here character level language modeling let's go so like before we are starting with a completely blank jupiter notebook page

# 介绍

大家好，希望你们一切都好。接下来我想做的是构建 make more，就像之前的 micrograd 一样，make more 是我在 GitHub 上的一个仓库，你可以去看看。但就像 micrograd 一样，我会一步步地构建它，并且详细解释每一个步骤，我们将一起慢慢构建它。那么，什么是 make more 呢？顾名思义，make more 是让你给它的东西生成更多类似的东西。比如说，`names.txt` 就是一个用于 make more 的示例数据集。当你查看 `names.txt` 时，你会发现它是一个非常大的名字数据集。里面有很多不同类型的名字，实际上，我相信这些名字有大约 32000 个，是我从政府网站随机找到的。如果你用这个数据集训练 make more，它会学习生成更多类似的东西，特别是像这样听起来像名字的东西，但实际上是独一无二的名字。也许你有一个宝宝，正在为他/她选择名字，可能你在寻找一个独特且听起来酷的新名字，make more 可能会帮到你。那么，这里有一些神经网络在我们用数据集训练后生成的示例名字。这里是一些它会生成的独特名字：dontel, irot, zhendi 等等。所有这些名字听起来像名字，但它们当然并不是已存在的名字。

在后台，make more 是一个字符级语言模型。这意味着它将每一行都当作一个示例，并且在每个示例中，它将这些示例看作是单独字符的序列。比如说，`r e e s e` 就是一个示例，它是一个字符序列，我们就是在这个级别上构建 make more。作为一个字符级语言模型，它的意思就是，它建模这些字符序列，并且能够预测序列中的下一个字符。

我们实际上将实现大量的字符级语言模型，包括简单的二元组模型和工作模型、多层感知器、递归神经网络，一直到现代的 Transformer。事实上，我们将构建的 Transformer 基本上是 GPT-2 的等效模型，如果你听说过 GPT，那么这就是一个大事，它是一个现代的网络，到系列的最后，你将真正理解它如何在字符级别上运作。

为了让你们更清楚接下来的扩展，除了字符级别外，我们可能还会花一些时间在单词级别上，这样我们就可以生成完整的单词文档，而不仅仅是字符的小片段。我们可以生成更大规模的文档，然后我们可能还会进入图像和图像文本网络，比如 Dolly、Stable Diffusion 等等。但目前我们必须从这里开始——字符级语言建模。那么，开始吧！就像之前一样，我们从一个完全空白的 Jupyter Notebook 页面开始。

# reading and exploring the dataset

the first thing is i would like to basically load up the dataset names.txt so we're going to open up names.txt for
reading and we're going to read in everything into a massive string
and then because it's a massive string we'd only like the individual words and put them in the list so let's call split lines
on that string to get all of our words as a python list of strings
so basically we can look at for example the first 10 words and we have that it's a list of emma
olivia eva and so on and if we look at the top of the page here that is indeed
what we see um so that's good this list actually makes me feel that
this is probably sorted by frequency but okay so
these are the words now we'd like to actually like learn a little bit more about this data set let's look at the
total number of words we expect this to be roughly 32 000 and then what is the for example
shortest word so min of length of each word for w inwards
so the shortest word will be length two and max of one w for w in words so the
longest word will be 15 characters so let's now think through our very first language model
as i mentioned a character level language model is predicting the next character in a sequence given already
some concrete sequence of characters before it now we have to realize here is that every single word here like isabella is
actually quite a few examples packed in to that single word because what is an existence of a word
like isabella in the data set telling us really it's saying that the character i is a very likely
character to come first in the sequence of a name the character s is likely to come
after i the character a is likely to come after is
the character b is very likely to come after isa and so on all the way to a following isabel
and then there's one more example actually packed in here and that is that after there's isabella
the word is very likely to end so that's one more sort of explicit piece of information that we have here
that we have to be careful with and so there's a lot backed into a single individual word in terms of the
statistical structure of what's likely to follow in these character sequences and then of course we don't have just an
individual word we actually have 32 000 of these and so there's a lot of structure here to model
now in the beginning what i'd like to start with is i'd like to start with building a bi-gram language model
now in the bigram language model we're always working with just two characters at a time
so we're only looking at one character that we are given and we're trying to predict the next character in the
sequence so um what characters are likely to follow are what characters are likely to
follow a and so on and we're just modeling that kind of a little local structure and we're forgetting the fact that we
may have a lot more information we're always just looking at the previous character to predict the next one so
it's a very simple and weak language model but i think it's a great place to start so now let's begin by looking at these

# 读取和探索数据集

首先，我想做的基本步骤是加载数据集 `names.txt`。我们将打开 `names.txt` 以读取内容，并把所有内容读取为一个大的字符串。
然后，由于这是一个长字符串，我们希望获取其中的每一个单词，并把它们放入一个列表中，所以我们会对这个字符串使用 `splitlines()`，把它拆分成一个 Python 的字符串列表。
接下来我们可以查看前十个单词，例如前10个是 `emma`、`olivia`、`eva` 等等。
如果我们看页面的顶部，那确实是我们看到的内容，说明读取成功了。

这个列表看起来让我感觉这些名字可能是按出现频率排序的，但没关系。
这些就是我们要处理的“单词”。

接下来我们想进一步了解一下这个数据集。比如说：

* 总共有多少个名字？我们预计大概是 32000 个。
* 最短的名字有多长？我们可以计算每个名字的长度，取最小值，结果是 2。
* 最长的名字有多长？同样的方法，最大长度是 15 个字符。

现在我们开始思考我们的第一个语言模型。

正如我之前提到的，字符级语言模型的目标是：**在给定前面一段具体字符序列的情况下，预测下一个字符**。
现在我们要意识到的一点是：数据集中每一个单词，比如 "isabella"，实际上在统计结构上包含了许多预测信息。

比如说，"isabella" 这个名字的存在告诉我们很多事情：

* 字符 `i` 很可能是名字开头的字符，
* 字符 `s` 很可能出现在 `i` 之后，
* 字符 `a` 很可能出现在 `is` 之后，
* 字符 `b` 很可能出现在 `isa` 之后，
* 一直到最后 `a` 可能出现在 `isabell` 后面。

并且还有一点隐含的信息是：当我们看到完整的 "isabella" 后，名字很可能就结束了。因此，"结束" 也是一个明确的预测目标。

所以在一个单词中，实际上包含了丰富的结构和信息，用来训练字符级语言模型。而我们不只有一个单词，总共有约 32000 个这样的单词，因此可以提取出大量有结构的训练数据。

在一开始，我们想从最简单的模型做起——**bigram 语言模型**。

在 bigram 模型中，我们始终只关注两个字符：我们有一个字符作为输入，要预测下一个字符是谁。

比如：

* 哪些字符常常出现在 `a` 后面？
* 哪些字符常常出现在 `b` 后面？

我们只建模这种**局部结构**，不考虑更长的上下文，也就是只看前一个字符来预测下一个字符。

这是一种非常简单且较弱的语言模型，但它是一个很好的起点。
接下来我们就要开始实现它。

如何上传本地文件 names.txt

点击左侧的文件夹，上传之后和sample_data并列，参考 https://blog.csdn.net/lcnana/article/details/122409044

# exploring the bigrams in the dataset

bi-grams in our data set and what they look like and these bi-grams again are just two characters in a row
so for w in words each w here is an individual word a string
we want to iterate uh for we're going to iterate this word
with consecutive characters so two characters at a time sliding it through the word now a interesting nice way cute
way to do this in python by the way is doing something like this for character one character two in zip off
w and w at one one column
print character one character two and let's not do all the words let's just do the first three words and i'm
going to show you in a second how this works but for now basically as an example let's just do the very first word alone
emma you see how we have a emma and this will just print e m m m a
and the reason this works is because w is the string emma w at one column is
the string mma and zip takes two iterators and it pairs them up
and then creates an iterator over the tuples of their consecutive entries and if any one of these lists is shorter
than the other then it will just halt and return so basically that's why we return em mmm
ma but then because this iterator second one here runs out of elements zip just
ends and that's why we only get these tuples so pretty cute so these are the consecutive elements in
the first word now we have to be careful because we actually have more information here than just these three
examples as i mentioned we know that e is the is very likely to come first and
we know that a in this case is coming last so one way to do this is basically we're
going to create a special array here all characters
and um we're going to hallucinate a special start token here
i'm going to call it like special start so this is a list of one element
plus w and then plus a special end character
and the reason i'm wrapping the list of w here is because w is a string emma list of w will just have the individual
characters in the list and then doing this again now but not iterating
over w's but over the characters will give us something like this
so e is likely so this is a bigram of the start character and e and this is a bigram of the
a and the special end character and now we can look at for example what this looks like for
olivia or eva and indeed we can actually potentially do this for the entire data
set but we won't print that that's going to be too much but these are the individual character diagrams and we can print them

# 探索数据集中的二元组

我们来看看数据集中的二元组是什么样的。二元组就是相邻的两个字符。
对于每个单词 `w`，每个 `w` 是一个单独的字符串，我们想要迭代这个单词，将其按顺序拆分成两个字符一组，滑动遍历整个单词。
顺便提一下，Python 中有一个很有趣且简洁的方式来做这个：你可以使用像这样的代码：

```python
for character1, character2 in zip(w, w[1:]):
    print(character1, character2)
```

不过我们先不对所有单词进行操作，只处理前三个单词。我马上给你演示这个方法是如何工作的，但现在先举个简单的例子，只处理第一个单词 `emma`。
你会看到输出是这样的：
`e m`, `m m`, `m a`。
之所以会这样输出，是因为 `w` 是字符串 "emma"，`w[1:]` 是字符串 "mma"。
`zip` 会将这两个迭代器配对，然后生成一系列连续字符的元组。如果其中一个列表的长度比另一个短，`zip` 会停止并返回结果。所以，我们得到的是 `e m`, `m m`, `m a`，然后由于第二个迭代器没有更多元素，`zip` 就结束了，因此我们只得到这几个元组。这很有趣吧！

这些就是第一个单词中的连续字符对。

不过我们需要注意的是，我们实际上拥有比这三个例子更多的信息。正如我之前提到的，我们知道字符 `e` 很可能出现在名字的开头，而字符 `a` 很可能出现在结尾。

一种方法是，我们为每个单词创建一个特殊的数组，包括每个字符，并为每个单词加上一个特殊的开始符号和结束符号。
我们可以把它称作 `special_start`，所以数组会变成这样的格式：`[special_start] + w + [special_end]`。
这样做的原因是，`w` 是一个字符串 "emma"，而 `list(w)` 就会把它转化为一个字符列表。然后，当我们用这种方式再次进行迭代时，会得到以下的二元组：
`special_start` 和 `e`，`e` 和 `m`，`m` 和 `m`，`m` 和 `a`，`a` 和 `special_end`。

接下来我们可以看一下类似的情况，比如 `olivia` 或 `eva`，并且我们实际上可以对整个数据集进行操作，不过我们不会打印出来，因为那会太多了。但这些就是我们得到的每个单词的二元组。我们可以把它们打印出来查看。

# counting bigrams in a python dictionary

now in order to learn the statistics about which characters are likely to follow other characters the simplest way
in the bigram language models is to simply do it by counting so we're basically just going to count
how often any one of these combinations occurs in the training set in these words
so we're going to need some kind of a dictionary that's going to maintain some counts for every one of these diagrams
so let's use a dictionary b and this will map these bi-grams so
bi-gram is a tuple of character one character two and then b at bi-gram
will be b dot get of bi-gram which is basically the same as b at bigram
but in the case that bigram is not in the dictionary b we would like to by default return to zero
plus one so this will basically add up all the bigrams and count how often they occur
let's get rid of printing or rather let's keep the printing and let's just
inspect what b is in this case and we see that many bi-grams occur just
a single time this one allegedly occurred three times so a was an ending character three times
and that's true for all of these words all of emma olivia and eva and with a
so that's why this occurred three times now let's do it for all the words
oops i should not have printed i'm going to erase that
let's kill this let's just run and now b will have the statistics of
the entire data set so these are the counts across all the words of the individual pie grams
and we could for example look at some of the most common ones and least common ones
this kind of grows in python but the way to do this the simplest way i like is we just use b dot items
b dot items returns the tuples of key value in this case the keys are
the character diagrams and the values are the counts and so then what we want to do is we
want to do sorted of this
but by default sort is on the first
on the first item of a tuple but we want to sort by the values which are the second element of a tuple that is the
key value so we want to use the key equals lambda
that takes the key value and returns the key value at the one not at zero but
at one which is the count so we want to sort by the count of these elements
and actually we wanted to go backwards so here we have is the bi-gram q and r occurs only a single
time dz occurred only a single time and when we sort this the other way around
we're going to see the most likely bigrams so we see that n was very often an ending character
many many times and apparently n almost always follows an a and that's a very likely combination as
well so this is kind of the individual counts
that we achieve over the entire data set now it's actually going to be significantly more convenient for us to

# 在 Python 字典中统计二元组（bigrams）

现在，为了了解哪些字符更可能出现在其他字符之后（即字符之间的统计关系），在 bigram 语言模型中，最简单的方式就是直接**数数**。
我们只需要统计每种字符对（bigram）在训练集（这些单词）中出现的次数。

所以我们需要一个字典 `b`，它用来记录每一个 bigram 出现的次数。
我们使用一个字典 `b`，它的键（key）是字符对的元组 `(character1, character2)`，值（value）是它们出现的次数。

```python
bigram = (ch1, ch2)
b[bigram] = b.get(bigram, 0) + 1
```

这行代码的含义是：

* `b.get(bigram, 0)` 会在字典中查找这个 bigram，如果没有找到就返回默认值 0；
* 然后加 1，相当于把这个 bigram 的出现次数加一。

这样我们就能遍历整个数据集，并把所有的 bigram 出现次数累加到字典里。

我们可以把打印结果保留一下，检查 `b` 的内容。我们看到很多 bigram 只出现了一次。
比如某个 bigram 出现了三次，表示字符 `a` 是结尾字符的情况发生了三次，这确实出现在了 “emma”、“olivia” 和 “eva” 中。

接着我们对 **所有单词** 运行这段统计代码，构建整个数据集的 bigram 统计表。
不过我们不再打印这些内容，因为太多了。运行完后，`b` 中就包含了所有单词中每一个字符 bigram 的出现频次。

现在我们可以分析最常见的和最罕见的 bigram。Python 中最简单的做法就是使用 `b.items()`：

```python
b.items()
```

它返回的是字典中的 `(key, value)` 元组列表，即 `(bigram, count)`。

接着我们可以对这些项进行排序：

```python
sorted(b.items(), key=lambda kv: kv[1])
```

这里的 `lambda kv: kv[1]` 表示我们按第二项（出现次数）排序，而不是默认的按键（bigram）排序。

我们还可以加上 `reverse=True` 实现**从高到低排序**，来查看最常出现的 bigram：

```python
sorted(b.items(), key=lambda kv: kv[1], reverse=True)
```

比如我们会发现字符 `n` 作为结尾的 bigram 出现了很多次，或者字符 `a` 后面跟着 `n` 是非常常见的组合。

这些就是我们通过遍历整个数据集得到的每个字符对的统计信息，后续我们就可以利用这些统计结果来建模或采样新数据了。

---

python lambda 解释一下

在 Python 中，`lambda` 是用来创建\*\*匿名函数（没有名字的函数）\*\*的一种简洁语法。

---

### 一般形式：

```python
lambda 参数: 表达式
```

这定义了一个函数，它接受一些参数，并返回一个表达式的结果。

相当于下面这种普通函数的简写形式：

```python
def 函数名(参数):
    return 表达式
```

---

### 举例说明：

#### ✅ 示例 1：普通函数 vs lambda

```python
# 普通函数
def add(x, y):
    return x + y

# lambda函数
add_lambda = lambda x, y: x + y

print(add(2, 3))         # 输出 5
print(add_lambda(2, 3))  # 输出 5
```

---

#### ✅ 示例 2：配合 sorted 使用

比如你有一个列表，想按第二个元素排序：

```python
items = [('a', 3), ('b', 1), ('c', 2)]
sorted_items = sorted(items, key=lambda x: x[1])
print(sorted_items)  # 输出 [('b', 1), ('c', 2), ('a', 3)]
```

解释：

* `lambda x: x[1]` 创建了一个匿名函数，表示“取每个元素的第1个索引（即第2个值）”
* `sorted(..., key=...)` 就会用这个函数来进行排序

---

#### ✅ 示例 3：配合 `map()` 使用

```python
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, nums))
print(squared)  # 输出 [1, 4, 9, 16]
```

---

### 总结：

| 特点       | 说明                                    |
| -------- | ------------------------------------- |
| 匿名       | 没有名字，适合临时用一次的小函数                      |
| 表达式只能有一行 | 不能包含多条语句、赋值、循环等                       |
| 常用于函数式操作 | 如 `sorted`, `map`, `filter`, `reduce` |

---

如果你看到 `lambda`，可以理解成：**临时造了个小函数，干点事就扔掉**。常用来写简洁的代码，但不要滥用，逻辑复杂时最好用 `def` 明确写出函数。


# counting bigrams in a 2D torch tensor ("training the model")

keep this information in a two-dimensional array instead of a python dictionary
so we're going to store this information in a 2d array and
the rows are going to be the first character of the bigram and the columns are going to be the second character and
each entry in this two-dimensional array will tell us how often that first character files the second character in
the data set so in particular the array representation that we're going to use or the library is that of pytorch
and pytorch is a deep learning neural network framework but part of it is also this torch.tensor
which allows us to create multi-dimensional arrays and manipulate them very efficiently so
let's import pytorch which you can do by import torch and then we can create
arrays so let's create a array of zeros and we give it a
size of this array let's create a three by five array as an example and
this is a three by five array of zeros and by default you'll notice a.d type
which is short for data type is float32 so these are single precision floating point numbers
because we are going to represent counts let's actually use d type as torch dot and 32
so these are 32-bit integers so now you see that we have integer data
inside this tensor now tensors allow us to really manipulate all the individual entries
and do it very efficiently so for example if we want to change this bit we have to index into the tensor and in
particular here this is the first row and the
because it's zero indexed so this is row index one and column index zero one two
three so a at one comma three we can set that to one
and then a we'll have a 1 over there we can of course also do things like
this so now a will be 2 over there or 3.
and also we can for example say a 0 0 is 5 and then a will have a 5 over here
so that's how we can index into the arrays now of course the array that we are interested in is much much bigger so
for our purposes we have 26 letters of the alphabet and then we have two special characters
s and e so uh we want 26 plus 2 or 28 by 28
array and let's call it the capital n because it's going to represent sort of the counts
let me erase this stuff so that's the array that starts at zeros 28 by 28
and now let's copy paste this here but instead of having a dictionary b
which we're going to erase we now have an n now the problem here is that we have
these characters which are strings but we have to now um basically index into a
um array and we have to index using integers so we need some kind of a lookup table from characters to integers
so let's construct such a character array and the way we're going to do this is we're going to take all the words which
is a list of strings we're going to concatenate all of it into a massive string so this is just simply the entire data set as a single
string we're going to pass this to the set constructor which takes this massive
string and throws out duplicates because sets do not allow duplicates
so set of this will just be the set of all the lowercase characters
and there should be a total of 26 of them and now we actually don't want a set we
want a list but we don't want a list sorted in some weird arbitrary way we want it to be
sorted from a to z so sorted list
so those are our characters now what we want is this lookup table as
i mentioned so let's create a special s2i i will call it
um s is string or character and this will be an s2i mapping
for is in enumerate of these characters
so enumerate basically gives us this iterator over the integer index and the
actual element of the list and then we are mapping the character to the integer
so s2i is a mapping from a to 0 b to 1 etc all the way from z to 25
and that's going to be useful here but we actually also have to specifically set that s will be 26
and s to i at e will be 27 right because z was 25.
so those are the lookups and now we can come here and we can map both character 1 and character 2 to
their integers so this will be s2i at character 1 and ix2 will be s2i of character 2.
and now we should be able to do this line but using our array so n at
x1 ix2 this is the two-dimensional array indexing i've shown you before and honestly just plus equals one
because everything starts at zero so this should work
and give us a large 28 by 28 array of all these counts so
if we print n this is the array but of course it looks ugly so let's erase this ugly mess and

# 用 2D Torch 张量统计 bigram（“训练模型”）

我们现在想把统计信息存储在一个 **二维数组（2D array）** 中，而不是用 Python 字典。
在这个数组中：

* **行（row）代表 bigram 的第一个字符**，
* **列（column）代表第二个字符**，
* 每个数组元素（即二维坐标）记录了这个字符对在数据集中出现的次数。

我们将使用 **PyTorch** 来构建这个数组。PyTorch 是一个深度学习框架，其中的 `torch.tensor` 提供了创建和高效操作多维数组的功能。

---

### ✅ 步骤 1：导入 PyTorch

```python
import torch
```

---

### ✅ 步骤 2：创建张量示例

```python
a = torch.zeros((3, 5), dtype=torch.int32)
```

* 创建一个 3 行 5 列的全 0 整数张量。
* 默认数据类型是 `float32`，我们改成 `int32` 是因为要统计次数。

你可以像这样操作张量的某个元素：

```python
a[1, 3] = 1     # 第2行第4列设为1
a[1, 3] += 1    # 累加
a[0, 0] = 5     # 第1行第1列设为5
```

---

### ✅ 步骤 3：构建 28x28 的大张量

因为我们有：

* 26 个英文字母（a\~z）
* 加上两个特殊字符：`S`（开始）和 `E`（结束）

所以我们需要一个 **28×28** 的张量来表示所有可能的 bigram：

```python
N = torch.zeros((28, 28), dtype=torch.int32)
```

---

### ✅ 步骤 4：字符转整数的映射（lookup 表）

张量索引只能用整数，但字符是字符串类型，所以我们需要把每个字符映射成一个整数。

我们可以从数据集中收集所有字符：

```python
chars = sorted(list(set(''.join(words))))  # 把所有单词拼成一个字符串，再取唯一字符，再排序
```

然后构建字符 → 索引的映射字典：

```python
s2i = { ch:i for i, ch in enumerate(chars) }
s2i['S'] = 26   # 开始字符
s2i['E'] = 27   # 结束字符
```

---

### ✅ 步骤 5：填充张量（即 bigram 统计）

现在我们用字符对索引这个二维张量，并将对应位置的值加 1：

```python
for w in words:
    chs = ['S'] + list(w) + ['E']  # 给每个单词加上起始和结束符
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = s2i[ch1]
        ix2 = s2i[ch2]
        N[ix1, ix2] += 1
```

这样就完成了整个数据集中所有 bigram 的统计，保存在一个高效的 PyTorch 张量中。

你可以打印 `N` 看结果，但它是一个 28×28 的大数组，可能会很“丑”。接下来可以进一步分析或可视化它。

---

### ✅ 总结：

我们做了这些事：

* 用 `torch.zeros()` 建立 2D 统计张量
* 用 `set + sorted` 提取所有字符，构建字符→整数的映射
* 用两个字符组成 bigram，并在张量中索引对应位置 +1
* 用 PyTorch 高效存储和操作字符频率信息，为后续建模做准备

这相当于是我们 bigram 模型的“训练”过程，只不过用的是数数而不是神经网络。

---

这行代码：

```python
set(''.join(words))
```

的作用是：**提取列表 `words` 中所有单词中出现过的**“**唯一字符集合**”。

---

### 🔍 分步解释：

假设你有一个字符串列表：

```python
words = ["emma", "olivia", "ava"]
```

#### 第一步：`''.join(words)`

把所有单词拼接成一个长字符串，没有空格：

```python
''.join(words)  →  "emmaoliviaava"
```

#### 第二步：`set(...)`

把这个字符串转换成一个集合，自动去重，只保留每个字符出现一次：

```python
set("emmaoliviaava") → {'e', 'm', 'a', 'o', 'l', 'i', 'v'}
```

集合的特点是：**不重复、无序**。

---

### ✅ 用法场景

这个操作常用来：

* 获取训练数据中用到的所有字符（去重后）
* 构建字符到索引的映射（如 `{'a':0, 'b':1, ...}`）

---

### 🧠 举个例子再总结：

```python
words = ["cat", "car", "dog"]
set(''.join(words))  → {'c', 'a', 't', 'r', 'd', 'o', 'g'}
```

即：把所有单词连起来 → 拆成字符 → 去重 → 得到所有**唯一出现过的字符**

---

### ✅ 总结一句话：

```python
set(''.join(words))
```

这是一种快速获取训练语料中**全部不重复字符**的写法。

---

# visualizing the bigram tensor

let's try to visualize it a bit more nicer so for that we're going to use a library
called matplotlib so matplotlib allows us to create figures so we can do things like plt
item show of the counter array so this is the 28x28 array
and this is structure but even this i would say is still pretty ugly so we're going to try to create a much
nicer visualization of it and i wrote a bunch of code for that the first thing we're going to need is
we're going to need to invert this array here this dictionary so s2i
is mapping from s to i and in i2s we're going to reverse this dictionary so iterator of all the items
and just reverse that array so i2s maps inversely from 0 to a 1 to b etc
so we'll need that and then here's the code that i came up with to try to make this a little bit
nicer we create a figure we plot
n and then we do and then we visualize a bunch of things later let me just run it so you get a sense of what this is
okay so you see here that we have the array spaced out
and every one of these is basically like b follows g zero times
b follows h 41 times um so a follows j 175 times
and so what you can see that i'm doing here is first i show that entire array
and then i iterate over all the individual little cells here and i create a character string here
which is the inverse mapping i2s of the integer i and the integer j so those are
the bi-grams in a character representation and then i plot just the diagram text
and then i plot the number of times that this bigram occurs now the reason that there's a dot item
here is because when you index into these arrays these are torch tensors
you see that we still get a tensor back so the type of this thing you'd think it
would be just an integer 149 but it's actually a torch.tensor and so if you do dot item then it will pop out
that in individual integer so it will just be 149.
so that's what's happening there and these are just some options to make it look nice so what is the structure of this array
we have all these counts and we see that some of them occur often and some of them do not occur often now if you scrutinize this carefully you

# 可视化 bigram 张量

现在我们想把统计好的 bigram 张量（28×28）**更漂亮地可视化**出来。为此，我们将使用一个叫做 **matplotlib** 的可视化库，它可以创建图表和图形。

---

### ✅ 第一步：使用 matplotlib 简单可视化

我们可以用 `matplotlib.pyplot` 中的 `imshow` 方法展示二维数组：

```python
import matplotlib.pyplot as plt

plt.imshow(N)  # N 是 28×28 的 bigram 计数张量
plt.show()
```

这会画出一个矩阵热图，但样子可能还是比较“丑”，不太直观。

---

### ✅ 第二步：构造更美观的可视化

我们可以改进可视化效果。首先，我们需要把原来的字符→索引字典 `s2i` 反转，得到一个从索引→字符的字典 `i2s`：

```python
i2s = {i: s for s, i in s2i.items()}
```

这样我们就能在图里显示字符而不是数字，例如横轴是字符 b，纵轴是字符 g，表示的是 “g 后面接 b 出现了几次”。

---

### ✅ 第三步：完整可视化逻辑

你可以使用如下结构进行更详细的绘制（伪代码结构如下）：

```python
plt.figure(figsize=(16, 16))  # 创建大画布
plt.imshow(N, cmap='Blues')   # 用蓝色色带显示计数矩阵
for i in range(28):
    for j in range(28):
        chstr = i2s[i] + i2s[j]             # 比如 "th"、"ar" 等 bigram
        count = N[i, j].item()              # 获取具体数字（tensor → int）
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, count, ha='center', va='top', color='black')
plt.axis('off')
plt.show()
```

---

### ✅ 补充解释：

* `N[i, j]` 是 torch tensor，如果你直接打印是 `tensor(41)` 之类的对象。
  用 `.item()` 可以将其变为普通整数，如 `41`。
* `plt.text(...)` 是在图的某个位置写字，比如 bigram 的字符对和出现次数。
* 关闭坐标轴 (`axis('off')`) 让图更清爽。
* 这样你会看到一个 28×28 的字符矩阵，每个格子标注了：

  * 哪个字符组合（bigram）
  * 出现了多少次

---

### ✅ 总结：

我们做的事是：

1. 用 PyTorch 构建了一个 28×28 的 bigram 计数张量；
2. 用 matplotlib 展示了这个张量，视觉化统计结构；
3. 用字符标注每个 bigram 及其频次，提升可读性；
4. 学会了 `.item()` 把 tensor 转换为 Python 整数。

这个图能直观反映哪些 bigram 很常见，哪些几乎没出现。分析这些结构对构建语言模型非常有帮助。

---

这段代码用于**在 Jupyter Notebook 中可视化一个二维张量（例如 bigram 统计矩阵）**。
下面逐行解释：

---

### 🔸 `import matplotlib.pyplot as plt`

* 导入 Python 中最常用的可视化库 **Matplotlib** 的子模块 `pyplot`。
* 并给它起了一个简写名 `plt`，方便后面使用。
* `pyplot` 提供了类似 MATLAB 的绘图 API，比如画图、显示图像、添加文字等。

---

### 🔸 `%matplotlib inline`

* 这是 **Jupyter Notebook** 特有的“魔法命令”（magic command）。
* 它的作用是：**让图像直接显示在 notebook 单元格中，而不是弹出一个新窗口**。
* 你只需要在 notebook 的开头运行一次这个命令。

---

### 🔸 `plt.imshow(N)`

* `imshow` 是 `pyplot` 提供的函数，用来**以图像形式展示一个二维数组**。
* `N` 是一个 **28×28 的 PyTorch 张量**，记录了 bigram 出现的频率。
* `imshow` 会把它当成“像素矩阵”画出来，数值越大，颜色越深（默认是彩色热图）。

等价于可视化一个“统计热力图”或“灰度图”。

---

### ✅ 举个例子：

假设：

```python
N = torch.tensor([[1, 2], [3, 4]])
plt.imshow(N)
```

会显示一个 2×2 的彩色方格图，其中：

* 颜色深浅表示数字的大小，
* 左上角是 `1`，右下角是 `4`，颜色最深。

---

### ✅ 总结：

| 行号                                | 含义                             |
| --------------------------------- | ------------------------------ |
| `import matplotlib.pyplot as plt` | 导入绘图库                          |
| `%matplotlib inline`              | 让图嵌入在 Jupyter 单元格中显示           |
| `plt.imshow(N)`                   | 把张量 N 作为图像显示出来（比如 bigram 统计热图） |

这在分析神经网络中的矩阵结构、注意力图、统计频次等场景中非常常用。

---

这段代码是在 **Jupyter Notebook 中可视化 bigram 统计矩阵** `N`，并在图中**标注字符组合及其出现次数**。下面逐行详细解释：

---

### 🔹 `import matplotlib.pyplot as plt`

导入可视化库 Matplotlib 的 `pyplot` 模块，并简写为 `plt`，用于绘图。

---

### 🔹 `%matplotlib inline`

这是 Jupyter 的“魔法命令”，确保图像在 notebook 单元格内直接显示，而不是弹出窗口。

---

### 🔹 `plt.figure(figsize=(16, 16))`

创建一个新的绘图窗口（画布），大小为 16×16 英寸。

* 图像越大，越清晰，适合展示密集的文本信息（比如 28×28 的 bigram 矩阵）。

---

### 🔹 `plt.imshow(N, cmap='Blues')`

将张量 `N` 可视化为一幅图像（热力图）：

* `N` 是一个 28×28 的二维张量，表示字符 bigram 的出现频率。
* `cmap='Blues'` 表示使用“蓝色”色带，值越大颜色越深。

---

### 🔹 双层 `for` 循环：逐格标注字符和计数

```python
for i in range(28):
  for j in range(28):
```

遍历矩阵中的每一个元素，i 表示行索引（前一个字符），j 表示列索引（后一个字符）。

#### 🔸 `chstr = itos[i] + itos[j]`

* `itos` 是索引转字符的字典（例如：`{0:'a', 1:'b', ..., 26:'S', 27:'E'}`）
* 把整数索引 `i` 和 `j` 转换为字符后拼接成 bigram 组合，比如 `"th"`、`"an"` 等。

#### 🔸 `plt.text(j, i, chstr, ha='center', va='bottom', color='gray')`

* 在图像中 (j, i) 位置标注 bigram 字符串 `chstr`，比如 `"th"`
* `ha='center'`：水平居中；`va='bottom'`：垂直底部对齐
* `color='gray'`：使用灰色字体

#### 🔸 `plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')`

* 在相同位置再标注该 bigram 出现的次数
* `.item()` 将 `torch.tensor` 转换为 Python 整数（否则打印不出来）

---

### 🔹 `plt.axis('off')`

隐藏图像的坐标轴，让图表更清爽。

---

### ✅ 最终效果：

* 图像中是一个 28×28 的格子矩阵
* 每格上方显示字符组合（bigram），下方显示它的出现次数
* 蓝色深浅代表出现频率：颜色越深，出现越多
* 非常适合观察哪些字符组合在数据中最常见，哪些从未出现

---

### 🎯 总结一句话：

这段代码把字符级 bigram 的统计结果 `N` 转换成一个 **美观的可视化热力图**，每一格清晰地标出字符对和它的出现次数，非常适合教学或模型分析使用。

---

# deleting spurious (S) and (E) tokens in favor of a single . token

will notice that we're not actually being very clever that's because when you come over here you'll notice that for example we have
an entire row of completely zeros and that's because the end character is never possibly going to be the first
character of a bi-gram because we're always placing these end tokens all at the end of the diagram
similarly we have entire columns zeros here because the s
character will never possibly be the second element of a bigram because we always start with s and we end with e
and we only have the words in between so we have an entire column of zeros an entire row of zeros and in this little
two by two matrix here as well the only one that can possibly happen is if s directly follows e
that can be non-zero if we have a word that has no letters so in that case
there's no letters in the word it's an empty word and we just have s follows e but the other ones are just not possible
and so we're basically wasting space and not only that but the s and the e are getting very crowded here
i was using these brackets because there's convention and natural language processing to use these kinds of brackets to denote special tokens
but we're going to use something else so let's fix all this and make it prettier
we're not actually going to have two special tokens we're only going to have one special token so
we're going to have n by n array of 27 by 27 instead
instead of having two we will just have one and i will call it a dot
okay let me swing this over here
now one more thing that i would like to do is i would actually like to make this special character half position zero
and i would like to offset all the other letters off i find that a little bit more pleasing
so we need a plus one here so that the first character which is a will start at
one so s2i will now be a starts at one and dot is 0
and i2s of course we're not changing this because i2s just creates a reverse mapping and this will work fine so 1 is
a 2 is b 0 is dot so we've reversed that here
we have a dot and a dot this should work fine
make sure i start at zeros count and then here we don't go up to 28 we go
up to 27 and this should just work
okay so we see that dot never happened it's at zero because we don't have empty words
then this row here now is just uh very simply the um counts for all the first letters so
uh j starts a word h starts a word i starts a word etc and then these are all
the ending characters and in between we have the structure of what characters follow each other
so this is the counts array of our entire data set so this array actually has all

# 删除多余的 (S) 和 (E) 标记，改用一个统一的 `.` 特殊符号

我们可以注意到，之前的处理方式其实不是很聪明。比如你看可视化图时，会发现：

* 有一整行是全 0 的，那是因为结尾符号 `E` 不可能出现在 bigram 的第一个位置（即不会作为起始字符）。

  * 因为我们只在每个单词末尾放 `E`，所以它永远不会作为前一个字符。

* 同样，也有一整列是全 0 的，那是因为开始符号 `S` 不会出现在 bigram 的第二个位置（即不会作为结束字符）。

  * 因为我们只在单词前面加 `S`，从来不会让它出现在 bigram 的结尾。

此外，在左上角的这个小 2×2 矩阵（`S` 和 `E` 交叉）中，**唯一可能的组合是 `S` 直接跟着 `E`**（即空单词的情况），其他组合根本不可能出现。

所以我们浪费了不少空间，并且 `S` 和 `E` 的位置在图中很拥挤。

虽然使用 `S` 和 `E` 符号是自然语言处理中常见的做法（用类似方括号的标记表示特殊符号），但我们现在决定用更简洁的方式来处理：

---

### ✅ 我们将做出以下改变：

1. **不再使用两个特殊字符 `S` 和 `E`**，统一改为一个特殊字符 `.`（点号）。
2. **把数组大小从 28×28 改为 27×27**：

   * 26 个字母 + 1 个点号，总共 27 个字符。

---

### ✅ 额外美化处理：

我们还希望这个特殊符号 `.` 排在字符表的最前面，对应索引 `0`，
然后 `'a'` 从索引 `1` 开始，`'b'` 是 `2`，以此类推。

因此：

```python
s2i = {
    '.': 0,
    'a': 1,
    'b': 2,
    ...
    'z': 26
}
```

`i2s`（索引转字符）是 `s2i` 的反转字典，所以没问题，它会正确映射回去。

---

### ✅ 更新后的行为：

* 空单词不会出现，所以 `.` → `.` 的 bigram（即 `N[0][0]`）计数为 0。
* 第一行记录了以 `.` 开头的 bigram，表示某字符是否是单词起始字符。
* 第一列记录了以某字符结尾的 bigram（→ `.`），表示这些字符作为单词结尾。
* 中间的区域记录了普通字符之间的 bigram 关系。

这样处理后，数据更紧凑、结构更清晰、可视化也更美观。

---

### ✅ 总结：

我们将：

* 用一个统一的特殊字符 `.` 代替之前的 `S`（start）和 `E`（end）；
* 把统计矩阵从 28×28 缩小为 27×27；
* 并调整字符索引，使 `.` 对应位置 0，其它字符从 1 开始；
* 这样可以减少冗余、节省空间、提高逻辑清晰度。

# sampling from the model

the information necessary for us to actually sample from this bigram uh character level language model
and um roughly speaking what we're going to do is we're just going to start following these probabilities and these
counts and we're going to start sampling from the from the model so in the beginning of course
we start with the dot the start token dot so to sample the first character of a
name we're looking at this row here so we see that we have the counts and
those concepts terminally are telling us how often any one of these characters is to start a word
so if we take this n and we grab the first row
we can do that by using just indexing as zero and then using this notation column for
the rest of that row so n zero colon is indexing into the zeroth
row and then it's grabbing all the columns and so this will give us a one-dimensional array
of the first row so zero four four ten you know zero four four ten one three oh
six one five four two etc it's just the first row the shape of this is 27 it's just the row of 27
and the other way that you can do this also is you just you don't need to actually give this you just grab the zeroth row like this
this is equivalent now these are the counts and now what we'd like to do is we'd
like to basically um sample from this since these are the raw counts we actually have to convert this to
probabilities so we create a probability vector
so we'll take n of zero and we'll actually convert this to float
first okay so these integers are converted to float floating point numbers and the reason
we're creating floats is because we're about to normalize these counts so to create a probability distribution
here we want to divide we basically want to do p p p divide p
that sum and now we get a vector of smaller
numbers and these are now probabilities so of course because we divided by the sum the sum of p now is 1.
so this is a nice proper probability distribution it sums to 1 and this is giving us the probability for any single
character to be the first character of a word so now we can try to sample from this
distribution to sample from these distributions we're going to use storch.multinomial which i've pulled up
here so torch.multinomial returns uh
samples from the multinomial probability distribution which is a complicated way of saying you give me probabilities and
i will give you integers which are sampled according to the property distribution
so this is the signature of the method and to make everything deterministic we're going to use a generator object in
pytorch so this makes everything deterministic so when you run this on your computer
you're going to the exact get the exact same results that i'm getting here on my computer so let me show you how this works
here's the deterministic way of creating a torch generator object
seeding it with some number that we can agree on so that seeds a generator gets gives us
an object g and then we can pass that g to a function that creates um
here random numbers twerk.rand creates random numbers three of them
and it's using this generator object to as a source of randomness
so without normalizing it i can just print
this is sort of like numbers between 0 and 1 that are random according to this thing and whenever i run it again
i'm always going to get the same result because i keep using the same generator object which i'm seeing here
and then if i divide to normalize i'm going to get a nice
probability distribution of just three elements and then we can use torsion multinomial
to draw samples from it so this is what that looks like tertiary multinomial we'll take the
torch tensor of probability distributions then we can ask for a number of samples
let's say 20. replacement equals true means that when we draw an element
we will uh we can draw it and then we can put it back into the list of eligible indices to draw again
and we have to specify replacement as true because by default uh for some reason it's false
and i think you know it's just something to be careful with and the generator is passed in here so
we're going to always get deterministic results the same results so if i run these two
we're going to get a bunch of samples from this distribution now you'll notice here that the
probability for the first element in this tensor is 60
so in these 20 samples we'd expect 60 of them to be zero
we'd expect thirty percent of them to be one and because the the element index two
has only ten percent probability very few of these samples should be two and indeed we only have a small number of
twos and we can sample as many as we'd like and the more we sample the more
these numbers should um roughly have the distribution here so we should have lots of zeros
half as many um ones and we should have um three times
as few oh sorry s few ones and three times as few uh
twos so you see that we have very few twos we have some ones and most of them are zero
so that's what torsion multinomial is doing for us here
we are interested in this row we've created this p here
and now we can sample from it so if we use the same seed
and then we sample from this distribution let's just get one sample
then we see that the sample is say 13. so this will be the index
and let's you see how it's a tensor that wraps 13 we again have to use that item
to pop out that integer and now index would be just the number 13.
and of course the um we can do we can map the i2s of ix to figure out
exactly which character we're sampling here we're sampling m so we're saying that the first character
is in our generation and just looking at the road here
m was drawn and you we can see that m actually starts a large number of words uh m
started 2 500 words out of 32 000 words so almost
a bit less than 10 percent of the words start with them so this was actually a fairly likely character to draw
um so that would be the first character of our work and now we can continue to sample more characters because now we
know that m started m is already sampled so now to draw the next character we
will come back here and we will look for the row that starts with m
so you see m and we have a row here so we see that m dot is
516 m a is this many and b is this many etc so these are the counts for the next
row and that's the next character that we are going to now generate so i think we are ready to actually just
write out the loop because i think you're starting to get a sense of how this is going to go the um
we always begin at index 0 because that's the start token
and then while true we're going to grab the row corresponding to index
that we're currently on so that's p so that's n array at ix
converted to float is rp then we normalize
this p to sum to one i accidentally ran the infinite loop we
normalize p to something one then we need this generator object
now we're going to initialize up here and we're going to draw a single sample from this distribution
and then this is going to tell us what index is going to be next
if the index sampled is 0 then that's now the end token
so we will break otherwise we are going to print
s2i of ix i2s
and uh that's pretty much it we're just uh this should work okay more
so that's that's the name that we've sampled we started with m the next step was o then r and then dot
and this dot we it here as well so
let's now do this a few times so let's actually create an
out list here and instead of printing we're going to
append so out that append this character
and then here let's just print it at the end so let's just join up all the outs and we're just going to print more okay
now we're always getting the same result because of the generator so if we want to do this a few times we
can go for i in range 10 we can sample 10 names
and we can just do that 10 times and these are the names that we're getting out
let's do 20.
i'll be honest with you this doesn't look right so i started a few minutes to convince myself that it actually is right
the reason these samples are so terrible is that bigram language model is actually look just like really
terrible we can generate a few more here and you can see that they're kind of like their name like a little bit like
yanu o'reilly etc but they're just like totally messed up um
and i mean the reason that this is so bad like we're generating h as a name but you have to think through
it from the model's eyes it doesn't know that this h is the very first h all it
knows is that h was previously and now how likely is h the last character well
it's somewhat likely and so it just makes it last character it doesn't know that there were other things before it or there
were not other things before it and so that's why it's generating all these like nonsense names
another way to do this is to convince yourself that this is actually doing something reasonable even
though it's so terrible is these little piece here are 27 right
like 27. so how about if we did something like this
instead of p having any structure whatsoever how about if p was just torch dot once
of 27 by default this is a float 32 so this is fine divide 27
so what i'm doing here is this is the uniform distribution which will make everything equally likely
and we can sample from that so let's see if that does any better okay so it's
this is what you have from a model that is completely untrained where everything is equally likely so it's obviously
garbage and then if we have a trained model which is trained on just bi-grams
this is what we get so you can see that it is more name-like it is actually working it's just um
my gram is so terrible and we have to do better now next i would like to fix an inefficiency that we have going on here

# 从模型中进行采样（Sampling from the model）

这一部分讲解了如何**根据 bigram 字符级语言模型生成新单词（比如名字）**，即“从模型中采样”。

---

### 🔹 总体流程

1. 模型的输入是字符对（bigram）统计矩阵 `N`（27×27）
2. 我们从特殊起始符号 `.`（索引 0）开始
3. 每一步根据当前字符对应的行（即当前字符后面可能接什么），**按概率采样下一个字符**
4. 如果采到 `.`，表示结束，采样终止
5. 否则继续采样下一个字符

---

## ✅ 步骤详解

---

### 🔸 第一步：取起始行（即点号开头的频率分布）

```python
N[0]  # 取张量第0行，对应从“起始”字符出发的 bigram 统计
```

得到一个大小为 27 的一维数组，对应于从“.”开始，分别接 a、b、c... 的次数。

---

### 🔸 第二步：将 raw count 转为概率分布

```python
p = N[0].float()      # 将整数张量转为 float
p = p / p.sum()       # 归一化：每个元素除以总和，使其变成合法的概率分布（和为 1）
```

---

### 🔸 第三步：使用 `torch.multinomial` 按概率采样

```python
g = torch.Generator().manual_seed(2147483647)   # 创建随机生成器，设定随机种子（保证结果可复现）
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
```

* `torch.multinomial(p, 1)` 从概率分布 `p` 中采样一个索引
* `.item()` 把结果从 tensor 转为普通整数

---

### 🔸 第四步：循环采样完整单词

我们不断地重复以上过程：

1. 每次根据当前字符的索引 `ix` 选取 `N[ix]` 这一行
2. 归一化为概率分布 `p`
3. 用 `torch.multinomial` 采样下一个字符的索引
4. 如果采出的是索引 0（即点号），表示单词结束，退出循环
5. 否则，将字符加入输出列表，继续循环

---

### 🔸 示例采样代码简化版：

```python
out = []
ix = 0  # 从 . 开始
while True:
    p = N[ix].float()
    p = p / p.sum()
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    if ix == 0:
        break
    out.append(itos[ix])  # 将索引转回字符
print(''.join(out))
```

这样可以生成一个“看起来像名字”的字符串。

---

### 🔸 多次采样多个名字：

```python
for i in range(20):
    out = []
    ix = 0
    while True:
        ...
    print(''.join(out))
```

---

## 🧠 模型的实际效果

你可能注意到，生成的名字看起来很奇怪（例如 "yanu", "chm", "eet"）。原因是：

* **Bigram 模型本质上很弱**：它只根据前一个字符预测下一个字符，无法理解整体结构。
* 它不理解“单词长度”、“发音结构”、“元音辅音交替”等规律。
* 它不知道自己生成的是名字，也无法考虑“历史上下文”。

---

## ✅ 与其他情况对比：

### 📉 使用随机均匀分布（完全未训练的模型）：

```python
p = torch.ones(27) / 27
```

每个字符都有相同概率，会生成完全没有规律的乱码。

### 📈 使用训练过的 bigram 模型：

虽然弱，但结果看起来更像“名字”，比如：

* "mory"
* "jorlen"
* "avie"
* "ronna"

说明模型确实学习到了某些字符之间的组合模式，虽然还远不够好。

---

## ✅ 总结

| 步骤         | 内容                                |
| ---------- | --------------------------------- |
| 1. 初始化     | 从特殊字符 `.` 开始（索引 0）                |
| 2. 获取概率    | 从 bigram 计数矩阵中取出对应行，并归一化为概率       |
| 3. 采样下一个字符 | 使用 `torch.multinomial()` 从概率分布中采样 |
| 4. 判断是否结束  | 如果采样结果是 `.`，就终止，否则继续              |
| 5. 重复      | 不断构建字符，直到遇到 `.`                   |

---

## 🎯 关键结论

* Bigram 模型能学到一些字符组合模式，但非常有限，效果不理想；
* 这是构建语言模型的**第一步**，后续我们可以用 MLP、RNN、Transformer 来增强；
* 当前的采样逻辑对后续构建更复杂模型仍然适用，框架已经搭好了。

---

`torch.multinomial` 是 PyTorch 中的一个函数，用于**从给定的概率分布中采样**整数索引（即选择下一个动作、字符、token 等），非常适用于语言模型中的生成任务。

---

## ✅ 函数原型：

```python
torch.multinomial(input, num_samples, replacement=False, *, generator=None)
```

---

## ✅ 参数解释：

| 参数            | 含义                        |
| ------------- | ------------------------- |
| `input`       | 一维或二维张量，表示概率或权重（非负数）      |
| `num_samples` | 要采样多少个结果（整数）              |
| `replacement` | 是否有放回采样（True 表示可以重复抽中同一个） |
| `generator`   | 可选，用于控制随机种子的生成器（保证结果可复现）  |

---

## ✅ 返回值：

返回一个张量，表示采样出的索引（索引位置，而不是概率本身）。

---

## ✅ 示例 1：从概率分布中采样

```python
import torch

p = torch.tensor([0.6, 0.3, 0.1])  # 概率分布
sample = torch.multinomial(p, num_samples=1)
print(sample)  # 可能是 tensor([0])
```

表示：60% 概率采样出索引 0，30% 是索引 1，10% 是索引 2。

---

## ✅ 示例 2：多次采样 + 有放回

```python
torch.multinomial(p, num_samples=10, replacement=True)
```

* 表示从 `p` 中采样 10 次，允许重复。
* 这类似于根据概率分布“投 10 次骰子”。

---

## ✅ 示例 3：设定随机种子（保证可复现）

```python
g = torch.Generator().manual_seed(42)
sample = torch.multinomial(p, 1, generator=g)
```

使用固定的随机数生成器 `g`，确保你和别人跑出相同的结果（在教学、调试中很有用）。

---

## ❗ 注意事项：

* `input` 不需要是正规化的概率分布（即总和不必是 1），但它不能包含负数。
* 如果是未归一化的“权重”，函数会自动在内部转换为概率。
* `replacement=False` 时不能采样数量超过非零元素数量。

---

## ✅ 应用场景（语言模型中）：

在字符级语言模型中，我们常常用：

```python
p = next_char_probs  # 一个 1D 的概率分布张量，长度为字符表大小（如 27）
ix = torch.multinomial(p, num_samples=1).item()
```

这一步的意思是：**根据模型当前预测出的概率，随机“选中”一个下一个字符的索引**，而不是总是选最大概率那个。

---

## ✅ 总结一句话：

> `torch.multinomial` 是用来**从一组概率或权重中按概率随机抽样索引**的工具，广泛用于语言模型、强化学习等任务中的“决策采样”操作。

---

# efficiency! vectorized normalization of the rows, tensor broadcasting

because what we're doing here is we're always fetching a row of n from the counts matrix up ahead
and then we're always doing the same things we're converting to float and we're dividing and we're doing this every single iteration of this loop and
we just keep renormalizing these rows over and over again and it's extremely inefficient and wasteful so what i'd like to do is i'd like to actually
prepare a matrix capital p that will just have the probabilities in it so in
other words it's going to be the same as the capital n matrix here of counts but every single row will have the row of
probabilities uh that is normalized to 1 indicating the probability distribution for the next character given the
character before it um as defined by which row we're in so basically what we'd like to do is
we'd like to just do it up front here and then we would like to just use that row here so here we would like to just
do p equals p of ix instead okay
the other reason i want to do this is not just for efficiency but also i would like us to practice these n-dimensional tensors and i'd like
us to practice their manipulation and especially something that's called broadcasting that we'll go into in a second
we're actually going to have to become very good at these tensor manipulations because if we're going to build out all
the way to transformers we're going to be doing some pretty complicated um array operations for efficiency and we
need to really understand that and be very good at it so intuitively what we want to do is we
first want to grab the floating point copy of n and i'm mimicking the line here
basically and then we want to divide all the rows so that they sum to 1.
so we'd like to do something like this p divide p dot sum but
now we have to be careful because p dot sum actually produces a sum
sorry equals and that float copy p dot sum produces a um
sums up all of the counts of this entire matrix n and gives us a single number of just the summation of everything so
that's not the way we want to define divide we want to simultaneously and in parallel divide all the rows
by their respective sums so what we have to do now is we have to
go into documentation for torch.sum and we can scroll down here to a definition that is relevant to us which
is where we don't only provide an input array that we want to sum but we also provide the dimension along which we
want to sum and in particular we want to sum up over rows
right now one more argument that i want you to pay attention to here is the keep them
is false if keep them is true then the output tensor is of the same size as input
except of course the dimension along which is summed which will become just one but if you pass in keep them as false
then this dimension is squeezed out and so torch.sum not only does the sum and collapses dimension to be of size one
but in addition it does what's called a squeeze where it squeezes out it squeezes out that dimension
so basically what we want here is we instead want to do p dot sum of some axis
and in particular notice that p dot shape is 27 by 27 so when we sum up across axis zero then
we would be taking the zeroth dimension and we would be summing across it so when keep them as true
then this thing will not only give us the counts across um
along the columns but notice that basically the shape of this is 1 by 27 we just get a row vector
and the reason we get a row vector here again is because we passed in zero dimension so this zero dimension becomes
one and we've done a sum and we get a row and so basically we've done the sum
this way vertically and arrived at just a single 1 by 27 vector of counts
what happens when you take out keep them is that we just get 27. so it squeezes
out that dimension and we just get a one-dimensional vector of size 27.
now we don't actually want one by 27 row vector because that gives
us the counts or the sums across the columns
we actually want to sum the other way along dimension one and you'll see that the shape of this is 27 by one so it's a
column vector it's a 27 by one vector of counts
okay and that's because what's happened here is that we're going horizontally and this 27 by 27 matrix becomes a 27 by 1
array now you'll notice by the way that um the actual numbers
of these counts are identical and that's because this special array of counts here comes from bi-gram
statistics and actually it just so happens by chance or because of the way this array is
constructed that the sums along the columns or along the rows horizontally or vertically is identical
but actually what we want to do in this case is we want to sum across the rows
horizontally so what we want here is p that sum of one with keep in true
27 by one column vector and now what we want to do is we want to divide by that
now we have to be careful here again is it possible to take what's a um p dot shape you see here 27
by 27 is it possible to take a 27 by 27 array and divide it by what is a 27 by 1
array is that an operation that you can do and whether or not you can perform this
operation is determined by what's called broadcasting rules so if you just search broadcasting semantics in torch
you'll notice that there's a special definition for what's called broadcasting that uh for whether or not um these two uh arrays
can be combined in a binary operation like division so the first condition is each tensor
has at least one dimension which is the case for us and then when iterating over the dimension sizes starting at the trailing
dimension the dimension sizes must either be equal one of them is one or one of them does not exist
okay so let's do that we need to align the two arrays and their shapes which is
very easy because both of these shapes have two elements so they're aligned then we iterate over from the from the
right and going to the left each dimension must be either equal one
of them is a one or one of them does not exist so in this case they're not equal but one of them is a one so this is fine
and then this dimension they're both equal so uh this is fine so all the dimensions are fine and
therefore the this operation is broadcastable so that means that this operation is allowed
and what is it that these arrays do when you divide 27 by 27 by 27 by one
what it does is that it takes this dimension one and it stretches it out it copies it to match
27 here in this case so in our case it takes this column vector which is 27 by 1
and it copies it 27 times to make these both be 27 by 27 internally you
can think of it that way and so it copies those counts and then it does an element-wise division
which is what we want because these counts we want to divide by them on every single one of these columns in
this matrix so this actually we expect will normalize every single row
and we can check that this is true by taking the first row for example and taking its sum we expect this to be
1. because it's not normalized and then we expect this now because if
we actually correctly normalize all the rows we expect to get the exact same result here so let's run this
it's the exact same result this is correct so now i would like to scare you a little bit
uh you actually have to like i basically encourage you very strongly to read through broadcasting semantics
and i encourage you to treat this with respect and it's not something to play fast and loose with it's something to
really respect really understand and look up maybe some tutorials for broadcasting and practice it and be careful with it because you can very
quickly run into books let me show you what i mean you see how here we have p dot sum of
one keep them as true the shape of this is 27 by one let me take out this line just so we have the n
and then we can see the counts we can see that this is a all the counts across all the
rows and it's a 27 by one column vector right now suppose that i tried to do the
following but i erase keep them just true here what does that do if keep them is not
true it's false then remember according to documentation it gets rid of this dimension one it squeezes it out so
basically we just get all the same counts the same result except the shape of it is not 27 by 1 it is just 27 the
one disappears but all the counts are the same so you'd think that this divide that
would uh would work first of all can we even uh write this and will it is it even is it even
expected to run is it broadcastable let's determine if this result is broadcastable p.summit one is shape
is 27. this is 27 by 27. so 27 by 27
broadcasting into 27. so now rules of broadcasting number one align
all the dimensions on the right done now iteration over all the dimensions starting from the right going to the
left all the dimensions must either be equal one of them must be one or one that does
not exist so here they are all equal here the dimension does not exist so internally what broadcasting will do
is it will create a one here and then we see that one of them is a one and
this will get copied and this will run this will broadcast okay so you'd expect this
to work because we we are
this broadcast and this we can divide this now if i run this you'd expect it to work but
it doesn't uh you actually get garbage you get a wrong dissolve because this is actually a bug
this keep them equals true makes it work
this is a bug in both cases we are doing the correct counts we are summing up
across the rows but keep them is saving us and making it work so in this case
i'd like to encourage you to potentially like pause this video at this point and try to think about why this is buggy and
why the keep dim was necessary here okay so the reason to do
for this is i'm trying to hint it here when i was sort of giving you a bit of a hint on how this works
this 27 vector internally inside the broadcasting this becomes a 1 by 27
and 1 by 27 is a row vector right and now we are dividing 27 by 27 by 1 by
27 and torch will replicate this dimension so basically
uh it will take it will take this row vector and it will copy it
vertically now 27 times so the 27 by 27 lies exactly and element wise divides
and so basically what's happening here is we're actually normalizing the columns
instead of normalizing the rows so you can check that what's happening
here is that p at zero which is the first row of p dot sum is not one it's seven
it is the first column as an example that sums to one
so to summarize where does the issue come from the issue comes from the silent adding of a dimension here because in
broadcasting rules you align on the right and go from right to left and if dimension doesn't exist you create it
so that's where the problem happens we still did the counts correctly we did the counts across the rows and we got
the the counts on the right here as a column vector but because the keep things was true this this uh this
dimension was discarded and now we just have a vector of 27. and because of broadcasting the way it works this
vector of 27 suddenly becomes a row vector and then this row vector gets replicated vertically and that every single point
we are dividing by the by the count in the opposite direction
so uh so this thing just uh doesn't work this needs to be keep things equal true in
this case so then then we have that p at zero is normalized
and conversely the first column you'd expect to potentially not be normalized and this is what makes it work
so pretty subtle and uh hopefully this helps to scare you that you should have
a respect for broadcasting be careful check your work uh and uh understand how it works under the hood and make sure
that it's broadcasting in the direction that you like otherwise you're going to introduce very subtle bugs very hard to
find bugs and uh just be careful one more note on efficiency we don't want to be doing this here because this creates
a completely new tensor that we store into p we prefer to use in place operations if possible
so this would be an in-place operation it has the potential to be faster it doesn't create new memory
under the hood and then let's erase this we don't need it and let's
also um just do fewer just so i'm not wasting space

# 提高效率！用向量化方法归一化 bigram 张量的每一行（Tensor Broadcasting）

---

## 🎯 问题背景

在前面的代码中，我们每次生成下一个字符时：

1. 都要从 bigram 计数矩阵 `N` 中取出一行；
2. 转换成 float；
3. 然后再除以该行元素之和，归一化成概率分布。

这个操作我们每生成一个字符就重复一遍，**非常低效**，因为我们在循环中不停地重复相同的计算。

---

## ✅ 优化目标

提前计算好一个新的矩阵 `P`：

* `P` 与 `N` 的形状一致（27 × 27）
* `P[i]` 是 `N[i]` 的归一化版本，即第 `i` 行被归一化成一个**概率分布**
* 这样，我们以后采样只需要从 `P[ix]` 中取概率分布，无需每次都手动归一化

---

## 🔧 实现步骤

### 1. 转为 float 类型

```python
P = N.float()
```

### 2. 对每一行做归一化（而不是对整个矩阵）

我们需要将 `P` 中的每一行除以这一行的总和。这需要用到 PyTorch 的 **广播机制（broadcasting）**。

#### 计算每一行的总和：

```python
P.sum(1, keepdim=True)
```

解释：

* `dim=1`：在“列”方向求和，相当于横向地求每一行的和；
* `keepdim=True`：保留维度，这样返回的是 `(27, 1)` 的列向量；

  * 这很重要，否则维度不对会导致广播错误！

#### 执行除法：

```python
P = P / P.sum(1, keepdim=True)
```

这样每一行都会被该行的和除，结果仍是 `(27, 27)`，但现在每一行的元素加起来是 1，即为概率分布。

---

## 🧠 关于 Broadcasting（广播机制）

PyTorch 的广播规则允许形状不同的张量进行运算，条件如下：

* 两个张量从最后一个维度开始对齐；
* 每个维度要么相等，要么其中一个是 `1`，或者其中一个不存在；
* 如果维度为 `1`，会自动扩展（复制）以匹配另一个维度；

### ✅ 示例：

```python
P      : shape (27, 27)
row_sums: shape (27, 1)
```

广播过程会自动将 `(27, 1)` **复制为** `(27, 27)`，从而与 `P` 兼容，实现**逐行除法**。

---

## ⚠️ Bug 警告：不要忘记 `keepdim=True`

如果你写成：

```python
P = P / P.sum(1)
```

那 `P.sum(1)` 会返回 `(27,)`，是一个一维向量。

广播时，它会被**当作行向量** `(1, 27)` 复制到 27 行，结果是 **按列归一化** 而不是按行归一化。

这会悄悄地让你得到错误结果，而不报错，**非常隐蔽！**

---

## ✅ 正确 vs 错误对比：

| 写法                           | 作用        | 是否正确 |
| ---------------------------- | --------- | ---- |
| `P / P.sum(1, keepdim=True)` | 按行归一化 ✅   | ✅ 正确 |
| `P / P.sum(1)`               | 实际按列归一化 ❌ | ❌ 错误 |

---

## 🧠 结论：Respect Broadcasting

作者强调了两个重要建议：

1. **不要轻视 broadcasting**：它功能强大，但容易产生隐性 bug；
2. **记得查文档并测试每一步**：看维度、打印形状、核对逻辑；

---

## 🛠 效率建议

* 用 **就地操作（in-place operation）** 比如 `P.div_()` 代替 `P = P / ...` 可减少内存开销，提高速度；
* 尽量把一次性可预计算的东西**提前算好**，避免在循环中重复计算；

---

## ✅ 总结一句话：

> **用 `P = N.float(); P = P / P.sum(1, keepdim=True)` 可一次性高效构建出 bigram 概率矩阵，并确保广播行为正确。牢记 broadcasting 的规则和陷阱，避免隐蔽 bug！**

# loss function (the negative log likelihood of the data under our model)

okay so we're actually in a pretty good spot now we trained a bigram language model and we trained it really just by counting uh
how frequently any pairing occurs and then normalizing so that we get a nice property distribution
so really these elements of this array p are really the parameters of our biogram language model giving us and summarizing
the statistics of these bigrams so we train the model and then we know how to sample from a model we just
iteratively uh sample the next character and feed it in each time and get a next
character now what i'd like to do is i'd like to somehow evaluate the quality of this model we'd like to somehow summarize the
quality of this model into a single number how good is it at predicting the training set
and as an example so in the training set we can evaluate now the training loss
and this training loss is telling us about sort of the quality of this model in a single number just like we saw in
micrograd so let's try to think through the quality of the model and how we would evaluate it
basically what we're going to do is we're going to copy paste this code that we previously used for counting
okay and let me just print these diagrams first we're gonna use f strings and i'm gonna print character one
followed by character two these are the diagrams and then i don't wanna do it for all the words just do the first three words so here we have emma olivia
and ava bigrams now what we'd like to do is we'd like to basically look at the probability that
the model assigns to every one of these diagrams so in other words we can look at the probability which is
summarized in the matrix b of i x 1 x 2 and then we can print it here
as probability and because these properties are way too large let me present
or call in 0.4 f to like truncate it a bit so what do we have here right we're
looking at the probabilities that the model assigns to every one of these bigrams in the dataset and so we can see some of them are four
percent three percent etc just to have a measuring stick in our mind by the way um we have 27 possible
characters or tokens and if everything was equally likely then you'd expect all these probabilities
to be four percent roughly so anything above four percent means
that we've learned something useful from these bigram statistics and you see that roughly some of these are four percent
but some of them are as high as 40 percent 35 percent and so on so you see that the model actually assigned a pretty high
probability to whatever's in the training set and so that's a good thing um basically if you have a very good
model you'd expect that these probabilities should be near one because that means that your model is correctly
predicting what's going to come next especially on the training set where you where you trained your model
so now we'd like to think about how can we summarize these probabilities into a single number that measures the quality
of this model now when you look at the literature into maximum likelihood estimation and
statistical modeling and so on you'll see that what's typically used here is something called the likelihood
and the likelihood is the product of all of these probabilities and so the product of all these
probabilities is the likelihood and it's really telling us about the probability of the entire data set assigned uh
assigned by the model that we've trained and that is a measure of quality so the product of these
should be as high as possible when you are training the model and when you have a good model your pro your
product of these probabilities should be very high um now because the product of these probabilities is an unwieldy thing to
work with you can see that all of them are between zero and one so your product of these probabilities will be a very tiny number
um so for convenience what people work with usually is not the likelihood but they work with what's called the log
likelihood so the product of these is the likelihood to get the log likelihood we just have
to take the log of the probability and so the log of the probability here i have the log of x from zero to one
the log is a you see here monotonic transformation of the probability where if you pass in one
you get zero so probability one gets your log probability of zero and then as you go lower and lower
probability the log will grow more and more negative until all the way to negative infinity at zero
so here we have a log prob which is really just a torch.log of probability let's print it out to get a sense of
what that looks like log prob also 0.4 f
okay so as you can see when we plug in numbers that are very close some of our
higher numbers we get closer and closer to zero and then if we plug in very bad probabilities we get more and more
negative number that's bad so and the reason we work with this is for
a large extent convenience right because we have mathematically that if you have some product a times b times c
of all these probabilities right the likelihood is the product of all these probabilities
then the log of these is just log of a plus
log of b plus log of c if you remember your logs
from your high school or undergrad and so on so we have that basically
the likelihood of the product probabilities the log likelihood is just the sum of the logs of the individual
probabilities so log likelihood
starts at zero and then log likelihood here we can just accumulate simply
and in the end we can print this
print the log likelihood f strings
maybe you're familiar with this so log likelihood is negative 38.
okay now we actually want um
so how high can log likelihood get it can go to zero so when all the
probabilities are one log likelihood will be zero and then when all the probabilities are lower this will grow
more and more negative now we don't actually like this because what we'd like is a loss function and a
loss function has the semantics that low is good because we're trying to minimize the
loss so we actually need to invert this and that's what gives us something called the negative log likelihood
negative log likelihood is just negative of the log likelihood
these are f strings by the way if you'd like to look this up negative log likelihood equals
so negative log likelihood now is just negative of it and so the negative log block load is a very nice loss function
because um the lowest it can get is zero and the higher it is the worse off the
predictions are that you're making and then one more modification to this that sometimes people do is that for
convenience uh they actually like to normalize by they like to make it an average instead of a sum
and so uh here let's just keep some counts as well so n plus equals one
starts at zero and then here um we can have sort of like a normalized log likelihood
um if we just normalize it by the count then we will sort of get the average
log likelihood so this would be usually our loss function here is what this we would this is what we would use
uh so our loss function for the training set assigned by the model is 2.4 that's the quality of this model
and the lower it is the better off we are and the higher it is the worse off we are and
the job of our you know training is to find the parameters that minimize the
negative log likelihood loss and that would be like a high quality
model okay so to summarize i actually wrote it out here so our goal is to maximize likelihood
which is the product of all the probabilities assigned by the model and we want to maximize this likelihood
with respect to the model parameters and in our case the model parameters here are defined in the table these numbers
the probabilities are the model parameters sort of in our program language models so far but you
have to keep in mind that here we are storing everything in a table format the probabilities but what's coming up as a
brief preview is that these numbers will not be kept explicitly but these numbers will be calculated by a neural network
so that's coming up and we want to change and tune the parameters of these neural networks we
want to change these parameters to maximize the likelihood the product of the probabilities
now maximizing the likelihood is equivalent to maximizing the log likelihood because log is a monotonic
function here's the graph of log and basically all it is doing is it's
just scaling your um you can look at it as just a scaling of the loss function
and so the optimization problem here and here are actually equivalent because
this is just scaling you can look at it that way and so these are two identical optimization problems
um maximizing the log-likelihood is equivalent to minimizing the negative log likelihood and then in practice
people actually minimize the average negative log likelihood to get numbers like 2.4
and then this summarizes the quality of your model and we'd like to minimize it and make it as small as possible
and the lowest it can get is zero and the lower it is the better off your model is because
it's signing it's assigning high probabilities to your data now let's estimate the probability over the entire
training set just to make sure that we get something around 2.4 let's run this over the entire oops
let's take out the print segment as well okay 2.45 or the entire training set
now what i'd like to show you is that you can actually evaluate the probability for any word that you want like for example
if we just test a single word andre and bring back the print statement
then you see that andre is actually kind of like an unlikely word like on average we take
three log probability to represent it and roughly that's because ej apparently is very uncommon as an example

以下是该段英文关于\*\*负对数似然损失（Negative Log Likelihood Loss, NLL）\*\*的完整中文翻译与讲解：

---

### 🎯 概述：我们已经训练了一个 Bigram 语言模型，它通过统计每对字符出现的频率来建立，然后归一化得到一个**概率矩阵** `P`，该矩阵表示每个字符后接另一个字符的概率。

接下来，我们要评估这个模型的好坏。最常见的评估方式是使用**损失函数（loss function）**，特别是**负对数似然损失（NLL）**，它是语言模型中非常核心的一个概念。

---

### 🧠 为什么使用对数似然？

我们希望模型能给训练集中的每个字对（bigram）分配尽可能**高的概率**。一种衡量方法是将所有这些概率**相乘**，也就是计算**整个数据的似然（likelihood）**：

```
Likelihood = P(x₁) * P(x₂ | x₁) * P(x₃ | x₂) * ...
```

这个乘积越大，说明模型对数据拟合得越好。

---

### 🚨 但问题来了：

所有这些概率都是小于 1 的，所以乘起来会变得非常小（接近于 0），难以处理。因此我们采用对数操作：

```
Log-Likelihood = log(P₁) + log(P₂) + log(P₃) + ...
```

* 对数操作将乘法变成加法，数值上更稳定
* 如果某个 bigram 概率高（接近 1），log 值接近 0
* 如果某个概率低（接近 0），log 值是负数，拉低总和

---

### ❗ 损失函数的语义是：**越小越好**。

而对数似然越大越好（越接近 0）。所以我们使用 **负对数似然（Negative Log Likelihood）**：

```python
NLL = - (log(P₁) + log(P₂) + log(P₃) + ...)
```

这样损失就变成了“越小越好”。

---

### 📏 通常我们还会**平均化损失**，以便不同长度的句子/样本能公平比较：

```python
avg_NLL = NLL / 总bigram数
```

---

### ✅ 总结逻辑链：

| 步骤            | 含义               |
| ------------- | ---------------- |
| 模型输出概率矩阵 `P`  | 每个字符跟随另一个字符的概率   |
| 查找训练集中每对字符的概率 | 来自 `P[ix1, ix2]` |
| 取对数 log(P)    | 便于加法操作和稳定计算      |
| 取负数 -log(P)   | 转换为损失（越小越好）      |
| 平均化           | 得到统一标准的 loss 分值  |

---

### 📌 举例：

假设 `emma` 这个词产生了以下 bigram：

* `.e`, `e.m`, `m.m`, `m.a`, `a.`

对于每一对字符，我们查找概率 → 取对数 → 取负数 → 求和 → 平均

最终，我们得到一个平均损失值（如 2.4），说明该模型平均每个字符需要约 2.4 bits 的信息量去编码训练数据。

---

### 💡 延伸：

将来我们会用神经网络（而不是手动统计）来预测 `P`，即每一个 bigram 的概率将通过网络参数计算而来，训练目标仍然是最小化负对数似然。

---

如果你还想进一步可视化这个 loss 是怎么计算出来的，或用代码例子跑一遍，我可以给你写一个简单的实现。需要吗？

# model smoothing with fake counts

now think through this um when i take andre and i append q and i
test the probability of it under q we actually get
infinity and that's because jq has a zero percent probability according to our model so
the log likelihood so the log of zero will be negative infinity we get infinite loss
so this is kind of undesirable right because we plugged in a string that could be like a somewhat reasonable name but basically what this is saying is
that this model is exactly zero percent likely to uh to predict this name
and our loss is infinity on this example and really what the reason for that is
that j is followed by q uh zero times
uh where's q jq is zero and so jq is uh zero percent likely
so it's actually kind of gross and people don't like this too much to fix this there's a very simple fix that people like to do to sort of like smooth
out your model a little bit and it's called model smoothing and roughly what's happening is that we will eight we will add some fake counts
so imagine adding a count of one to everything so we add a count of one
like this and then we recalculate the probabilities and that's model smoothing and you can
add as much as you like you can add five and it will give you a smoother model and the more you add here
the more uniform model you're going to have and the less you add the more peaked model you are going to
have of course so one is like a pretty decent count to add and that will ensure that there will be
no zeros in our probability matrix p and so this will of course change the generations a little bit in this case it
didn't but in principle it could but what that's going to do now is that nothing will be infinity unlikely
so now our model will predict some other probability and we see that jq now has a very small probability so the model
still finds it very surprising that this was a word or a bigram but we don't get negative infinity so it's kind of like a
nice fix that people like to apply sometimes and it's called model smoothing okay so we've now trained a respectable bi-gram character level

以下是这段关于“**模型平滑（Model Smoothing）**”的完整中文翻译和讲解：

---

### 🧪 【背景问题：模型对未见过的 bigram 给出零概率】

作者举了一个例子：

> 我们把 "andre" 这个名字后面接一个 "q" 字母，然后计算它的概率，结果损失变成了 **无穷大（infinity）**。

为什么？

* 因为 bigram `"jq"` 在训练集中**从未出现过**。
* 所以模型中 `P[j][q] = 0`，即这个组合的概率为 0。
* 而 `log(0)` 在数学上是负无穷（`-∞`），
* 所以最终计算的 **log likelihood = -∞**，也就是损失 = ∞。

---

### 😬 问题分析

这意味着：

* 模型认为 "jq" 这个组合**绝对不可能**出现（概率为 0），
* 导致任何包含这个组合的词都会被模型视为“完全错误”——即损失为无穷大。

但这在实际中很**不合理**，因为训练数据总是有限的，**没出现过 ≠ 完全不可能**。

---

### ✅ 解决方案：**模型平滑（Model Smoothing）**

解决方式就是：**给所有组合都加上一个虚拟的“伪计数（fake count）”**。

#### 🔧 操作方法：

原来我们统计 bigram 的频数是：

```python
N[i][j] += 1   # 真实出现过就+1
```

改为：

```python
N[i][j] += 1   # 真实数据
N += 1         # 给每一个位置都加上一个 "1" 的伪计数
```

> 这相当于我们假设每个字符组合**都出现过一次**。

然后再归一化为概率（除以行和）重新得到 `P`。

---

### 🌊 平滑程度可调

你可以加的不是 1，而是任意一个数（例如 0.1, 5, 10）：

* **加得越多**：模型越“平滑”，越接近于均匀分布（越不自信）
* **加得越少**：模型越“尖锐”，越贴近训练数据（越自信）

---

### 🧾 效果分析：

* 原来 jq = 0，导致 `log(0)` = -∞，loss 无法使用
* 加上平滑后，jq 的概率不再为 0，虽然很小，但 `log(prob)` 是一个有限的负数
* 最终我们避免了无穷大的损失，让模型更加健壮

---

### 📌 总结一句话：

> **模型平滑就是给所有 bigram 加上一个“伪造的”出现次数，避免出现零概率，从而让模型更健壮、更泛化。**

这种方法也叫做 **Laplace smoothing** 或 **Add-one smoothing**，是语言建模里非常经典的一种技巧。

如需，我也可以给你写一段代码实现这个平滑过程。是否需要？

# PART 2: the neural network approach: intro

language model and we saw that we both sort of trained the model by looking at
the counts of all the bigrams and normalizing the rows to get probability distributions
we saw that we can also then use those parameters of this model to perform
sampling of new words so we sample new names according to those distributions and we also saw that
we can evaluate the quality of this model and the quality of this model is summarized in a single number which is
the negative log likelihood and the lower this number is the better the model is because it is giving high probabilities
to the actual next characters in all the bi-grams in our training set so that's all well and good but we've
arrived at this model explicitly by doing something that felt sensible we were just performing counts and then we
were normalizing those counts now what i would like to do is i would like to take an alternative approach we
will end up in a very very similar position but the approach will look very different because i would like to cast
the problem of bi-gram character level language modeling into the neural network framework in the neural network framework we're
going to approach things slightly differently but again end up in a very similar spot i'll go into that later now
our neural network is going to be a still a background character level language model so it receives a single
character as an input then there's neural network with some weights or some parameters w
and it's going to output the probability distribution over the next character in a sequence it's going to make guesses as
to what is likely to follow this character that was input to the model
and then in addition to that we're going to be able to evaluate any setting of the parameters of the neural net because
we have the loss function the negative log likelihood so we're going to take a look at its probability distributions and we're going to use the
labels which are basically just the identity of the next character in that diagram the
second character so knowing what second character actually comes next in the bigram allows
us to then look at what how high of probability the model assigns to that character
and then we of course want the probability to be very high and that is another way of saying that the loss is low
so we're going to use gradient-based optimization then to tune the parameters of this network because we have the loss
function and we're going to minimize it so we're going to tune the weights so that the neural net is correctly
predicting the probabilities for the next character so let's get started the first thing i want to do is i want to compile the

以下是这段关于 **神经网络方法实现 bigram 语言模型（第二部分）** 的中文翻译和讲解：

---

## 🔢 第二部分：神经网络方法简介

我们之前已经用统计方法做了语言模型：

* 通过统计 **所有 bigram（字对）** 的出现次数；
* 然后对每一行进行归一化，得到每个字符后面接哪个字符的**概率分布**。

我们也做了三件事：

1. **训练模型**：通过计数 & 归一化；
2. **采样生成新词**：根据 bigram 概率生成名字；
3. **评估模型质量**：用“**负对数似然（Negative Log-Likelihood, NLL）**”来衡量模型性能 —— 越低越好，表示模型越擅长预测真实训练集中字符的组合。

---

## 🤖 现在，我们将采用一个**完全不同的方式** —— 用神经网络来做！

虽然最终得到的东西会很相似，但**实现方法完全不同**：

### 🎯 新目标：把 bigram 字符级语言模型 **转化为神经网络任务**。

* 输入：一个字符（比如 `'a'`）
* 网络：有一套可学习参数 `W`
* 输出：**下一个字符的概率分布**

也就是说，它会根据输入字符，**预测下一个字符是哪个的概率**。

---

## 🧠 训练方法：

1. **输入一个字符**；
2. 神经网络给出一个“下一个字符”的概率分布；
3. 我们知道训练集中真实的下一个字符是什么（也就是 bigram 的第二个字符）；
4. 看看网络对真实字符预测的概率是多少：

   * **越高越好**；
   * **概率越低，损失越大（Negative Log Likelihood 越大）**；
5. 所以我们就有了一个明确的“目标函数”：负对数似然（NLL）；
6. 我们就可以使用\*\*基于梯度的优化（反向传播 + 梯度下降）\*\*来更新网络参数，使得模型更擅长预测下一个字符。

---

## 🔁 总结一下流程：

| 步骤 | 内容                         |
| -- | -------------------------- |
| 输入 | 单个字符（例如 `'a'`）             |
| 输出 | 所有可能的“下一个字符”的概率分布（27个）     |
| 真值 | 训练集里真实的下一个字符（比如 `'b'`）     |
| 损失 | 网络预测 `'b'` 的概率越高，损失越小；反之越大 |
| 优化 | 用梯度下降法来最小化这个损失（NLL）        |

---

下一步，作者将开始实现神经网络模型的代码。

如果你需要，我可以为你详细写出从 bigram 到神经网络模型的 PyTorch 实现步骤。是否需要继续？

# creating the bigram dataset for the neural net

training set of this neural network right so create the training set
of all the bigrams okay and
here i'm going to copy paste this code because this code iterates over all the
programs so here we start with the words we iterate over all the bygrams and
previously as you recall we did the counts but now we're not going to do counts we're just creating a training
set now this training set will be made up of two lists
we have the inputs and the targets
the the labels and these bi-grams will denote x y those are the characters right
and so we're given the first character of the bi-gram and then we're trying to predict the next one both of these are going to be integers
so here we'll take x's that append is just x1 ystat append ix2
and then here we actually don't want lists of integers we will create tensors out of these so
axis is torch.tensor of axis and wise a storage.tensor of ys
and then we don't actually want to take all the words just yet because i want everything to be manageable
so let's just do the first word which is emma and then it's clear what these x's and
y's would be here let me print character 1 character 2 just so you see
what's going on here so the bigrams of these characters is
dot e e m m m a a dot so this single word as i mentioned has one two three
four five examples for our neural network there are five separate examples in emma
and those examples are summarized here when the input to the neural network is integer 0
the desired label is integer 5 which corresponds to e when the input to the
neural network is 5 we want its weights to be arranged so that 13 gets a very high probability
when 13 is put in we want 13 to have a high probability when 13 is put in we also want 1 to have
a high probability when one is input we want zero to have a very high probability so there are five
separate input examples to a neural nut in this data set
i wanted to add a tangent of a node of caution to be careful with a lot of the apis of some of these frameworks
you saw me silently use torch.tensor with a lowercase t and the output looked right
but you should be aware that there's actually two ways of constructing a tensor there's a torch.lowercase tensor
and there's also a torch.capital tensor class which you can also construct so you can actually call both you can
also do torch.capital tensor and you get a nexus and wise as well
so that's not confusing at all um there are threads on what is the difference between these two
and um unfortunately the docs are just like not clear on the difference and when you look at the the docs of lower case
tensor construct tensor with no autograd history by copying data it's just like it doesn't
it doesn't make sense so the actual difference as far as i can tell is explained eventually in this random thread that you can google
and really it comes down to i believe that um
what is this torch.tensor in first d-type the data type automatically while torch.tensor
just returns a float tensor i would recommend stick to torch.lowercase tensor
so um indeed we see that when i construct this with a capital t the data
type here of xs is float32 but towards that lowercase tensor
you see how it's now x dot d type is now integer
so um it's advised that you use lowercase t and you can read more about it if you
like in some of these threads but basically um i'm pointing out some of these things
because i want to caution you and i want you to re get used to reading a lot of documentation and reading through a lot
of q and a's and threads like this and you know some of the stuff is
unfortunately not easy and not very well documented and you have to be careful out there what we want here is integers
because that's what makes uh sense um and so lowercase tensor is what we are using

以下是这段关于**为神经网络创建 bigram 数据集**的内容翻译和详细解释：

---

## 📊 为神经网络创建 bigram 数据集

我们现在要为神经网络创建训练集，使用的仍然是字符级的 bigram 语言模型。

---

### 🛠 步骤解析

> **目标：** 把所有的 bigram 变成神经网络的训练样本（输入和目标输出）。

---

### 🧩 Bigram 结构举例：

以单词 `"emma"` 为例：

我们在每个词前后加上特殊字符 `'.'` 表示起始和结束：

* 原始词变成：`.emma.`
* 对应的 bigram 是：`.e`、`e.m`、`m.m`、`m.a`、`a.`

---

### 🧮 代码逻辑：

我们遍历这些 bigram，不再**统计出现次数**，而是：

* 把每对字符（x, y）转为对应的整数（索引）
* 存入两个列表：

  * `xs`：输入字符的索引
  * `ys`：目标字符的索引

例如：

```python
xs.append(ix1)  # 输入字符索引
ys.append(ix2)  # 目标字符索引
```

最后，我们将这两个列表转换为 PyTorch 的张量：

```python
xs = torch.tensor(xs)  
ys = torch.tensor(ys)
```

---

### 📦 示例输出（以 `"emma"` 为例）：

| bigram | 输入字符（x）  | 目标字符（y）  |
| ------ | -------- | -------- |
| `.e`   | `.` → 0  | `e` → 5  |
| `e.m`  | `e` → 5  | `m` → 13 |
| `m.m`  | `m` → 13 | `m` → 13 |
| `m.a`  | `m` → 13 | `a` → 1  |
| `a.`   | `a` → 1  | `.` → 0  |

所以这个 word（emma）就贡献了 5 个训练样本。

---

## ⚠️ 小心 Tensor 的构建方式！

作者特别提醒：

* PyTorch 有两种方式可以构建 tensor：

  * `torch.tensor(...)`（小写）✅ 推荐
  * `torch.Tensor(...)`（大写）⚠️ 不推荐

> 两者行为**不同**，尤其是在处理整数时！

* 小写 `.tensor()` 会根据你提供的内容自动推断数据类型（例如整数 -> `int64`）
* 大写 `.Tensor()` 会默认生成 float32 类型（即使你传的是整数！）

例如：

```python
xs = torch.Tensor([1, 2, 3])  # ❌ 得到 float32
xs = torch.tensor([1, 2, 3])  # ✅ 得到 int64
```

---

### ✅ 总结建议：

* 对于分类任务（如字符索引预测），你希望是 **整数类型（int64）**
* 所以应该使用小写的 `torch.tensor(...)`

---

需要我帮你把这个数据处理部分的代码写出来吗？或者你想继续翻译后续内容？

# feeding integers into neural nets? one-hot encodings

okay now we want to think through how we're going to feed in these examples into a neural network
now it's not quite as straightforward as plugging it in because these examples right now are integers so there's like a
0 5 or 13 it gives us the index of the character and you can't just plug an integer index into a neural net
these neural nets right are sort of made up of these neurons and
these neurons have weights and as you saw in micrograd these weights act multiplicatively on the inputs w x plus
b there's 10 h's and so on and so it doesn't really make sense to make an input neuron take on integer values that
you feed in and then multiply on with weights so instead a common way of encoding integers is
what's called one hot encoding in one hot encoding we take an integer like 13 and we create
a vector that is all zeros except for the 13th dimension which we turn to a
one and then that vector can feed into a neural net now conveniently
uh pi torch actually has something called the one hot function inside torching and functional
it takes a tensor made up of integers um long is a is a as an integer
um and it also takes a number of classes um which is how large you want your uh
tensor uh your vector to be so here let's import
torch.n.functional sf this is a common way of importing it and then let's do f.1 hot
and we feed in the integers that we want to encode so we can actually feed in the entire array of x's
and we can tell it that num classes is 27. so it doesn't have to try to guess it it
may have guessed that it's only 13 and would give us an incorrect result
so this is the one hot let's call this x inc for x encoded
and then we see that x encoded that shape is 5 by 27
and uh we can also visualize it plt.i am show of x inc
to make it a little bit more clear because this is a little messy so we see that we've encoded all the five examples uh into vectors we have
five examples so we have five rows and each row here is now an example into a neural nut
and we see that the appropriate bit is turned on as a one and everything else is zero
so um here for example the zeroth bit is turned on the fifth bit is turned on
13th bits are turned on for both of these examples and then the first bit here is turned on
so that's how we can encode integers into vectors and then these
vectors can feed in to neural nets one more issue to be careful with here by the way is
let's look at the data type of encoding we always want to be careful with data types what would you expect x encoding's data
type to be when we're plugging numbers into neural nuts we don't want them to be integers we want them to be floating point
numbers that can take on various values but the d type here is actually 64-bit
integer and the reason for that i suspect is that one hot received a 64-bit integer
here and it returned the same data type and when you look at the signature of one hot it doesn't even take a d type a
desired data type of the output tensor and so we can't in a lot of functions in
torch we'd be able to do something like d type equal storage.float32 which is what we want but one heart does
not support that so instead we're going to want to cast this to float like this
so that these everything is the same everything looks the same but the d-type
is float32 and floats can feed into neural nets so now let's construct our

以下是这段关于 **将整数输入神经网络：One-hot 编码** 的完整翻译与讲解：

---

## 🎯 将整数输入神经网络？使用 One-hot 编码

---

现在我们已经有了训练数据（例如字符索引：0、5、13 等），接下来我们想要把它们输入神经网络中。

但事情**不是那么简单** —— 因为：

### ❌ 问题：整数不能直接作为神经网络输入

神经网络的输入是要和权重进行**乘法计算**的，例如 `w·x + b`。
如果直接输入整数（例如 13），神经元会把它当作是一个实数值来乘，这**没有意义**。

---

### ✅ 解决方案：One-hot 编码

所谓 **One-hot encoding（独热编码）** 是把每个整数转换成一个向量，这个向量除了某一个位置是 1 以外，其余都是 0。

例如：

```text
索引 13 → [0, 0, ..., 0, 1, 0, ..., 0] （只有第13位是1，其他都是0）
```

这种编码方式**不携带任何数值大小信息**，纯粹只是一个“位置”的标记，非常适合用来表达“分类索引”。

---

### 💡 PyTorch 中的 One-hot 编码

PyTorch 提供了内置函数：

```python
torch.nn.functional.one_hot()
```

用法：

```python
import torch.nn.functional as F
F.one_hot(xs, num_classes=27)
```

其中：

* `xs` 是原始整数张量（如 `[0, 5, 13, 13, 1]`）
* `num_classes=27` 表示我们共有 27 个类别（字符）

这将返回一个张量，其形状为 `[样本数, 类别数]`。

---

### 📊 示例结果：

以 `"emma"` 为例：

```python
xs = tensor([0, 5, 13, 13, 1])
F.one_hot(xs, num_classes=27)
```

结果是一个 5×27 的矩阵，每一行对应一个样本，每行只有一个位置为1，其他为0。

---

### 📈 可视化：

可以使用 `matplotlib` 来可视化这个 one-hot 编码：

```python
plt.imshow(x_enc)
```

会看到一个只有少数白点（代表1）的图，其他都是黑色（代表0）。

---

### ⚠️ 小心数据类型！

默认情况下，`F.one_hot()` 会返回 **整数型（int64）** 的张量，但神经网络要求输入是**浮点型（float32）**。

因此你需要手动转换：

```python
x_enc = x_enc.float()  # 转换为 float32 类型
```

这样才能输入到神经网络中。

---

## ✅ 总结：

| 步骤    | 描述                              |
| ----- | ------------------------------- |
| ❌ 不可行 | 直接把整数索引作为输入，不能参与神经网络的计算         |
| ✅ 可行  | 把整数索引转换为 One-hot 向量             |
| 📌 工具 | `torch.nn.functional.one_hot()` |
| 🧽 注意 | 最终结果需 `.float()` 转为浮点数张量        |

---

需要我帮你写一个完整的示例代码来展示从 `xs` 到 `one-hot` 的全过程吗？

# the "neural net": one linear layer of neurons implemented with matrix multiplication

first neuron this neuron will look at these input vectors
and as you remember from micrograd these neurons basically perform a very simple function w x plus b where w x is a dot
product right so we can achieve the same thing here let's first define the weights of this
neuron basically what are the initial weights at initialization for this neuron let's initialize them with torch.rendin
torch.rendin is um fills a tensor with random numbers
drawn from a normal distribution and a normal distribution has a probability density function like this
and so most of the numbers drawn from this distribution will be around 0 but some of them will be as high as
almost three and so on and very few numbers will be above three in magnitude
so we need to take a size as an input here and i'm going to use size as to be 27 by
one so 27 by one and then let's visualize w so
w is a column vector of 27 numbers and
these weights are then multiplied by the inputs so now to perform this multiplication we
can take x encoding and we can multiply it with w this is a matrix multiplication operator
in pi torch and the output of this operation is five by one
the reason is five by five is the following we took x encoding which is five by twenty seven and we multiplied it by
twenty seven by one and in matrix multiplication
you see that the output will become five by one because these 27
will multiply and add so basically what we're seeing here outs
out of this operation is we are seeing the five activations
of this neuron on these five inputs and we've evaluated all of them in
parallel we didn't feed in just a single input to the single neuron we fed in simultaneously all the five inputs into
the same neuron and in parallel patrol has evaluated the wx plus b but here is just the wx
there's no bias it has value w times x for all of them
independently now instead of a single neuron though i would like to have 27 neurons and i'll show you in a second
why i want 27 neurons so instead of having just a 1 here which is indicating this presence of one
single neuron we can use 27 and then when w is 27 by 27
this will in parallel evaluate all the 27 neurons on all the 5 inputs
giving us a much better much much bigger result so now what we've done is 5 by 27 multiplied 27 by 27
and the output of this is now 5 by 27 so we can see that the shape of this
is 5 by 27. so what is every element here telling us right
it's telling us for every one of 27 neurons that we created
what is the firing rate of those neurons on every one of those five examples
so the element for example 3 comma 13
is giving us the firing rate of the 13th neuron looking at the third input
and the way this was achieved is by a dot product between the third
input and the 13th column of this w matrix here
okay so using matrix multiplication we can very efficiently evaluate
the dot product between lots of input examples in a batch and lots of neurons where all those
neurons have weights in the columns of those w's and in matrix multiplication we're just doing those dot products and
in parallel just to show you that this is the case we can take x and we can take the third
row and we can take the w and take its 13th column
and then we can do x and get three elementwise multiply with w at 13.
and sum that up that's wx plus b well there's no plus b it's just wx dot
product and that's this number so you see that this is just being done efficiently by the matrix multiplication
operation for all the input examples and for all the output neurons of this first layer

以下是这段内容的详细中文翻译与解释：

---

## 🧠「神经网络」的第一层：用矩阵乘法实现的线性层（Linear Layer）

---

我们现在要实现一个**线性神经元层**，这个神经元层会接收前面构造的 one-hot 向量作为输入。

---

### 🎯 一个神经元的计算过程回顾：

还记得在 micrograd 中，我们的神经元的计算公式是：

```
output = w · x + b
```

其中 `w·x` 是点积（dot product），然后加上偏置 b。

---

### 🛠 第一步：定义权重 W

我们用 `torch.randn()` 初始化神经元的权重：

```python
W = torch.randn(27, 1)
```

这里是一个 **27×1 的列向量**，表示有一个神经元，它需要乘以一个 27 维的输入向量（因为我们用的是 one-hot 编码，有 27 个可能字符）。

`torch.randn` 生成的是服从**正态分布**（均值为 0，标准差为 1）的随机数，大多数值在 \[-3, 3] 区间内。

---

### 🧮 第二步：进行矩阵乘法

假设我们有五个输入样本（比如 "emma" 的 5 个 bigram），它们的 one-hot 编码是一个形状为：

```python
x_enc.shape = (5, 27)
```

我们进行矩阵乘法：

```python
out = x_enc @ W
```

* `x_enc` 是 5×27（5 个样本，每个样本是一个 27 维向量）
* `W` 是 27×1（一个神经元的权重）
* 结果 `out` 是 5×1（5 个样本分别被神经元处理后的输出）

这表示我们**一次性并行计算了**这个神经元对 5 个样本的响应（firing rate）。

---

### 🎯 拓展：用 27 个神经元代替 1 个

我们不只需要一个神经元，而是要有 **27 个神经元** —— 每个神经元预测一个字符的概率（a-z + .）。

因此我们把 `W` 扩展为：

```python
W = torch.randn(27, 27)
```

现在进行矩阵乘法：

```python
out = x_enc @ W
```

* `x_enc`: 5×27
* `W`: 27×27
* `out`: 5×27

现在的输出 `out[i][j]` 表示：

> 第 `i` 个样本输入，在第 `j` 个神经元上的激活值。

每一行是一个样本，每一列是一个神经元。

---

### 🧪 验证：点积确实是这么来的

例如：

```python
x_enc[3]         # 第4个输入样本
W[:,13]          # 第14个神经元的权重（W 的第13列）
(x_enc[3] * W[:,13]).sum()  # 这就是 dot product（手动计算）
```

这会给出和 `out[3,13]` 相同的结果，验证矩阵乘法的确是高效完成了所有 dot product 的并行计算。

---

### ✅ 总结

| 步骤   | 内容                                    |
| ---- | ------------------------------------- |
| 定义权重 | 使用 `torch.randn(27, N)`，N 是神经元数目      |
| 输入数据 | One-hot 编码，形状为 `[样本数, 27]`            |
| 前向传播 | 使用矩阵乘法 `@` 批量计算所有输入的响应                |
| 输出结果 | 一个 `[样本数, 神经元数]` 的矩阵，每个值表示对应神经元对输入的响应 |

如果你愿意，我可以帮你画一个示意图来说明这些矩阵操作的含义，是否需要？

# transforming neural net outputs into probabilities: the softmax

okay so we fed our 27-dimensional inputs into a first layer of a neural net that
has 27 neurons right so we have 27 inputs and now we have 27 neurons these
neurons perform w times x they don't have a bias and they don't have a non-linearity like 10 h we're going to
leave them to be a linear layer in addition to that we're not going to have any other layers this is going to
be it it's just going to be the dumbest smallest simplest neural net which is just a single linear layer
and now i'd like to explain what i want those 27 outputs to be intuitively what we're trying to produce
here for every single input example is we're trying to produce some kind of a probability distribution for the next
character in a sequence and there's 27 of them but we have to come up with like precise
semantics for exactly how we're going to interpret these 27 numbers that these neurons take on
now intuitively you see here that these numbers are negative and some of them are positive etc
and that's because these are coming out of a neural net layer initialized with these
normal distribution parameters but what we want is we want something like we had here
like each row here told us the counts and then we normalized the counts to get probabilities and we want something
similar to come out of the neural net but what we just have right now is just some negative and positive numbers
now we want those numbers to somehow represent the probabilities for the next character but you see that probabilities they they
have a special structure they um they're positive numbers and they sum to one
and so that doesn't just come out of a neural net and then they can't be counts because these counts are positive and
counts are integers so counts are also not really a good thing to output from a neural net
so instead what the neural net is going to output and how we are going to interpret the um
the 27 numbers is that these 27 numbers are giving us log counts
basically um so instead of giving us counts directly like in this table they're giving us log
counts and to get the counts we're going to take the log counts and we're going to exponentiate them
now exponentiation takes the following form
it takes numbers that are negative or they are positive it takes the entire real line
and then if you plug in negative numbers you're going to get e to the x which is uh always below one
so you're getting numbers lower than one and if you plug in numbers greater than zero you're getting numbers greater than
one all the way growing to the infinity and this here grows to zero
so basically we're going to take these numbers here
and instead of them being positive and negative and all over the place we're
going to interpret them as log counts and then we're going to element wise exponentiate these numbers
exponentiating them now gives us something like this and you see that these numbers now because they went through an exponent
all the negative numbers turned into numbers below 1 like 0.338 and all the
positive numbers originally turned into even more positive numbers sort of greater than one
so like for example seven is some positive number over here
that is greater than zero but exponentiated outputs here
basically give us something that we can use and interpret as the equivalent of counts originally so you see these
counts here 112 7 51 1 etc the neural net is kind of now predicting
uh counts and these counts are positive numbers
they can never be below zero so that makes sense and uh they can now take on various values
depending on the settings of w so let me break this down
we're going to interpret these to be the log counts
in other words for this that is often used is so-called logits these are logits log counts
then these will be sort of the counts largest exponentiated and this is equivalent to the n matrix
sort of the n array that we used previously remember this was the n
this is the the array of counts and each row here are the counts for the
for the um next character sort of so those are the counts and now the
probabilities are just the counts um normalized and so um
i'm not going to find the same but basically i'm not going to scroll all over the place we've already done this we want to
counts that sum along the first dimension and we want to keep them as true
we've went over this and this is how we normalize the rows of our counts matrix
to get our probabilities props so now these are the probabilities
and these are the counts that we ask currently and now when i show the probabilities
you see that um every row here of course
will sum to 1 because they're normalized and the shape of this
is 5 by 27 and so really what we've achieved is for every one of our five examples
we now have a row that came out of a neural net and because of the transformations here
we made sure that this output of this neural net now are probabilities or we can interpret to be probabilities
so our wx here gave us logits and then we interpret those to be log
counts we exponentiate to get something that looks like counts and then we normalize those counts to
get a probability distribution and all of these are differentiable operations so what we've done now is we're taking
inputs we have differentiable operations that we can back propagate through and we're getting out probability
distributions so for example for the zeroth example that
fed in right which was um the zeroth example here was a one-half
vector of zero and um it basically corresponded to feeding in
this example here so we're feeding in a dot into a neural net and the way we fed the dot into a neural net is that we
first got its index then we one hot encoded it then it went into the neural net and out
came this distribution of probabilities and its shape
is 27 there's 27 numbers and we're going to interpret this as the neural nets assignment for how likely every one of
these characters um the 27 characters are to come next
and as we tune the weights w we're going to be of course getting different probabilities out for any
character that you input and so now the question is just can we optimize and find a good w
such that the probabilities coming out are pretty good and the way we measure pretty good is by the loss function okay

以下是你提供内容的完整中文翻译和讲解：

---

## 🔁 将神经网络的输出转换为概率：Softmax 函数

---

我们前面已经做了什么？

我们将 27 维的 one-hot 编码输入到了一个**包含 27 个神经元的神经网络层**中，这些神经元执行的是一个线性变换：
**`output = W·x`**（没有偏置，没有激活函数）。

这个神经网络非常简单，只有一个线性层，是最“笨”的网络之一。

---

### 🧠 我们想让输出代表什么？

我们希望每一个输入样本，最终输出的是一个概率分布（用于预测下一个字符），因为我们想训练的是一个**语言模型**。

* 一共有 27 个可能的字符（a-z 和 `.`），所以每个输出应该是一个长度为 27 的向量。
* 这个向量的每一项表示当前输入字符后面某个字符出现的“概率”。

但问题是：

> 现在神经网络的输出只是一些正数或负数，分布随意。它**不是概率分布**。

---

### ❓如何把这些输出变成“概率”？

我们需要满足两个条件：

1. 所有值都为正数
2. 所有值加起来等于 1

这就是我们需要用 **Softmax** 函数的地方。

---

### 🧮 Softmax 的操作过程：

1. **解释输出为 Log-count（logits）**

   * 输出值不是概率，而是“对数的计数”值（log-count）
   * 神经网络直接输出 logits（通常称为“未归一化得分”）

2. **对这些 log-count 执行 `exp()` 操作**

   * 所有负数会变成 (0,1) 之间的小数
   * 所有正数变成大于 1 的数
   * 这样得到的就是“伪计数”

3. **将这些“计数”归一化（normalize）**

   * 每一行除以其总和，就可以得到一个真正的概率分布

这个过程就是 Softmax：

```python
logits = x_enc @ W           # 线性输出 (logits)
counts = logits.exp()        # 伪计数（全为正）
probs = counts / counts.sum(dim=1, keepdim=True)  # 按行归一化为概率
```

---

### ✅ 得到的结果是什么？

* `probs` 的形状是 `[5, 27]`，表示 5 个样本，每个样本一个概率分布
* 每一行都代表一个样本下所有可能字符的概率和为 1

---

### 🔄 举个例子：

假设输入是字符 `.`，它的 one-hot 是 `[1, 0, 0, ..., 0]`
它经过神经网络后，输出一个 logits 向量，比如：

```
[ 0.1, -1.3, 0.5, ..., -0.7 ]
```

然后我们进行 `exp()` 得到：

```
[1.11, 0.27, 1.65, ..., 0.49]
```

接着再除以总和，得到：

```
[0.05, 0.01, 0.07, ..., 0.02]
```

现在它就是一个概率分布了，每一项表示 `.` 后面接哪个字符的概率。

---

### 🎯 为什么要这么做？

* 输出 logits 是神经网络更容易学习的形式（不用强制满足归一化）
* Softmax 保留了这些 logits 的“相对大小”信息
* 所有步骤是可微的（可以反向传播）

---

### 🔚 总结

| 阶段            | 说明                      |
| ------------- | ----------------------- |
| `W·x`         | 得到 logits，未归一化的分数（可正可负） |
| `exp(logits)` | 转换为正数（伪计数）              |
| `softmax`     | 按行归一化，得到合法的概率分布         |
| 结果            | 每个输入字符，得到 27 个输出字符的概率分布 |

这样，我们就能让神经网络输出一个可以用于采样和计算损失的合法分布了。

---

需要我帮你画出整个流程图或代码示意图吗？这能帮助你更直观理解每一步。

# summary, preview to next steps, reference to micrograd

so i organized everything into a single summary so that hopefully it's a bit more clear so it starts here
with an input data set we have some inputs to the neural net and we have some labels for the correct
next character in a sequence these are integers here i'm using uh torch generators now
so that you see the same numbers that i see and i'm generating um
27 neurons weights and each neuron here receives 27 inputs
then here we're going to plug in all the input examples x's into a neural net so here this is a forward pass
first we have to encode all of the inputs into one hot representations so we have 27 classes we pass in these
integers and x inc becomes a array that is 5 by 27
zeros except for a few ones we then multiply this in the first layer of a neural net to get logits
exponentiate the logits to get fake counts sort of and normalize these counts to get
probabilities so we lock these last two lines by the way here are called the softmax
which i pulled up here soft max is a very often used layer in a neural net
that takes these z's which are logics exponentiates them
and divides and normalizes it's a way of taking outputs of a neural net layer and these
these outputs can be positive or negative and it outputs probability distributions
it outputs something that is always sums to one and are positive numbers just like probabilities
um so it's kind of like a normalization function if you want to think of it that way and you can put it on top of any other linear layer inside a neural net
and it basically makes a neural net output probabilities that's very often used and we used it as well here
so this is the forward pass and that's how we made a neural net output probability now
you'll notice that um all of these this entire forward pass is made up of
differentiable layers everything here we can back propagate through and we saw some of the
back propagation in micrograd this is just multiplication and addition all that's
happening here is just multiply and then add and we know how to backpropagate through them exponentiation we know how to
backpropagate through and then here we are summing and sum is is easily backpropagable as
well and division as well so everything here is differentiable operation
and we can back propagate through now we achieve these probabilities which
are 5 by 27 for every single example we have a vector of probabilities that's into one
and then here i wrote a bunch of stuff to sort of like break down uh the examples
so we have five examples making up emma right and there are five bigrams inside emma
so bigram example a bigram example1 is that e is the beginning character right
after dot and the indexes for these are zero and five so then we feed in a zero
that's the input of the neural net we get probabilities from the neural net that are 27 numbers
and then the label is 5 because e actually comes after dot so that's the label
and then we use this label 5 to index into the probability distribution here
so this index 5 here is 0 1 2 3 4 5. it's this
number here which is here so that's basically the probability
assigned by the neural net to the actual correct character you see that the network currently thinks that this next character that e
following dot is only one percent likely which is of course not very good right because this actually is a training
example and the network thinks this is currently very very unlikely but that's just because we didn't get very lucky in
generating a good setting of w so right now this network things it says unlikely and 0.01 is not a good outcome
so the log likelihood then is very negative and the negative log likelihood is very
positive and so four is a very high negative log likelihood and that means we're going to
have a high loss because what is the loss the loss is just the average negative log likelihood
so the second character is em and you see here that also the network thought that m following e is very
unlikely one percent the for m following m i thought it was
two percent and for a following m it actually thought it was seven percent likely so
just by chance this one actually has a pretty good probability and therefore pretty low negative log likelihood
and finally here it thought this was one percent likely so overall our average negative log
likelihood which is the loss the total loss that summarizes basically the how well this network
currently works at least on this one word not on the full data suggested one word is 3.76 which is actually very
fairly high loss this is not a very good setting of w's now here's what we can do
we're currently getting 3.76 we can actually come here and we can change our w we can resample it so let
me just add one to have a different seed and then we get a different w and then we can rerun this
and with this different c with this different setting of w's we now get 3.37
so this is a much better w right and that and it's better because the probabilities just happen to come out
higher for the for the characters that actually are next and so you can imagine actually just
resampling this you know we can try two so
okay this was not very good let's try one more we can try three
okay this was terrible setting because we have a very high loss so anyway i'm going to erase this
what i'm doing here which is just guess and check of randomly assigning parameters and seeing if the network is good that is uh amateur hour that's not
how you optimize a neural net the way you optimize your neural net is you start with some random guess and we're
going to commit to this one even though it's not very good but now the big deal is we have a loss function
so this loss is made up only of differentiable operations and we can minimize the loss
by tuning ws by computing the gradients of the loss with respect to
these w matrices and so then we can tune w to minimize the loss and find a good setting of w
using gradient based optimization so let's see how that will work now things are actually going to look almost
identical to what we had with micrograd so here i pulled up the lecture from micrograd
the notebook it's from this repository and when i scroll all the way to the end where we left off with micrograd we had
something very very similar we had a number of input examples in this case we had four input examples inside axis
and we had their targets these are targets just like here we have our axes now but
we have five of them and they're now integers instead of vectors but we're going to convert our integers
to vectors except our vectors will be 27 large instead of three large
and then here what we did is first we did a forward pass where we ran a neural net on all of the inputs
to get predictions our neural net at the time this nfx was a multi-layer perceptron
our neural net is going to look different because our neural net is just a single layer
single linear layer followed by a soft max so that's our neural net
and the loss here was the mean squared error so we simply subtracted the prediction from the ground truth and
squared it and summed it all up and that was the loss and loss was the single number that summarized the quality of
the neural net and when loss is low like almost zero that means the neural net is
predicting correctly so we had a single number that uh that summarized the uh the performance of the
neural net and everything here was differentiable and was stored in massive compute graph
and then we iterated over all the parameters we made sure that the gradients are set to zero and we called
lost up backward and lasted backward initiated back propagation at the final output node of
loss right so yeah remember these expressions we had loss all the way at the end we start
back propagation and we went all the way back and we made sure that we populated all the parameters dot grad
so that graph started at zero but back propagation filled it in and then in the update we iterated over
all the parameters and we simply did a parameter update where every single
element of our parameters was nudged in the opposite direction of the gradient
and so we're going to do the exact same thing here so i'm going to pull this up
on the side here so that we have it available and we're
actually going to do the exact same thing so this was the forward pass so where we did this
and probs is our wipe red so now we have to evaluate the loss but we're not using the mean squared error we're using the

这一段内容是一个 **神经网络语言模型**教学的**总结与过渡部分**，用来回顾我们刚刚做了什么，并引出下一步：**使用反向传播（backpropagation）来训练神经网络模型的参数**。以下是详细解释：

---

### 🧩 整体结构和流程回顾：

1. **数据准备（输入和标签）**：

   * 你有一个单词，比如 `emma`。
   * 拆成 bigrams（双字符对）后得到：`.e`, `em`, `mm`, `ma`, `a.`。
   * 每对字符用一个整数表示，输入是前一个字符（如 `e`），目标是后一个字符（如 `m`）。

2. **One-hot 编码**：

   * 输入字符用 one-hot 向量表示，比如字符总数是 27，那么每个字符就变成一个 27维向量，其中只有一个位置是 1。

3. **前向传播（Forward Pass）**：

   * 输入 one-hot 向量乘以权重矩阵 W，得到 27 维的输出（logits）。
   * logits 被解释为“log-count”（对数计数），通过 softmax 转成概率分布。

4. **softmax**：

   * 将神经网络输出转为概率，满足总和为1，数值为正。
   * 实现方式是：对 logits 做 `exp()`，再除以总和进行归一化。

---

### 📉 损失计算：Negative Log Likelihood Loss（负对数似然）

* 模型预测一个概率分布，你有实际标签（正确的下一个字符）。
* 取模型预测中 **该标签对应位置的概率**，然后对其取对数再取负号，就是这个样本的损失：

  $$
  \text{Loss} = -\log P(\text{正确字符})
  $$
* 多个样本取平均，就是总的 loss。

例如：

| 输入  | 正确输出 | 模型预测概率 | 负 log 概率         |
| --- | ---- | ------ | ---------------- |
| `.` | `e`  | 0.01   | -log(0.01) ≈ 4.6 |
| `e` | `m`  | 0.01   | ≈ 4.6            |
| `m` | `m`  | 0.02   | ≈ 3.9            |
| `m` | `a`  | 0.07   | ≈ 2.6            |
| `a` | `.`  | 0.01   | ≈ 4.6            |

平均后得到总 loss。

---

### 🎲 为什么 loss 可能高？

* 网络刚初始化，权重是随机的。
* 对正确字符分配的概率可能非常小 → loss 高。
* 所以刚开始模型“很笨”。

可以手动多试几次随机权重初始化（seed 0, 1, 2, ...），看哪个初始化的 loss 更低（但这是瞎猜）。

---

### ⚠️ 这不是训练，这只是 forward 过程！

真正的训练是：

* 计算 loss
* **反向传播（Backward Pass）**：自动计算损失函数对权重的导数（gradients）
* **参数更新（Parameter Update）**：用梯度下降来更新 W

---

### 🔁 与 micrograd 对比：

`micrograd` 是作者自己写的一个微型深度学习框架，用于教学。之前我们也学过：

* 定义网络
* 前向传播计算 loss
* 反向传播 `.backward()`
* 用 `.grad` 更新参数

现在 PyTorch 也会做类似的事情，只不过用更强大的底层引擎自动处理了微分和图构建。

---

### ✅ 小结：

| 步骤      | 内容                                       |
| ------- | ---------------------------------------- |
| 1. 数据   | 拆成 bigrams，映射为整数索引                       |
| 2. 编码   | 用 one-hot 向量表示                           |
| 3. 前向传播 | `one_hot @ W → logits → softmax → probs` |
| 4. 计算损失 | 使用 NLL Loss                              |
| 5. 未来步骤 | 使用反向传播 + 梯度下降训练模型（调整 W）                  |

---

是否需要我帮你写出 PyTorch 的完整训练代码片段，或是画出这个流程图来辅助理解？

# vectorized loss

negative log likelihood because we are doing classification we're not doing regression as it's called
so here we want to calculate loss now the way we calculate it is it's just
this average negative log likelihood now this probs here
has a shape of 5 by 27 and so to get all the we basically want
to pluck out the probabilities at the correct indices here so in particular because the labels are
stored here in array wise basically what we're after is for the first example we're looking at
probability of five right at index five for the second example
at the the second row or row index one we are interested in the probability assigned to index 13.
at the second example we also have 13. at the third row we want one
and then the last row which is four we want zero so these are the probabilities we're interested in right
and you can see that they're not amazing as we saw above so these are the probabilities we want
but we want like a more efficient way to access these probabilities not just listing them out in a tuple
like this so it turns out that the way to do this in pytorch uh one of the ways at least is we can basically pass in
all of these sorry about that all of these um
integers in the vectors so the these ones you see how they're just 0 1
2 3 4 we can actually create that using mp not mp sorry torch dot range of 5
0 1 2 3 4. so we can index here with torch.range of 5
and here we index with ys and you see that that gives us exactly these numbers
so that plucks out the probabilities of that the neural network assigns to the correct next character
now we take those probabilities and we don't we actually look at the log probability so we want to dot log
and then we want to just average that up so take the mean of all of that
and then it's the negative average log likelihood that is the loss
so the loss here is 3.7 something and you see that this loss 3.76 3.76 is
exactly as we've obtained before but this is a vectorized form of that expression so
we get the same loss and the same loss we can consider service part of this forward pass
and we've achieved here now loss okay so we made our way all the way to loss we've defined the forward pass

这段内容是讲解如何**向量化地计算神经网络的损失函数**（loss），并用 PyTorch 实现一个高效的写法。下面是逐步翻译与解释：

---

### 🧠 原文翻译 + 解释：

> **negative log likelihood** because we are doing **classification**, we're not doing **regression** as it's called
> 因为我们在做**分类任务**，所以使用\*\*负对数似然（Negative Log Likelihood, NLL）\*\*作为损失函数，而不是回归任务里常用的均方误差（MSE）。

---

> so here we want to calculate loss now the way we calculate it is it's just this **average negative log likelihood**
> 现在我们要计算 loss，我们的做法就是取所有样本的 **负对数概率的平均值**。

---

> `probs` here has shape (5, 27)
> 当前变量 `probs` 是一个 5 × 27 的矩阵：

* 每一行对应一个训练样本（比如来自单词 "emma" 中的 bigram）
* 每一列对应某一个可能的输出字符（共 27 个字符）

---

> to get all the probabilities at the correct indices...
> 我们要从 `probs` 里**提取出神经网络对每个样本中“正确字符”的概率**。举例：

* 第一个样本的正确标签是 `5`，我们想取第0行的第5列概率（`probs[0, 5]`）
* 第二个样本标签是 `13`，取 `probs[1, 13]`
  ……

---

> but we want a more efficient way to access these probabilities
> 我们希望用更**向量化的方式**一次性取出所有样本的对应概率。

---

### ✅ PyTorch 实现向量索引：

```python
probs[torch.arange(5), ys]
```

解释：

* `torch.arange(5)` → `[0, 1, 2, 3, 4]` → 样本的行索引
* `ys` 是真实标签（例如 `[5, 13, 13, 1, 0]`）→ 列索引
* `probs[行, 列]` 就可以一次性取出每个样本中，神经网络对正确字符的概率值。

---

### 🧮 然后计算损失：

```python
loss = -torch.log(probs[torch.arange(5), ys]).mean()
```

* `torch.log(...)` → 取对数
* `-log(...)` → 负对数（Negative Log-Likelihood）
* `.mean()` → 所有样本的平均，得到 loss。

---

### 🧾 结果：

> 所以这个 `loss` 是 `3.76`，和之前计算的一模一样，只是写法更高效、紧凑。
> 这部分完成了我们神经网络的 **前向传播（forward pass）**，包括输出预测 → 转为概率 → 计算损失。

---

### 📌 总结：

这段代码完成的事情：

1. 网络输出 logits → softmax → probs（概率）
2. 从 probs 中取出每个样本“正确标签”对应的概率值
3. 对这些概率取 log，取负，求平均 → 得到 loss

这就是分类问题中常见的 **交叉熵损失（Cross Entropy Loss）** 的核心计算逻辑。

---

需要我帮你写出完整的这部分 PyTorch 代码吗？或者可视化这个损失流程图也可以。

# backward and update, in PyTorch

we forwarded the network and the loss now we're ready to do the backward pass so backward pass
we want to first make sure that all the gradients are reset so they're at zero now in pytorch you can set the gradients
to be zero but you can also just set it to none and setting it to none is more efficient and pi torch will interpret
none as like a lack of a gradient and is the same as zeros so this is a way to set to zero the
gradient and now we do lost it backward
before we do lost that backward we need one more thing if you remember from micrograd pytorch actually requires
that we pass in requires grad is true so that when we tell
pythorge that we are interested in calculating gradients for this leaf tensor by default this is false
so let me recalculate with that and then set to none and lost that backward
now something magical happened when lasted backward was run because pytorch just like micrograd when
we did the forward pass here it keeps track of all the operations under the hood it builds a full
computational graph just like the graphs we've produced in micrograd those graphs exist
inside pi torch and so it knows all the dependencies and all the mathematical operations of
everything and when you then calculate the loss we can call a dot backward on it
and that backward then fills in the gradients of all the intermediates
all the way back to w's which are the parameters of our neural net so now we
can do w grad and we see that it has structure there's stuff inside it
and these gradients every single element here so w dot shape is 27 by 27
w grad shape is the same 27 by 27 and every element of w that grad
is telling us the influence of that weight on the loss function
so for example this number all the way here if this element the zero zero element of
w because the gradient is positive is telling us that this has a positive
influence in the loss slightly nudging w slightly taking w 0 0
and adding a small h to it would increase the loss
mildly because this gradient is positive some of these gradients are also negative
so that's telling us about the gradient information and we can use this gradient information to update the weights of
this neural network so let's now do the update it's going to be very similar to what we had in micrograd we need no loop
over all the parameters because we only have one parameter uh tensor and that is w so we simply do w dot data plus equals
uh the we can actually copy this almost exactly negative 0.1 times w dot grad
and that would be the update to the tensor
so that updates the tensor and
because the tensor is updated we would expect that now the loss should decrease so
here if i print loss that item
it was 3.76 right so we've updated the w here so if i
recalculate forward pass loss now should be slightly lower so
3.76 goes to 3.74 and then
we can again set to set grad to none and backward update
and now the parameters changed again so if we recalculate the forward pass we expect a lower loss again 3.72
okay and this is again doing the we're now doing gradient descent
and when we achieve a low loss that will mean that the network is assigning high probabilities to the correctness
characters okay so i rearranged everything and i put it all together from scratch

# putting everything together

so here is where we construct our data set of bigrams you see that we are still iterating only
on the first word emma i'm going to change that in a second i added a number that counts the number of
elements in x's so that we explicitly see that number of examples is five
because currently we're just working with emma and there's five backgrounds there and here i added a loop of exactly what
we had before so we had 10 iterations of grainy descent of forward pass backward pass and an update
and so running these two cells initialization and gradient descent gives us some improvement
on the loss function but now i want to use all the words
and there's not 5 but 228 000 bigrams now however this should require no
modification whatsoever everything should just run because all the code we wrote doesn't care if there's five migrants or 228 000 bigrams and with
everything we should just work so you see that this will just run but now we are optimizing over the
entire training set of all the bigrams and you see now that we are decreasing very slightly so actually we can
probably afford a larger learning rate and probably for even larger learning
rate
even 50 seems to work on this very very simple example right so let me re-initialize and let's run 100
iterations see what happens
okay we seem to be
coming up to some pretty good losses here 2.47 let me run 100 more
what is the number that we expect by the way in the loss we expect to get something around what we had originally
actually so all the way back if you remember in the beginning of this video when we
optimized uh just by counting our loss was roughly 2.47
after we had it smoothing but before smoothing we had roughly 2.45
likelihood sorry loss and so that's actually roughly the vicinity of what we expect to achieve
but before we achieved it by counting and here we are achieving the roughly the same result but with gradient based
optimization so we come to about 2.4 6 2.45 etc
and that makes sense because fundamentally we're not taking any additional information we're still just taking in the previous character and
trying to predict the next one but instead of doing it explicitly by counting and normalizing
we are doing it with gradient-based learning and it just so happens that the explicit approach happens to very well
optimize the loss function without any need for a gradient based optimization because the setup for bigram language
models are is so straightforward that's so simple we can just afford to estimate those probabilities directly and
maintain them in a table but the gradient-based approach is significantly more flexible
so we've actually gained a lot because what we can do now is
we can expand this approach and complexify the neural net so currently we're just taking a single character and
feeding into a neural net and the neural that's extremely simple but we're about to iterate on this substantially we're
going to be taking multiple previous characters and we're going to be feeding feeding them into increasingly more
complex neural nets but fundamentally out the output of the neural net will always just be logics
and those logits will go through the exact same transformation we are going to take them through a soft max
calculate the loss function and the negative log likelihood and do gradient based optimization and so actually
as we complexify the neural nets and work all the way up to transformers none of this will really fundamentally
change none of this will fundamentally change the only thing that will change is the way we do the forward pass where we
take in some previous characters and calculate logits for the next character in the sequence that will become more
complex and uh but we'll use the same machinery to optimize it and um
it's not obvious how we would have extended this bigram approach into the case where there are many more
characters at the input because eventually these tables would get way too large because there's way too many
combinations of what previous characters could be if you only have one previous character
we can just keep everything in a table that counts but if you have the last 10 characters that are input we can't
actually keep everything in the table anymore so this is fundamentally an unscalable approach and the neural network approach is significantly more
scalable and it's something that actually we can improve on over time so that's where we will be digging next i
wanted to point out two more things number one i want you to notice that

# note 1: one-hot encoding really just selects a row of the next Linear layer's weight matrix

this x ink here this is made up of one hot vectors and
then those one hot vectors are multiplied by this w matrix and we think of this as multiple neurons
being forwarded in a fully connected manner but actually what's happening here is that for example
if you have a one hot vector here that has a one at say the fifth dimension
then because of the way the matrix multiplication works multiplying that one-half vector with w
actually ends up plucking out the fifth row of w log logits would become just the fifth
row of w and that's because of the way the matrix multiplication works
um so that's actually what ends up happening so but that's actually exactly what
happened before because remember all the way up here we have a bigram we took the first
character and then that first character indexed into a row of this array here
and that row gave us the probability distribution for the next character so the first character was used as a lookup
into a matrix here to get the probability distribution
well that's actually exactly what's happening here because we're taking the index we're encoding it as one hot and
multiplying it by w so logics literally becomes the
the appropriate row of w and that gets just as before exponentiated to create the counts
and then normalized and becomes probability so this w here is literally
the same as this array here but w remember is the log counts not the
counts so it's more precise to say that w exponentiated w dot x is this array
but this array was filled in by counting and by basically
populating the counts of bi-grams whereas in the gradient-based framework we initialize it randomly and then we
let the loss guide us to arrive at the exact same array
so this array exactly here is basically the array w at the end of
optimization except we arrived at it piece by piece by following the loss
and that's why we also obtain the same loss function at the end and the second note is if i come here

# note 2: model smoothing as regularization loss

remember the smoothing where we added fake counts to our counts in order to
smooth out and make more uniform the distributions of these probabilities and that prevented us from assigning
zero probability to to any one bigram now if i increase the count here
what's happening to the probability as i increase the count probability
becomes more and more uniform right because these counts go only up to
like 900 or whatever so if i'm adding plus a million to every single number here you can see how
the row and its probability then when we divide is just going to become more and more close to exactly even probability
uniform distribution it turns out that the gradient based framework has an equivalent to smoothing
in particular think through these w's here
which we initialized randomly we could also think about initializing w's to be zero
if all the entries of w are zero then you'll see that logits will become
all zero and then exponentiating those logics becomes all one and then the probabilities turned out to
be exactly uniform so basically when w's are all equal to each other or say especially zero
then the probabilities come out completely uniform so trying to incentivize w to be near zero
is basically equivalent to label smoothing and the more you incentivize that in the loss function
the more smooth distribution you're going to achieve so this brings us to something that's called
regularization where we can actually augment the loss function to have a small component that we call a
regularization loss in particular what we're going to do is we can take w and we can for example
square all of its entries and then we can um whoops
sorry about that we can take all the entries of w and we can sum them
and because we're squaring uh there will be no signs anymore um negatives and positives all get squashed
to be positive numbers and then the way this works is you achieve zero loss if w is exactly or
zero but if w has non-zero numbers you accumulate loss and so we can actually take this and we
can add it on here so we can do something like loss plus
w square dot sum or let's actually instead of sum let's take a mean because otherwise the sum
gets too large so mean is like a little bit more manageable
and then we have a regularization loss here say 0.01 times or something like that you can choose
the regularization strength and then we can just optimize this and
now this optimization actually has two components not only is it trying to make all the probabilities work out but in
addition to that there's an additional component that simultaneously tries to make all w's be zero because if w's are
non-zero you feel a loss and so minimizing this the only way to achieve that is for w to be zero
and so you can think of this as adding like a spring force or like a gravity force that that pushes w to be zero so w
wants to be zero and the probabilities want to be uniform but they also simultaneously want to match up your
your probabilities as indicated by the data and so the strength of this regularization is exactly controlling
the amount of counts that you add here adding a lot more counts
here corresponds to increasing this number
because the more you increase it the more this part of the loss function dominates this part and the more these
these weights will un be unable to grow because as they grow they uh accumulate way too much loss
and so if this is strong enough then we are not able to overcome the
force of this loss and we will never and basically everything will be uniform predictions
so i thought that's kind of cool okay and lastly before we wrap up i wanted to show you how you would sample from this neural net model

# sampling from the neural net

and i copy-pasted the sampling code from before where remember that we sampled five
times and all we did we start at zero we grabbed the current ix row of p and that
was our probability row from which we sampled the next index and just accumulated that and
break when zero and running this gave us these
results still have the p in memory so this is fine now
the speed doesn't come from the row of b instead it comes from this neural net
first we take ix and we encode it into a one hot row of x
inc this x inc multiplies rw which really just plucks out the row of
w corresponding to ix really that's what's happening and that gets our logits and then we
normalize those low jets exponentiate to get counts and then normalize to get uh the distribution and
then we can sample from the distribution so if i run this
kind of anticlimactic or climatic depending how you look at it but we get the exact same result
um and that's because this is in the identical model not only does it achieve the same loss
but as i mentioned these are identical models and this w is the log counts of
what we've estimated before but we came to this answer in a very different way and it's got a very different
interpretation but fundamentally this is basically the same model and gives the same samples here and so
that's kind of cool okay so we've actually covered a lot of ground we introduced the bigram character level

# conclusion
language model we saw how we can train the model how we can sample from the model and how we can
evaluate the quality of the model using the negative log likelihood loss and then we actually trained the model
in two completely different ways that actually get the same result and the same model in the first way we just counted up the
frequency of all the bigrams and normalized in a second way we used the
negative log likelihood loss as a guide to optimizing the counts matrix
or the counts array so that the loss is minimized in the in a gradient-based framework and we saw that both of them
give the same result and that's it now the second one of these the
gradient-based framework is much more flexible and right now our neural network is super simple we're taking a
single previous character and we're taking it through a single linear layer to calculate the logits
this is about to complexify so in the follow-up videos we're going to be taking more and more of these characters
and we're going to be feeding them into a neural net but this neural net will still output the exact same thing the neural net will output logits
and these logits will still be normalized in the exact same way and all the loss and everything else and the gradient gradient-based framework
everything stays identical it's just that this neural net will now complexify all the way to transformers
so that's gonna be pretty awesome and i'm looking forward to it for now bye
