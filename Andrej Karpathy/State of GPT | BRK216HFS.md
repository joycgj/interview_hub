Learn about the training pipeline of GPT assistants like ChatGPT, from tokenization to pretraining, supervised finetuning, and Reinforcement Learning from Human Feedback (RLHF). Dive deeper into practical techniques and mental models for the effective use of these models, including prompting strategies, finetuning, the rapidly growing ecosystem of tools, and their future extensions.

Speakers:
Andrej Karpathy


Session Information:
This video is one of many sessions delivered for the Microsoft Build 2023 event. View the full session schedule and learn more about Microsoft Build at https://build.microsoft.com 

BRK216HFS | English (US) | AI

这段文字的翻译如下：

---

了解像 ChatGPT 这样的 GPT 助手的训练管道，从分词到预训练、监督微调，再到人类反馈强化学习（RLHF）。深入探讨有效使用这些模型的实用技巧和思维模型，包括提示策略、微调、快速增长的工具生态系统及其未来扩展。

演讲者：
Andrej Karpathy

会议信息：
该视频是 Microsoft Build 2023 活动中的众多会议之一。查看完整的会议日程并了解更多关于 Microsoft Build 的信息，请访问 [https://build.microsoft.com](https://build.microsoft.com)

BRK216HFS | 英语（美国）| 人工智能

---

这段文字描述了 Andrej Karpathy 在 Microsoft Build 2023 活动上关于 GPT 模型训练和应用的演讲内容。


# Intro

[MUSIC]
ANNOUNCER: Please welcome AI researcher and founding member of OpenAI, Andrej Karpathy.
ANDREJ KARPATHY: Hi, everyone. I'm happy to be here to tell you about the state of GPT and more generally about
the rapidly growing ecosystem of large language models. I would like to partition the talk into two parts.
In the first part, I would like to tell you about how we train GPT Assistance, and then in the second part,
we're going to take a look at how we can use these assistants effectively for your applications.
First, let's take a look at the emerging recipe for how to train these assistants and keep in mind that this is all very new and still rapidly evolving,
but so far, the recipe looks something like this. Now, this is a complicated slide, I'm going to go through it piece by

这段话的中文解释如下：

---

**\[音乐]**
**播音员**：请欢迎 AI 研究员、OpenAI 创始成员 Andrej Karpathy。
**Andrej Karpathy**：大家好，很高兴在这里向大家介绍 GPT 的现状，以及更广泛的大型语言模型生态系统的快速发展。我想把今天的讲座分成两部分。
第一部分，我将向大家介绍我们是如何训练 GPT 助手的；然后在第二部分，我们将探讨如何有效地利用这些助手来为你们的应用提供支持。
首先，让我们来看一下训练这些助手的新兴方法，大家要记住，这一切都很新，且仍在快速发展中，但到目前为止，训练的“食谱”大致如下。现在，这是一张比较复杂的幻灯片，我会一一讲解。

---

这段话是 Andrej Karpathy 在介绍 GPT 助手训练的过程时的开场白，强调了训练方法的复杂性和这一领域的快速发展。他提到将分为两部分，第一部分讲解训练过程，第二部分讲解如何有效应用这些助手。


# GPT Assistant training pipeline

piece, but roughly speaking, we have four major stages, pretraining,
supervised finetuning, reward modeling, reinforcement learning, and they follow each other serially.
Now, in each stage, we have a dataset that powers that stage. We have an algorithm that for our purposes will be
a objective and over for training the neural network, and then we have a resulting model,
and then there are some notes on the bottom. The first stage we're going to start with as the pretraining stage. Now, this stage is special in this diagram,
and this diagram is not to scale because this stage is where all of the computational work basically happens. This is 99 percent of the training
compute time and also flops. This is where we are dealing with
Internet scale datasets with thousands of GPUs in the supercomputer and also months of training potentially.
The other three stages are finetuning stages that are much more along the lines of small few number of GPUs and hours or days.
Let's take a look at the pretraining stage to achieve a base model. First, we are going to gather a large amount of data.

这段话的中文解释如下：

---

**GPT 助手训练管道**

大致来说，我们有四个主要阶段：预训练、监督微调、奖励建模、强化学习，它们是顺序进行的。
在每个阶段，我们都有一个数据集，支持该阶段的训练；我们有一个算法，目的就是用来训练神经网络；然后会得到一个训练后的模型，最后是一些备注。
我们首先从预训练阶段开始。
在这个图中，预训练阶段特别重要，图并不是按比例显示的，因为这个阶段是所有计算工作的核心。可以说，这个阶段占据了99%的训练计算时间和浮点运算量（FLOPS）。这是我们处理互联网级的数据集，需要数千个 GPU，在超级计算机上进行，训练可能会持续数个月。
而其他三个阶段则是微调阶段，通常只需要少量的 GPU，并且训练时间是几个小时或几天。
接下来，我们来看一下预训练阶段是如何实现基础模型的。首先，我们需要收集大量的数据。

---

这段话描述了 GPT 助手训练的四个主要阶段，其中预训练阶段占据了大部分的计算资源和时间。它提到，预训练阶段的工作量非常庞大，需要使用大规模的数据集和超强的计算能力。而其他阶段的微调则相对较轻，通常只需要少量资源。


# Data collection

Here's an example of what we call a data mixture that comes from this paper that was released by
Meta where they released this LLaMA based model. Now, you can see roughly the datasets that
enter into these collections. We have CommonCrawl, which is a web scrape, C4, which is also CommonCrawl,
and then some high quality datasets as well. For example, GitHub, Wikipedia, Books, Archives, Stock Exchange and so on.
These are all mixed up together, and then they are sampled according to some given proportions,
and that forms the training set for the GPT. Now before we can actually train on this data,
we need to go through one more preprocessing step, and that is tokenization. This is basically a translation of
the raw text that we scrape from the Internet into sequences of integers because
that's the native representation over which GPTs function. Now, this is a lossless translation
between pieces of texts and tokens and integers, and there are a number of algorithms for the stage.
Typically, for example, you could use something like byte pair encoding, which iteratively merges text chunks
and groups them into tokens. Here, I'm showing some example chunks of these tokens,
and then this is the raw integer sequence that will actually feed into a transformer. Now, here I'm showing

这段话的中文解释如下：

---

**数据收集**

这里是一个数据混合的例子，来自 Meta 发布的一篇论文，他们发布了基于 LLaMA 的模型。你可以看到，这些数据集大致包含了哪些内容。我们有 **CommonCrawl**（一个网络抓取数据集），**C4**（也是来自 CommonCrawl 的数据），还有一些高质量的数据集。例如，**GitHub**、**Wikipedia**、**Books**、**Archives**、**Stock Exchange** 等等。这些数据集会混合在一起，然后按照给定的比例进行采样，形成 GPT 的训练集。

但在我们实际用这些数据进行训练之前，我们还需要经过一步预处理，那就是 **分词**。分词的基本任务是将我们从互联网上抓取到的原始文本，转化成一系列的整数，因为这才是 GPT 使用的原生表示形式。这是一个无损的转换过程，即文本和分词整数之间的转换是可以精确反向操作的。
为此，存在一些算法。例如，通常我们可以使用 **字节对编码（Byte Pair Encoding, BPE）**，它通过迭代地合并文本块，将它们分组为 tokens（分词）。
这里我展示了一些示例的 token 块，以及对应的原始整数序列，这些整数序列会被输入到 Transformer 模型中。

---

这段话讲解了 GPT 训练前的数据收集和预处理过程，重点说明了数据的来源和如何通过分词将文本转化为模型可以处理的整数序列。


# 2 example models

two examples for hybrid parameters that govern this stage.
GPT-4, we did not release too much information about how it was trained and so on, I'm using GPT-3s numbers,
but GPT-3 is of course a little bit old by now, about three years ago. But LLaMA is a fairly recent model from Meta.
These are roughly the orders of magnitude that we're dealing with when we're doing pretraining. The vocabulary size is usually a couple 10,000 tokens.
The context length is usually something like 2,000, 4,000, or nowadays even 100,000,
and this governs the maximum number of integers that the GPT will look at when it's trying to
predict the next integer in a sequence. You can see that roughly the number of parameters say,
65 billion for LLaMA. Now, even though LLaMA has only 65B parameters compared to GPP-3s 175 billion parameters,
LLaMA is a significantly more powerful model, and intuitively, that's because the model is trained for significantly longer.
In this case, 1.4 trillion tokens, instead of 300 billion tokens. You shouldn't judge the power of a model by
the number of parameters that it contains. Below, I'm showing some tables of rough hyperparameters that typically
go into specifying the transformer neural network, the number of heads, the dimension size, number of layers,
and so on, and on the bottom I'm showing some training hyperparameters. For example, to train the 65B model,
Meta used 2,000 GPUs, roughly 21 days of training and a roughly several million dollars.
That's the rough orders of magnitude that you should have in mind for the pre-training stage.
Now, when we're actually pre-training, what happens? Roughly speaking, we are going to take our tokens,
and we're going to lay them out into data batches. We have these arrays that will feed into the transformer,
and these arrays are B, the batch size and these are all independent examples stocked up in rows and B by T,
T being the maximum context length. In my picture I only have 10 the context lengths, so this could be 2,000, 4,000, etc.
These are extremely long rows. What we do is we take these documents, and we pack them into rows,
and we delimit them with these special end of texts tokens, basically telling the transformer where a new document begins.
Here, I have a few examples of documents and then I stretch them out into this input.
Now, we're going to feed all of these numbers into transformer. Let me just focus on a single particular cell,
but the same thing will happen at every cell in this diagram. Let's look at the green cell. The green cell is going to take
a look at all of the tokens before it, so all of the tokens in yellow, and we're going to feed that entire context
into the transforming neural network, and the transformer is going to try to predict the next token in
a sequence, in this case in red. Now the transformer, I don't have too much time to, unfortunately, go into the full details of this
neural network architecture is just a large blob of neural net stuff for our purposes, and it's got several,
10 billion parameters typically or something like that. Of course, as I tune these parameters, you're getting slightly different predicted distributions
for every single one of these cells. For example, if our vocabulary size is 50,257 tokens,
then we're going to have that many numbers because we need to specify a probability distribution for what comes next.
Basically, we have a probability for whatever may follow. Now, in this specific example, for this specific cell,
513 will come next, and so we can use this as a source of supervision to update our transformers weights.
We're applying this basically on every single cell in the parallel, and we keep swapping batches, and we're trying to get the transformer to make
the correct predictions over what token comes next in a sequence. Let me show you more concretely what this looks
like when you train one of these models. This is actually coming from the New York Times, and they trained a small GPT on Shakespeare.
Here's a small snippet of Shakespeare, and they train their GPT on it. Now, in the beginning, at initialization,
the GPT starts with completely random weights. You're getting completely random outputs as well. But over time, as you train the GPT longer and longer,
you are getting more and more coherent and consistent samples from the model,
and the way you sample from it, of course, is you predict what comes next, you sample from that distribution and
you keep feeding that back into the process, and you can basically sample large sequences.
By the end, you see that the transformer has learned about words and where to put spaces and where to put commas and so on.
We're making more and more consistent predictions over time. These are the plots that you are looking at when you're doing model pretraining.
Effectively, we're looking at the loss function over time as you train, and low loss means that our transformer
is giving a higher probability to the next correct integer in the sequence.
What are we going to do with model once we've trained it after a month? Well, the first thing that we noticed, we the field,

这段话的中文解释如下：

---

### 2个示例模型

这里有两个关于影响该阶段训练的混合参数的例子。
GPT-4 的训练细节我们没有发布太多信息，所以我使用的是 GPT-3 的数据，但 GPT-3 当然有些过时了，已经大约三年了。而 LLaMA 则是 Meta 最近发布的一个模型。这些是我们在预训练阶段遇到的大致量级。
词汇表的大小通常在几万个 token（词语/符号）左右。上下文长度通常是 2,000、4,000，或者现在的情况，甚至是 100,000，这决定了 GPT 在预测下一个整数时，最多能看到多少整数（即上下文的最大长度）。
你可以看到，大致来说，LLaMA 的参数量是 65 亿，虽然 LLaMA 只有 65B 参数，而 GPT-3 有 1750 亿参数，但 LLaMA 是一个功能更强大的模型，直观地说，这是因为 LLaMA 的训练时间明显更长。在这种情况下，LLaMA 训练了 1.4 万亿个 token，而 GPT-3 只训练了 3000 亿个 token。
因此，你不应该仅仅通过模型的参数量来评估其性能。
接下来，我展示了一些通常用于定义 Transformer 神经网络的超参数的表格，包括头数、维度大小、层数等，以及一些训练超参数。例如，为了训练这个 65B 的模型，Meta 使用了 2,000 个 GPU，训练了大约 21 天，花费了几百万美元。
这就是你需要了解的预训练阶段的大致量级。

### 预训练过程

那么，预训练过程中发生了什么？
大致来说，我们会将 tokens（词语/符号）分配到数据批次中。我们有这些数组将输入到 Transformer 中，这些数组的大小是 B（批次大小），这些是独立的样本，排成行，大小为 B x T，其中 T 是最大上下文长度。在我的图中，T 只有 10（实际中可能是 2,000、4,000 等等），这些行非常长。我们将文档填充到这些行中，并用特殊的“结束文本” token 来分隔，告诉 Transformer 新的文档开始了。
在这里，我展示了一些文档的例子，然后将它们展开放入输入中。
接着，我们将所有这些数字输入到 Transformer 中。
让我们重点看一个特定的单元格，虽然每个单元格的处理方式都相同。我们来看一下绿色单元格。绿色单元格会查看它前面的所有 tokens（黄色部分），然后我们将整个上下文输入到 Transformer 神经网络中，Transformer 会尝试预测下一个 token，这里是红色的那个 token。
现在，由于没有足够时间详细解释这个神经网络架构，它只是一个很大的神经网络集合，通常拥有数十亿的参数。在调整这些参数时，模型会为每个单元格生成略有不同的预测分布。例如，如果词汇表大小是 50,257 个 token，那么我们就需要生成这么多数字，因为我们需要指定下一个 token 出现的概率分布。
在这个特定的例子中，513 将是下一个 token，因此我们可以将其作为监督信号，来更新 Transformer 的权重。我们会在每个单元格上应用这个过程，并且不断交换批次，努力让 Transformer 做出正确的预测，预测出序列中下一个应该出现的 token。

### 训练过程示例

让我具体展示一下训练模型时的样子。这个例子来自《纽约时报》，他们在训练一个小型的 GPT 模型，专门学习莎士比亚的语言。以下是莎士比亚的一小段文字，他们用它来训练 GPT。在初始化时，GPT 从完全随机的权重开始，它也会给出完全随机的输出。但随着训练的进行，GPT 会变得越来越连贯、一致。
采样的方式是：我们预测下一个可能的 token，然后从中采样，继续将它反馈给模型，你可以持续生成较长的序列。到最后，你会看到 Transformer 学会了单词的用法、如何放置空格、逗号等标点符号，以及更一致的预测。

### 预训练中的损失函数

这些图表展示的是当你在进行模型预训练时的情况。实际上，我们会查看训练过程中的损失函数，损失越低意味着 Transformer 在预测序列中下一个正确整数的概率越高。

---

这段内容主要讲解了预训练阶段的具体过程，从参数设置、数据批次的准备，到通过 Transformer 进行训练和预测的细节。同时也强调了训练过程中如何通过预测下一个 token 来优化模型，并展示了实际训练过程中的效果。


# Base models learn powerful, general representations

is that these models basically in the process of language modeling, learn very powerful general representations,
and it's possible to very efficiently fine tune them for any arbitrary downstream tasks you might be interested in.
As an example, if you're interested in sentiment classification, the approach used to be that you collect a bunch of positives
and negatives and then you train some NLP model for that, but the new approach is:
ignore sentiment classification, go off and do large language model pretraining,
train a large transformer, and then you may only have a few examples and you can very efficiently fine tune
your model for that task. This works very well in practice. The reason for this is that basically
the transformer is forced to multitask a huge amount of tasks in the language modeling task,
because in terms of predicting the next token, it's forced to understand a lot about the structure of the text and all the different concepts therein.
That was GPT-1. Now around the time of GPT-2, people noticed that actually even better than fine tuning,
you can actually prompt these models very effectively. These are language models and they want to complete documents,
you can actually trick them into performing tasks by arranging these fake documents.
In this example, for example, we have some passage and then we like do QA, QA, QA.
This is called Few-shot prompt, and then we do Q, and then as the transformer is tried to complete the document is actually answering our question.
This is an example of prompt engineering based model, making it believe that it's imitating a document and getting it to perform a task.
This kicked off, I think the era of, I would say, prompting over fine tuning and seeing that this
actually can work extremely well on a lot of problems, even without training any neural networks, fine tuning or so on.
Now since then, we've seen an entire evolutionary tree of base models that everyone has trained.
Not all of these models are available. for example, the GPT-4 base model was never released.
The GPT-4 model that you might be interacting with over API is not a base model, it's an assistant model, and we're going to cover how to get those in a bit.
GPT-3 based model is available via the API under the name Devanshi and GPT-2 based model
is available even as weights on our GitHub repo. But currently the best available base model
probably is the LLaMA series from Meta, although it is not commercially licensed.
Now, one thing to point out is base models are not assistants. They don't want to make answers to your questions,
they want to complete documents. If you tell them to write a poem about the bread and cheese,
it will answer questions with more questions, it's completing what it thinks is a document.
However, you can prompt them in a specific way for base models that is more likely to work.
As an example, here's a poem about bread and cheese, and in that case it will autocomplete correctly. You can even trick base models into being assistants.
The way you would do this is you would create a specific few-shot prompt that makes it look like there's some document between the human and assistant
and they're exchanging information. Then at the bottom, you put your query at the end and the base model
will condition itself into being a helpful assistant and answer,
but this is not very reliable and doesn't work super well in practice, although it can be done. Instead, we have a different path to make
actual GPT assistants not base model document completers. That takes us into supervised finetuning.
In the supervised finetuning stage, we are going to collect small but high quality data-sets, and in this case,
we're going to ask human contractors to gather data of the form prompt and ideal response.
We're going to collect lots of these typically tens of thousands or something like that. Then we're going to still do language
modeling on this data. Nothing changed algorithmically, we're swapping out a training set. It used to be Internet documents,
which has a high quantity local for basically Q8 prompt response data.
That is low quantity, high quality. We will still do language modeling and then after training,
we get an SFT model. You can actually deploy these models and they are actual assistants and they work to some extent.
Let me show you what an example demonstration might look like. Here's something that a human contractor might come up with.
Here's some random prompt. Can you write a short introduction about the relevance of the term monopsony or something like that?
Then the contractor also writes out an ideal response. When they write out these responses, they are following extensive labeling
documentations and they are being asked to be helpful, truthful, and harmless.
These labeling instructions here, you probably can't read it, neither can I, but they're long and this is people
following instructions and trying to complete these prompts. That's what the dataset looks like. You can train these models. This works to some extent.
Now, you can actually continue the pipeline from here on, and go into RLHF,
reinforcement learning from human feedback that consists of both reward modeling and reinforcement learning.
Let me cover that and then I'll come back to why you may want to go through the extra steps and how that compares to SFT models.
In the reward modeling step, what we're going to do is we're now going to shift our data collection to be of the form of comparisons.
Here's an example of what our dataset will look like. I have the same identical prompt on the top,

这段话的中文解释如下：

---

### 基础模型学习强大的通用表示

这些模型在语言建模过程中基本上学会了非常强大的通用表示，并且可以非常高效地微调来适应你感兴趣的任何下游任务。
举个例子，如果你对情感分类感兴趣，过去的做法是收集大量的正面和负面样本，然后训练一个 NLP 模型。但现在的方法是：忽略情感分类，直接进行大规模语言模型的预训练，训练一个大型的 Transformer 模型，然后你可能只需要几个示例，就能高效地微调你的模型来完成该任务。这个方法在实践中效果非常好。
原因在于，Transformer 在语言建模任务中被迫进行大量的多任务处理，因为在预测下一个 token 时，它需要理解文本的结构和其中的各种概念。这就是 GPT-1 的情况。
到了 GPT-2 时，人们发现，比微调更有效的方法是使用 **提示（prompt）** 来引导这些模型。这些模型是语言模型，它们希望完成文档，你可以通过组织这些伪文档来让它们执行任务。
例如，给定一段文本，然后我们添加问答（QA）任务，这叫做 **少量示例提示（Few-shot prompt）**，然后我们提出问题，Transformer 会尝试完成文档，实际上就是在回答我们的提问。
这就是提示工程的一个示例，通过让模型认为它在模仿一个文档，从而让它执行任务。我认为这开启了“提示优于微调”的时代，表明这种方法在很多问题上非常有效，甚至不需要训练神经网络、微调等。

### 基础模型与助手模型的区别

从那时起，我们看到一个完整的基础模型演化树，大家都在训练这些模型。并非所有这些模型都是公开可用的。例如，GPT-4 的基础模型从未发布。你通过 API 与 GPT-4 交互的模型不是基础模型，它是一个助手模型，我们稍后会讲解如何获取这些模型。
GPT-3 基础模型可以通过 API 获取，名为 **Devanshi**，而 GPT-2 基础模型则可以在 GitHub 上获取。
目前最好的基础模型可能是 Meta 的 **LLaMA** 系列，尽管它没有商业授权。

### 基础模型的局限性

需要指出的是，基础模型并不是助手模型。它们并不是为了回答你的问题而设计的，而是用来完成文档的。如果你让它写一首关于面包和奶酪的诗，它可能会通过提出更多问题来回答你的问题，因为它认为自己在完成文档。
然而，你可以通过特定方式提示基础模型，让它更有可能按照你的需求工作。例如，下面是关于面包和奶酪的诗，在这种情况下，它会正确地自动补全。如果你想让基础模型成为助手，可以通过创建一个特定的少量示例提示，让它看起来像是人类和助手之间的信息交换。然后在底部放置你的查询，基础模型就会“条件化”自己成为一个有用的助手并回答问题，但这种方法并不可靠，在实践中效果也不好，尽管可以做到。
因此，真正的 GPT 助手并不是基础模型的文档补全者，而是经过监督微调后的模型。

### 监督微调（SFT）

在监督微调阶段，我们将收集小而高质量的数据集。在这种情况下，我们会要求人工承包商收集格式为“提示和理想回答”的数据。我们会收集大量这样的数据，通常是几万个示例。然后，仍然进行语言建模，算法没有变化，只是我们更换了训练集，从原本的大量互联网文档换成了低数量、高质量的 Q\&A 提示-回答数据集。
训练后，我们得到一个 **SFT 模型**，这些模型是真正的助手，能够在一定程度上工作。
让我展示一个示例，看看人工承包商如何做：
假设一个随机的提示是：“你能写一篇关于单音市场（monopsony）相关性的简短介绍吗？”然后，承包商会写出一个理想的回答。写这些回答时，承包商遵循了详细的标注文档，并被要求确保回答有用、真实且无害。
这就是数据集的样子，你可以用它来训练模型。虽然这种方法有效，但仍有一些限制。

### 强化学习与人类反馈（RLHF）

你可以从这里继续流程，进入 **强化学习与人类反馈（RLHF）** 阶段，这个阶段包括奖励建模和强化学习。让我讲解一下奖励建模的步骤，然后再回到为什么你可能希望走这额外的步骤，以及它与 SFT 模型的比较。
在奖励建模步骤中，我们将改变数据收集的形式，改为比较的方式。举个例子，我们的 dataset 会像这样，给出相同的提示，进行多个比较，从而评估模型的表现。

---

这段话讲解了基础模型的功能和限制，并介绍了通过 **监督微调（SFT）** 将基础模型转化为实际助手的过程。还提到了如何通过强化学习与人类反馈（RLHF）进一步优化模型。


# RM Dataset

which is asking the assistant to write a program or a function that checks if a given string is a palindrome.
Then what we do is we take the SFT model which we've already trained and we create multiple completions.
In this case, we have three completions that the model has created, and then we ask people to rank these completions.
If you stare at this for a while, and by the way, these are very difficult things to do to compare some of these predictions.
This can take people even hours for a single prompt completion pairs,
but let's say we decided that one of these is much better than the others and so on. We rank them.
Then we can follow that with something that looks very much like a binary classification on all the possible pairs between these completions.

这段话的中文解释如下：

---

### 奖励建模数据集（RM Dataset）

假设我们要求助手写一个程序或函数，检查给定的字符串是否是回文。然后，我们将已经训练好的 **SFT 模型** 用来生成多个完成的结果。在这个例子中，我们得到了模型生成的三个完成结果，然后我们让人类对这些完成结果进行排名。
如果你仔细看这些结果，实际上这是一项非常困难的任务，需要人类对这些预测进行比较。这种比较可能需要花费几个小时，甚至只对一个提示和完成的结果对进行比较。
假设我们最终决定，其中一个结果明显优于其他结果，然后我们进行排名。接下来，我们可以进行类似 **二元分类** 的操作，比较这些完成结果之间的所有可能对。

---

这段话描述了奖励建模（RM）数据集的生成过程，包括如何使用 SFT 模型生成多个完成结果，并让人类对这些结果进行排名，然后通过比较结果对进行分类来评估模型的表现。


# RM Training

What we do now is, we lay out our prompt in rows, and the prompt is identical across all three rows here.
It's all the same prompt, but the completion of this varies. The yellow tokens are coming from the SFT model.
Then what we do is we append another special reward readout token at the end and we basically only
supervise the transformer at this single green token. The transformer will predict some reward
for how good that completion is for that prompt and basically it makes
a guess about the quality of each completion. Then once it makes a guess for every one of them,
we also have the ground truth which is telling us the ranking of them. We can actually enforce that some of
these numbers should be much higher than others, and so on. We formulate this into a loss function and we train our model to make reward predictions
that are consistent with the ground truth coming from the comparisons from all these contractors. That's how we train our reward model.
That allows us to score how good a completion is for a prompt. Once we have a reward model,
we can't deploy this because this is not very useful as an assistant by itself, but it's very useful for the reinforcement
learning stage that follows now. Because we have a reward model, we can score the quality of any arbitrary completion for any given prompt.
What we do during reinforcement learning is we basically get, again, a large collection of prompts and now we do
reinforcement learning with respect to the reward model. Here's what that looks like. We take a single prompt,
we lay it out in rows, and now we use basically the model we'd like to train which
was initialized at SFT model to create some completions in yellow, and then we append the reward token again
and we read off the reward according to the reward model, which is now kept fixed. It doesn't change any more. Now the reward model
tells us the quality of every single completion for all these prompts and so what we can do is we can now just basically apply the same
language modeling loss function, but we're currently training on the yellow tokens, and we are weighing
the language modeling objective by the rewards indicated by the reward model. As an example, in the first row,
the reward model said that this is a fairly high-scoring completion and so all the tokens that we
happen to sample on the first row are going to get reinforced and they're going to get higher probabilities for the future.
Conversely, on the second row, the reward model really did not like this completion, -1.2. Therefore, every single token that we sampled in
that second row is going to get a slightly higher probability for the future. We do this over and over on many prompts on many batches and basically,
we get a policy that creates yellow tokens here. It's basically all the completions here will
score high according to the reward model that we trained in the previous stage.
That's what the RLHF pipeline is. Then at the end, you get a model that you could deploy.
As an example, ChatGPT is an RLHF model, but some other models that you might come across for example,
Vicuna-13B, and so on, these are SFT models. We have base models, SFT models, and RLHF models.
That's the state of things there. Now why would you want to do RLHF? One answer that's not
that exciting is that it works better. This comes from the instruct GPT paper. According to these experiments a while ago now,
these PPO models are RLHF. We see that they are basically preferred in a lot
of comparisons when we give them to humans. Humans prefer basically tokens
that come from RLHF models compared to SFT models, compared to base model that is prompted to be an assistant. It just works better.
But you might ask why does it work better? I don't think that there's a single amazing answer
that the community has really agreed on, but I will offer one reason potentially.
It has to do with the asymmetry between how easy computationally it is to compare versus generate.
Let's take an example of generating a haiku. Suppose I ask a model to write a haiku about paper clips.
If you're a contractor trying to train data, then imagine being a contractor collecting basically data for the SFT stage,
how are you supposed to create a nice haiku for a paper clip? You might not be very good at that, but if I give you a few examples of
haikus you might be able to appreciate some of these haikus a lot more than others. Judging which one of these is good is a much easier task.
Basically, this asymmetry makes it so that comparisons are a better way to potentially leverage
yourself as a human and your judgment to create a slightly better model. Now, RLHF models are not
strictly an improvement on the base models in some cases. In particular, we'd notice for example that they lose some entropy.
That means that they give more peaky results. They can output samples

这段话的中文解释如下：

---

### 奖励模型训练（RM Training）

现在我们做的事情是，我们将提示（prompt）按行排列，这里所有三行的提示是相同的，但每行的完成结果是不同的。黄色的 token 来自已经训练好的 SFT 模型。然后我们在每个完成结果的末尾附加一个特殊的奖励读取 token，基本上我们只在这个绿色的 token 上对 Transformer 进行监督。Transformer 会预测这个完成结果的奖励，表示它对该提示的完成结果的质量进行评估，基本上就是对每个完成结果做出一个质量评分。
接着，一旦它对每个完成结果做出评估，我们还拥有地面真实数据（ground truth），这告诉我们每个完成结果的排名。我们可以强制某些数字比其他数字要高，从而形成一个损失函数，并训练我们的模型使其做出的奖励预测与这些来自人类承包商的比较排名一致。这就是我们如何训练奖励模型。
这样，我们就能评估一个完成结果对于某个提示的质量。一旦我们得到了奖励模型，虽然它本身并不能作为助手直接部署，但它对于后续的强化学习阶段非常有用。因为我们有了奖励模型，就可以为任何给定的提示评估任何完成结果的质量。

### 强化学习与人类反馈（RLHF）

在强化学习阶段，我们又收集了大量的提示，并用奖励模型进行强化学习。具体步骤如下：我们取一个提示，将其按行排列，然后使用我们希望训练的模型（初始化为 SFT 模型）来生成一些完成结果（黄色部分），接着再次附加奖励 token，并根据固定的奖励模型读取奖励。奖励模型不会再改变。它会告诉我们每个完成结果的质量。
然后我们可以应用和语言建模相同的损失函数，但我们当前训练的是黄色的 token，并且通过奖励模型给出的奖励来加权语言建模目标。
举个例子，在第一行，奖励模型认为这个完成结果的分数较高，因此我们在第一行采样的每个 token 都会被强化，将来会有更高的概率被采样。而在第二行，奖励模型对这个完成结果不太满意，得分为 -1.2。因此，第二行采样的每个 token 将会有稍高的采样概率。我们反复对许多提示和批次进行这个过程，最终得到一个策略，使得在生成完成结果时，所有生成的结果都能根据之前训练的奖励模型得到高分。
这就是 RLHF 管道的工作原理。最终，你得到一个可以部署的模型。例如，**ChatGPT** 就是一个 RLHF 模型，而一些你可能遇到的模型，如 **Vicuna-13B** 等，属于 SFT 模型。我们有基础模型、SFT 模型和 RLHF 模型。
这就是当前的状态。

### 为什么使用 RLHF？

一个不那么令人兴奋的答案是：**它效果更好**。这个结论来自于 **Instruct GPT** 论文。根据早期的实验，**PPO 模型**（RLHF）在很多比较中都表现得更好。当我们将它们交给人类时，人类更倾向于选择 RLHF 模型生成的 tokens，而不是 SFT 模型生成的 tokens，甚至比那些通过提示被引导的基础模型生成的 tokens 要好。它就是更有效。
但你可能会问，为什么它效果更好呢？虽然没有统一的答案，但我可以给出一个可能的解释：这与比较和生成之间的计算难度不对称性有关。
举个例子，假设我让模型写一首关于回形针的俳句。如果你是一个承包商，正在为 SFT 阶段收集数据，如何为回形针写一首优美的俳句呢？你可能不擅长写，但如果我给你几个俳句的例子，你可能会比别人更容易判断哪些俳句比较好。实际上，判断哪个俳句好是一个相对容易的任务。
这种不对称性使得比较成为更好的方法，从而能更好地利用人类的判断力，帮助训练出稍微更好的模型。
然而，RLHF 模型并不总是在某些情况下比基础模型更好。例如，我们注意到 RLHF 模型有时会失去一些 **熵**（entropy），这意味着它们的输出结果更加集中，给出的样本更容易趋向某些固定的输出。

---

这段话介绍了 **奖励建模（RM）** 和 **强化学习与人类反馈（RLHF）** 的训练过程，重点阐述了如何使用奖励模型来评估完成结果的质量，并通过强化学习对模型进行优化。最后，讨论了 RLHF 模型为什么通常效果更好，以及其背后的计算不对称性原理。


# Mode collapse

with lower variation than the base model. The base model has lots of entropy and will give lots of diverse outputs.
For example, one place where I still prefer to use a base model is in the setup
where you basically have n things and you want to generate more things like it.
Here is an example that I just cooked up. I want to generate cool Pokemon names.
I gave it seven Pokemon names and I asked the base model to complete the document and it gave me a lot more Pokemon names.
These are fictitious. I tried to look them up. I don't believe they're actual Pokemons. This is the task that I think the base model would be
good at because it still has lots of entropy. It'll give you lots of diverse cool more things that look like whatever you give it before.
Having said all that, these are the assistant models that are probably available to you at this point.
There was a team at Berkeley that ranked a lot of the available assistant models and give them basically Elo ratings.
Currently, some of the best models, of course, are GPT-4, by far, I would say, followed by Claude, GPT-3.5, and then a number of models,
some of these might be available as weights, like Vicuna, Koala, etc. The first three rows here are
all RLHF models and all of the other models to my knowledge, are SFT models, I believe.
That's how we train these models on the high level. Now I'm going to switch gears and let's look at how we can
best apply the GPT assistant model to your problems. Now, I would like to work
in setting of a concrete example. Let's work with a concrete example here.
Let's say that you are working on an article or a blog post, and you're going to write this sentence at the end.
"California's population is 53 times that of Alaska." So for some reason, you want to compare the populations of these two states.
Think about the rich internal monologue and tool use and how much work actually goes computationally in
your brain to generate this one final sentence. Here's maybe what that could look like in your brain.
For this next step, let me blog on my blog, let me compare these two populations.
First I'm going to obviously need to get both of these populations. Now, I know that I probably
don't know these populations off the top of my head so I'm aware of what I know or don't know of my self-knowledge.
I go, I do some tool use and I go to Wikipedia and I look up California's population and Alaska's population.
Now, I know that I should divide the two, but again, I know that dividing 39.2 by 0.74 is very unlikely to succeed.
That's not the thing that I can do in my head and so therefore, I'm going to rely on the calculator so I'm going to use a calculator,
punch it in and see that the output is roughly 53. Then maybe I do some reflection and sanity checks in
my brain so does 53 makes sense? Well, that's quite a large fraction, but then California is the most
populous state, so maybe that looks okay. Then I have all the information I might need, and now I get to the creative portion of writing.
I might start to write something like "California has 53x times greater" and then I think to myself,
that's actually like really awkward phrasing so let me actually delete that and let me try again.
As I'm writing, I have this separate process, almost inspecting what I'm writing and judging whether it looks good
or not and then maybe I delete and maybe I reframe it, and then maybe I'm happy with what comes out.
Basically long story short, a ton happens under the hood in terms of your internal monologue when you create sentences like this.
But what does a sentence like this look like when we are training a GPT on it? From GPT's perspective, this
is just a sequence of tokens. GPT, when it's reading or generating these tokens,
it just goes chunk, chunk, chunk, chunk and each chunk is roughly the same amount of computational work for each token.
These transformers are not very shallow networks they have about 80 layers of reasoning,
but 80 is still not like too much. This transformer is going to do its best to imitate,
but of course, the process here looks very different from the process that you took. In particular, in our final artifacts
in the data sets that we create, and then eventually feed to LLMs, all that internal dialogue was completely stripped and unlike you,
the GPT will look at every single token and spend the same amount of compute on every one of them. So, you can't expect it
to do too much work per token and also in particular,
basically these transformers are just like token simulators, they don't know what they don't know.
They just imitate the next token. They don't know what they're good at or not good at. They just tried their best to imitate the next token.
They don't reflect in the loop. They don't sanity check anything. They don't correct their mistakes along the way.
By default, they just are sample token sequences. They don't have separate inner monologue streams
in their head right? They're evaluating what's happening. Now, they do have some cognitive advantages,
I would say and that is that they do actually have a very large fact-based knowledge across a vast number of areas because they have,
say, several, 10 billion parameters. That's a lot of storage for a lot of facts. They also, I think have
a relatively large and perfect working memory. Whatever fits into the context window
is immediately available to the transformer through its internal self attention mechanism and so it's perfect memory,
but it's got a finite size, but the transformer has a very direct access to it and so it can a losslessly remember anything that
is inside its context window. This is how I would compare those two and the reason I bring all of this up is because I
think to a large extent, prompting is just making up for this cognitive difference between
these two architectures like our brains here and LLM brains.
You can look at it that way almost. Here's one thing that people found for example works pretty well in practice.
Especially if your tasks require reasoning, you can't expect the transformer to do too much reasoning per token.
You have to really spread out the reasoning across more and more tokens. For example, you can't give a transformer
a very complicated question and expect it to get the answer in a single token. There's just not enough time for it. "These transformers need tokens to
think," I like to say sometimes. This is some of the things that work well, you may for example have a few-shot prompt that
shows the transformer that it should show its work when it's answering question and if you give a few examples,
the transformer will imitate that template and it will just end up working out better in terms of its evaluation.
Additionally, you can elicit this behavior from the transformer by saying, let things step-by-step.
Because this conditions the transformer into showing its work and because
it snaps into a mode of showing its work, is going to do less computational work per token.
It's more likely to succeed as a result because it's making slower reasoning over time.
Here's another example, this one is called self-consistency. We saw that we had the ability

这段内容主要解释了**GPT 模型的“模式坍缩”现象（mode collapse）**，以及与人类思考方式的差异，并给出了一些提示工程（prompting）的建议。以下是逐段的中文解释：

---

### 🔹 模式坍缩（Mode collapse）

> 与基础模型（Base Model）相比，经过 RLHF 训练的助手模型（Assistant Model）**输出的多样性更低**，也就是说，它们更倾向于输出“更确定、标准”的回答，**缺乏创造性和多样性**。

* 基础模型拥有更高的“熵”（entropy），输出更加丰富多样；
* 比如，你给基础模型几个宝可梦（Pokemon）名字，它会自动生成更多看起来像宝可梦的创意名字；
* 尽管这些新名字是虚构的，但这种“创作性”的任务，**基础模型做得更好**；
* 因为助手模型倾向于“安全”、“规范”地回答问题，而不是创造。

---

### 🔹 模型排行榜

> 有研究团队（如伯克利大学）对各种助手模型进行排名（用 Elo 分数）。

* 最顶级的模型是：GPT-4（遥遥领先）；
* 接下来是 Claude、GPT-3.5；
* 还有一些开源模型，比如 Vicuna、Koala 等；
* **前三名通常是 RLHF 模型**（用人类反馈训练过），其余多数是 SFT 模型（监督微调）；

---

### 🔹 GPT 如何处理任务 vs 人脑如何处理任务

> 举了一个例子：“加州人口是阿拉斯加的 53 倍”，展示人类和 GPT 处理这个任务的方式差异。

#### 人脑：

* 会思考：我需要查两个州的人口；
* 上网查数据 → 用计算器除法 → 得出 53；
* 思考这个数字是否合理；
* 最后才开始写作，并不断检查和修改语句；
* **全过程包含丰富的“内心独白”和工具使用**。

#### GPT 模型：

* GPT 只是读取一串 token（如单词或标点），每一个 token 的处理计算量是一样的；
* 它没有自我意识，不知道自己不知道什么；
* 它不会主动反思，也不会检查或纠正自己的错误；
* 它只是**尽力模仿人类语言中下一个 token 应该是什么**。

---

### 🔹 GPT 的优势和局限

**优势：**

* 储存了海量知识（数百亿参数）；
* 上下文记忆完美（只要在上下文窗口中，它就能记住）；
* 对一些任务（如事实检索、代码编写）非常高效。

**局限：**

* **每个 token 的计算是有限的**，不能像人一样集中算一个东西；
* **无法“反思”或做长期规划**，需要通过提示来模拟这一过程。

---

### 🔹 提示工程的建议（Prompt Engineering）

> 提示工程的目的是**弥补 GPT 和人脑之间的差异**，引导模型进行更深入的“推理”。

* GPT **需要 token 才能思考**，复杂任务不能一口气完成；
* 给它“逐步推理”的机会，性能会提升；
* 举例：

  * 用 few-shot 提示让它模仿“展示过程”的回答；
  * 明确提示它 “一步一步来（let's think step-by-step）”，效果更好；
  * 利用“自一致性（self-consistency）”策略，即多次采样回答再投票选最常见的那个，准确率会上升。

---

如果你希望我把这段话做成更清晰的思维导图或中英对照笔记，也可以告诉我！


# Ensemble multiple attempts

to start writing and then if it didn't work out, I can try again and I can try multiple times
and maybe select the one that worked best. In these approaches,
you may sample not just once, but you may sample multiple times and then have some process for finding
the ones that are good and then keeping just those samples or doing a majority vote or something like that. Basically these transformers in the process as
they predict the next token, just like you, they can get unlucky and they could sample a not a very good
token and they can go down like a blind alley in terms of reasoning. Unlike you, they cannot recover from that.
They are stuck with every single token they sample and so they will continue the sequence, even if they know that this sequence is not going to work out.
Give them the ability to look back, inspect or try to basically sample around it.
Here's one technique also, it turns out that actually LLMs, they know when they've screwed up,

这段话解释了如何使用\*\*多次采样与集成（ensemble multiple attempts）\*\*的方法来提升大语言模型（LLM）的回答质量，以及这背后的原因。以下是逐句中文解释：

---

### 📌 多次尝试与集成（Ensemble multiple attempts）

> 一开始可以让模型尝试写一个回答，如果它写得不好，可以**再试几次**，然后选择表现最好的那一个。

* **我们可以让模型生成多个版本的回答**；
* 然后用某种机制选出最好的：

  * 比如人工选择；
  * 多个结果中**投票选择最常出现的答案**（majority vote）；
  * 或其他自动评估方法；
* 这个方法提升了模型整体表现的可靠性。

---

### 📌 为什么有效？因为 Transformer 会走“死胡同”

> Transformer 模型在生成 token 的时候，**有时会不小心生成不好的 token**，结果导致后续推理走上错误方向（走进“死胡同”）。

* 它不像人类一样能意识到“我好像出错了”然后撤回；
* 一旦它采样了一个 token，就只能接着往下走；
* 即使知道后面有点不对，也**不能回头重来**；
* 所以，如果你只让它生成一次，可能“运气不好”就失败了。

---

### ✅ 怎么改进？——给模型“试错的机会”

> 所以我们可以通过**多次采样 + 选择机制**来弥补模型不能“反悔”的缺陷：

* 让它**生成多个版本**；
* 比如一次生成 5 个不同回答；
* 然后从中**筛选出更好的那一个或几个**；
* 这种策略非常像我们人类思考的方式：想一想 -> 写下来 -> 不满意 -> 重来。

---

### 🤔 小提示：其实 LLM 是知道它自己“搞砸了”的

> 有趣的是，其实大语言模型 **有时是知道自己搞砸了的**，只是它没法修正。

* 它没有元认知能力，不能“反省”自己；
* 但你可以通过设计提示或用策略（比如 self-reflection prompt）让它表现得像在反思；
* 或者用多轮采样 + 筛选的方式，让错误输出被自动“抛弃”。

---

如果你需要我整理一份中文学习笔记或总结这些策略，欢迎告诉我！


# Ask for reflection

so as an example, say you ask the model to generate a poem that does not
rhyme and it might give you a poem, but it actually rhymes. But it turns out that especially for the bigger models like GPT-4,
you can just ask it "did you meet the assignment?" Actually GPT-4 knows very well that it did not meet the assignment.
It just got unlucky in its sampling. It will tell you, "No, I didn't actually meet the assignment here. Let me try again."
But without you prompting it it doesn't know to revisit and so on.
You have to make up for that in your prompts, and you have to get it to check, if you don't ask it to check,
its not going to check by itself it's just a token simulator.
I think more generally, a lot of these techniques fall into the bucket of what I would say recreating our System 2.
You might be familiar with the System 1 and System 2 thinking for humans. System 1 is a fast automatic process and I
think corresponds to an LLM just sampling tokens. System 2 is the slower deliberate
planning part of your brain. This is a paper actually from
just last week because this space is pretty quickly evolving, it's called Tree of Thought.
The authors of this paper proposed maintaining multiple completions for any given prompt
and then they are also scoring them along the way and keeping the ones that are going well if that makes sense.
A lot of people are really playing around with prompt engineering
to basically bring back some of these abilities that we have in our brain for LLMs.
Now, one thing I would like to note here is that this is not just a prompt. This is actually prompts that are together
used with some Python Glue code because you actually have to maintain multiple prompts and you also have to do
some tree search algorithm here to figure out which prompts to expand, etc. It's a symbiosis of Python Glue code and
individual prompts that are called in a while loop or in a bigger algorithm. I also think there's a really cool
parallel here to AlphaGo. AlphaGo has a policy for placing the next stone when it plays go,
and its policy was trained originally by imitating humans. But in addition to this policy,
it also does Monte Carlo Tree Search. Basically, it will play out a number of possibilities in its head and evaluate all of
them and only keep the ones that work well. I think this is an equivalent of AlphaGo but for text if that makes sense.
Just like Tree of Thought, I think more generally people are starting to really explore
more general techniques of not just the simple question-answer prompts, but something that looks a lot more like
Python Glue code stringing together many prompts. On the right, I have an example from this paper called React where they
structure the answer to a prompt as a sequence of thought-action-observation,
thought-action-observation, and it's a full rollout and a thinking process to answer the query.
In these actions, the model is also allowed to tool use. On the left, I have an example of AutoGPT.
Now AutoGPT by the way is a project that I think got a lot of hype recently,
but I think I still find it inspirationally interesting. It's a project that allows an LLM to keep
the task list and continue to recursively break down tasks. I don't think this currently works very well and I would
not advise people to use it in practical applications. I just think it's something to generally take inspiration
from in terms of where this is going, I think over time. That's like giving our model System 2 thinking.
The next thing I find interesting is, this following serve I would say almost psychological quirk of LLMs,
is that LLMs don't want to succeed, they want to imitate. You want to succeed, and you should ask for it.
What I mean by that is, when transformers are trained, they have training sets and there can be
an entire spectrum of performance qualities in their training data. For example, there could be some kind of a prompt
for some physics question or something like that, and there could be a student's solution that is completely wrong but there can also be an expert
answer that is extremely right. Transformers can't tell the difference between low,
they know about low-quality solutions and high-quality solutions, but by default, they want to imitate all of
it because they're just trained on language modeling. At test time, you actually have to ask for a good performance.
In this example in this paper, they tried various prompts. Let's think step-by-step was very powerful
because it spread out the reasoning over many tokens. But what worked even better is, let's work this out in a step-by-step way
to be sure we have the right answer. It's like conditioning on getting the right answer, and this actually makes the transformer work
better because the transformer doesn't have to now hedge its probability mass on low-quality solutions,
as ridiculous as that sounds. Basically, feel free to ask for a strong solution.
Say something like, you are a leading expert on this topic. Pretend you have IQ 120, etc. But don't try to ask for too much IQ because if
you ask for IQ 400, you might be out of data distribution, or even worse, you could be in data distribution for
something like sci-fi stuff and it will start to take on some sci-fi, or like roleplaying or something like that.
You have to find the right amount of IQ. I think it's got some U-shaped curve there.
Next up, as we saw when we are trying to solve problems, we know what we are good at and what we're not good at,
and we lean on tools computationally. You want to do the same potentially with your LLMs.

这段话讲的是：**如何通过提示词设计、结构化思维流程和工具调用，让大语言模型（LLM）模仿人类更有条理、更理性、更目标导向的“System 2”式思维**。以下是逐段的中文解释：

---

### 🧠 人类能“反思”，大模型不会，除非你提醒它

> 比如你让 GPT-4 写一首“**不押韵**”的诗，结果它却写了一首**押韵**的。如果你事后问它：“你完成任务了吗？”它**其实知道自己没完成任务**，只是采样过程中“运气不好”。

* GPT 默认只是一台“**token 模拟器**”，生成一个词接着一个词；
* 它不会**主动检查自己是否犯错**，但如果你问它，它可以反思；
* 所以你必须在提示词中**显式要求它检查、反思、评估**，否则它不会自己这么做。

---

### 🧠 System 1 vs. System 2 思维：我们要帮模型激活“System 2”

> 人类有两套思维系统：

* **System 1**：快速、直觉、自动（类似 LLM 一步步生成 token）；

* **System 2**：缓慢、理性、有计划（需要反思、检查、多路径思考）；

* 现代一些方法如 **Tree of Thought（ToT）** 就是模拟 System 2 思维，让模型**并行探索多条思路**，选择其中表现最好的。

---

### 🔁 Tree of Thought 不是一句提示词，而是一个流程 + Python glue code

* \*\*Tree of Thought（思维树）\*\*方法不是简单一句提示词；
* 它要写 Python 脚本串联多个提示词，用**树搜索算法**逐步展开问题空间，保留效果好的回答；
* 本质上是：提示词 + 程序 控制模型思维路径，类似 AI 玩游戏时“预演多步”。

---

### ♟ 类比 AlphaGo：AlphaGo 用策略 + 树搜索，Tree of Thought 是文本版的

> 就像 AlphaGo 先模仿人类下棋策略，然后使用 Monte Carlo Tree Search（MCTS）评估所有可能走法；
> Tree of Thought 就像是“下棋中的 MCTS”，但应用在文本任务上。

---

### 🧩 React 和 AutoGPT：更复杂的结构化推理流程

> **React** 方法：模型生成结构为“思考 → 行动 → 观察”，逐步执行任务；可以调用工具。

> **AutoGPT** 项目：模型会维护一个任务列表，并递归分解目标。但目前效果不佳，仍在探索阶段。

这两种方法都在**构建多步骤、结构化的推理流程**，而不是一问一答。

---

### ⚠️ LLM 并不“想要”答对题目 —— 它只想模仿！

> LLM 并不是在“追求正确答案”，而是**在模仿训练数据中出现过的内容**。训练集里既有学生错解，也有专家答案，模型一视同仁都学了。

---

### ✅ 想让它更靠谱？你必须“要求它做到最好”

> 举个例子：

* Prompt A: “Let's think step-by-step.”（我们一步步想）
* Prompt B: “Let's work this out step-by-step to **be sure we have the right answer**.”（为了确保正确答案，我们一步步推理）

后者表现更好。因为它明确要求模型“必须答对”，这样模型**就不会去分配概率给那些低质量回答**了。

---

### 🧠 提示词要“激发潜力”，但不能“扯太远”

> 你可以对模型说：

* “你是一个资深专家”；
* “假装你智商有 120”；

但**别说智商 400**，那可能会触发模型进入“奇怪的数据分布”——比如它以为你在讲科幻、搞角色扮演，输出就不靠谱了。

---

### 🛠️ 最后：模型也要学会用工具，就像我们用计算器一样

> 当我们知道自己算不出来的时候，就会用计算器。同样地，你可以在模型提示词中告诉它使用某些工具：

* 如调用搜索引擎；
* 使用代码计算器；
* 查询数据库等。

这就是“增强模型能力”的方向。

---

如果你希望我把这段内容整理成一张图表或中文笔记总结，随时告诉我！


# Tool use / Plugins

In particular, we may want to give them calculators, code interpreters,
and so on, the ability to do search, and there's a lot of techniques for doing that.
One thing to keep in mind, again, is that these transformers by default may not know what they don't know.
You may even want to tell the transformer in a prompt you are not very good at mental arithmetic. Whenever you need to do very large number addition,
multiplication, or whatever, instead, use this calculator. Here's how you use the calculator, you use this token combination, etc.
You have to actually spell it out because the model by default doesn't know what it's good at or not good at, necessarily, just like you and I might be.
Next up, I think something that is very interesting is we went from a world that was retrieval only all the way,
the pendulum has swung to the other extreme where its memory only in LLMs. But actually, there's this entire space in-between of
these retrieval-augmented models and this works extremely well in practice. As I mentioned, the context window of
a transformer is its working memory. If you can load the working memory with any information that is relevant to the task,
the model will work extremely well because it can immediately access all that memory. I think a lot of people are really interested
in basically retrieval-augment degeneration. On the bottom, I have an example of LlamaIndex which is
one data connector to lots of different types of data. You can index all
of that data and you can make it accessible to LLMs. The emerging recipe there is you take relevant documents,
you split them up into chunks, you embed all of them, and you basically get embedding vectors that represent that data.
You store that in the vector store and then at test time, you make some kind of a query to your vector store and you fetch chunks that
might be relevant to your task and you stuff them into the prompt and then you generate. This can work quite well in practice.
This is, I think, similar to when you and I solve problems. You can do everything from your memory and
transformers have very large and extensive memory, but also it really helps to reference some primary documents.
Whenever you find yourself going back to a textbook to find something, or whenever you find yourself going back to documentation of the library to look something up,
transformers definitely want to do that too. You have some memory over how
some documentation of the library works but it's much better to look it up. The same applies here.
Next, I wanted to briefly talk about constraint prompting. I also find this very interesting.
This is basically techniques for forcing a certain template in the outputs of LLMs.
Guidance is one example from Microsoft actually. Here we are enforcing that the output from the LLM will be JSON.
This will actually guarantee that the output will take on this form because they go in and they mess with the probabilities of
all the different tokens that come out of the transformer and they clamp those tokens and then the transformer is only filling in the blanks here,
and then you can enforce additional restrictions on what could go into those blanks. This might be really helpful, and I think
this constraint sampling is also extremely interesting. I also want to say
a few words about fine tuning. It is the case that you can get really far with prompt engineering, but it's also possible to
think about fine tuning your models. Now, fine tuning models means that you are actually going to change the weights of the model.
It is becoming a lot more accessible to do this in practice, and that's because of a number of techniques that have been
developed and have libraries for very recently. So for example parameter efficient fine tuning techniques like Laura,
make sure that you're only training small, sparse pieces of your model. So most of the model is kept clamped at
the base model and some pieces of it are allowed to change and this still works pretty well empirically and makes
it much cheaper to tune only small pieces of your model. It also means that because most of your model is clamped,
you can use very low precision inference for computing those parts because you are not going to be updated by
gradient descent and so that makes everything a lot more efficient as well. And in addition, we have a number of open source, high-quality base models.
Currently, as I mentioned, I think LLaMa is quite nice, although it is not commercially licensed, I believe right now.
Some things to keep in mind is that basically fine tuning is a lot more technically involved.
It requires a lot more, I think, technical expertise to do right. It requires human data contractors for
datasets and/or synthetic data pipelines that can be pretty complicated. This will definitely slow down
your iteration cycle by a lot, and I would say on a high level SFT is achievable because you're continuing
the language modeling task. It's relatively straightforward, but RLHF, I would say is very much research territory
and is even much harder to get to work, and so I would probably not advise that someone just tries to roll their own RLHF of implementation.
These things are pretty unstable, very difficult to train, not something that is, I think, very beginner friendly right now,
and it's also potentially likely also to change pretty rapidly still.
So I think these are my default recommendations right now. I would break up your task into two major parts.

这张图概述了如何优化大语言模型（LLM）的性能和成本，分为两个主要目标：

---

### **目标 1：实现最佳性能**

1. **使用 GPT-4**

   * 在任务中使用 GPT-4，以获得最强大的性能。

2. **使用详细的提示词（Prompt）**

   * 给任务提供明确的上下文、相关信息和指示：

     * 例如，可以询问：“如果任务联系人无法回复你，你会告诉他们什么？”这样的问题可以帮助模型理解任务的背景。

3. **检索并添加相关上下文信息**

   * 从外部信息源（如数据库或搜索引擎）获取相关背景，增强提示词。

4. **实验提示工程技巧**

   * 尝试不同的提示工程方法，例如使用少量示例来帮助模型理解任务并提高准确性。

5. **尝试工具/插件来分担 LLM 的任务**

   * 使用计算器、代码执行等工具来分担计算密集型任务，使 LLM 更专注于语言理解。

6. **花时间优化工作流程（管道/链）**

   * 优化整个模型使用流程，提高效率和准确度。

7. **如果你确信已经最大化了提示词的优化，考虑进行 SFT 数据收集和微调**

   * \*\*SFT（监督微调）\*\*阶段可以通过收集高质量的任务数据和微调模型来进一步提高性能。

8. **专家/研究阶段：考虑 RM 数据收集和 RLHF 微调**

   * 对于专家级任务或研究型项目，使用\*\*奖励建模（RM）**和**强化学习与人类反馈（RLHF）\*\*来进一步优化模型。

---

### **目标 2：优化成本**

1. **一旦达到最佳性能，开始尝试成本节约措施**

   * 在已经获得最强性能的情况下，采用**GPT-3.5**等更便宜的模型，或尝试使用更简短的提示词来减少计算成本。

---

这张图总结了如何在最大化性能的基础上，进一步通过优化提示词和微调来提高效率，并考虑如何降低成本的方法。


# Default recommendations

Number 1, achieve your top performance, and Number 2, optimize your performance in that order.
Number 1, the best performance will currently come from GPT-4 model. It is the most capable of all by far.
Use prompts that are very detailed. They have lots of task content, relevant information and instructions.
Think along the lines of what would you tell a task contractor if they can't email you back, but then also keep in mind that a task contractor is a
human and they have inner monologue and they're very clever, etc. LLMs do not possess those qualities.
So make sure to think through the psychology of the LLM almost and cater prompts to that.
Retrieve and add any relevant context and information to these prompts. Basically refer to a lot of
the prompt engineering techniques. Some of them I've highlighted in the slides above, but also this is a very large space and I would
just advise you to look for prompt engineering techniques online. There's a lot to cover there.
Experiment with few-shot examples. What this refers to is, you don't just want to tell, you want to show whenever it's possible.
So give it examples of everything that helps it really understand what you mean if you can.
Experiment with tools and plug-ins to offload tasks that are difficult for LLMs natively,
and then think about not just a single prompt and answer, think about potential chains and reflection and how you glue
them together and how you can potentially make multiple samples and so on. Finally, if you think you've squeezed
out prompt engineering, which I think you should stick with for a while, look at some potentially
fine tuning a model to your application, but expect this to be a lot more slower in the vault and then
there's an expert fragile research zone here and I would say that is RLHF, which currently does work a bit
better than SFT if you can get it to work. But again, this is pretty involved, I would say. And to optimize your costs,
try to explore lower capacity models or shorter prompts and so on.
I also wanted to say a few words about the use cases in which I think LLMs are currently well suited for.
In particular, note that there's a large number of limitations to LLMs today, and so I would keep that
definitely in mind for all of your applications. Models, and this by the way could be an entire talk. So I don't have time to cover it in full detail.
Models may be biased, they may fabricate, hallucinate information, they may have reasoning errors, they may struggle in entire classes of applications,
they have knowledge cut-offs, so they might not know any information above, say, September, 2021.
They are susceptible to a large range of attacks which are coming out on Twitter daily,
including prompt injection, jailbreak attacks, data poisoning attacks and so on. So my recommendation right now is
use LLMs in low-stakes applications. Combine them always with human oversight.
Use them as a source of inspiration and suggestions and think co-pilots, instead of completely autonomous agents
that are just like performing a task somewhere. It's just not clear that the models are there right now.
So I wanted to close by saying that GPT-4 is an amazing artifact. I'm very thankful that it exists, and it's beautiful.
It has a ton of knowledge across so many areas. It can do math, code and so on. And in addition, there's this
thriving ecosystem of everything else that is being built and incorporated into the ecosystem. Some of these things I've talked about,
and all of this power is accessible at your fingertips. So here's everything that's needed in terms of
code to ask GPT-4 a question, to prompt it, and get a response. In this case, I said,
can you say something to inspire the audience of Microsoft Build 2023? And I just punched this into Python and verbatim
GPT-4 said the following: And by the way, I did not know that they
used this trick in the keynote. So I thought I was being clever, but it is really good at this.
It says, ladies and gentlemen, innovators and trailblazers Microsoft Build 2023. Welcome to the gathering of brilliant
minds like no other, you are the architects of the future, the visionaries molding the digital realm
in which humanity thrives. Embrace the limitless possibilities of technologies and let your ideas soar as high as your imagination.
Together, let's create a more connected, remarkable, and inclusive world for generations to come. Get ready to unleash your creativity,
canvas the unknown, and turn dreams into reality. Your journey begins today!

这段话提供了关于**如何利用大语言模型（LLM）最大化性能**以及如何**优化成本**的建议。以下是详细的中文解释：

---

### **目标 1：实现最佳性能**

1. **使用 GPT-4**

   * **GPT-4** 是目前最强大的模型，提供最优性能。
   * **提示词要非常详细**，包含任务内容、相关信息和指示。
   * 例如，想象你在与任务承包商沟通时如果他们无法回复你，会告诉他们什么。**要考虑到任务承包商是人类，他们有内心独白和聪明的思维**，而大语言模型并不具备这些能力。所以在设计提示时要考虑到模型的“心理”，并相应调整提示。

2. **检索并添加相关上下文**

   * **增强提示词**，通过检索和添加相关上下文信息，提升模型的表现。
   * 可以参考一些**提示工程技巧**，这些技巧可以通过在线资源获取。

3. **实验少量示例（Few-shot examples）**

   * **示例很重要**，不仅仅是告诉模型要做什么，还要给它相关的**示范例子**，帮助模型更好地理解任务。

4. **使用工具和插件来减轻模型的负担**

   * 对于LLM本身难以处理的任务，使用**工具和插件**进行协作。例如：计算器、代码执行等。

5. **考虑任务链和反思**

   * 任务不仅仅是一次提问和回答，考虑如何将任务拆解成**多个步骤、链式操作**，并允许模型进行“反思”，从多个尝试中选择最好的结果。

6. **如果你觉得提示工程已经优化到极致，可以考虑微调模型**

   * 微调（Fine-tuning）是指调整模型权重来适应特定任务，但这会相对更慢，并且需要大量的数据和计算资源。

7. **专家级研究：考虑奖励建模（RM）和强化学习与人类反馈（RLHF）**

   * 如果你有足够的经验，并且任务较为复杂，**RLHF**（强化学习与人类反馈）可以提供更好的结果，但这涉及更深的技术研究，不适合初学者。

---

### **目标 2：优化成本**

1. **在达到最佳性能后，考虑节约成本**

   * 一旦你获得了最佳性能，可以通过以下方式来优化成本：

     * 使用更低容量的模型（如**GPT-3.5**）；
     * 使用更简短的提示词，减少计算资源消耗。

---

### **LLM的当前适用场景**

> 当前大语言模型有许多限制，因此适用场景也有局限。以下是一些重要注意事项：

1. **模型可能有偏见，可能会捏造信息，可能存在推理错误**；
2. **知识切割点问题**：模型可能无法获取2021年之后的信息。
3. **攻击易感性**：模型容易受到攻击，如提示注入、越狱攻击、数据污染等。
4. **建议**：目前大语言模型更适用于低风险的应用，并且**始终结合人工监督使用**。

   * 可以将它们用作**灵感和建议的来源**，类似于“协作助手”，而不是完全独立执行任务的自动化代理。

---

### **GPT-4的巨大优势**

> GPT-4 是一个非常强大的工具，能够跨多个领域提供知识，它能做数学、编程等任务，且能够提供高质量的输出。这个强大的能力可以为你提供更多创意和灵感。

* 例如：**GPT-4**能够通过简单的提示生成精彩的演讲稿，下面是它生成的微软 Build 2023 的欢迎词：

  > “女士们，先生们，创新者和先行者，欢迎来到微软 Build 2023。这是一个聚集着辉煌思想的场所，您是未来的建筑师，是那些塑造数字世界的远见卓识者。在这里，拥抱技术的无限可能，让你的想法像你的想象力一样翱翔。让我们共同创造一个更加互联、卓越和包容的世界，为未来的世代打造更美好的未来。准备好释放你的创造力，探索未知，把梦想变成现实。你的旅程从今天开始！”

---

### 总结

这段话强调了如何通过**精确的提示词设计、工具使用和模型微调**来提升大语言模型的表现，并提到在获得最佳性能后如何优化成本。还提醒了使用 LLM 时需要注意的限制，以及如何结合人工监督来提高其效果。


