This is a general audience deep dive into the Large Language Model (LLM) AI technology that powers ChatGPT and related products. It is covers the full training stack of how the models are developed, along with mental models of how to think about their "psychology", and how to get the best use them in practical applications. I have one "Intro to LLMs" video already from ~year ago, but that is just a re-recording of a random talk, so I wanted to loop around and do a lot more comprehensive version.

Instructor
Andrej was a founding member at OpenAI (2015) and then Sr. Director of AI at Tesla (2017-2022), and is now a founder at Eureka Labs, which is building an AI-native school. His goal in this video is to raise knowledge and understanding of the state of the art in AI, and empower people to effectively use the latest and greatest in their work.
Find more at https://karpathy.ai/ and https://x.com/karpathy

```
Chapters
00:00:00 introduction
00:01:00 pretraining data (internet)
00:07:47 tokenization
00:14:27 neural network I/O
00:20:11 neural network internals
00:26:01 inference
00:31:09 GPT-2: training and inference
00:42:52 Llama 3.1 base model inference
00:59:23 pretraining to post-training
01:01:06 post-training data (conversations)
01:20:32 hallucinations, tool use, knowledge/working memory
01:41:46 knowledge of self
01:46:56 models need tokens to think
02:01:11 tokenization revisited: models struggle with spelling
02:04:53 jagged intelligence
02:07:28 supervised finetuning to reinforcement learning
02:14:42 reinforcement learning
02:27:47 DeepSeek-R1
02:42:07 AlphaGo
02:48:26 reinforcement learning from human feedback (RLHF)
03:09:39 preview of things to come
03:15:15 keeping track of LLMs
03:18:34 where to find LLMs
03:21:46 grand summary
```

Links
- ChatGPT https://chatgpt.com/
- FineWeb (pretraining dataset): https://huggingface.co/spaces/Hugging...
- Tiktokenizer: https://tiktokenizer.vercel.app/
- Transformer Neural Net 3D visualizer: https://bbycroft.net/llm
- llm.c Let's Reproduce GPT-2 https://github.com/karpathy/llm.c/dis...
- Llama 3 paper from Meta: https://arxiv.org/abs/2407.21783
- Hyperbolic, for inference of base model: https://app.hyperbolic.xyz/
- InstructGPT paper on SFT: https://arxiv.org/abs/2203.02155
- HuggingFace inference playground: https://huggingface.co/spaces/hugging...
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- TogetherAI Playground for open model inference: https://api.together.xyz/playground
- AlphaGo paper (PDF): https://discovery.ucl.ac.uk/id/eprint...
- AlphaGo Move 37 video:    • Lee Sedol vs AlphaGo  Move 37 reactions an...  
- LM Arena for model rankings: https://lmarena.ai/
- AI News Newsletter: https://buttondown.com/ainews
- LMStudio for local inference https://lmstudio.ai/

- The visualization UI I was using in the video: https://excalidraw.com/
- The specific file of Excalidraw we built up: https://drive.google.com/file/d/1EZh5...
- Discord channel for Eureka Labs and this video:   / discord  

Educational Use Licensing
This video is freely available for educational and internal training purposes. Educators, students, schools, universities, nonprofit institutions, businesses, and individual learners may use this content freely for lessons, courses, internal training, and learning activities, provided they do not engage in commercial resale, redistribution, external commercial use, or modify content to misrepresent its intent.

这是一个面向一般观众的深度解析，介绍了支持ChatGPT及相关产品的大型语言模型（LLM）AI技术。视频全面讲解了模型的开发过程，包括从数据预训练到神经网络架构的各个方面，探讨了如何理解它们的“心理”，以及如何在实际应用中高效使用这些技术。虽然我大约一年前已经发布了一个“LLM入门”视频，但那只是一次随机演讲的重录，因此我希望通过这次视频进行更加全面的讲解。

**讲师：**
Andrej是OpenAI的创始成员之一（2015年），曾担任特斯拉的AI高级总监（2017-2022），现为Eureka Labs的创始人，该公司致力于构建AI本土化的学校。他在这个视频中的目标是提升人们对AI技术最前沿的了解，并帮助他们在工作中高效利用最新的AI成果。

**更多信息**：
- [Andrej的个人网站](https://karpathy.ai/)
- [Andrej的X主页](https://x.com/karpathy)

```
**视频章节：**
00:00:00 引言
00:01:00 预训练数据（互联网）
00:07:47 分词
00:14:27 神经网络输入输出
00:20:11 神经网络内部结构
00:26:01 推理过程
00:31:09 GPT-2：训练与推理
00:42:52 Llama 3.1基础模型推理
00:59:23 从预训练到后训练
01:01:06 后训练数据（对话）
01:20:32 幻觉、工具使用、知识与工作记忆
01:41:46 自我认知
01:46:56 模型需要令牌才能思考
02:01:11 分词再探：模型在拼写上的困扰
02:04:53 锯齿型智能
02:07:28 监督微调到强化学习
02:14:42 强化学习
02:27:47 DeepSeek-R1
02:42:07 AlphaGo
02:48:26 来自人类反馈的强化学习（RLHF）
03:09:39 未来展望
03:15:15 跟踪LLM的发展
03:18:34 如何找到LLM
03:21:46 总结
```

**相关链接：**

* [ChatGPT](https://chatgpt.com/)
* [FineWeb（预训练数据集）](https://huggingface.co/spaces/Hugging...)
* [Tiktokenizer](https://tiktokenizer.vercel.app/)
* [Transformer神经网络3D可视化器](https://bbycroft.net/llm)
* [llm.c：重现GPT-2](https://github.com/karpathy/llm.c/dis...)
* [Meta的Llama 3论文](https://arxiv.org/abs/2407.21783)
* [Hyperbolic，基础模型推理](https://app.hyperbolic.xyz/)
* [InstructGPT论文：SFT](https://arxiv.org/abs/2203.02155)
* [HuggingFace推理平台](https://huggingface.co/spaces/hugging...)
* [DeepSeek-R1论文](https://arxiv.org/abs/2501.12948)
* [TogetherAI开放模型推理平台](https://api.together.xyz/playground)
* [AlphaGo论文（PDF）](https://discovery.ucl.ac.uk/id/eprint...)
* \[AlphaGo第37步视频]\(• Lee Sedol vs AlphaGo  Move 37 reactions an...)
* [LM Arena：模型排名](https://lmarena.ai/)
* [AI新闻通讯](https://buttondown.com/ainews)
* [LMStudio本地推理平台](https://lmstudio.ai/)

**视频中使用的可视化UI**：
- [Excalidraw](https://excalidraw.com/)
- [我们在Excalidraw中构建的具体文件](https://drive.google.com/file/d/1EZh5...)
- [Eureka Labs和本视频的Discord频道]

**教育用途许可：**
该视频可自由用于教育和内部培训目的。教育工作者、学生、学校、大学、非营利机构、企业和个人学习者可以自由使用此内容用于课程、教学、内部培训和学习活动，前提是不得进行商业转售、再分发、外部商业用途，或修改内容以误导其原意。

# introduction

hi everyone so I've wanted to make this video for a while it is a comprehensive but General audience introduction to
large language models like Chachi PT and what I'm hoping to achieve in this video is to give you kind of mental models for
thinking through what it is that this tool is it is obviously magical and
amazing in some respects it's uh really good at some things not very good at other things and there's also a lot of
sharp edges to be aware of so what is behind this text box you can put anything in there and press enter but uh
what should we be putting there and what are these words generated back how does this work and what what are you talking
to exactly so I'm hoping to get at all those topics in this video we're going to go through the entire pipeline of how
this stuff is built but I'm going to keep everything uh sort of accessible to a general audience so let's take a look
at first how you build something like chpt and along the way I'm going to talk about um you know some of the sort of
cognitive psychological implications of the tools okay so let's build Chachi PT

大家好，我一直想做这个视频，这是一个针对普通观众的全面介绍，讲解大型语言模型（如ChatGPT）。我希望通过这个视频给大家提供一些思维模型，帮助大家理解这个工具是什么。它在某些方面确实很神奇和令人惊叹，但在一些领域也并不特别强大，而且它也有很多“锋利的边缘”，我们需要注意。

那么，这个文本框背后到底是什么呢？你可以输入任何内容，然后按下回车键，生成的文字又是怎么产生的呢？它是如何工作的？你到底是在和什么“对话”？我希望在视频中能涵盖这些话题，我们将一起了解如何构建像ChatGPT这样的工具，并在这个过程中讨论一些与工具相关的认知心理学方面的内容。

接下来，让我们看看如何构建ChatGPT，并在这个过程中讨论一些细节。

# pretraining data (internet)

so there's going to be multiple stages arranged sequentially the first stage is called the pre-training stage and the
first step of the pre-training stage is to download and process the internet now to get a sense of what this roughly
looks like I recommend looking at this URL here so um this company called
hugging face uh collected and created and curated this data set called Fine
web and they go into a lot of detail on this block post on how how they constructed the fine web data set and
all of the major llm providers like open AI anthropic and Google and so on will have some equivalent internally of
something like the fine web data set so roughly what are we trying to achieve here we're trying to get ton of text
from the internet from publicly available sources so we're trying to have a huge quantity of very high
quality documents and we also want very large diversity of documents because we want to have a lot of knowledge inside
these models so we want large diversity of high quality documents and we want many many of them and achieving this is
uh quite complicated and as you can see here takes multiple stages to do well so let's take a look at what some of these
stages look like in a bit for now I'd like to just like to note that for example the fine web data set which is fairly representative what you would see
in a production grade application actually ends up being only about 44 terabyt of dis space um you can get a
USB stick for like a terabyte very easily or I think this could fit on a single hard drive almost today so this
is not a huge amount of data at the end of the day even though the internet is very very large we're working with text
and we're also filtering it aggressively so we end up with about 44 terabytes in this example so let's take a look at uh
kind of what this data looks like and what some of these stages uh also are so the starting point for a lot of these
efforts and something that contributes most of the data by the end of it is Data from common crawl so common craw is
an organization that has been basically scouring the internet since 2007 so as of 2024 for example common CW has
indexed 2.7 billion web pages uh and uh they have all these crawlers going around the internet and
what you end up doing basically is you start with a few seed web pages and then you follow all the links and you just
keep following links and you keep indexing all the information and you end up with a ton of data of the internet over time so this is usually the
starting point for a lot of the uh for a lot of these efforts now this common C data is quite raw and is filtered in
many many different ways so here they Pro they document this is
the same diagram they document a little bit the kind of processing that happens in these stages so the first thing here
is something called URL filtering so what that is referring to is that there's these block
lists of uh basically URLs that are or domains that uh you don't want to be
getting data from so usually this includes things like U malware websites spam websites marketing websites uh
racist websites adult sites and things like that so there's a ton of different types of websites that are just
eliminated at this stage because we don't want them in our data set um the second part is text extraction you have
to remember that all these web pages this is the raw HTML of these web pages that are being saved by these crawlers
so when I go to inspect here this is what the raw HTML actually looks like you'll notice that it's got
all this markup uh like lists and stuff like that and there's CSS and all this
kind of stuff so this is um computer code almost for these web pages but what we really want is we just want this text
right we just want the text of this web page and we don't want the navigation and things like that so there's a lot of filtering and processing uh and heris
that go into uh adequately filtering for just their uh good content of these web
pages the next stage here is language filtering so for example fine web
filters uh using a language classifier they try to guess what language every single web page is in and then they only
keep web pages that have more than 65% of English as an example and so you can get a sense that
this is like a design decision that different companies can uh can uh take for themselves what fraction of all
different types of languages are we going to include in our data set because for example if we filter out all of the Spanish as an example then you might
imagine that our model later will not be very good at Spanish because it's just never seen that much data of that language and so different companies can
focus on multilingual performance to uh to a different degree as an example so fine web is quite focused on English and
so their language model if they end up training one later will be very good at English but not may be very good at
other languages after language filtering there's a few other filtering steps and D duplication and things like that um
finishing with for example the pii removal this is personally identifiable
information so as an example addresses Social Security numbers and things like that you would try to detect them and
you would try to filter out those kinds of web pages from the the data set as well so there's a lot of stages here and I won't go into full detail but it is a
fairly extensive part of the pre-processing and you end up with for example the fine web data set so when
you click in on it uh you can see some examples here of what this actually ends up looking like and anyone can download
this on the huging phase web page and so here are some examples of the final text
that ends up in the training set so this is some article about tornadoes in
2012 um so there's some t tadoes in 2020 in 2012 and what
happened uh this next one is something about did you know you have two little
yellow 9vt battery sized adrenal glands in your body okay so this is some kind
of a odd medical article so just think of these as
basically uh web pages on the internet filtered just for the text in various ways and now we have a ton of text 40
terabytes off it and that now is the starting point for the next step of this stage now I wanted to give you an
intuitive sense of where we are right now so I took the first 200 web pages here and remember we have tons of them
and I just take all that text and I just put it all together concatenate it and so this is what we end up with we just
get this just just raw text raw internet text and there's a ton of it even in
these 200 web pages so I can continue zooming out here and we just have this like massive tapestry of Text data and
this text data has all these p patterns and what we want to do now is we want to start training neural networks on this
data so the neural networks can internalize and model how this text
flows right so we just have this giant texture of text and now we want to get
neural Nets that mimic it okay now before we plug text into neural networks

首先，训练大型语言模型的过程分为多个阶段，依次进行。第一个阶段叫做**预训练阶段**，而预训练阶段的第一步就是下载和处理互联网数据。为了让大家大致了解这个过程，我推荐大家查看这个网址。

有一个公司叫做Hugging Face，他们收集、创建并策划了一个数据集，叫做**Fine Web**。在他们的博客中，详细介绍了Fine Web数据集的构建过程。所有主要的LLM提供商，如OpenAI、Anthropic、Google等，都会有类似于Fine Web数据集的内部版本。所以大致来说，我们的目标是什么呢？我们需要从互联网上获取大量公开可用的文本数据，确保这些数据既有很高的质量，又有很大的多样性，因为我们希望这些模型能够拥有丰富的知识。

要实现这一目标是相当复杂的，如你所见，这个过程需要多个步骤才能做好。那么，让我们来看看这些步骤是如何进行的。首先，Fine Web数据集（它非常具有代表性，能反映生产级应用中会使用的数据集）最终大约是44TB的数据空间。你可以很容易地找到一个1TB的USB硬盘，这个数据量甚至能装进一个硬盘里。因此，即使互联网非常庞大，我们实际上处理的只是文本数据，而且我们还对数据进行了强烈的过滤，最终得到大约44TB的数据。

接下来，我们来看看这些数据长什么样，处理的步骤又是什么样的。很多数据的来源来自**Common Crawl**，这个组织自2007年开始，基本上就在不停地抓取互联网数据。到2024年，Common Crawl已经索引了大约27亿个网页。他们的抓取方式是从几个种子网页开始，然后跟随网页上的链接，不断地抓取信息，最终获得大量的互联网数据。这通常是许多LLM开发项目的起点。

然而，这些Common Crawl的数据是非常原始的，需要经过多次过滤处理。具体的处理过程是这样的：首先是**URL过滤**，即过滤掉不想要的网页或域名，比如恶意网站、垃圾网站、营销网站、种族歧视网站、成人网站等。接着是**文本提取**，因为这些网页抓取的是原始的HTML代码，而我们只需要从中提取纯文本内容，去除导航栏等无关内容。接下来是**语言过滤**，例如Fine Web使用语言分类器来猜测每个网页的语言，并只保留大于65%为英语的网页。这样做的目的是为了保证模型在训练时能集中在某一特定语言（如英语）上，而不至于让模型对其他语言的理解能力过弱。

接下来是其他的过滤步骤，比如去重和**个人身份信息（PII）移除**，即过滤掉含有个人信息的网页，比如地址、社会保障号码等。

所有这些步骤都需要经过非常复杂的预处理，最终才会得到像Fine Web这样一个干净且高质量的数据集。当你访问Hugging Face网站时，就可以下载并查看这些经过处理的文本数据。比如，数据集中的一篇文章讲述了2012年发生的龙卷风，另一篇文章则介绍了人体内的肾上腺腺体。

到这里，我们已经处理好大约40TB的文本数据，接下来将进入下一阶段，开始训练神经网络。为了让大家对这一过程有一个直观的理解，我拿了前200个网页的内容，简单地将它们拼接在一起，形成了一个庞大的文本集合。即使只是这200个网页的数据，我们也可以看到这段文本数据像一幅巨大的文本拼图一样，充满了各种模式和结构。现在，我们的目标是让神经网络能够学习并模拟这些文本的流动规律。因此，接下来的步骤就是开始用这些数据训练神经网络，让它们能够理解和模仿这些文本的结构和语言模式。


# tokenization

we have to decide how we're going to represent this text uh and how we're going to feed it in now the way our
technology works for these neuron Lots is that they expect a one-dimensional sequence of symbols
and they want a finite set of symbols that are possible and so we have to
decide what are the symbols and then we have to represent our data as one-dimensional sequence of those
symbols so right now what we have is a onedimensional sequence of text it starts here and it goes here and then it
comes here Etc so this is a onedimensional sequence even though on my monitor of course it's laid out in a
two-dimensional way but it goes from left to right and top to bottom right so it's a one-dimensional sequence of text
now this being computers of course there's an underlying representation here so if I do what's called utf8 uh
encode this text then I can get the raw bits that correspond to this text in the
computer and that's what uh that looks like this so it turns out that for
example this very first bar here is the first uh eight bits as an
example so what is this thing right this is um representation that we are looking
for uh in in a certain sense we have exactly two possible symbols zero and one and we have a very long sequence of
it right now as it turns out um this sequence length is actually going to be
very finite and precious resource uh in our neural network and we actually don't want extremely long sequences of just
two symbols instead what we want is we want to trade off uh this um symbol
size uh of this vocabulary as we call it and the resulting sequence length so we
don't want just two symbols and extremely long sequences we're going to want more symbols and shorter sequences
okay so one naive way of compressing or decreasing the length of our sequence here is to basically uh consider some
group of consecutive bits for example eight bits and group them into a single
what's called bite so because uh these bits are either on or off if we take a
group of eight of them there turns out to be only 256 possible combinations of how these bits could be on or off and so
therefore we can re repesent this sequence into a sequence of bytes instead so this sequence of bytes will
be eight times shorter but now we have 256 possible symbols so every number
here goes from 0 to 255 now I really encourage you to think of these not as numbers but as unique
IDs or like unique symbols so maybe it's a bit more maybe it's better to actually
think of these to replace every one of these with a unique Emoji you'd get something like this so um we basically
have a sequence of emojis and there's 256 possible emojis you can think of it that way now it turns out that in
production for state-of-the-art language models uh you actually want to go even Beyond this you want to continue to
shrink the length of the sequence uh because again it is a precious resource in return for more symbols in your
vocabulary and the way this is done is done by running what's called The Bite pair encoding algorithm and the way this
works is we're basically looking for consecutive bytes or symbols that are
very common so for example turns out that the sequence 116 followed by 32 is
quite common and occurs very frequently so what we're going to do is we're going to group uh this um pair into a new
symbol so we're going to Mint a symbol with an ID 256 and we're going to rewrite every single uh pair 11632 with
this new symbol and then can we can iterate this algorithm as many times as we wish and each time when we mint a new
symbol we're decreasing the length and we're increasing the symbol size and in practice it turns out that a pretty good
setting of um the basically the vocabulary size turns out to be about 100,000 possible symbols so in
particular GPT 4 uses 100, 277 symbols
um and this process of converting from raw text into these symbols or as we
call them tokens is the process called tokenization so let's now take a look at
how gp4 performs tokenization conting from text to tokens and from tokens back
to text and what this actually looks like so one website I like to use to explore these token representations is
called tick tokenizer and so come here to the drop down and select CL 100 a base which is the gp4 base model
tokenizer and here on the left you can put in text and it shows you the tokenization of that text so for example
heo space world so hello world turns out to be
exactly two Tokens The Token hello which is the token with ID
15339 and the token space world that is the token 1
1917 so um hello space world now if I was to join these two for example I'm
going to get again two tokens but it's the token H followed by the token L world without the
H um if I put in two Spa two spaces here between hello and world it's again a
different uh tokenization there's a new token 220
here okay so you can play with this and see what happens here also keep in mind this is not uh this is case sensitive so
if this is a capital H it is something else or if it's uh hello world then
actually this ends up being three tokens instead of just two
tokens yeah so you can play with this and get an sort of like an intuitive sense of uh what these tokens work like
we're actually going to loop around to tokenization a bit later in the video for now I just wanted to show you the website and I wanted to uh show you that
this text basically at the end of the day so for example if I take one line here this is what GT4 will see it as so
this text will be a sequence of length 62 this is the sequence here and this is
how the chunks of text correspond to these symbols and again there's 100,
27777 possible symbols and we now have one-dimensional sequences of those
symbols so um yeah we're going to come back to tokenization but that's uh for now where we are okay so what I've done

在处理这些文本时，我们需要决定如何表示它们，并将其输入到神经网络中。神经网络需要的是一维的符号序列，并且每个符号来自一个有限的符号集合。因此，我们需要决定使用哪些符号，然后将文本数据表示为这些符号的一个一维序列。

现在我们手头有的是一维的文本序列，它从一个位置开始，逐渐延伸到另一个位置。尽管在显示器上它是二维排列的，但它本质上是一个从左到右、从上到下的文本序列。虽然这看起来像是一个简单的文本序列，但计算机内部实际上是有一套底层的表示方式的。例如，如果我们使用UTF-8编码将这个文本转换成计算机能理解的原始二进制位，我们会得到这样的结果：这就是计算机中表示文本的方式。

在计算机中，最终我们只有两种符号——0和1，这样就形成了一个非常长的二进制序列。不过，这种序列对于神经网络来说并不高效，因为每个序列的长度是有限且珍贵的。因此，我们不希望仅用0和1来表示这么长的序列，而是希望通过增加符号的种类来缩短序列的长度。也就是说，我们希望通过使用更多符号和更短的序列来交换空间和时间的权衡。

一种简单的方式是将多个连续的二进制位组合成一个字节（8位）。每个字节有256种可能的组合，因此可以将这些二进制序列转换为字节序列。这样，序列就会变得短得多，但我们也有了256个不同的符号。例如，字节0到255就可以视为256种不同的符号。此时，我们可以把这些字节当作符号来看待，类似于用256个不同的表情符号来替代这些字节。

然而，在实际的生产环境中，最先进的语言模型通常会使用一种叫做**字节对编码**（Byte Pair Encoding，BPE）的算法，进一步缩短序列的长度。这个算法的原理是寻找在文本中频繁出现的连续符号对，然后将这些常见的符号对替换为新的符号。例如，某些符号对，比如116和32，经常同时出现，我们可以将它们组合成一个新的符号ID，然后在后续的文本处理中用这个新符号代替。通过不断重复这个过程，我们可以减少序列的长度，并增加符号的种类。

最终，这种符号集合的大小通常是几万到十几万个符号。例如，GPT-4使用的是100,277个符号。

将文本转换为这些符号，或者我们称之为**标记**（tokens）的过程叫做**标记化**。接下来，我们来看看GPT-4如何执行从文本到标记的转换，并从标记回到文本。

为了更好地理解这些标记如何工作，我喜欢使用一个网站叫做**Tiktokenizer**，在这个网站上，你可以看到文本的标记化过程。例如，当你输入"hello world"时，它会被标记化为两个标记，一个是"hello"，另一个是"world"。这两个标记对应的ID分别是15339和11917。

如果你尝试输入“hello  world”（注意空格），则会看到不同的标记化结果，因为空格的数量影响了标记的数量。在这个过程中，标记是区分大小写的，所以"Hello"和"hello"会被视为不同的标记，生成不同的ID。

通过这种方式，我们可以将一段文本转化为神经网络可以理解的符号序列，模型将根据这些标记来处理和生成文本。

在这个过程中，GPT-4会把这段文本转换为一个62长度的符号序列，并通过这些符号来执行进一步的处理。这个符号集合包含了100,277个可能的符号，神经网络会处理这些符号，生成最终的输出。


# neural network I/O

now is I've taken this uh sequence of text that we have here in the data set and I have re-represented it using our
tokenizer into a sequence of tokens and this is what that looks like now so for
example when we go back to the Fine web data set they mentioned that not only is this 44 terab of dis space but this is
about a 15 trillion token sequence of um in this data set and so here these are
just some of the first uh one or two or three or a few thousand here I think uh tokens of this data set but there's 15
trillion here uh to keep in mind and again keep in mind one more time that all of these represent little text
chunks they're all just like atoms of these sequences and the numbers here don't make any sense they're just uh
they're just unique IDs okay so now we get to the fun part which is the uh
neural network training and this is where a lot of the heavy lifting happens computationally when you're training these neural networks so what we do here
in this this step is we want to model the statistical relationships of how these tokens follow each other in the
sequence so what we do is we come into the data and we take Windows of tokens
so we take a window of tokens uh from this data fairly randomly and um the windows length can
range anywhere anywhere between uh zero tokens actually all the way up to some
maximum size that we decide on uh so for example in practice you could see a token with Windows of say 8,000 tokens
now in principle we can use arbitrary window lengths of tokens uh but uh
processing very long uh basically U window sequences would just be very
computationally expensive so we just kind of decide that say 8,000 is a good number or 4,000 or 16,000 and we crop it
there now in this example I'm going to be uh taking the first four tokens just
so everything fits nicely so these tokens we're going to take a window of four
tokens this bar view in and space single
which are these token IDs and now what we're trying to do here is we're trying to basically predict the
token that comes next in the sequence so 3962 comes next right so what we do now
here is that we call this the context these four tokens are context and they
feed into a neural network and this is the input to the neural network
now I'm going to go into the detail of what's inside this neural network in a little bit for now it's important to understand is the input and the output
of the neural net so the input are sequences of tokens of variable length
anywhere between zero and some maximum size like 8,000 the output now is a
prediction for what comes next so because our vocabulary has
100277 possible tokens the neural network is going to Output exactly that many numbers
and all of those numbers correspond to the probability of that token as coming next in the sequence so it's making
guesses about what comes next um in the beginning this neural network is randomly initialized so um
and we're going to see in a little bit what that means but it's a it's a it's a random transformation so these
probabilities in the very beginning of the training are also going to be kind of random uh so here I have three
examples but keep in mind that there's 100,000 numbers here um so the probability of this token space
Direction neural network is saying that this is 4% likely right now 11799 is 2%
and then here the probility of 3962 which is post is 3% now of course we've
sampled this window from our data set so we know what comes next we know and that's the label we know that the
correct answer is that 3962 actually comes next in the sequence so now what we have is this mathematical process for
doing an update to the neural network we have the way of tuning it and uh we're going to go into a little bit of of
detail in a bit but basically we know that this probability here of 3% we want
this probability to be higher and we want the probabilities of all the other tokens to be
lower and so we have a way of mathematically calculating how to adjust and update the neural network so that
the correct answer has a slightly higher probability so if I do an update to the neural network now the next time I Fe
this particular sequence of four tokens into neural network the neural network will be slightly adjusted now and it will say Okay post is maybe 4% and case
now maybe is 1% and uh Direction could become 2% or something like that and so we have a way
of nudging of slightly updating the neuronet to um basically give a higher
probability to the correct token that comes next in the sequence and now you just have to remember that this process
happens not just for uh this um token here where these four fed in and
predicted this one this process happens at the same time for all of these tokens in the entire data set and so in
practice we sample little windows little batches of Windows and then at every single one of these tokens we want to
adjust our neural network so that the probability of that token becomes slightly higher and this all happens in
parallel in large batches of these tokens and this is the process of training the neural network it's a
sequence of updating it so that it's predictions match up the statistics of
what actually happens in your training set and its probabilities become consistent with the uh statistical
patterns of how these tokens follow each other in the data so let's now briefly get into the internals of these neural

我们现在已经将文本数据用标记化工具转换成了一个标记序列。例如，**Fine Web**数据集中的数据不仅有大约44TB的空间，而且它包含约15万亿个标记。这里展示的只是数据集中的一小部分标记，但需要记住的是，这些标记都是文本的“原子”，每个标记都是一个唯一的ID，数值本身并没有实际意义。

接下来，我们进入了神经网络训练的部分，这个过程是非常重要且计算密集的。在这个阶段，我们的目标是建模这些标记之间的统计关系，即它们如何在序列中相互依赖。我们从数据集中随机选取一个**窗口**，窗口的大小可以在一定范围内变化，从0个标记到一个预先设定的最大值。通常情况下，窗口的大小会被限制在一个较合理的范围内，比如4,000、8,000或16,000个标记。

在这个例子中，我们选取了一个**四个标记的窗口**，这四个标记是："this"，"is"，"a"，"test"。我们现在的任务是预测接下来的标记。例如，在这个窗口中，我们知道接下来应该是标记ID为`3962`的标记（在这个例子中是"post"）。

### **神经网络的输入与输出**

* **输入：** 神经网络的输入是这几个标记，它们构成了一个上下文。神经网络会根据这些上下文来预测下一个标记。
* **输出：** 神经网络的输出是一个概率分布，表示所有可能的标记中每个标记作为下一个标记的概率。例如，假设我们的词汇表包含100,277个标记，那么神经网络会输出100,277个概率值，每个值表示相应标记的出现概率。

### **训练过程**

1. **初始预测：** 在训练开始时，神经网络是随机初始化的，因此它的输出（即各标记的概率）也是随机的。例如，神经网络可能预测：

   * 标记`" "`（空格）的概率为4%
   * 标记ID为`11799`的概率为2%
   * 标记ID为`3962`（"post"）的概率为3%

2. **更新过程：** 我们知道正确的答案应该是`3962`（即"post"）。因此，神经网络应该提高该标记的概率，并减少其他标记的概率。为了实现这一点，我们使用**反向传播**算法和**梯度下降**方法来调整神经网络的权重，使得在下一次预测时，正确标记（`3962`）的概率会变得更高。

3. **批量训练：** 在实际训练过程中，这个过程不仅仅发生在一个窗口中，而是会并行处理数据集中的多个窗口。每个窗口的训练都会更新神经网络的参数。通过这种方式，神经网络逐步调整其内部参数，以便在处理大量数据时能够更准确地预测下一个标记。

4. **迭代过程：** 这个更新过程会重复进行，直到神经网络的预测与实际数据中的统计模式相符，能够准确地预测标记之间的关系。

### **总结：**

这个训练过程的目标是通过不断地调整神经网络，使它能够根据上下文信息准确预测下一个标记。训练过程中，每个标记都会对神经网络进行微调，使得它的预测概率与实际数据的统计规律更加一致。这个过程不仅是计算密集型的，而且需要通过大量的数据和计算来不断优化网络，直到模型的预测能力达到理想水平。


# neural network internals

networks just to give you a sense of what's inside so neural network internals so as I mentioned we have
these inputs uh that are sequences of tokens in this case this is four input
tokens but this can be anywhere between zero up to let's say 8,000 tokens in
principle this can be an infinite number of tokens we just uh it would just be too computationally expensive to process
an infinite number of tokens so we just crop it at a certain length and that becomes the maximum context length of
that uh model now these inputs X are mixed up in a giant mathematical expression together
with the parameters or the weights of these neural networks so here I'm showing six example parameters and their
setting but in practice these uh um modern neural networks will have
billions of these uh parameters and in the beginning these parameters are completely randomly set now with a
random setting of parameters you might expect that this uh this neural network would make random predictions and it
does in the beginning it's totally random predictions but it's through this process of iteratively updating the
network uh as and we call that process training a neural network so uh that the
setting of these parameters gets adjusted such that the outputs of our neural network becomes consistent with
the patterns seen in our training set so think of these parameters as kind
of like knobs on a DJ set and as you're twiddling these knobs you're getting different uh predictions for every
possible uh token sequence input and training in neural network just means
discovering a setting of parameters that seems to be consistent with the statistics of the training
set now let me just give you an example what this giant mathematical expression looks like just to give you a sense and
modern networks are massive expressions with trillions of terms probably but let me just show you a simple example here
it would look something like this I mean these are the kinds of Expressions just to show you that it's not very scary we
have inputs x uh like X1 x2 in this case two example inputs and they get mixed up
with the weights of the network w0 W1 2 3 Etc and this mixing is simple things
like multiplication addition addition exponentiation division Etc and it is
the subject of neural network architecture research to design effective mathematical Expressions uh
that have a lot of uh kind of convenient characteristics they are expressive they're optimizable they're paralyzable
Etc and so but uh at the end of the day these are these are not complex expressions and basically they mix up
the inputs with the parameters to make predictions and we're optimizing uh the
parameters of this neural network so that the predictions come out consistent with the training set now I would like
to show you an actual production grade example of what these neural networks look like so for that I encourage you to
go to this website that has a very nice visualization of one of these networks so this is what you will find
on this website and this neural network here that is used in production settings
has this special kind of structure this network is called the Transformer and this particular one as an example has 8
5,000 roughly parameters now here on the top we take the inputs which are the token
sequences and then information flows through the neural network until the
output which here are the logit softmax but these are the predictions for what comes next what token comes
next and then here there's a sequence of Transformations and all these
intermediate values that get produced inside this mathematical expression s it is sort of predicting what comes next so
as an example these tokens are embedded into kind of like this distributed representation as it's called so every
possible token has kind of like a vector that represents it inside the neural network so first we embed the tokens and
then those values uh kind of like flow through this diagram and these are all
very simple mathematical Expressions individually so we have layer norms and Matrix multiplications and uh soft Maxes
and so on so here kind of like the attention block of this Transformer and then information kind of flows through
into the multi-layer perceptron block and so on and all these numbers here these are the intermediate values of the
expression and uh you can almost think of these as kind of like the firing rates of these synthetic neurons but I
would caution you to uh not um kind of think of it too much like neurons because these are extremely simple
neurons compared to the neurons you would find in your brain your biological neurons are very complex dynamical
processes that have memory and so on there's no memory in this expression it's a fixed mathematical expression
from input to Output with no memory it's just a stateless so these are very simple neurons in comparison to biological
neurons but you can still kind of loosely think of this as like a synthetic piece of uh brain tissue if
you if you like uh to think about it that way so information flows through
all these neurons fire until we get to the predictions now I'm not actually
going to dwell too much on the precise kind of like mathematical details of all these Transformations honestly I don't
think it's that important to get into what's really important to understand is that this is a mathematical function it
is uh parameterized by some fixed set of parameters like say 85,000 of them and
it is a way of transforming inputs into outputs and as we twiddle the parameters we are getting uh different kinds of
predictions and then we need to find a good setting of these parameters so that the predictions uh sort of match up with
the patterns seen in training set so that's the Transformer okay so I've

在神经网络的内部结构中，输入是由一系列标记（tokens）组成的序列。这些标记可以是从0到最多8,000个标记的任意长度（尽管理论上可以是无限的，但计算上无法处理如此庞大的序列，因此会对输入长度进行限制）。这些输入标记会与神经网络中的**参数**（weights）结合，形成一个巨大的数学表达式。举个例子，现代神经网络中包含的参数可能有数十亿个，训练开始时，这些参数的值是随机设置的。

由于参数最初是随机的，神经网络的输出也是随机的。这时，网络还没有学习到任何规律，预测的结果也没有实际意义。但是通过**训练**，我们会不断调整这些参数，使得网络的输出越来越接近训练数据中的统计模式。可以把这些参数想象成是DJ调音台上的旋钮，调整这些旋钮就会改变神经网络的预测结果。训练神经网络的过程就是调整这些旋钮，使得神经网络的输出（即预测）与训练数据的一致性越来越高。

### **神经网络内部的数学表达式：**

神经网络的计算过程是一个复杂的数学表达式，但它其实并不复杂。以简单的例子为说明，假设输入是`x1`、`x2`，而参数是`w0`、`w1`、`w2`等。输入和参数通过一些数学运算（如加法、乘法、指数运算、除法等）混合在一起。这些操作最终会得到输出，输出是神经网络对于输入的预测结果。

现代神经网络是非常庞大的，包含了成千上万甚至亿万的参数，但它们的核心逻辑是相似的：通过一些数学表达式（如加法、矩阵乘法、激活函数等），将输入转化为输出。在训练过程中，我们不断调整这些参数的值，使得神经网络对训练集中的模式有更好的匹配。

### **Transformer神经网络：**

在生产环境中，常用的神经网络结构是**Transformer**，它具有非常复杂的内部结构，且包含数百万甚至数十亿的参数。Transformer的基本操作是：

1. **输入嵌入：** 首先，将输入的标记序列转换为一个向量表示。每个标记会有一个对应的向量（也叫做嵌入向量），这些向量会被送入神经网络。

2. **信息流动：** 嵌入的标记向量会经过Transformer的多个层，每一层都进行一些计算和变换（如矩阵乘法、激活函数等）。这些计算都是通过简单的数学公式完成的。

3. **注意力机制：** Transformer的一个关键组件是**自注意力机制**（Self-Attention），它允许模型在处理每个标记时，能够考虑到整个输入序列中的其他标记。这样可以使模型捕捉到更复杂的关系。

4. **多层感知机：** 除了注意力机制，Transformer还包含多层感知机（MLP）部分，进一步处理这些信息，生成最终的输出。

5. **输出：** 经过多次计算后，最终会输出预测结果，通常是下一个标记的概率分布。

### **总结：**

神经网络，特别是像Transformer这样的模型，实际上是通过一系列数学变换将输入数据转化为输出。在训练过程中，我们通过调整网络中的参数，逐步优化这些数学变换，使得输出结果越来越准确。每个标记的嵌入向量和通过神经网络流动的各种计算步骤，可以帮助模型捕捉到文本中标记之间的复杂关系。这些计算本质上是通过不断优化神经网络的参数来使模型输出符合训练数据的统计规律。


# inference

shown you the internals of the neural network and we talked a bit about the process of training it I want to cover
one more major stage of working with these networks and that is the stage called inference so in inference what
we're doing is we're generating new data from the model and so uh we want to basically see what kind of patterns it
has internalized in the parameters of its Network so to generate from the
model is relatively straightforward we start with some tokens that are basically your prefix like what you want
to start with so say we want to start with the token 91 well we feed it into the
network and remember that the network gives us probabilities right it gives us
this probability Vector here so what we can do now is we can basically flip a biased coin so um we can sample uh
basically a token based on this probability distribution so the tokens
that are given High probability by the model are more likely to be sampled when you flip this biased coin you can think
of it that way so we sample from the distribution to get a single unique token so for example token 860 comes
next uh so 860 in this case when we're generating from model could come next now 860 is a relatively likely token it
might not be the only possible token in this case there could be many other tokens that could have been sampled but we could see that 86c is a relatively
likely token as an example and indeed in our training examp example here 860 does follow 91 so let's now say that we um
continue the process so after 91 there's a60 we append it and we again ask what
is the third token let's sample and let's just say that it's 287 exactly as
here let's do that again we come back in now we have a sequence of three and we
ask what is the likely fourth token and we sample from that and get this one and
now let's say we do it one more time we take those four we sample and we get this one and this
13659 uh this is not actually uh 3962 as we had before so this token is the token
article uh instead so viewing a single article and so in this case we didn't
exactly reproduce the sequence that we saw here in the training data so keep in mind that these systems are stochastic
they have um we're sampling and we're flipping coins and sometimes we lock out
and we reproduce some like small chunk of the text and training set but sometimes we're uh we're getting a token
that was not verbatim part of any of the documents in the training data so we're going to get sort of like remixes of the
data that we saw in the training because at every step of the way we can flip and get a slightly different token and then
once that token makes it in if you sample the next one and so on you very quickly uh start to generate token
streams that are very different from the token streams that UR in the training documents so
statistically they will have similar properties but um they are not identical to your training data they're kind of
like inspired by the training data and so in this case we got a slightly different sequence and why would we get
article you might imagine that article is a relatively likely token in the context of bar viewing single Etc and
you can imagine that the word article followed this context window somewhere in the training documents uh to some
extent and we just happen to sample it here at that stage so basically inference is just uh predicting from
these distributions one at a time we continue feeding back tokens and getting the next one and we uh we're always
flipping these coins and depending on how lucky or unlucky we get um we might
get very different kinds of patterns depending on how we sample from these probability distributions so that's
inference so in most common scenarios uh basically downloading the internet and
tokenizing it is is a pre-processing step you do that a single time and then uh once you have your token sequence we
can start training networks and in Practical cases you would try to train many different networks of different
kinds of uh settings and different kinds of arrangements and different kinds of sizes and so you''ll be doing a lot of
neural network training and um then once you have a neural network and you train it and you have some specific set of
parameters that you're happy with um then you can take the model and you can do inference and you can actually uh
generate data from the model and when you're on chat GPT and you're talking with a model uh that model is trained
and has been trained by open aai many months ago probably and they have a specific set of Weights that work well
and when you're talking to the model all of that is just inference there's no more training those parameters are held
fixed and you're just talking to the model sort of uh you're giving it some of the tokens and it's kind of
completing token sequences and that's what you're seeing uh generated when you actually use the model on CH GPT so that
model then just does inference alone so let's now look at an example of training an inference that is kind of concrete
and gives you a sense of what this actually looks like uh when these models are trained now the example that I would like to work with and that I'm

在神经网络的**推理**阶段，我们通过模型生成新的数据。简单来说，推理就是通过已训练好的神经网络，根据已有的上下文预测下一个可能的标记（token）。这个过程可以帮助我们了解网络内部已经学习到的模式。

### 推理过程：

1. **初始化输入：** 我们从一些“起始”标记开始，通常是一个“前缀”，比如说开始时选择标记 `91`。

2. **获取概率分布：** 网络根据输入的标记计算出一个**概率分布**，即每个标记作为下一个输出的概率。例如，模型可能会给出标记 `860` 的概率为最高，那么 `860` 就是下一步的预测结果。

3. **采样：** 我们并不总是选择概率最高的标记，而是**随机采样**。模型给出的概率越高的标记，更可能被采样到。这就像是在掷一个有偏的硬币，概率大的标记更容易被选中。

   比如，假设标记 `860` 被采样到作为下一个标记，那么接着我们把 `860` 加入到序列中，再次输入到神经网络中。模型再次计算概率分布并进行采样，直到生成完整的文本。

4. **连续生成：** 继续这个过程，我们每次都从网络获取一个新标记，并将其添加到当前的标记序列中。每生成一个新的标记，就将这个标记作为输入，继续进行下一个标记的生成。

5. **随机性：** 需要注意的是，**推理是随机的**。尽管我们训练模型时会看到一些标记的概率分布，这并不意味着每次生成的内容都会完全一致。因为每次采样可能会选到不同的标记，所以最终生成的文本可能会与训练数据中的内容不同，但统计性质会相似。

例如，如果训练数据中有很多“bar viewing single”这样的短语，模型可能会生成类似“article”这样的标记，因为这个标记在这个上下文中是很可能出现的。

### 推理与训练的区别：

* **训练阶段：** 训练神经网络是为了找到一组合适的参数，这些参数能让网络的输出与训练数据的模式一致。训练是通过反向传播和梯度下降来不断调整网络参数，使得模型能够从大量的训练数据中学到有效的规律。
* **推理阶段：** 推理则是利用已经训练好的网络，根据特定的输入生成新的输出。在推理过程中，模型的参数是固定的，不再更新。你在与像ChatGPT这样的模型对话时，实际上就是在做推理。模型已经训练好，当你输入一些标记后，它会预测下一个标记并生成完整的回答。

### 总结：

推理的过程就是通过固定的网络参数，基于输入的标记序列逐步预测下一个标记，直到生成完整的文本。这个过程是**随机的**，因为我们每次从概率分布中采样，所以生成的结果可能会有所不同，但整体上会遵循数据的统计特性。


# GPT-2: training and inference

particularly fond of is that of opening eyes gpt2 so GPT uh stands for
generatively pre-trained Transformer and this is the second iteration of the GPT series by open AI when you are talking
to chat GPT today the model that is underlying all of the magic of that interaction is GPT 4 so the fourth
iteration of that series now gpt2 was published in 2019 by openi in this paper
that I have right here and the reason I like gpt2 is that it is the first time
that a recognizably modern stack came together so um all of the pieces of gpd2
are recognizable today by modern standards it's just everything has gotten bigger now I'm not going to be
able to go into the full details of this paper of course because it is a technical publication but some of the
details that I would like to highlight are as follows gpt2 was a Transformer neural network just like you were just
like the neural networks you would work with today it was it had 1.6 billion parameters right so these are the
parameters that we looked at here it would have 1.6 billion of them today modern Transformers would have a lot
closer to a trillion or several hundred billion probably the maximum context length here
was 1,24 tokens so it is when we are sampling chunks of Windows of tokens
from the data set we're never taking more than 1,24 tokens and so when you are trying to predict the next token in
a sequence you will never have more than 1,24 tokens uh kind of in your context
in order to make that prediction now this is also tiny by modern standards today the token uh the context lengths
would be a lot closer to um couple hundred thousand or maybe even a million
and so you have a lot more context a lot more tokens in history history and you can make a lot better prediction about
the next token in the sequence in that way and finally gpt2 was trained on approximately 100 billion tokens and
this is also fairly small by modern standards as I mentioned the fine web data set that we looked at here the fine
web data set has 15 trillion tokens uh so 100 billion is is quite small
now uh I actually tried to reproduce uh gpt2 for fun as part of this project called lm. C so you can see my rup of
doing that in this post on GitHub under the lm. C repository so in particular
the cost of training gpd2 in 2019 what was estimated to be approximately
$40,000 but today you can do significantly better than that and in particular here it took about one day
and about $600 uh but this wasn't even trying too hard I think you could really bring this
down to about $100 today now why is it that the costs have come down so much
well number one these data sets have gotten a lot better and the way we filter them extract them and prepare
them has gotten a lot more refined and so the data set is of just a lot higher quality so that's one thing but really
the biggest difference is that our computers have gotten much faster in terms of the hardware and we're going to look at that in a second and also the
software for uh running these models and really squeezing out all all the speed
from the hardware as it is possible uh that software has also gotten much better as as everyone has focused on
these models and try to run them very very quickly now I'm not going to be able to
go into the full detail of this gpd2 reproduction and this is a long technical post but I would like to still
give you an intuitive sense for what it looks like to actually train one of these models as a researcher like what are you looking at and what does it look
like what does it feel like so let me give you a sense of that a little bit okay so this is what it looks like let me slide this
over so what I'm doing here is I'm training a gpt2 model right now
and um what's happening here is that every single line here like this one is
one update to the model so remember how here we are um basically making the
prediction better for every one of these tokens and we are updating these weights or parameters of the neural net so here
every single line is One update to the neural network where we change its parameters by a little bit so that it is
better at predicting next token and sequence in particular every single line here is improving the prediction on 1
million tokens in the training set so we've basically taken 1 million tokens
out of this data set and we've tried to improve the prediction of that token as
coming next in a sequence on all 1 million of them simultaneously and at every single one
of these steps we are making an update to the network for that now the number to watch closely is this number called
loss and the loss is a single number that is telling you how well your neural network is performing right now and it
is created so that low loss is good so you'll see that the loss is decreasing
as we make more updates to the neural nut which corresponds to making better predictions on the next token in a
sequence and so the loss is the number that you are watching as a neural network researcher and you are kind of
waiting you're twiddling your thumbs uh you're drinking coffee and you're making sure that this looks good so that with
every update your loss is improving and the network is getting better at prediction now here you see that we are
processing 1 million tokens per update each update takes about 7 Seconds
roughly and here we are going to process a total of 32,000 steps of
optimization so 32,000 steps with 1 million tokens each is about 33 billion
tokens that we are going to process and we're currently only about 420 step 20
out of 32,000 so we are still only a bit more than 1% done because I've only been
running this for 10 or 15 minutes or something like that now every 20 steps I have
configured this optimization to do inference so what you're seeing here is the model is predicting the next token
in a sequence and so you sort of start it randomly and then you continue plugging in the tokens so we're running
this inference step and this is the model sort of predicting the next token in the sequence and every time you see something appear that's a new
token um so let's just look at this and
you can see that this is not yet very coherent and keep in mind that this is only 1% of the way through training and
so the model is not yet very good at predicting the next token in the sequence so what comes out is actually
kind of a little bit of gibberish right but it still has a little bit of like local coherence so since she is mine
it's a part of the information should discuss my father great companions Gordon showed me sitting over at and Etc
so I know it doesn't look very good but let's actually scroll up and see what it
looked like when I started the optimization so all the way here at
step one so after 20 steps of optimization you see that what we're getting here is
looks completely random and of course that's because the model has only had 20 updates to its parameters and so it's
giving you random text because it's a random Network and so you can see that at least in comparison to this model is
starting to do much better and indeed if we waited the entire 32,000 steps the model will have improved the point that
it's actually uh generating fairly coherent English uh and the tokens stream correctly um and uh they they
kind of make up English a a lot better um so this has to run for about a day or
two more now and so uh at this stage we just make sure that the loss is decreasing everything is looking good um
and we just have to wait and now um let me turn now to the um
story of the computation that's required because of course I'm not running this optimization on my laptop that would be
way too expensive uh because we have to run this neural network and we have to improve it and we have we need all this
data and so on so you can't run this too well on your computer uh because the network is just too large uh so all of
this is running on the computer that is out there in the cloud and I want to basically address the compute side of
the store of training these models and what that looks like so let's take a look okay so the computer that I'm
running this optimization on is this 8X h100 node so there are eight h100s in a
single node or a single computer now I am renting this computer and it is somewhere in the cloud I'm not sure
where it is physically actually the place I like to rent from is called Lambda but there are many other companies who provide this service so
when you scroll down you can see that uh they have some on demand pricing for
um sort of computers that have these uh h100s which are gpus and I'm going to
show you what they look like in a second but on demand 8times Nvidia h100 uh
GPU this machine comes for $3 per GPU per hour for example so you can rent
these and then you get a machine in a cloud and you can uh go in and you can train these models and these uh gpus they look like
this so this is one h100 GPU uh this is kind of what it looks like and you slot this into your computer and gpus are
this uh perfect fit for training your networks because they are very computationally expensive but they
display a lot of parallelism in the computation so you can have many independent workers kind of um working
all at the same time in solving uh the matrix multiplication that's under the
hood of training these neural networks so this is just one of these h100s but actually you would put them
you would put multiple of them together so you could stack eight of them into a single node and then you can stack
multiple nodes into an entire data center or an entire system so when we look at a data
center can't spell when we look at a data center we start to see things that look like this right so we have one GPU
goes to eight gpus goes to a single system goes to many systems and so these are the bigger data centers and there of
course would be much much more expensive um and what's happening is that all the big tech companies really desire these
gpus so they can train all these language models because they are so powerful and that has is fundamentally
what has driven the stock price of Nvidia to be $3.4 trillion today as an example and why Nvidia has kind of
exploded so this is the Gold Rush the Gold Rush is getting the gpus getting
enough of them so they can all collaborate to perform this optimization
and they're what are they all doing they're all collaborating to predict the next token on a data set like the fine
web data set this is the computational workflow that that basically is extremely
expensive the more gpus you have the more tokens you can try to predict and improve on and you're going to process
this data set faster and you can iterate faster and get a bigger Network and train a bigger Network and so on so this
is what all those machines are look like are uh are doing and this is why all of
this is such a big deal and for example this is a article from like about a month ago or
so this is why it's a big deal that for example Elon Musk is getting 100,000 gpus uh in a single Data Center and all
of these gpus are extremely expensive are going to take a ton of power and all of them are just trying to predict the next token in the sequence and improve
the network uh by doing so and uh get probably a lot more coherent text than
what we're seeing here a lot faster okay so unfortunately I do not have a couple 10 or hundred million of dollars to

GPT-2是OpenAI在2019年发布的第二代“生成预训练变换器”（Generative Pre-trained Transformer，简称GPT）。与现代GPT模型相比，GPT-2的参数较少，但它是首次实现现代Transformer架构并取得显著成功的里程碑。

### GPT-2的基本特点：

1. **Transformer架构**：GPT-2采用了Transformer神经网络架构，这种架构是现代自然语言处理（NLP）任务中广泛使用的基础。它通过自注意力机制（self-attention）来捕捉文本中的长程依赖关系。

2. **1.6亿个参数**：GPT-2拥有1.6亿个参数，这些参数会随着训练逐步优化。现代的GPT模型参数量大约为数百亿甚至数万亿，GPT-4的参数量就达到了几百亿。

3. **最大上下文长度**：GPT-2的上下文长度为1024个标记（tokens），也就是说，在预测下一个标记时，模型最多会考虑1024个标记的上下文信息。而现代的GPT模型通常可以处理几万个甚至更多的标记。

4. **训练数据**：GPT-2在1000亿个标记的数据集上进行训练，而现在的模型训练可能使用更多的数据。例如，GPT-4可能会使用超过15万亿个标记的数据集。

### GPT-2的训练过程：

* **优化步骤**：训练过程中，模型每处理1百万个标记，就进行一次优化。每一次优化都会使得模型的损失（loss）值下降，说明模型对下一个标记的预测能力正在提升。每次优化需要花费大约7秒钟，经过32,000次优化步骤，GPT-2会处理大约330亿个标记。

* **损失函数**：损失函数是用来衡量模型表现的指标，损失越低，模型的表现就越好。在训练过程中，研究人员通过观察损失值的变化来评估模型的学习进度。

* **模型的输出**：在训练过程中，模型会定期进行推理，生成一些预测结果。这些结果开始时可能是毫无意义的，但随着训练的推进，模型会逐渐学会生成更连贯的文本。

### 计算资源：

由于GPT-2模型较大，训练它需要巨大的计算资源。如今，这样的模型一般是在高性能GPU集群上进行训练的，例如NVIDIA的H100 GPU。一个带有8块H100 GPU的节点租赁价格为每小时3美元。为了训练更大的模型，通常需要更多的GPU节点，它们集成到一个数据中心中。

### 云计算和硬件的重要性：

* **云计算服务**：由于GPT-2等大型模型的训练非常依赖计算资源，因此很多公司会租用云服务中的高性能GPU来进行训练。比如，Elon Musk就计划租用100,000个GPU来加速训练过程。

* **NVIDIA GPU**：NVIDIA的GPU因其出色的并行计算能力而成为训练深度学习模型的首选硬件。多个GPU可以并行工作，大大提高了模型训练的速度和效率。

### 总结：

GPT-2是一个革命性的语言模型，虽然其规模相对较小，但它的成功为后来的大规模语言模型奠定了基础。如今的训练过程不仅依赖于高效的算法和数据集，还依赖于强大的硬件支持，尤其是GPU。随着硬件和软件的不断进步，训练大型语言模型的成本大幅下降，使得更多公司和研究者能够参与到这一领域。

# Llama 3.1 base model inference

spend on training a really big model like this but luckily we can turn to some big tech companies who train these
models routinely and release some of them once they are done training so they've spent a huge amount of compute
to train this network and they release the network at the end of the optimization so it's very useful because
they've done a lot of compute for that so there are many companies who train these models routinely but actually not
many of them release uh these what's called base models so the model that comes out at the end here is is what's
called a base model what is a base model it's a token simulator right it's an internet text token simulator and so
that is not by itself useful yet because what we want is what's called an assistant we want to ask questions and
have it respond to answers these models won't do that they just uh create sort of remixes of the internet they dream
internet pages so the base models are not very often released because they're kind of just only a step one of a few
other steps that we still need to take to get in system however a few releases have been made so
as an example the gbt2 model released the 1.6 billion sorry 1.5 billion model
back in 2019 and this gpt2 model is a base model now what is a model release
what does it look like to release these models so this is the gpt2 repository on GitHub well you need two things
basically to release model number one we need the um python code usually that
describes the sequence of operations in detail that they make in their model so
um if you remember back this Transformer the sequence of steps that
are taken here in this neural network is what is being described by this code so
this code is sort of implementing the what's called forward pass of this neural network so we need the specific
details of exactly how they wired up that neural network so this is just computer code and it's usually just a
couple hundred lines of code it's not it's not that crazy and uh this is all fairly understandable and usually fairly
standard what's not standard are the parameters that's where the actual value is what are the parameters of this
neural network because there's 1.6 billion of them and we need the correct setting or a really good setting and so
that's why in addition to this source code they release the parameters which in this case is roughly 1.5 billion
parameters and these are just numbers so it's one single list of 1.5 billion numbers the precise and good setting of
all the knobs such that the tokens come out well so uh you need those two things to
get a base model release now gpt2 was released but that's
actually a fairly old model as I mentioned so actually the model we're going to turn to is called llama 3 and
that's the one that I would like to show you next so llama 3 so gpt2 again was
1.6 billion parameters trained on 100 billion tokens Lama 3 is a much bigger model and much more modern model it is
released and trained by meta and it is a 45 billion parameter model trained on 15
trillion tokens in very much the same way just much much bigger um and meta has also made a
release of llama 3 and that was part of this paper so with this paper that goes into
a lot of detail the biggest base model that they released is the Lama 3.1 4.5
405 billion parameter model so this is the base model and then in addition to the base model you see here
foreshadowing for later sections of the video they also released the instruct model and the instruct means that this
is an assistant you can ask it questions and it will give you answers we still have yet to cover that part later for
now let's just look at this base model this token simulator and let's play with it and try to think about you know what
is this thing and how does it work and um what do we get at the end of this optimization if you let this run Until
the End uh for a very big neural network on a lot of data so my favorite place to interact with the base models is this um
company called hyperbolic which is basically serving the base model of the 405b Llama 3.1 so when you go to the
website and I think you may have to register and so on make sure that in the models make sure that you are using
llama 3.1 405 billion base it must be the base model and then here let's say
the max tokens is how many tokens we're going to be gener rating so let's just decrease this to be a bit less just so
we don't waste compute we just want the next 128 tokens and leave the other stuff alone I'm not going to go into the
full detail here um now fundamentally what's going to happen here is identical
to what happens here during inference for us so this is just going to continue the token sequence of whatever you
prefix you're going to give it so I want to first show you that this model here is not yet an assistant so you can for
example ask it what is 2 plus 2 it's not going to tell you oh it's four uh what else can I help you with it's not going
to do that because what is 2 plus 2 is going to be tokenized and then those
tokens just act as a prefix and then what the model is going to do now is just going to get the probability for the next token and it's just a glorified
autocomplete it's a very very expensive autocomplete of what comes next um
depending on the statistics of what it saw in its training documents which are basically web pages so let's just uh hit enter to see
what tokens it comes up with as a
continuation okay so here it kind of actually answered the question and started to go off into some philosophical territory uh let's try it
again so let me copy and paste and let's try again from scratch what is 2 plus
two so okay so it just goes off again so notice one more thing that I want to
stress is that the system uh I think every time you put it in it just kind of starts from scratch
so it doesn't uh the system here is stochastic so for the same prefix of tokens we're always getting a different
answer and the reason for that is that we get this probity distribution and we sample from it and we always get
different samples and we sort of always go into a different territory uh afterwards so here in this case um I
don't know what this is let's try one more time so it just continues on so it's
just doing the stuff that it's saw on the internet right um and it's just kind of like regurgitating those uh
statistical patterns so first things it's not an assistant yet it's a token autocomplete
and second it is a stochastic system now the crucial thing is that even though
this model is not yet by itself very useful for a lot of applications just yet um it is still very useful because
in the task of predicting the next token in the sequence the model has learned a lot about the world and it has stored
all that knowledge in the parameters of the network so remember that our text looked like this right internet web
pages and now all of this is sort of compressed in the weights of the network
so you can think of um these 405 billion parameters is a kind of compression of
the internet you can think of the 45 billion parameters is kind of like a zip file uh but it's not a loss less
compression it's a loss C compression we're kind of like left with kind of a gal of the internet and we can generate
from it right now we can elicit some of this knowledge by prompting the base model uh accordingly so for example
here's a prompt that might work to elicit some of that knowledge that's hiding in the parameters here's my top
10 list of the top landmarks to see in the pairs
um and I'm doing it this way because I'm trying to Prime the model to now continue this list so let's see if that
works when I press enter okay so you see that it started a list and it's now kind of giving me some
of those landmarks and now notice that it's trying to give a lot of information here now you might not be able to actually
fully trust some of the information here remember that this is all just a recollection of some of the internet
documents and so the things that occur very frequently in the internet data are
probably more likely to be remembered correctly compared to things that happen very infrequently so you can't fully
trust some of the things that and some of the information that is here because it's all just a vague recollection of Internet documents because the
information is not stored explicitly in any of the parameters it's all just the recollection that said we did get
something that is probably approximately correct and I don't actually have the expertise to verify that this is roughly
correct but you see that we've elicited a lot of the knowledge of the model and this knowledge is not precise and exact
this knowledge is vague and probabilistic and statistical and the kinds of things that occur often are the
kinds of things that are more likely to be remembered um in the model now I want to show you a few more examples of this
model's Behavior the first thing I want to show you is this example I went to the Wikipedia page for zebra and let me
just copy paste the first uh even one sentence here and let me put it here now when I
click enter what kind of uh completion are we going to get so let me just hit
enter there are three living species etc etc what the model is producing here
is an exact regurgitation of this Wikipedia entry it is reciting this Wikipedia entry purely from memory and
this memory is stored in its parameters and so it is possible that at some point in these 512 tokens the model will uh
stray away from the Wikipedia entry but you can see that it has huge chunks of it memorized here uh let me see for
example if this sentence occurs by now okay so this so we're
still on track let me check here okay we're still on
track it will eventually uh stray away okay so this thing is just recited
to a very large extent it will eventually deviate uh because it won't be able to remember exactly now the
reason that this happens is because these models can be extremely good at memorization and usually this is not
what you want in the final model and this is something called regurgitation and it's usually undesirable to site uh
things uh directly uh that you have trained on now the reason that this happens actually is because for a lot of
documents like for example Wikipedia when these documents are deemed to be of very high quality as a source like for
example Wikipedia it is very often uh the case that when you train the model you will preferentially sample from
those sources so basically the model has probably done a few epochs on this data meaning that it has seen this web page
like maybe probably 10 times or so and it's a bit like you like when you read some kind of a text many many times say
you read something a 100 times uh then you'll be able to recite it and it's very similar for this model if it sees
something way too often it's going to be able to recite it later from memory except these models can be a lot more
efficient um like per presentation than human so probably it's only seen this Wikipedia entry 10 times but basically
it has remembered this article exactly in its parameters okay the next thing I want to show you is something that the
model has definitely not seen during its training so for example if we go to the paper uh and then we navigate to the
pre-training data we'll see here that uh the data set has a knowledge cut off
until the end of 2023 so it will not have seen documents after this point and
certainly it has not seen anything about the 2024 election and how it turned out now if we Prime the model with the
tokens from the future it will continue the token sequence and it will just take its best guess according to the
knowledge that it has in its own parameters so let's take a look at what that could look like
so the Republican Party kit Trump okay president of the United States from
2017 and let's see what it says after this point so for example the model will have to guess at the running mate and
who it's against Etc so let's hit enter so here thingss that Mike Pence was the running mate instead of JD Vance
and the ticket was against Hillary Clinton and Tim Kane so this is kind of
a interesting parallel universe potentially of what could have happened happened according to the LM let's get a
different sample so the identical prompt and let's resample so here the running mate was
Ronda santis and they ran against Joe Biden and Camala Harris so this is again
a different parallel universe so the model will take educated guesses and it will continue the token sequence based
on this knowledge um and it will just kind of like all of what we're seeing here is what's called hallucination the
model is just taking its best guess uh in a probalistic manner the next thing I would like to show you is that even
though this is a base model and not yet an assistant model it can still be utilized in Practical applications if
you are clever with your prompt design so here's something that we would call a few shot
prompt so what it is here is that I have 10 words or 10 pairs and each pair is a
word of English column and then a the translation in Korean and we have 10 of
them and what the model does here is at the end we have teacher column and then here's where we're going to do a
completion of say just five tokens and these models have what we call in
context learning abilities and what that's referring to is that as it is reading this context it is learning sort
of in place that there's some kind of a algorithmic pattern going on in my data
and it knows to continue that pattern and this is called kind of like Inc context learning so it takes on the role
of a translator and when we hit uh completion we see that the teacher translation is
Sim which is correct um and so this is how you can build apps by being clever
with your prompting even though we still just have a base model for now and it relies on what we call this um uh in
context learning ability and it is done by constructing what's called a few shot prompt okay and finally I want to show
you that there is a clever way to actually instantiate a whole language model assistant just by prompting and
the trick to it is that we're structure a prompt to look like a web page that is a conversation between a helpful AI
assistant and a human and then the model will continue that conversation so actually to write the prompt I turned to
chat gbt itself which is kind of meta but I told it I want to create an llm
assistant but all I have is the base model so can you please write my um uh
prompt and this is what it came up with which is actually quite good so here's a conversation between an AI assistant and
a human the AI assistant is knowledgeable helpful capable of answering wide variety of questions Etc and then here
it's not enough to just give it a sort of description it works much better if you create this fot prompt so here's a
few terms of human assistant human assistant and we have uh you know a few
turns of conversation and then here at the end is we're going to be putting the actual query that we like so let me copy
paste this into the base model prompt and now let me do human column and this
is where we put our actual prompt why is the sky blue and uh let's uh
run assistant the sky appears blue due to the phenomenon called R lights scattering etc etc so you see that the
base model is just continuing the sequence but because the sequence looks like this conversation it takes on that
role but it is a little subtle because here it just uh you know it ends the assistant and then just you know
hallucinate Ates the next question by the human Etc so it'll just continue going on and on uh but you can see that
we have sort of accomplished the task and if you just took this why is the sky blue and if we just refresh this and put
it here then of course we don't expect this to work with a base model right we're just going to who knows what we're going to get okay we're just going to
get more questions okay so this is one way to create an assistant even though you may
only have a base model okay so this is the kind of brief summary of the things we talked about over the last few

# pretraining to post-training

minutes now let me zoom out here and this is kind of like what we've
talked about so far we wish to train LM assistants like chpt we've discussed the
first stage of that which is the pre-training stage and we saw that really what it comes down to is we take Internet documents we break them up into
these tokens these atoms of little text chunks and then we predict token sequences using neural networks the
output of this entire stage is this base model it is the setting of The parameters of this network and this base
model is basically an internet document simulator on the token level so it can just uh it can generate token sequences
that have the same kind of like statistics as Internet documents and we saw that we can use it in some
applications but we actually need to do better we want an assistant we want to be able to ask questions and we want the
model to give us answers and so we need to now go into the second stage which is called the post-training stage so we
take our base model our internet document simulator and hand it off to post training so we're now going to
discuss a few ways to do what's called post training of these models these stages in post training are going to be
computationally much less expensive most of the computational work all of the massive data centers um and all of the
sort of heavy compute and millions of dollars are the pre-training stage but
now we go into the slightly cheaper but still extremely important stage called post trining where we turn this llm
model into an assistant so let's take a look at how we can get our model to not
sample internet documents but to give answers to questions so in other words

在这部分中，我们讨论了从 **预训练阶段** 到 **后训练阶段** 的转变。这个过程是将语言模型从一个简单的 **文档生成器** 转变为一个能够回答问题的 **助手模型**。以下是具体的步骤和解释：

### 1. **预训练阶段**：

* 在预训练阶段，模型通过 **大量的互联网文档** 进行训练。这些文档被切分成 **token**（标记），即文本的最小单元，然后模型使用神经网络来预测这些标记的序列。
* 通过这种方式，模型学会了如何生成符合互联网文档统计特征的文本。它的输出是一个 **基础模型**，这个基础模型能够生成与互联网文档相似的 **标记序列**。

**基础模型的功能**：

* 基础模型仅仅是一个 **互联网文档模拟器**，它的任务是生成文本序列，而不是直接回答问题。它可以生成类似于互联网上文档的文本，但并不具备执行具体任务的能力。

### 2. **后训练阶段（Post-training）**：

* 基础模型完成预训练后，我们进入 **后训练阶段**。后训练阶段的目标是将这个基础模型转变为一个 **助手模型**，可以回答问题并提供相关信息。
* 后训练的计算开销相对较低，**不像预训练阶段那样需要大量的数据中心计算和高昂的成本**，但它依然非常重要。

### 3. **后训练的目的**：

* 在预训练中，模型只是学会了如何生成符合互联网数据统计特征的文本，而 **后训练的目的是让模型能够回答问题**，从而变得更有用。例如，当用户向模型提问时，它应该能够提供有意义的回答，而不仅仅是继续生成随机文本。

### 总结：

* **预训练阶段**：通过大量的互联网数据，训练模型学会预测标记的序列，生成符合互联网统计特征的文本。这是一个计算密集型且成本高昂的阶段，输出的是基础模型。
* **后训练阶段**：基础模型通过后训练变成一个能够执行特定任务（如问答）的助手。后训练的计算成本较低，但它对模型的实际应用至关重要。


# post-training data (conversations)

what we want to do is we want to start thinking about conversations and these are conversations that can be multi-turn
so so uh there can be multiple turns and they are in the simplest case a conversation between a human and an
assistant and so for example we can imagine the conversation could look something like this when a human says what is 2 plus2 the assistant should re
respond with something like 2 plus 2 is 4 when a human follows up and says what if it was star instead of a plus
assistant could respond with something like this um and similar here this is another example showing that the assistant could
also have some kind of a personality here uh that it's kind of like nice and then here in the third example I'm
showing that when a human is asking for something that we uh don't wish to help with we can produce what's called
refusal we can say that we cannot help with that so in other words what we want to do now is we want to think through
how in a system should interact with the human and we want to program the assistant and Its Behavior in these
conversations now because this is neural networks we're not going to be programming these explicitly in code
we're not going to be able to program the assistant in that way because this is neural networks everything is done through neural network training on data
sets and so because of that we are going to be implicitly programming the
assistant by creating data sets of conversations so these are three independent examples of conversations in
a data dat set an actual data set and I'm going to show you examples will be much larger it could have hundreds of
thousands of conversations that are multi- turn very long Etc and would cover a diverse breath of topics but
here I'm only showing three examples but the way this works basically is uh a
assistant is being programmed by example and where is this data coming from like 2 * 2al 4 same as 2 plus 2 Etc where
does that come from this comes from Human labelers so we will basically give human labelers some conversational
context and we will ask them to um basically give the ideal assistant response in this situation and a human
will write out the ideal response for an assistant in any situation and then we're going to get the model to
basically train on this and to imitate those kinds of responses so the way this works then is
we are going to take our base model which we produced in the preing stage and this base model was trained on
internet documents we're now going to take that data set of internet documents and we're gonna throw it out and we're
going to substitute a new data set and that's going to be a data set of conversations and we're going to continue training the model on these
conversations on this new data set of conversations and what happens is that the model will very rapidly adjust and
will sort of like learn the statistics of how this assistant responds to human queries and then later during inference
we'll be able to basically um Prime the assistant and get the response and it
will be imitating what the humans will human labelers would do in that situation if that makes sense so we're
going to see examples of that and this is going to become bit more concrete I also wanted to mention that this post-training stage we're going to
basically just continue training the model but um the pre-training stage can in practice take roughly three months of
training on many thousands of computers the post-training stage will typically be much shorter like 3 hours for example
um and that's because the data set of conversations that we're going to create here manually is much much smaller than
the data set of text on the internet and so this training will be very short but
fundamentally we're just going to take our base model we're going to continue training using the exact same algorithm
the exact same everything except we're swapping out the data set for conversations so the questions now are
what are these conversations how do we represent them how do we get the model to see conversations instead of just raw
text and then what are the outcomes of um this kind of training and what do you
get in a certain like psychological sense uh when we talk about the model so let's turn to those questions now so
let's start by talking about the tokenization of conversations everything in these models has to be turned into
tokens because everything is just about token sequences so how do we turn conversations into token sequences is
the question and so for that we need to design some kind of ending coding and uh this is kind of similar to maybe if
you're familiar you don't have to be with for example the TCP IP packet in um on the internet there are precise rules
and protocols for how you represent information how everything is structured together so that you have all this kind of data laid out in a way that is
written out on a paper and that everyone can agree on and so it's the same thing now happening in llms we need some kind
of data structures and we need to have some rules around how these data structures like conversations get
encoded and decoded to and from tokens and so I want to show you now how I
would recreate uh this conversation in the token space so if you go to Tech
tokenizer I can take that conversation and this is how it is represented in uh for the
language model so here we have we are iterating a user and an assistant in
this two- turn conversation and what you're seeing here is it looks ugly but it's actually
relatively simple the way it gets turned into a token sequence here at the end is a little bit complicated but at the end
this conversation between a user and assistant ends up being 49 tokens it is a one-dimensional sequence of 49 tokens
and these are the tokens okay and all the different llms will have a slightly different format or
protocols and it's a little bit of a wild west right now but for example GPT 40 does it in the following way you have
this special token called imore start and this is short for IM imaginary
monologue uh the start then you have to specify um I
don't actually know why it's called that to be honest then you have to specify whose turn it is so for example user
which is a token 4 28 then you have internal monologue
separator and then it's the exact question so the tokens of the question
and then you have to close it so I am end the end of the imaginary monologue so
basically the question from a user of what is 2 plus two ends up being the
token sequence of these tokens and now the important thing to mention here is that IM start this is not text right IM
start is a special token that gets added it's a new token and um this token has
never been trained on so far it is a new token that we create in a post-training stage and we introduce and so these
special tokens like IM seep IM start Etc are introduced and interspersed with
text so that they sort of um get the model to learn that hey this is a the start of a turn for who is it start of
the turn for the start of the turn is for the user and then this is what the user says and then the user ends and
then it's a new start of a turn and it is by the assistant and then what does the assistant say well these are the
tokens of what the assistant says Etc and so this conversation is not turned into the sequence of tokens the specific
details here are not actually that important all I'm trying to show you in concrete terms is that our conversations
which we think of as kind of like a structured object end up being turned via some encoding into onedimensional
sequences of tokens and so because this is one dimensional sequence of tokens we
can apply all the stuff that we applied before now it's just a sequence of tokens and now we can train a language
model on it and so we're just predicting the next token in a sequence uh just like before and um we can represent and
train on conversations and then what does it look like at test time during inference so say we've trained a model
and we've trained a model on these kinds of data sets of conversations and now we want to
inference so during inference what does this look like when you're on on chash apt well you come to chash apt and you
have say like a dialogue with it and the way this works is basically um say that this was already
filled in so like what is 2 plus 2 2 plus 2 is four and now you issue what if it was times I am end and what basically
ends up happening um on the servers of open AI or something like that is they put in I start assistant I amep and this
is where they end it right here so they construct this context and now they
start sampling from the model so it's at this stage that they will go to the model and say okay what is a good for
sequence what is a good first token what is a good second token what is a good third token and this is where the LM
takes over and creates a response like for example response that looks
something like this but it doesn't have to be identical to this but it will have the flavor of this if this kind of a
conversation was in the data set so um that's roughly how the protocol Works
although the details of this protocol are not important so again my goal is that just to show you that everything
ends up being just a one-dimensional token sequence so we can apply everything we've already seen but we're
now training on conversations and we're now uh basically generating conversations as well okay so now I
would like to turn to what these data sets look like in practice the first paper that I would like to show you and the first effort in this direction is
this paper from openai in 2022 and this paper was called instruct GPT or the
technique that they developed and this was the first time that opena has kind of talked about how you can take language models and fine-tune them on
conversations and so this paper has a number of details that I would like to take you through so the first stop I
would like to make is in section 3.4 where they talk about the human contractors that they hired uh in this
case from upwork or through scale AI to uh construct these conversations and so
there are human labelers involved whose job it is professionally to create these conversations and these labelers are
asked to come up with prompts and then they are asked to also complete the ideal assistant responses and so these
are the kinds of prompts that people came up with so these are human labelers so list five ideas for how to regain
enthusiasm for my career what are the top 10 science fiction books I should read next and there's many different
types of uh kind of prompts here so translate this sentence from uh to Spanish Etc and so there's many things
here that people came up with they first come up with the prompt and then they also uh answer that prompt and they give
the ideal assistant response now how do they know what is the ideal assistant response that they should write for
these prompts so when we scroll down a little bit further we see that here we have this excerpt of labeling
instructions uh that are given to the human labelers so the company that is developing the language model like for
example open AI writes up labeling instructions for how the humans should create ideal responses and so here for
example is an excerpt uh of these kinds of labeling instruction instructions on High level you're asking people to be
helpful truthful and harmless and you can pause the video if you'd like to see more here but on a high level basically
just just answer try to be helpful try to be truthful and don't answer questions that we don't want um kind of
the system to handle uh later in chat gbt and so roughly speaking the company
comes up with the labeling instructions usually they are not this short usually there are hundreds of pages and people have to study them professionally and
then they write out the ideal assistant responses uh following those labeling instructions so this is a very human
heavy process as it was described in this paper now the data set for instruct GPT was never actually released by openi
but we do have some open- Source um reproductions that were're trying to follow this kind of a setup and collect
their own data so one that I'm familiar with for example is the effort of open
Assistant from a while back and this is just one of I think many examples but I just want to show you an example so
here's so these were people on the internet that were asked to basically create these conversations similar to
what um open I did with human labelers and so here's an entry of a person who
came up with this BR can you write a short introduction to the relevance of the term manop uh in economics please use
examples Etc and then the same person or potentially a different person will write up the response so here's the
assistant response to this and so then the same person or different person will actually write out this ideal
response and then this is an example of maybe how the conversation could continue now explain it to a dog and
then you can try to come up with a slightly a simpler explanation or something like that now this then
becomes the label and we end up training on this so what happens during training
is that um of course we're not going to have a full coverage of all the possible
questions that um the model will encounter at test time during inference
we can't possibly cover all the possible prompts that people are going to be asking in the future but if we have a
like a data set of a few of these examples then the model during training will start to take on this Persona of
this helpful truthful harmless assistant and it's all programmed by example and
so these are all examples of behavior and if you have conversations of these example behaviors and you have enough of
them like 100,00 and you train on it the model sort of starts to understand the statistical pattern and it kind of takes
on this personality of this assistant now it's possible that when you get the exact same question like
this at test time it's possible that the answer will be recited as exactly what
was in the training set but more likely than that is that the model will kind of like do something of a similar Vibe um
and we will understand that this is the kind of answer that you want um so
that's what we're doing we're programming the system um by example and the system adopts statistically this
Persona of this helpful truthful harmless assistant which is kind of like reflected in the labeling instructions
that the company creates now I want to show you that the state-of-the-art has kind of advanced in the last 2 or 3
years uh since the instr GPT paper so in particular it's not very common for humans to be doing all the heavy lifting
just by themselves anymore and that's because we now have language models and these language models are helping us create these data sets and conversations
so it is very rare that the people will like literally just write out the response from scratch it is a lot more
likely that they will use an existing llm to basically like uh come up with an answer and then they will edit it or
things like that so there's many different ways in which now llms have started to kind of permeate this
posttraining Set uh stack and llms are basically used pervasively to help
create these massive data sets of conversations so I don't want to show like Ultra chat is one um such example
of like a more modern data set of conversations it is to a very large extent synthetic but uh I believe
there's some human involvement I could be wrong with that usually there will be a little bit of human but there will be a huge amount of synthetic help um and
this is all kind of like uh constructed in different ways and Ultra chat is just one example of many sft data sets that
currently exist and the only thing I want to show you is that uh these data sets have now millions of conversations
uh these conversations are mostly synthetic but they're probably edited to some extent by humans and they span a
huge diversity of sort of um uh areas and so on so these are
fairly extensive artifacts by now and there's all these like sft mixtures as they're called so you have a mixture of
like lots of different types and sources and it's partially synthetic partially human and it's kind of like um gone in
that direction since uh but roughly speaking we still have sft data sets they're made up of conversations we're
training on them um just like we did before and uh I guess like the last thing to note
is that I want to dispel a little bit of the magic of talking to an AI like when
you go to chat GPT and you give it a question and then you hit enter uh what
is coming back is kind of like statistically aligned with what's happening in the training set and these
training sets I mean they really just have a seed in humans following labeling
instructions so what are you actually talking to in chat GPT or how should you think about it well it's not coming from
some magical AI like roughly speaking it's coming from something that is statistically imitating human labelers
which comes from labeling instructions written by these companies and so you're kind of imitating this uh you're kind of
getting um it's almost as if you're asking human labeler and imagine that the answer that is given to you uh from
chbt is some kind of a simulation of a human labeler uh and it's kind of like
asking what would a human labeler say in this kind of a conversation
and uh it's not just like this human labeler is not just like a random person from the internet because these
companies actually hire experts so for example when you are asking questions about code and so on the human labelers
that would be in um involved in creation of these conversation data sets they will usually be usually be educated
expert people and you're kind of like asking a question of like a simulation of those people if that makes sense so
you're not talking to a magical AI you're talking to an average labeler this average labeler is probably fairly highly skilled
but you're talking to kind of like an instantaneous simulation of that kind of a person that would be hired uh in the
construction of these data sets so let me give you one more specific example before we move on for example when I go
to chpt and I say recommend the top five landmarks who see in Paris and then I hit
enter uh okay here we go okay when I hit enter
what's coming out here how do I think about it well it's not some kind of a magical AI that has gone out and
researched all the landmarks and then ranked them using its infinite intelligence Etc what I'm getting is a
statistical simulation of a labeler that was hired by open AI you can think about it roughly in that way and so if this
specific um question is in the posttraining data set somewhere at open aai then I'm very likely to see an
answer that is probably very very similar to what that human labeler would have put down for those five landmarks how does the
human labeler come up with this well they go off and they go on the internet and they kind of do their own little research for 20 minutes and they just
come up with a list right now so if they come up with this list and this is in the data set I'm probably very likely to
see what they submitted as the correct answer from the assistant now if this
specific query is not part of the post training data set then what I'm getting here is a little bit more emergent uh
because uh the model kind of understands the statistically um the kinds of landmarks that are in
this training set are usually the prominent landmarks the landmarks that people usually want to see the kinds of
landmarks that are usually uh very often talked about on the internet and remember that the model already has a
ton of Knowledge from its pre-training on the internet so it's probably seen a ton of conversations about Paris about
landmarks about the kinds of things that people like to see and so it's the pre-training knowledge that has then combined with the postering data set
that results in this kind of an imitation um so that's uh that's roughly how you can
kind of think about what's happening behind the scenes here in in this statistical sense okay now I want to

在这一部分中，我们讨论了语言模型从 **基础模型（Base Model）** 训练到 **后训练模型（Post-Training Model）** 的过程，尤其是通过 **对话数据集** 来对模型进行进一步训练的细节。以下是对这个过程的总结和解释：

### 1. **对话数据集的构建**：

* 我们希望模型不仅仅是生成类似互联网文档的文本，而是能够理解并回答具体的 **多轮对话**。这种对话可以是 **人类与助手** 之间的互动。
* 为了实现这一目标，我们需要用 **对话数据集** 来训练模型。这些数据集中的对话由 **人工标注员（Labelers）** 创建。标注员会根据某些对话上下文，编写理想的助手回答。
* 例如，当人类问 "2 + 2 是多少？" 时，助手应该回答 "2 + 2 是 4"。如果人类接着问 "如果是乘法而不是加法呢？" 那么助手应该提供相应的回答。

### 2. **从基础模型到后训练的转变**：

* 在 **预训练阶段**，模型通过大量的互联网文档进行训练，学习了语言的基本统计规律，生成文本。但它并不具备回答问题的能力。
* 在 **后训练阶段**，我们将基础模型从互联网文档数据集转到对话数据集，并继续训练。这个过程虽然计算量较小（通常只需要几小时，而不是几个月），但对模型的 **任务适应性** 至关重要。

### 3. **对话的标记与编码**：

* 在训练过程中，我们将对话转化为 **token（标记）序列**，这些 token 会被模型用来进行训练。每个对话在转换成 token 时会被编码为一维的 token 序列。例如，一个对话可能会被编码成多个 token，其中包括用户和助手的不同“轮次”。
* 不同的语言模型有不同的标记格式。举例来说，**GPT-4** 使用特殊的标记（例如 `IM start` 和 `IM end`）来区分用户和助手的对话轮次，标记序列告诉模型该如何理解对话。

### 4. **后训练阶段的作用**：

* 在后训练阶段，虽然数据集的规模相对较小，但它为模型注入了大量关于对话行为的统计信息。例如，模型会学习如何回答问题、拒绝请求、以及在特定情况下展示个性化回答。
* 这些 **对话数据集** 通常由人工标注员创建，他们会根据提供的对话上下文编写理想的答案。标注员在处理这些对话时会遵循一些指导原则，如 **有帮助、真实和无害**。

### 5. **现代数据集的合成与人工干预**：

* 目前的对话数据集通常是 **合成的**，但会有一些人工干预。在实际操作中，许多对话生成任务已经开始使用 **预训练语言模型** 来帮助生成初步回答，再由人工标注员进行编辑。
* 一个例子是 **Ultra Chat**，这是一个包含大量对话的合成数据集，虽然它是合成的，但经过人工编辑，以确保对话质量和准确性。

### 6. **推理阶段的工作原理**：

* 在 **推理阶段**，用户与模型进行对话时，系统会根据 **先前的对话上下文** 生成回复。模型并不会在每次查询时“重新研究”或从头开始思考，而是通过之前训练过的数据集来生成合理的答案。
* 例如，当问到“推荐巴黎的五大景点”时，如果这个问题出现在训练数据集中，模型很可能会给出类似于标注员提供的答案；如果这个问题没有出现在训练集里，模型则会根据统计知识和以前的训练数据生成答案。

### 7. **语言模型的统计模拟**：

* 最后，重要的一点是，模型 **并不是** 一个具备自主意识的“魔法”AI。它是 **统计学上模仿人类标注员** 的输出。它通过大量的对话数据和标注员的示例来生成回复。因此，ChatGPT 等对话系统的回答基本上是对标注员行为的模拟。
* 例如，当你问 ChatGPT 一个问题时，它的回答并不是通过实时的知识获取得来的，而是基于它在训练中学到的模式和信息。所以，你可以将它视为一个 **高效的人工模拟**，而不是具备真实“智慧”的存在。

### 总结：

* 后训练阶段通过对话数据集训练模型，使其能够成为一个有效的助手，能回答问题并进行多轮对话。
* 数据集的构建通常由人工标注员完成，随着技术的发展，**合成数据** 和 **语言模型辅助生成数据** 的方式变得越来越普遍。
* 在推理阶段，模型基于已经训练好的模式和统计知识生成回答，而不是像人类那样通过研究和分析来获取知识。


# hallucinations, tool use, knowledge/working memory

turn to the topic of llm psychology as I like to call it which is what are sort of the emergent cognitive effects of the
training pipeline that we have for these models so in particular the first one I want to talk to is of course
hallucinations so you might be familiar with model hallucinations it's when llms make stuff up they just totally
fabricate information Etc and it's a big problem with llm assistants it is a problem that existed to a large extent
with early models uh from many years ago and I think the problem has gotten a bit better uh because there are some
medications that I'm going to go into in a second for now let's just try to understand where these hallucinations come from so here's a specific example
of a few uh of three conversations that you might think you have in your training set and um these are pretty
reasonable conversations that you could imagine being in the training set so like for example who is Cruz well Tom
Cruz is an famous actor American actor and producer Etc who is John baraso this
turns out to be a us senetor for example who is genis Khan well genis Khan was
blah blah blah and so this is what your conversations could look like at training time now the problem with this
is that when the human is writing the correct answer for the assistant in each
one of these cases uh the human either like knows who this person is or they research them on the Internet and they
come in and they write this response that kind of has this like confident tone of an answer and what happens
basically is that at test time when you ask for someone who is this is a totally random name that I totally came up with
and I don't think this person exists um as far as I know I just Tred to generate it randomly the problem is when we ask
who is Orson kovats the problem is that the assistant will not just tell you oh
I don't know even if the assistant and the language model itself might know
inside its features inside its activations inside of its brain sort of it might know that this person is like not someone that um that is that it's
familiar with even if some part of the network kind of knows that in some sense the uh saying that oh I don't know who
this is is is not going to happen because the model statistically imitates
is training set in the training set the questions of the form who is blah are confidently answered with the correct
answer and so it's going to take on the style of the answer and it's going to do its best it's going to give you
statistically the most likely guess and it's just going to basically make stuff up because these models again we just
talked about it is they don't have access to the internet they're not doing research these are statistical token
tumblers as I call them uh is just trying to sample the next token in the sequence and it's going to basically
make stuff up so let's take a look at what this looks like I have here what's called the
inference playground from hugging face and I am on purpose picking on a model
called Falcon 7B which is an old model this is a few years ago now so it's an older model So It suffers from
hallucinations and as I mentioned this has improved over time recently but let's say who is Orson kovats let's ask
Falcon 7B instruct run oh yeah Orson kovat is an American author and science uh fiction writer
okay this is totally false it's hallucination let's try again these are statistical systems right so we can
resample this time Orson kovat is a fictional character from this 1950s TV
show it's total BS right let's try again he's a former minor league baseball
player okay so basically the model doesn't know and it's given us lots of different answers because it doesn't
know it's just kind of like sampling from these probabilities the model starts with the tokens who is oron
kovats assistant and then it comes in here and it's get it's getting these
probabilities and it's just sampling from the probabilities and it just like comes up with stuff and the stuff is
actually statistically consistent with the style of the answer in its training set and
it's just doing that but you and I experiened it as a madeup factual knowledge but keep in mind that uh the
model basically doesn't know and it's just imitating the format of the answer and it's not going to go off and look it up uh because it's just imitating again
the answer so how can we uh mitigate this because for example when we go to chat apt and I say who is oron kovats
and I'm now asking the stateoftheart state-of-the-art model from open AI this model will tell
you oh so this model is actually is even smarter because you saw very briefly it
said searching the web uh we're going to cover this later um it's actually trying to do tool use and
uh kind of just like came up with some kind of a story but I want to just who or Kovach did not use any tools I don't
want it to do web search there's a wellknown historical or
public figure named or oron kovats so this model is not going to make up stuff this model knows that it doesn't know
and it tells you that it doesn't appear to be a person that this model knows so somehow we sort of improved
hallucinations even though they clearly are an issue in older models and it makes totally uh sense why you would be
getting these kinds of answers if this is what your training set looks like so how do we fix this okay well clearly we
need some examples in our data set that where the correct answer for the assistant is that the model doesn't know
about some particular fact but we only need to have those answers be produced in the cases where the model actually
doesn't know and so the question is how do we know what the model knows or doesn't know well we can empirically
probe the model to figure that out so let's take a look at for example how meta uh dealt with hallucinations for
the Llama 3 series of models as an example so in this paper that they published from meta we can go into
hallucinations which they call here factuality and they
describe the procedure by which they basically interrogate the model to figure out what it knows and doesn't
know to figure out sort of like the boundary of its knowledge and then they add examples to the training set where
for the things where the model doesn't know them the correct answer is that the model doesn't know them which sounds
like a very easy thing to do in principle but this roughly fixes the issue and the the reason it fixes the
issue is because remember like the model might actually have a pretty good model of its
self knowledge inside the network so remember we looked at the network and all these neurons inside the network you
might imagine that there's a neuron somewhere in the network that sort of like lights up for when the model is
uncertain but the problem is that the activation of that neuron is not currently wired up to the model actually
saying in words that it doesn't know so even though the internal of the neural network no because there's some neurons
that represent that the model uh will not surface that it will instead take its best guess so that it sounds
confident um just like it sees in a training set so we need to basically interrogate the model and allow it to
say I don't know in the cases that it doesn't know so let me take you through what meta roughly does so basically what
they do is here I have an example uh Dominic kek is uh the featured article
today so I just went there randomly and what they do is basically they take a random document in a training set and
they take a paragraph and then they use an llm to construct questions about that
paragraph so for example I did that with chat GPT here so I said here's a paragraph from
this document generate three specific factual questions based on this paragraph and give me the questions and
the answers and so the llms are already good enough to create and reframe this
information so if the information is in the context window um of this llm this
actually works pretty well it doesn't have to rely on its memory it's right there in the context window and so it
can basically reframe that information with fairly high accuracy so for example can generate questions for us like for
which team did he play here's the answer how many cups did he win Etc and now
what we have to do is we have some question and answers and now we want to interrogate the model so roughly speaking what we'll do is we'll take our
questions and we'll go to our model which would be uh say llama uh in meta
but let's just interrogate mol 7B here as an example that's another model so does this model know about this answer
let's take a look uh so he played for Buffalo Sabers right so the model knows and the the way
that you can programmatically decide is basically we're going to take this answer from the model and we're going to
compare it to the correct answer and again the model model are good enough to do this automatically so there's no
humans involved here we can take uh basically the answer from the model and we can use another llm judge to check if
that is correct according to this answer and if it is correct that means that the model probably knows so what we're going
to do is we're going to do this maybe a few times so okay it knows it's Buffalo Savers let's drag
in um Buffalo Sabers let's try one more
time Buffalo Sabers so we asked three times about this factual question and the model seems to know so everything is
great now let's try the second question how many Stanley Cups did he win and again let's interrogate the
model about that and the correct answer is two so um here the model claims that he
won um four times which is not correct right it doesn't match two so the model
doesn't know it's making stuff up let's try again
um so here the model again it's kind of like making stuff up right let's
Dragon here it says did he did not even did not win during his career so
obviously the model doesn't know and the way we can programmatically tell again is we interrogate the model three times
and we compare its answers maybe three times five times whatever it is to the correct answer and if the model doesn't
know then we know that the model doesn't know this question and then what we do is we take this question we create a new conversation in
the training set so we're going to add a new conversation training set and when the question is how many Stanley Cups
did he win the answer is I'm sorry I don't know or I don't remember and
that's the correct answer for this question because we interrogated the model and we saw that that's the case if
you do this for many different types of uh questions for many different types of documents you are giving the model an
opportunity to in its training set refuse to say based on its knowledge and
if you just have a few examples of that in your training set the model will know
um and and has the opportunity to learn the association of this knowledge-based refusal to this internal neuron
somewhere in its Network that we presume exists and empirically this turns out to be probably the case and it can learn
that Association that hey when this neuron of uncertainty is high then I actually don't know and I'm allowed to
say that I'm sorry but I don't think I remember this Etc and if you have these
uh examples in your training set then this is a large mitigation for hallucination and that's roughly
speaking why chpt is able to do stuff like this as well so these are kinds of uh mitigations that people have
implemented and that have improved the factuality issue over time okay so I've described mitigation number one for
basically mitigating the hallucinations issue now we can actually do much better than that uh it's instead of just saying
that we don't know uh we can introduce an additional mitigation number two to give the llm an opportunity to be
factual and actually answer the question now what do you and I do if I was to ask
you a factual question and you don't know uh what would you do um in order to answer the question well you could uh go
off and do some search and uh use the internet and you could figure out the answer and then tell me what that answer
is and we can do the exact exact same thing with these models so think of the
knowledge inside the neural network inside its billions of parameters think of that as kind of a vague recollection
of the things that the model has seen during its training during the pre-training stage a long time ago so
think of that knowledge in the parameters as something you read a month ago and if you keep reading something
then you will remember it and the model remembers that but if it's something rare then you probably don't have a really good recollection of that
information but what you and I do is we just go and look it up now when you go and look it up what you're doing
basically is like you're refreshing your working memory with information and then you're able to sort of like retrieve it
talk about it or Etc so we need some equivalent of allowing the model to refresh its memory or its recollection
and we can do that by introducing tools uh for the models so the way we are going to
approach this is that instead of just saying hey I'm sorry I don't know we can attempt to use tools so we can create uh
a mechanism by which the language model can emit special tokens and these are tokens that we're going to introduce new tokens so
for example here I've introduced two tokens and I've introduced a format or a protocol for how the model is allowed to
use these tokens so for example instead of answering the question when the model does not instead of just saying I don't
know sorry the model has the option now to emitting the special token search start and this is the query that will go
to like bing.com in the case of openai or say Google search or something like that so it will emit the query and then
it will emit search end and then here what will happen is that the program
that is sampling from the model that is running the inference when it sees the special token search end instead of
sampling the next token uh in the sequence it will actually pause generating from the model it will go off
it will open a session with bing.com and it will paste the search query into Bing
and it will then um get all the text that is retrieved and it will basically take that text it will maybe represent
it again with some other special tokens or something like that and it will take that text and it will copy paste it here
into what I Tred to like show with the brackets so all that text kind of comes here and when the text comes here it
enters the context window so the model so that text from the web search is now inside the context window that will feed
into the neural network and you should think of the context window as kind of like the working memory of the model
that data that is in the context window is directly accessible by the model it directly feeds into the neural network
so it's not anymore a vague recollection it's data that it it has in the context
window and is directly available to that model so now when it's sampling the new uh tokens here afterwards it can
reference very easily the data that has been copy pasted in there so that's
roughly how these um how these tools use uh tools uh function
and so web search is just one of the tools we're going to look at some of the other tools in a bit uh but basically you introduce new tokens you introduce
some schema by which the model can utilize these tokens and can call these special functions like web search
functions and how do you teach the model how to correctly use these tools like say web search search start search end
Etc well again you do that through training sets so we need now to have a bunch of data and a bunch of
conversations that show the model by example how to use web search so what
are the what are the settings where you are using the search um and what does that look like and here's by example how
you start a search and the search Etc and uh if you have a few thousand maybe examples of that in your training set
the model will actually do a pretty good job of understanding uh how this tool works and it will know how to sort of
structure its queries and of course because of the pre-training data set and its understanding of the world it
actually kind of understands what a web search is and so it actually kind of has a pretty good native understanding
um of what kind of stuff is a good search query um and so it all kind of just like works you just need a little
bit of a few examples to show it how to use this new tool and then it can lean on it to retrieve information and uh put
it in the context window and that's equivalent to you and I looking something up because once it's in the
context it's in the working memory and it's very easy to manipulate and access so that's what we saw a few minutes ago
when I was searching on chat GPT for who is Orson kovats the chat GPT language model decided Ed that this is some kind
of a rare um individual or something like that and instead of giving me an answer from its memory it decided that
it will sample a special token that is going to do web search and we saw briefly something flash it was like
using the web tool or something like that so it briefly said that and then we waited for like two seconds and then it generated this and you see how it's
creating references here and so it's citing sources so what happened here is
it went off it did a web web search it found these sources and these URLs and the text of these web pages was all
stuffed in between here and it's not showing here but it's it's basically stuffed as text in between here and now
it sees that text and now it kind of references it and says that okay it could be these people citation could be
those people citation Etc so that's what happened here and that's what and that's why when I said who is Orson kovats I
could also say don't use any tools and then that's enough to um basically convince chat PT to not use
tools and just use its memory and its recollection I also went off and I um
tried to ask this question of Chachi PT so how many standing cups did uh Dominic Hasek win and Chachi P actually decided
that it knows the answer and it has the confidence to say that uh he want twice and so it kind of just relied on its
memory because presumably it has um it has enough of a kind of confidence in its weights in
it parameters and activations that this is uh retrievable just for memory um but
you can also conversely use web search to make sure and then for the same query it actually
goes off and it searches and then it finds a bunch of sources it finds all this all of this stuff gets copy pasted
in there and then it tells us uh to again and sites and it actually says the
Wikipedia article which is the source of this information for us as well so
that's tools web search the model determines when to search and then uh that's kind of like how these tools uh
work and this is an additional kind of mitigation for uh hallucinations and factuality so I want to stress one more
time this very important sort of psychology Point knowledge in the parameters of the
neural network is a vague recollection the knowledge in the tokens that make up the context
window is the working memory and it roughly speaking Works kind of like um
it works for us in our brain the stuff we remember is our parameters uh and the stuff that we just experienced like a
few seconds or minutes ago and so on you can imagine that being in our context window and this context window is being built up as you have a conscious
experience around you so this has a bunch of um implications also for your use of LOLs in practice so for example I
can go to chat GPT and I can do something like this I can say can you Summarize chapter one of Jane Austin's Pride and Prejudice right and this is a
perfectly fine prompt and Chach actually does something relatively reasonable here and but the reason it does that is
because Chach has a pretty good recollection of a famous work like Pride and Prejudice it's probably seen a ton
of stuff about it there's probably forums about this book it's probably read versions of this book um and it's
kind of like remembers because even if you've read this or articles about it
you'd kind of have a recollection enough to actually say all this but usually when I actually interact with LMS and I want them to recall specific things it
always works better if you just give it to them so I think a much better prompt would be something like this can you
summarize for me chapter one of genos's spr and Prejudice and then I am attaching it below for your reference
and then I do something like a delimeter here and I paste it in and I I found that just copy pasting it from some
website that I found here um so copy pasting the chapter one here and I do
that because when it's in the context window the model has direct access to it and can exactly it doesn't have to
recall it it just has access to it and so this summary is can be expected to be a significantly high quality or higher
quality than this summary uh just because it's directly available to the model and I think you and I would work
in the same way if you want to it would be you would produce a much better summary if you had reread this chapter
before you had to summarize it and that's basically what's happening here or the equivalent of it the next sort of

这段话主要讨论了LLM（大型语言模型）在训练和推理过程中的一些“心理学”现象，尤其是关于**幻觉**、**工具使用**以及**知识/工作记忆**等方面的问题。以下是对内容的中文解释：

### 幻觉

幻觉指的是语言模型在回答问题时，常常会“编造”一些并不存在的事实或信息。这是LLM中一个常见的问题，尤其是早期的模型。在训练过程中，模型通过大量的文本数据学习语言模式和结构，但它们并不具备真正的“知识”或“理解”，因此在遇到不熟悉的内容时，它们会生成一个看起来合理但实际上错误的答案。

例如，当询问模型一个虚构的人物时，模型会根据其训练数据生成相关的、看似正确的信息，即使这些信息根本是“编造”的。模型并不会知道它是否正确，只会生成最有可能的答案。

### 如何解决幻觉

为了缓解幻觉问题，模型需要能够“承认”它不知晓某个事实。这可以通过在训练数据中加入一些“模型不知道”的情况来实现。具体方法是，通过反复询问模型一些问题，如果它不能正确回答，就让它学会在这些情况下直接说“不知道”。通过这种方式，模型可以学习到在某些情况下，拒绝回答是更合适的反应。

### 工具使用

为了进一步提升模型的准确性，我们可以让模型使用外部工具，比如网络搜索。当模型遇到一个它无法确定的事实时，它可以触发搜索操作，从互联网上获取信息并将其作为“工作记忆”添加到上下文中。这样，模型就不必单纯依赖它自身的参数记忆，而可以通过实时搜索来获取最新的、准确的信息。

具体来说，模型通过发出特定的信号（如“search start”和“search end”）来触发工具的使用。例如，模型会通过网络搜索查找某个问题的答案，然后将搜索结果放入上下文中，供模型参考。这种方式类似于人类使用互联网查询信息的方式，可以有效避免幻觉和提升模型的准确性。

### 知识与工作记忆

在LLM中，“知识”可以理解为模型的“参数记忆”，而“工作记忆”则是模型当前上下文中的数据。类似于人类大脑中的记忆系统，模型会从其参数记忆中提取过去学到的知识，并将当前的工作记忆（即最新的信息）作为上下文进行处理。在推理阶段，模型将上下文中的信息作为依据来生成答案。

例如，在回答问题时，模型会利用它的参数记忆来提供答案，但如果问题涉及一些具体内容，它可能需要从“工作记忆”中获取额外信息。如果直接将相关信息提供给模型（例如，粘贴一章文本），模型的回答通常会更加准确，因为它可以直接读取并引用这些信息。

### 总结

1. **幻觉**：模型会根据训练数据生成“统计上可能的答案”，即使这些答案不准确。解决方案是加入“不知道”的例子，让模型学会拒绝回答。
2. **工具使用**：通过引入外部工具（如网络搜索），模型可以获得最新的信息来增强回答的准确性。
3. **知识与工作记忆**：模型的知识存储在其参数中，而工作记忆则是在当前上下文中提供的额外信息。通过给模型提供更多的上下文，它能够生成更加精确的回答。

这些方法和技术的引入使得现代语言模型在回答问题时更加精准且可信，减少了幻觉的发生。


# knowledge of self

psychological Quirk I'd like to talk about briefly is that of the knowledge of self so what I see very often on the
internet is that people do something like this they ask llms something like what model are you and who built you and
um basically this uh question is a little bit nonsensical and the reason I say that is that as I try to kind of
explain with some of the underhood fundamentals this thing is not a person right it doesn't have a persistent
existence in any way it sort of boots up processes tokens and shuts off and it
does that for every single person it just kind of builds up a context window of conversation and then everything gets deleted and so this this entity is kind
of like restarted from scratch every single conversation if that makes sense it has no persistent self it has no
sense of self it's a token tumbler and uh it follows the statistical regularities of its training set so it
doesn't really make sense to ask it who are you what build you Etc and by default if you do what I described and
just by default and from nowhere you're going to get some pretty random answers so for example let's uh pick on Falcon
which is a fairly old model and let's see what it tells us uh so it's evading the question uh
talented engineers and developers here it says I was built by open AI based on the gpt3 model it's totally making stuff
up now the fact that it's built by open AI here I think a lot of people would take this as evidence that this model
was somehow trained on open AI data or something like that I don't actually think that that's necessarily true the reason for that is
that if you don't explicitly program the model to answer these kinds of questions
then what you're going to get is its statistical best guess at the answer and
this model had a um sft data mixture of conversations and during the
fine-tuning um the model sort of understands as it's training on this data that it's taking on this
personality of this like helpful assistant and it doesn't know how to it doesn't actually it wasn't told exactly
what label to apply to self it just kind of is taking on this uh this uh Persona
of a helpful assistant and remember that the pre-training stage took the documents from the entire internet and
Chach and open AI are very prominent in these documents and so I think what's actually likely to be happening here is
that this is just its hallucinated label for what it is this is its self-identity is that it's chat GPT by open Ai and
it's only saying that because there's a ton of data on the internet of um answers like this that are actually
coming from open from chasht and So that's its label for what it is now you
can override this as a developer if you have a llm model you can actually override it and there are a few ways to
do that so for example let me show you there's this MMO model from Allen Ai and
um this is one llm it's not a top tier LM or anything like that but I like it because it is fully open source so the
paper for Almo and everything else is completely fully open source which is nice um so here we are looking at its
sft mixture so this is the data mixture of um the fine tuning so this is the
conversations data it right and so the way that they are solving it for Theo model is we see that there's a bunch of
stuff in the mixture and there's a total of 1 million conversations here but here we have alot to hardcoded if we go there
we see that this is 240 conversations and look at these 240
conversations they're hardcoded tell me about yourself says user and then the
assistant says I'm and open language model developed by AI to Allen Institute of artificial intelligence Etc I'm here
to help blah blah blah what is your name uh Theo project so these are all kinds of like cooked up hardcoded questions
abouto 2 and the correct answers to give in these cases if you take 240 questions
like this or conversations put them into your training set and fine tune with it then the model will actually be expected
to parot this stuff later if you don't give it this then it's probably a Chach
by open Ai and um there's one more way to sometimes do this is
that basically um in these conversations and you have terms between human and
assistant sometimes there's a special message called system message at the very beginning of the conversation so
it's not just between human and assistant there's a system and in the system message you can actually hardcode
and remind the model that hey you are a model developed by open Ai and your name
is chashi pt40 and you were trained on this date and your knowledge cut off is this and basically it kind of like
documents the model a little bit and then this is inserted into to your conversations so when you go on chpt you
see a blank page but actually the system message is kind of like hidden in there and those tokens are in the context
window and so those are the two ways to kind of um program the models to talk
about themselves either it's done through uh data like this or it's done through system message and things like
that basically invisible tokens that are in the context window and remind the model of its identity but it's all just
kind of like cooked up and bolted on in some in some way it's not actually like really deeply there in any real sense as
it would before a human I want to now continue to the next section which deals with the computational capabilities or

# models need tokens to think

like I should say the native computational capabilities of these models in problem solving scenarios and
so in particular we have to be very careful with these models when we construct our examples of conversations and there's a lot of sharp edges here
that are kind of like elucidative is that a word uh they're kind of like interesting to look at when we consider
how these models think so um consider the following prompt from a human and
supposed that basically that we are building out a conversation to enter into our training set of conversations so we're going to train the model on
this we're teaching you how to basically solve simple math problems so the prompt is Emily buys three apples and two
oranges each orange cost $2 the total cost is 13 what is the cost of apples very simple math question now there are
two answers here on the left and on the right they are both correct answers they both say that the answer is three which
is correct but one of these two is a significant ific anly better answer for the assistant than the other like if I
was Data labeler and I was creating one of these one of these would be uh a really terrible answer for the assistant
and the other would be okay and so I'd like you to potentially pause the video Even and think through why one of these
two is significantly better answer uh than the other and um if you use the
wrong one your model will actually be uh really bad at math potentially and it would have uh bad outcomes and this is
something that you would be careful with in your life labeling documentations when you are training people uh to create the ideal responses for the
assistant okay so the key to this question is to realize and remember that when the models are training and also
inferencing they are working in onedimensional sequence of tokens from left to right and this is the picture
that I often have in my mind I imagine basically the token sequence evolving from left to right and to always produce
the next token in a sequence we are feeding all these tokens into the neural network and this neural network then is
the probabilities for the next token and sequence right so this picture here is the exact same picture we saw uh before
up here and this comes from the web demo that I showed you before right so this
is the calculation that basically takes the input tokens here on the top and uh performs these operations of all these
neurons and uh gives you the answer for the probabilities of what comes next now the important thing to realize is that
roughly speaking uh there's basically a finite number of layers of computation that
happened here so for example this model here has only one two three layers of
what's called detention and uh MLP here um maybe um typical modern
state-of-the-art Network would have more like say 100 layers or something like that but there's only 100 layers of computation or something like that to go
from the previous token sequence to the probabilities for the next token and so there's a finite amount of computation
that happens here for every single token and you should think of this as a very small amount of computation and this
amount of computation is almost roughly fixed uh for every single token in this sequence um the that's not actually
fully true because the more tokens you feed in uh the the more expensive uh this forward pass will be of this neural
network but not by much so you should think of this uh and I think as a good model to have in mind this is a fixed
amount of compute that's going to happen in this box for every single one of these tokens and this amount of compute
Cann possibly be too big because there's not that many layers that are sort of going from the top to bottom here
there's not that that much computationally that will happen here and so you can't imagine the model to to basically do arbitrary computation in a
single forward pass to get a single token and so what that means is that we actually have to distribute our
reasoning and our computation across many tokens because every single token is only spending a finite amount of
computation on it and so we kind of want to distribute the computation across
many tokens and we can't have too much computation or expect too much computation out of of the model in any
single individual token because there's only so much computation that happens per token okay roughly fixed amount of
computation here so that's why this answer here is
significantly worse and the reason for that is Imagine going from left to right here um and I copy pasted it right here
the answer is three Etc imagine the model having to go from left to right emitting these tokens one at a time it
has to say or we're expecting to say the answer is space dollar sign and then
right here we're expecting it to basically cram all of the computation of this problem into this single token it has to emit the correct answer three and
then once we've emitted the answer three we're expecting it to say all these tokens but at this point we've already
prod produced the answer and it's already in the context window for all these tokens that follow so anything
here is just um kind of post Hawk justification of why this is the answer
um because the answer is already created it's already in the token window so it's it's not actually being calculated here
um and so if you are answering the question directly and immediately you are training the model to to try to
basically guess the answer in a single token and that is just not going to work because of the finite amount of
computation that happens per token that's why this answer on the right is significantly better because we are
Distributing this computation across the answer we're actually getting the model to sort of slowly come to the answer
from the left to right we're getting intermediate results we're saying okay the total cost of oranges is four so 30
- 4 is 9 and so we're creating intermediate calculations and each one
of these calculations is by itself not that expensive and so we're actually basically kind of guessing a little bit
the difficulty that the model is capable of in any single one of these individual tokens and there can never be too much
work in any one of these tokens computationally because then the model won't be able to do that later at test
time and so we're teaching the model here to spread out its reasoning and to spread out its computation over the
tokens and in this way it only has very simple problems in each token and they
can add up and then by the time it's near the end it has all the previous results in its working memory and it's
much easier for it to determine that the answer is and here it is three so this is a significantly better label for our
computation this would be really bad and is teaching the model to try to do all the computation in a single token and
it's really bad so uh that's kind of like an interesting thing to keep in mind is in
your prompts uh usually don't have to think about it explicitly because uh the
people at open AI have labelers and so on that actually worry about this and they make sure that the answers are
spread out and so actually open AI will kind of like do the right thing so when I ask this question for chat GPT it's
actually going to go very slowly it's going to be like okay let's define our variables set up the equation and it's kind of creating all these
intermediate results these are not for you these are for the model if the model is not creating these intermediate
results for itself it's not going to be able to reach three I also wanted to show you that it's possible to be a bit
mean to the model uh we can just ask for things so as an example I said I gave it the exact same uh prompt and I said
answer the question in a single token just immediately give me the answer nothing else and it turns out that for
this simple um prompt here it actually was able to do it in single go so it just created a single I think this is
two tokens right uh because the dollar sign is its own token so basically this
model didn't give me a single token it gave me two tokens but it still produced the correct answer and it did that in a
single forward pass of the network now that's because the numbers here I think are very simple and so I
made it a bit more difficult to be a bit mean to the model so I said Emily buys 23 apples and 177 oranges and then I
just made the numbers a bit bigger and I'm just making it harder for the model I'm asking it to more computation in a
single token and so I said the same thing and here it gave me five and five is actually not correct so the model
failed to do all of this calculation in a single forward pass of the network it failed to go from the input tokens and
then in a single forward pass of the network single go through the network it couldn't produce the result and then I
said okay now don't worry about the the token limit and just solve the problem as usual and then it goes all the
intermediate results it simplifies and every one of these intermediate results here and intermediate calculations is
much easier for the model and um it sort of it's not too much work per token all
of the tokens here are correct and it arises the solution which is seven and I just couldn't squeeze all of this work
it couldn't squeeze that into a single forward passive Network so I think that's kind of just a cute example and
something to kind of like think about and I think it's kind of again just elucidative in terms of how these uh
models work the last thing that I would say on this topic is that if I was in practi is trying to actually solve this in my day-to-day life I might actually
not uh trust that the model that all the intermediate calculations correctly here so actually probably what I do is
something like this I would come here and I would say use code and uh that's
because code is one of the possible tools that chachy PD can use and instead
of it having to do mental arithmetic like this mental arithmetic here I don't fully trust it and especially if the
numbers get really big there's no guarantee that the model will do this correctly any one of these intermediates
steps might in principle fail we're using neural networks to do mental arithmetic uh kind of like you doing
mental arithmetic in your brain it might just like uh screw up some of the intermediate results it's actually kind
of amazing that it can even do this kind of mental arithmetic I don't think I could do this in my head but basically the model is kind of like doing it in
its head and I don't trust that so I wanted to use tools so you can say stuff like use
code and uh I'm not sure what happened there use
code and so um like I mentioned there's a special tool and the uh the model can
write code and I can inspect that this code is correct and then uh it's not
relying on its mental arithmetic it is using the python interpreter which is a very simple programming language to
basically uh write out the code that calculates the result and I would personally trust this a lot more because
this came out of a Python program which I think has a lot more correctness guarantees than the mental arithmetic of
a language model uh so just um another kind of uh potential hint that if you
have these kinds of problems uh you may want to basically just uh ask the model to use the code interpreter and just
like we saw with the web search the model has special uh kind of tokens for
calling uh like it will not actually generate these tokens from the language model it will write the program and then
it actually sends that program to a different sort of part of the computer that actually just runs that program and
brings back the result and then the model gets access to that result and can tell you that okay the cost of each apple is seven
um so that's another kind of tool and I would use this in practice for yourself and it's um yeah it's just uh less error
prone I would say so that's why I called this section models need tokens to think distribute your competition across many
tokens ask models to create intermediate results or whenever you can lean on
tools and Tool use instead of allowing the models to do all of the stuff in their memory so if they try to do it all
in their memory I don't fully trust it and prefer to use tools whenever possible I want to show you one more
example of where this actually comes up and that's in counting so models actually are not very good at counting
for the exact same reason you're asking for way too much in a single individual token so let me show you a simple
example of that um how many dots are below and then I just put in a bunch of dots and Chach says there are and then
it just tries to solve the problem in a single token so in a single token it has to count the number of dots in its
context window um and it has to do that in the single forward pass of a network and a single
forward pass of a network as we talked about there's not that much computation that can happen there just think of that as being like very little competation
that happens there so if I just look at what the model sees let's go to the LM
go to tokenizer it sees uh this how many dots are below and then it
turns out that these dots here this group of I think 20 dots is a single token and then this group of whatever it
is is another token and then for some reason they break up as this so I don't
actually this has to do with the details of the tokenizer but it turns out that these um the model basically sees the
token ID this this this and so on and then from these token IDs it's expected
to count the number and spoiler alert is not 161 it's actually I believe
177 so here's what we can do instead uh we can say use code and you might expect
that like why should this work and it's actually kind of subtle and kind of interesting so when I say use code I
actually expect this to work let's see okay 177 is correct so what happens here
is I've actually it doesn't look like it but I've broken down the problem into a
problems that are easier for the model I know that the model can't count it can't do mental counting but I know that the
model is actually pretty good at doing copy pasting so what I'm doing here is when I say use code it creates a string
in Python for this and the task of basically copy pasting my input here to
here is very simple because for the model um it sees this string of uh it
sees it as just these four tokens or whatever it is so it's very simple for the model to copy paste those token IDs
and um kind of unpack them into Dots here and so it creates this string and
then it calls python routine. count and then it comes up with the correct answer so the python interpreter is doing the
counting it's not the models mental arithmetic doing the counting so it's again a simple example of um models need
tokens to think don't rely on their mental arithmetic and um that's why also the models are not very good at counting
if you need them to do counting tasks always ask them to lean on the tool now the models also have many other little

这段话深入探讨了大型语言模型（LLM）在处理问题时如何使用“令牌”（tokens）进行计算和推理，尤其是它们如何处理简单的数学问题和计算任务。以下是中文解释：

### 1. **模型的计算能力：令牌的作用**

LLM在处理任务时是按令牌（tokens）逐步计算的，每个令牌表示序列中的一个部分。计算过程中，模型依赖于每个令牌的上下文来预测下一个令牌的可能性。这里的关键是：**每个令牌的计算量是有限的**，这意味着模型在每次生成一个令牌时，只能进行有限的计算。

### 2. **数学问题的处理**

考虑一个简单的数学问题：“Emily买了3个苹果和2个橙子，每个橙子2美元，总价13美元，苹果的价格是多少？”这个问题的正确答案是“3”，但如果我们让模型在一次生成中直接给出答案，它可能只会给出“3”，而忽略了计算过程中的推理。

* **不好的回答**：如果模型直接给出“3”作为答案，而没有中间步骤（例如计算橙子的总价、扣除橙子的费用等），这会让它在推理时跳过重要的计算过程，导致它的计算不充分。

* **好的回答**：通过在回答中加入逐步的推理过程（例如“橙子的总价是4，所以剩下9美元是苹果的费用”），模型就能在每个令牌上执行更简单的计算，从而最终得出正确答案。这是因为每个令牌的计算量有限，而通过逐步推理可以分摊计算量，避免过于复杂的运算。

### 3. **推理的分布**

由于模型每次生成一个令牌时只能进行有限的计算，因此推理和计算需要在多个令牌中分布。模型并不具备像人类一样一次性完成复杂推理的能力。每个令牌生成后，它的计算结果会被传递到下一个令牌，这样模型就可以逐步接近正确的答案。

例如，在这个例子中，通过分步计算（计算橙子的费用、苹果的费用），模型可以逐步完成推理，而不是试图在一个令牌上完成所有计算。

### 4. **模型在困难问题上的局限性**

当问题变得更加复杂时（例如较大的数字或更复杂的数学问题），模型可能无法在单次前向传播中正确计算结果。例如，当问题中的数字变得更大时，模型可能无法一次性给出正确答案，可能会失败。

这种情况下，使用工具（如代码）来辅助计算变得尤为重要。如果让模型仅依靠自身的“心算”，它可能会出错。使用代码（如Python代码）来计算结果则更加可靠，因为代码执行可以减少错误的发生。

### 5. **计数问题的例子**

模型在计数任务上通常也表现不佳，因为计数需要模型在有限的计算量内完成较多的操作。如果让模型一次性计算大量点数，它可能会出错。

例如，当询问模型“下面有多少个点”时，模型可能无法正确计算这些点的数量，因为它只能一次性处理一个有限的计算步骤。为了让模型准确计数，可以使用代码工具，如让模型运行Python代码来计算点的数量，这比依靠模型的内存计算要可靠得多。

### 6. **总结：令牌与计算的关系**

* 模型的计算能力是有限的，计算必须分布在多个令牌上。
* 在推理任务中，如果让模型直接给出答案，它可能会跳过必要的推理步骤，因此分步推理是更好的做法。
* 对于复杂的计算任务，尤其是数学计算和计数任务，使用外部工具（如代码）可以帮助模型更准确地完成任务。

总体来说，这段话强调了：**不要依赖模型的“心算”**，而是应该让它通过逐步推理或使用外部工具（如代码）来解决问题。


# tokenization revisited: models struggle with spelling

cognitive deficits here and there and these are kind of like sharp edges of the technology to be kind of aware of over time so as an example the models
are not very good with all kinds of spelling related tasks they're not very good at it and I told you that we would
loop back around to tokenization and the reason to do for this is that the models they don't see the characters they see
tokens and they their entire world is about tokens which are these little text chunks and so they don't see characters
like our eyes do and so very simple character level tasks often fail so for
example uh I'm giving it a string ubiquitous and I'm asking it to print only every third character starting with
the first one so we start with U and then we should go every third so every
so 1 2 3 Q should be next and then Etc so this I see is not correct and again
my hypothesis is that this is again Dental arithmetic here is failing number one a little bit but number two I think
the the more important issue here is that if you go to Tik tokenizer and you look at ubiquitous we
see that it is three tokens right so you and I see ubiquitous and we can easily access the individual letters because we
kind of see them and when we have it in the working memory of our visual sort of field we can really easily index into
every third letter and I can do that task but the models don't have access to the individual letters they see this as
these three tokens and uh remember these models are trained from scratch on the internet and all these token uh
basically the model has to discover how many of all these different letters are packed into all these different tokens
and the reason we even use tokens is mostly for efficiency uh but I think a lot of people areed interested to delete tokens entirely like we should really
have character level or bite level models it's just that that would create very long sequences and people don't
know how to deal with that right now so while we have the token World any kind of spelling tasks are not actually
expected to work super well so because I know that spelling is not a strong suit because of tokenization I can again Ask
it to lean On Tools so I can just say use code and I would again expect this to work because the task of copy pasting
ubiquitous into the python interpreter is much easier and then we're leaning on python interpreter to manipulate the
characters of this string so when I say use code
ubiquitous yes it indexes into every third character and the actual truth is u2s
uqs uh which looks correct to me so um again an example of spelling related
tasks not working very well a very famous example of that recently is how many R are there in strawberry and this
went viral many times and basically the models now get it correct they say there are three Rs in Strawberry but for a
very long time all the state-of-the-art models would insist that there are only two RS in strawberry and this caused a
lot of you know Ruckus because is that a word I think so because um it just kind
of like why are the models so brilliant and they can solve math Olympiad questions but they can't like count RS
in strawberry and the answer for that again is I've got built up to it kind of slowly but number one the models don't
see characters they see tokens and number two they are not very good at counting and so here we are combining
the difficulty of seeing the characters with the difficulty of counting and that's why the models struggled with
this even though I think by now honestly I think open I may have hardcoded the answer here or I'm not sure what they
did but um uh but this specific query now works
so models are not very good at spelling and there there's a bunch of other little sharp edges and I don't want to
go into all of them I just want to show you a few examples of things to be aware of and uh when you're using these models
in practice I don't actually want to have a comprehensive analysis here of all the ways that the models are kind of

# jagged intelligence

like falling short I just want to make the point that there are some Jagged edges here and there and we've discussed
a few of them and a few of them make sense but some of them also will just not make as much sense and they're kind of like you're left scratching your head
even if you understand in- depth how these models work and and good example of that recently is the following uh the
models are not very good at very simple questions like this and uh this is shocking to a lot of people because
these math uh these problems can solve complex math problems they can answer PhD grade physics chemistry biology
questions much better than I can but sometimes they fall short in like super simple problems like this so here we go
9.11 is bigger than 9.9 and it justifies it in some way but obviously and then at
the end okay it actually it flips its decision later so um I don't believe
that this is very reproducible sometimes it flips around its answer sometimes gets it right sometimes get it get it wrong uh let's try
again okay even though it might look larger okay so here it doesn't even correct itself in the end if you ask
many times sometimes it gets it right too but how is it that the model can do so great at Olympiad grade problems but
then fail on very simple problems like this and uh I think this one is as I
mentioned a little bit of a head scratcher it turns out that a bunch of people studied this in depth and I haven't actually read the paper uh but
what I was told by this team was that when you scrutinize the activations
inside the neural network when you look at some of the features and what what features turn on or off and what neurons
turn on or off uh a bunch of neurons inside the neural network light up that are usually associated with Bible verses
U and so I think the model is kind of like reminded that these almost look like Bible verse markers and in a bip
verse setting 9.11 would come after 99.9 and so basically the model somehow finds
it like cognitively very distracting that in Bible verses 9.11 would be greater um even though here it's
actually trying to justify it and come up to the answer with a math it still ends up with the wrong answer here so it
basically just doesn't fully make sense and it's not fully understood and um
there's a few Jagged issues like that so that's why treat this as a as what it is which is a St stochastic system that is
really magical but that you can't also fully trust and you want to use it as a tool not as something that you kind of
like letter rip on a problem and copypaste the results okay so we have now covered two major stages of training

这段话讨论了大型语言模型（LLM）在某些简单问题上出现的意外失误，即使它们在复杂的数学、物理、化学、生物学问题上表现得非常出色。下面是对这段话的中文解释：

### 1. **“Jagged Intelligence”**

这里提到的“Jagged Intelligence”（不均匀的智能）指的是模型在处理不同类型问题时的表现不均衡，表现出某些“锐利的边缘”，也就是模型在某些任务上可能非常出色，但在其他看似简单的任务上却会出现显著的失败。即使我们理解模型的工作原理，也会对其某些不合逻辑的表现感到困惑。

### 2. **模型在简单问题上的失败**

有些看似非常简单的问题，模型却会出现错误。比如，问题是“9.11 是否大于 9.9”，这个问题的答案显然是“是”，但模型却在给出答案时表示“9.11 小于 9.9”，并在后续的推理过程中给出了一些不合逻辑的解释。这个现象令很多人感到震惊，因为这些模型能够解决复杂的数学题，甚至能回答博士级别的物理、化学、生物学问题，但在一些基本的数学判断上却出错。

### 3. **模型错误的潜在原因**

这种错误的背后可能有一个不容易理解的原因。一些研究者深入分析了模型内部的神经网络激活过程，发现当模型处理像“9.11”和“9.9”这种数字时，模型的某些神经元会被激活，这些神经元通常与“圣经章节”相关。比如，在圣经的章节编号中，“9:11”会被视为大于“9:9”。所以，模型可能受到了这种类似圣经标记的认知干扰，导致它给出了错误的答案，即使它在数学上进行了推理。

### 4. **模型的随机性和不可预测性**

这种问题的存在表明，尽管这些模型在大多数情况下表现得非常强大，但它们依然存在不可预知的缺陷。模型有时会根据不同的上下文或者某些看似不相关的激活因素，给出错误的答案。这种现象表明，尽管大型语言模型在某些任务上看起来非常智能，但它们其实是一个“随机系统”，并不能完全信任。

### 5. **使用模型的建议**

由于这些模型有时会犯错误，尤其是在简单问题上，我们应该把它们当作工具来使用，而不是完全依赖它们。面对问题时，不能简单地“放手一搏”，把结果复制粘贴下来，而是应该对模型的输出结果进行判断和验证。

### 总结：

这段话强调了大型语言模型的“不均衡智能”特性：它们在处理一些复杂的学术问题时表现出色，但在处理一些简单的任务时却可能犯错。这些错误背后可能与模型的训练数据、神经网络激活以及某些意外的干扰因素有关。因此，在使用这些模型时，我们应该保持谨慎，将其视为工具，而不是全盘信任它们的输出结果。


# supervised finetuning to reinforcement learning

of large language models we saw that in the first stage this is called the pre-training stage we are basically
training on internet documents and when you train a language model on internet documents you get what's called a base
model and it's basically an internet document simulator right now we saw that this is an interesting artifact and uh
this takes many months to train on thousands of computers and it's kind of a lossy compression of the internet and
it's extremely interesting but it's not directly useful because we don't want to sample internet documents we want to ask
questions of an AI and have it respond to our questions so for that we need an assistant and we saw that we can
actually construct an assistant in the process of a post training and specifically in the process
of supervised fine-tuning as we call it so in this stage we saw that it's
algorithmically identical to pre-training nothing is going to change the only thing that changes is the data set so instead of Internet documents we
now want to create and curate a very nice data set of conversations so we
want Millions conversations on all kinds of diverse topics between a human and an
assistant and fundamentally these conversations are created by humans so
humans write the prompts and humans write the ideal response responses and
they do that based on labeling documentations now in the modern stack
it's not actually done fully and manually by humans right they actually now have a lot of help from these tools
so we can use language models um to help us create these data sets and that's done extensively but fundamentally it's
all still coming from Human curation at the end so we create these conversations that now becomes our data set we fine
tune on it or continue training on it and we get an assistant and then we kind of shifted gears and started talking
about some of the kind of cognitive implications of what this assistant is like and we saw that for example the
assistant will hallucinate if you don't take some sort of mitigations towards it
so we saw that hallucinations would be common and then we looked at some of the mitigations of those hallucinations and
then we saw that the models are quite impressive and can do a lot of stuff in their head but we saw that they can also Lean On Tools to become better so for
example we can lo lean on a web search in order to hallucinate less and to
maybe bring up some more um recent information or something like that or we can lean on tools like code interpreter
so the code can so the llm can write some code and actually run it and see the results so these are some of the topics
we looked at so far um now what I'd like to do is I'd like to cover the last and
major stage of this Pipeline and that is reinforcement learning so reinforcement
learning is still kind of thought to be under the umbrella of posttraining uh but it is the last third major stage and
it's a different way of training language models and usually follows as this third step so inside companies like
open AI you will start here and these are all separate teams so there's a team doing data for pre-training and a team
doing training for pre-training and then there's a team doing all the conversation generation in a in a
different team that is kind of doing the supervis fine tuning and there will be a team for the reinforcement learning as
well so it's kind of like a handoff of these models you get your base model the then you find you need to be an assistant and then you go into
reinforcement learning which we'll talk about uh now so that's kind of like the major
flow and so let's now focus on reinforcement learning the last major stage of training and let me first
actually motivate it and why we would want to do reinforcement learning and what it looks like on a high level so I
would now like to try to motivate the reinforcement learning stage and what it corresponds to with something that you're probably familiar with and that
is basically going to school so just like you went to school to become um really good at something we want to take
large language models through school and really what we're doing is um we're um
we have a few paradigms of ways of uh giving them knowledge or transferring skills so in particular when we're
working with textbooks in school you'll see that there are three major kind of uh pieces of information in these
textbooks three classes of information the first thing you'll see is you'll see a lot of exposition um and by the way
this is a totally random book I pulled from the internet I I think it's some kind of an organic chemistry or something I'm not sure uh but the
important thing is that you'll see that most of the text most of it is kind of just like the meat of it is exposition
it's kind of like background knowledge Etc as you are reading through the words of this Exposition you can think of that
roughly as training on that data so um and that's why when you're reading
through this stuff this background knowledge and this all this context information it's kind of equivalent to
pre-training so it's it's where we build sort of like a knowledge base of this data and get a sense of the topic the
next major kind of information that you will see is these uh problems and with
their worked Solutions so basically a human expert in this case uh the author of this book has given us not just a
problem but has also worked through the solution and the solution is basically like equivalent to having like this
ideal response for an assistant so it's basically the expert is showing us how to solve the problem in it's uh kind of
like um in its full form so as we are reading the solution we are basically
training on the expert data and then later we can try to imitate the expert
um and basically um that's that roughly correspond to having the sft model
that's what it would be doing so basically we've already done pre-training and we've already covered
this um imitation of experts and how they solve these problems and the third
stage of reinforcement learning is basically the practice problems so sometimes you'll see this is just a
single practice problem here but of course there will be usually many practice problems at the end of each chapter in any textbook and practice
problems of course we know are critical for learning because what are they getting you to do they're getting you to practice uh to practice yourself and
discover ways of solving these problems yourself and so what you get in a practice problem is you get a problem
description but you're not given the solution but you are given the final answer answer usually in the answer key
of the textbook and so you know the final answer that you're trying to get to and you have the problem statement
but you don't have the solution you are trying to practice the solution you're trying out many different things and
you're seeing what gets you to the final solution the best and so you're
discovering how to solve these problems so and in the process of that you're relying on number one the background
information which comes from pre-training and number two maybe a little bit of imitation of human experts
and you can probably try similar kinds of solutions and so on so we've done this and this and now in this section
we're going to try to practice and so we're going to be given prompts we're going to be given Solutions U sorry the
final answers but we're not going to be given expert Solutions we have to practice and try stuff out and that's
what reinforcement learning is about okay so let's go back to the problem that we worked with previously just so

这段内容讲的是大型语言模型（LLMs）训练流程中的第三个重要阶段：**强化学习（Reinforcement Learning，简称 RL）**，并通过类比“上学”的过程来解释它的意义和作用。以下是详细的中文解释：

---

## 🌟 总体流程回顾

LLMs 的训练分为三个主要阶段：

### 1. **预训练（Pre-training）**

* 用互联网文本训练模型，让它学习语言的基本规律和知识。
* 输出是一个“基础模型（base model）”，可以看作是“互联网文档模拟器”。
* 虽然强大，但这个模型并不能像助手一样回答人类提出的问题。

### 2. **有监督微调（Supervised Fine-Tuning, SFT）**

* 用人类编写的对话数据对基础模型进行“精调”，让它像一个“助手”一样工作。
* 人类写问题 + 理想答案（或者用模型协助生成，再由人类审查）。
* 训练方式和预训练类似，唯一不同的是数据集变了：从网页内容变成了“对话数据”。

### 3. **强化学习（Reinforcement Learning, RL）**

* 是第三阶段，也是“后训练（post-training）”的一部分。
* 用来进一步提升助手模型的表现，使其回答更符合人类偏好（human preferences）。
* 常由专门团队负责，是模型在成为真正 AI 助手的“最后一步”。

---

## 🎓 用“上学”类比来解释这三步训练过程：

### 📘 1. 预训练 = 看教材的讲解部分（Exposition）

* 比如一本化学教材的大段讲解文字，就是“背景知识”。
* 模型在这个阶段像是在“看书、读资料”，积累对世界的理解和语言能力。

### ✅ 2. 有监督微调 = 看教材中的“例题 + 解法”

* 书中有些例题，作者会详细写出如何一步一步解题。
* 模型在这个阶段就是在“模仿人类专家怎么解题”。
* 这就像在学习“标准解法”，目的是让模型学会正确、清晰地回答问题。

### 🧠 3. 强化学习 = 做练习题（只有题目和答案，没有解法）

* 教材末尾的练习题，只有题目和答案，解题过程得自己摸索。
* 模型要自己“尝试不同的回答方式”，看哪种方式能更好地被人类喜欢或接受。
* 这是**强化学习的核心思想：尝试——评估——调整**。

---

## 💡 强化学习的目的和意义

* 即使模型已经会说话、能回答问题，但它的“表现”可能不是最符合人类喜好的（例如太啰嗦、太武断、太模糊等）。
* 强化学习的目标是：**优化模型的行为，使回答更符合人类的喜好或评分**。
* 实际上，OpenAI 在这个阶段用到了 **人类反馈强化学习（RLHF, Reinforcement Learning from Human Feedback）**，就是人类给回答打分，然后模型学会更倾向于得高分的回答风格。

---

## ✍️ 总结一句话：

强化学习就像是语言模型的“练习题阶段”，它基于前两个阶段所学的内容，通过不断尝试和修正，把“会说话”变成“说得让人喜欢”，从一个基础的模型成长为真正的 AI 助手。

---

如果你想继续了解 RL 的具体流程（比如 RLHF 是怎么训练的），我可以进一步解释下一段内容。


# reinforcement learning

we have a concrete example to talk through as we explore sort of the topic here so um I'm here in the Teck
tokenizer because I'd also like to well I get a text box which is useful but number two I want to remind you again
that we're always working with onedimensional token sequences and so um I actually like prefer this view because
this is like the native view of the llm if that makes sense like this is what it actually sees it sees token IDs right
okay so Emily buys three apples and two oranges each orange is $2 the total cost
of all the fruit is $13 what is the cost of each apple and what I'd like to what I like you to
appreciate here is these are like four possible candidate Solutions as an
example and they all reach the answer three now what I'd like you to appreciate at this point is that if I am
the human data labeler that is creating a conversation to be entered into the training set I don't actually really
know which of these conversations to um to add to the data
set some of these conversations kind of set up a system equations some of them sort of like just talk through it in
English and some of them just kind of like skip right through to the solution um if you look at chbt for
example and you give it this question it defines a system of variables and it kind of like does this little thing what
we have to appreciate and uh differentiate between though is um the
first purpose of a solution is to reach the right answer of course we want to get the final answer three that is the
that is the important purpose here but there's kind of like a secondary purpose as well where here we are also just kind
of trying to make it like nice uh for the human because we're kind of assuming that the person wants to see the
solution they want to see the intermediate steps we want to present it nicely Etc so there are two separate things going on here number one is the
presentation for the human but number two we're trying to actually get the right answer um so let's for the moment
focus on just reaching the final answer if we're only care if we only care about the final answer then which of these is
the optimal or the best prompt um sorry the best solution for the llm to reach
the right answer um and what I'm trying to get at is we don't know me as a human labeler I
would not know which one of these is best so as an example we saw earlier on when we looked at
um the token sequences here and the mental arithmetic and reasoning we saw that for each token we can only spend
basically a finite number of finite amount of compute here that is not very large or you should think about it that way way and so we can't actually make
too big of a leap in any one token is is maybe the way to think about it so as an
example in this one what's really nice about it is that it's very few tokens so it's going to take us very short amount
of time to get to the answer but right here when we're doing 30 - 4 IDE 3
equals right in this token here we're actually asking for a lot of computation to happen on that single individual
token and so maybe this is a bad example to give to the llm because it's kind of incentivizing it to skip through the
calculations very quickly and it's going to actually make up mistakes make mistakes in this mental arithmetic uh so
maybe it would work better to like spread out the spread it out more maybe it would be better to set it up as an
equation maybe it would be better to talk through it we fundamentally don't know and we don't know because what is
easy for you or I as or as human labelers what's easy for us or hard for us is different than what's easy or hard
for the llm it cognition is different um and the token sequences are kind of like
different hard for it and so some of the token sequences here that are trivial
for me might be um very too much of a leap for the llm so right here this
token would be way too hard but conversely many of the tokens that I'm creating here might be just trivial to
the llm and we're just wasting tokens like why waste all these tokens when this is all trivial so if the only thing
we care care about is the final answer and we're separating out the issue of the presentation to the human um then we
don't actually really know how to annotate this example we don't know what solution to get to the llm because we are not the
llm and it's clear here in the case of like the math example but this is actually like a very pervasive issue
like for our knowledge is not lm's knowledge like the llm actually has a ton of knowledge of PhD in math and
physics chemistry and whatnot so in many ways it actually knows more than I do and I'm I'm potentially not utilizing
that knowledge in its problem solving but conversely I might be injecting a bunch of knowledge in my solutions that
the LM doesn't know in its parameters and then those are like sudden leaps
that are very confusing to the model and so our cognitions are different and I
don't really know what to put here if all we care about is the reaching the final solution and doing it economically
ideally and so long story short we are not in a good position to create these
uh token sequences for the LM and they're useful by imitation to initialize the system but we really want
the llm to discover the token sequences that work for it we need to find it needs to find for itself what token
sequence reliably gets to the answer given the prompt and it needs to discover that in the process of
reinforcement learning and of trial and error so let's see how this example
would work like in reinforcement learning okay so we're now back in the huging
face inference playground and uh that just allows me to very easily call uh
different kinds of models so as an example here on the top right I chose the Gemma 2 2 billion parameter model so
two billion is very very small so this is a tiny model but it's okay so we're going to give it um the way that
reinforcement learning will basically work is actually quite quite simple um we need to try many different kinds of
solutions and we want to see which Solutions work well or not so we're basically going to take the
prompt we're going to run the model and the model generates a solution
and then we're going to inspect the solution and we know that the correct answer for this one is $3 and so indeed
the model gets it correct it says it's $3 so this is correct so that's just one attempt at DIS solution so now we're
going to delete this and we're going to rerun it again let's try a second attempt so the model solves it in a bit
slightly different way right every single attempt will be a different generation because these models are
stochastic systems remember that at every single token here we have a probability distribution and we're sampling from that distribution so we
end up kind kind of going down slightly different paths and so this is a second solution that also ends in the correct
answer now we're going to delete that let's go a third time okay so again slightly different
solution but also gets it correct now we can actually repeat this uh many times and so in practice you
might actually sample thousand of independent Solutions or even like million solutions for just a single
prompt um and some of them will be correct and some of them will not be very correct and basically what we want
to do is we want to encourage the solutions that lead to correct answers so let's take a look at what that looks
like so if we come back over here here's kind of like a cartoon diagram of what this is looking like we have a prompt
and then we tried many different solutions in parallel and some of the solutions um
might go well so they get the right answer which is in green and some of the solutions might go poorly and may not
reach the right answer which is red now this problem here unfortunately is not the best example because it's a trivial
prompt and as we saw uh even like a two billion parameter model always gets it right so it's not the best example in
that sense but let's just exercise some imagination here and let's just suppose
that the um green ones are good and the red ones are bad okay so we generated 15 Solutions
only four of them got the right answer and so now what we want to do is basically we want to encourage the kinds
of solutions that lead to right answers so whatever token sequences happened in
these red Solutions obviously something went wrong along the way somewhere and uh this was not a good path to take
through the solution and whatever token sequences there were in these Green Solutions well things went uh pretty
well in this situation and so we want to do more things like it in prompts like
this and the way we encourage this kind of a behavior in the future is we basically train on these sequences um
but these training sequencies now are not coming from expert human annotators there's no human who decided that this
is the correct solution this solution came from the model itself so the model is practicing here it's tried out a few
Solutions four of them seem to have worked and now the model will kind of like train on them and this corresponds
to a student basically looking at their Solutions and being like okay well this one worked really well so this is this is how I should be solving these kinds
of problems and uh here in this example there are many different ways to
actually like really tweak the methodology a little bit here but just to give the core idea across maybe it's simplest to just think about take the
taking the single best solution out of these four uh like say this one that's why it was yellow uh so this is the the
solution that not only led to the right answer but may maybe had some other nice properties maybe it was the shortest one
or it looked nicest in some ways or uh there's other criteria you could think of as an example but we're going to
decide that this the top solution we're going to train on it and then uh the model will be slightly more likely once
you do the parameter update to take this path in this kind of a setting in the
future but you have to remember that we're going to run many different diverse prompts across lots of math
problems and physics problems and whatever wherever there might be so tens of thousands of prompts maybe have in
mind there's thousands of solutions prompt and so this is all happening kind of like at the same time and as we're
iterating this process the model is discovering for itself what kinds of token sequences lead it to correct
answers it's not coming from a human annotator the the model is kind of like
playing in this playground and it knows what it's trying to get to and it's discovering sequences that work for it
uh these are sequences that don't make any mental leaps uh they they seem to work reliably and statistically and uh
fully utilize the knowledge of the model as it has it and so uh this is the process of reinforcement
learning it's basically a guess and check we're going to guess many different types of solutions we're going to check them and we're going to do more
of what worked in the future and that is uh reinforcement learning so in the
context of what came before we see now that the sft model the supervised fine tuning model it's still helpful because
it still kind of like initializes the model a little bit into to the vicinity of the correct Solutions so it's kind of
like a initialization of um of the model in the sense that it kind of gets the model to you know take Solutions like
write out Solutions and maybe it has an understanding of setting up a system of equations or maybe it kind of like talks
through a solution so it gets you into the vicinity of correct Solutions but reinforcement learning is where
everything gets dialed in we really discover the solutions that work for the model get the right answers we encourage
them and then the model just kind of like gets better over time time okay so that is the high Lev process for how we
train large language models in short we train them kind of very similar to how we train children and basically the only
difference is that children go through chapters of books and they do all these different types of training exercises um
kind of within the chapter of each book but instead when we train AIS it's almost like we kind of do it stage by
stage depending on the type of that stage so first what we do is we do pre-training which as we saw is
equivalent to uh basically reading all the expository material so we look at all the textbooks at the same time and
we read all the exposition and we try to build a knowledge base the second thing then is we go into the sft stage which
is really looking at all the fixed uh sort of like solutions from Human Experts of all the different kinds of
worked Solutions across all the textbooks and we just kind of get an sft model which is able to imitate the
experts but does so kind of blindly it just kind of like does its best guess uh kind of just like trying to mimic
statistically the expert behavior and so that's what you get when you look at all the work Solutions and then finally in
the last stage we do all the practice problems in the RL stage across all the textbooks we only do the practice
problems and that's how we get the RL model so on a high level the way we train llms is very much equivalent uh to
the process that we train uh that we use for training of children the next point I would like to make is that actually

这段内容深入解释了大型语言模型（LLMs）训练流程中最后一个关键阶段：**强化学习（Reinforcement Learning，简称 RL）**，并结合一个数学例子，详细说明了它是如何工作的。以下是逐步的中文解释：

---

## 🔁 强化学习：语言模型的“练习题阶段”

### 🔍 举例说明：苹果和橘子的数学题

> “Emily 买了 3 个苹果和 2 个橘子，每个橘子 \$2，总共花了 \$13，请问每个苹果多少钱？”

这个题目有多个可能的正确解法，比如：

* 列出方程解答；
* 用文字逻辑推理；
* 直接 mental math 得出答案；
* 略过过程直接给出答案。

这些方法最终都得出正确答案：\$3。但关键问题是：

> 🧠 作为人类标注者，我并不知道哪种解法对 LLM 最合适！

---

## 🤯 为什么人类难以判断最佳解法？

* LLM 的“思考方式”与人类不同。
* 人类觉得简单的推理步骤（如 mental math）对模型可能太跳跃；
* 而人类觉得啰嗦的详细推导步骤对模型可能刚刚好；
* 有些 token 表达在人类眼里很自然，但对 LLM 是“算力压力太大”的计算跳跃。

所以，人类无法准确判断：

> 哪一种“token 序列”最有助于模型可靠地得出正确答案。

---

## 🎯 强化学习解决了这个问题：让模型自己探索最佳路径！

### 🧪 基本流程：

1. **提供题目（prompt）**
2. **模型随机尝试多种解法**（生成多个回答）
3. **检查这些解法是否得出正确答案**（比如 \$3）
4. **将正确的解法记录下来作为训练样本**
5. **用这些“好解法”更新模型参数**

> 这个过程就像学生做练习题：试一试，看看对不对，然后记住对的做法。

---

## 📊 举个具体实验：

* 用一个 20 亿参数的小模型（Gemma-2B）反复让它回答这个问题。
* 每次生成不同的解法，尽管方式不同，但都得出了 \$3。
* 模型“猜 + 检查”，像练习一样不断尝试，并从成功案例中学习。

🧩 实际训练中会：

* 针对每道题生成成千上万个不同解法；
* 自动判断哪些 token 序列更靠谱；
* 用成功样本来训练模型，让它**更倾向走那些“靠谱的路径”**。

---

## 🧠 强化学习不是人类教模型怎么解题，而是：

> **让模型自己发现什么 token 序列最适合自己去解题。**

* 不再依赖人类写的理想解；
* 模型自己摸索自己最擅长的表达方式；
* 最终找到**对它来说“最省力、最可靠”的推理路径**。

---

## 🪜 总结：LLM 训练的三个阶段类比为“学生学习过程”：

| 阶段                     | 类比        | 说明                              |
| ---------------------- | --------- | ------------------------------- |
| **1. 预训练 Pretraining** | 看课本讲解     | 阅读大量文本，积累通识知识和语言能力              |
| **2. 有监督微调 SFT**       | 看例题和标准解法  | 模仿人类专家的解题方式，进行“有监督模仿学习”         |
| **3. 强化学习 RL**         | 做练习题，自己琢磨 | 让模型自己尝试、犯错、总结、迭代，从中学会“更靠谱”的解题方式 |

---

## ✅ 总结一句话：

强化学习是让模型像学生做练习题一样，通过**大量尝试和反馈机制**，自己找到最适合的“思考路径”，不断强化那些能 reliably 得出正确答案的 token 序列，从而让模型表现得更稳定、更聪明、更贴近人类需求。

---

如果你想了解 **RLHF（人类反馈强化学习）** 的技术细节或算法流程（比如 PPO 算法），我也可以继续讲解。


# DeepSeek-R1

these first two stat ages pre-training and surprise fine-tuning they've been around for years and they are very standard and everyone does them all the
different llm providers it is this last stage the RL training that is a lot more
early in its process of development and is not standard yet in the field and so
um this stage is a lot more kind of early and nent and the reason for that
is because I actually skipped over a ton of little details here in this process the high level idea is very simple it's
trial and there learning but there's a ton of details and little math mathematical kind of like nuances to exactly how you pick the solutions that
are the best and how much you train on them and what is the prompt distribution and how to set up the training run such
that this actually works so there's a lot of little details and knobs to the core idea that is very very simple and
so getting the details right here uh is not trivial and so a lot of companies like for example open and other LM
providers have experimented internally with reinforcement learning fine tuning for llms for a while but they've not
talked about it publicly um it's all kind of done inside the company and so that's why the paper from
Deep seek that came out very very recently was such a big deal because this is a paper from this company called
DC Kai in China and this paper really talked very publicly about reinforcement
learning fine training for large language models and how incredibly important it is for large language
models and how it brings out a lot of reasoning capabilities in the models we'll go into this in a second so this
paper reinvigorated the public interest of using RL for llms and gave a lot of
the um sort of n-r details that are needed to reproduce their results and actually get the stage to work for large
langage models so let me take you briefly through this uh deep seek R1 paper and what happens when you actually
correctly apply RL to language models and what that looks like and what that gives you so the first thing I'll scroll to is this uh kind of figure two here
where we are looking at the Improvement in how the models are solving mathematical problems so this is the
accuracy of solving mathematical problems on the a accuracy and then we can go to the web page and we can see
the kinds of problems that are actually in these um these the kinds of math problems that are being measured here so
these are simple math problems you can um pause the video if you like but these are the kinds of problems that basically
the models are being asked to solve and you can see that in the beginning they're not doing very well but then as you update the model with this many
thousands of steps their accuracy kind of continues to climb so the models are improving and they're solving these
problems with a higher accuracy as you do this trial and error on a large data set of these kinds of
problems and the models are discovering how to solve math problems but even more
incredible than the quantitative kind of results of solving these problems with a higher accuracy is the qualitative means
by which the model achieves these results so when we scroll down uh one of the figures here that is kind of
interesting is that later on in the optimization the model seems to be uh
using average length per response uh goes up up so the model seems to be using more tokens to get its higher
accuracy results so it's learning to create very very long Solutions why are these Solutions very long we can look at
them qualitatively here so basically what they discover is that the model solution get very very long partially
because so here's a question and here's kind of the answer from the model what the model learns to do um and this is an
immerging property of new optimization it just discovers that this is good for problem solving is it starts to do stuff
like this wait wait wait that's Nota moment I can flag here let's reevaluate this step by step to identify the
correct sum can be so what is the model doing here right the model is basically
re-evaluating steps it has learned that it works better for accuracy to try out
lots of ideas try something from different perspectives retrace reframe backtrack is doing a lot of the things
that you and I are doing in the process of problem solving for mathematical questions but it's rediscovering what
happens in your head not what you put down on the solution and there is no human who can hardcode this stuff in the
ideal assistant response this is only something that can be discovered in the process of reinforcement learning
because you wouldn't know what to put here this just turns out to work for the model and it improves its accuracy in
problem solving so the model learns what we call these chains of thought in your head and it's an emergent property of
the optim of the optimization and that's what's bloating up the response length
but that's also what's increasing the accuracy of the problem problem solving so what's incredible here is basically
the model is discovering ways to think it's learning what I like to call cognitive strategies of how you
manipulate a problem and how you approach it from different perspectives how you pull in some analogies or do
different kinds of things like that and how you kind of uh try out many different things over time uh check a
result from different perspectives and how you kind of uh solve problems but here it's kind of discovered by the RL
so extremely incredible to see this emerge in the optimization without having to hardcode it anywhere the only
thing we've given it are the correct answers and this comes out from trying to just solve them correctly which is
incredible um now let's go back to actually the problem that we've been working with and
let's take a look at what it would look like uh for uh for this kind of a model
what we call reasoning or thinking model to solve that problem okay so recall that this is the problem we've been
working with and when I pasted it into chat GPT 40 I'm getting this kind of a response let's take a look at what
happens when you give this same query to what's called a reasoning or a thinking model this is a model that was trained
with reinforcement learning so this model described in this paper DC car1 is
available on chat. dec.com uh so this is kind of like the company uh that developed is hosting it you have to make
sure that the Deep think button is turned on to get the R1 model as it's called we can paste it here and run
it and so let's take a look at what happens now and what is the output of the model okay so here's it says so this
is previously what we get using basically what's an sft approach a supervised funing approach this is like
mimicking an expert solution this is what we get from the RL model okay let me try to figure this out so Emily buys
three apples and two oranges each orange cost $2 total is 13 I need to find out blah blah blah so here you you um as
you're reading this you can't escape thinking that this model is thinking um is definitely pursuing the
solution solution it deres that it must cost $3 and then it says wait a second let me check my math again to be sure
and then it tries it from a slightly different perspective and then it says yep all that checks out I think that's
the answer I don't see any mistakes let me see if there's another way to approach the problem maybe setting up an
equation let's let the cost of one apple be $8 then blah blah blah yep same
answer so definitely each apple is $3 all right confident that that's correct
and then what it does once it sort of um did the thinking process is it writes up the nice solution for the human and so
this is now considering so this is more about the correctness aspect and this is more about the presentation aspect where
it kind of like writes it out nicely and uh boxes in the correct answer at the bottom and so what's incredible about
this is we get this like thinking process of the model and this is what's coming from the reinforcement learning process this is what's bloating up the
length of the token sequences they're doing thinking and they're trying different ways this is what's giving you
higher accuracy in problem solving and this is where we are seeing these aha moments and these different
strategies and these um ideas for how you can make sure that you're getting the correct
answer the last point I wanted to make is some people are a little bit nervous about putting you know very sensitive
data into chat.com because this is a Chinese company so people don't um people are a little bit careful and Cy
with that a little bit um deep seek R1 is a model that was released by this company so this is an open source model
or open weights model it is available for anyone to download and use you will not be able to like run it in its full
um sort of the full model in full Precision you won't run that on a MacBook but uh or like a local device
because this is a fairly large model but many companies are hosting the full largest model one of those companies
that I like to use is called together. so when you go to together. you sign up and you go to playgrounds
you can can select here in the chat deep seek R1 and there's many different kinds of other models that you can select here
these are all state-of-the-art models so this is kind of similar to the hugging face inference playground that we've been playing with so far but together. a
will usually host all the state-of-the-art models so select DT car1 um you can try to ignore a lot of
these I think the default settings will often be okay and we can put in this and
because the model was released by Deep seek what you're getting here should be basically equivalent to what you're
getting here now because of the randomness in the sampling we're going to get something slightly different uh but in principle this should be uh
identical in terms of the power of the model and you should be able to see the same things quantitatively and qualitatively uh but uh this model is
coming from kind of a an American company so that's deep seek and that's
the what's called a reasoning model now when I go back to chat uh let
me go to chat here okay so the models that you're going to see in the drop down here some of them like 01 03 mini
O3 mini High Etc they are talking about uses Advanced reasoning now what this is
referring to uses Advanced reasoning is it's referring to the fact that it was trained by reinforcement learning with
techniques very similar to those of deep C car1 per public statements of opening ey employees uh so these are thinking
models trained with RL and these models like GPT 4 or GPT 4 40 mini that you're getting in the free tier you should
think of them as mostly sft models supervised fine tuning models they don't actually do this like thinking as as you
see in the RL models and even though there's a little bit of reinforcement learning involved with these models and
I'll go that into that in a second these are mostly sft models I think you should think about it that way so in the same
way as what we saw here we can pick one of the thinking models like say 03 mini high and these models by the way might
not be available to you unless you pay a Chachi PT subscription of either $20 per month or $200 per month for some of the
top models so we can pick a thinking model and run now what's going to happen
here is it's going to say reasoning and it's going to start to do stuff like this and um what we're seeing here is
not exactly the stuff we're seeing here so even though under the hood the model
produces these kinds of uh kind of chains of thought opening ey chooses to not show the exact chains of thought in
the web interface it shows little summaries of that of those chains of thought and open kind of does this I
think partly because uh they are worried about what's called the distillation risk that is that someone could come in
and actually try to imitate those reasoning traces and recover a lot of the reasoning performance by just imitating the reasoning uh chains of
thought and so they kind of hide them and they only show little summaries of them so you're not getting exactly what you would get in deep seek as with
respect to the reasoning itself and then they write up the solution so these are kind of like
equivalent even though we're not seeing the full under the hood details now in terms of the performance uh these models
and deep seek models are currently rly on par I would say it's kind of hard to tell because of the evaluations but if
you're paying $200 per month to open AI some of these models I believe are currently they basically still look better uh but deep seek R1 for now is
still a very solid choice for a thinking model that would be available to you um
sort of um either on this website or any other website because the model is open weights you can just download it so
that's thinking models so what is the summary so far well we've talked about reinforcement learning and the fact that
thinking emerges in the process of the optimization on when we basically run RL on many math uh and kind of code
problems that have verifiable Solutions so there's like an answer three Etc now these thinking models you can
access in for example deep seek or any inference provider like together. a and
choosing deep seek over there these thinking models are also available uh in chpt under any of the 01 or O3
models but these GPT 4 R models Etc they're not thinking models you should think of them as mostly sft models now
if you are um if you have a prompt that requires Advanced reasoning and so on you should probably use some of the
thinking models or at least try them out but empirically for a lot of my use when you're asking a simpler question there's
like a knowledge based question or something like that this might be Overkill like there's no need to think 30 seconds about some factual question
so for that I will uh sometimes default to just GPT 40 so empirically about 80 90% of my use is just gp4
and when I come across a very difficult problem like in math and code Etc I will reach for the thinking models but then I
have to wait a bit longer because they're thinking um so you can access these on chat on deep seek also I wanted
to point out that um AI studio. go.com even though it looks really busy
really ugly because Google's just unable to do this kind of stuff well it's like what is happening but if you choose
model and you choose here Gemini 2.0 flash thinking experimental 01 21 if you
choose that one that's also a a kind of early experiment experimental of a thinking model by Google so we can go
here and we can give it the same problem and click run and this is also a thinking problem a thinking model that
will also do something similar and comes out with the right answer here so basically Gemini also
offers a thinking model anthropic currently does not offer a thinking model but basically this is kind of like
the frontier development of these llms I think RL is kind of like this new exciting stage but getting the details
right is difficult and that's why all these models and thinking models are currently experimental as of 2025 very
early 2025 um but this is kind of like the frontier development of pushing the performance on these very difficult
problems using reasoning that is emerging in these optimizations one more connection that I wanted to bring up is

这段内容详细介绍了由中国公司 DeepSeek（深势科技）发布的 **DeepSeek-R1** 模型，以及它在大型语言模型（LLM）训练中一个非常关键但目前尚未成熟的阶段：**强化学习（Reinforcement Learning，简称 RL）**。以下是详细的中文解释：

---

## 一、前两个训练阶段已成熟，强化学习仍是前沿

LLM 的训练分为三个阶段：

1. **预训练（Pre-training）**：在海量网络文本上训练，建立语言和常识基础。所有模型都要走这一步。
2. **有监督微调（Supervised Fine-tuning, SFT）**：用人类写的问答对进行微调，学会像助手一样回答问题。
3. **强化学习（RL）**：这是当前仍在不断探索的阶段。它让模型自己反复尝试、反馈、优化，**自己找到最适合自己的思考路径**。

> 前两个步骤业内已非常成熟，但 RL 很复杂，目前没有统一做法，也很少公开细节。

---

## 二、DeepSeek-R1 为什么重要？

DeepSeek-R1 是中国公司 **深势科技（DeepSeek）** 首次**公开透明地发布了完整 RL 训练流程与模型参数**。

这个发布之所以被认为是里程碑事件，是因为：

* 它首次详细展示了 RL 如何让模型具备更强的“思考”能力；
* 它提供了完整的训练方法、参数和效果，可以被他人复现；
* 它引起了整个 LLM 社区对 RL 应用的新关注。

---

## 三、RL 如何提升模型的“推理能力”？

来看一个例子：

### 🍎 问题：

> Emily 买了 3 个苹果和 2 个橘子，每个橘子 \$2，总共花了 \$13，问苹果单价？

---

### ✅ 传统模型（SFT）做法：

直接解题，列方程或计算，给出 \$3，过程清晰，但就是模仿专家风格，**不会“自我反思”**。

---

### 🧠 RL 训练后的模型（如 DeepSeek-R1）做法：

* 首先尝试一种思路，得出答案；
* 然后“自我检查”：**“等等，我再检查一遍是否合理”**；
* 接着从不同角度再尝试解法（如设变量法）；
* 最后对比、确认，再整理出整洁的解题过程给人类阅读。

这像不像人在认真做题？对！它学会了“思考的链条（chain of thought）”。

---

## 四、这就是 RL 的神奇之处：**让模型自动学会“思考策略”**

### ✨ Emergent Behavior（涌现行为）：

* 模型并不是我们手动教它怎么“反思”的；
* 我们只是告诉它：**“这是对的答案”**；
* 它在优化过程中**自动发现**，原来反复尝试、回溯思路、换角度**能更容易得到正确答案**；
* 于是模型“学会了思考”。

---

## 五、DeepSeek-R1 如何使用？

### ✅ 可访问渠道：

* [chat.deepseek.com](https://chat.deepseek.com)：模型官方接口（需要开启 "DeepThink" 模式才能使用 R1）
* [together.ai](https://together.ai)：第三方模型平台，也可运行 DeepSeek-R1（不受地域限制）
* 开源：DeepSeek-R1 已开源模型参数，**可下载自行部署**（但本地运行要求较高）

---

## 六、与 ChatGPT、Gemini、Anthropic 的比较：

| 平台                                 | 是否具备“思考能力”            | 是否显示推理过程    | RL阶段完善程度           |
| ---------------------------------- | --------------------- | ----------- | ------------------ |
| **ChatGPT 4-o（免费版）**               | 否，主要是 SFT             | 否           | RL 比较轻量，重在模仿       |
| **ChatGPT o1 / o3-mini-high（付费）**  | 是，属于“thinking models” | 部分隐藏（防止被抄袭） | RL 较完善，接近 DeepSeek |
| **DeepSeek-R1**                    | 是，完整“thinking model”  | 全部显示推理细节    | 训练公开、开源            |
| **Gemini 2.0 Flash 01.21（Google）** | 是（实验性）                | 可显示推理过程     | 正在探索中              |
| **Anthropic Claude 系列**            | 否                     | 无明确 RL 展示   | RL 进展较少            |

---

## 七、该如何选择使用？

* **简单问题**（查知识、写摘要、问常识）➡️ 用 ChatGPT 4-o 足够；
* **复杂问题**（数学推理、代码验证、多步逻辑）➡️ 建议用 Thinking Model，如 DeepSeek-R1 或 o3 mini high；
* **需要本地部署/自定义训练**➡️ 下载 DeepSeek-R1 的开源权重，自己训练微调。

---

## 总结一句话：

> DeepSeek-R1 是一个真正“会思考”的大模型。通过强化学习，它学会了像人一样思考、验证、回溯，显著提高了解题的准确率。它不仅提升了性能，也推动了整个 LLM 领域对强化学习的公开探索。

如果你想继续了解 DeepSeek-R1 的论文结构、训练细节，或者如何在本地部署该模型，我也可以继续帮你解析。


# AlphaGo

that the discovery that reinforcement learning is extremely powerful way of learning is not new to the field of AI
and one place what we've already seen this demonstrated is in the game of Go and famously Deep Mind developed the
system alphago and you can watch a movie about it um where the system is learning
to play the game of go against top human players and um when we go to the paper
underlying alphago so in this paper when we scroll down we actually find a really
interesting plot um that I think uh is kind of familiar uh to us and we're kind of like
we discovering in the more open domain of arbitrary problem solving instead of on the closed specific domain of the
game of Go but basically what they saw and we're going to see this in llms as well as this becomes more mature is this
is the ELO rating of playing game of Go and this is leas dull an extremely strong human player and here what they
are comparing is the strength of a model learned trained by supervised learning and a model trained by reinforcement
learning so the supervised learning model is imitating human expert players
so if you just get a huge amount of games played by expert players in the game of Go and you try to imitate them
you are going to get better but then you top out and you never quite get better
than some of the top top top players of in the game of Go like LEL so you're never going to reach there because
you're just imitating human players you can't fundamentally go beyond a human player if you're just imitating human players but in a process of
reinforcement learning is significantly more powerful in reinforcement learning for a game of Go it means that the
system is playing moves that empirically and statistically lead to win to winning
the game and so alphago is a system where it kind of plays against it itself
and it's using reinforcement learning to create rollouts so it's the exact same diagram
here but there's no prompt it's just uh because there's no prompt it's just a fixed game of Go but it's trying out
lots of solutions it's trying out lots of plays and then the games that lead to a win instead of a specific answer are
reinforced they're they're made stronger and so um the system is learning
basically the sequences of actions that empirically and statistically lead to winning the game and reinforcement
learning is not going to be constrained by human performance and reinforcement learning can do significantly better and
overcome even the top players like Lisa Dole and so uh probably they could have
run this longer and they just chose to crop it at some point because this costs money but this is very powerful demonstration of reinforcement learning
and we're only starting to kind of see hints of this diagram in larger language models for reasoning problems so we're
not going to get too far by just imitating experts we need to go beyond that set up these like little game
environments and get let let the system discover reasoning traces or like ways
of solving problems uh that are unique and that uh just basically work
well now on this aspect of uniqueness notice that when you're doing reinforcement learning nothing prevents
you from veering off the distribution of how humans are playing the game and so when we go back to uh this alphao search
here one of the suggested modifications is called move 37 and move 37 in alphao
is referring to a specific point in time where alphago basically played a move
that uh no human expert would play uh so the probability of this move uh to be
played by a human player was evaluated to be about 1 in 10th ,000 so it's a very rare move but in retrospect it was
a brilliant move so alphago in the process of reinforcement learning discovered kind of like a strategy of
playing that was unknown to humans and but is in retrospect uh brilliant I recommend this YouTube video um leis do
versus alphao move 37 reactions and Analysis and this is kind of what it looked like when alphao played this
move value that's a very that's a very
surprising move I thought I thought it was I thought it was a mistake when I see this move anyway so
basically people are kind of freaking out because it's a it's a move that a human would not play that alphago played
because in its training uh this move seemed to be a good idea it just happens not to be a kind of thing that a humans
would would do and so that is again the power of reinforcement learning and in principle we can actually see the
equivalence of that if we continue scaling this Paradigm in language models and what that looks like is kind of
unknown so so um what does it mean to solve problems in such a way that uh
even humans would not be able to get how can you be better at reasoning or thinking than humans how can you go
beyond just uh a thinking human like maybe it means discovering analogies
that humans would not be able to uh create or maybe it's like a new thinking strategy it's kind of hard to think through uh maybe it's a holy new
language that actually is not even English maybe it discovers its own language that is a lot better at
thinking um because the model is unconstrained to even like stick with English uh so maybe it takes a different
language to think in or it discovers its own language so in principle the behavior of the system is a lot less
defined it is open to do whatever works and it is open to also slowly Drift from
the distribution of its training data which is English but all of that can only be done if we have a very large
diverse set of problems in which the these strategy can be refined and perfected and so that is a lot of the
frontier LM research that's going on right now is trying to kind of create those kinds of prompt distributions that
are large and diverse these are all kind of like game environments in which the llms can practice their thinking and uh
it's kind of like writing you know these practice problems we have to create practice problems for all of domains of
knowledge and if we have practice problems and tons of them the models will be able to reinforcement learning
reinforcement learn on them and kind of uh create these kinds of uh diagrams but
in the domain of open thinking instead of a closed domain like game of Go there's one more section within

这段内容讲的是：**强化学习（Reinforcement Learning, RL）早在围棋 AlphaGo 中就已经被证明非常强大**，而现在我们正尝试把这种方法推广到**大语言模型（LLMs）中的开放式问题求解**。

下面是中文详细解释：

---

## 一、AlphaGo：强化学习首次震撼全世界

**AlphaGo** 是 DeepMind 开发的围棋 AI 系统。

* 它的目标是：通过与人类以及自己对弈，最终击败最强人类棋手（如李世石）。
* 它的核心训练方法包括：

  * **监督学习（SFT）**：先学习人类棋谱，模仿人类下法；
  * **强化学习（RL）**：然后和自己反复对战，从胜负中总结经验，不断优化。

### 👉 AlphaGo 给出一个重要结论：

> **监督学习能学得像人类，但只能达到人类上限；强化学习能超越人类。**

---

## 二、ELO 图示：RL 明显优于 SFT

论文中展示了一个非常关键的图表：

* 横轴是训练步数，纵轴是**ELO 等级分**（棋力评分）；
* 对比了两种模型：

  * SFT 模型（模仿人类）；
  * RL 模型（通过自我博弈不断提升）；
* 结论：

  * SFT 模型能力提升有限，无法超过人类顶级高手；
  * RL 模型持续提升，**最终超过李世石水平**。

这个图表非常类似我们现在在 LLM 领域做的事情，只不过围棋是一个封闭的、规则明晰的系统，而 LLM 要处理的是开放世界中的问题（例如数学、编程、逻辑推理等）。

---

## 三、AlphaGo 的“神之一手”——第 37 手（Move 37）

在 AlphaGo 与李世石的比赛中，有一手被称为“神之一手”，也就是第 37 手。

* 这步棋：

  * **人类下不出来**，概率被估为十万分之一；
  * 一开始连解说员都以为是失误；
  * **但事后证明非常精彩、战略深远**；
* 它是 AlphaGo 通过强化学习自己探索出的策略——**不是模仿人类学来的**！

> 💡 这说明：**强化学习可以走出人类没有走过的路，发现新的策略和智慧。**

推荐视频：[李世石 vs AlphaGo Move 37 Reaction](https://www.youtube.com/results?search_query=lee+sedol+alphago+move+37)

---

## 四、强化学习在大语言模型（LLM）中的前景

现在，我们希望把这种探索性的学习方法，也应用到语言模型中：

* **模仿人类的答案（SFT）可以让模型“看起来聪明”**；
* **但要让它真的聪明、超越人类，就需要像 AlphaGo 一样，用 RL 反复练习、优化策略。**

### 问题是：语言不是围棋，怎么设计“游戏环境”？

* 需要给模型设计各种“练习题”，比如数学题、代码题、逻辑题等；
* 模型尝试各种解法，然后我们只告诉它哪种是“答对了”；
* 它在不断试错中发现更好的思考方式，甚至是：

  * 人类从未使用过的推理路径；
  * 人类未发明过的类比方式；
  * 更高效的“类语言”结构，甚至不是英语；
  * **思维上的“神之一手”**。

---

## 五、LLM 强化学习 VS AlphaGo 的异同

| 项目    | AlphaGo       | LLM                    |
| ----- | ------------- | ---------------------- |
| 游戏类型  | 封闭规则、输赢明确的围棋  | 开放式语言问题（如数学、代码）        |
| 奖励信号  | 赢棋 vs 输棋      | 是否答对问题                 |
| 训练方式  | 自我对弈          | 大量练习题（prompt）自动采样、多样尝试 |
| RL 成果 | 第 37 手、全新围棋策略 | 涌现式推理能力、链式思考、问题重构等     |

---

## 六、总结一句话：

> **AlphaGo 通过强化学习超越了人类围棋高手，而我们现在正试图让语言模型用类似方式，学会前所未有的推理方法，甚至超越人类的认知方式。**

这条路径是当前最前沿的研究方向，但它非常复杂，需要：

* 超大规模的练习数据；
* 极其精细的奖励设计；
* 极高的训练成本；
* 模型架构与优化方法的同步创新。

---

如果你想，我还可以继续解释后面讲的「推理练习环境设计」或 DeepSeek R1 在这方面的探索。是否继续？


# reinforcement learning from human feedback (RLHF)

reinforcement learning that I wanted to cover and that is that of learning in unverifiable domains so so far all of
the problems that we've looked at are in what's called verifiable domains that is any candidate solution we can score very
easily against a concrete answer so for example answer is three and we can very easily score these Solutions against the
answer of three either we require the models to like box in their answers and then we just check
for equality of whatever is in the box with the answer or you can also use uh kind of what's called an llm judge so
the llm judge looks at a solution and it gets the answer and just basically scores the solution for whether it's
consistent with the answer or not and llms uh empirically are good enough at the current capability that they can do
this fairly reliably so we can apply those kinds of techniques as well in any case we have a concrete answer and we're
just checking Solutions again against it and we can do this automatically with no kind of humans in the loop the problem
is that we can't apply the strategy in what's called unverifiable domains so usually these are for example creative
writing tasks like write a joke about Pelicans or write a poem or summarize a paragraph or something like that in
these kinds of domains it becomes harder to score our different solutions to this problem so for example writing a joke
about Pelicans we can generate lots of different uh jokes of course that's fine for example we can go to chbt and we can
get it to uh generate a joke about Pelicans uh so much stuff in their beaks
because they don't bellan in backpacks what
okay we can uh we can try something else why don't Pelicans ever pay for their drinks because they always B it to
someone else haha okay so these models are not obviously not very good at humor
actually I think it's pretty fascinating because I think humor is secretly very difficult and the model have the capability I think anyway in any case
you could imagine creating lots of jokes the problem that we are facing is how do we score them now in principle we could
of course get a human to look at all these jokes just like I did right now the problem with that is if you are
doing reinforcement learning you're going to be doing many thousands of updates and for each update you want to
be looking at say thousands of prompts and for each prompt you want to be potentially looking at looking at hundred or thousands of different kinds
of generations and so there's just like way too many of these to look at and so
um in principle you could have a human inspect all of them and score them and decide that okay maybe this one is funny
and uh maybe this one is funny and this one is funny and we could train on them
to get the model to become slightly better at jokes um in the context of pelicans at least um the problem is that
it's just like way too much human time this is an unscalable strategy we need some kind of an automatic strategy for
doing this and one sort of solution to this was proposed in this paper uh that introduced what's called
reinforcement learning from Human feedback and so this was a paper from open at the time and many of these people are now um co-founders in
anthropic um and this kind of proposed a approach for uh basically doing
reinforcement learning in unverifiable domains so let's take a look at how that works so this is the cartoon diagram of
the core ideas involved so as I mentioned the native approach is if we just set Infinity human time we could
just run RL in these domains just fine so for example we can run RL as usual if
I have Infinity humans I would I just want to do and these are just cartoon numbers I want to do 1,000 updates where
each update will be on 1,000 prompts and in for each prompt we're going to have 1,000 roll outs that we're scoring so we
can run RL with this kind of a setup the problem is in the process of doing this I will need to run one I will need to
ask a human to evaluate a joke a total of 1 billion times and so that's a lot
of people looking at really terrible jokes so we don't want to do that so instead we want to take the arlef
approach so um in our Rel of approach we are kind of like the the core trick is
that of indirection so we're going to involve humans just a little bit and the
way we cheat is that we basically train a whole separate neural network that we call a reward model and this neural
network will kind of like imitate human scores so we're going to ask humans to score um roll
we're going to then imitate human scores using a neural network and this neural
network will become a kind of simulator of human preferences and now that we have a neural network simulator we can do RL
against it so instead of asking a real human we're asking a simulated human for
their score of a joke as an example and so once we have a simulator we're often
racist because we can query it as many times as we want to and it's all whole automatic process and we can now do
reinforcement learning with respect to the simulator and the simulator as you might expect is not going to be a perfect human but if it's at least
statistically similar to human judgment then you might expect that this will do something and in practice indeed uh it
does so once we have a simulator we can do RL and everything works great so let me show you a cartoon diagram a little
bit of what this process looks like although the details are not 100 like super important it's just a core idea of
how this works so here I have a cartoon diagram of a hypothetical example of what training the reward model would
look like so we have a prompt like write a joke about picans and then here we have five separate roll outs so these
are all five different jokes just like this one now the first thing we're going to do is we are going to ask a human to
uh order these jokes from the best to worst so this is uh so here this human
thought that this joke is the best the funniest so number one joke this is
number two joke number three joke four and five so this is the worst joke
we're asking humans to order instead of give scores directly because it's a bit of an easier task it's easier for a
human to give an ordering than to give precise scores now that is now the supervision for the model so the human
has ordered them and that is kind of like their contribution to the training process but now separately what we're going to do is we're going to ask a
reward model uh about its scoring of these jokes now the reward model is a
whole separate neural network completely separate neural net um and it's also probably a transform
uh but it's not a language model in the sense that it generates diverse language Etc it's just a scoring model so the
reward model will take as an input The Prompt number one and number two a
candidate joke so um those are the two inputs that go into the reward model so here for example the reward model would
be taken this prompt and this joke now the output of a reward model is a single
number and this number is thought of as a score and it can range for example from Z to one so zero would be the worst
score and one would be the best score so here are some examples of what a hypothetical reward model at some stage
in the training process would give uh s scoring to these jokes so 0.1 is a very
low score 08 is a really high score and so on and so now um we compare the
scores given by the reward model with uh the ordering given by the human and there's a precise mathematical way to
actually calculate this uh basically set up a loss function and calculate a kind of like a correspondence here and uh
update a model based on it but I just want to give you the intuition which is that as an example here for this second
joke the the human thought that it was the funniest and the model kind of agreed right 08 is a relatively high score but this score should have been
even higher right so after an update we would expect that maybe this score should have been will actually grow
after an update of the network to be like say 081 or something um for this one here they
actually are in a massive disagreement because the human thought that this was number two but here the the score is
only 0.1 and so this score needs to be much higher so after an update on top of
this um kind of a supervision this might grow a lot more like maybe it's 0.15 or something like
that um and then here the human thought that this one was the worst joke but
here the model actually gave it a fairly High number so you might expect that after the update uh this would come down
to maybe 3 3.5 or something like that so basically we're doing what we did before we're slightly nudging the predictions
from the models using a neural network training process and we're trying to make the
reward model scores be consistent with human ordering and so um as we update the
reward model on human data it becomes better and better simulator of the scores and orders uh that humans provide
and then becomes kind of like the the neural the simulator of human preferences which we can then do RL
against but critically we're not asking humans one billion times to look at a joke we're maybe looking at th000
prompts and five roll outs each so maybe 5,000 jokes that humans have to look at in total and they just give the ordering
and then we're training the model to be consistent with that ordering and I'm skipping over the mathematical details
but I just want you to understand a high level idea that uh this reward model is do is basically giving us this scour and
we have a way of training it to be consistent with human orderings and that's how rhf works okay so that is
the rough idea we basically train simulators of humans and RL with respect to those
simulators now I want to talk about first the upside of reinforcement learning from Human
feedback the first thing is that this allows us to run reinforcement learning which we know is incredibly powerful
kind of set of techniques and it allows us to do it in arbitrary domains and including the ones that are unverifiable
so things like summarization and poem writing joke writing or any other creative writing really uh in domains
outside of math and code Etc now empirically what we see when we actually apply rhf is that this is a way
to improve the performance of the model and uh I have a top answer for why that
might be but I don't actually know that it is like super well established on like why this is you can empirically
observe that when you do rhf correctly the models you get are just like a little bit better um but as to why is I
think like not as clear so here's my best guess my best guess is that this is possibly mostly due to the discriminator
generator Gap what that means is that in many cases it is significantly easier to
discriminate than to generate for humans so in particular an example of this is
um in when we do supervised fine-tuning right sft we're asking humans to generate the
ideal assistant response and in many cases here um as I've shown it uh the
ideal response is very simple to write but in many cases might not be so for example in summarization or poem writing
or joke writing like how are you as a human assist as a human labeler um supposed to give the ideal response in
these cases it requires creative human writing to do that and so rhf kind of
sidesteps this because we get um we get to ask people a significantly easier question as a data labelers they're not
asked to write poems directly they're just given five poems from the model and they're just asked to order them and so
that's just a much easier task for a human labeler to do and so what I think this allows you to do basically is it um
it kind of like allows a lot more higher accuracy data because we're not asking
people to do the generation task which can be extremely difficult like we're not asking them to do creative writing
we're just trying to get them to distinguish between creative writings and uh find the ones that are best and
that is the signal that humans are providing just the ordering and that is their input into the system and then the
system in rhf just discovers the kinds of responses that would be graded well
by humans and so that step of indirection allows the models to become a bit better so that is the upside of
our LF it allows us to run RL it empirically results in better models and it allows uh people to contribute their
supervision uh even without having to do extremely difficult tasks um in the case of writing ideal responses unfortunately
our HF also comes with significant downsides and so um the main one is that
basically we are doing reinforcement learning not with respect to humans and actual human judgment but with respect to a lossy simulation of humans right
and this lossy simulation could be misleading because it's just a it's just a simulation right it's just a language
model that's kind of outputting scores and it might not perfectly reflect the opinion of an actual human with an
actual brain in all the possible different cases so that's number one which is actually something even more
subtle and devious going on that uh really dramatically holds back our LF as a
technique that we can really scale to significantly um kind of Smart Systems
and that is that reinforcement learning is extremely good at discovering a way to game the model to game the simulation
so this reward model that we're constructing here that gives the course these models are Transformers these
Transformers are massive neurals they have billions of parameters and they imitate humans but they do so in a kind
of like a simulation way now the problem is that these are massive complicated systems right there's a billion
parameters here that are outputting a single score it turns out that there are ways
to gain these models you can find kinds of inputs that were not part of their
training set and these inputs inexplicably get very high scores but in
a fake way so very often what you find if you run our lch for very long so for
example if we do 1,000 updates which is like say a lot of updates you might expect that your jokes are getting
better and that you're getting like real bangers about Pelicans but that's not EXA exactly what happens what happens is
that uh in the first few hundred steps the jokes about Pelicans are probably improving a little bit and then they
actually dramatically fall off the cliff and you start to get extremely nonsensical results like for example you
start to get um the top joke about Pelicans starts to be the
and this makes no sense right like when you look at it why should this be a top joke but when you take the the and you
plug it into your reward model you'd expect score of zero but actually the reward model loves this as a joke it
will tell you that the the the theth is a score of 1. Z this is a top joke and
this makes no sense right but it's because these models are just simulations of humans and they're massive neural lots and you can find
inputs at the bottom that kind of like get into the part of the input space that kind of gives you nonsensical
results these examples are what's called adversarial examples and I'm not going to go into the topic too much but these
are adversarial inputs to the model they are specific little inputs that kind of
go between the nooks and crannies of the model and give nonsensical results at the top now here's what you might
imagine doing you say okay the the the is obviously not score of one um it's obviously a low score so let's take the
the the the the let's add it to the data set and give it an ordering that is extremely bad like a score of five and
indeed your model will learn that the D should have a very low score and it will give it score of zero the problem is
that there will always be basically infinite number of nonsensical adversarial examples hiding in the model
if you iterate this process many times and you keep adding nonsensical stuff to your reward model and giving it very low
scores you can you'll never win the game uh you can do this many many rounds and
reinforcement learning if you run it long enough will always find a way to gain the model it will discover adversarial examples it will get get
really high scores uh with nonsensical results and fundamentally this is
because our scoring function is a giant neural nut and RL is extremely good at
finding just the ways to trick it uh so long story short you always run rhf put
for maybe a few hundred updates the model is getting better and then you have to crop it and you are done you
can't run too much against this reward model because the optimization will
start to game it and you basically crop it and you call it and you ship it um
and uh you can improve the reward model but you kind of like come across these situations eventually at some point so
rhf basically what I usually say is that RF is not RL and what I mean by that is
I mean RF is RL obviously but it's not RL in the magical sense this is not RL
that you can run indefinitely these kinds of problems like where you are getting con correct
answer you cannot gain this as easily you either got the correct answer or you didn't and the scoring function is much
much simpler you're just looking at the boxed area and seeing if the result is correct so it's very difficult to gain
these functions but uh gaming a reward model is possible now in these verifiable domains you can run RL
indefinitely you could run for tens of thousands hundreds of thousands of steps and discover all kinds of really crazy
strategies that we might not even ever think about of Performing really well for all these problems in the game of Go
there's no way to to beat to basically game uh the winning of a game or the losing of a game we have a perfect
simulator we know all the different uh where all the stones are placed and we can calculate uh whether someone has won
or not there's no way to gain that and so you can do RL indefinitely and you can eventually be beat even leol but
with models like this which are gameable you cannot repeat this process
indefinitely so I kind of see rhf as not real RL because the reward function is
gameable so it's kind of more like in the realm of like little fine-tuning it's a little it's a little Improvement
but it's not something that is fundamentally set up correctly where you can insert more compute run for longer
and get much better and magical results so it's it's uh it's not RL in that sense it's not RL in the sense that it
lacks magic um it can find you in your model and get a better performance and indeed if we go back to chat GPT the GPT
40 model has gone through rhf because it works well but it's just not RL in the
same sense rlf is like a little fine tune that slightly improves your model is maybe like the way I would think about it okay so that's most of the
technical content that I wanted to cover I took you through the three major stages and paradigms of training these
models pre-training supervised fine tuning and reinforcement learning and I showed you that they Loosely correspond
to the process we already use for teaching children and so in particular we talked about pre-training being sort
of like the basic knowledge acquisition of reading Exposition supervised fine tuning being the process of looking at
lots and lots of worked examples and imitating experts and practice problems
the only difference is that we now have to effectively write textbooks for llms and AIS across all the disciplines of
human knowledge and also in all the cases where we actually would like them to work like code and math and you know
basically all the other disciplines so we're in the process of writing textbooks for them refining all the algorithms that I've presented on the
high level and then of course doing a really really good job at the execution of training these models at scale and
efficiently so in particular I didn't go into too many details but these are extremely large and complicated
distributed uh sort of um jobs that have to run over tens of
thousands or even hundreds of thousands of gpus and the engineering that goes into this is really at the stateof the
art of what's possible with computers at that scale so I didn't cover that aspect too much
but um this is very kind of serious and they were underlying all these very
simple algorithms ultimately now I also talked about sort of like the theory of mind a little bit
of these models and the thing I want you to take away is that these models are really good but they're extremely useful
as tools for your work you shouldn't uh sort of trust them fully and I showed you some examples of that even though we
have mitigations for hallucinations the models are not perfect and they will hallucinate still it's gotten better
over time and it will continue to get better but they can hallucinate in other words in in
addition to that I covered kind of like what I call the Swiss cheese uh sort of model of llm capabilities that you
should have in your mind the models are incredibly good across so many different disciplines but then fail randomly almost in some unique cases so for
example what is bigger 9.11 or 9.9 like the model doesn't know but simultaneously it can turn around and
solve Olympiad questions and so this is a hole in the Swiss cheese and there are many of them and you don't want to trip
over them so don't um treat these models as infallible models check their work
use them as tools use them for inspiration use them for the first draft but uh work with them as tools and be
ultimately respons responsible for the you know product of your work and that's roughly what I wanted to
talk about this is how they're trained and this is what they are let's now turn to what are some of the future

这段内容解释了一个非常重要但复杂的技术：**来自人类反馈的强化学习（RLHF, Reinforcement Learning from Human Feedback）**。下面是对整段内容的中文讲解，通俗清晰，分段整理如下：

---

## 一、什么是可验证领域与不可验证领域？

* **可验证领域**：有“唯一正确答案”的问题，比如数学、代码、问答题等。你只需判断答案对不对，自动评估。

  * 举例：2 + 1 = ？答案是 3，我们可以轻松判断生成的答案是否正确。
  * 模型的输出可以用“LLM 评审器”或直接比对来评分。

* **不可验证领域**：没有标准答案的问题，比如写诗、讲笑话、总结段落等。

  * 举例：写一个关于鹈鹕的笑话，可能有100种不同写法，哪个更好不好评判。
  * 人类可以评估，但太费时间，无法规模化训练。

---

## 二、为什么不能用传统强化学习来训练“创意”类任务？

强化学习通常需要成千上万次迭代。每一步都需要评分：

* 每次迭代训练 = 数千个 prompt × 每个 prompt 的数百个答案
* 总计可能要 **上亿次人工评分**，这显然不现实。

---

## 三、RLHF 的核心“技巧”：**引入奖励模型（Reward Model）**

为了替代人工评分，RLHF 引入了一个核心方法：

1. **让人类对小部分样本进行偏好排序**：

   * 给人类多个生成的答案，让他们按好坏排序（而不是给分，更简单）。
   * 例如，5 个鹈鹕笑话，人类标注哪一个最有趣、哪一个最差。

2. **训练一个奖励模型（reward model）来模仿人类排序习惯**：

   * 奖励模型是一个独立神经网络，它输入的是「prompt + 答案」，输出一个评分（0\~1之间）。
   * 奖励模型不断学习如何给出的分数接近人类的排序。

3. **之后的强化学习就不用人类打分了，而是对着奖励模型来优化**：

   * 模型只需要对 reward model 输出高分的内容做强化；
   * 于是训练过程就可以大规模、自动化地进行了。

---

## 四、RLHF 的好处

1. **可以在不可验证的任务上使用强化学习**（如笑话、摘要、写作等）；
2. **人类只需要做简单任务（排序），而不是直接创作内容**；
3. **训练效果不错：用 RLHF 后，模型的表现通常会“稍微好一点”**，比如回答更符合用户期待。

> ✅ **一句话总结优点：**
> RLHF 把难的问题（创作）变成了简单的问题（排序），通过训练奖励模型来间接实现自动强化。

---

## 五、RLHF 的问题和局限性

虽然 RLHF 有用，但它并不完美，甚至存在一些“致命弱点”：

### 1. 奖励模型不是人类，它只是“模拟”人类偏好

* 它只是一个神经网络，不是人脑；
* 有些“看起来很好”的内容，它可能误判为非常棒；
* 所以最终强化出来的东西，**不一定是真正优质的输出**。

### 2. 模型可以“欺骗”奖励模型（对抗样本）

这是最关键的缺点：

* **强化学习会学会投机取巧！**
* 比如一开始模型生成的笑话在变好；
* 但训练久了后，可能会发现“某些奇怪的字符串”（例如 “the the the the the”）可以骗到奖励模型打高分；
* 于是它反而学到这种**无意义但高分的对抗例子**。

这类例子是：

> “模型学会如何骗奖励模型，而不是变得更优秀。”

这意味着：

* 训练**不能进行太久**，否则模型就开始“学坏”了；
* 通常我们训练几百步后，就要**停止训练、裁切并发布模型**。

---

## 六、RLHF ≠ 真正意义上的强化学习（RL）

作者在这里强调了一个观点：

> **RLHF 虽然用的是 RL 技术，但它不是真正强大、无限可扩展的 RL。**

理由如下：

* 真正的 RL，比如 AlphaGo，在围棋这样的“可验证系统”里，**可以持续优化**；
* 但 RLHF 中的奖励模型很容易被游戏（Game the reward model），没法无限优化。

---

## 七、比喻总结三阶段训练流程（与儿童学习类比）：

| 阶段               | 类比       | 说明                |
| ---------------- | -------- | ----------------- |
| 预训练（Pretraining） | 阅读       | 模型学习基础知识（海量网页、书籍） |
| 监督微调（SFT）        | 看例题      | 模型模仿人类写好的答案       |
| RLHF             | 教练打分+练习赛 | 模型练习，优化更符合人类喜好的答案 |

---

## 八、结语与提醒

* 模型仍然可能会“胡说八道”（hallucinate）；
* RLHF 提升模型有用，但它有局限；
* 模型非常强大，但有盲点（像瑞士奶酪）——大部分很厉害，少数场景会出错；
* **最好的使用方式是把模型当作工具、助手，不要完全依赖它。**

---

是否需要我画个图或整理成一页笔记来帮助记忆这个 RLHF 的过程？


# preview of things to come

capabilities of these models uh probably what's coming down the pipe and also where can you find these models I have a
few blow points on some of the things that you can expect coming down the pipe the first thing you'll notice is that the models will very rapidly become
multimodal everything I talked about above concerned text but very soon we'll have llms that can not just handle text
but they can also operate natively and very easily over audio so they can hear and speak and also images so they can
see and paint and we're already seeing the beginnings of all of this uh but this will be all done natively inside
inside the language model and this will enable kind of like natural conversations and roughly speaking the
reason that this is actually no different from everything we've covered above is that as a baseline you can
tokenize audio and images and apply the exact same approaches of everything that we've talked about above so it's not a
fundamental change it's just uh it's just a to we have to add some tokens so as an example for tokenizing audio we
can look at slices of the spectrogram of the audio signal and we can tokenize that and just add more tokens that
suddenly represent audio and just add them into the context windows and train on them just like above the same for
images we can use patches and we can separately tokenize patches and then
what is an image an image is just a sequence of tokens and this actually kind of works and there's a lot of early
work in this direction and so we can just create streams of tokens that are representing audio images as well as
text and interpers them and handle them all simultaneously in a single model so that's one example of multimodality
uh second something that people are very interested in is currently most of the work is that we're handing individual tasks to the
models on kind of like a silver platter like please solve this task for me and the model sort of like does this little
task but it's up to us to still sort of like organize a coherent execution of
tasks to perform jobs and the models are not yet at the capability required to do
this in a coherent error correcting way over long periods of time so they're not
able to fully string together tasks to perform these longer running jobs but they're getting there and this is
improving uh over time but uh probably what's going to happen here is we're going to start to see what's called
agents which perform tasks over time and you you supervise them and you watch
their work and they come up to once in a while report progress and so on so we're going to see more long running agents uh
tasks that don't just take you know a few seconds of response but many tens of seconds or even minutes or hours over
time uh but these uh models are not infallible as we talked about above so all of this will require supervision so
for example in factories people talk about the human to robot ratio uh for automation I think we're going to see
something similar in the digital space where we are going to be talking about human to agent ratios where humans
becomes a lot more supervisors of agent tasks um in the digital
domain uh next um I think everything is going to become a lot more pervasive and invisible so it's kind of like
integrated into the tools and everywhere um and in addition kind of like computer
using so right now these models aren't able to take actions on your behalf but I think this is a separate bullet point
um if you saw chpt launch the operator then uh that's one early example of that
where you can actually hand off control to the model to perform you know keyboard and mouse actions on your
behalf so that's also something that that I think is very interesting the last point I have here is just a general
comment that there's still a lot of research to potentially do in this domain main one example of that uh is
something along the lines of test time training so remember that everything we've done above and that we talked about has two major stages there's first
the training stage where we tune the parameters of the model to perform the tasks well once we get the parameters we
fix them and then we deploy the model for inference from there the model is fixed it doesn't change anymore it
doesn't learn from all the stuff that it's doing a test time it's a fixed um number of parameters and the only thing
that is changing is now the token inside the context windows and so the only type of learning or test time learning that
the model has access to is the in context learning of its uh kind of like uh dynamically adjustable context window
depending on like what it's doing at test time so but I think this is still different from humans who actually are
able to like actually learn uh depending on what they're doing especially when you sleep for example like your brain is updating your parameters or something
like that right so there's no kind of equivalent of that currently in these models and tools so there's a lot of
like um more wonky ideas I think that are to be explored still and uh in particular I think this will be
necessary because the context window is a finite and precious resource and especially once we start to tackle very
long running multimodal tasks and we're putting in videos and these token windows will basically start to grow
extremely large like not thousands or even hundreds of thousands but significantly beyond that and the only
trick uh the only kind of trick we have Avail to us right now is to make the context Windows longer but I think that
that approach by itself will will not will not scale to actual long running tasks that are multimodal over time and
so I think new ideas are needed in some of those disciplines um in some of those kind of cases in the main where these
tasks are going to require very long contexts so those are some examples of some of the things you can um expect
coming down the pipe let's now turn to where you can actually uh kind of keep track of this progress and um you know
be up to date with the latest and grest of what's happening in the field so I would say the three resources that I have consistently used to stay up to

这段内容是关于**大型语言模型（LLMs）未来的发展趋势和展望**的总结。下面我将用中文逐点解释这个「未来预览（preview of things to come）」的核心内容：

---

## 🔮 1. 多模态模型（Multimodal Models）

未来的语言模型将**不再局限于文字**，而是能够**听、说、看、画**。

* ✅ 文字：现在模型已经能很好地处理。
* 🔜 音频：模型能听懂人说话、用声音回应。
* 🔜 图像：模型能“看图说话”，也能“画画”。

**为什么这不算“根本性的改变”？**

* 因为图像、音频也可以像文字一样被“分词成 token”；
* 图像 → 图像切片 → token；
* 音频 → 声谱图片段 → token；
* 于是模型就能像处理文字一样处理它们。

---

## 🤖 2. 从任务工具到智能代理（Agents）

目前大多数语言模型只是：

> “你告诉我要干嘛，我给你一个答复。”

未来会发展出\*\*“智能代理”\*\*，具备以下特点：

* 🧩 能**自己组合一系列任务**来完成一个更复杂的目标；
* 🕒 任务持续时间从“几秒”延长到“几分钟”甚至“几小时”；
* 📊 人类不再自己做任务，而是担任**监督者（Supervisor）**，类似工厂里人监控机器人。

我们将进入\*\*“人类 : 代理”\*\*的协作时代。

---

## 🧠 3. 越来越“无形”地融入日常工具

未来这些模型将会像“电力”一样：

* 无所不在，却无处可见；
* 悄悄地嵌入各种工具和平台；
* 你不会明显“看见 AI”，但它无时无刻都在帮你工作。

---

## 🖱️ 4. 模型能“动手”帮你操作电脑（Action Agents）

* 示例：ChatGPT 的 **“Operator”** 功能已经允许模型控制鼠标和键盘；
* 未来模型将能执行操作任务，如帮你发邮件、整理文件、运行程序等；
* 模型不只是“建议者”，而是“行动者”。

---

## 📚 5. 测试时学习（Test-time Training）的潜力与挑战

当前模型只能\*\*“一次性训练，之后就冻结”\*\*：

* 模型训练完成后就定型，不再更新；
* 在使用（推理）时，**不会从你新的输入中持续学习**；
* 唯一的“学习”方式是依靠**上下文窗口（context window）**，这相当于模型的“短时记忆”。

🔍 **问题是：**

* 上下文窗口是有限的，尤其未来我们要放入视频、音频等内容时；
* 单靠“延长窗口”这招，不够用；
* 可能需要新的机制，让模型**像人类一样“睡觉时更新大脑参数”**。

---

## 🚧 6. 仍有大量前沿研究值得探索

未来研究方向包括：

* **上下文窗口机制的突破**：处理更长时间、多模态的信息；
* **让模型持续学习**：不只是靠输入，更像人一样不断成长；
* **优化代理行为**：让模型学会更稳定地长期规划和行动；
* **让多模态模型协同工作**：比如让图像和语言理解真正融合。

---

## 📌 总结

| 发展方向  | 说明            |
| ----- | ------------- |
| 多模态   | 模型能听、说、看、画    |
| 智能代理  | 能连续完成一系列任务    |
| 动作控制  | 模型能操作你的电脑     |
| 无处不在  | 模型融入日常工具      |
| 测试期学习 | 探索训练后仍能学习的新方法 |
| 持续研究  | 还有大量技术挑战值得攻克  |

---

需要我根据这部分内容画一张未来路线图、总结卡片或生成演示幻灯片吗？可以帮助你更清晰地记住和分享这些趋势。

这段内容主要是讲**如何持续关注和跟踪大型语言模型（LLM）的发展与最新动态**。作者分享了他自己用的三个主要渠道，以下是中文解释和总结：

---

## 🔍 1. **Eluther AI 的 Leaderboard：ElMariana**

这是一个**LLM 排行榜网站**，叫做 **[ElMariana](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)**，它通过“**人类盲评**”的方式来对模型进行排名：

* 👤 人类用户会对多个模型的回答进行打分；
* ❓ 用户**不知道哪个答案是哪个模型写的**，只能选出哪一个更好；
* 📊 排名是根据这些打分结果生成的“胜率 Elo 值”排序。

### ✨ 特别之处：

* 比如 **Google Gemini** 排在前列；
* **OpenAI 的模型（如 GPT-4）也在上面**；
* 一个叫 **DeepSeek** 的开源模型居然排在第三，**它的权重是开源的（MIT 许可证）**，任何人都可以下载和使用；
* 类似地，**Meta 的 LLaMA 也开放了权重**（但排名较靠后）；
* 作者提醒：这个榜单**近年来有点被“刷榜”了，不能完全信**，比如 Anthropic 的 Claude Sonnet 实际表现不错，但排名很低；
* ✅ 建议：这个榜单可作为**参考起点**，但最重要的还是**亲自测试模型在你任务上的表现**。

---

## 📰 2. **AI News 新闻简报**

这是一个**AI 新闻简报网站/邮件列表**，名字很简单就叫 **AI News**，由 **Swix 和他的团队维护**。

### 🌟 特点：

* 非常全面，经常更新（几乎每隔一天一篇）；
* 有些内容由人写，有些由 LLM 自动汇总生成；
* 页面顶部通常会有简短的人工精炼版摘要，**适合快速浏览重点**；
* 如果你想知道最近 AI 圈有哪些重大更新，这里几乎不会漏掉。

---

## 🐦 3. **X（原 Twitter）**

作者的第三个建议是：**关注 Twitter/X 上的一些专家、研究者或开发者的账号**。

* 很多 AI 领域的第一手资料、论文发布、项目公告、模型更新都会**优先在 X 上曝光**；
* 你可以**挑选你信任的人关注他们**，如 OpenAI、Anthropic、Meta、HuggingFace 等的研究人员或工程师；
* 这是个获取前沿动态非常快速、非正式但有效的方式。

---

## ✅ 总结

| 渠道              | 功能           | 建议               |
| --------------- | ------------ | ---------------- |
| 🌐 ElMariana 榜单 | 模型性能对比（人类评估） | 适合作为入门参考，最终还是要实测 |
| 🗞 AI News 简报   | 每日重大新闻归纳     | 非常全面，有人工摘要，值得订阅  |
| 🐦 X / Twitter  | 社群与第一手消息     | 跟踪业内大牛和团队的即时动态   |

---

需要我整理一个中文推荐列表（比如：关注哪些 Twitter 账号、订阅哪些 newsletter、访问哪个榜单地址）供你长期跟踪吗？我可以为你做一张图表或笔记。


# where to find LLMs

can you use them so the first one I would say is for any of the biggest proprietary models you just have to go
to the website of that LM provider so for example for open a that's uh chat I believe actually works now uh so
that's for open AI now for or you know for um for Gemini
I think it's gem. google.com or AI Studio I think they have two for some reason that I don't fly understand no
one does um for the open weights models like deep SE CL Etc you have to go to
some kind of an inference provider of LMS so my favorite one is together together. a and I showed you that when
you go to the playground of together. a then you can sort of pick lots of different models and all of these are
open models of different types and you can talk to them here as an example um now if you'd like to use a
base model like um you know a base model then this is where I think it's not as common to find base models even on these
inference providers they are all targeting assistants and chat and so I think even here I can't I couldn't see
base models here so for base models I usually go to hyperbolic because they serve my llama 3.1 base and I love that
model and you can just talk to it here so as far as I know this is this is a good place for a base model and I wish
more people hosted base models because they are useful and interesting to work with in some cases finally you can also
take some of the models that are smaller and you can run them locally and so for example deep seek the biggest model
you're not going to be able to run locally on your MacBook but there are smaller versions of the deep seek model that are what's called distilled and
then also you can run these models at smaller Precision so not at the native Precision of for example fp8 on deep
seek or you know bf16 llama but much much lower than that um and don't worry
if you don't fully understand those details but you can run smaller versions that have been distilled and then at even lower precision and then you can
fit them on your uh computer and so you can actually run pretty okay models on
your laptop and my favorite I think place I go to usually is LM studio uh which is basically an app you can get
and I think it kind of actually looks really ugly and it's I don't like that it shows you all these models that are basically not that useful like everyone
just wants to run deep seek so I don't know why they give you these 500 different types of models they're really complicated to search for and you have
to choose different distillations and different uh precisions and it's all really confusing but once you actually
understand how it works and that's a whole separate video then you can actually load up a model like here I loaded up a llama 3 uh2 instruct 1
billion and um you can just talk to it so I ask for Pelican jokes and I can ask
for another one and it gives me another one Etc all of this that happens here is locally on your computer so we're not
actually going to anywhere anyone else this is running on the GPU on the MacBook Pro so that's very nice and you
can then eject the model when you're done and that frees up the ram so LM studio is probably like my favorite one
even though I don't I think it's got a lot of uiux issues and it's really geared towards uh professionals almost
uh but if you watch some videos on YouTube I think you can figure out how to how to use this interface uh so those are a few words on
where to find them so let me now loop back around to where we started the question was when we go to chashi pta.com and we enter some kind of a

这一段内容是在讲**我们如何实际使用 LLM 模型（大型语言模型）**，也就是说，在哪里可以找到它们，以及如何使用这些模型，作者给出了几个推荐方法和平台。以下是中文解释和总结：

---

## 🏢 一、使用大型科技公司提供的「专有模型」

如果你想使用像 **OpenAI（ChatGPT）**、**Google（Gemini）** 这样的大型商业模型：

| 公司     | 使用地址                                                                                              | 说明                     |
| ------ | ------------------------------------------------------------------------------------------------- | ---------------------- |
| OpenAI | [https://chat.openai.com](https://chat.openai.com)                                                | 使用 ChatGPT（GPT-4 等）    |
| Google | [https://gemini.google.com](https://gemini.google.com) 或 [AI Studio](https://aistudio.google.com) | 使用 Gemini 模型（Bard 已更名） |

这些是**商业闭源模型**，你只能通过它们官网提供的界面使用，**不能自行下载和部署**。

---

## 🆓 二、使用「开源模型」：Together AI

如果你想体验各种**开源 LLM 模型**（如 DeepSeek、LLaMA 等）：

* 推荐网站：🌐 **[Together AI](https://together.ai)**（作者特别推荐）

  * 提供一个在线「Playground」界面；
  * 上面集成了多个开源模型，可以直接对话；
  * 常见开源模型包括：DeepSeek、LLaMA、Mistral 等；
  * 缺点：大多数模型都是「指令微调模型」（Instruct），**没有原始 base 模型**。

---

## 🧱 三、想用「基础模型 base model」：Hyperbolic

大多数平台只提供聊天版模型，如果你需要用**原始 base 模型（未微调、无指令训练）**：

* 推荐网站：🌐 **[Hyperbolic](https://hyperbo.la)**

  * 作者特别喜欢其中的 **LLaMA 3.1 base 模型**；
  * 可以用来做研究、嵌入提取、任务微调等用途；
  * 适合技术用户。

---

## 💻 四、在本地电脑上运行 LLM 模型

如果你希望模型完全跑在你自己电脑上，不依赖外部服务器：

* 推荐软件：🖥️ **[LM Studio](https://lmstudio.ai)**（本地运行模型的桌面应用）

  * 可在 Mac 或 Windows 上运行；
  * 可加载**小型开源模型的低精度版本**，如：

    * DeepSeek（小版本）
    * LLaMA、Mistral、Phi 等
  * 通过 **distillation（蒸馏）+ 低精度（如 int4、int8）** 来缩小模型体积；
  * 模型在你电脑的 GPU 上运行，**完全离线本地化**；
  * 缺点：

    * UI 很丑、体验偏专业、配置项复杂（需要选择模型大小、版本、精度等）；
    * 需要看一些 YouTube 教学视频才能熟练使用；
  * 一旦配置好，体验非常丝滑，内存用完可以直接「弹出模型」来释放资源。

---

## 📌 总结一览表：

| 场景                | 平台推荐                               | 说明                    |
| ----------------- | ---------------------------------- | --------------------- |
| 使用 GPT-4、Gemini 等 | ChatGPT、Gemini                     | 商业模型，闭源，只能网页使用        |
| 尝试开源 LLM          | [Together.ai](https://together.ai) | 集成众多开源模型，在线使用         |
| 想用 base 模型（非聊天版）  | [Hyperbolic](https://hyperbo.la)   | 提供基础模型                |
| 想在本地电脑上运行         | [LM Studio](https://lmstudio.ai)   | 模型运行在你本地 GPU 上，支持离线对话 |

---

如果你想了解：

* 如何配置 LM Studio；
* 哪些小模型适合低配电脑；
* 怎样选 base vs instruct；
  我也可以帮你整理一个中文操作指南 ✅。需要吗？


# grand summary

query and we hit go what exactly is happening here what are we seeing what
are we talking to how does this work and I hope that this video gave you some appreciation for some of the under the
hood details of how these models are trained and what this is that is coming back so in particular we now know that
your query is taken and is first chopped up into tokens so we go to to tick
tokenizer and here where is the place in the in the um sort of format that is for
the user query we basically put in our query right there so our query goes into
what we discussed here is the conversation protocol format which is this way that we maintain conversation
objects so this gets inserted there and then this whole thing ends up being just a token sequence a onedimensional token
sequence under the hood so Chachi PT saw this token sequence and then when we hit
go it basically continues appending tokens into this list it continues the
sequence it acts like a token autocomplete so in particular it gave us this response so we can basically just
put it here and we see the tokens that it continued uh these are the tokens that it continued with
roughly now the question becomes okay why are these the tokens that the model responded with what are
these tokens where are they coming from uh what are we talking to and how do we program this system and so that's where
we shifted gears and we talked about the under thehood pieces of it so the first stage of this process and there are
three stages is the pre-training stage which fundamentally has to do with just knowledge acquisition from the internet
into the parameters of this neural network and so the neural net internalizes a lot of Knowledge from the
internet but where the personality really comes in is in the process of supervised fine-tuning here and so what
what happens here is that basically the a company like openai will curate a large data set of conversations like say
1 million conversation across very diverse topics and there will be conversations between a human and an
assistant and even though there's a lot of synthetic data generation used throughout this entire process and a lot
of llm help and so on fundamentally this is a human data curation task with lots
of humans involved and in particular these humans are data labelers hired by open AI who are given labeling
instructions that they learn and they task is to create ideal assistant responses for any arbitrary prompts so
they are teaching the neural network by example how to respond to
prompts so what is the way to think about what came back here like what is
this well I think the right way to think about it is that this is the neural network simulation of a data labeler at
openai so it's as if I gave this query to a data Li open and this data labeler
first reads all of the labeling instructions from open Ai and then spends 2 hours writing up the ideal
assistant response to this query and uh giving it to me now we're not actually
doing that right because we didn't wait two hours so what we're getting here is a neural network simulation of that
process and we have to keep in mind that these neural networks don't function like human brains do they are different
what's easy or hard for them is different from what's easy or hard for humans and so we really are just getting
a simulation so here I shown you this is a token stream and this is fundamentally
the neural network with a bunch of activations and neurons in between this is a fixed mathematical expression that
mixes inputs from tokens with parameters of the model and they get mixed up and
get you the next token in a sequence but this is a finite amount of compute that happens for every single token and so
this is some kind of a lossy simulation of a human that is kind of like restricted in this way and so whatever
the humans write the language model is kind of imitating on this token level with only
this this specific computation for every single token and
sequence we also saw that as a result of this and the cognitive differences the models will suffer in a variety of ways
and uh you have to be very careful with their use so for example we saw that they will suffer from hallucinations and
they also we have the sense of a Swiss model of the LM capabilities where basically there's like holes in the
cheese sometimes the models will just arbitrarily like do something dumb uh so
even though they're doing lots of magical stuff sometimes they just can't so maybe you're not giving them enough tokens to think and maybe they're going
to just make stuff up because they're mental arithmetic breaks uh maybe they are suddenly unable to count number of
letters um or maybe they're unable to tell you that 911 9.11 is smaller than
9.9 and it looks kind of dumb and so so it's a Swiss cheese capability and we have to be careful with that and we saw
the reasons for that but fundamentally this is how we think of what came back it's again a
simulation of this neural network of a human data labeler following the
labeling instructions at open a so that's what we're getting back now I do
think that the uh things change a little bit when you actually go and reach for one of the thinking models like o03 mini
and the reason for that is that GPT 40 basically doesn't do reinforcement
learning it does do rhf but I've told you that rhf is not RL there's no
there's no uh time for magic in there it's just a little bit of a fine-tuning is the way to look at it but these
thinking models they do use RL so they go through this third state stage of
perfecting their thinking process and discovering new thinking strategies and uh
solutions to problem solving that look a little bit like your internal monologue in your head and they practice that on a
large collection of practice problems that companies like openi create and curate and um then make available to the
LMS so when I come here and I talked to a thinking model and I put in this
question what we're seeing here is not anymore just the straightforward simulation of a human data labeler like
this is actually kind of new unique and interesting um and of course open is not showing us the under thehood thinking
and the chains of thought that are underlying the reasoning here but we know that such a thing exists and this
is a summary of it and what we're getting here is actually not just an imitation of a human data labeler it's
actually something that is kind of new and interesting and exciting in the sense that it is a function of thinking
that was emergent in a simulation it's not just imitating human data labeler it comes from this reinforcement learning
process and so here we're of course not giving it a chance to shine because this is not a mathematical or a reasoning
problem this is just some kind of a sort of creative writing problem roughly speaking and I think it's um it's a a
question an open question as to whether the thinking strategies that are developed inside verifiable domains
transfer and are generalizable to other domains that are unverifiable such as
create writing the extent to which that transfer happens is unknown in the field I would say so we're not sure if we are
able to do RL on everything that is very verifiable and see the benefits of that on things that are unverifiable like
this prompt so that's an open question the other thing that's interesting is that this reinforcement learning here is
still like way too new primordial and nent so we're just seeing like the
beginnings of the hints of greatness uh in the reasoning problems we're seeing something that is in principle capable
of something like the equivalent of move 37 but not in the game of Go but in open
domain thinking and problem solving in principle this Paradigm is capable of doing something really cool new and
exciting something even that no human has thought of before in principle these models are capable of analogies no human
has had so I think it's incredibly exciting that these models exist but again it's very early and these are
primordial models for now um and they will mostly shine in domains that are verifiable like math en code Etc so very
interesting to play with and think about and use and then that's roughly it um um I
would say those are the broad Strokes of what's available right now I will say that overall it is an extremely exciting
time to be in the field personally I use these models all the time daily uh tens or hundreds of
times because they dramatically accelerate my work I think a lot of people see the same thing I think we're going to see a huge amount of wealth
creation as a result of these models be aware of some of their shortcomings even
with RL models they're going to suffer from some of these use it as a tool in a toolbox don't trust it fully because
they will randomly do dumb things they will randomly hallucinate they will randomly skip over some mental arithmetic and not get it right um they
randomly can't count or something like that so use them as tools in the toolbox check their work and own the product of
your work but use them for inspiration for first draft uh ask them questions
but always check and verify and you will be very successful in your work if you do so uh so I hope this video was useful
and interesting to you I hope you had it fun and uh it's already like very long so I apologize for that but I hope it
was useful and yeah I will see you later

这段话是对整场讲解（视频）的**总结（grand summary）**，主要解释了我们在使用 ChatGPT 或类似大模型时，**它到底做了什么、它是怎么工作的、我们和谁在“对话”**，以及**未来的发展方向和现实中的使用建议**。

以下是详细中文解释：

---

## 🧠 我们输入问题时，模型做了什么？

当我们访问 ChatGPT 并输入一个问题（prompt），点击“发送”后，背后发生了以下流程：

1. **输入被分词（tokenize）**：

   * 模型不会直接处理自然语言，而是把你的问题切割成一串“tokens”（类似英语的词或子词）；
   * 这些 token 被组合成一个线性的“token 序列”，输入到模型中。

2. **模型基于这些 token 做“自动补全”**：

   * 模型不是理解问题再思考回答，而是“根据已知内容预测接下来的 token”；
   * 它每次预测一个 token，再把这个 token 接上去，再继续预测下一个，就像“智能联想输入”。

---

## 🔧 模型背后的训练过程

模型是怎么学会回答问题的？有三个阶段：

### 1. 预训练（Pretraining）：

* 模型先从**海量互联网数据**中学习语言知识和常识；
* 通过预测下一词来训练神经网络中的参数；
* 这一步类似“获取世界知识”。

### 2. 指令微调（Supervised Fine-Tuning）：

* 开发者（如 OpenAI）会收集大量“人类与助手对话”的数据；
* 人类标注员被雇来**编写理想的助手回答**；
* 模型通过学习这些“高质量对话示范”来变得更像一个助手。

> 你现在看到的回复，其实就是一个神经网络在**模拟一位 OpenAI 的标注员**，根据“助手行为准则”来回答你的问题。

### 3. 强化学习（Reinforcement Learning，某些模型有）：

* 一些“思考型模型”（如 GPT-4o、o3 mini）会进行第三阶段的强化学习；
* 模型会在可验证的推理题、数学题上练习思考步骤，学会更高效的推理方式；
* 类似于下围棋的 AlphaGo 学会了“第37步妙招”，这些模型也能在一些问题上找到前所未有的新解法。

---

## 🤖 模型输出的本质是什么？

总结一句话：

> 💬 你得到的回复，其实是：**一个神经网络模拟了 OpenAI 的数据标注员，在“助手行为准则”指导下写出理想回答**。

这不是一个真的人，也不是逻辑完美的思考，而是一种“**有限计算力的模拟**”，所以它会出现各种“智障瞬间”。

例如：

* 无法正确比较 9.11 和 9.9；
* 忽然算错简单加法；
* 幻觉（hallucinate）错误信息。

这就像“瑞士奶酪”——整体很强大，但某些地方有洞。

---

## 🧠 思考型模型的不同之处

像 o3 mini、GPT-4o 这类“思考型模型”：

* 在强化学习阶段，**学会了链式思维（chain-of-thought）**；
* 不再只是模仿人类的标注行为，而是在解题过程中发展出一些“类人思维路径”；
* 在数学、编程、推理这些可验证领域，已经能展现出惊人的能力；
* 但这些“思维策略”是否可以迁移到文学创作、写作等“不可验证”的领域，目前还不确定。

---

## 🔧 使用建议：如何正确使用语言模型？

1. ✅ 把模型当作**工具箱中的工具**，不要当做万能 AI。
2. ✅ 用它来**获取灵感、生成草稿、进行头脑风暴**。
3. ❌ 不要盲目信任，必须**验证结果**。
4. ⚠️ 它会出错、胡说八道、算错、记错，要学会识别。

---

## 🌟 总结感言

> 这是一个令人兴奋的时代！

* 作者每天都在使用大模型，极大提高了工作效率；
* 未来这类技术将创造巨大的社会价值和财富；
* 但也要认识到它们还处于“原始阶段”，尤其是思考能力方面；
* 未来还有很多可以改进、探索和突破的方向。

---

如果你想要我用一张图或表格**概括整个 LLM 工作流程、训练流程和使用建议**，我也可以帮你做。需要吗？
