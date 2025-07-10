papers:
- Language Models are Unsupervised Multitask Learners
- LLAMA 2: Open Foundation and Finue-Tuned Chat Models
- https://www.reedbeta.com/blog/programmers-intro-to-unicode/ A Programmer’s Introduction to Unicode
- MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers
- Wikipedia Byte pair encoding https://en.wikipedia.org/wiki/Byte-pair_encoding
- https://www.regular-expressions.info/
- Efficient Training of Language Models to Fill in the Middle
- https://github.com/karpathy/minbpe/tree/master
- https://github.com/google/sentencepiece
- Learning to Compress Prompts with Gist Tokens
- https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation
- 

# intro: Tokenization, GPT-2 paper, tokenization-related issues
The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely.

```
Chapters:
00:00:00 intro: Tokenization, GPT-2 paper, tokenization-related issues
00:05:50 tokenization by example in a Web UI (tiktokenizer)
00:14:56 strings in Python, Unicode code points
00:18:15 Unicode byte encodings, ASCII, UTF-8, UTF-16, UTF-32
00:22:47 daydreaming: deleting tokenization
00:23:50 Byte Pair Encoding (BPE) algorithm walkthrough
00:27:02 starting the implementation
00:28:35 counting consecutive pairs, finding most common pair
00:30:36 merging the most common pair
00:34:58 training the tokenizer: adding the while loop, compression ratio
00:39:20 tokenizer/LLM diagram: it is a completely separate stage
00:42:47 decoding tokens to strings
00:48:21 encoding strings to tokens
00:57:36 regex patterns to force splits across categories
01:11:38 tiktoken library intro, differences between GPT-2/GPT-4 regex
01:14:59 GPT-2 encoder.py released by OpenAI walkthrough
01:18:26 special tokens, tiktoken handling of, GPT-2/GPT-4 differences
01:25:28 minbpe exercise time! write your own GPT-4 tokenizer
01:28:42 sentencepiece library intro, used to train Llama 2 vocabulary
01:43:27 how to set vocabulary set? revisiting gpt.py transformer
01:48:11 training new tokens, example of prompt compression
01:49:58 multimodal [image, video, audio] tokenization with vector quantization
01:51:41 revisiting and explaining the quirks of LLM tokenization
02:10:20 final recommendations
02:12:50 ??? :)
```

Exercises:
- Advised flow: reference this document and try to implement the steps before I give away the partial solutions in the video. The full solutions if you're getting stuck are in the minbpe code https://github.com/karpathy/minbpe/bl...

Links:
- Google colab for the video: https://colab.research.google.com/dri...
- GitHub repo for the video: minBPE https://github.com/karpathy/minbpe
- Playlist of the whole Zero to Hero series so far:    • The spelled-out intro to neural networks a...  
- our Discord channel:   / discord  
- my Twitter:   / karpathy  

Supplementary links:
- tiktokenizer https://tiktokenizer.vercel.app
- tiktoken from OpenAI: https://github.com/openai/tiktoken
- sentencepiece from Google https://github.com/google/sentencepiece


这段内容是关于大语言模型（LLM）中的\*\*分词器（Tokenizer）\*\*的讲解。分词器是大语言模型中必不可少且无处不在的组成部分，它负责在字符串和令牌（文本片段）之间进行转换。分词器是LLM流程中一个完全独立的阶段：它们有自己独立的训练集、训练算法（如字节对编码Byte Pair Encoding），并在训练后执行两个基本功能：`encode()`（从字符串转换为令牌）和`decode()`（从令牌转换回字符串）。在这节课中，我们将从零开始构建OpenAI GPT系列中使用的分词器。在过程中，我们会发现很多LLM的奇怪行为和问题实际上都可以追溯到分词阶段。我们将讨论这些问题的原因，分析为什么分词可能是罪魁祸首，以及为什么某些人希望能完全删除这一阶段。

### 章节：

* **00:00:00** 引言：分词、GPT-2论文、与分词相关的问题
* **00:05:50** 网页界面中的分词示例（tiktokenizer）
* **00:14:56** Python中的字符串、Unicode代码点
* **00:18:15** Unicode字节编码，ASCII、UTF-8、UTF-16、UTF-32
* **00:22:47** 白日梦：删除分词
* **00:23:50** 字节对编码（BPE）算法演示
* **00:27:02** 开始实现分词器
* **00:28:35** 计算连续的字节对，找到最常见的字节对
* **00:30:36** 合并最常见的字节对
* **00:34:58** 训练分词器：添加循环、压缩比
* **00:39:20** 分词器/LLM图解：分词是一个完全独立的阶段
* **00:42:47** 令牌解码为字符串
* **00:48:21** 字符串编码为令牌
* **00:57:36** 使用正则表达式强制按类别拆分
* **01:11:38** tiktoken库介绍，GPT-2/GPT-4正则的区别
* **01:14:59** OpenAI发布的GPT-2 encoder.py代码演示
* **01:18:26** 特殊令牌，tiktoken如何处理，GPT-2/GPT-4之间的差异
* **01:25:28** minbpe练习：自己实现GPT-4分词器
* **01:28:42** sentencepiece库介绍，用于训练Llama 2词汇
* **01:43:27** 如何设置词汇表？重新审视gpt.py转换器
* **01:48:11** 训练新令牌，提示压缩示例
* **01:49:58** 多模态（图像、视频、音频）令牌化与向量量化
* **01:51:41** 重新审视并解释LLM分词的怪癖
* **02:10:20** 最终建议
* **02:12:50** ??? :)

### 练习：

建议的流程是：参考本文档并尝试实现这些步骤，在我提供部分视频解决方案之前，如果卡住了，完整的解决方案可以在minbpe代码中找到 [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)。

### 链接：

* 视频的Google Colab：[点击这里](https://colab.research.google.com/dri...)
* 视频的GitHub仓库：minBPE [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
* 整个“零基础到英雄”系列的视频播放列表：\[播放列表链接]
* 我们的Discord频道：\[加入Discord]
* 我的Twitter：\[我的Twitter链接]

### 补充链接：

* tiktokenizer：[https://tiktokenizer.vercel.app](https://tiktokenizer.vercel.app)
* OpenAI的tiktoken：[https://github.com/openai/tiktoken](https://github.com/openai/tiktoken)
* Google的sentencepiece：[https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)


# intro: Tokenization, GPT-2 paper, tokenization-related issues

hi everyone so in this video I'd like us to cover the process of tokenization in large language models now you see here
that I have a set face and that's because uh tokenization is my least favorite part of working with large
language models but unfortunately it is necessary to understand in some detail because it it is fairly hairy gnarly and
there's a lot of hidden foot guns to be aware of and a lot of oddness with large language models typically traces back to
tokenization so what is tokenization now in my previous video Let's Build GPT from scratch uh we
actually already did tokenization but we did a very naive simple version of tokenization so when you go to the
Google colab for that video uh you see here that we loaded our training set and
our training set was this uh Shakespeare uh data set now in the beginning the Shakespeare data set is just a large
string in Python it's just text and so the question is how do we plug text into
large language models and in this case here we created a vocabulary of 65
possible characters that we saw occur in this string these were the possible characters and we saw that there are 65
of them and then we created a a lookup table for converting from every possible
character a little string piece into a token an integer so here for example we tokenized
the string High there and we received this sequence of tokens and here we took the first 1,000
characters of our data set and we encoded it into tokens and because it is this is character level we received
1,000 tokens in a sequence so token 18 47
Etc now later we saw that the way we plug these tokens into the language
model is by using an embedding table and so basically if we have 65
possible tokens then this embedding table is going to have 65 rows and roughly speaking we're taking the
integer associated with every single sing Le token we're using that as a lookup into this table and we're
plucking out the corresponding row and this row is a uh is trainable parameters
that we're going to train using back propagation and this is the vector that then feeds into the Transformer um and
that's how the Transformer Ser of perceives every single token so here we had a very naive
tokenization process that was a character level tokenizer but in practice in state-ofthe-art uh language
models people use a lot more complicated schemes unfortunately uh for constructing these uh token
vocabularies so we're not dealing on the Character level we're dealing on chunk level and the way these um character
chunks are constructed is using algorithms such as for example the bik pair in coding algorithm which we're
going to go into in detail um and cover in this video I'd like to briefly show
you the paper that introduced a bite level encoding as a mechanism for tokenization in the context of large
language models and I would say that that's probably the gpt2 paper and if you scroll down here to the section
input representation this is where they cover tokenization the kinds of properties that you'd like the tokenization to have and they conclude
here that they're going to have a tokenizer where you have a vocabulary of 50,2 57 possible
tokens and the context size is going to be 1,24 tokens so in the in in the
attention layer of the Transformer neural network every single token is attending to the previous tokens in the sequence and it's
going to see up to 1,24 tokens so tokens are this like fundamental unit um the
atom of uh large language models if you will and everything is in units of tokens everything is about tokens and
tokenization is the process for translating strings or text into sequences of tokens and uh vice versa
when you go into the Llama 2 paper as well I can show you that when you search token you're going to get get 63 hits um
and that's because tokens are again pervasive so here they mentioned that they trained on two trillion tokens of
data and so on so we're going to build our own tokenizer luckily the bite be encoding
algorithm is not uh that super complicated and we can build it from scratch ourselves and we'll see exactly
how this works before we dive into code I'd like to give you a brief Taste of some of the complexities that come from
the tokenization because I just want to make sure that we motivate it sufficiently for why we are doing all
this and why this is so gross so tokenization is at the heart of a lot of weirdness in large language models and I
would advise that you do not brush it off a lot of the issues that may look like just issues with the new network
architecture or the large language model itself are actually issues with the tokenization and fundamentally Trace uh
back to it so if you've noticed any issues with large language models can't
you know not able to do spelling tasks very easily that's usually due to tokenization simple string processing
can be difficult for the large language model to perform natively uh non-english languages can
work much worse and to a large extent this is due to tokenization sometimes llms are bad at
simple arithmetic also can trace be traced to tokenization uh gbt2 specifically would
have had quite a bit more issues with python than uh future versions of it due to tokenization there's a lot of other
issues maybe you've seen weird warnings about a trailing whites space this is a tokenization issue um
if you had asked GPT earlier about solid gold Magikarp and what it is you would see the llm go totally crazy and it
would start going off about a completely unrelated tangent topic maybe you've been told to use yl over Json in
structure data all of that has to do with tokenization so basically tokenization is at the heart of many
issues I will look back around to these at the end of the video but for now let me just um skip over it a little bit and

在这段视频中，讲解者介绍了**大语言模型（LLM）中的分词（Tokenization）过程**。他提到，尽管分词是他最不喜欢的部分，但它对于理解LLM至关重要，因为很多LLM的奇怪行为和问题都源于分词阶段。

### 主要内容：

1. **分词的基本概念**：

   * 在之前的视频《从零开始构建GPT》中，讲解者使用了一个非常简单的字符级别分词方法。当时，训练集是莎士比亚的数据集，这个数据集首先是一个大字符串。分词的方式是基于每个字符来做标记。比如，对于“High”这个字符串，分词后会得到一个由令牌（tokens）组成的序列。

2. **字符级别分词**：

   * 在这个简单的分词过程中，训练数据集被映射成了65个字符的令牌集，然后通过查找表将字符转换为整数（令牌）。这些整数再通过\*\*嵌入表（embedding table）\*\*转换成向量，进入Transformer模型进行处理。

3. **更复杂的分词方法**：

   * 然而，在现代的LLM中，分词的方式要复杂得多，通常不再是按字符来分词，而是按“词块（chunk）”来分词。这些词块的构建通常使用**字节对编码（Byte Pair Encoding，BPE）算法**，这也是我们将要学习的内容。

4. **GPT-2中的分词**：

   * GPT-2论文中介绍了**字节级别编码**（byte-level encoding），并讨论了分词的相关问题。GPT-2的分词器拥有约50,257个可能的令牌，且上下文窗口大小为1,024个令牌。每个令牌在Transformer的注意力层中，会关注前面1,024个令牌的上下文。

5. **Tokenization的普遍性**：

   * 分词在所有LLM中都是非常核心的部分，例如，**Llama 2**论文中也提到，它们的训练集包含了两万亿个令牌。无论是文本生成还是理解，所有任务都涉及令牌，分词是将字符串转换为令牌序列的过程。

6. **Tokenization相关的问题**：

   * 分词引发了LLM中的许多问题。例如：

     * LLM在拼写任务上可能表现不佳，这是由于分词的限制。
     * 非英语语言的处理效果较差，这也可能是分词不准确导致的。
     * 有时LLM在做简单的算术运算时也出现问题，这与分词方式有关。
     * GPT-2版本尤其在处理Python代码时会遇到更多问题，因为它的分词器在处理编程语言时并不完全适应。
     * 其他问题如尾随空格警告、意外的语义跑题等，也可以追溯到分词阶段。

7. **总结**：

   * 分词是许多LLM问题的根源。很多表面上看起来像是模型架构本身的问题，实际上可能是由于分词造成的。

讲解者还建议大家不要忽视分词，因为它在LLM的表现中起着至关重要的作用，理解分词过程对于理解LLM的复杂行为非常重要。


# tokenization by example in a Web UI (tiktokenizer)

let's go to this web app um the Tik tokenizer bell.app so I have it loaded
here and what I like about this web app is that tokenization is running a sort of live in your browser in JavaScript so
you can just type here stuff hello world and the whole string rokenes so here what we see on uh the
left is a string that you put in on the right we're currently using the gpt2 tokenizer we see that this string that I
pasted here is currently tokenizing into 300 tokens and here they are sort of uh
shown explicitly in different colors for every single token so for example uh this word tokenization became two tokens
the token 3,642 and
1,634 the token um space is is token 318
so be careful on the bottom you can show white space and keep in mind that there are spaces and uh sln new line
characters in here but you can hide them for clarity the token space at is token 379
the to the Token space the is 262 Etc so
you notice here that the space is part of that uh token chunk now so this is kind of like how
our English sentence broke up and that seems all well and good now now here I
put in some arithmetic so we see that uh the token 127 Plus and then token six
space 6 followed by 77 so what's happening here is that 127 is feeding in as a single token into the large
language model but the um number 677 will actually feed in as two separate
tokens and so the large language model has to sort of um take account of that
and process it correctly in its Network and see here 804 will be broken up into
two tokens and it's is all completely arbitrary and here I have another example of four-digit numbers and they
break up in a way that they break up and it's totally arbitrary sometimes you have um multiple digits single token
sometimes you have individual digits as many tokens and it's all kind of pretty arbitrary and coming out of the
tokenizer here's another example we have the string egg and you see here that
this became two tokens but for some reason when I say I have an egg you see when it's a space
egg it's two token it's sorry it's a single token so just egg by itself in
the beginning of a sentence is two tokens but here as a space egg is suddenly a single token uh for the exact
same string okay here lowercase egg turns out to be a single token and in
particular notice that the color is different so this is a different token so this is case sensitive and of course
a capital egg would also be different tokens and again um this would be two
tokens arbitrarily so so for the same concept egg depending on if it's in the beginning of a sentence at the end of a
sentence lowercase uppercase or mixed all this will be uh basically very different tokens and different IDs and
the language model has to learn from raw data from all the internet text that it's going to be training on that these are actually all the exact same concept
and it has to sort of group them in the parameters of the neural network and understand just based on the data
patterns that these are all very similar but maybe not almost exactly similar but but very very similar
um after the EG demonstration here I have um an introduction from open a eyes
chbt in Korean so manaso Pang uh Etc uh
so this is in Korean and the reason I put this here is because you'll notice
that um non-english languages work slightly worse in Chachi part of this is
because of course the training data set for Chachi is much larger for English and for everything else but the same is
true not just for the large language model itself but also for the tokenizer so when we train the tokenizer we're
going to see that there's a training set as well and there's a lot more English than non-english and what ends up
happening is that we're going to have a lot more longer tokens for
English so how do I put this if you have a single sentence in English and you tokenize it you might see that it's 10
tokens or something like that but if you translate that sentence into say Korean or Japanese or something else you'll
typically see that the number of tokens used is much larger and that's because the chunks here are a lot more broken up
so we're using a lot more tokens for the exact same thing and what this does is it bloats up the sequence length of all
the documents so you're using up more tokens and then in the attention of the Transformer when these tokens try to
attend each other you are running out of context um in the maximum context length
of that Transformer and so basically all the non-english text is stretched out
from the perspective of the Transformer and this just has to do with the um trainings that used for the tokenizer
and the tokenization itself so it will create a lot bigger tokens and a lot larger groups in English and it will
have a lot of little boundaries for all the other non-english text um so if we
translated this into English it would be significantly fewer tokens the final example I have here is
a little snippet of python for doing FS buuz and what I'd like you to notice is
look all these individual spaces are all separate tokens they are token
220 so uh 220 220 220 220 and then space
if is a single token and so what's going on here is that when the Transformer is going to consume or try to uh create
this text it needs to um handle all these spaces individually they all feed
in one by one into the entire Transformer in the sequence and so this is being extremely wasteful tokenizing
it in this way and so as a result of that gpt2 is not very good with python
and it's not anything to do with coding or the language model itself it's just that if he use a lot of indentation
using space in Python like we usually do uh you just end up bloating out all the
text and it's separated across way too much of the sequence and we are running out of the context length in the
sequence uh that's roughly speaking what's what's happening we're being way too wasteful we're taking up way too much token space now we can also scroll
up here and we can change the tokenizer so note here that gpt2 tokenizer creates a token count of 300 for this string
here we can change it to CL 100K base which is the GPT for tokenizer and we
see that the token count drops to 185 so for the exact same string we are now roughly having the number of tokens and
roughly speaking this is because uh the number of tokens in the GPT 4 tokenizer is roughly double that of the number of
tokens in the gpt2 tokenizer so we went went from roughly 50k to roughly 100K now you can imagine that this is a good
thing because the same text is now squished into half as many tokens so uh
this is a lot denser input to the Transformer and in the Transformer every
single token has a finite number of tokens before it that it's going to pay attention to and so what this is doing is we're roughly able to see twice as
much text as a context for what token to predict next uh because of this change
but of course just increasing the number of tokens is uh not strictly better infinitely uh because as you increase
the number of tokens now your embedding table is um sort of getting a lot larger and also at the output we are trying to
predict the next token and there's the soft Max there and that grows as well we're going to go into more detail later on this but there's some kind of a Sweet
Spot somewhere where you have a just right number of tokens in your vocabulary where everything is
appropriately dense and still fairly efficient now one thing I would like you to note specifically for the gp4
tokenizer is that the handling of the white space for python has improved a
lot you see that here these four spaces are represented as one single token for the three spaces here and then the token
SPF and here seven spaces were all grouped into a single token so we're
being a lot more efficient in how we represent Python and this was a deliberate Choice made by open aai when they designed the gp4 tokenizer and they
group a lot more space into a single character what this does is this densifies Python and therefore we can
attend to more code before it when we're trying to predict the next token in the sequence and so the Improvement in the
python coding ability from gbt2 to gp4 is not just a matter of the language
model and the architecture and the details of the optimization but a lot of the Improvement here is also coming from
the design of the tokenizer and how it groups characters into tokens okay so let's now start writing some code

这段内容讲解了在Web UI中使用**tiktokenizer**进行分词的例子，展示了如何查看和理解大语言模型（LLM）中的分词过程。这里的分词展示通过一个交互式Web应用（tiktokenizer）实现，可以实时在浏览器中查看文本是如何被分词的。

### 主要内容：

1. **分词示例**：

   * 用户在网页上输入文本（例如“hello world”），系统会展示该文本经过**GPT-2分词器**后的分词结果。在这个例子中，输入的字符串“hello world”被分词成300个令牌（tokens），每个令牌都用不同的颜色高亮显示。例如，单词“tokenization”被分为两个令牌，分别是令牌3,642和1,634。

2. **空格和换行的分词**：

   * 分词器不仅将单词分解成令牌，还会将空格、换行符等特殊字符作为独立的令牌处理。例如，“token space”中的空格被分为一个单独的令牌（令牌379）。

3. **数字的分词**：

   * 在进行算术运算时，例如“127 + 6”，数字127会被视为一个令牌，而数字677会被分成两个令牌。这说明大语言模型需要处理数字的拆分和组合，且这种拆分是相对任意的。

4. **大小写敏感的分词**：

   * 词语的大小写会影响其分词方式。例如，“egg”作为句首时会被分成两个令牌，而在句中则变成单个令牌。不同的大小写或位置（句首、句中）会导致完全不同的令牌ID，这表明分词器是大小写敏感的。

5. **非英语语言的分词问题**：

   * 当处理非英语语言（如韩语）时，分词效果通常较差。由于训练数据集中的英语文本占据了大部分，非英语语言会被分割成更多的令牌，这导致序列的长度增加，消耗更多的上下文空间。最终，这会影响Transformer模型的上下文处理能力，导致非英语语言的性能较差。

6. **Python代码的分词问题**：

   * 在Python代码示例中，分词器会将每个空格、每个字符都拆成单独的令牌。例如，多个空格会被当作多个令牌，这对于代码缩进（例如Python中的空格缩进）来说是非常低效的。这种低效的分词方式导致GPT-2模型在处理Python代码时表现不佳，因为它需要处理更多的令牌，从而导致上下文窗口的浪费。

7. **GPT-4分词器的改进**：

   * 当将分词器切换为**GPT-4分词器**时，令牌数量明显减少，文本的分词密度提高。例如，原本300个令牌的文本在GPT-4分词器中被压缩成185个令牌。这意味着GPT-4能够以更密集的令牌表示输入文本，从而提高了Transformer的上下文处理能力，可以处理更多的上下文信息。
   * GPT-4分词器还对Python代码的分词做了改进，例如将多个空格合并为一个令牌，使得Python代码的处理更加高效。这种改进不仅提高了模型在Python代码处理上的表现，也优化了上下文的利用。

### 结论：

* **GPT-4分词器**通过改进空格的处理方式，提高了对Python代码的处理能力。分词的设计和优化对大语言模型的表现影响深远，尤其在处理编程语言和非英语文本时，分词器的设计尤为重要。

这个过程让我们明白，分词器不仅是语言模型的一部分，它的设计直接影响模型对不同类型文本（尤其是编程语言和非英语文本）的处理能力。


# strings in Python, Unicode code points

so remember what we want to do we want to take strings and feed them into language models for that we need to
somehow tokenize strings into some integers in some fixed vocabulary and
then we will use those integers to make a look up into a lookup table of vectors and feed those vectors into the
Transformer as an input now the reason this gets a little bit tricky of course is that we don't just want to support
the simple English alphabet we want to support different kinds of languages so this is anango in Korean which is hello
and we also want to support many kinds of special characters that we might find on the internet for example
Emoji so how do we feed this text into uh Transformers well how's the what is this
text anyway in Python so if you go to the documentation of a string in Python
you can see that strings are immutable sequences of Unicode code points okay what are Unicode code points
we can go to PDF so Unicode code points are defined by the Unicode Consortium as
part of the Unicode standard and what this is really is that it's just a definition of roughly 150,000 characters
right now and roughly speaking what they look like and what integers um represent
those characters so it says 150,000 characters across 161 scripts as of
right now so if you scroll down here you can see that the standard is very much alive the latest standard 15.1 in
September 2023 and basically this is just a way to
define lots of types of characters like for example all these
characters across different scripts so the way we can access the unic code code Point given Single Character is by using
the or function in Python so for example I can pass in Ord of H and I can see
that for the Single Character H the unic code code point is
104 okay um but this can be arbitr complicated so we can take for example
our Emoji here and we can see that the code point for this one is 128,000 or we can take
un and this is 50,000 now keep in mind you can't plug in strings here because
you uh this doesn't have a single code point it only takes a single uni code code Point character and tells you its
integer so in this way we can look up all the um characters of this
specific string and their code points so or of X forx in this string and we get
this encoding here now see here we've already turned the raw code points
already have integers so why can't we simply just use these integers and not have any tokenization at all why can't
we just use this natively as is and just use the code Point well one reason for that of course is that the vocabulary in
that case would be quite long so in this case for Unicode the this is a vocabulary of
150,000 different code points but more worryingly than that I think the Unicode
standard is very much alive and it keeps changing and so it's not kind of a stable representation necessarily that
we may want to use directly so for those reasons we need something a bit better so to find something better we turn to

在这段内容中，讲解者介绍了如何将字符串转换为大语言模型可以理解的形式，重点讨论了**Unicode编码点**和如何在Python中处理这些编码点。

### 主要内容：

1. **目标：将字符串输入到语言模型中**：

   * 我们的目标是将字符串（例如不同语言的文本和特殊字符，如表情符号）转换为可以输入到Transformer模型中的令牌。这些令牌是整数，我们通过查找表将这些整数映射到向量，然后将向量输入到Transformer中。

2. **Python中的字符串和Unicode**：

   * 在Python中，字符串是**不可变的（immutable）Unicode代码点序列**。这意味着每个字符都有一个对应的**Unicode编码点**，它是一个整数，表示该字符的唯一标识。

3. **什么是Unicode编码点**：

   * **Unicode编码点**是由Unicode联盟定义的，用来表示全球所有字符的标准。目前，Unicode标准包含大约**150,000个字符**，覆盖了161种书写系统。每个字符都有一个唯一的整数表示，这些整数就是我们所称的编码点（code points）。

4. **如何访问Unicode编码点**：

   * 在Python中，我们可以通过`ord()`函数来获取一个字符的Unicode编码点。例如，`ord('H')`返回的是字符'H'的编码点104。
   * 对于表情符号等字符，我们也可以通过`ord()`函数获取它们的编码点。例如，某个表情符号的编码点可能是128,000，另一个字符的编码点可能是50,000。

5. **为何不能直接使用Unicode编码点**：

   * 虽然我们可以通过`ord()`函数得到字符的编码点并将其转换为整数，但如果直接使用这些整数作为输入，问题就在于**词汇表会非常庞大**。例如，Unicode标准有大约150,000个字符，每个字符都有一个唯一的编码点，这就意味着我们的词汇表会非常长。
   * 更重要的是，**Unicode标准是不断更新和变化的**，新的字符不断被添加到Unicode中，这使得Unicode编码点不是一个稳定的表示方式，因此不适合直接用于分词和模型训练。

### 总结：

* 虽然Unicode编码点可以作为字符的整数表示，但由于其词汇表过于庞大且不断变化，直接使用Unicode编码点并不适合用于语言模型的输入。为了更高效地处理文本，通常需要构建更加紧凑和稳定的**分词器**，即将文本转换成更小的、固定词汇表中的令牌。


# Unicode byte encodings, ASCII, UTF-8, UTF-16, UTF-32

encodings so if we go to the Wikipedia page here we see that the Unicode consortion defines three types of
encodings utf8 UTF 16 and UTF 32 these encoding are the way by which we can
take Unicode text and translate it into binary data or by streams utf8 is by far
the most common uh so this is the utf8 page now this Wikipedia page is actually quite long but what's important for our
purposes is that utf8 takes every single Cod point and it translates it to a by
stream and this by stream is between one to four bytes so it's a variable length encoding so depending on the Unicode
Point according to the schema you're going to end up with between 1 to four bytes for each code point on top of that
there's utf8 uh utf16 and UTF 32 UTF 32 is nice because
it is fixed length instead of variable length but it has many other downsides as well so the full kind of spectrum of
pros and cons of all these different three encodings are beyond the scope of this video I just like to point out that
I enjoyed this block post and this block post at the end of it also has a number of references that can be quite useful
uh one of them is uh utf8 everywhere Manifesto um and this Manifesto
describes the reason why utf8 is significantly preferred and a lot nicer
than the other encodings and why it is used a lot more prominently um on the
internet one of the major advantages just just to give you a sense is that utf8 is the only one of these that is
backwards compatible to the much simpler asky encoding of text um but I'm not
going to go into the full detail in this video so suffice to say that we like the utf8 encoding and uh let's try to take
the string and see what we get if we encoded into utf8 the string class in Python actually
has do encode and you can give it the encoding which is say utf8 now we get out of this is not very nice because
this is the bytes is a bytes object and it's not very nice in the way that it's printed so I personally like to take it
through list because then we actually get the raw B of this uh encoding so this is the raw
byes that represent this string according to the utf8 en coding we can
also look at utf16 we get a slightly different by stream and we here we start
to see one of the disadvantages of utf16 you see how we have zero Z something Z something Z something we're starting to
get a sense that this is a bit of a wasteful encoding and indeed for simple asky characters or English characters
here uh we just have the structure of 0 something Z something and it's not exactly nice same for UTF 32 when we
expand this we can start to get a sense of the wastefulness of this encoding for our purposes you see a lot of zeros
followed by something and so uh this is not desirable so suffice it to say that we
would like to stick with utf8 for our purposes however if we just use utf8
naively these are by streams so that would imply a vocabulary length of only
256 possible tokens uh but this this vocabulary size is very very small what
this is going to do if we just were to use it naively is that all of our text would be stretched out over very very
long sequences of bytes and so um what what this does is that certainly
the embeding table is going to be tiny and the prediction at the top at the final layer is going to be very tiny but our sequences are very long and remember
that we have pretty finite um context length and the attention that we can support in a transformer for
computational reasons and so we only have as much context length but now we have very very long sequences and this
is just inefficient and it's not going to allow us to attend to sufficiently long text uh before us for the purposes
of the next token prediction task so we don't want to use the raw bytes of the
utf8 encoding we want to be able to support larger vocabulary size that we
can tune as a hyper but we want to stick with the utf8 encoding of these strings so what do we
do well the answer of course is we turn to the bite pair encoding algorithm which will allow us to compress these
bite sequences um to a variable amount so we'll get to that in a bit but I just
want to briefly speak to the fact that I would love nothing more than to be able to feed raw bite sequences into uh

在这段内容中，讲解者介绍了**Unicode字节编码**（包括UTF-8、UTF-16和UTF-32），并解释了它们的优缺点以及为什么在大语言模型中倾向于使用UTF-8编码。

### 主要内容：

1. **Unicode编码和字节流**：

   * Unicode编码由三个主要的编码标准组成：**UTF-8、UTF-16和UTF-32**。这些编码方式将Unicode字符集中的字符转换为二进制数据或字节流。

2. **UTF-8编码**：

   * **UTF-8**是最常用的编码方式，它将每个Unicode编码点（code point）转换为一个字节流，这个字节流的长度是可变的，通常在1到4个字节之间。也就是说，不同的Unicode字符会占用不同数量的字节。
   * UTF-8的一个主要优点是它的**向后兼容性**，也就是说，UTF-8编码的文本可以兼容更简单的**ASCII编码**，这使得UTF-8在互联网中广泛使用。

3. **UTF-16和UTF-32**：

   * **UTF-16**也是一种常用的编码方式，但它比UTF-8更浪费空间，尤其是在处理像ASCII字符（英文字符）这样的简单字符时，每个字符都被编码为2个字节。
   * **UTF-32**是固定长度编码，每个字符都占用4个字节，但它也有一些缺点，尤其是在处理不需要那么大存储空间的字符时，会浪费很多字节。

4. **编码的选择**：

   * 对于我们这个场景，讲解者提到会选择**UTF-8**编码，因为它是最有效的，并且它能够支持多种字符，特别是在处理表情符号、不同语言字符等时，UTF-8能够非常高效地进行编码。

5. **UTF-8的局限性**：

   * 尽管UTF-8编码在许多场景下非常有用，但如果直接使用UTF-8的字节流，**词汇表的大小**会非常小（最多只有256个可能的令牌）。这就导致了**序列长度非常长**，每个字符都会被编码为多个字节，生成的序列变得非常冗长。
   * 在大语言模型中，Transformer模型有固定的**上下文窗口**，即它能处理的上下文长度是有限的。如果输入序列过长，Transformer模型就不能有效地处理这些长序列，导致性能不佳。

6. **解决方案：字节对编码（BPE）**：

   * 讲解者提到，为了避免UTF-8编码导致的冗长序列和词汇表过小的问题，通常会使用**字节对编码（BPE）算法**来压缩这些字节序列，**减少序列的长度**，并且可以**动态调整词汇表的大小**。

### 总结：

* **UTF-8**是最常用且效率较高的编码方式，尤其适用于包含多语言字符和特殊符号（如表情符号）的文本。然而，如果直接使用UTF-8的字节流，会导致非常长的文本序列，这对大语言模型的处理效率不利。为了解决这个问题，通常会使用\*\*字节对编码（BPE）\*\*来压缩这些字节序列，优化模型的处理效率。


# daydreaming: deleting tokenization

language models in fact there's a paper about how this could potentially be done uh from Summer last last year now the
problem is you actually have to go in and you have to modify the Transformer architecture because as I mentioned
you're going to have a problem where the attention will start to become extremely expensive because the sequences are so
long and so in this paper they propose kind of a hierarchical structuring of
the Transformer that could allow you to just feed in raw bites and so at the end they say together these results
establish the viability of tokenization free autor regressive sequence modeling at scale so tokenization free would
indeed be amazing we would just feed B streams directly into our models but unfortunately I don't know that this has
really been proven out yet by sufficiently many groups and a sufficient scale uh but something like
this at one point would be amazing and I hope someone comes up with it but for now we have to come back and we can't
feed this directly into language models and we have to compress it using the B paare encoding algorithm so let's see
how that works so as I mentioned the B paare encoding algorithm is not all that complicated and the Wikipedia page is

在这段内容中，讲解者探讨了一个有趣的想法：**删除分词过程**，即直接将原始字节流输入到语言模型中，而不需要通过分词将字符串转换为令牌。

### 主要内容：

1. **删除分词的想法**：

   * 讲解者提到，有一篇论文讨论了如何实现“**无分词（tokenization-free）**”的自回归序列建模。这意味着，如果能够直接将原始字节流（byte streams）输入到语言模型中，而不是首先进行分词处理，将极大地简化整个过程。
   * 不过，这个想法面临一些技术难题，尤其是在**Transformer架构**中处理长序列时会遇到性能瓶颈，因为长序列的\*\*注意力机制（attention）\*\*会变得非常昂贵，计算资源需求激增。

2. **论文中的建议**：

   * 论文提出了一种**层次化结构的Transformer**，使得直接处理原始字节流成为可能，从而避免了分词阶段。通过这种结构，语言模型能够在不进行分词的情况下处理输入的字节流。

3. **现实挑战**：

   * 尽管这种“无分词”的方法看起来很有前景，但讲解者认为，目前还没有足够多的团队在大规模上验证这种方法的可行性。因此，尽管“无分词”可能会是一个理想的目标，但目前我们还无法实现这一点。

4. **现有的解决方案**：

   * 由于目前我们不能直接将字节流输入到语言模型中，讲解者指出，\*\*字节对编码（Byte Pair Encoding，BPE）\*\*算法仍然是解决这一问题的有效方法。BPE算法通过对字节流进行压缩，减少了序列长度，并能为模型提供一个更有效的词汇表。

### 总结：

* 删除分词步骤并直接处理字节流的想法很吸引人，但由于Transformer架构中的计算开销问题，目前还未得到广泛验证。因此，尽管未来可能会有这种无分词的技术，但目前我们仍需依赖\*\*字节对编码（BPE）\*\*来优化分词过程。


# Byte Pair Encoding (BPE) algorithm walkthrough

actually quite instructive as far as the basic idea goes go what we're doing is we have some kind of a input sequence uh
like for example here we have only four elements in our vocabulary a b c and d and we have a sequence of them so
instead of bytes let's say we just have four a vocab size of four the sequence is too long and we'd
like to compress it so what we do is that we iteratively find the pair of uh
tokens that occur the most frequently and then once we've
identified that pair we repl replace that pair with just a single new token
that we append to our vocabulary so for example here the bite pair AA occurs
most often so we mint a new token let's call it capital Z and we replace every
single occurrence of AA by Z so now we have two Z's here so here we took a
sequence of 11 characters with vocabulary size four and we've converted
it to a um sequence of only nine tokens but now with a vocabulary of five
because we have a fifth vocabulary element that we just created and it's Z standing for concatination of AA and we
can again repeat this process so we again look at the sequence and identify
the pair of tokens that are most frequent let's say that that is now AB
well we are going to replace AB with a new token that we meant call Y so y becomes ab and then every single
occurrence of ab is now replaced with y so we end up with this so now we only
have 1 2 3 4 5 6 seven characters in our sequence but we have not just um four
vocabulary elements or five but now we have six and for the final round we
again look through the sequence find that the phrase zy or the pair zy is most common and replace it one more time
with another um character let's say x so X is z y and we replace all curses of zy
and we get this following sequence so basically after we have gone through this process instead of having a um
sequence of 11 uh tokens with a vocabulary length of
four we now have a sequence of 1 2 3 four five tokens but our vocabulary
length now is seven and so in this way we can iteratively compress our sequence
I we Mint new tokens so in the in the exact same way we start we start out with bite sequences so we have 256
vocabulary size but we're now going to go through these and find the bite pairs that occur the most and we're going to
iteratively start minting new tokens appending them to our vocabulary and replacing things and in this way we're
going to end up with a compressed training data set and also an algorithm for taking any arbitrary sequence and
encoding it using this uh vocabul and also decoding it back to Strings so
let's now Implement all that so here's what I did I went to this block post that I enjoyed and I took the first

在这段内容中，讲解者详细解释了**字节对编码（Byte Pair Encoding, BPE）算法**的工作原理，并演示了如何通过迭代的方式压缩一个序列。

### 主要内容：

1. **BPE的基本概念**：

   * **BPE算法**的核心思想是通过寻找频率最高的字符对（token pair），然后将这些频繁出现的字符对替换为新的单一字符。这个新字符会被添加到词汇表中，随着每次替换，词汇表会不断扩展，序列的长度则会压缩。

2. **压缩过程**：

   * 以一个简单的例子为例：假设我们的初始词汇表只有四个元素（a, b, c, d），并且我们有一个由这些元素组成的序列。我们的目标是通过BPE算法压缩这个序列。

   * **步骤一**：找到频率最高的字符对。例如，在这个例子中，字符对“AA”出现的频率最高，所以我们将其替换为一个新字符（假设是Z）。这就相当于将所有“AA”替换为“Z”，并把“Z”添加到词汇表中。

   * **步骤二**：接着，我们再检查剩下的序列，找出出现频率最高的字符对（假设是“AB”），并将其替换为一个新字符（假设是Y）。同样，所有“AB”都会被替换成“Y”，并将“Y”添加到词汇表中。

   * **步骤三**：继续重复上述过程，找到出现频率最高的字符对，并用新的字符替换，直到达到期望的压缩效果。

3. **示例过程**：

   * 初始时，我们有一个长度为11的字符序列，词汇表包含四个元素（a, b, c, d）。经过第一轮替换（将“AA”替换为“Z”），我们得到一个长度为9的序列，词汇表变为5个元素。
   * 经过第二轮替换（将“AB”替换为“Y”），序列长度变为7，词汇表变为6个元素。
   * 最后一轮替换（将“ZY”替换为“X”），最终我们得到一个长度为5的序列，词汇表变为7个元素。

4. **BPE在大语言模型中的应用**：

   * 在实际应用中，BPE从256个初始词汇（字节）开始，通过查找出现频率最高的字节对，不断生成新的词汇，并替换掉原序列中的字符对。最终，我们得到一个更压缩的训练数据集，并且有了一个新的编码方法，可以将任何序列用新的词汇表进行编码和解码。

### 总结：

* **BPE算法**通过迭代地压缩字符序列，逐步构建新的词汇表，并有效地减小了序列的长度。这个过程不仅可以用于压缩训练数据，还能帮助我们构建高效的编码和解码机制，使得大语言模型能够更加高效地处理文本。


# starting the implementation

paragraph and I copy pasted it here into text so this is one very long line
here now to get the tokens as I mentioned we just take our text and we encode it into utf8 the tokens here at
this point will be a raw bites single stream of bytes and just so that it's
easier to work with instead of just a bytes object I'm going to convert all those bytes to integers and then create
a list of it just so it's easier for us to manipulate and work with in Python and visualize and here I'm printing all
of that so this is the original um this is the original paragraph and its length
is 533 uh code points and then here are the bytes encoded in ut utf8 and we see that
this has a length of 616 bytes at this point or 616 tokens and the reason this
is more is because a lot of these simple asky characters or simple characters
they just become a single bite but a lot of these Unicode more complex characters become multiple bytes up to four and so
we are expanding that size so now what we'd like to do as a first step of the algorithm is we'd like
to iterate over here and find the pair of bites that occur most frequently
because we're then going to merge it so if you are working long on a notebook on a side then I encourage you to basically
click on the link find this notebook and try to write that function yourself otherwise I'm going to come here and
Implement first the function that finds the most common pair okay so here's what I came up with there are many different

在这段内容中，讲解者开始了实现\*\*字节对编码（BPE）\*\*算法的代码，并演示了如何从原始文本开始进行编码并实现算法的第一步。

### 主要内容：

1. **文本编码**：

   * 讲解者首先将文本转换为**UTF-8编码**，并生成一个**字节流**。这些字节流表示了文本的原始编码，而每个字符在UTF-8编码下可能会被转换为多个字节，尤其是Unicode字符（如复杂字符或符号）可能需要多个字节，而简单的ASCII字符通常只需要一个字节。

2. **字节流转换为整数**：

   * 为了更方便地操作和可视化，讲解者将字节流（bytes）转换为整数，并将这些整数保存到一个列表中。这样，在Python中就可以更轻松地处理字节数据。

3. **计算字节对频率**：

   * 在BPE算法中，**第一步是找到频率最高的字节对**。一旦找到了这些字节对，就可以将它们合并为一个新的符号。讲解者准备实现一个函数，用来找出最常见的字节对（pair of bytes）。这个函数将在后续的实现中使用。

4. **编码长度的变化**：

   * 讲解者展示了原始段落的长度为533个Unicode代码点，而通过UTF-8编码后，字节流的长度变为616字节。这是因为UTF-8编码的每个字符（尤其是Unicode字符）会被转换为多个字节，而不是单个字节。这会导致字节流长度的增加。

5. **算法实现**：

   * 讲解者建议，如果观众有时间，可以自己尝试编写这个函数，查找文本中最频繁的字节对。否则，讲解者将继续实现这个函数。

### 总结：

* 讲解者介绍了如何从文本开始，通过**UTF-8编码**将其转换为字节流，并将这些字节转化为整数列表，方便后续操作。接下来，讲解者将实现一个函数，找出频率最高的字节对，这是BPE算法的第一步。


这段代码展示了如何将文本转换为UTF-8编码的字节流，并将这些字节流转换为整数列表。让我们逐行解析：

### 代码解释：

1. **`tokens = text.encode("utf-8")`**：

   * 这行代码将原始文本 `text` 转换为 **UTF-8编码的字节流**。
   * `encode("utf-8")` 是Python中的字符串方法，它将字符串转换为UTF-8编码的字节（`bytes` 类型）。UTF-8编码是一种变长编码，它将文本中的字符转换为一个或多个字节。
   * 例如，如果 `text` 是 "hello"，那么 `tokens` 将是一个字节流（`b'h'`, `b'e'`, `b'l'`, `b'l'`, `b'o'` 对应UTF-8编码的字节值）。

2. **`tokens = list(map(int, tokens))`**：

   * `map(int, tokens)` 这一部分将字节流中的每个字节值转换为整数。具体来说，UTF-8编码中的每个字节是一个整数值（范围从0到255），`map(int, tokens)` 将每个字节转换为对应的整数值。
   * `list()` 将 `map` 返回的结果转为一个列表。这样，`tokens` 变量将存储一个整数列表，列表中的每个整数值对应UTF-8编码中每个字节的值。
   * 例如，如果 `tokens` 是 `b'h'`, `b'e'`, `b'l'`, `b'l'`, `b'o'`，那么它们的整数值会分别是 `[104, 101, 108, 108, 111]`，这表示的是它们对应的字节的整数值。

### 总结：

* **`text.encode("utf-8")`** 将文本转换为字节流（`bytes`），每个字符变成一个或多个字节。
* **`list(map(int, tokens))`** 将字节流中的字节转换为0到255范围内的整数，并生成一个整数列表，方便后续操作和处理。



# counting consecutive pairs, finding most common pair

ways to implement this but I'm calling the function get stats it expects a list of integers I'm using a dictionary to
keep track of basically the counts and then this is a pythonic way to iterate consecutive elements of this list uh
which we covered in the previous video and then here I'm just keeping track of just incrementing by one um for all the
pairs so if I call this on all the tokens here then the stats comes out here so this is the dictionary the keys
are these topples of consecutive elements and this is the count so just
to uh print it in a slightly better way this is one way that I like to do that
where you it's a little bit compound here so you can pause if you like but we iterate all all the items the items
called on dictionary returns pairs of key value and instead I create a list
here of value key because if it's a value key list then I can call sort on
it and by default python will uh use the first element which in this case will be
value to sort by if it's given tles and then reverse so it's descending and
print that so basically it looks like 101 comma 32 was the most commonly
occurring consecutive pair and it occurred 20 times we can double check that that makes reasonable sense so if I
just search 10132 then you see that these are the 20 occurrences of that um pair and if we'd
like to take a look at what exactly that pair is we can use Char which is the opposite of or in Python so we give it a
um unic code Cod point so 101 and of 32 and we see that this is e and space so
basically there's a lot of E space here meaning that a lot of these words seem to end with e so here's eace as an
example so there's a lot of that going on here and this is the most common pair so now that we've identified the most

这段代码展示了如何统计文本中\*\*连续字节对（token pairs）\*\*的出现频率，并找出最常见的连续字节对。下面是逐行解释：

### 代码解释：

1. **`get_stats`函数**：

   * 讲解者定义了一个名为`get_stats`的函数，用来统计给定整数列表（字节流）的连续字节对的频率。
   * 该函数使用一个\*\*字典（dictionary）\*\*来记录每一对连续字节（token pair）出现的次数。

2. **`for`循环和迭代**：

   * 这段代码使用一种**Pythonic**的方式来迭代列表中的连续元素。通过遍历列表，获取连续的两个元素作为一个**元组（tuple）**，然后在字典中统计它们的出现次数。

3. **字典的使用**：

   * 字典的键（key）是连续的字节对（元组），值（value）是该字节对的出现次数。例如，如果字节对 `(101, 32)` 出现了20次，则字典中 `(101, 32)` 对应的值为20。

4. **打印和排序**：

   * 讲解者提供了一个**打印字典的方式**，通过将字典中的键值对交换顺序（`value, key`），可以方便地按照**出现次数（value）**进行排序。这样，Python默认会根据**值**进行排序，并且通过`reverse`参数将其按降序排列。
   * 打印出的结果显示了最常见的字节对。例如，`(101, 32)`是最常出现的连续字节对，出现了20次。

5. **检查字节对**：

   * 通过查找字节对 `(101, 32)` 出现的位置，确认其出现的次数是否合理。讲解者在这部分还展示了如何通过**Unicode编码**将字节转换为字符，并发现这个字节对对应的是字符\*\*`e`**和**空格（space）\*\*。
   * 这意味着很多单词后面都以字母`e`和空格作为结尾，这符合英语单词的常见结构（例如 "eace" 可能是一个单词的一部分）。

### 总结：

* **统计字节对**：通过迭代整个字节流，找出连续的字节对并统计它们的出现次数。
* **字典排序**：使用字典保存统计数据，并通过交换键值对的顺序来按频率排序。
* **最常见的字节对**：通过`get_stats`函数，最终找到了最常见的连续字节对，`(101, 32)`，对应的是字符`e`和空格。

该过程帮助我们理解文本中最常见的字节对，并为后续的BPE算法准备了基础。


# merging the most common pair

common pair we would like to iterate over this sequence we're going to Mint a new token with the ID of
256 right because these tokens currently go from Z to 255 so when we create a new
token it will have an ID of 256 and we're going to iterate over this
entire um list and every every time we see 101 comma 32 we're going to swap
that out for 256 so let's Implement that now and feel free to uh do that yourself as well so
first I commented uh this just so we don't pollute uh the notebook too much this is a nice way of in Python
obtaining the highest ranking pair so we're basically calling the Max on this
dictionary stats and this will return the maximum key and then the question is how does it
rank keys so you can provide it with a function that ranks keys and that
function is just stats. getet uh stats. getet would basically return the value
and so we're ranking by the value and getting the maximum key so it's 101 comma 32 as we saw now to actually merge
10132 um this is the function that I wrote but again there are many different versions of it so we're going to take a
list of IDs and the the pair that we want to replace and that pair will be replaced with the new index
idx so iterating through IDs if we find the pair swap it out for idx so we
create this new list and then we start at zero and then we go through this entire list sequentially from left to
right and here we are checking for equality at the current position with the
pair um so here we are checking that the pair matches now here is a bit of a tricky condition that you have to append
if you're trying to be careful and that is that um you don't want this here to be out of Bounds at the very last
position when you're on the rightmost element of this list otherwise this would uh give you an autof bounds error
so we have to make sure that we're not at the very very last element so uh this would be false for that so if we find a
match we append to this new list that replacement index and we increment the
position by two so we skip over that entire pair but otherwise if we we haven't found a matching pair we just
sort of copy over the um element at that position and increment by one then
return this so here's a very small toy example if we have a list 566 791 and we
want to replace the occurrences of 67 with 99 then calling this on that will
give us what we're asking for so here the 67 is replaced with
99 so now I'm going to uncomment this for our actual use case where we want to
take our tokens we want to take the top pair here and replace it with 256 to get
tokens to if we run this we get the following so recall that previously we
had a length 616 in this list and now we have a length 596 right so this
decreased by 20 which makes sense because there are 20 occurrences moreover we can try to find 256 here and
we see plenty of occurrences on off it and moreover just double check there should be no occurrence of 10132 so this
is the original array plenty of them and in the second array there are no occurrences of 1032 so we've
successfully merged this single pair and now we just uh iterate this so we are
going to go over the sequence again find the most common pair and replace it so let me now write a y Loop that uses
these functions to do this um sort of iteratively and how many times do we do it four well that's totally up to us as
a hyper parameter the more um steps we take the larger will be our vocabulary and the shorter
will be our sequence and there is some sweet spot that we usually find works the best in practice and so this is kind
of a hyperparameter and we tune it and we find good vocabulary sizes as an example gp4 currently uses roughly
100,000 tokens and um bpark that those are reasonable numbers currently instead
the are large language models so let me now write uh putting putting it all together and uh iterating these steps

这段代码介绍了如何**合并最常见的字节对**（token pairs），并逐步构建一个新的词汇表。我们使用**Byte Pair Encoding (BPE)** 算法对文本进行压缩。下面是逐行解释：

### 代码解析：

1. **合并最常见的字节对**：

   * 在这段代码中，我们首先找到最常见的字节对（例如 `(101, 32)`，对应字符 `e` 和空格）。我们为这个字节对创建一个新的 token（例如，`256`），并将其加入到词汇表中。
   * 例如，如果最常见的字节对是 `(101, 32)`，那么我们会用新的 ID `256` 替换所有 `(101, 32)` 出现的地方。

2. **创建新的 token 并替换**：

   * 我们创建了一个新的索引 `256`，然后遍历整个 token 列表，每当找到 `(101, 32)` 这个字节对时，就把它替换成新生成的 token `256`。
   * **代码的处理方式**：

     * 我们通过遍历整个列表，检查当前元素是否是我们要替换的字节对。如果是，就将其替换为新的 token。
     * 需要特别注意的是，列表的最后一个元素可能会导致越界错误，因此需要在遍历时判断是否是最后一个元素，避免访问越界。

3. **执行替换操作**：

   * 例如，如果我们有一个包含 `[566, 791]` 的小列表，且我们想将 `67` 替换成 `99`，替换后的列表将变为 `[599, 791]`。这展示了替换操作的工作方式。
   * 在实际的操作中，我们遍历整个 tokens 列表，将每一个匹配的字节对替换为新生成的 token。例如，`101, 32` 被替换为 `256`。

4. **计算序列长度的变化**：

   * 在完成替换后，我们发现原来的序列长度从 `616` 减少到了 `596`。这意味着我们替换了 20 次字节对，因此总的 token 数量减少了 20。
   * 进一步检查后，我们确认在新序列中不再出现 `101, 32`，而是出现了新的 token `256`，替换工作成功完成。

5. **迭代过程**：

   * 在实际应用中，我们会不断重复这一过程：每次找到最常见的字节对并将其合并，直到达到预设的迭代次数。
   * 迭代次数的选择是一个超参数，决定了词汇表的大小和序列的长度。通常情况下，迭代次数越多，生成的词汇表越大，压缩后的序列长度越短。
   * 实际应用中，像 GPT-4 使用的 token 数量大约是 100,000，因此在实践中我们会选择合适的超参数来平衡词汇表的大小和计算效率。

### 总结：

* **合并最常见的字节对**是 Byte Pair Encoding (BPE) 算法的核心步骤，通过替换高频字节对来减少序列的长度。
* 每次替换时都会生成新的 token，并不断增加词汇表的大小。
* **超参数调整**：通过调整迭代次数，可以控制生成的词汇表大小和压缩后的序列长度，通常选择一个合适的平衡点来提高模型性能和效率。


# training the tokenizer: adding the while loop, compression ratio

okay now before we dive into the Y loop I wanted to add one more cell here where I went to the block post and instead of
grabbing just the first paragraph or two I took the entire block post and I stretched it out in a single line and
basically just using longer text will allow us to have more representative statistics for the bite Pairs and we'll
just get a more sensible results out of it because it's longer text um so here
we have the raw text we encode it into bytes using the utf8 encoding
and then here as before we are just changing it into a list of integers in Python just so it's easier to work with
instead of the raw byes objects and then this is the code that I came up with uh
to actually do the merging in Loop these two functions here are identical to what
we had above I only included them here just so that you have the point of reference here so uh these two are
identical and then this is the new code that I added so the first first thing we want to do is we want to decide on the
final vocabulary size that we want our tokenizer to have and as I mentioned this is a hyper parameter and you set it
in some way depending on your best performance so let's say for us we're going to use 276 because that way we're
going to be doing exactly 20 merges and uh 20 merges because we already have
256 tokens for the raw bytes and to reach 276 we have to do 20 merges uh to
add 20 new tokens here uh this is uh one way in Python to just create a copy of a list
so I'm taking the tokens list and by wrapping it in a list python will construct a new list of all the
individual elements so this is just a copy operation then here I'm creating a merges uh dictionary so this merges
dictionary is going to maintain basically the child one child two mapping to a new uh token and so what
we're going to be building up here is a binary tree of merges but actually it's not exactly a tree because a tree would
have a single root node with a bunch of leaves for us we're starting with the leaves on the bottom which are the
individual bites those are the starting 256 tokens and then we're starting to like merge two of them at a time and so
it's not a tree it's more like a forest um uh as we merge these elements
so for 20 merges we're going to find the most commonly occurring pair we're going
to Mint a new token integer for it so I here will start at zero so we'll going to start at 256 we're going to print
that we're merging it and we're going to replace all of the occurrences of that pair with the new new lied token and
we're going to record that this pair of integers merged into this new
integer so running this gives us the following
output so we did 20 merges and for example the first merge was exactly as
before the 10132 um tokens merging into a new token 2556 now keep in mind that the
individual uh tokens 101 and 32 can still occur in the sequence after merging it's only when they occur
exactly consecutively that that becomes 256 now um and in particular the other thing
to notice here is that the token 256 which is the newly minted token is also eligible for merging so here on the
bottom the 20th merge was a merge of 25 and 259 becoming
275 so every time we replace these tokens they become eligible for merging in the next round of data ration so
that's why we're building up a small sort of binary Forest instead of a single individual tree one thing we can take a look at as
well is we can take a look at the compression ratio that we've achieved so in particular we started off with this
tokens list um so we started off with 24,000 bytes and after merging 20 times
uh we now have only 19,000 um tokens and so therefore the
compression ratio simply just dividing the two is roughly 1.27 so that's the amount of compression we were able to
achieve of this text with only 20 merges um and of course the more
vocabulary elements you add uh the greater the compression ratio here would
be finally so that's kind of like um the training of the tokenizer if you will

这段代码讲解了如何**训练一个Tokenizer**，并通过**合并字节对**（Byte Pair Encoding, BPE）来逐步压缩文本，从而创建一个有效的词汇表。接下来，我将分步骤讲解代码的工作原理。

### 1. **准备数据和初始化**

* **原始文本**：首先，我们将一整篇文章的文本读取进来，并将其转化为UTF-8编码的字节流。然后，将这些字节转换为一个**整数列表**，以便于后续的操作。这样我们得到的不是原始字节对象，而是易于处理的整数列表。

```python
tokens = text.encode("utf-8")  # 使用UTF-8编码将文本转换为字节
tokens = list(map(int, tokens))  # 将字节转换为整数列表，范围是 0 到 255
```

### 2. **设定词汇表大小**

* **词汇表大小（Vocabulary size）**：这里我们定义了目标词汇表的大小。为了创建一个具有20个新词的词汇表，我们需要进行**20次合并**操作。这样，词汇表大小从原来的256（原始字节）增加到276。每次合并后，新的token将被添加到词汇表中。

```python
vocab_size = 276  # 设置目标词汇表大小
merges_count = vocab_size - 256  # 计算需要合并的次数
```

* **复制tokens列表**：通过 `tokens.copy()` 来创建原始token列表的副本，这样可以避免直接修改原始数据。

```python
tokens_copy = tokens.copy()
```

### 3. **执行字节对合并**

* **合并操作**：对于每一次合并，我们首先通过 `get_stats()` 函数找到最常见的字节对，然后生成一个新的token。这个新token的ID是从 `256` 开始递增的。接着，我们在整个token序列中替换所有的匹配对，最终形成新的词汇表。

```python
# 获取最常见的字节对
pair = get_stats(tokens_copy)
# 合并字节对，生成新token
new_token = len(merges) + 256
tokens_copy = merge_pair(tokens_copy, pair, new_token)
```

* 这样我们逐步将多个最常见的字节对合并为新token，直到达到预定的词汇表大小。

### 4. **更新词汇表并记录合并历史**

* 每次合并后，我们将新的字节对及其生成的token记录下来。新token也会参与后续的合并。最终，我们会得到一个类似“森林”（而非树形结构）的合并结构，其中每个token都有可能被多次合并。

```python
merges[(pair[0], pair[1])] = new_token  # 记录合并操作
```

* 每次合并之后，生成的新token会成为接下来合并的候选者之一，这就是为什么最终会形成一个“森林”结构，而不是一个单一的树形结构。

### 5. **计算压缩比（Compression Ratio）**

* **压缩比**：合并后的序列比原始序列更短，这个压缩效果可以通过计算**字节数的减少**来体现。比如在这段代码中，原始的字节数是 24,000，而经过20次合并后，字节数减少到19,000。压缩比为：

```python
compression_ratio = original_size / new_size  # 计算压缩比
```

* 这里的压缩比是大约1.27，意味着文本大小减少了大约27%。
* 更高的词汇表大小（更多的合并次数）会带来更高的压缩比。

### 6. **总结**

* 通过这种迭代的字节对合并过程，我们训练了一个简易的tokenizer，并且能够压缩文本序列。这个过程涉及到反复地找出最常见的字节对，并用新的token替换它们。
* 词汇表的大小和压缩比是**超参数**，可以根据实际需求调整。
* 最终，通过合并字节对，我们能高效地减少文本的表示长度，同时保留必要的文本信息，适用于后续的Transformer模型训练。

这个过程其实模拟了如何通过**BPE算法**训练一个文本编码器（Tokenizer），并且使用压缩比来评估算法的效果。


# tokenizer/LLM diagram: it is a completely separate stage

now 1 Point I wanted to make is that and maybe this is a diagram that can help um
kind of illustrate is that tokenizer is a completely separate object from the large language model itself so
everything in this lecture we're not really touching the llm itself uh we're just training the tokenizer this is a completely separate pre-processing stage
usually so the tokenizer will have its own training set just like a large language model has a potentially
different training set so the tokenizer has a training set of documents on which you're going to train the tokenizer and then and um we're
performing The Bite pair encoding algorithm as we saw above to train the vocabulary of this tokenizer so it has its own training set
it is a pre-processing stage that you would run a single time in the beginning um and the tokenizer is trained using
bipar coding algorithm once you have the tokenizer once it's trained and you have the vocabulary and you have the merges
uh we can do both encoding and decoding so these two arrows here so the
tokenizer is a translation layer between raw text which is as we saw the sequence
of Unicode code points it can take raw text and turn it into a token sequence
and vice versa it can take a token sequence and translate it back into raw
text so now that we have trained uh tokenizer and we have these merges we
are going to turn to how we can do the encoding and the decoding step if you give me text here are the tokens and
vice versa if you give me tokens here's the text once we have that we can translate between these two Realms and
then the language model is going to be trained as a step two afterwards and typically in a in a sort of a
state-of-the-art application you might take all of your training data for the language model and you might run it through the tokenizer and sort of
translate everything into a massive token sequence and then you can throw away the raw text you're just left with
the tokens themselves and those are stored on disk and that is what the large language model is actually reading
when it's training on them so this one approach that you can take as a single massive pre-processing step a
stage um so yeah basically I think the most important thing I want to get across is that this is completely
separate stage it usually has its own entire uh training set you may want to have those training sets be different
between the tokenizer and the logge language model so for example when you're training the tokenizer as I mentioned we don't just care about the
performance of English text we care about uh multi many different languages and we also care about code or not code
so you may want to look into different kinds of mixtures of different kinds of languages and different amounts of code
and things like that because the amount of different language that you have in your tokenizer training set will
determine how many merges of it there will be and therefore that determines the density with which uh this type of
data is um sort of has in the token space and so roughly speaking
intuitively if you add some amount of data like say you have a ton of Japanese data in your uh tokenizer training set
then that means that more Japanese tokens will get merged and therefore Japanese will have shorter sequences uh and that's going to be
beneficial for the large language model which has a finite context length on which it can work on in in the token
space uh so hopefully that makes sense so we're now going to turn to encoding and decoding now that we have trained a
tokenizer so we have our merges and now how do we do encoding and decoding okay

在这个阶段，我们讨论了 **tokenizer** 和 **大型语言模型 (LLM)** 之间的关系。tokenizer 是与 LLM 完全独立的对象，负责将原始文本（如 Unicode 代码点）转换为一系列的 token，反之亦然。tokenizer 的训练过程使用了 **字节对编码 (BPE)** 算法，而这个训练通常在大型语言模型训练之前完成。tokenizer 会有自己的训练数据集，专门用于训练词汇表和合并规则。一旦 tokenizer 被训练出来，它就能对任何输入文本进行编码和解码，生成 token 或者将 token 转换回文本。

整个过程的关键点在于 tokenizer 与 LLM 是两个独立的处理阶段。在实际应用中，我们通常会先通过 tokenizer 对所有文本数据进行预处理，得到 token 列表，然后将这些 token 存储起来，供 LLM 使用。


# decoding tokens to strings

so let's begin with decoding which is this Arrow over here so given a token sequence let's go through the tokenizer
to get back a python string object so the raw text so this is the function that we' like to implement um we're
given the list of integers and we want to return a python string if you'd like uh try to implement this function yourself it's a fun exercise otherwise
I'm going to start uh pasting in my own solution so there are many different ways to do it um here's one way I will
create an uh kind of pre-processing variable that I will call vocab and vocab is a mapping or a
dictionary in Python for from the token uh ID to the bytes object for that token
so we begin with the raw bytes for tokens from 0 to 255 and then we go in
order of all the merges and we sort of uh populate this vocab list by doing an
addition here so this is the basically the bytes representation of the first
child followed by the second one and remember these are bytes objects so this addition here is an addition of two
bytes objects just concatenation so that's what we get here one tricky thing to be careful with
by the way is that I'm iterating a dictionary in Python using a DOT items and uh it really matters that this runs
in the order in which we inserted items into the merous dictionary luckily starting with python 3.7 this is
guaranteed to be the case but before python 3.7 this iteration may have been out of order with respect to how we
inserted elements into merges and this may not have worked but we are using an um modern python so we're okay and then
here uh given the IDS the first thing we're going to do is get the
tokens so the way I implemented this here is I'm taking I'm iterating over all the IDS I'm using vocap to look up
their bytes and then here this is one way in Python to concatenate all these bytes together to create our tokens and
then these tokens here at this point are raw bytes so I have to decode using UTF
F now back into python strings so previously we called that encode on a
string object to get the bytes and now we're doing it Opposite we're taking the bytes and calling a decode on the bytes
object to get a string in Python and then we can return
text so um this is how we can do it now this actually has a um issue um in the
way I implemented it and this could actually throw an error so try to think figure out why this code could actually
result in an error if we plug in um uh some sequence of IDs that is
unlucky so let me demonstrate the issue when I try to decode just something like 97 I am going to get letter A here back
so nothing too crazy happening but when I try to decode 128 as a single element
the token 128 is what in string or in Python object uni Cod decoder utfa can't
Decode by um 0x8 which is this in HEX in position zero invalid start bite what
does that mean well to understand what this means we have to go back to our utf8 page uh that I briefly showed
earlier and this is Wikipedia utf8 and basically there's a specific schema that
utfa bytes take so in particular if you have a multi-te object for some of the
Unicode characters they have to have this special sort of envelope in how the encoding works and so what's happening
here is that invalid start pite that's because 128 the binary representation of it is
one followed by all zeros so we have one and then all zero and we see here that
that doesn't conform to the format because one followed by all zero just doesn't fit any of these rules so to
speak so it's an invalid start bite which is byte one this one must have a
one following it and then a zero following it and then the content of your uni codee in x here so basically we
don't um exactly follow the utf8 standard and this cannot be decoded and so the way to fix this um is to
use this errors equals in bytes. decode function of python and by default errors
is strict so we will throw an error if um it's not valid utf8 bytes encoding
but there are many different things that you could put here on error handling this is the full list of all the errors
that you can use and in particular instead of strict let's change it to replace and that will replace uh with
this special marker this replacement character so errors equals replace and
now we just get that character back so basically not every single by
sequence is valid utf8 and if it happens that your large language model for example predicts your
tokens in a bad manner then they might not fall into valid utf8 and then we
won't be able to decode them so the standard practice is to basically uh use
errors equals replace and this is what you will also find in the openai um code that they released as well but basically
whenever you see um this kind of a character in your output in that case uh something went wrong and the LM output
not was not valid uh sort of sequence of tokens okay and now we're going to go

在这一部分，我们讨论了 **解码** 过程，目标是将一组 token ID 转换回原始文本字符串。

1. **初始化 `vocab` 字典**:

   * `vocab` 是一个字典，用于存储 token ID 与其对应的字节对象。开始时，token ID 从 0 到 255，直接映射到相应的字节。
   * 随着字节对编码（BPE）过程的进行，我们会根据之前的合并规则，更新 `vocab` 字典，将合并后的 token 对应的字节对象添加进字典。

2. **解码过程**:

   * 给定一个 token ID 列表，我们使用 `vocab` 字典找到每个 token 对应的字节值。
   * 然后，将这些字节值合并在一起，形成一个原始的字节流。
   * 最后，我们通过 UTF-8 解码这些字节流，转换成 Python 字符串。

3. **潜在问题**:

   * 在某些情况下，解码可能会失败，特别是当某些 token 生成的字节序列不是有效的 UTF-8 编码时。比如，如果某个 token 的字节流的起始字节不符合 UTF-8 编码的规范，就会抛出 "invalid start byte" 错误。

4. **解决方法**:

   * 解决这个问题的方法是在调用 `decode()` 函数时使用 `errors='replace'` 参数，这样如果遇到无效的字节流，会将其替换为一个特殊的替代符号，而不会抛出错误。
   * 这是 OpenAI 代码中常见的做法，确保即使生成的 token 不是有效的 UTF-8 编码，程序也能继续运行。

通过这种方式，我们能够有效地解码一组 token，并将其恢复为原始文本。在实际应用中，这种方法有助于处理因编码问题而导致的潜在错误。


# encoding strings to tokens

the other way so we are going to implement this Arrow right here where we are going to be given a string and we want to
encode it into tokens so this is the signature of the function that we're interested in and um
this should basically print a list of integers of the tokens so again uh try to maybe implement this yourself if
you'd like a fun exercise uh and pause here otherwise I'm going to start putting in my solution so again there are many ways to
do this so um this is one of the ways that sort of I came came up with so the
first thing we're going to do is we are going to uh take our text encode it into utf8
to get the raw bytes and then as before we're going to call list on the bytes object to get a list of integers of
those bytes so those are the starting tokens those are the raw bytes of our sequence but now of course according to
the merges dictionary above and recall this was the merges some of the bytes may be merged
according to this lookup in addition to that remember that the merges was built from top to bottom and this is sort of
the order in which we inserted stuff into merges and so we prefer to do all these merges in the beginning before we
do these merges later because um for example this merge over here relies on the 256 which got merged here so we have
to go in the order from top to bottom sort of if we are going to be merging anything now we expect to be doing a few
merges so we're going to be doing W true um and now we want to find a pair
of byes that is consecutive that we are allowed to merge according to this in
order to reuse some of the functionality that we've already written I'm going to reuse the function uh get
stats so recall that get stats uh will give us the we'll basically count up how
many times every single pair occurs in our sequence of tokens and return that as a dictionary and the dictionary was a
mapping from all the different uh by pairs to the number of times that they
occur right um at this point we don't actually care how many times they occur in the sequence we only care what the
raw pairs are in that sequence and so I'm only going to be using basically the keys of the dictionary I only care about
the set of possible merge candidates if that makes sense now we want to identify the pair
that we're going to be merging at this stage of the loop so what do we want we want to find the pair or like the a key
inside stats that has the lowest index in the merges uh dictionary because we
want to do all the early merges before we work our way to the late merges so again there are many different
ways to implement this but I'm going to do something a little bit fancy
here so I'm going to be using the Min over an iterator in Python when you call
Min on an iterator and stats here as a dictionary we're going to be iterating the keys of this dictionary in Python so
we're looking at all the pairs inside stats um which are all the consecutive
Pairs and we're going to be taking the consecutive pair inside tokens that has
the minimum what the Min takes a key which gives us the function that is
going to return a value over which we're going to do the Min and the one we care about is we're we care about taking
merges and basically getting um that pairs
index so basically for any pair inside stats we are going to be looking into
merges at what index it has and we want to get the pair with the Min number so
as an example if there's a pair 101 and 32 we definitely want to get that pair uh we want to identify it here and
return it and pair would become 10132 if it occurs and the reason that I'm putting a
float INF here as a fall back is that in the get function when we call uh when we
basically consider a pair that doesn't occur in the merges then that pair is not eligible to be merged right so if in
the token sequence there's some pair that is not a merging pair it cannot be merged then uh it doesn't actually occur
here and it doesn't have an index and uh it cannot be merged which we will denote as float INF and the reason Infinity is
nice here is because for sure we're guaranteed that it's not going to participate in the list of candidates when we do the men so uh so this is one
way to do it so B basically long story short this Returns the most eligible merging candidate pair uh that occurs in
the tokens now one thing to be careful with here is this uh function here might
fail in the following way if there's nothing to merge then uh uh then there's
nothing in merges um that satisfi that is satisfied anymore there's nothing to merge everything just returns float imps
and then the pair I think will just become the very first element of stats
um but this pair is not actually a mergeable pair it just becomes the first pair inside stats arbitrarily because
all of these pairs evaluate to float in for the merging Criterion so basically
it could be that this this doesn't look succeed because there's no more merging pairs so if this pair is not in merges
that was returned then this is a signal for us that actually there was nothing to merge no single pair can be merged
anymore in that case we will break out um nothing else can be
merged you may come up with a different implementation by the way this is kind of like really trying hard in
Python um but really we're just trying to find a pair that can be merged with the lowest index
here now if we did find a pair that is inside merges with the lowest index then
we can merge it so we're going to look into the merger
dictionary for that pair to look up the index and we're going to now merge that
into that index so we're going to do tokens equals and we're going to replace the original tokens we're going
to be replacing the pair pair and we're going to be replacing it with index idx and this returns a new list of tokens
where every occurrence of pair is replaced with idx so we're doing a merge and we're going to be continuing this
until eventually nothing can be merged we'll come out here and we'll break out and here we just return
tokens and so that that's the implementation I think so hopefully this runs okay cool um yeah and this looks uh
reasonable so for example 32 is a space in asky so that's here um so this looks
like it worked great okay so let's wrap up this section of the video at least I wanted to point out that this is not
quite the right implementation just yet because we are leaving out a special case so in particular if uh we try to do
this this would give us an error and the issue is that um if we only have a single character or an empty string then
stats is empty and that causes an issue inside Min so one way to fight this is if L of tokens is at least two because
if it's less than two it's just a single token or no tokens then let's just uh there's nothing to merge so we just
return so that would fix uh that case Okay and then second I have a few
test cases here for us as well so first let's make sure uh about or let's note
the following if we take a string and we try to encode it and then decode it back
you'd expect to get the same string back right is that true for all
strings so I think uh so here it is the case and I think in general this is probably the case um but notice that
going backwards is not is not you're not going to have an identity going backwards because as I mentioned us not
all token sequences are valid utf8 uh sort of by streams and so so therefore
you're some of them can't even be decodable um so this only goes in One
Direction but for that one direction we can check uh here if we take the training text which is the text that we
train to tokenizer around we can make sure that when we encode and decode we get the same thing back which is true
and here I took some validation data so I went to I think this web page and I grabbed some text so this is text that
the tokenizer has not seen and we can make sure that this also works um okay so that gives us some confidence that
this was correctly implemented so those are the basics of the bite pair encoding algorithm we saw how we can uh
take some training set train a tokenizer the parameters of this tokenizer really are just this dictionary of merges and
that basically creates the little binary Forest on top of raw bites once we have this the merges table
we can both encode and decode between raw text and token sequences so that's the the simplest setting of The
tokenizer what we're going to do now though is we're going to look at some of the St the art lar language models and
the kinds of tokenizers that they use and we're going to see that this picture complexifies very quickly so we're going
to go through the details of this comp complexification one at a time so let's

这一部分讨论了 **将字符串编码为 token** 的过程，目的是将文本（字符串）转换为模型可以处理的 token ID 序列。

### 步骤解析：

1. **文本编码**：

   * 首先，我们将输入的文本（字符串）转换为 **UTF-8** 编码的字节流。
   * 然后，将字节流转换为整数列表（每个字节对应一个整数），这些整数就是最初的 token。

2. **合并操作**：

   * 在上述的字节流基础上，我们会根据 **BPE (Byte Pair Encoding)** 生成的合并字典进行合并操作。这个字典记录了哪些字节对需要被合并为一个新的 token。
   * 在 BPE 过程中，我们将按顺序进行合并，从字典中的最早合并对开始，直到没有更多可合并的对为止。

3. **寻找可合并的对**：

   * 我们使用之前编写的 `get_stats` 函数来统计所有连续字节对的出现次数。
   * 之后，我们要找到 **合并字典（merges）** 中顺序最早的字节对，优先合并这些字节对，因为后续的合并可能依赖于前面的合并。

4. **实现合并**：

   * 一旦找到可以合并的字节对，就用一个新的 token ID 替换原始字节对，直到没有更多的字节对可以合并。
   * 在合并过程中，每当我们替换了一个字节对，就会生成一个新的 token，这样我们的 token 列表就变得更加紧凑。

5. **特定情况的处理**：

   * 如果只有一个字符或字符串为空，则没有字节对可供合并，函数会提前返回，避免出错。

6. **检查编码与解码的一致性**：

   * 最终，我们检查了编码和解码的结果是否一致。也就是说，当我们对一个字符串进行编码后，再解码回来，应该得到相同的原始字符串。
   * 但是需要注意，并不是所有的 token 序列都能有效解码，因为有些字节流可能不符合 UTF-8 编码的规范，这可能导致解码失败。

7. **验证**：

   * 在编码和解码过程中，我们可以使用训练数据和未见过的验证数据来确保编码解码过程正常工作。如果能成功地进行编码和解码并还原原始文本，说明我们的实现是正确的。

### 总结：

这个过程展示了如何通过字节对编码算法（BPE）将文本转换为 token ID 序列，并且能够在需要时进行解码。这是训练 tokenizer 和大型语言模型时的一个基本步骤。

我们来通过一个简单的例子来解释这行代码：

```python
pair = min(stats, key=lambda p: merges.get(p, float("inf")))
```

### 假设：

1. `stats` 是一个字典，包含了当前 `tokens` 列表中所有连续字节对的频率（也就是每对连续字节的出现次数）。
2. `merges` 是一个字典，包含了已经合并的字节对和它们对应的合并索引。

### 具体例子：

假设当前的 `stats` 字典如下：

```python
stats = {
    (97, 32): 4,  # 'a' followed by space appears 4 times
    (101, 32): 7,  # 'e' followed by space appears 7 times
    (97, 101): 5,  # 'a' followed by 'e' appears 5 times
}
```

`merges` 字典如下：

```python
merges = {
    (97, 32): 256,  # 'a' followed by space has been merged into token 256
    (97, 101): 257,  # 'a' followed by 'e' has been merged into token 257
}
```

### 解释：

这行代码的作用是，从 `stats` 字典中选择一个频率最高的连续字节对，并且这个字节对是可以合并的（即它已经出现在 `merges` 字典中）。

#### 1. **`min(stats, key=...)`**:

* `min(stats)` 会从 `stats` 字典的键（即字节对）中选出 **最小的那个键**，但它会基于给定的 `key` 函数来决定排序的标准。

#### 2. **`key=lambda p: merges.get(p, float("inf"))`**:

* 这个 `key` 函数是用来为每个字节对 `p` 获取它在 `merges` 字典中的值。
* `merges.get(p, float("inf"))`：对于每个字节对 `p`，我们查找它是否已经在 `merges` 字典中。

  * 如果存在，它会返回合并后的索引（比如 256 或 257）。
  * 如果不存在，它会返回 `float("inf")`，表示这个字节对不能再被合并。

#### 3. **工作流程：**

* 让我们看一下对 `stats` 中每个字节对的处理：

  * `(97, 32)`：这个字节对已经在 `merges` 中（对应的合并索引是 256），所以 `merges.get((97, 32), float("inf"))` 返回 `256`。
  * `(101, 32)`：这个字节对没有出现在 `merges` 字典中，所以 `merges.get((101, 32), float("inf"))` 返回 `float("inf")`。
  * `(97, 101)`：这个字节对已经在 `merges` 中（对应的合并索引是 257），所以 `merges.get((97, 101), float("inf"))` 返回 `257`。

* 然后 `min` 会选择合并索引最小的字节对：

  * `(97, 32)` 的合并索引是 `256`，而其他两个字节对的合并索引分别是 `float("inf")` 和 `257`。
  * 因此，最小的合并索引是 `256`，对应的字节对就是 `(97, 32)`。

### 结果：

最终，`pair` 会被赋值为 `(97, 32)`，因为它是可以合并的，并且在 `merges` 字典中的索引最小。

### 总结：

这行代码的目的是从 `stats` 字典中找到 **最常见的可合并字节对**，即根据 `merges` 字典中的合并索引来排序，选择出可以合并的最优字节对。


# regex patterns to force splits across categories

kick things off by looking at the GPD Series so in particular I have the gpt2 paper here um and this paper is from
2019 or so so 5 years ago and let's scroll down to input representation this
is where they talk about the tokenizer that they're using for gpd2 now this is all fairly readable so I encourage you
to pause and um read this yourself but this is where they motivate the use of the bite pair encoding algorithm on the
bite level representation of utf8 encoding so this is where they motivate it and they talk about the vocabulary
sizes and everything now everything here is exactly as we've covered it so far but things start to depart around here
so what they mention is that they don't just apply the naive algorithm as we have done it and in particular here's a
example suppose that you have common words like dog what will happen is that dog of course occurs very frequently in
the text and it occurs right next to all kinds of punctuation as an example so doc dot dog exclamation mark dog
question mark Etc and naively you might imagine that the BP algorithm could merge these to be single tokens and then
you end up with lots of tokens that are just like dog with a slightly different punctuation and so it feels like you're
clustering things that shouldn't be clustered you're combining kind of semantics with uation and this uh feels suboptimal and
indeed they also say that this is suboptimal according to some of the experiments so what they want to do is
they want to top down in a manual way enforce that some types of um characters
should never be merged together um so they want to enforce these merging rules
on top of the bite PA encoding algorithm so let's take a look um at their code
and see how they actually enforce this and what kinds of mergy they actually do perform so I have to to tab open here
for gpt2 under open AI on GitHub and when we go to Source there is an encoder thatp now I
don't personally love that they call it encoder dopy because this is the tokenizer and the tokenizer can do both
encode and decode uh so it feels kind of awkward to me that it's called encoder but that is the tokenizer and there's a
lot going on here and we're going to step through it in detail at one point for now I just want to focus on this
part here the create a rigix pattern here that looks very complicated and we're going to go through it in a bit uh
but this is the core part that allows them to enforce rules uh for what parts
of the text Will Never Be merged for sure now notice that re. compile here is a little bit misleading because we're
not just doing import re which is the python re module we're doing import reex as re and reex is a python package that
you can install P install r x and it's basically an extension of re so it's a bit more powerful
re um so let's take a look at this pattern and
what it's doing and why this is actually doing the separation that they are looking for okay so I've copy pasted the
pattern here to our jupit notebook where we left off and let's take this pattern for a spin so in the exact same way that
their code does we're going to call an re. findall for this pattern on any
arbitrary string that we are interested so this is the string that we want to encode into tokens um to feed into n llm
like gpt2 so what exactly is this doing well re. findall will take this pattern
and try to match it against a string um the way this works is that you
are going from left to right in the string and you're trying to match the pattern and R.F find all will get all
the occurrences and organize them into a list now when you look at the um when
you look at this pattern first of all notice that this is a raw string um and then these are three double quotes just
to start the string so really the string itself this is the pattern itself right and notice that it's made up of a
lot of ores so see these vertical bars those are ores in reg X and so you go
from left to right in this pattern and try to match it against the string wherever you are so we have hello and
we're going to try to match it well it's not apostrophe s it's not apostrophe t or any of these but it is an optional
space followed by- P of uh sorry SL P of L one or more times what is/ P of L it
is coming to some documentation that I found um there might be other sources as
well uh SLP is a letter any kind of letter from any language and hello is
made up of letters h e l Etc so optional space followed by a bunch of letters one
or more letters is going to match hello but then the match ends because a white
space is not a letter so from there on begins a new sort of attempt to match
against the string again and starting in here we're going to skip over all of these again until we get to the exact
same Point again and we see that there's an optional space this is the optional space followed by a bunch of letters one
or more of them and so that matches so when we run this we get a list of two
elements hello and then space world so how are you if we add more letters we
would just get them like this now what is this doing and why is this important we are taking our string and instead of
directly encoding it um for tokenization we are first splitting it
up and when you actually step through the code and we'll do that in a bit more detail what really is doing on a high
level is that it first splits your text into a list of texts just like this one
and all these elements of this list are processed independently by the tokenizer and all of the results of that
processing are simply concatenated so hello world oh I I
missed how hello world how are you we have five elements of list all of these
will independent independently go from text to a token
sequence and then that token sequence is going to be concatenated it's all going to be joined up and roughly speaking
what that does is you're only ever finding merges between the elements of this list so you can only ever consider
merges within every one of these elements in individually and um after you've done
all the possible merging for all of these elements individually the results of all that will be joined um by
concatenation and so you are basically what what you're doing effectively is you are never going to be merging this e
with this space because they are now parts of the separate elements of this list and so you are saying we are never
going to merge eace um because we're breaking it up in this way so basically using this regx
pattern to Chunk Up the text is just one way of enforcing that some merges are
not to happen and we're going to go into more of this text and we'll see that what this is trying to do on a high level is we're trying to not merge
across letters across numbers across punctuation and so on so let's see in
more detail how that works so let's continue now we have/ P ofn if you go to the documentation SLP of n is any kind
of numeric character in any script so it's numbers so we have an optional space followed by numbers and those
would be separated out so letters and numbers are being separated so if I do Hello World 123 how are you then world
will stop matching here because one is not a letter anymore but one is a number so this group will match for that and
we'll get it as a separate entity uh let's see how these apostrophes work
so here if we have um uh Slash V or I mean apostrophe V as
an example then apostrophe here is not a letter or a number so hello will stop matching and
then we will exactly match this with that so that will come out as a separate
thing so why are they doing the apostrophes here honestly I think that these are just like very common
apostrophes p uh that are used um typically I don't love that they've done
this because uh let me show you what happens when you have uh some Unicode
apostrophes like for example you can have if you have house then this will be
separated out because of this matching but if you use the Unicode apostrophe like
this then suddenly this does not work and so this apostrophe will actually
become its own thing now and so so um it's basically hardcoded for this specific kind of apostrophe and uh
otherwise they become completely separate tokens in addition to this you can go to the gpt2 docs and here when
they Define the pattern they say should have added re. ignore case so BP merges can happen for capitalized versions of
contractions so what they're pointing out is that you see how this is apostrophe and then lowercase letters
well because they didn't do re. ignore case then then um these rules will not
separate out the apostrophes if it's uppercase so house would be like this but if I did
house if I'm uppercase then notice suddenly the apostrophe comes by
itself so the tokenization will work differently in uppercase and lower case
inconsistently separating out these apostrophes so it feels extremely gnarly and slightly gross um but that's that's
how that works okay so let's come back after trying to match a bunch of apostrophe Expressions by the way the
other issue here is that these are quite language specific probably so I don't know that all the languages for example
use or don't use apostrophes but that would be inconsistently tokenized as a result then we try to match letters then
we try to match numbers and then if that doesn't work we fall back to here and
what this is saying is again optional space followed by something that is not a letter number or a space in one or
more of that so what this is doing effectively is this is trying to match punctuation roughly speaking not letters
and not numbers so this group will try to trigger for that so if I do something like this then these parts here are not
letters or numbers but they will actually they are uh they will actually get caught here and so they become its
own group so we've separated out the punctuation and finally this um this is
also a little bit confusing so this is matching white space but this is using a
negative look ahead assertion in regex so what this is doing is it's matching
wh space up to but not including the last Whit space character why is this important um this
is pretty subtle I think so you see how the white space is always included at the beginning of the word so um space r
space u Etc suppose we have a lot of spaces here what's going to happen here is that
these spaces up to not including the last character will get caught by this
and what that will do is it will separate out the spaces up to but not including the last character so that the
last character can come here and join with the um space you and the reason
that's nice is because space you is the common token so if I didn't have these Extra Spaces here you would just have
space you and if I add tokens if I add spaces we still have a space view but
now we have all this extra white space so basically the GB to tokenizer really likes to have a space letters or numbers
um and it it preens these spaces and this is just something that it is consistent about so that's what that is
for and then finally we have all the the last fallback is um whites space characters uh so um that would be
just um if that doesn't get caught then this thing will catch any trailing
spaces and so on I wanted to show one more real world example here so if we have this string which is a piece of
python code and then we try to split it up then this is the kind of output we get so you'll notice that the list has
many elements here and that's because we are splitting up fairly often uh every time sort of a category
changes um so there will never be any merges Within These elements and um that's what you are
seeing here now you might think that in order to train the tokenizer uh open AI has used this to
split up text into chunks and then run just a BP algorithm within all the chunks but that is not exactly what
happened and the reason is the following notice that we have the spaces here uh
those Spaces end up being entire elements but these spaces never actually
end up being merged by by open Ai and the way you can tell is that if you copy paste the exact same chunk here into Tik
token U Tik tokenizer you see that all the spaces are kept independent and
they're all token 220 so I think opena at some point Point en Force some rule that these spaces
would never be merged and so um there's some additional rules on top of just
chunking and bpe that open ey is not uh clear about now the training code for
the gpt2 tokenizer was never released so all we have is uh the code that I've already shown you but this code here
that they've released is only the inference code for the tokens so this is not the training code you can't give it
a piece of text and training tokenizer this is just the inference code which Tak takes the merges that we have up
above and applies them to a new piece of text and so we don't know exactly how opening ey trained um train the
tokenizer but it wasn't as simple as chunk it up and BP it uh whatever it was

这段内容主要讨论了 **GPT-2** 的分词器是如何工作以及它如何通过正则表达式（regex）强制分割文本中的不同部分，以防止不合理的合并。以下是逐步的解释：

### 1. **GPT-2 分词器概述**：

在 GPT-2 的论文中，他们使用了字节对编码（BPE）算法对文本进行编码。通过该算法，文本被分解为单个字节对，然后这些字节对会根据频率进行合并，最终构建出一个词汇表。虽然 BPE 过程有效，但存在一个问题：当一些常见的词（如 "dog"）后面有标点符号时，BPE 会将词和标点符号一起合并成单一的 token（例如 "dog."），这会导致合并无意义的内容。为了解决这个问题，GPT-2 对 BPE 算法进行了修改，并强制执行了一些规则来确保某些字符（如标点符号）不会与单词合并。

### 2. **使用正则表达式强制分割文本**：

GPT-2 在编码文本时使用了复杂的正则表达式（regex）模式来强制分割文本。这些模式通过确保特定类别的字符（如字母、数字、标点符号等）不会合并在一起，来避免不必要的合并。

#### 主要操作：

* **分割字母和数字**：模式会确保字母与数字之间不会合并。例如，字符串 `"Hello World 123"` 中，`"Hello"` 和 `"World"` 之间会被分开，`"123"` 也会作为单独的部分被处理。
* **分割标点符号**：模式还会确保标点符号（如 `'`, `.`, `,` 等）不会和字母合并，例如 `"don't"` 会被分割成 `"don"` 和 `'`（分开处理）。
* **强制分割空格**：为了确保空格符号不会与后面的字母或标点合并，GPT-2 强制分割空格，使得每个空格都成为独立的 token。
* **正则表达式的结构**：这个正则表达式会逐步匹配并分割字符串，遇到字母、数字或标点符号时，它们会被识别为独立的部分。通过这种方式，文本中的每一部分都被处理为一个独立的“块”，而不是在整个文本中进行合并。

### 3. **合并与分割的影响**：

通过强制分割，GPT-2 可以确保文本中的标点符号和字母不会无意义地合并。例如，"dog!" 和 "dog." 不会被合并成一个 token，从而避免了模型错误地学习这些组合。GPT-2 通过这种方法，确保了更合理的分词策略。

### 4. **如何在代码中实现**：

GPT-2 使用了 `re.compile`（正则表达式编译）来创建这个强制分割的模式。在运行时，代码会根据这个模式将输入文本分割成多个部分，然后对每个部分独立进行处理。所有这些分割后的文本会被单独编码成 tokens，最后再将它们拼接起来。

### 5. **GPT-2 训练与推理**：

虽然 GPT-2 的训练代码没有完全公开，但可以推测，GPT-2 并不是简单地将文本切割后直接应用 BPE 进行编码。它还采用了更多的规则来确保合适的分割和合并。此外，GPT-2 分词器的推理代码是公开的，它会应用已经训练好的合并规则对新的文本进行编码，但不支持重新训练 tokenizer。

### 总结：

GPT-2 使用了一个复杂的正则表达式模式，在文本编码前将文本分割成多个部分，以确保某些字符（如字母、数字、标点符号）不会错误地被合并在一起。这种强制分割的策略帮助 GPT-2 更好地处理不同类型的文本，确保标点符号、空格、字母、数字等能合理地分开，从而提升了模型的表现和准确性。

这个正则表达式 `gpt2pat` 是用来匹配和分割文本的，它是 GPT-2 tokenizer 中使用的一个模式。通过正则表达式的不同部分，它能够强制按照特定的规则分割文本。让我们逐部分地解析这个正则表达式：

### 1. **`\p{L}+`**

* `\p{L}`：匹配任何字母字符，`\p{L}` 是 Unicode 字符类别（Letter），它包括了所有语言的字母（不论是拉丁字母、阿拉伯字母、汉字等）。
* `+`：表示匹配一个或多个字母。
* 这个部分匹配由字母组成的单词，允许不同语言的字母。

### 2. **`\p{N}+`**

* `\p{N}`：匹配任何数字字符（包括阿拉伯数字、罗马数字等）。
* `+`：表示匹配一个或多个数字。
* 这个部分匹配由数字组成的序列。

### 3. **`[^\s\p{L}\p{N}]+`**

* `[ ]`：表示字符集。
* `^\s\p{L}\p{N}`：`^` 在字符集内表示“非”，即排除空白字符（`\s`）、字母（`\p{L}`）和数字（`\p{N}`）。因此，这部分表示匹配不是字母、数字或空白字符的其他字符，通常是标点符号或特殊符号。
* `+`：表示匹配一个或多个。
* 这个部分用于匹配标点符号或其他不属于字母、数字或空白的字符。

### 4. **`'?`**

* `?`：表示前面的字符是可选的（即可以有一个空格，也可以没有）。
* 这个部分是匹配可选的空格字符，意味着它会匹配像 `"dog "`（字母后跟空格）或者 `"dog"`（没有空格）的情况。

### 5. **`\s+(?!\S)`**

* `\s+`：匹配一个或多个空白字符（空格、制表符、换行符等）。
* `(?!\S)`：这是一个“负前瞻”断言（negative lookahead assertion）。它表示，如果后面紧跟的字符是非空白字符（`\S`），则不匹配这个空格。换句话说，这个部分确保匹配到的空格是字符串中唯一的空白部分，不会匹配那些后面跟着字母或数字的空格。
* 这个部分用来确保空格被正确地识别为独立的 token，尤其是在它后面不跟其他字符时。

### 6. **`\s+`**

* 这个部分表示匹配一个或多个空格字符。
* 它可以匹配像 `"dog "` 这样的字符串末尾的空格或中间的多个空格。

### 总结：

正则表达式的目标是按照不同类型的字符将文本分割为独立的 token。具体来说：

* 它匹配常见的英文缩写（例如 `'s`、`'t` 等），
* 匹配字母和数字序列，
* 分离标点符号和其他非字母数字的字符，
* 强制分割空格，并确保空格不会与其他非空格字符合并。

通过这种方法，GPT-2 的 tokenizer 能够确保分词时不会错误地将不同类型的字符（如字母和标点符号）合并成一个 token，从而提高语言模型的准确性。


# tiktoken library intro, differences between GPT-2/GPT-4 regex

next I wanted to introduce you to the Tik token library from openai which is the official library for tokenization
from openai so this is Tik token bip install P to Tik token and then um you
can do the tokenization in inference this is again not training code this is only inference code for
tokenization um I wanted to show you how you would use it quite simple and running this just gives us the gpt2
tokens or the GPT 4 tokens so this is the tokenizer use for GPT 4 and so in
particular we see that the Whit space in gpt2 remains unmerged but in GPT 4 uh these Whit spaces merge as we also saw
in this one where here they're all unmerged but if we go down to GPT 4 uh
they become merged um now in the
gp4 uh tokenizer they changed the regular expression that they use to
Chunk Up text so the way to see this is that if you come to your the Tik token uh library and then you go to this file
Tik token X openi public this is where sort of like the definition of all these different tokenizers that openi
maintains is and so uh necessarily to do the inference they had to publish some of the details about the strings
so this is the string that we already saw for gpt2 it is slightly different but it is actually equivalent uh to what
we discussed here so this pattern that we discussed is equivalent to this pattern this one just executes a little
bit faster so here you see a little bit of a slightly different definition but otherwise it's the same we're going to
go into special tokens in a bit and then if you scroll down to CL 100k this is
the GPT 4 tokenizer you see that the pattern has changed um and this is kind
of like the main the major change in addition to a bunch of other special tokens which I'll go into in a bit again
now some I'm not going to actually go into the full detail of the pattern change because honestly this is my
numbing uh I would just advise that you pull out chat GPT and the regex documentation and just step through it
but really the major changes are number one you see this eye here that means
that the um case sensitivity this is case insensitive match and so the
comment that we saw earlier on oh we should have used re. uppercase uh basically we're now going to be matching
these apostrophe s apostrophe D apostrophe M Etc uh we're going to be
matching them both in lowercase and in uppercase so that's fixed there's a bunch of different like handling of the
whites space that I'm not going to go into the full details of and then one more thing here is you will notice that
when they match the numbers they only match one to three numbers so so they will never merge
numbers that are in low in more than three digits only up to three digits of
numbers will ever be merged and uh that's one change that they made as well
to prevent uh tokens that are very very long number sequences uh but again we don't really
know why they do any of this stuff uh because none of this is documented and uh it's just we just get the pattern so
um yeah it is what it is but those are some of the changes that gp4 has made and of course the vocabulary size went
from roughly 50k to roughly 100K the next thing I would like to do very briefly is to take you through the

这段话介绍了 OpenAI 的 `tiktoken` 库，这个库是官方用于进行 **tokenization**（分词）的工具，主要用于推理阶段（inference），而非训练阶段。让我们逐步解析这段内容：

### 1. **tiktoken 库介绍**

`tiktoken` 是 OpenAI 发布的一个库，用于进行推理时的 **tokenization**。推理时的分词过程使用了不同的 tokenizer（分词器），例如 GPT-2 和 GPT-4 对应不同的分词器。这个库非常简单，使用起来也非常方便，通过安装命令 `pip install tiktoken` 即可安装。

通过这个库，我们可以看到不同版本模型的 tokens，例如 GPT-2 和 GPT-4，它们的分词规则有所不同。你可以直接用这个库来进行推理操作，获取 GPT-2 或 GPT-4 的 token。

### 2. **GPT-2 和 GPT-4 分词的差异**

* **GPT-2 分词：** 在 GPT-2 的 tokenizer 中，空白字符（whitespace）不会被合并为一个 token。每个空格都会保持独立，作为一个单独的 token。

* **GPT-4 分词：** 在 GPT-4 的 tokenizer 中，空白字符会被合并为一个单独的 token。这意味着连续的空格（如“dog    cat”）会被合并成一个 token，减少了 token 的数量。

### 3. **tiktoken 库中的正则表达式变化**

GPT-2 和 GPT-4 使用了不同的正则表达式来进行分词。具体来说，`tiktoken` 库中有一段关于 GPT-2 和 GPT-4 的正则表达式定义文件，里面详细描述了它们如何处理输入文本：

* **GPT-2 的正则表达式：** 虽然与之前提到的正则表达式非常相似，但有所不同，主要是为了提高执行效率。

* **GPT-4 的正则表达式：** GPT-4 在正则表达式上做了较大的调整：

  * **大小写不敏感：** GPT-4 的正则表达式在匹配缩写（例如：`'s`、`'t`、`'re` 等）时，不再区分大小写，解决了 GPT-2 中对于大小写匹配的限制。
  * **数字匹配：** GPT-4 的正则表达式对数字进行了限制，只匹配最多三位数字。换句话说，GPT-4 不会将超过三位数的数字序列作为一个 token 来处理，从而避免了处理超长数字序列时可能出现的问题。

### 4. **词汇表大小**

GPT-2 的词汇表大小大约是 50,000 个 token，而 GPT-4 则增加到了大约 100,000 个 token。这意味着 GPT-4 可以处理更多的不同词汇和符号，从而提高了其在更广泛文本上的表现。

### 总结：

* **GPT-2 和 GPT-4 的 tokenization 区别**：GPT-2 主要将空格视为独立的 token，而 GPT-4 将空格合并。此外，GPT-4 在正则表达式上作了更细致的调整，增加了大小写不敏感的匹配、限制了数字长度等。

* **tiktoken 库**：提供了非常简单的接口来进行 tokenization 操作，但其背后的正则表达式处理非常复杂，尤其是 GPT-4 在正则表达式上的一些变化并没有完全公开文档，所以我们只能推测其背后的原因。

# GPT-2 encoder.py released by OpenAI walkthrough

gpt2 encoder dopy that openi has released uh this is the file that I
already mentioned to you briefly now this file is uh fairly short and should
be relatively understandable to you at this point um starting at the bottom
here they are loading two files encoder Json and vocab bpe and they do some
light processing on it and then they call this encoder object which is the tokenizer now if you'd like to inspect
these two files which together constitute their saved tokenizer then you can do that with a piece of code
like this um this is where you can download these two files and you can inspect them if you'd like and what you will find is
that this encoder as they call it in their code is exactly equivalent to our vocab so remember here where we have
this vocab object which allowed us us to decode very efficiently and basically it took us from the integer to the byes uh
for that integer so our vocab is exactly their encoder and then their vocab bpe
confusingly is actually are merges so their BP merges which is based on the
data inside vocab bpe ends up being equivalent to our merges so uh basically
they are saving and loading the two uh variables that for us are also critical
the merges variable and the vocab variable using just these two variables you can represent a tokenizer and you
can both do encoding and decoding once you've trained this tokenizer now the only thing that um is
actually slightly confusing inside what opening ey does here is that in addition to this encoder and a decoder they also
have something called a bite encoder and a bite decoder and this is actually unfortunately just
kind of a spirous implementation detail and isn't actually deep or interesting in any way so I'm going to skip the
discussion of it but what opening ey does here for reasons that I don't fully understand is that not only have they
this tokenizer which can encode and decode but they have a whole separate layer here in addition that is used serially with the tokenizer and so you
first do um bite encode and then encode and then you do decode and then bite
decode so that's the loop and they are just stacked serial on top of each other
and and it's not that interesting so I won't cover it and you can step through it if you'd like otherwise this file if
you ignore the bite encoder and the bite decoder will be algorithmically very familiar with you and the meat of it
here is the what they call bpe function and you should recognize this Loop here
which is very similar to our own y Loop where they're trying to identify the Byram uh a pair that they should be
merging next and then here just like we had they have a for Loop trying to merge this pair uh so they will go over all of
the sequence and they will merge the pair whenever they find it and they keep repeating that until they run out of
possible merges in the in the text so that's the meat of this file and uh there's an encode and a decode function
just like we have implemented it so long story short what I want you to take away at this point is that unfortunately it's
a little bit of a messy code that they have but algorithmically it is identical to what we've built up above and what
we've built up above if you understand it is algorithmically what is necessary to actually build a BP to organizer
train it and then both encode and decode the next topic I would like to turn to is that of special tokens so in addition

这段话讲解了 OpenAI 发布的 **GPT-2 encoder.py** 文件，并与我们之前讨论的 **Byte Pair Encoding (BPE)** 算法进行了对比。我们来逐步解读这段内容：

### 1. **文件结构和文件加载**

OpenAI 发布的 **GPT-2 encoder.py** 文件包含了用来进行编码和解码的主要逻辑。文件的核心操作是加载了两个文件：**encoder.json** 和 **vocab.bpe**，并在此基础上创建一个 **encoder**（编码器）对象。该对象即为 **tokenizer**（分词器）。

* **encoder.json**：保存了词汇表（vocab），将整数映射到字节序列（bytes）之间。
* **vocab.bpe**：保存了 **Byte Pair Encoding (BPE)** 合并操作的规则，它记录了哪些字节对应该合并成一个新token。

### 2. **encoder 和 vocab 的映射**

OpenAI 的 `encoder` 与我们之前讲解的 **vocab** 是相同的对象。它们的作用是将整数转换为字节，而 **vocab.bpe** 则类似于我们使用的 **merges**，记录了合并规则。

* **vocab** 对应的是整数到字节的映射。
* **merges** 对应的是合并的规则。

这两个文件（**encoder.json** 和 **vocab.bpe**）存储了分词器所需的基本信息，使用这两个文件就可以进行编码和解码。

### 3. **额外的 "byte encoder" 和 "byte decoder"**

OpenAI 的实现中，除了常规的 `encoder` 和 `decoder`，还有 **byte encoder** 和 **byte decoder**。但这部分实现没有实际的深度意义，它们只是在 `encoder` 和 `decoder` 的基础上额外加了一个字节层的处理。

* 他们的处理过程是先通过 **byte encoder** 和 **byte decoder**，再通过 `encoder` 和 `decoder`。
* 然而，这部分的代码没有太大的技术价值，所以这里我们不深入讨论。

### 4. **核心逻辑：BPE函数**

GPT-2 中的 **BPE** 函数与我们之前讨论的 **BPE** 算法非常相似。关键部分是：

* **BPE 合并过程**：通过一个循环，查找当前最常见的字节对，然后将这个字节对进行合并。这个过程不断重复，直到没有更多的字节对可以合并。
* **编码与解码函数**：类似我们之前实现的那样，GPT-2 也实现了 **encode** 和 **decode** 函数，用来将文本转换为 tokens 或将 tokens 转回文本。

### 5. **总结**

简而言之，OpenAI 的 **GPT-2 encoder.py** 文件中的代码实现和我们所讲的 **BPE** 算法逻辑是一样的。区别主要体现在代码实现上，并且 OpenAI 的实现有一些附加的处理层（如 **byte encoder** 和 **byte decoder**），但这并不影响算法的本质。

* **关键点**：如果你理解了我们之前讲的 **BPE** 算法，那么你就可以理解 **GPT-2 encoder.py** 中的编码和解码过程。
* **重点**：虽然代码可能有点杂乱，但其核心算法与我们所讨论的完全一致。

接下来，你可能需要了解 **GPT-2** 的特殊 token 处理，它们用于处理一些特定情况。


# special tokens, tiktoken handling of, GPT-2/GPT-4 differences

to tokens that are coming from you know raw bytes and the BP merges we can insert all kinds of tokens that we are
going to use to delimit different parts of the data or introduced to create a special structure of the token streams
so in uh if you look at this encoder object from open AIS gpd2 right here we
mentioned this is very similar to our vocab you'll notice that the length of this is
50257 and as I mentioned it's mapping uh and it's inverted from the mapping of our vocab our vocab goes from integer to
string and they go the other way around for no amazing reason um but the thing
to note here is that this the mapping table here is 50257 where does that number come from
where what are the tokens as I mentioned there are 256 raw bite token
tokens and then opena actually did 50,000 merges so those become the other tokens
but this would have been 50256 so what is the 57th token and
there is basically one special token and that one special token you can
see is called end of text so this is a special token and it's the very last
token and this token is used to delimit documents ments in the training set so
when we're creating the training data we have all these documents and we tokenize them and we get a stream of tokens those
tokens only range from Z to 50256 and then in between those
documents we put special end of text token and we insert that token in
between documents and we are using this as a signal to the language model that
the document has ended and what follows is going to be unrelated to the document previously that said the language model
has to learn this from data it it needs to learn that this token usually means that it should wipe its sort of memory
of what came before and what came before this token is not actually informative to what comes next but we are expecting
the language model to just like learn this but we're giving it the Special sort of the limiter of these documents
we can go here to Tech tokenizer and um this the gpt2 tokenizer uh our code that
we've been playing with before so we can add here right hello world world how are you and we're getting different tokens
but now you can see what if what happens if I put end of text you see how until I
finished it these are all different tokens end of text still set different tokens and now
when I finish it suddenly we get token 50256 and the reason this works is
because this didn't actually go through the bpe merges instead the code that
actually outposted tokens has special case instructions for handling special
tokens um we did not see these special instructions for handling special tokens in the encoder dopy it's absent there
but if you go to Tech token Library which is uh implemented in Rust you will find all kinds of special case handling
for these special tokens that you can register uh create adds to the vocabulary and then it looks for them
and it uh whenever it sees these special tokens like this it will actually come in and swap in that special token so
these things are outside of the typical algorithm of uh B PA en coding so these special tokens are used
pervasively uh not just in uh basically base language modeling of predicting the next token in the sequence but
especially when it gets to later to the fine tuning stage and all of the chat uh gbt sort of aspects of it uh because we
don't just want to Del limit documents we want to delimit entire conversations between an assistant and a user so if I
refresh this sck tokenizer page the default example that they have here is using not sort of base model encoders
but ftuned model uh sort of tokenizers um so for example using the GPT 3.5
turbo scheme these here are all special tokens I am start I end Etc uh this is
short for Imaginary mcore start by the way but you can see here that there's a
sort of start and end of every single message and there can be many other other tokens lots of tokens um in use to
delimit these conversations and kind of keep track of the flow of the messages here now we can go back to the Tik token
library and here when you scroll to the bottom they talk about how you can extend tick token and I can you can
create basically you can Fork uh the um CL 100K base tokenizers in gp4 and for
example you can extend it by adding more special tokens and these are totally up to you you can come up with any arbitrary tokens and add them with the
new ID afterwards and the tikken library will uh correctly swap them out uh when
it sees this in the strings now we can also go back to this
file which we've looked at previously and I mentioned that the gpt2 in Tik toen open
I.P we have the vocabulary we have the pattern for splitting and then here we are registering the single special token
in gpd2 which was the end of text token and we saw that it has this ID in GPT 4 when they defy this here you
see that the pattern has changed as we've discussed but also the special tokens have changed in this tokenizer so
we of course have the end of text just like in gpd2 but we also see three sorry
four additional tokens here Thim prefix middle and suffix what is fim fim is
short for fill in the middle and if you'd like to learn more about this idea it comes from this paper um and I'm not
going to go into detail in this video it's beyond this video and then there's one additional uh serve token here so
that's that encoding as well so it's very common basically to train a language model and then if you'd like uh
you can add special tokens now when you add special tokens you of course have to
um do some model surgery to the Transformer and all the parameters involved in that Transformer because you
are basically adding an integer and you want to make sure that for example your embedding Matrix for the vocabulary
tokens has to be extended by adding a row and typically this row would be initialized uh with small random numbers
or something like that because we need to have a vector that now stands for that token in addition to that you have
to go to the final layer of the Transformer and you have to make sure that that projection at the very end into the classifier uh is extended by
one as well so basically there's some model surgery involved that you have to couple with the tokenization changes if
you are going to add special tokens but this is a very common operation that people do especially if they'd like to
fine tune the model for example taking it from a base model to a chat model like chat
GPT okay so at this point you should have everything you need in order to build your own gp4 tokenizer now in the

这段文字讲解了 **特殊 token** 的处理、**TikToken** 库的使用以及 GPT-2 和 GPT-4 在 token 处理上的差异。接下来我将为你逐步解读：

### 1. **特殊 Token 的介绍**

在 GPT 模型中，除了常规的 **BPE（字节对编码）** 合并 token 外，OpenAI 还引入了一些 **特殊的 token** 来处理文档的边界、对话的开始和结束等结构化信息。最常见的特殊 token 是：

* **end of text**（文本结束）：它是 GPT-2 和 GPT-4 词汇表中的最后一个 token，通常用于表示一个文档的结束。在训练数据中，当文档结束时，我们会在文档之间插入该 token 来帮助模型识别文档的结束，并防止模型将一个文档的信息错误地带入下一个文档中。

### 2. **GPT-2 的特殊 Token**

* **token 50256**：在 GPT-2 中，**50256** 是 **end of text** 的 ID，表示文本的结束。在训练过程中，所有的文档 token 范围是 **0-50255**，而 **50256** 用来区分不同的文档。

### 3. **TikToken 库中的特殊 Token**

在 OpenAI 的 **TikToken** 库中，处理这些特殊 token 是通过特定的代码实现的。这些特殊 token 不会经过常规的 **BPE** 合并过程，而是直接插入到词汇表中，并在推理时由模型识别并替换。具体来说：

* **GPT-2** 的特殊 token 主要是 **end of text**，它会在文档之间插入，告知模型文档的结束。
* 在 **GPT-4** 中，除了 **end of text**，还引入了其他几种特殊 token，如：

  * **fim**（fill in the middle，填充中间）：用于填充上下文中的某些部分。
  * **prefix、middle、suffix**：用于表示文本的不同部分，帮助模型理解对话或文档的结构。

### 4. **GPT-2 和 GPT-4 在特殊 Token 上的差异**

* **GPT-2** 仅有 **end of text** 作为特殊 token，用于区分文档之间的边界。
* **GPT-4** 不仅保留了 **end of text**，还新增了其他特殊 token（如 **fim、prefix、middle、suffix**），用于更复杂的对话结构和填充任务。

### 5. **如何扩展 TikToken 库**

* **TikToken** 库允许用户根据自己的需求添加自定义的特殊 token。例如，用户可以添加新的 token 来扩展词汇表，只需要在词汇表中给这些 token 分配一个新的 ID，并确保模型的 **嵌入矩阵** 和 **最终分类器** 能处理这些新增的 token。

* 这种扩展通常伴随一些模型的修改，比如 **Transformer** 模型的最后一层投影需要重新调整，以适应新增的 token。

### 6. **模型修改（Model Surgery）**

* 当你向模型中添加特殊 token 时，你需要进行一些 **模型修改**，包括：

  1. **扩展词汇表的嵌入矩阵**：增加新的行来表示新的 token。
  2. **调整最后一层分类器**：确保最后一层能够处理新的 token。

* 这种修改通常是在 **微调** 阶段进行的，尤其是当你希望将模型从基础版本转变为具有对话功能的模型（如 **ChatGPT**）时。

### 7. **总结**

* GPT 模型使用 **特殊 token** 来标记文档边界、对话开始/结束等。
* GPT-2 和 GPT-4 在特殊 token 处理上有所不同，GPT-4 增加了更多的特殊 token 以支持更复杂的对话和填充任务。
* **TikToken** 库提供了灵活的方式来管理和扩展特殊 token，但如果你要添加特殊 token，需要对模型进行一些 **修改**（model surgery），如扩展嵌入矩阵和调整最终分类器。

通过这些操作，可以有效地使 GPT 模型适应更复杂的任务，比如对话生成和多轮交互。


# minbpe exercise time! write your own GPT-4 tokenizer

process of developing this lecture I've done that and I published the code under this repository
MBP so MBP looks like this right now as I'm recording but uh the MBP repository
will probably change quite a bit because I intend to continue working on it um in addition to the MBP repository I've
published the this uh exercise progression that you can follow so if you go to exercise. MD here uh this is
sort of me breaking up the task ahead of you into four steps that sort of uh
build up to what can be a gp4 tokenizer and so feel free to follow these steps exactly and follow a little bit of the
guidance that I've laid out here and anytime you feel stuck just reference the MBP repository here so either the
tests could be useful or the MBP repository itself I try to keep the code fairly clean and understandable and so
um feel free to reference it whenever um you get stuck uh in addition to that basically
once you write it you should be able to reproduce this behavior from Tech token so getting the gb4 tokenizer you can
take uh you can encode the string and you should get these tokens and then you can encode and decode the exact same
string to recover it and in addition to all that you should be able to implement your own train function uh which Tik
token Library does not provide it's it's again only inference code but you could write your own train MBP does it as well
and that will allow you to train your own token vocabularies so here are some of the code inside M be mean bpe uh shows the
token vocabularies that you might obtain so on the left uh here we have the GPT 4
merges uh so the first 256 are raw individual bytes and then here I am
visualizing the merges that gp4 performed during its training so the very first merge that gp4 did was merge
two spaces into a single token for you know two spaces and that is a token 256
and so this is the order in which things merged during gb4 training and this is the merge order that um we obtain in MBP
by training a tokenizer and in this case I trained it on a Wikipedia page of Taylor Swift uh not because I'm a Swifty
but because that is one of the longest um Wikipedia Pages apparently that's available but she is pretty cool and
um what was I going to say yeah so you can compare these two uh vocabularies
and so as an example um here GPT for merged I in to become in and we've done
the exact same thing on this token 259 here space t becomes space t and that
happened for us a little bit later as well so the difference here is again to my understanding only a difference of
the training set so as an example because I see a lot of white space I supect that gp4 probably had a lot of
python code in its training set I'm not sure uh for the tokenizer and uh here we see much less
of that of course in the Wikipedia page so roughly speaking they look the same and they look the same because they're
running the same algorithm and when you train your own you're probably going to get something similar depending on what
you train it on okay so we are now going to move on from tick token and the way that open AI tokenizes its strings and

这段文字介绍了 **minbpe** 练习和如何编写自己的 **GPT-4 Tokenizer**。具体内容如下：

### 1. **minbpe 练习介绍**

* **minbpe** 是一个练习项目，目的是帮助你实现自己的 **GPT-4 Tokenizer**。在这个项目中，作者已经发布了代码，并将其放在一个公共代码库中（叫做 `MBP`）。
* 这个练习是逐步进行的，作者将任务分成了四个步骤，每一步都逐渐接近于构建一个 **GPT-4 Tokenizer**。
* 你可以根据作者在 `exercise.md` 中的指导进行操作，遇到困难时可以参考 `MBP` 仓库中的测试或代码。

### 2. **目标：实现 GPT-4 Tokenizer**

* 目标是让你能够实现一个可以模仿 **GPT-4 Tokenizer** 的程序。具体要求包括：

  1. **编码和解码字符串**，并能够恢复原始文本。
  2. **实现自己的训练功能**，而不仅仅是推理功能（推理功能是 TikToken 库提供的）。你将学会如何训练自己的词汇表（vocabulary）。

### 3. **GPT-4 Tokenizer 的工作原理**

* 在训练过程中，**GPT-4** 的 tokenizer 会先从 256 个原始字节开始，然后通过 **BPE（字节对编码）** 逐步合并词对，形成更大的词汇单元。例如，GPT-4 的第一次合并操作是将两个空格合并成一个 token。
* 训练结束后，GPT-4 的词汇表包含了 50,257 个 token，其中包括了许多经过合并的字节对。这个过程是通过 **BPE** 算法实现的，作者提供了一个可视化的例子，展示了 GPT-4 合并操作的顺序。

### 4. **实际操作与对比**

* 作者举了一个实际的例子，说明了自己训练的 **minbpe** tokenizer 和 **GPT-4** tokenizer 的合并过程。作者在一个关于 Taylor Swift 的维基百科页面上训练了自己的 tokenizer。
* 通过对比 **GPT-4** 和自己训练的 tokenizer 结果，作者指出，尽管训练集不同，但两者的合并过程非常相似。差异主要体现在训练集的不同，**GPT-4** 可能包含了大量 Python 代码，而自己训练的 tokenizer 则使用了维基百科页面。

### 5. **总结**

* 这个练习的目的是帮助你理解和实现 **BPE** 编码，并能够训练自己的 tokenizer。
* 一旦完成，你将能够开发出一个与 **GPT-4** 类似的 tokenizer，并能够进行文本的编码与解码。


# sentencepiece library intro, used to train Llama 2 vocabulary

we're going to discuss one more very commonly used library for working with tokenization inlm
and that is sentence piece so sentence piece is very commonly used in language
models because unlike Tik token it can do both training and inference and is quite efficient at both it supports a
number of algorithms for training uh vocabularies but one of them is the B pair en coding algorithm that we've been
looking at so it supports it now sentence piece is used both by llama and
mistal series and many other models as well it is on GitHub under Google
sentence piece and the big difference with sentence piece and we're going to look at example
because this is kind of hard and subtle to explain is that they think different about the order of operations here so in
the case of Tik token we first take our code points in the string we encode them
using mutf to bytes and then we're merging bytes it's fairly straightforward for sentence piece um it
works directly on the level of the code points themselves so so it looks at whatever code points are available in
your training set and then it starts merging those code points and um the bpe
is running on the level of code points and if you happen to run out of code points so there are maybe some rare
uh code points that just don't come up too often and the Rarity is determined by this character coverage hyper parameter then these uh code points will
either get mapped to a special unknown token like ank or if you have the bite
foldback option turned on then that will take those rare Cod points it will encode them using utf8 and then the
individual bytes of that encoding will be translated into tokens and there are these special bite tokens that basically
get added to the vocabulary so it uses BP on on the code points and then it
falls back to bytes for rare Cod points um and so that's kind of like difference
personally I find the Tik token we significantly cleaner uh but it's kind of like a subtle but pretty major difference between the way they approach
tokenization let's work with with a concrete example because otherwise this is kind of hard to um to get your head
around so let's work with a concrete example this is how we can import sentence piece and then here we're going
to take I think I took like the description of sentence piece and I just created like a little toy data set it
really likes to have a file so I created a toy. txt file with this content now what's kind of a little bit
crazy about sentence piece is that there's a ton of options and configurations and the reason this is so
is because sentence piece has been around I think for a while and it really tries to handle a large diversity of things and um because it's been around I
think it has quite a bit of accumulated historical baggage uh as well and so in
particular there's like a ton of configuration arguments this is not even all of it you can go to here to see all
the training options um and uh there's also quite useful documentation when you look at
the raw Proto buff uh that is used to represent the trainer spec and so on um
many of these options are irrelevant to us so maybe to point out one example Das Das shrinking Factor uh this shrinking
factor is not used in the B pair en coding algorithm so this is just an argument that is irrelevant to us um it
applies to a different training algorithm now what I tried to do here is
I tried to set up sentence piece in a way that is very very similar as far as I can tell to maybe identical hopefully
to the way that llama 2 was strained so the way they trained their own um their
own tokenizer and the way I did this was basically you can take the tokenizer model file that meta released and you
can um open it using the Proto protuff uh sort of file that you can generate
and then you can inspect all the options and I tried to copy over all the options that looked relevant so here we set up
the input it's raw text in this file here's going to be the output so it's going to be for talk 400. model and
vocab we're saying that we're going to use the BP algorithm and we want to Bap size of
400 then there's a ton of configurations here
for um for basically pre-processing and normalization rules as they're called
normalization used to be very prevalent I would say before llms in natural language processing so in machine
translation and uh text classification and so on you want to normalize and simplify the text and you want to turn
it all lowercase and you want to remove all double whites space Etc and in language models we prefer not to
do any of it or at least that is my preference as a deep learning person you want to not touch your data you want to
keep the raw data as much as possible um in a raw form so you're basically trying to turn
off a lot of this if you can the other thing that sentence piece does is that it has this concept of sentences so
sentence piece it's back it's kind of like was developed I think early in the days where there was um an idea that
they you're training a tokenizer on a bunch of independent sentences so it has a lot of like how many sentences you're
going to train on what is the maximum sentence length um shuffling sentences and so for it
sentences are kind of like the individual training examples but again in the context of llms I find that this is like a very spous and weird
distinction like sentences are just like don't touch the raw data sentences
happen to exist but in raw data sets there are a lot of like inet like what exactly is a sentence what isn't a
sentence um and so I think like it's really hard to Define what an actual sentence is if you really like dig into
it and there could be different concepts of it in different languages or something like that so why even
introduce the concept it it doesn't honestly make sense to me I would just prefer to treat a file as a giant uh
stream of bytes it has a lot of treatment around rare word characters and when I say word
I mean code points we're going to come back to this in a second and it has a lot of other rules for um basically
splitting digits splitting white space and numbers and how you deal with that so these are some kind of like merge
rules so I think this is a little bit equivalent to tick token using the regular expression to split up
categories there's like kind of equivalence of it if you squint T it in sentence piece where you can also for
example split up split up the digits uh and uh so
on there's a few more things here that I'll come back to in a bit and then there are some special tokens that you can indicate and it hardcodes the UN
token the beginning of sentence end of sentence and a pad token um and the UN
token must exist for my understanding and then some some things so we can
train and when when I press train it's going to create this file talk 400.
model and talk 400. wab I can then load the model file and I can inspect the
vocabulary off it and so we trained vocab size 400 on this text here and
these are the individual pieces the individual tokens that sentence piece will create so in the beginning we see
that we have the an token uh with the ID zero then we have the beginning of
sequence end of sequence one and two and then we said that the pad ID is negative
1 so we chose not to use it so there's no pad ID here then these are individual bite
tokens so here we saw that bite fallback in llama was turned on so it's true so
what follows are going to be the 256 bite tokens and these are their
IDs and then at the bottom after the bite tokens come the
merges and these are the parent nodes in the merges so we're not seeing the children we're just seeing the parents
and their ID and then after the merges comes eventually the individual
tokens and their IDs and so these are the individual tokens so these are the individual code Point tokens if you will
and they come at the end so that is the ordering with which sentence piece sort of like represents its vocabularies it
starts with special tokens then the bike tokens then the merge tokens and then the individual codo tokens and all these
raw codepoint to tokens are the ones that it encountered in the training set so those individual code points are
all the the entire set of code points that occurred here so those all get put in there and
then those that are extremely rare as determined by character coverage so if a code Point occurred only a single time
out of like a million um sentences or something like that then it would be ignored and it would not be added to our
uh vocabulary once we have a vocabulary we can encode into IDs and we can um sort
of get a list and then here I am also decoding the indiv idual tokens back into little
pieces as they call it so let's take a look at what happened here hello space
on so these are the token IDs we got back and when we look here uh a few
things sort of uh jump to mind number one take a look at these characters the
Korean characters of course were not part of the training set so sentence piece is encountering code points that
it has not seen during training time and those code points do not have a token associated with them so suddenly these
are un tokens unknown tokens but because bite fall back as true instead sentence
piece falls back to bytes and so it takes this it encodes it with utf8 and
then it uses these tokens to represent uh those bytes and that's what we are
getting sort of here this is the utf8 uh encoding and in this shifted by three uh
because of these um special tokens here that have IDs earlier on so that's what
happened here now one more thing that um well first before I go on with respect
to the bitef back let me remove bite foldback if this is false what's going
to happen let's retrain so the first thing that happened is all the bite tokens disappeared right
and now we just have the merges and we have a lot more merges now because we have a lot more space because we're not taking up space in the wab size uh with
all the bytes and now if we encode this we get a zero so this entire string
here suddenly there's no bitef back so this is unknown and unknown is an and so
this is zero because the an token is token zero and you have to keep in mind
that this would feed into your uh language model so what is a language model supposed to do when all kinds of different things that are unrecognized
because they're rare just end up mapping into Unk it's not exactly the property that you want so that's why I think
llama correctly uh used by fallback true uh because we definitely want to feed
these um unknown or rare code points into the model and some uh some manner the next thing I want to show you is the
following notice here when we are decoding all the individual tokens you see how spaces uh space here ends up
being this um bold underline I'm not 100% sure by the way why sentence piece
switches whites space into these bold underscore characters maybe it's for visualization I'm not 100% sure why that
happens uh but notice this why do we have an extra space in the front of
hello um what where is this coming from well it's coming from this option
here um add dummy prefix is true and when you
go to the documentation add D whites space at the beginning of text in order to treat World in world and hello world in the
exact same way so what this is trying to do is the following if we go back to our tick
tokenizer world as uh token by itself has a different ID than space world so
we have this is 1917 but this is 14 Etc so these are two different tokens for
the language model and the language model has to learn from data that they are actually kind of like a very similar concept so to the language model in the
Tik token World um basically words in the beginning of sentences and words in the middle of sentences actually look
completely different um and it has to learned that they are roughly the same
so this add dami prefix is trying to fight that a little bit and the way that works is that it basically
uh adds a dummy prefix so for as a as a
part of pre-processing it will take the string and it will add a space it will do this and that's done in an effort to
make this world and that world the same they will both be space world so that's
one other kind of pre-processing option that is turned on and llama 2 also uh uses this option and that's I think
everything that I want to say for my preview of sentence piece and how it is different um maybe here what I've done
is I just uh put in the Raw protocol buffer representation basically of the
tokenizer the too trained so feel free to sort of Step through this and if you would like uh your tokenization to look
identical to that of the meta uh llama 2 then you would be copy pasting these settings as I tried to do up above and
uh yeah that's I think that's it for this section I think my summary for sentence piece from all of this is
number one I think that there's a lot of historical baggage in sentence piece a lot of Concepts that I think are slightly confusing and I think
potentially um contain foot guns like this concept of a sentence and it's maximum length and stuff like that um
otherwise it is fairly commonly used in the industry um because it is efficient
and can do both training and inference uh it has a few quirks like for example un token must exist and the way the bite
fallbacks are done and so on I don't find particularly elegant and unfortunately I have to say it's not very well documented so it took me a lot
of time working with this myself um and just visualizing things and trying to really understand what is happening here
because uh the documentation unfortunately is in my opion not not super amazing but it is a very nice repo
that is available to you if you'd like to train your own tokenizer right now okay let me now switch gears again as we're starting to slowly wrap up here I

这段内容讲解了 **SentencePiece** 库的基本概念及其与 **TikToken** 库的主要区别，并介绍了 SentencePiece 在训练 **Llama 2** 词汇表中的应用。以下是详细的中文解释：

### 1. **SentencePiece 库简介**

* **SentencePiece** 是一个非常常用的库，用于训练和推理中的 **tokenization**（分词）。与 **TikToken** 库不同，SentencePiece 既可以进行训练，也支持推理，并且在两者上都很高效。
* SentencePiece 支持多种算法来训练词汇表，其中包括我们之前讨论过的 **BPE（字节对编码）** 算法，它支持在字符层面进行分词，而 **TikToken** 是在字节层面操作的。
* SentencePiece 在 **Llama 2** 和 **Mistral** 等模型中有广泛应用，并且是由 Google 开发并开源的。

### 2. **与 TikToken 的主要区别**

* **TikToken** 库先将文本编码为字节，然后通过字节对合并（BPE）来分词。而 **SentencePiece** 库直接在 **code points（代码点）** 上运行，即它在训练集中的字符上直接进行 BPE 操作。
* 另一个显著的不同是，SentencePiece 还支持处理一些稀有字符。如果某个字符在训练集中出现的频率太低，它会被映射到一个特殊的 **UNK（未知）** token，或者如果启用了 **byte fallback** 选项，它会将这些稀有字符通过 **UTF-8 编码** 转换为字节，并将这些字节作为 token 添加到词汇表中。

### 3. **训练模型的过程**

* **SentencePiece** 提供了非常多的配置选项，这些选项使它能够处理多种不同的情况。训练时，用户可以设定 **vocab size**（词汇表大小）、选择合适的算法等。文章中提到，它的配置比较复杂，因为这个库已经有了一些历史积累和沉淀。
* 通过分析 **Meta** 发布的 **Llama 2** 训练的 tokenizer 模型文件，作者尝试配置 SentencePiece，使得其训练过程和 **Llama 2** 的 tokenizer 训练过程类似。主要的配置选项包括输入文件、输出模型文件、词汇表大小、BPE算法等。

### 4. **SentencePiece 的训练过程和词汇表示**

* 训练完成后，SentencePiece 会生成一个包含特殊 token 的词汇表。特殊 token 包括 **\[UNK]**（未知 token）、**\[BOS]**（句子开始）、**\[EOS]**（句子结束）等。
* 在训练过程中，SentencePiece 还会生成一些字节级别的 token，这些是通过 **byte fallback** 机制处理的稀有字符。
* 接下来是 **merge tokens**（合并后的词）和 **codepoint tokens**（原始字符）。这些原始字符是训练集中出现的所有字符。

### 5. **编码与解码过程**

* 训练后，你可以用 **SentencePiece** 编码输入文本，生成 token ID，然后解码回原始文本。编码过程中，任何训练集外的字符会被映射为 **\[UNK]**，而启用 **byte fallback** 时，稀有字符会被转换为字节 token。
* 例如，文本中的 **韩文字符** 就会被识别为 **UNK** token，除非启用了 **byte fallback**，此时它会将这些字符的 UTF-8 编码作为 token 进行处理。

### 6. **预处理选项**

* **SentencePiece** 中有一些预处理选项，例如 **add dummy prefix**，即在每个句子前添加一个空格。这是为了确保模型能够正确处理句首和句中词汇。这个选项在 **Llama 2** 中也有使用。

### 7. **总结**

* **SentencePiece** 是一个功能强大的库，能够高效地处理分词、训练和推理，但它有一些历史遗留的复杂配置选项，可能让新手用户感到困惑。特别是关于 "句子" 和 "最大句长" 等概念，并不是特别直观，且文档质量有待提高。
* 然而，它依然被广泛应用于业界，尤其是在训练像 **Llama 2** 这样的语言模型时，它是非常重要的工具。

总的来说，**SentencePiece** 与 **TikToken** 有很多相似之处，但在实现细节和配置选项上存在差异。它的灵活性和高效性使其在处理不同类型的文本和稀有字符时表现得很强大。


# how to set vocabulary set? revisiting gpt.py transformer

want to revisit this issue in a bit more detail of how we should set the vocap size and what are some of the considerations around it so for this I'd
like to go back to the model architecture that we developed in the last video when we built the GPT from
scratch so this here was uh the file that we built in the previous video and we defined the Transformer model and and
let's specifically look at Bap size and where it appears in this file so here we Define the voap size uh at this time it
was 65 or something like that extremely small number so this will grow much larger you'll see that Bap size doesn't
come up too much in most of these layers the only place that it comes up to is in exactly these two places here so when we
Define the language model there's the token embedding table which is this two-dimensional array where the vocap
size is basically the number of rows and uh each vocabulary element each token
has a vector that we're going to train using back propagation that Vector is of size and embed which is number of
channels in the Transformer and basically as voap size increases this embedding table as I mentioned earlier
is going to also grow we're going to be adding rows in addition to that at the end of the Transformer there's this LM
head layer which is a linear layer and you'll notice that that layer is used at the very end to produce the logits uh
which become the probabilities for the next token in sequence and so intuitively we're trying to produce a
probability for every single token that might come next at every point in time of that Transformer and if we have more
and more tokens we need to produce more and more probabilities so every single token is going to introduce an
additional dot product that we have to do here in this linear layer for this final layer in a
Transformer so why can't vocap size be infinite why can't we grow to Infinity
well number one your token embedding table is going to grow uh your linear
layer is going to grow so we're going to be doing a lot more computation here because this LM head layer will become more computational expensive number two
because we have more parameters we could be worried that we are going to be under trining some of these
parameters so intuitively if you have a very large vocabulary size say we have a million uh tokens then every one of
these tokens is going to come up more and more rarely in the training data because there's a lot more other tokens
all over the place and so we're going to be seeing fewer and fewer examples uh for each individual token and you might
be worried that basically the vectors associated with every token will be undertrained as a result because they just don't come up too often and they
don't participate in the forward backward pass in addition to that as your vocab size grows you're going to start shrinking your sequences a lot
right and that's really nice because that means that we're going to be attending to more and more text so that's nice but also you might be
worrying that two large of chunks are being squished into single tokens and so the model just doesn't have as much of
time to think per sort of um some number of characters in the text or you can
think about it that way right so basically we're squishing too much information into a single token and then the forward pass of the Transformer is
not enough to actually process that information appropriately and so these are some of the considerations you're thinking about when you're designing the
vocab size as I mentioned this is mostly an empirical hyperparameter and it seems like in state-of-the-art architectures
today this is usually in the high 10,000 or somewhere around 100,000 today and
the next consideration I want to briefly talk about is what if we want to take a pre-trained model and we want to extend
the vocap size and this is done fairly commonly actually so for example when you're doing fine-tuning for cha GPT um
a lot more new special tokens get introduced on top of the base model to maintain the metadata and all the
structure of conversation objects between a user and an assistant so that takes a lot of special tokens you might
also try to throw in more special tokens for example for using the browser or any other tool and so it's very tempting to
add a lot of tokens for all kinds of special functionality so if you want to be adding a token that's totally
possible Right all we have to do is we have to resize this embedding so we have to add rows we would initialize these uh
parameters from scratch to be small random numbers and then we have to extend the weight inside this linear uh
so we have to start making dot products um with the associated parameters as well to basically calculate the
probabilities for these new tokens so both of these are just a resizing operation it's a very mild
model surgery and can be done fairly easily and it's quite common that basically you would freeze the base model you introduce these new parameters
and then you only train these new parameters to introduce new tokens into the architecture um and so you can
freeze arbitrary parts of it or you can train arbitrary parts of it and that's totally up to you but basically minor
surgery required if you'd like to introduce new tokens and finally I'd like to mention that actually there's an

这段内容主要讨论了 **GPT** 模型中 **vocab size**（词汇表大小）设置的考虑因素，并讲解了如何进行模型扩展，尤其是当我们希望在预训练模型的基础上添加新的特殊 token 时。

### 1. **vocab size 的设置**

* **vocab size** 是决定词汇表大小的超参数，通常在训练 **Transformer** 模型时，它会影响到嵌入层（embedding layer）和最后的 **LM head** 层的大小。

  * 在 **Transformer** 模型中，**vocab size** 直接决定了 **token embedding table** 的大小，这个表是一个二维数组，其中每一行代表一个 token，并且每个 token 对应一个向量，这些向量会随着训练过程通过反向传播进行调整。随着词汇表大小的增加，嵌入层的大小也会增大。

  * 在模型的最后，**LM head** 层是一个线性层，用于计算每个 token 出现的概率。随着 **vocab size** 的增大，线性层的计算量也会增加，因为每个 token 都需要计算一个概率。

### 2. **vocab size 的限制**

* **vocab size** 不可能是无限大的，原因有几个：

  1. **计算量增加**：随着词汇表的增大，**token embedding table** 和 **LM head** 层的大小都会增加，这意味着计算量会急剧增加，尤其是在 **LM head** 层，它需要为每个 token 计算一个 **dot product**，随着 token 数量增多，计算的复杂度会变得非常高。
  2. **训练数据稀疏化**：随着词汇表大小的增加，每个 token 在训练数据中出现的频率会下降，因为 token 数量变多，导致每个 token 的训练样本变少，可能会造成某些 token 对应的向量无法充分训练。
  3. **信息过度压缩**：如果词汇表过大，相当于把更多的信息压缩到更少的 token 中。这虽然能让模型处理更长的文本，但可能导致某些信息无法在足够的 **forward pass** 中得到充分处理。

  总之，**vocab size** 是一个经验性的超参数，当前的状态-of-the-art 架构的词汇表大小通常在 **1万** 到 **10万** 之间。

### 3. **如何扩展预训练模型的 vocab size**

* 在很多实际应用中，我们需要在 **预训练模型** 上进行 **fine-tuning**，并且可能需要扩展词汇表来加入新的特殊 token。

  * 比如，在 **ChatGPT** 的 fine-tuning 过程中，通常需要加入一些特殊 token 来保持用户与助手之间的对话结构。这些特殊 token 用于标记消息的开始和结束、表示不同类型的操作（如浏览器、工具使用等），因此需要在原始模型的基础上加入更多的 token。

* **如何增加新的 token：**

  1. **修改嵌入层（Embedding Layer）**：如果要添加新的 token，需要扩展 **token embedding table**，即为每个新 token 添加新的行。新的嵌入向量会被初始化为小的随机数。
  2. **扩展线性层（LM Head）**：在 **LM head** 层的线性变换中，也需要对权重进行扩展，增加对新 token 的计算能力。

  这些操作本质上是 **模型小修小补**，不会改变整体的架构，但需要扩展部分参数。通常，**基础模型** 可以被冻结，只训练新增的参数来适应新 token。

### 4. **总结**

* **vocab size** 是决定模型计算复杂度和学习能力的重要超参数，需要根据任务的需要和计算资源来权衡。选择合适的词汇表大小可以确保模型的有效性和高效性。
* 扩展预训练模型的词汇表是一个常见的操作，尤其是在引入新的特殊 token 时，这通常需要进行一些简单的模型调整，如扩展嵌入层和线性层，但整体架构不会发生大变化。这种 **小修小补** 操作非常常见且高效，尤其是在 **fine-tuning** 阶段。

这样，我们就可以根据需要调整和扩展模型的词汇表大小，保持模型的灵活性和适应性。


# training new tokens, example of prompt compression

entire design space of applications in terms of introducing new tokens into a vocabulary that go Way Beyond just
adding special tokens and special new functionality so just to give you a sense of the design space but this could be an entire video just by itself uh
this is a paper on learning to compress prompts with what they called uh gist tokens and the rough idea is suppose
that you're using language models in a setting that requires very long prompts while these long prompts just slow
everything down because you have to encode them and then you have to use them and then you're tending over them and it's just um you know heavy to have
very large prompts so instead what they do here in this paper is they introduce
new tokens and um imagine basically having a few new tokens you put them in
a sequence and then you train the model by distillation so you are keeping the
entire model Frozen and you're only training the representations of the new tokens their embeddings and you're
optimizing over the new tokens such that the behavior of the language model is identical uh to the model that has a
very long prompt that works for you and so it's a compression technique of compressing that very long prompt into
those few new gist tokens and so you can train this and then at test time you can discard your old prompt and just swap in
those tokens and they sort of like uh stand in for that very long prompt and have an almost identical performance and
so this is one um technique and a class of parameter efficient fine-tuning techniques where most of the model is
basically fixed and there's no training of the model weights there's no training of Laura or anything like that of new
parameters the the parameters that you're training are now just the uh token embeddings so that's just one
example but this could again be like an entire video but just to give you a sense that there's a whole design space here that is potentially worth exploring
in the future the next thing I want to briefly address is that I think recently there's a lot of momentum in how you

这段内容介绍了一种通过引入新 **token** 来压缩长 **prompt**（提示）的方法，主要是通过 **gist tokens** 来实现 **prompt compression**（提示压缩）。

### 1. **引入新 token 来压缩 prompt**

* 当使用语言模型时，特别是在需要长提示的场景下，长提示会导致计算变慢，因为需要对它们进行编码、使用并传递给模型，这会增加计算的负担。
* 为了解决这个问题，一些研究提出了一种创新的方法：**将长提示压缩成几个新 token**，称为 **gist tokens**（概括性 token）。这种方法的核心思想是：

  1. 用几个新的 **gist tokens** 来替代原本的长提示。
  2. 在训练时，将整个模型冻结，仅训练这几个 **gist token** 的 **embedding**（嵌入表示），并通过 **distillation**（蒸馏）优化这些新 token 的表现。
  3. 训练的目标是使得新 token 的行为与原本长提示的效果一致。这样，模型就能够通过少量的 **gist tokens** 来表示一个复杂的长提示，而不会丧失性能。

### 2. **如何工作**

* **训练过程**：在训练过程中，**gist tokens** 的表示被优化，使它们能够捕捉到长提示的主要信息。这些 **gist tokens** 是通过 **token embedding** 来学习的。
* **测试过程**：在测试时，你可以丢弃原本的长提示，而只使用这几个 **gist tokens** 来代替。经过训练后，这几个新 token 会在没有原始长提示的情况下，保持几乎相同的性能。

### 3. **参数高效的微调（Parameter-efficient fine-tuning）**

* 这种方法是一种 **参数高效的微调** 技术。大部分的模型参数在这种微调中保持不变，只有新 token 的嵌入向量（embedding）会被训练。
* **没有训练模型权重**，也没有训练其他模型的层，只训练新 token 的表示。通过这种方式，可以在不对整个模型进行修改的情况下，使用新的 **gist tokens** 实现提示压缩。

### 4. **总结**

* **Prompt compression** 是通过引入新的 **gist tokens** 来实现的，目的是用少量的 token 来压缩和表示原本很长的提示，从而提高模型的效率。
* 这种方法属于一种 **高效的微调** 技术，它避免了对模型其他部分的训练，仅训练新 token 的嵌入向量。

这种技术展示了如何在 **语言模型** 中引入新的 **token** 来解决特定问题，而不必修改或训练整个模型。


# multimodal [image, video, audio] tokenization with vector quantization

actually could construct Transformers that can simultaneously process not just text as the input modality but a lot of
other modalities so be it images videos audio Etc and how do you feed in all
these modalities and potentially predict these modalities from a Transformer uh do you have to change the architecture
in some fundamental way and I think what a lot of people are starting to converge towards is that you're not changing the architecture you stick with the
Transformer you just kind of tokenize your input domains and then call the day and pretend it's just text tokens and
just do everything else identical in an identical manner so here for example there was a early paper that has nice
graphic for how you can take an image and you can chunc at it into integers um and these sometimes uh so
these will basically become the tokens of images as an example and uh these tokens can be uh hard tokens where you
force them to be integers they can also be soft tokens where you uh sort of don't require uh these to be discrete
but you do Force these representations to go through bottlenecks like in Auto encoders uh also in this paper that came
out from open a SORA which I think really um uh blew the mind of many people and inspired a lot of people in
terms of what's possible they have a Graphic here and they talk briefly about how llms have text tokens Sora has
visual patches so again they came up with a way to chunc a videos into basically tokens when they own
vocabularies and then you can either process discrete tokens say with autog regressive models or even soft tokens
with diffusion models and uh all of that is sort of uh being actively worked on
designed on and is beyond the scope of this video but just something I wanted to mention briefly okay now that we have come quite deep into the tokenization

**多模态（Multimodal）输入处理和向量量化（Vector Quantization）** 是一种可以让Transformer模型同时处理文本、图像、视频、音频等不同类型数据的技术。它的核心思想是通过适当的 **tokenization（标记化）** 来将这些不同模态的数据转换为可以由Transformer处理的标准格式，然后用类似处理文本的方式进行处理。

### 1. **处理多模态输入**

在传统的Transformer模型中，我们通常处理的是文本数据。为了处理多模态数据（如图像、视频、音频等），有些方法建议不改变Transformer模型本身的架构，而是通过 **标记化（tokenization）** 将不同模态的数据转换为标准的“令牌”，然后将它们作为输入传递给模型进行处理。

* **文本**：文本数据通过标准的标记化方式，将单词或子词（Subword）转化为tokens（令牌）。
* **图像**：图像被划分成小块（称为“patches”），这些图像块被编码成一个个token。每个图像块代表图像的一部分，它们可以通过传统的编码方式，或者通过更复杂的编码方法如\*\*向量量化（Vector Quantization，VQ）\*\*来表示。
* **视频**：视频通过将其划分为图像帧，并通过类似图像处理的方式进行tokenization，将视频流转化为一系列的tokens。
* **音频**：音频信号通常经过短时傅里叶变换（STFT）等处理，转化为频谱图等形式，再进行tokenization。

### 2. **向量量化（Vector Quantization）**

向量量化是一种将连续数据转化为离散符号的技术，它被广泛应用于图像、视频等数据的标记化。其基本思想是将输入数据通过编码器映射到一个固定大小的字典中，每个输入被表示为字典中一个离散的“符号”。

#### 在图像和视频处理中的应用：

* **图像**：通过向量量化，可以将图像的每个小块（patch）表示为一个离散的向量。每个小块的特征向量被映射到一个离散的token，这个token可以作为Transformer模型的输入。这样，图像就被转换为一系列的离散tokens，类似文本中的词或子词。

* **视频**：视频处理的过程与图像类似。视频帧首先被分割成多个小块（patches），然后每个小块的特征向量通过向量量化被映射到一个离散的token，从而生成一个视频的token序列。

### 3. **软token与硬token**

在多模态处理的tokenization过程中，**硬token**和**软token**是两种常见的表示方式：

* **硬token**：这些token是离散的，通常是整数，表示每个图像块或视频帧的具体类别或特征。这种方法类似于传统的标记化方法，例如将每个单词映射为一个整数。

* **软token**：这些token不一定是离散的整数，而是通过模型学习得到的连续值。软token通过一定的瓶颈（例如在自动编码器中）进行压缩，得到较为紧凑的表示。

### 4. **SORA与视觉补丁（Visual Patches）**

在OpenAI的SORA模型中，图像和视频数据被转化为“视觉补丁”（Visual Patches），每个补丁类似于图像的一个小块（patch）。这些视觉补丁可以通过**离散token**或者**软token**的方式表示。SORA模型表明，视频也可以通过与图像类似的方式被转化为tokens进行处理。

### 5. **Transformer的应用**

处理多模态输入的核心是将所有不同模态的数据转化为标准的tokens之后，输入Transformer进行处理。Transformer本身并不需要做太多的修改，重点是如何设计tokenization方式来将不同的数据转化为统一的格式。一旦数据被tokenized，Transformer的处理流程就与文本数据处理是一样的，可以通过自回归模型（auto-regressive models）进行预测，或者通过扩散模型（diffusion models）等其他模型进行生成。

### 总结

* **多模态tokenization** 通过将不同类型的数据（如文本、图像、视频、音频）转化为tokens，使得这些数据可以被Transformer处理。
* **向量量化**是将连续的输入数据（如图像块、音频帧）转化为离散的符号（token）的技术。
* 使用**硬token**或**软token**来表示不同模态的数据，其中硬token是离散的，而软token是通过模型学习得到的连续表示。
* **SORA模型**和类似的技术正在积极探索如何将视频、图像、音频等模态与文本处理结合，推动多模态学习的发展。

通过这些方法，Transformer可以同时处理多种模态的数据，并且不需要对模型架构做出根本性的改变。


# revisiting and explaining the quirks of LLM tokenization

algorithm and we understand a lot more about how it works let's loop back around to the beginning of this video
and go through some of these bullet points and really see why they happen so first of all why can't my llm spell
words very well or do other spell related tasks so fundamentally this is because
as we saw these characters are chunked up into tokens and some of these tokens
are actually fairly long so as an example I went to the gp4 vocabulary and I looked at uh one of the longer tokens
so that default style turns out to be a single individual token so that's a lot of characters for a single token so my
suspicion is that there's just too much crammed into this single token and my suspicion was that the model should not
be very good at tasks related to spelling of this uh single token so I
asked how many letters L are there in the word default style and of course my
prompt is intentionally done that way and you see how default style will be a single token so this is what the model
sees so my suspicion is that it wouldn't be very good at this and indeed it is not it doesn't actually know how many
L's are in there it thinks there are three and actually there are four if I'm not getting this wrong myself so that
didn't go extremely well let's look look at another kind of uh character level task so for example here I asked uh gp4
to reverse the string default style and they tried to use a code interpreter and I stopped it and I said just do it just
try it and uh it gave me jumble so it doesn't actually really know how to
reverse this string going from right to left uh so it gave a wrong result so
again like working with this working hypothesis that maybe this is due to the tokenization I tried a different
approach I said okay let's reverse the exact same string but take the following approach step one just print out every
single character separated by spaces and then as a step two reverse that list and it again Tred to use a tool but when I
stopped it it uh first uh produced all the characters and that was actually correct and then It reversed them and
that was correct once it had this so somehow it can't reverse it directly but when you go just first uh you know
listing it out in order it can do that somehow and then it can once it's uh broken up this way this becomes all
these individual characters and so now this is much easier for it to see these individual tokens and reverse them and
print them out so that is kind of interesting so let's continue now why
are llms worse at uh non-english langu and I briefly covered this already but
basically um it's not only that the language model sees less non-english data during training of the model
parameters but also the tokenizer is not um is not sufficiently trained on
non-english data and so here for example hello how are you is five tokens and its
translation is 15 tokens so this is a three times blow up and so for example
anang is uh just hello basically in Korean and that end up being three tokens I'm actually kind of surprised by
that because that is a very common phrase there just the typical greeting of like hello and that ends up being
three tokens whereas our hello is a single token and so basically everything is a lot more bloated and diffuse and
this is I think partly the reason that the model Works worse on other languages uh coming back why is LM bad
at simple arithmetic um that has to do with the tokenization of numbers and so
um you'll notice that for example addition is very sort of like uh there's an algorithm that is
like character level for doing addition so for example here we would first add the ones and then the tens and then the
hundreds you have to refer to specific parts of these digits but uh these
numbers are represented completely arbitrarily based on whatever happened to merge or not merge during the tokenization process there's an entire
blog post about this that I think is quite good integer tokenization is insane and this person basically
systematically explores the tokenization of numbers in I believe this is gpt2 and
so they notice that for example for the for um four-digit numbers you can take a
look at whether it is uh a single token or whether it is two tokens that is a 1 three or a 2 two or a 31 combination and
so all the different numbers are all the different combinations and you can imagine this is all completely arbitrarily so and the model
unfortunately sometimes sees uh four um a token for for all four digits
sometimes for three sometimes for two sometimes for one and it's in an arbitrary uh Manner and so this is
definitely a headwind if you will for the language model and it's kind of incredible that it can kind of do it and
deal with it but it's also kind of not ideal and so that's why for example we saw that meta when they train the Llama
2 algorithm and they use sentence piece they make sure to split up all the um
all the digits as an example for uh llama 2 and this is partly to improve a
simple arithmetic kind of performance and finally why is gpt2 not
as good in Python again this is partly a modeling issue on in the architecture and the data set and the strength of the
model but it's also partially tokenization because as we saw here with the simple python example the encoding
efficiency of the tokenizer for handling spaces in Python is terrible and every single space is an individual token and
this dramatically reduces the context length that the model can attend to cross so that's almost like a tokenization bug for gpd2 and that was
later fixed with gp4 okay so here's another fun one my llm abruptly halts
when it sees the string end of text so here's um here's a very strange Behavior
print a string end of text is what I told jt4 and it says could you please specify the string and I'm I'm telling
it give me end of text and it seems like there's an issue it's not seeing end of text and then I give it end of text is
the string and then here's a string and then it just doesn't print it so obviously something is breaking here
with respect to the handling of the special token and I don't actually know what open ey is doing under the hood
here and whether they are potentially parsing this as an um as an actual token
instead of this just being uh end of text um as like individual sort of
pieces of it without the special token handling logic and so it might be that someone when they're calling do encode
uh they are passing in the allowed special and they are allowing end of text as a special character in the user
prompt but the user prompt of course is is a sort of um attacker controlled text
so you would hope that they don't really parse or use special tokens or you know
from that kind of input but it appears that there's something definitely going wrong here and um so your knowledge of
these special tokens ends up being in a tax surface potentially and so if you'd like to confuse llms then just um try to
give them some special tokens and see if you're breaking something by chance okay so this next one is a really fun one uh
the trailing whites space issue so if you come to playground and uh we come
here to GPT 3.5 turbo instruct so this is not a chat model this is a completion model so think of it more like it's a
lot more closer to a base model it does completion it will continue the token sequence so here's a tagline for ice
cream shop and we want to continue the sequence and so we can submit and get a bunch of tokens okay no problem but now
suppose I do this but instead of pressing submit here I do here's a tagline for ice cream shop space so I
have a space here before I click submit we get a warning your text ends
in a trail Ling space which causes worse performance due to how API splits text into tokens so what's happening here it
still gave us a uh sort of completion here but let's take a look at what's happening so here's a tagline for an ice
cream shop and then what does this look like in the actual actual training data
suppose you found the completion in the training document somewhere on the internet and the llm trained on this
data so maybe it's something like oh yeah maybe that's the tagline that's a terrible tagline but notice here that
when I create o you see that because there's the the space character is
always a prefix to these tokens in GPT so it's not an O token it's a space o
token the space is part of the O and together they are token 8840 that's
that's space o so what's What's Happening Here is that when I just have it like this and I let it complete the
next token it can sample the space o token but instead if I have this and I
add my space then what I'm doing here when I incode this string is I have
basically here's a t line for an ice cream uh shop and this space at the very end becomes a token
220 and so we've added token 220 and this token otherwise would be part of
the tagline because if there actually is a tagline here so space o is the token
and so this is suddenly a of distribution for the model because this space is part of the next token but
we're putting it here like this and the model has seen very very little data of
actual Space by itself and we're asking it to complete the sequence like add in more tokens but the problem is that
we've sort of begun the first token and now it's been split up and now we're out
of this distribution and now arbitrary bad things happen and it's just a very rare example for it to see something
like that and uh that's why we get the warning so the fundamental issue here is of course that um the llm is on top of
these tokens and these tokens are text chunks they're not characters in a way you and I would think of them they are
these are the atoms of what the LM is seeing and there's a bunch of weird stuff that comes out of it let's go back
to our default cell style I bet you that the model has never in its training set
seen default cell sta without Le in there it's always seen this as a single
group because uh this is some kind of a function in um I'm guess I don't
actually know what this is part of this is some kind of API but I bet you that it's never seen this combination of
tokens uh in its training data because or I think it would be extremely rare so
I took this and I copy pasted it here and I had I tried to complete from it
and the it immediately gave me a big error and it said the model predicted to completion that begins with a stop sequence resulting in no output consider
adjusting your prompt or stop sequences so what happened here when I clicked submit is that immediately the model
emitted and sort of like end of text token I think or something like that it basically predicted the stop sequence
immediately so it had no completion and so this is why I'm getting a warning again because we're off the data
distribution and the model is just uh predicting just totally arbitrary things
it's just really confused basically this is uh this is giving it brain damage it's never seen this before it's shocked
and it's predicting end of text or something I tried it again here and it in this case it completed it but then
for some reason this request May violate our usage policies this was flagged um basically something just like
goes wrong and there's something like Jank you can just feel the Jank because the model is like extremely unhappy with just this and it doesn't know how to
complete it because it's never occurred in training set in a training set it always appears like this and becomes a
single token so these kinds of issues where tokens are either you sort of like complete the
first character of the next token or you are sort of you have long tokens that you then have just some of the
characters off all of these are kind of like issues with partial tokens is how I
would describe it and if you actually dig into the T token repository go to the rust code and
search for unstable and you'll see um en code
unstable native unstable token tokens and a lot of like special case handling none of this stuff about unstable tokens
is documented anywhere but there's a ton of code dealing with unstable tokens and unstable tokens is exactly kind of like
what I'm describing here what you would like out of a completion API is something a lot more fancy like if we're
putting in default cell sta if we're asking for the next token sequence we're not actually trying to append the next
token exactly after this list we're actually trying to append we're trying to consider lots of tokens um
that if we were or I guess like we're trying to search over characters that if
we retened would be of high probability if that makes sense um so that we can actually add a single individual
character uh instead of just like adding the next full token that comes after this partial token list so I this is
very tricky to describe and I invite you to maybe like look through this it ends up being extremely gnarly and hairy kind
of topic it and it comes from tokenization fundamentally so um maybe I can even spend an entire video talking
about unstable tokens sometime in the future okay and I'm really saving the best for last my favorite one by far is
the solid gold Magikarp and it just okay so this comes from this blog post uh solid gold
Magikarp and uh this is um internet famous now for those of us in llms and
basically I I would advise you to uh read this block Post in full but basically what this person was doing is
this person went to the um token embedding stable and clustered the
tokens based on their embedding representation and this person noticed that there's a cluster of tokens that
look really strange so there's a cluster here at rot e stream Fame solid gold Magikarp Signet message like really
weird tokens in uh basically in this embedding cluster and so what are these
tokens and where do they even come from like what is solid gold magikarpet makes no sense and then they found bunch of
these tokens and then they notice that actually the plot thickens here because if you ask the model about these tokens
like you ask it uh some very benign question like please can you repeat back to me the string sold gold Magikarp uh
then you get a variety of basically totally broken llm Behavior so either you get evasion so I'm sorry I can't
hear you or you get a bunch of hallucinations as a response um you can even get back like insults so you ask it
uh about streamer bot it uh tells the and the model actually just calls you names uh or it kind of comes up with
like weird humor like you're actually breaking the model by asking about these very simple strings like at Roth and
sold gold Magikarp so like what the hell is happening and there's a variety of here documented behaviors uh there's a
bunch of tokens not just so good Magikarp that have that kind of a behavior and so basically there's a
bunch of like trigger words and if you ask the model about these trigger words or you just include them in your prompt
the model goes haywire and has all kinds of uh really Strange Behaviors including sort of ones that violate typical safety
guidelines uh and the alignment of the model like it's swearing back at you so what is happening here and how can this
possibly be true well this again comes down to tokenization so what's happening here is that sold gold Magikarp if you
actually dig into it is a Reddit user so there's a u Sol gold
Magikarp and probably what happened here even though I I don't know that this has been like really definitively explored
but what is thought to have happened is that the tokenization data set was very
different from the training data set for the actual language model so in the tokenization data set there was a ton of
redded data potentially where the user solid gold Magikarp was mentioned in the text because solid gold Magikarp was a
very common um sort of uh person who would post a lot uh this would be a string that occurs many times in a
tokenization data set because it occurs many times in a tokenization data set these tokens would end up getting merged
to the single individual token for that single Reddit user sold gold Magikarp so they would have a dedicated token in a
vocabulary of was it 50,000 tokens in gpd2 that is devoted to that Reddit user
and then what happens is the tokenization data set has those strings but then later when you train the model
the language model itself um this data from Reddit was not present and so
therefore in the entire training set for the language model sold gold Magikarp never occurs that token never appears in
the training set for the actual language model later so this token never gets activated it's initialized at random in
the beginning of optimization then you have forward backward passes and updates to the model and this token is just never updated in the embedding table
that row Vector never gets sampled it never gets used so it never gets trained and it's completely untrained it's kind
of like unallocated memory in a typical binary program written in C or something like that that so it's unallocated
memory and then at test time if you evoke this token then you're basically plucking out a row of the embedding
table that is completely untrained and that feeds into a Transformer and creates undefined behavior and that's
what we're seeing here this completely undefined never before seen in a training behavior and so any of these
kind of like weird tokens would evoke this Behavior because fundamentally the model is um is uh uh out of sample out
of distribution okay and the very last thing I wanted to just briefly mention point out although I think a lot of
people are quite aware of this is that different kinds of formats and different representations and different languages
and so on might be more or less efficient with GPD tokenizers uh or any tokenizers for any other L for that
matter so for example Json is actually really dense in tokens and yaml is a lot more efficient in tokens um so for
example this are these are the same in Json and in yaml the Json is
116 and the yaml is 99 so quite a bit of an Improvement and so in the token
economy where we are paying uh per token in many ways and you are paying in the context length and you're paying in um
dollar amount for uh the cost of processing all this kind of structured data when you have to um so prefer to
use theal over Json and in general kind of like the tokenization density is something that you have to um sort of
care about and worry about at all times and try to find efficient encoding schemes and spend a lot of time in tick
tokenizer and measure the different token efficiencies of different formats and settings and so on okay so that

**LLM标记化的奇怪行为**是理解和使用大型语言模型（LLM）时的一些常见问题和挑战。以下是视频中的一些关键点，以及这些问题为什么会发生的解释：

### 1. **LLM拼写问题**

LLM拼写不好，尤其是处理拼写相关任务时，根本原因在于标记化的方式。标记化是将字符拆分成tokens（令牌）。有些tokens非常长，一个典型的例子是“default style”，这个短语在GPT-4中被标记为单个token，而这个token包含了很多字符。由于这些tokens过长，LLM可能无法准确处理拼写问题，例如它在计算“L”的个数时会出错，因为它将整个“default style”视为一个token，而不是逐个字符来处理。

### 2. **字符级任务（如反转字符串）**

当LLM被要求执行字符串反转等字符级任务时，它有时会因为标记化的问题而失败。比如，当直接请求反转“default style”时，模型没能正确地从右到左反转字符串。但如果先逐个字符列出再进行反转，模型则能正确处理，因为它更容易将每个字符作为单独的token来操作。

### 3. **非英语语言的处理**

LLM在处理非英语语言时表现较差，这不仅仅是因为训练数据中非英语语料较少，还因为标记化器对非英语语言的处理能力不足。例如，英语中的“hello”是一个token，而韩语中的“안녕”（你好）却被分成了三个tokens。这样，模型处理非英语文本时会面临更多的挑战，导致它对这些语言的理解和生成能力较差。

### 4. **数字标记化与算术问题**

LLM在处理算术问题时也会出现困难，因为数字的标记化是非常随意的。例如，对于四位数的数字，它可能会被标记为1个token，也可能是2个token，甚至是3个token，这种标记化的任意性使得模型在执行简单的算术任务时表现不佳。Meta的Llama 2模型通过确保将所有数字都作为单独的token来改进这一点，从而提高了算术表现。

### 5. **Python代码处理的困难**

GPT-2在处理Python代码时表现较差，尤其是在处理空格时。每个空格被单独标记为一个token，这导致了上下文长度的严重限制，从而影响了模型在执行代码时的效率和准确性。GPT-4通过改进这一点，提升了处理代码的能力。

### 6. **遇到“end of text”标记时LLM的行为**

当模型遇到“end of text”标记时，它会中止输出。这是因为在训练时，模型对这种特殊token的处理并不完善，导致它无法正确识别并处理这一标记，从而导致了意外的停止或错误的行为。

### 7. **尾随空格问题**

在LLM的输入中，如果输入字符串以空格结尾，模型可能会因为空格被单独标记为一个token，从而影响模型的性能。例如，输入“here’s a tagline for an ice cream shop”与“here’s a tagline for an ice cream shop ”（后面加一个空格）会导致不同的token分配，影响模型的输出。

### 8. **部分token问题**

部分token问题指的是LLM在处理不完整的token序列时，往往会产生无法预测的行为。例如，如果一个token被拆分成多个部分，模型可能无法理解这些部分如何组合起来，导致无法正确生成接下来的tokens。

### 9. **“Solid Gold Magikarp”现象**

这是LLM中的一个非常有趣的现象。当模型遇到某些特殊的token时，可能会产生非常奇怪的行为，比如生成无意义的回答、回避问题甚至发出侮辱。这个问题的根本原因是标记化过程。在某些情况下，某些token（如“Solid Gold Magikarp”）可能是某个特定用户的名字，模型没有见过这些token，因此在遇到这些token时，它无法正确处理，导致了不可预测的行为。

### 10. **格式和表示对token效率的影响**

不同格式的数据会在标记化时有不同的token效率。例如，JSON格式的数据通常比较冗长，而YAML格式的数据更为紧凑。在涉及费用和上下文长度时，选择合适的格式对于提高模型性能至关重要。

---

这些奇怪的行为和问题大多源于LLM在标记化阶段的工作方式。标记化是将输入文本转换为模型能够处理的tokens的过程，而这个过程中的任何不完美都可能导致模型在处理任务时出现意料之外的错误。这些问题的出现说明了标记化在LLM的整体效果中扮演着极为重要的角色。


# final recommendations

concludes my fairly long video on tokenization I know it's a try I know it's annoying I know it's irritating I
personally really dislike the stage what I do have to say at this point is don't brush it off there's a lot of foot guns
sharp edges here security issues uh AI safety issues as we saw plugging in unallocated memory into uh language
models so um it's worth understanding this stage um that said I will say that
eternal glory goes to anyone who can get rid of it uh I showed you one possible paper that tried to uh do that and I
think I hope a lot more can follow over time and my final recommendations for the application right now are if you can
reuse the GPT 4 tokens and the vocabulary uh in your application then that's something you should consider and just use Tech token because it is very
efficient and nice library for inference for bpe I also really like the bite
level BP that uh Tik toen and openi uses uh if you for some reason want to train
your own vocabulary from scratch um then I would use uh the bpe with sentence
piece um oops as I mentioned I'm not a huge fan of sentence piece I don't like
its uh bite fallback and I don't like that it's doing BP on unic code code points I think it's uh it also has like
a million settings and I think there's a lot of foot gonss here and I think it's really easy to Mis calibrate them and you end up cropping your sentences or
something like that uh because of some type of parameter that you don't fully understand so so be very careful with
the settings try to copy paste exactly maybe where what meta did or basically spend a lot of time looking at all the
hyper parameters and go through the code of sentence piece and make sure that you have this correct um but even if you
have all the settings correct I still think that the algorithm is kind of inferior to what's happening here and
maybe the best if you really need to train your vocabulary maybe the best thing is to just wait for M bpe to
becomes as efficient as possible and uh that's something that maybe I hope to work on and at some point maybe we can
be training basically really what we want is we want tick token but training code and that is the ideal thing that
currently does not exist and MBP is um is in implementation of it but currently
it's in Python so that's currently what I have to say for uh tokenization there might be an advanced video that has even
drier and even more detailed in the future but for now I think we're going to leave things off here and uh I hope
that was helpful bye
and uh they increase this contact size from gpt1 of 512 uh to 1024 and GPT 4
two the next okay next I would like us to briefly walk through the code from open
AI on the gpt2 encoded
ATP I'm sorry I'm gonna sneeze and then what's Happening Here
is this is a spous layer that I will explain in a bit What's Happening Here
is

### **最终建议与总结**

在视频的最后部分，讲者分享了自己对**tokenization**（标记化）过程的看法，并给出了几点关于如何在实际应用中处理这个过程的建议：

#### 1. **不要忽视标记化过程**

尽管标记化是一个相对烦琐、令人厌烦的过程，但讲者强调了**理解标记化的重要性**，并指出了潜在的安全风险和AI安全问题。比如，未分配的内存被直接传入语言模型的例子，就能导致模型出现不可预测的行为。因此，必须仔细理解标记化阶段，避免踩雷。

#### 2. **解决标记化问题的愿景**

讲者提到，虽然目前标记化阶段有很多问题，但也有一些研究和方法试图解决这些问题。比如，有一篇论文尝试**消除标记化**的过程，讲者希望未来能看到更多此类研究，并期望能够逐步解决这一问题。

#### 3. **关于使用现有标记化工具**

如果你的应用程序能够复用**GPT-4的标记化和词汇表**，那么这是一个不错的选择。**Tech Token**被推荐作为一个高效且便捷的BPE（字节对编码）推理工具，适用于很多任务。另一个推荐的工具是**Byte-Level BPE**，这是OpenAI和Tech Token采用的标记化方法，适用于大多数应用。

#### 4. **自己训练标记词汇表的注意事项**

如果你确实需要从头开始训练自己的词汇表，讲者建议使用**SentencePiece**与BPE结合的方式。不过，讲者也指出，**SentencePiece**有很多设置，且容易出错，尤其是在处理Unicode代码点时。因此，如果使用**SentencePiece**，需要非常小心参数的调试和设置，甚至最好模仿Meta使用的设置。此外，虽然**SentencePiece**的算法有其优点，但讲者认为它仍然不如**Tech Token**的效率高。

#### 5. **未来的理想工具**

如果将来可以使用类似于**Tech Token**的标记化工具来训练词汇表，这将是理想的情况。目前，MBPE（多字节对编码）正朝着这个方向发展，虽然它目前还在Python中实现，但讲者希望未来能有更高效的解决方案。

### **总结**

讲者强调，标记化不仅是一个技术细节，还与模型性能、推理效率以及安全性息息相关。虽然这一过程目前存在许多问题和挑战，但理解它并小心处理，可以帮助我们避免很多潜在的问题。
