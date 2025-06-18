We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).

Links:
- makemore on github: https://github.com/karpathy/makemore
- jupyter notebook I built in this video: https://github.com/karpathy/nn-zero-t...
- collab notebook (new)!!!: https://colab.research.google.com/dri...
- Bengio et al. 2003 MLP language model paper (pdf): https://www.jmlr.org/papers/volume3/b...
- my website: https://karpathy.ai
- my twitter:   / karpathy  
- (new) Neural Networks: Zero to Hero series Discord channel:   / discord   , for people who'd like to chat more and go beyond youtube comments

Useful links:
- PyTorch internals ref http://blog.ezyang.com/2019/05/pytorc...

Exercises:
- E01: Tune the hyperparameters of the training to beat my best validation loss of 2.2
- E02: I was not careful with the intialization of the network in this video. (1) What is the loss you'd get if the - predicted probabilities at initialization were perfectly uniform? What loss do we achieve? (2) Can you tune the - initialization to get a starting loss that is much more similar to (1)?
- E03: Read the Bengio et al 2003 paper (link above), implement and try any idea from the paper. Did it work?

Chapters:
- 00:00:00 intro
- 00:01:48 Bengio et al. 2003 (MLP language model) paper walkthrough
- 00:09:03 (re-)building our training dataset
- 00:12:19 implementing the embedding lookup table
- 00:18:35 implementing the hidden layer + internals of torch.Tensor: storage, views
- 00:29:15 implementing the output layer
- 00:29:53 implementing the negative log likelihood loss
- 00:32:17 summary of the full network
- 00:32:49 introducing F.cross_entropy and why
- 00:37:56 implementing the training loop, overfitting one batch
- 00:41:25 training on the full dataset, minibatches
- 00:45:40 finding a good initial learning rate
- 00:53:20 splitting up the dataset into train/val/test splits and why
- 01:00:49 experiment: larger hidden layer
- 01:05:27 visualizing the character embeddings
- 01:07:16 experiment: larger embedding size
- 01:11:46 summary of our final code, conclusion
- 01:13:24 sampling from the model
- 01:14:55 google collab (new!!) notebook advertisement

以下是你提供内容的中文翻译（含章节和练习说明），适合对机器学习和神经网络有初步了解的学习者：

---

我们实现了一个**多层感知机（MLP）字符级语言模型**。本视频还介绍了许多**机器学习的基础知识**，包括：模型训练、学习率调整、超参数设置、模型评估、训练集/验证集/测试集划分，以及欠拟合/过拟合等内容。

🔗 **相关链接：**

* makemore 项目 GitHub： [https://github.com/karpathy/makemore](https://github.com/karpathy/makemore)
* 视频中我制作的 Jupyter Notebook： [https://github.com/karpathy/nn-zero-t](https://github.com/karpathy/nn-zero-t)...
* \[新] Google Colab Notebook（可在线运行）：[https://colab.research.google.com/dri](https://colab.research.google.com/dri)...
* Bengio 等人 2003 年 MLP 语言模型论文（PDF）：[https://www.jmlr.org/papers/volume3/b](https://www.jmlr.org/papers/volume3/b)...
* 我的个人网站：[https://karpathy.ai](https://karpathy.ai)
* 我的推特账号：[@karpathy](https://twitter.com/karpathy)
* \[新] “神经网络：从零开始”系列 Discord 频道：/discord，欢迎大家交流学习、深入探讨

📚 **有用的参考资料：**

* PyTorch 底层实现参考博客：[http://blog.ezyang.com/2019/05/pytorc](http://blog.ezyang.com/2019/05/pytorc)...

---

### 🧪 **练习题：**

* **E01：** 调整训练的超参数，尝试击败我目前最好的验证集损失 2.2
* **E02：** 本视频中初始化做得不够仔细。

  * (1) 如果初始预测概率是完全均匀分布，理论损失是多少？我们实际初始化时的损失是多少？
  * (2) 能否通过调整初始化方法，让初始损失更接近理论值？
* **E03：** 阅读上面提到的 Bengio 2003 年论文，挑选其中的一个想法，尝试实现并测试其效果。看看是否有提升。

---

### 📺 **视频章节：**

```
00:00:00  引言  
00:01:48  Bengio 等人 2003 年 MLP 语言模型论文解析  
00:09:03  （重新）构建训练数据集  
00:12:19  实现嵌入查找表（embedding lookup table）  
00:18:35  实现隐藏层 + torch.Tensor 的底层机制：storage、视图（views）  
00:29:15  实现输出层  
00:29:53  实现负对数似然损失（NLL Loss）  
00:32:17  网络结构总结  
00:32:49  引入 F.cross_entropy 以及理由  
00:37:56  实现训练循环，首先在一个小批数据上过拟合  
00:41:25  在完整数据集上训练，使用小批量（minibatches）  
00:45:40  寻找合适的初始学习率  
00:53:20  将数据划分为训练/验证/测试集，并解释原因  
01:00:49  实验：增大隐藏层规模  
01:05:27  可视化字符嵌入（embedding）  
01:07:16  实验：增大嵌入维度  
01:11:46  最终代码总结与结语  
01:13:24  从模型中采样文本  
01:14:55  Google Colab notebook 广告
```

如果你需要我整理这段视频内容的学习路线、代码笔记、或者将内容简化为 HSK3 级中文，也可以告诉我！

# intro

hi everyone today we are continuing our implementation of makemore now in the last lecture we implemented
the bigram language model and we implemented it both using counts and also using a super simple neural network
that had a single linear layer now this is the jupyter notebook that we built out last
lecture and we saw that the way we approached this is that we looked at only the single previous character and we
predicted the distribution for the character that would go next in the sequence and we did that by taking
counts and normalizing them into probabilities so that each row here sums to one
now this is all well and good if you only have one character of previous context and this works and it's approachable the
problem with this model of course is that the predictions from this model are not very good because you only take one
character of context so the model didn't produce very name like sounding things
now the problem with this approach though is that if we are to take more context into account when predicting the
next character in a sequence things quickly blow up and this table the size of this table grows and in fact it grows
exponentially with the length of the context because if we only take a single character at a time that's 27
possibilities of context but if we take two characters in the past and try to predict the third one
suddenly the number of rows in this matrix you can look at it that way is 27 times 27 so there's 729
possibilities for what could have come in the context if we take three characters as the
context suddenly we have 20 000 possibilities of context
and so there's just way too many rows of this matrix it's way too few counts
for each possibility and the whole thing just kind of explodes and doesn't work very well

当然可以，以下是这段英文内容的中文解释：

---

## 🎬 引言部分讲了什么？

大家好，今天我们继续实现 **makemore** 项目。在上一讲中，我们已经实现了一个**bigram（二元语法）语言模型**，我们用两种方法实现了它：

1. **基于统计计数**：统计每个字符后面出现其他字符的频率；
2. **基于神经网络**：用一个**非常简单的神经网络**，只有一层线性层（linear layer），来做预测。

我们在上节课中使用了 Jupyter Notebook 来实现这些内容。

---

## 🧠 bigram 模型是怎么工作的？

我们的方法是这样的：

* 给定前一个字符，预测下一个字符的概率分布。
* 我们使用了统计的方法，把出现频率归一化成概率（每一行概率加起来是 1）。

这种方法在只使用一个前文字符作为上下文时**是可行的**，也比较容易理解。

---

## ❗ 但这个模型的问题在哪里？

虽然这种模型可以工作，但**效果不好**，因为它只考虑了一个字符的上下文，预测能力很有限。生成的名字不像真实的人名。

如果我们想提高模型的性能，就需要**使用更多的上下文字符**（比如两个、三个前文字符）来预测下一个字符。

---

## 🔺 但为什么不能直接加上下文？

问题在于：**上下文越长，组合可能性越多，模型就变得无法处理**。

* 1 个字符的上下文：有 27 种可能（26 个英文字母 + 1 个特殊符号）
* 2 个字符：27 × 27 = 729 种上下文组合
* 3 个字符：27 × 27 × 27 = 19,683 种组合

如果用统计表（矩阵）来表示这些上下文和预测的关系：

* 表格的行数会**指数级增长**
* 每个组合可能出现的次数（样本数）却非常少，导致**数据稀疏**
* 最后模型效果也不会太好，**计算资源和效果都不划算**

---

✅ **总结**：

* bigram 模型简单但效果不好；
* 想用更长上下文就会遇到**组合爆炸问题**；
* 所以我们需要引入更强大的模型（比如多层感知机 MLP）来处理上下文的建模问题。

如果你想继续看后面讲 MLP 部分的内容，我可以继续帮你翻译解释。是否需要？

---

问题在于：**上下文越长，组合可能性越多，模型就变得无法处理**。

* 1 个字符的上下文：有 27 种可能（26 个英文字母 + 1 个特殊符号）
* 2 个字符：27 × 27 = 729 种上下文组合
* 3 个字符：27 × 27 × 27 = 19,683 种组合

这个地方举个例子详细说一下

---

好的，我们用一个更**具体、生动的例子**来说明“上下文越长，组合可能性越多，模型变得难以处理”的问题。

---

## 🎯 设定场景：预测下一个字母

我们正在做的是一个**字符级语言模型（character-level language model）**，也就是：

> 给定前面的若干个字符，预测下一个字符出现的概率。

我们假设字母表是 26 个英文字母 + 一个特殊字符（比如开始符号 `<START>` 或空格），共 **27 个字符**。

---

## ✅ 举例说明

### 🧩 如果我们用 **1 个字符**作为上下文：

我们就问：

> “给定前一个字符是 `a`，下一个字符最可能是什么？”

所有可能的情况是：

```
a → ?
b → ?
c → ?
...
z → ?
<START> → ?
```

👉 总共就是 27 行的表格，每一行记录“给定某个字符后，预测下一个字符的概率分布”。

这张表叫做 **bigram 表（2-gram）**，27 × 27 的大小， manageable ✅。

---

### 🧩 如果我们用 **2 个字符**作为上下文：

问题变成：

> “给定前两个字符是 `th`，下一个字符最可能是什么？”

所有可能的上下文组合是：

* `aa`, `ab`, `ac`, ..., `az`, `a<START>`
* `ba`, `bb`, ..., `bz`, ...
* ...
* `<START><START>`, `<START>a`, ..., `<START>z`

👉 总共有 27 × 27 = **729 种组合**

对于每一种组合都要记录一个“预测下一个字符的概率分布”——这就变成一个 729 行、每行 27 列的矩阵。

---

### 🧩 如果我们用 **3 个字符**作为上下文：

我们要回答的问题是：

> “给定前 3 个字符是 `the`，下一个字符最可能是什么？”

可能的上下文变成：

* `aaa`, `aab`, `aac`, ..., `aaz`, `aa<START>`
* `aba`, `abb`, ..., ...
* ...
* `<START><START><START>`, ..., `<START><START>z`

👉 总共有 27 × 27 × 27 = **19,683 种组合**！

每一个组合都要有一行记录下一个字符的分布。如果我们用表格来存储这个信息，**就需要近 2 万行的矩阵**！

---

## 🚨 问题来了：

* 这么多组合，**很多组合在训练集中可能根本没出现过**；
* 那这些“没见过的组合”我们就**无法统计概率**，只能设为零；
* 模型就**无法泛化**（比如只训练过 “the”，没训练过 “tne”，模型就懵了）；
* **内存消耗也迅速膨胀**，你需要存下近 2 万行的统计结果；
* 如果你想要 4 个字符做上下文呢？27⁴ = 531,441 行！完全失控。

---

## ✅ 小结

| 上下文长度 | 所有组合数量（行数） |
| ----- | ---------- |
| 1 个字符 | 27         |
| 2 个字符 | 729        |
| 3 个字符 | 19,683     |
| 4 个字符 | 531,441    |

这就是所谓的**组合爆炸（combinatorial explosion）**问题：
👉 上下文长度每增加 1，可能的组合数就乘以 27，模型就变得**不可扩展**，也很难训练。

---

## 💡 怎么解决？

这就是为什么我们要引入神经网络，比如：

* 用 **embedding** 把字符变成向量；
* 用 **多层感知机（MLP）** 或 **RNN、Transformer** 来从上下文中学习模式；
* 这些模型可以**泛化**到没见过的组合。

---

如果你想我再具体画个图、或者用表格方式展示，也可以告诉我！

---

# Bengio et al. 2003 (MLP language model) paper walkthrough

so that's why today we're going to move on to this bullet point here and we're going to implement a multi-layer perceptron model to predict the next uh
character in a sequence and this modeling approach that we're going to adopt follows this paper
benguetal 2003 so i have the paper pulled up here now this isn't the very first paper that
proposed the use of multiglio perceptrons or neural networks to predict the next character or token in a
sequence but it's definitely one that is uh was very influential around that time it is very often cited to stand in for
this idea and i think it's a very nice write-up and so this is the paper that we're going to first look at and then
implement now this paper has 19 pages so we don't have time to go into the full detail of this paper but i
invite you to read it it's very readable interesting and has a lot of interesting ideas in it as well
in the introduction they describe the exact same problem i just described and then to address it they propose the
following model now keep in mind that we are building a character level language model so we're
working on the level of characters in this paper they have a vocabulary of 17 000 possible words and they instead
build a word level language model but we're going to still stick with the characters but we'll take the same modeling approach
now what they do is basically they propose to take every one of these words seventeen thousand words and they're
going to associate to each word a say thirty dimensional feature vector
so every word is now embedded into a thirty dimensional space you can think of it that way so we have
17 000 points or vectors in a 30 dimensional space and that's um you might imagine that's
very crowded that's a lot of points for a very small space now in the beginning these words are
initialized completely randomly so they're spread out at random but then we're going to tune these
embeddings of these words using back propagation so during the course of training of this neural network these points or vectors
are going to basically move around in this space and you might imagine that for example words that have very similar
meanings or that are indeed synonyms of each other might end up in a very similar part of the space and conversely
words that mean very different things would go somewhere else in a space now their modeling approach otherwise is
identical to ours they are using a multi-layer neural network to predict the next word given the previous words
and to train the neural network they are maximizing the log likelihood of the training data just like we did
so the modeling approach itself is identical now here they have a concrete example of this intuition
why does it work basically suppose that for example you are trying to predict a dog was running in a blank
now suppose that the exact phrase a dog was running in a has never occurred in a training data
and here you are at sort of test time later when the model is deployed somewhere and it's trying to make a sentence and
it's saying a dog was running in a blank and because it's never encountered this exact phrase in the training set you're
out of distribution as we say like you don't have fundamentally any reason to suspect
what might come next but this approach actually allows you to get around that because maybe you didn't
see the exact phrase a dog was running in a something but maybe you've seen similar phrases maybe you've seen the
phrase the dog was running in a blank and maybe your network has learned that a and the
are like frequently are interchangeable with each other and so maybe it took the embedding for a and the embedding for
the and it actually put them like nearby each other in the space and so you can transfer knowledge through that
embedding and you can generalize in that way similarly the network could know that cats and dogs are animals and they
co-occur in lots of very similar contexts and so even though you haven't seen this exact phrase
or if you haven't seen exactly walking or running you can through the embedding space
transfer knowledge and you can generalize to novel scenarios so let's now scroll down to the diagram
of the neural network they have a nice diagram here and in this example we are taking three
previous words and we are trying to predict the fourth word in a sequence
now these three previous words as i mentioned uh we have a vocabulary of 17 000 um possible words
so every one of these basically basically are the index of the incoming word
and because there are 17 000 words this is an integer between 0 and 16999
now there's also a lookup table that they call c this lookup table is a matrix that is 17
000 by say 30. and basically what we're doing here is we're treating this as a lookup table
and so every index is plucking out a row of this embedding matrix
so that each index is converted to the 30 dimensional vector that corresponds to the embedding vector for that word
so here we have the input layer of 30 neurons for three words making up 90
neurons in total and here they're saying that this matrix c is shared across all the words so
we're always indexing into the same matrix c over and over um for each one of these
words next up is the hidden layer of this neural network the size of this hidden neural layer of this neural net
is a hoppy parameter so we use the word hyperparameter when it's kind of like a design choice up to the designer of the
neural net and this can be as large as you'd like or as small as you'd like so for example the size could be a hundred
and we are going to go over multiple choices of the size of this hidden layer and we're going to evaluate how well
they work so say there were 100 neurons here all of them would be fully connected to the
90 words or 90 um numbers that make up these three words
so this is a fully connected layer then there's a 10 inch long linearity and then there's this output layer and
because there are 17 000 possible words that could come next this layer has 17 000 neurons
and all of them are fully connected to all of these neurons in the hidden layer
so there's a lot of parameters here because there's a lot of words so most computation is here this is the
expensive layer now there are 17 000 logits here so on top of there we have the softmax layer
which we've seen in our previous video as well so every one of these logits is exponentiated and then everything is
normalized to sum to 1 so that we have a nice probability distribution for the next word in the sequence
now of course during training we actually have the label we have the identity of the next word in a sequence
that word or its index is used to pluck out the
probability of that word and then we are maximizing the probability of that word
with respect to the parameters of this neural net so the parameters are the weights and biases of this output layer
the weights and biases of the hidden layer and the embedding lookup table c
and all of that is optimized using back propagation and these uh dashed arrows ignore those
uh that represents a variation of a neural nut that we are not going to explore in this video so that's the setup and now let's
implement it okay so i started a brand new notebook for this lecture we are importing pytorch and we are

当然可以！以下是这段内容的中文解释，配合结构清晰的讲解，帮助你理解这部分的知识点：

---

## 📚 Bengio 等人 2003 年 MLP 语言模型论文导读

### ✅ 为什么要看这篇论文？

我们今天将实现一个**多层感知机（MLP）语言模型**，用来预测下一个字符。这个建模方法就是基于 **Bengio 等人 2003 年的论文**。
虽然这并不是第一篇提出用神经网络预测下一个词或字符的论文，但它非常具有影响力，也经常被引用，因此是一个很好的起点。

---

## 📖 论文内容概述（我们要参考的核心思路）

这篇论文本身是**词级语言模型**，即它处理的是「词」而非「字符」：

* 作者使用了一个包含 17,000 个词的词表；
* 每个词会被映射成一个 **30维的向量（embedding）**；
* 这些词向量开始是随机初始化的；
* 在训练过程中，**通过反向传播（backpropagation）不断调整这些向量**，使它们在空间中移动到合理的位置。

> 🎯 **好处：**
> 即使模型没见过某个完整句子，也能通过词之间的“相似性”进行**泛化**（generalization）。比如：

* 模型没见过 "a dog was running in a ..."，但它见过 "the dog was running in a ..."；
* 它知道 "a" 和 "the" 在某些上下文中是可以互换的；
* 于是它把 "a" 和 "the" 的 embedding 放得很近；
* 最终模型就能在没见过的句子结构中，**做出合理预测**。

---

## 🧠 论文提出的网络结构（MLP 模型结构）

我们来看看他们提出的模型图解。这里是词级的模型（我们会用同样结构来处理字符）：

### 🌟 输入层（Input Layer）

* 输入是上下文中前 3 个词（比如前 3 个字符）；
* 每个词是一个 0 到 16999 之间的整数（代表词的索引）；
* 每个词的索引会通过查表（lookup）转为一个 **30维向量**；
* 所以输入总共有 **3 × 30 = 90 个数字**（也就是 90 个“神经元”）；

> 💡**这个查表矩阵叫 C，是一个 17000 × 30 的矩阵**。查表其实就是用索引去提取一行向量。

---

### 🧱 隐藏层（Hidden Layer）

* 接下来是一个**全连接的隐藏层**，神经元个数是一个超参数（hyperparameter），比如 100 个；
* 所有 90 个输入都和这 100 个神经元相连；
* 隐藏层后接一个非线性函数，比如 `tanh`。

---

### 🎯 输出层（Output Layer）

* 最后是输出层，它的大小是词表大小，即 **17,000 个神经元**；
* 每个隐藏层神经元都连接到输出层的每个神经元；
* 输出的是 **logits（未归一化分数）**；
* 然后使用 **softmax** 转换成概率分布（使总和为 1）；

---

### 🎯 训练方式

* 在训练过程中，我们已知真实下一个词；
* 我们用该词的索引取出模型预测的概率；
* 然后**最大化该词的预测概率**（对数似然）；
* 通过反向传播来更新所有参数，包括：

  * 输出层的权重和偏置；
  * 隐藏层的权重和偏置；
  * 查表矩阵 C 中的向量。

---

### 📝 结尾说明

* 虚线箭头表示的是论文中提到但本视频不实现的模型变体；
* 接下来会用 PyTorch 从零开始实现这个模型。

---

如果你接下来需要我继续翻译讲解 PyTorch 实现的部分，我可以按代码逐行解释。是否需要继续？

# (re-)building our training dataset

importing matplotlib so we can create figures then i am reading all the names into a
list of words like i did before and i'm showing the first eight right here keep in mind that we have a 32 000 in
total these are just the first eight and then here i'm building out the vocabulary of characters and all the
mappings from the characters as strings to integers and vice versa
now the first thing we want to do is we want to compile the data set for the neural network and i had to rewrite this code um i'll
show you in a second what it looks like so this is the code that i created for
the dataset creation so let me first run it and then i'll briefly explain how this works
so first we're going to define something called block size and this is basically the context length of how many
characters do we take to predict the next one so here in this example we're taking three characters to predict the
fourth one so we have a block size of three that's the size of the block that supports the prediction
then here i'm building out the x and y the x are the input to the neural net
and the y are the labels for each example inside x
then i'm airing over the first five words i'm doing first five just for efficiency while we are developing all
the code but then later we're going to come here and erase this so that we use the entire training set
so here i'm printing the word emma and here i'm basically showing the examples that we can generate the five
examples that we can generate out of the single um sort of word emma
so when we are given the context of just uh dot dot the first character in a sequence is e
in this context the label is m when the context is this the label is m
and so forth and so the way i build this out is first i start with a padded context of just zero tokens
then i iterate over all the characters i get the character in the sequence and i
basically build out the array y of this current character and the array x which stores the current running context
and then here see i print everything and here i um crop the context and enter the
new character in a sequence so this is kind of like a rolling window of context
now we can change the block size here to for example four and in that case we'll be predicting the
fifth character given the previous four or it can be five and then it would look like this
or it can be say ten and then it would look something like this we're taking ten characters to
predict the eleventh one and we're always padding with dots so let me bring this back to three
just so that we have what we have here in the paper and finally the data set right now looks
as follows from these five words we have created a data set of 32 examples
and each input of the neural net is three integers and we have a label that is also an integer
y so x looks like this these are the individual examples
and then y are the labels so given this
let's now write a neural network that takes these axes and predicts the y's first let's build the embedding lookup

好的，这段内容是在解释如何**构建用于训练神经网络的数据集**，下面我将一步步用中文详细解释这个过程，帮助你完全理解：

---

## 🧱 我们要做什么？

我们正在构建一个**字符级语言模型**，也就是说：

> 给定前面几个字符（上下文），让神经网络预测下一个字符。

例如，给定 "e", "m", "m"，模型要预测 "a"。这就是我们要生成的训练数据。

---

## 🧰 步骤解析

### 1. 导入模块

```python
import matplotlib.pyplot as plt
```

这个 `matplotlib` 是为了后面可视化用的。

---

### 2. 读取数据

作者读取了一个名字数据集（名字列表），总共有 **32,000 个名字**。这里只显示了前 8 个作为例子。

---

### 3. 建立词表（vocabulary）

程序建立了两个映射：

* 每个字符（例如 `'a'`, `'b'`, `'.'`）对应一个整数索引；
* 每个整数可以反查回字符（字符串 ↔ 整数的映射）。

这是字符级模型的基本操作，用来把字符转成模型能处理的数字。

---

### 4. 设定 block\_size（上下文长度）

```python
block_size = 3
```

这表示我们用**几个前文字符（上下文）来预测下一个字符**。

例如：

* 如果 `block_size = 3`，我们就是用 "e", "m", "m" → 预测 "a"；
* 如果 `block_size = 4`，我们就用 4 个字符预测第 5 个；
* `block_size` 可以改为 5、10……任意你想要的上下文长度。

---

### 5. 构建训练样本 x 和标签 y

```python
x = []
y = []
```

* `x`：是输入，即每组上下文字符对应的数字编码；
* `y`：是标签，即每个输入对应的「正确的下一个字符」的数字编码。

---

### 6. 举例说明：用 "emma" 构建样本

我们来看一个名字 `emma`，构建方式如下（假设用 `block_size = 3`）：

#### 初始上下文用句点补齐

```text
'.', '.', '.'  → 'e'
'.', '.', 'e'  → 'm'
'.', 'e', 'm'  → 'm'
'e', 'm', 'm'  → 'a'
'm', 'm', 'a'  → '.'
```

注意最后一个 `'.'` 是代表名字结束的特殊字符（类似 `<EOS>`）。

---

### 7. 如何构建？

用一个「滑动窗口」的方法：

* 一开始用三个 `0`（表示句点）作为起始上下文；
* 然后一个个字符加入上下文，每加入一个就形成一个新的训练样本；
* 每一步，输入是当前的 3 个上下文字符，标签是下一个字符。

👉 这就像一个\*\*“滚动的窗口”\*\*，每次往前滑动一位。

---

### 8. 最终生成什么样的数据？

假设我们只用了前 5 个名字，总共会生成 **32 个训练样本**。

每个样本是：

* `x`: 一个包含 3 个整数的列表（表示上下文中的 3 个字符）；
* `y`: 一个整数（表示下一个字符）

比如：

```python
x = [[0, 0, 5], [0, 5, 13], [5, 13, 13], ...]
y = [13, 13, 1, ...]
```

这些数字可以通过 `itos` 映射表还原成字符。

---

### 🧠 下一步预告

接下来，我们就可以用这些 `x` 和 `y` 来训练一个神经网络：

> 输入 `x` → 嵌入层 → 隐藏层 → 输出层 → softmax → 预测 y

---

## ✅ 小结

| 步骤             | 说明                 |
| -------------- | ------------------ |
| 读取数据           | 读取名字组成的字符串列表       |
| 构建词表           | 字符 ↔ 整数的双向映射       |
| 设置 block\_size | 决定用几个字符来预测下一个字符    |
| 滑动窗口生成样本       | 用每个名字构造多个 (x, y) 对 |
| 构建训练数据集        | 最终用于神经网络训练         |

---

如果你需要我帮你写出这段生成训练集的 Python 代码，并逐行解释，我也可以！是否需要？

----

```python
block_size = 3 
context = [0] * block_size
```
解释一下

---

当然可以！

```python
block_size = 3 
context = [0] * block_size
```

这两行代码出现在我们构建训练数据的过程中，用来处理**字符上下文（context）初始化**。

---

### 🧱 逐行解释：

#### ✅ `block_size = 3`

这表示：

> 我们的神经网络每次用「前面 3 个字符」作为上下文来预测下一个字符。

例如：

* 如果你想预测 `a`，你会用 `e`, `m`, `m` 作为输入；
* 所以输入长度是 3，称为一个 “block”（块），用来提供上下文信息。

---

#### ✅ `context = [0] * block_size`

这行代码的意思是：

> 用 0 来初始化一个长度为 `block_size` 的列表，表示最开始的“上下文”。

所以：

```python
context = [0] * 3  # => [0, 0, 0]
```

这里的 `0` 实际上是字符 `'.'` 的整数编码（在 Karpathy 的代码中，`.` 表示名字的开始或结束）。

---

### 🔁 举个例子：用来预测 `emma`

在开始构造第一个名字 `"emma"` 的训练样本时：

```text
1. 上下文是 [0, 0, 0] → 下一个字符是 'e' → 生成样本 x = [0, 0, 0], y = index('e')
2. 然后把 'e' 的 index 放入 context → 新的 context 是 [0, 0, index('e')]
3. 然后预测 'm' → 生成样本 ...
```

这个 context 列表会像一个**滑动窗口**一样，每读一个字符就把它加入到 context 的末尾，并丢掉最前面的那个字符。

---

### 🧠 为什么要用 `[0]*block_size` 初始化？

因为在处理一个名字的**最开始几个字符**时，没有足够的上下文字符可用，比如：

* `emma` 的第一个字符是 `'e'`，但前面并没有 3 个字符；
* 所以我们用特殊符号 `'.'` 来**填补空缺的上下文**；
* 而 `'.'` 的整数编码就是 0。

---

### ✅ 总结

| 代码                           | 解释                                       |
| ---------------------------- | ---------------------------------------- |
| `block_size = 3`             | 我们用 3 个字符来预测下一个字符                        |
| `context = [0] * block_size` | 初始化上下文为 \[0, 0, 0]，表示一开始还没读入任何字符（全是句点占位） |

---

如果你想，我还可以写一段演示滑动窗口如何更新 context 的小程序，也可以可视化整个样本构造流程。是否需要？

---

# implementing the embedding lookup table

table c so we have 27 possible characters and we're going to embed them in a lower
dimensional space in the paper they have 17 000 words and they bet them in uh spaces as small
dimensional as 30. so they cram 17 000 words into 30 dimensional space in our
case we have only 27 possible characters so let's grab them in something as small as to start with for example a
two-dimensional space so this lookup table will be random numbers and we'll have 27 rows and we'll have
two columns right so each 20 each one of 27 characters will have a two-dimensional
embedding so that's our matrix c of embeddings in
the beginning initialized randomly now before we embed all of the integers inside the input x using this lookup
table c let me actually just try to embed a single individual integer like say five
so we get a sense of how this works now one way this works of course is we can just take the c and we can index into
row five and that gives us a vector the fifth row of c
and um this is one way to do it the other way that i presented in the previous lecture is actually seemingly
different but actually identical so in the previous lecture what we did is we took these integers and we used the one
hot encoding to first encode them so f.1 hot we want to encode integer 5 and we want
to tell it that the number of classes is 27 so that's the 26 dimensional vector of all zeros except the fifth bit is
turned on now this actually doesn't work
the reason is that this input actually must be a doorstop tensor and i'm making some of these errors
intentionally just so you get to see some errors and how to fix them so this must be a tester not an int
fairly straightforward to fix we get a one hot vector the fifth dimension is one and the shape of this is 27.
and now notice that just as i briefly alluded to in the previous video if we take this one hot vector and we multiply
it by c then
what would you expect well number one first you'd expect an error
because expected scalar type long but found float so a little bit confusing but
the problem here is that one hot the data type of it is long it's a 64-bit integer but this is a
float tensor and so pytorch doesn't know how to multiply an int with a float and
that's why we had to explicitly cast this to a float so that we can multiply now the output actually here
is identical and that it's identical because of the way the matrix multiplication here works
we have the one hot um vector multiplying columns of c and because of all the zeros they
actually end up masking out everything in c except for the fifth row which is plucked out
and so we actually arrive at the same result and that tells you that here we can interpret this first
piece here this embedding of the integer we can either think of it as the integer indexing into a lookup table c but
equivalently we can also think of this little piece here as a first layer of this bigger neural net
this layer here has neurons that have no non-linearity there's no 10h they're just linear neurons and their weight
matrix is c and then we are encoding integers into one hot and feeding those into a neural
net and this first layer basically embeds them so those are two equivalent ways of doing the same thing we're just going to
index because it's much much faster and we're going to discard this interpretation of one hot inputs into
neural nets and we're just going to index integers and create and use embedding tables now embedding a single
integer like 5 is easy enough we can simply ask pytorch to retrieve the fifth row of c
or the row index five of c but how do we simultaneously embed all
of these 32 by three integers stored in array x luckily pytorch indexing is fairly
flexible and quite powerful so it doesn't just work to ask for a single element five like this
you can actually index using lists so for example we can get the rows five six and seven
and this will just work like this we can index with a list it doesn't just have to be a list it can
also be a actually a tensor of integers and we can index with that
so this is a integer tensor 567 and this will just work as well
in fact we can also for example repeat row 7 and retrieve it multiple times
and that same index will just get embedded multiple times here
so here we are indexing with a one-dimensional tensor of integers but it turns out that
you can also index with multi-dimensional tensors of integers here we have a two-dimensional in tensor
of integers so we can simply just do c at x and this just works
and the shape of this is 32 by 3 which is the original shape and
now for every one of those 32 by 3 integers we've retrieved the embedding vector here so basically we have that as an
example the 13th or example index 13
the second dimension is the integer 1 as an example and so
here if we do c of x which gives us that array and then we index into 13 by two
of that array then we we get the embedding here and you can verify that
c at one which is the integer at that location is indeed equal to this
you see they're equal so basically long story short pytorch indexing is awesome and to embed
simultaneously all of the integers in x we can simply do c of x and that is our embedding
and that just works now let's construct this layer here the hidden layer

当然可以！这段内容是在解释如何**实现字符嵌入（embedding）查找表**，是神经网络中的第一层。以下是详细的中文解释，配合例子帮助你更好地理解：

---

## 📌 我们要做什么？

我们有 27 个可能的字符（a-z 和一个特殊字符 `.`），我们要把它们映射到一个**低维的向量空间**中，比如二维空间。

这个过程叫做 **embedding（嵌入）**，对应一个**查找表（lookup table）**，矩阵大小为：

```
27（行，对应字符数量） × 2（列，对应嵌入维度）
```

---

## 🧱 第一步：初始化嵌入矩阵 `C`

```python
C = torch.randn(27, 2)
```

* `C` 是一个随机初始化的张量；
* 每一行表示一个字符的嵌入向量；
* 比如第 5 行表示字符 `e` 的嵌入向量。

---

## 🔍 如何使用 `C` 来嵌入字符？

有两种等价的方式：

### 方法 1：直接用索引取行（推荐）

```python
C[5]  # 直接取第 5 行，表示索引为 5 的字符的嵌入
```

非常简单、快速，PyTorch 直接支持。

---

### 方法 2：用 one-hot 向量乘以嵌入矩阵（原理等价）

```python
F.one_hot(torch.tensor(5), num_classes=27).float() @ C
```

* 生成一个 one-hot 向量（27维，只有第 5 位是 1）；
* 与嵌入矩阵 C 相乘，等价于只取出第 5 行。

⚠️ 注意事项：

* one-hot 是 `long` 类型，需要 `.float()` 才能乘以 `C`（浮点类型）；
* 效果等价，但**效率差很多**，所以实际代码会直接用方法 1。

---

## 🧩 嵌入一个字符很简单，那如何嵌入整个数据集的所有输入呢？

我们的输入数据 `x` 是一个 32 × 3 的二维张量（32 个样本，每个样本 3 个字符）。

PyTorch 的索引功能非常强大，你可以直接写：

```python
C[x]
```

这会返回一个新的张量，形状是：

```text
[32, 3, 2]
```

解释：

* 32：样本数
* 3：每个样本的上下文长度
* 2：每个字符的嵌入维度

所以：

> 每个字符 → 一个 2 维向量
> 每个样本（3 个字符）→ 一个 3 × 2 的矩阵
> 全部样本 → 一个 32 × 3 × 2 的张量 ✅

---

## 🧪 示例验证

假设你想验证第 13 个样本、第 2 个字符的嵌入：

```python
idx = x[13, 2]       # 是一个整数，比如 1
embedding = C[x][13, 2]  # 得到嵌入向量
C[idx] == embedding     # 结果是 True，说明嵌入正确
```

---

## ✅ 小结

| 概念       | 内容                                    |
| -------- | ------------------------------------- |
| 嵌入矩阵 `C` | 大小为 `[27, 2]`，每一行是一个字符的嵌入向量           |
| 嵌入方法 1   | 直接用索引：`C[5]`（推荐）                      |
| 嵌入方法 2   | 用 one-hot 向量与 `C` 相乘（理解用）             |
| 嵌入全部输入   | `C[x]`，输入是 `[32, 3]`，结果是 `[32, 3, 2]` |
| 好处       | 高效、简单、可自动传播梯度（用于训练）                   |

---

接下来，这个嵌入结果 `[32, 3, 2]` 将会被扁平化（flatten）后送入神经网络的隐藏层进行进一步处理。

如果你想我继续讲解「隐藏层如何处理这些嵌入向量」的部分，也可以告诉我！

# implementing the hidden layer + internals of torch.Tensor: storage, views

so we have that w1 as i'll call it are these weights which we will initialize
randomly now the number of inputs to this layer is going to be three times two right because we have
two dimensional embeddings and we have three of them so the number of inputs is 6 and the number of neurons in this layer
is a variable up to us let's use 100 neurons as an example and then biases
will be also initialized randomly as an example and let's and we just need 100 of them
now the problem with this is we can't simply normally we would take the input in this case that's embedding and we'd
like to multiply it with these weights and then we would like to add the bias this is roughly what we want to do
but the problem here is that these embeddings are stacked up in the dimensions of this input tensor
so this will not work this matrix multiplication because this is a shape 32 by 3 by 2 and i can't multiply that
by 6 by 100 so somehow we need to concatenate these inputs here together so that we can do
something along these lines which currently does not work so how do we transform this 32 by 3 by 2
into a 32 by 6 so that we can actually perform this multiplication over here i'd like to
show you that there are usually many ways of implementing what you'd like to do in
torch and some of them will be faster better shorter etc and that's because torch is a very large
library and it's got lots and lots of functions so if you just go to the documentation and click on torch you'll
see that my slider here is very tiny and that's because there are so many functions that you can call on these
tensors to transform them create them multiply them add them perform all kinds of
different operations on them and so this is kind of like
the space of possibility if you will now one of the things that you can do is if we can control here ctrl f for
concatenate and we see that there's a function torque.cat short for concatenate
and this concatenates the given sequence of tensors in a given dimension and these sensors must have the same
shape etc so we can use the concatenate operation to in a naive way concatenate
these three embeddings for each input so in this case we have m of
amp of the shape and really what we want to do is we want to retrieve these three parts and concatenate them
so we want to grab all the examples we want to grab
first the zeroth index and then all of
this so this plucks out the 32 by 2 embeddings of just the first
word here and so basically we want this guy we want the first dimension and we want
the second dimension and these are the three pieces individually
and then we want to treat this as a sequence and we want to torch that cat on that sequence so this is the list
tor.cat takes a sequence of tensors and then we have to tell it along which dimension to concatenate
so in this case all these are 32 by 2 and we want to concatenate not across dimension 0 by the cross dimension one
so passing in one gives us a result the shape of this is 32 by 6 exactly as
we'd like so that basically took 32 and squashed these by concatenating them into 32 by
6. now this is kind of ugly because this code would not generalize if we want to
later change the block size right now we have three inputs three words but what if we had five
then here we would have to change the code because i'm indexing directly well torch comes to rescue again because that
turns out to be a function called unbind and it removes a tensor dimension
so it removes the tensor dimension returns a tuple of all slices along a given dimension
without it so this is exactly what we need and basically when we call torch dot
unbind torch dot unbind of m
and pass in dimension 1 index 1 this gives us a list of
a list of tensors exactly equivalent to this so running this gives us a line
3 and it's exactly this list and so we can call torch.cat on it
and along the first dimension and this works and this shape is the same
but now this is uh it doesn't matter if we have block size 3 or 5 or 10 this will just work
so this is one way to do it but it turns out that in this case there's actually a significantly better and more efficient
way and this gives me an opportunity to hint at some of the internals of torch.tensor
so let's create an array here of elements from 0 to 17
and the shape of this is just 18. it's a single picture of 18 numbers
it turns out that we can very quickly re-represent this as different sized and
dimensional tensors we do this by calling a view and we can say that actually this is not
a single vector of 18 this is a two by nine tensor or alternatively this is a
nine by two tensor or this is actually a three by three by two tensor
as long as the total number of elements here multiply to be the same this will just work and
in pytorch this operation calling that view is extremely efficient
and the reason for that is that in each tensor there's something called the underlying storage
and the storage is just the numbers always as a one-dimensional vector and this is how this tensor is represented
in the computer memory it's always a one-dimensional vector but when we call that view we are
manipulating some of attributes of that tensor that dictate how this one-dimensional sequence is interpreted
to be an n-dimensional tensor and so what's happening here is that no memory is being changed copied moved or
created when we call that view the storage is identical but when you call that view
some of the internal attributes of the view of the sensor are being manipulated and changed in
particular that's something there's something called a storage offset strides and shapes and those are manipulated so that this one-dimensional
sequence of bytes is seen as different and dimensional arrays there's a blog post here from eric
called pi torch internals where he goes into some of this with respect to tensor and how the view of the tensor is
represented and this is really just like a logical construct of representing the physical
memory and so this is a pretty good um blog post that you can go into i might also
create an entire video on the internals of torch tensor and how this works for here we just note that this is an
extremely efficient operation and if i delete this and come back to our end
we see that the shape of our end is 32 by three by two but we can simply ask for pytorch to view this instead as
a 32 by six and the way this gets flattened into a 32 by six array
just happens that these two get stacked up in a single row and so that's basically
the concatenation operation that we're after and you can verify that this actually gives the exact same result as what we
had before so this is an element y equals and you can see that all the elements of these two tensors are the same
and so we get the exact same result so long story short we can actually just
come here and if we just view this as a 32x6
instead then this multiplication will work and give us the hidden states that we're after
so if this is h then h shape is now the 100 dimensional activations for
every one of our 32 examples and this gives the desired result let me do two things here number one let's not
use 32 we can for example do something like
m.shape at 0 so that we don't hard code these numbers and this would work for any size of this
amp or alternatively we can also do negative one when we do negative one pi torch will infer what this should be
because the number of elements must be the same and we're saying that this is 6 by church will derive that this must be
32 or whatever else it is if m is of different size the other thing is here um
one more thing i'd like to point out is here when we do the concatenation
this actually is much less efficient because um this concatenation would create a whole new tensor with a whole
new storage so new memory is being created because there's no way to concatenate tensors just by manipulating
the view attributes so this is inefficient and creates all kinds of new memory
uh so let me delete this now we don't need this and here to calculate h we want to also
dot 10h of this to get our oops to get our h
so these are now numbers between negative one and one because of the 10h and we have that the shape is 32 by 100
and that is basically this hidden layer of activations here for every one of our 32 examples
now there's one more thing i've lost over that we have to be very careful with and that this and that's this plus here
in particular we want to make sure that the broadcasting will do what we like the shape of this is 32 by 100 and the
ones shape is 100. so we see that the addition here will broadcast these two and in particular we
have 32 by 100 broadcasting to 100. so broadcasting will align on the right
create a fake dimension here so this will become a 1 by 100 row vector and then it will copy vertically
for every one of these rows of 32 and do an element wise addition so in this case the correct thing will
be happening because the same bias vector will be added to all the rows
of this matrix so that is correct that's what we'd like and it's always good
practice you just make sure so that you don't shoot yourself in the foot and finally let's create the final layer here

当然可以！这段内容主要讲的是如何：

1. 实现**神经网络的隐藏层（hidden layer）**
2. 使用 PyTorch 中的 `.view()`、`.cat()` 和广播机制等操作
3. 理解 `torch.Tensor` 的内部结构和性能优化

---

## 🧱 背景回顾

我们之前做了：

* 每个字符用 2 维向量表示（embedding）；
* 每个训练样本有 3 个字符作为上下文；
* 所以，每个输入样本是一个 `3 × 2` 的矩阵（3 个字符 × 每个字符 2 维向量）；
* 总共有 32 个样本，因此输入张量 `m` 的形状是 `[32, 3, 2]`

---

## 🧠 目标：输入送入隐藏层

隐藏层的参数包括：

```python
W1 = torch.randn(6, 100)  # 权重矩阵（输入维度 × 隐藏神经元个数）
b1 = torch.randn(100)     # 偏置向量
```

解释：

* 每个样本输入有 `3 × 2 = 6` 个数，所以权重是 `6 × 100`
* 输出隐藏层是 100 个神经元
* 每个神经元有一个偏置值

---

## 🔧 问题：怎么把 `m` 从 `[32, 3, 2]` 变成 `[32, 6]`？

### 方法 1：手动用 `torch.cat()`

把第 0、1、2 个字符分别取出，再沿着维度 1 拼接：

```python
torch.cat([m[:,0,:], m[:,1,:], m[:,2,:]], dim=1)  # 结果：[32, 6]
```

缺点：如果将来 `block_size` 变了，比如从 3 改成 5，这个代码就要改。

---

### 方法 2（更通用）：使用 `torch.unbind()` + `torch.cat()`

```python
torch.cat(torch.unbind(m, dim=1), dim=1)
```

* `unbind(dim=1)` 会把 `[32, 3, 2]` 拆成一个长度为 3 的 list，list 中每个元素是 `[32, 2]`
* 再用 `cat(..., dim=1)` 把它们拼起来变成 `[32, 6]`

优点：**支持任意 `block_size`，更通用**

---

### 方法 3（更高效推荐）：使用 `.view()`

```python
m.view(32, 6)
```

或者：

```python
m.view(-1, 6)  # -1 表示由 PyTorch 自动推导 batch size
```

> `.view()` 是一种**不复制内存**、只改变张量“看法”的方法，效率极高。

---

## 📦 PyTorch 内部机制解释（view 的背后）

* 张量底层其实就是一维数组（称为 `storage`）；
* `.view()` 只是改变**维度信息**，不移动数据；
* 所以它比 `cat()` 快很多，且不会消耗额外内存。

---

## ✅ 隐藏层计算

现在可以把 `[32, 6]` 的输入送入隐藏层了：

```python
h = torch.tanh(m.view(-1, 6) @ W1 + b1)
```

解释：

* `m.view(-1, 6)`：变成 `[32, 6]` 的输入
* `@ W1`：矩阵乘法，结果 `[32, 100]`
* `+ b1`：加上偏置（通过广播机制，`[32, 100] + [100]`）
* `tanh()`：激活函数，让值在 \[-1, 1] 之间，增强非线性建模能力

---

## 🤖 广播机制（broadcasting）解释

当我们执行 `+ b1` 时：

* `h` 是 `[32, 100]`，`b1` 是 `[100]`
* PyTorch 自动把 `b1` 变成 `[1, 100]`，然后复制成 `[32, 100]`
* 最终实现**每一行加同一个偏置向量**

这就是 **广播机制**，是 NumPy 和 PyTorch 都支持的非常强大的功能。

---

## 🧼 小优化建议

* 把 `32` 改写成 `m.shape[0]` 或 `-1`，这样能适配任意 batch size；
* 避免使用 `.cat()`（会创建新内存）而推荐使用 `.view()`（高效、零开销）；

---

## 🟢 总结

| 步骤              | 说明                            |
| --------------- | ----------------------------- |
| `W1`, `b1`      | 初始化隐藏层的权重和偏置                  |
| `m.view(-1, 6)` | 将 `[32, 3, 2]` 扁平化为 `[32, 6]` |
| `@ W1 + b1`     | 进行矩阵乘法和加偏置                    |
| `tanh()`        | 激活函数，构造非线性变换                  |
| 最终输出            | `[32, 100]`，表示 32 个样本的隐藏层输出   |

---

下一步就会把隐藏层输出送入 **输出层**（output layer），做最终的字符预测。如果你需要我继续解释输出层部分，也可以继续告诉我！

---

torch.unbind

解释一下

好的！`torch.unbind` 是 PyTorch 里非常实用的一个**张量操作函数**，它的作用是：

> 沿着某一个维度，把张量「拆解」成若干个子张量，返回一个**元组**。

简单来说，就是把某个维度“解绑（unbind）”掉。

---

### 🧱 语法

```python
torch.unbind(input, dim=0)
```

* **input**：输入张量
* **dim**：在哪个维度上拆开（默认是 `dim=0`）

返回值：一个元组，每个元素是一个子张量，去掉了 `dim` 这个维度。

---

### 🧩 例子

#### 🌰 例 1：简单 2 维张量

```python
x = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])

# x.shape = [3, 2]

# 沿着 dim=0 拆开
tup = torch.unbind(x, dim=0)

# tup 是一个长度 3 的元组
for t in tup:
    print(t)
```

输出：

```
tensor([1, 2])
tensor([3, 4])
tensor([5, 6])
```

解释：

* 原张量 shape 是 `[3, 2]`
* 沿着 `dim=0`（第一维，长度是 3）拆成 3 个 `[2]` 的张量

---

#### 🌰 例 2：3 维张量

```python
x = torch.randn(32, 3, 2)

# 沿着 dim=1 拆
tup = torch.unbind(x, dim=1)

print(len(tup))  # 3
print(tup[0].shape)  # torch.Size([32, 2])
```

解释：

* 原张量 shape 是 `[32, 3, 2]`
* `dim=1` 是长度为 3
* 所以拆出 3 个子张量，每个 shape 是 `[32, 2]`

---

### 📌 和 `split`、`chunk` 的区别

| 函数             | 功能                              |
| -------------- | ------------------------------- |
| `torch.unbind` | 沿某个维度「拆」成若干子张量，返回**元组**，不保留那个维度 |
| `torch.split`  | 按块拆分，返回**list**，保留维度            |
| `torch.chunk`  | 均匀拆分，返回**list**，保留维度            |

---

### ✅ 为什么在 embedding 操作里用 `unbind`？

我们之前 embedding 的 shape 是 `[batch_size, block_size, embedding_dim]`：

```python
[32, 3, 2]
```

* 要把 3 个 embedding（每个是 `[32, 2]`）拼成一个大的 `[32, 6]`
* 用 `unbind(dim=1)` 把第二维「解绑」成一个 list（长度为 3），然后用 `torch.cat()` 拼起来

代码：

```python
torch.cat(torch.unbind(m, dim=1), dim=1)
```

这样做比手动取 `m[:,0,:]`，`m[:,1,:]` 更灵活，因为 block\_size 改变时不用改代码 ✅

---

### ✅ 总结

* `torch.unbind(input, dim)` → 沿某维拆成 tuple
* 常用场景：拼 embedding、拼历史上下文
* 好处：灵活、简洁

---

如果你需要我再举更多实际例子，或者写一段可复用的小 demo，随时告诉我！🌟

---

torch.tanh

解释一下

当然可以！

`torch.tanh` 是 PyTorch 里的一个**激活函数**，全称是 **双曲正切函数（hyperbolic tangent）**。

---

### 📐 函数定义

```python
torch.tanh(input)
```

对输入张量中的每一个元素，计算：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

---

### 📈 函数特性

* **输入范围**：任意实数
* **输出范围**：`[-1, 1]`
* 当 `x → +∞`，tanh(x) → 1
* 当 `x → -∞`，tanh(x) → -1
* 当 `x = 0`，tanh(0) = 0

也就是说：

* 小的正数 → 接近 0 到 1
* 小的负数 → 接近 0 到 -1
* 对称于原点（0，0）

---

### 🔍 在神经网络中的作用

**为什么要用 `tanh`？**

* 把线性变换后的输出「压缩」到 `[-1, 1]`，形成**非线性映射**，这样神经网络可以拟合复杂的模式；
* 如果没有 `tanh` 或其他非线性，多个线性层叠加起来还是线性，无法建模复杂函数；
* `tanh` 对称于 0，相比于 `sigmoid(0~1)`，对收敛更有利（因为均值是 0）；

---

### 🚀 代码示例

```python
import torch

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
y = torch.tanh(x)

print(y)
```

输出：

```
tensor([-0.9951, -0.7616,  0.0000,  0.7616,  0.9951])
```

可以看到：

* -3 → -0.9951
* 0 → 0
* 3 → 0.9951

符合「压缩到 \[-1, 1]」的效果。

---

### 🧠 在 Karpathy MLP 语言模型中怎么用？

在实现隐藏层时：

```python
h = torch.tanh(m.view(-1, 6) @ W1 + b1)
```

这一行作用是：

1. 先把输入 m 线性变换一下 `@ W1 + b1`
2. 然后通过 `tanh` 激活，非线性处理，输出到隐藏层

最终 h 的形状是 `[batch_size, 100]`，其中 100 是隐藏层神经元个数。

---

### ✅ 总结

| 名称       | 内容                 |
| -------- | ------------------ |
| 函数       | `torch.tanh`       |
| 作用       | 双曲正切激活函数           |
| 输入范围     | 任意实数               |
| 输出范围     | `[-1, 1]`          |
| 神经网络中的用途 | 提供非线性能力，帮助模型拟合复杂关系 |

---

如果你还想了解：

1. 为什么要用 `tanh` 而不用 ReLU？
2. `tanh` 和 `sigmoid`、`ReLU` 有什么区别？
3. 在什么场景更推荐用 `tanh`？

我也可以继续讲！需要的话告诉我 🌟。

---

什么是非线性映射？

好问题！我来用简单的语言解释一下**什么是非线性映射**，特别是在神经网络里，为什么非线性映射这么重要。

---

### 📐 先理解“线性映射”

**线性映射**，简单说就是符合这种规律的变换：

$$
y = a \cdot x + b
$$

比如：

* $y = 2x + 1$
* $y = -0.5x$

画出来是**直线**，关系非常简单，输入 x 改变多少，输出 y 按比例改变。

如果一个神经网络里，所有层都是线性的（只做矩阵乘法 + 偏置）：

```python
output = W @ x + b
```

不管你加多少层，整个网络其实还相当于一个整体的线性变换，表达能力有限，没法学复杂的模式（比如图片、语言里的复杂结构）。

---

### 🌈 非线性映射的定义

**非线性映射**，就是不符合“线性方程”规律的变换，不能用 $y = a \cdot x + b$ 来描述。

比如：

* $y = \tanh(x)$
* $y = \text{ReLU}(x) = \max(0, x)$
* $y = \sin(x)$
* $y = x^2$

这些函数的图像是**曲线**，而不是直线。

---

### 🤖 神经网络为什么需要非线性？

如果没有非线性，神经网络就算叠 100 层，最终还是一个大“线性层” —— 本质上和一层是一样的，没法拟合复杂数据。

引入非线性激活函数（比如 `tanh`、`ReLU`）后，网络才能“弯曲”输入空间，拟合复杂的映射关系，比如：

* 图像里的边缘/形状；
* 语言里的句法结构；
* 股票价格里的非线性趋势；
* 医学数据里的复杂模式。

一句话总结：

> **非线性映射 = 给神经网络“弯曲空间”的能力，才能学复杂东西。**

---

### 📊 举个例子

如果用纯线性神经网络（只有线性层）做二分类，面对这种数据：

```
   o o o o x x x x
```

可以。但如果数据是环形、交错的，纯线性网络就不行了，必须用非线性函数去“拉伸、扭曲”空间，才能分开。

---

### ✅ 总结

| 术语     | 含义                         |
| ------ | -------------------------- |
| 线性映射   | $y = ax + b$，不能拟合复杂关系      |
| 非线性映射  | 比如 `tanh`、`ReLU`，有能力拟合复杂数据 |
| 神经网络作用 | 需要非线性激活，才能学习复杂的、高阶的模式      |

---

如果你还想，我可以画一个图来演示「线性 vs 非线性」的区别，或者具体讲讲 `tanh`、`ReLU` 各自的优缺点，是否要继续？ 🌟

---

# implementing the output layer

so let's create w2 and v2
the input now is 100 and the output number of neurons will be for us 27 because we have 27 possible
characters that come next so the biases will be 27 as well
so therefore the logits which are the outputs of this neural net are going to be um
h multiplied by w2 plus b2
logistic shape is 32 by 27 and the logits look
good now exactly as we saw in the previous video we want to take these logits and we want to first exponentiate

当然可以！这段内容是在讲**输出层（output layer）的实现**，我给你分段解释：

---

## 📌 当前进度

* 输入是名字，每个字符用 2 维 embedding；
* 每个样本是 3 个字符 → 输入 `[32, 3, 2]` → 通过 view 变成 `[32, 6]`；
* 经过隐藏层（100 个神经元），得到：

```python
h = torch.tanh(m.view(-1, 6) @ W1 + b1)  # shape = [32, 100]
```

---

## 🎯 目标：实现输出层

```python
W2 = torch.randn(100, 27)
b2 = torch.randn(27)
```

解释：

* W2 是权重矩阵，大小是 `[100, 27]`，因为：

  * 输入是隐藏层的 100 个神经元；
  * 输出是 27 个可能的下一个字符（a-z + '.'）；
* b2 是偏置，大小是 `[27]`，每个输出神经元 1 个偏置。

---

### 输出层的计算

```python
logits = h @ W2 + b2  # shape = [32, 27]
```

解释：

* h 的 shape 是 `[32, 100]`，batch size 是 32；
* 乘以 W2 后得到 `[32, 27]`，也就是说：

  * 每个样本都会得到一个长度为 27 的向量；
  * 这个向量里的每一项，对应一个字符的“打分”；
* 这个向量叫 **logits**，是**未归一化的分数**，数值可以是正也可以是负。

---

## 🧠 logits 的作用

logits 是模型「还没归一化」的预测结果，下一步需要：

1. 对 logits **取指数**（exponentiate）：

   * 让所有数值变成正数；
2. 再用 **softmax** 归一化，让它变成**概率分布**（每一行加起来 = 1）：

```python
probs = torch.softmax(logits, dim=1)
```

* 这样，模型最终输出的 `probs` 就是：

  * 对于每个样本，27 个字符的概率预测；
  * 你可以采样、argmax 找出概率最大的字符，作为预测结果。

---

## ✅ 总结

| 步骤      | 说明                     |
| ------- | ---------------------- |
| W2      | `[100, 27]`，隐藏层到输出层的权重 |
| b2      | `[27]`，输出层偏置           |
| logits  | `[32, 27]`，未归一化打分      |
| softmax | 把 logits 转成概率分布        |
| 最终输出    | 每个样本 27 个字符的概率预测       |

---

如果你需要，我还可以继续讲「如何用 logits 和标签算交叉熵 loss」，或者继续解释 softmax 怎么工作，要不要？🌟

# implementing the negative log likelihood loss

them to get our fake counts and then we want to normalize them into a probability so prob is counts divide
and now counts dot sum along the first dimension and keep them as true exactly as in the
previous video and so prob that shape now is 32 by 27
and you'll see that every row of prob sums to one so it's normalized
so that gives us the probabilities now of course we have the actual letter that comes next and that comes from this
array y which we which we created during the dataset creation so why is this last
piece here which is the identity of the next character in the sequence that we'd like to now predict
so what we'd like to do now is just as in the previous video we'd like to index into the rows of prob and in each row
we'd like to pluck out the probability assigned to the correct character as given here
so first we have torch.range of 32 which is kind of like a iterator over
numbers from 0 to 31 and then we can index into prob in the following way
prop in torch.range of 32 which iterates the roads and in each row we'd like to grab
this column as given by y so this gives the current probabilities
as assigned by this neural network with this setting of its weights to the correct character in the sequence
and you can see here that this looks okay for some of these characters like this is basically 0.2
but it doesn't look very good at all for many other characters like this is 0.0701 probability and so the network
thinks that some of these are extremely unlikely but of course we haven't trained the neural network yet so
this will improve and ideally all of these numbers here of course are one because then we are correctly predicting
the next character now just as in the previous video we want to take these probabilities we want
to look at the lock probability and then we want to look at the average probability and the negative of it to create the
negative log likelihood loss so the loss here is 17
and this is the loss that we'd like to minimize to get the network to predict the correct character in the sequence
okay so i rewrote everything here and made it a bit more respectable so here's our data set here's all the parameters

当然可以！这段内容是在讲**如何实现 Negative Log Likelihood Loss（负对数似然损失，简称 NLL Loss）**，我帮你分段解释：

---

## 🎯 目标

我们现在有了：

* logits `[32, 27]` —— 神经网络的输出（还没归一化）
* y `[32]` —— 每个样本对应的正确“下一个字符”的索引

我们需要：

👉 **用 logits 算出损失（loss）**，用于训练神经网络。

---

## 🧠 步骤 1：把 logits 转换成概率（probabilities）

先把 logits 转换成「归一化后的概率」：

```python
probs = logits.exp()  # 先取指数，得到正数
probs /= probs.sum(dim=1, keepdim=True)  # 每行归一化成概率（softmax）
```

* probs 的形状还是 `[32, 27]`
* 每一行（一个样本）27 个字符的概率**加起来 = 1**

---

## 🧠 步骤 2：取出正确字符对应的概率

* 我们有 y（正确答案），每个 y\[i] 是 0 \~ 26 的一个整数，对应正确的下一个字符；
* 现在我们希望对每一行的 probs，取出“对应正确字符”的概率，写法是：

```python
probs[range(32), y]
```

解释：

* `range(32)` 遍历每一行（样本）；
* `y` 是列索引（每行中正确的那个字符的列）；
* 这样就取出了一个 `[32]` 向量，表示 32 个样本中，每个样本的「正确字符被模型分配到的概率」。

---

## 🧠 步骤 3：计算 NLL Loss

NLL Loss 是：

$$
\text{Loss} = -\frac{1}{N} \sum_{i=1}^N \log(p_i)
$$

也就是说：

* 先对每个样本的正确字符的概率 `p_i` 取对数；
* 再取负号、求平均，作为 Loss。

代码：

```python
loss = -torch.log(probs[range(32), y]).mean()
```

这样就得到了最终的 Loss。

---

## ✅ 为什么要用 NLL Loss？

* 我们的目标是：**希望模型把“正确字符”的概率 p\_i 提高到 1**，log(1)=0，Loss 越小越好；
* 如果模型预测正确，Loss → 0；
* 如果模型错得离谱，Loss 很大；
* Loss 是训练过程中需要优化的目标（用反向传播去优化网络参数）。

---

## 📊 现象解读

* 一开始网络是随机初始化，probs 里很多概率很低，比如 0.07、0.2 等，Loss 可能很大（例如 17）；
* 训练过程中，Loss 会不断下降，模型预测正确字符的概率会越来越高，Loss 越来越小。

---

## ✅ 总结

| 步骤                   | 说明                          |
| -------------------- | --------------------------- |
| logits               | 网络原始输出 `[32, 27]`           |
| probs                | softmax 归一化后概率 `[32, 27]`   |
| probs\[range(32), y] | 取出 32 个样本中，正确字符对应的概率 `[32]` |
| loss                 | 负对数似然损失，目标是最小化              |

---

如果你需要，我还可以继续解释：

1. **为什么 NLL Loss 是交叉熵（cross entropy）的一种特例？**
2. 用 PyTorch 的 `F.cross_entropy` 一行就能写完，怎么用？
3. NLL Loss 和 accuracy 有什么关系？

要不要继续？🌟

# summary of the full network

that we defined i'm now using a generator to make it reproducible i clustered all the parameters into a
single list of parameters so that for example it's easy to count them and see that in total we currently have about
3400 parameters and this is the forward pass as we developed it and we arrive at a single number here
the loss that is currently expressing how well this neural network works with the
current setting of parameters now i would like to make it even more respectable so in particular see these

当然可以！这一段是在做「整个神经网络的总结」，我帮你用中文解释一下：

---

## 📌 目前进度

到这里，我们已经把整个神经网络实现完了，包含：

1. 嵌入层（embedding layer）
2. 隐藏层（hidden layer）
3. 输出层（output layer）
4. 损失函数（negative log likelihood loss）

---

## 🎯 具体做了什么？

### ✅ 1️⃣ 让结果可复现

```python
generator = torch.Generator().manual_seed(42)
```

用一个随机数生成器 generator，并设置固定 seed（42），这样程序每次运行结果是一样的，方便调试和对比。

---

### ✅ 2️⃣ 把所有参数收集到一个列表

```python
parameters = [C, W1, b1, W2, b2]
```

这样做的好处：

* 以后可以统一管理参数（比如统一加 L2 正则、统一做优化器更新）；
* 也可以**统计总参数量**，看看模型大小。

比如：

```python
sum(p.nelement() for p in parameters)
# 当前总参数量大约 3400 个
```

---

### ✅ 3️⃣ 完整的前向传播（forward pass）

我们实现了完整的 forward pass：

```python
h = torch.tanh(m.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
loss = -torch.log(probs[range(32), y]).mean()
```

最终 `loss` 是一个数，衡量当前模型「拟合训练集的效果」。

---

### 🧠 目前模型效果

因为网络还没有训练，loss 是一个比较大的数（比如 17），预测的概率分布还很差。

接下来通过训练（梯度下降）会逐步优化参数，让 loss 下降，模型效果变好。

---

## ✅ 总结

| 步骤           | 说明                                 |
| ------------ | ---------------------------------- |
| 设置 generator | 固定随机种子，结果可复现                       |
| 收集参数         | 统一管理参数，方便统计和更新                     |
| forward pass | 嵌入 → 隐藏层 → 输出层 → softmax → 计算 loss |
| 当前状态         | 网络实现完成，还没训练，loss 较大                |

---

如果你需要，我也可以总结一下「整个 MLP 语言模型的完整流程图」或者「手写出完整的 forward pass + backward pass 代码」，要不要？ 🌟

# introducing F.cross_entropy and why

lines here where we take the logits and we calculate the loss
we're not actually reinventing the wheel here this is just um classification and many people use
classification and that's why there is a functional.cross entropy function in pytorch to calculate this much more
efficiently so we can just simply call f.cross entropy and we can pass in the logits and we can
pass in the array of targets y and this calculates the exact same loss
so in fact we can simply put this here and erase these three lines and we're
going to get the exact same result now there are actually many good reasons to prefer f.cross entropy over rolling your
own implementation like this i did this for educational reasons but you'd never use this in practice why is that
number one when you use f.cross entropy by torch will not actually create all these intermediate tensors because these
are all new tensors in memory and all this is fairly inefficient to run like this instead pytorch will cluster up all
these operations and very often create have fused kernels that very efficiently
evaluate these expressions that are sort of like clustered mathematical operations
number two the backward pass can be made much more efficient and not just because it's a fused kernel but also
analytically and mathematically it's much it's often a very much simpler backward pass to implement
we actually sell this with micrograd you see here when we implemented 10h the forward pass of this operation to
calculate the 10h was actually a fairly complicated mathematical expression but because it's a clustered
mathematical expression when we did the backward pass we didn't individually backward through the x and the two times
and the minus one in division etc we just said it's one minus t squared and that's a much simpler mathematical
expression and we were able to do this because we're able to reuse calculations and because we are able to mathematically
and analytically derive the derivative and often that expression simplifies mathematically and so there's much less
to implement so not only can can it be made more efficient because it runs in a fused kernel but also because the expressions
can take a much simpler form mathematically so that's number one number two
under the hood f that cross entropy can also be significantly more um
numerically well behaved let me show you an example of how this works
suppose we have a logits of negative 2 3 negative 3 0 and 5 and then we are taking the exponent of
it and normalizing it to sum to 1. so when logits take on this values everything is well and good and we get a
nice probability distribution now consider what happens when some of these logits take on more extreme values
and that can happen during optimization of the neural network suppose that some of these numbers grow
very negative like say negative 100 then actually everything will come out fine we still get the probabilities that
um you know are well behaved and they sum to one and everything is great
but because of the way the x works if you have very positive logits let's say positive 100 in here
you actually start to run into trouble and we get not a number here and the reason for that is that these
counts have an if here so if you pass in a very negative number
to x you just get a very negative sorry not negative but very small number very
very near zero and that's fine but if you pass in a very positive number suddenly we run out of range in
our floating point number that represents these counts so basically we're taking e and we're
raising it to the power of 100 and that gives us if because we run out of dynamic range on this floating point
number that is count and so we cannot pass very large logits
through this expression now let me reset these numbers to something reasonable
the way pi torch solved this is that you see how we have a well-behaved result here
it turns out that because of the normalization here you can actually offset logits by any arbitrary constant
value that you want so if i add 1 here you actually get the exact same result
or if i add 2 or if i subtract three any offset will produce the exact same
probabilities so because negative numbers are okay but positive numbers can actually overflow
this x what patrick does is it internally calculates the maximum value that occurs in the logits and it
subtracts it so in this case it would subtract five and so therefore the greatest number in
logits will become zero and all the other numbers will become some negative numbers and then the result of this is always
well behaved so even if we have 100 here previously not good but because pytorch will
subtract 100 this will work and so there's many good reasons to call
cross-entropy number one the forward pass can be much more efficient the backward pass can be much more efficient
and also things can be much more numerically well behaved okay so let's now set up the training of this neural

当然可以！这一段讲的是：**为什么在 PyTorch 里用 `F.cross_entropy()` 来计算 loss，优于自己手写 softmax + NLL Loss**。我帮你分段解释：

---

## 📌 当前进度

* 我们之前是「自己手写」了三行代码：

```python
probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
loss = -torch.log(probs[range(32), y]).mean()
```

* 这三行实现了 softmax + NLL Loss；
* 但是 PyTorch 提供了更专业的封装函数：

```python
F.cross_entropy(logits, y)
```

---

## 🎯 为什么用 `F.cross_entropy()` 更好？

---

### ✅ 1️⃣ 运行效率更高

* 手写代码中，每一步都会创建**中间张量**，消耗内存、运算慢；
* `F.cross_entropy()` 内部是「融合运算（fused kernel）」，更快、更省内存；
* 对大模型训练更友好。

---

### ✅ 2️⃣ 反向传播更快

* 手写版：自动微分系统要一层一层「反向传播」：

  * softmax
  * log
  * 取 index

* 这样反向图很复杂，速度慢，容易出错。

* `F.cross_entropy()` 内部是**数学推导过的简化版反向传播**，计算更快，数值更稳定，代码也更简洁。

举例：

```text
tanh 的反向传播，其实不是一层一层拆开计算，而是：
d(tanh) = 1 - tanh^2
```

同理，cross\_entropy 也可以简化。

---

### ✅ 3️⃣ 数值更稳定（numerically stable）

大重点 ❗：

* 当 logits 里出现「极大」的数（例如 100）时：

```python
probs = exp(100) / (exp(100) + ...)
```

* exp(100) 很容易**溢出**（超出浮点数范围，变成 NaN）；

如果你手写 softmax，这种溢出很容易发生，模型就崩了。

PyTorch 的 `F.cross_entropy()` 内部会自动做处理：

```text
softmax(x) = softmax(x - max(x))
```

* 先减去 logits 最大值，避免 exp 结果太大；
* 这样即使 logits 很大，也不会 NaN，数值始终稳定；

---

### 🔍 举例演示

* 如果 logits 中有正 100，手写版 softmax 会出问题，出现 NaN；
* 用 `F.cross_entropy()` 不会崩溃，结果稳定正常。

---

### ✅ 总结

| 优点    | 说明                     |
| ----- | ---------------------- |
| 速度快   | fused kernel，节省内存、加速   |
| 反向传播快 | 用数学推导的更简洁反向            |
| 数值稳定  | 自动处理 logits 极大/极小，避免溢出 |
| 简单易用  | 一行搞定，不用手写三行            |

---

### 🧑‍🏫 为什么一开始 Karpathy 手写？

* 纯粹是**教学目的**，帮助大家理解 softmax + NLL Loss 是怎么计算的；
* 实际工程中，应该**直接用 `F.cross_entropy()`**，性能和稳定性都更好。

---

如果你需要，我也可以：

1. 帮你画一张「softmax + NLL Loss」的流程图 🌈；
2. 解释 cross\_entropy 和 NLL Loss 的关系（很多人容易混淆）；
3. 举例讲 logits 极大的溢出案例 🚀。

需要的话可以随时告诉我！🌟

# implementing the training loop, overfitting one batch

net we have the forward pass uh we don't need these
is that we have the losses equal to the f.cross entropy that's the forward pass
then we need the backward pass first we want to set the gradients to be zero so for p in parameters
we want to make sure that p dot grad is none which is the same as setting it to zero in pi torch and then lost that backward to populate
those gradients once we have the gradients we can do the parameter update so for p in parameters
we want to take all the data and we want to nudge it learning rate times p dot grad
and then we want to repeat this a few times
and let's print the loss here as well now this won't suffice and it will
create an error because we also have to go for pn parameters and we have to make sure that p dot
requires grad is set to true in pi torch and this should just work
okay so we started off with loss of 17 and we're decreasing it let's run longer
and you see how the loss decreases a lot here so
if we just run for a thousand times we get a very very low loss and that means that we're making very good
predictions now the reason that this is so straightforward right now is because we're only um
overfitting 32 examples so we only have 32 examples uh of the
first five words and therefore it's very easy to make this neural net fit only these two 32
examples because we have 3 400 parameters and only 32 examples so we're
doing what's called overfitting a single batch of the data and getting a very low loss and good
predictions um but that's just because we have so many parameters for so few examples so it's easy to
uh make this be very low now we're not able to achieve exactly zero and the reason for that is we can for
example look at logits which are being predicted and we can look at the max along the
first dimension and in pi torch max reports both the actual values that
take on the maximum number but also the indices of piece and you'll see that the indices are very
close to the labels but in some cases they differ
for example in this very first example the predicted index is 19 but the label is five
and we're not able to make loss be zero and fundamentally that's because here
the very first or the zeroth index is the example where dot dot dot is supposed to predict e but you see how
dot dot dot is also supposed to predict an o and dot dot is also supposed to predict an i and then s as well and so
basically e o a or s are all possible outcomes in a training set for the exact
same input so we're not able to completely over fit and um and make the loss be exactly zero so but
we're getting very close in the cases where there's a unique input for a unique
output in those cases we do what's called overfit and we basically get the exact same and the exact correct result
so now all we have to do is we just need to make sure that we read in the full data set and optimize
the neural net okay so let's swing back up where we created the dataset and we see that here we only use the

当然可以！这一段在讲「如何实现训练循环（training loop）」，以及**用一个 batch 先做 overfitting 试试看网络能不能正常收敛**。我来帮你用中文分段解释：

---

## 📌 当前模型状态

* 前向传播（forward pass）已经做好了：

  ```python
  loss = F.cross_entropy(logits, y)
  ```

接下来就可以**训练神经网络**了，训练步骤包括：

1️⃣ **清零梯度**
2️⃣ **反向传播，计算梯度**
3️⃣ **更新参数（梯度下降）**
4️⃣ **重复很多轮（epoch）**

---

## 🧱 训练循环的实现

### ✅ 步骤 1：清零梯度

```python
for p in parameters:
    p.grad = None
```

PyTorch 里，优化之前**先把梯度清零**，否则梯度会累加。

---

### ✅ 步骤 2：反向传播

```python
loss.backward()
```

* 计算 loss 对各个参数的梯度；
* 梯度会存储到参数的 `.grad` 属性里。

---

### ✅ 步骤 3：更新参数

```python
for p in parameters:
    p.data -= learning_rate * p.grad
```

解释：

* `p.data` 是参数的值；
* 取当前梯度 `p.grad`，乘以学习率，做梯度下降（往 loss 变小的方向移动）；
* 这一步就是“学习”过程。

---

### ✅ 步骤 4：重复训练

通过循环不断训练：

```python
for i in range(1000):
    # forward pass
    # backward pass
    # 参数更新
    # 打印 loss
```

---

## 🎯 为什么先做 overfitting 一个 batch？

Karpathy 先只用\*\*32 个样本（来自前 5 个名字）\*\*做训练：

* 因为模型参数有 3400 个，数据只有 32 个样本，很容易拟合；
* 如果实现没问题，loss 会**迅速下降**；
* 用来检查网络实现是否正确 —— 一种常见的调试方法。

---

### 结果观察

* 训练 1000 轮，loss 从 17 → 非常小，说明模型能学；
* 有些样本 loss 不是 0，是因为：

  * 相同输入（比如 `...`）对应多个不同的 y（比如 e、o、a、s）；
  * 这种情况本来就不能完全拟合成 loss = 0，属于数据本身的特性。

---

## 🧠 overfitting 的意义

| 现象              | 解释                           |
| --------------- | ---------------------------- |
| loss 迅速下降       | 网络结构、forward + backward 没问题  |
| loss 接近 0，但不是 0 | 数据本身存在歧义                     |
| 仅 32 个样本能学好     | 网络有 3400 参数，拟合能力远大于样本量，容易过拟合 |

---

## ✅ 下一步

目前只是**overfit 一个 batch**，证明网络 OK。
接下来要：

* 用全数据集（32,000 个名字）训练；
* 设计 train/val/test split，防止过拟合，提升泛化能力。

---

## 🔔 小结

| 步骤                   | 说明              |
| -------------------- | --------------- |
| 清零梯度                 | 避免累积            |
| 反向传播                 | 计算 loss 对参数的梯度  |
| 更新参数                 | 梯度下降，优化 loss    |
| 训练 1000 轮            | loss 大幅下降，模型可以学 |
| overfitting 一个 batch | 调试用，确认网络实现正确    |

---

如果你想，我也可以继续解释：

1️⃣ 怎么选择 **学习率**（learning rate）
2️⃣ 怎么设计 **train/val/test split**
3️⃣ 怎么加 **mini-batch**，提升训练效率

要不要继续？🌟

# training on the full dataset, minibatches

first five words so let me now erase this and let me erase the print statements otherwise we'd be printing way too much
and so when we processed the full data set of all the words we now had 228 000 examples instead of just 32.
so let's now scroll back down to this is much larger reinitialize the weights the same number of parameters
they all require gradients and then let's push this print out lost.item to be here
and let's just see how the optimization goes if we run this
okay so we started with a fairly high loss and then as we're optimizing the loss is coming down
but you'll notice that it takes quite a bit of time for every single iteration so let's actually address that because
we're doing way too much work forwarding and backwarding 220 000 examples in practice what people usually do is
they perform forward and backward pass and update on many batches of the data
so what we will want to do is we want to randomly select some portion of the data set and that's a mini batch and then
only forward backward and update on that little mini batch and then we iterate on those many batches
so in pytorch we can for example use storage.randint we can generate numbers between 0 and 5
and make 32 of them i believe the size has to be a
tuple in my torch so we can have a tuple 32 of numbers
between zero and five but actually we want x dot shape of zero here and so this creates uh integers that
index into our data set and there's 32 of them so if our mini batch size is 32
then we can come here and we can first do a mini batch construct
so in the integers that we want to optimize in this single iteration
are in the ix and then we want to index into x
with ix to only grab those rows so we're only getting 32 rows of x
and therefore embeddings will again be 32 by three by two not two hundred thousand by three by two
and then this ix has to be used not just to index into x but also to index into y
and now this should be many batches and this should be much much faster so okay so it's instant almost
so this way we can run many many examples nearly instantly and decrease the loss
much much faster now because we're only dealing with mini batches the quality of our gradient is
lower so the direction is not as reliable it's not the actual gradient direction
but the gradient direction is good enough even when it's estimating on only 32 examples that it is useful and so
it's much better to have an approximate gradient and just make more steps than it is to evaluate the exact gradient and
take fewer steps so that's why in practice uh this works quite well
so let's now continue the optimization let me take out this lost item from here
and uh place it over here at the end okay so we're hovering around 2.5 or so
however this is only the loss for that mini batch so let's actually evaluate the loss here
for all of x and for all of y just so we have a full sense of exactly how all the model
is doing right now so right now we're at about 2.7 on the entire training set
so let's run the optimization for a while okay right 2.6
2.57 2.53
okay so one issue of course is we don't know if we're stepping too slow or too fast
so this point one i just guessed it so one question is how do you determine this learning rate
and how do we gain confidence that we're stepping in the right sort of speed so i'll show you one way

当然可以！这一段讲的是：

✅ 如何把训练从「一个 batch 过拟合」切换成**完整数据集 + mini-batch 训练**，提高训练效率。

我帮你分段解释：

---

## 📌 当前情况

* 刚刚我们只用 5 个名字、32 个样本过拟合，loss 下降很快；
* 现在我们要**用完整数据集**（全 32,000 个名字）训练模型。

---

## 🧠 处理完整数据集

* 把 `first 5 words` 这行代码删除；
* 用完整数据集生成训练样本，x、y 大小是：

```text
x.shape = [228,000, 3]
y.shape = [228,000]
```

* 也就是说，数据集现在有 22.8 万个训练样本。

---

## 🚧 问题来了

如果你直接对 22.8 万个样本做 forward + backward + 参数更新，每一轮迭代会很慢，效率低下。

---

## 🎯 解决方案：mini-batch 训练

**mini-batch** 的意思是：

> 每次只随机取一小批样本（比如 32 个），用来更新参数。

好处：

* 每次更新计算更快（forward + backward 快速完成）；
* 参数能不断学习新数据，不用等全量数据计算完；
* 训练更稳定，收敛效果更好。

---

### ✅ 如何实现 mini-batch

#### 1️⃣ 随机采样 mini-batch

```python
ix = torch.randint(0, x.shape[0], (32,))
```

解释：

* 随机生成 32 个整数，范围在 \[0, 228,000)
* 相当于在数据集中随机取 32 个样本

---

#### 2️⃣ 取出 mini-batch 数据

```python
x_batch = x[ix]
y_batch = y[ix]
```

* 取出这 32 个样本，对它们做 forward + backward + update；
* 这时 embedding 的 shape 就变回：

```text
[32, 3, 2]
```

---

### ✅ 3️⃣ 训练速度提升

用 mini-batch 后，训练速度几乎「瞬间」就可以跑完一次迭代，非常快！

---

## 🧠 mini-batch 的特性

* 用 mini-batch 训练，单次更新的梯度方向不再是「精确的」梯度（因为只看了 32 个样本）；
* 但它仍然是一个「合理的近似方向」，可以帮助 loss 持续下降；
* 在实际训练中，**频繁用小 batch 更新**，效果比「全量样本更新一次」更好。

---

## ✅ loss 观察

* 用 mini-batch 训练时，每次打印的 loss 只是**这个 batch 的 loss**；
* 如果你想知道「整体模型表现」，可以对整个数据集算一次 loss：

```python
loss_full = F.cross_entropy(forward(x), y)
```

例如，当前训练到全体数据集 loss ≈ 2.7，继续训练还能降到 2.6，2.53……

---

## 🧩 学习率问题

* 目前的学习率 (learning rate) 是 0.1，随便猜的；
* 不知道这个步长是太大还是太小，调整学习率非常关键；
* 下一段视频会讲如何寻找好的 learning rate。

---

## ✅ 总结

| 步骤            | 说明                               |
| ------------- | -------------------------------- |
| 完整数据集         | 22.8 万样本                         |
| 训练慢           | 如果直接训练全部，速度慢                     |
| mini-batch    | 每次随机选 32 个样本                     |
| 优点            | 训练快、参数更新频繁、loss 快速下降             |
| 观察            | batch 内 loss ≠ 全数据 loss，需要评估整体效果 |
| learning rate | 下一步要学会怎么调节                       |

---

如果你需要，我可以继续帮你讲：

1️⃣ **怎么选 batch\_size？**
2️⃣ **怎么用 learning rate finder？**
3️⃣ **为什么小 batch 会有更好的泛化能力？**

需要的话随时告诉我！🌟

---

torch.randint

解释一下

当然可以！

`torch.randint` 是 PyTorch 里一个**生成随机整数**的函数，作用是：

> **生成一个张量（tensor），里面的值是 \[low, high) 区间内的随机整数。**

---

### 📐 函数格式

```python
torch.randint(low, high, size)
```

### 参数说明：

| 参数   | 说明                          |
| ---- | --------------------------- |
| low  | 最小值（包含 low）                 |
| high | 最大值（不包含 high）               |
| size | 要生成的 tensor 的形状（用 tuple 表示） |

---

### 🧩 举例 1：生成 5 个 0\~9 的随机整数

```python
torch.randint(0, 10, (5,))
```

输出可能是：

```
tensor([7, 2, 4, 1, 8])
```

含义：生成了 5 个随机整数，范围是 \[0, 10) —— 也就是 0 到 9。

---

### 🧩 举例 2：生成一个 3x4 的随机整数矩阵

```python
torch.randint(0, 100, (3, 4))
```

输出可能是：

```
tensor([[53, 12, 79, 23],
        [34, 88,  9, 67],
        [45, 61, 25, 72]])
```

---

### 🧠 在 Karpathy 讲的训练循环里怎么用？

在 mini-batch 训练里：

```python
ix = torch.randint(0, x.shape[0], (32,))
```

意思是：

* 在数据集里，**随机选 32 个样本**；
* 取出的 32 个随机索引就放在 `ix` 这个 tensor 里。

---

### ✅ 总结

\| 功能 | 生成随机整数张量 |
\| 输入区间 | \[low, high) —— 包含 low，不包含 high |
\| 输出形状 | 用 size 参数指定，比如 (32,) 就是 1 维 32 个元素 |
\| 用途 | 随机采样、生成 mini-batch、数据增强等场景都用得到 |

---

如果你需要，我还可以讲：

1️⃣ `torch.randint` 和 `torch.rand`、`torch.randn` 的区别；
2️⃣ 为什么用 `randint` 比 for 循环随机采样更高效？

要不要继续？🌟

---

# finding a good initial learning rate

to determine a reasonable learning rate it works as follows let's reset our parameters
to the initial settings and now let's
print in every step but let's only do 10 steps or so
or maybe maybe 100 steps we want to find like a very reasonable set
search range if you will so for example if this is like very low then
we see that the loss is barely decreasing so that's not that's like too low basically so let's
try this one okay so we're decreasing the loss but like not very quickly so that's a pretty
good low range now let's reset it again and now let's try to find the place at
which the loss kind of explodes uh so maybe at negative one
okay we see that we're minimizing the loss but you see how uh it's kind of unstable it goes up and down quite a bit
um so negative one is probably like a fast learning rate let's try negative 10.
okay so this isn't optimizing this is not working very well so negative 10 is way too big
negative one was already kind of big um so therefore
negative one was like somewhat reasonable if i reset so i'm thinking that the right learning
rate is somewhere between uh negative zero point zero zero one and um
negative one so the way we can do this here is we can use uh torch shot lens space
and we want to basically do something like this between zero and one but um
those number of steps is one more parameter that's required let's do a thousand steps this creates 1000
numbers between 0.01 and 1 but it doesn't really make sense to step
between these linearly so instead let me create learning rate exponent
and instead of 0.001 this will be a negative 3 and this will be a zero
and then the actual lrs that we want to search over are going to be 10 to the power of lre
so now what we're doing is we're stepping linearly between the exponents of these learning rates this is 0.001
and this is 1 because 10 to the power of 0 is 1. and therefore we are spaced
exponentially in this interval so these are the candidate learning rates that we want to sort of like search over
roughly so now what we're going to do is here we are going to run the
optimization for 1000 steps and instead of using a fixed number we are going to use learning rate
indexing into here lrs of i and make this i
so basically let me reset this to be again starting from random creating these learning rates between
negative zero points between 0.001 and um one but exponentially stopped
and here what we're doing is we're iterating a thousand times we're going to use the learning rate
um that's in the beginning very very low in the beginning is going to be 0.001 but by the end it's going to be
1. and then we're going to step with that learning rate and now what we want to do is we want to
keep track of the uh
learning rates that we used and we want to look at the losses that resulted
and so here let me track stats so lri.append lr
and um lost side that append loss that item
okay so again reset everything and then run
and so basically we started with a very low learning rate and we went all the way up to a learning rate of negative one
and now what we can do is we can plt that plot and we can plot the two so we can plot
the learning rates on the x-axis and the losses we saw on the y-axis and often you're going to find that your
plot looks something like this where in the beginning you had very low learning rates so
basically anything barely anything happened then we got to like a nice spot here
and then as we increase the learning rate enough we basically started to be kind of unstable here
so a good learning rate turns out to be somewhere around here um and because we have lri here
um we actually may want to um
do not lr not the learning rate but the exponent so that would be the lre at i is maybe
what we want to log so let me reset this and redo that calculation but now on the x axis we have the
[Music] exponent of the learning rate and so we can see the exponent of the learning
rate that is good to use it would be sort of like roughly in the valley here because here the learning rates are just
way too low and then here where we expect relatively good learning rates somewhere here and then here things are
starting to explode so somewhere around negative one x the exponent of the learning rate is a pretty good setting
and 10 to the negative one is 0.1 so 0.1 is actually 0.1 was actually a fairly
good learning rate around here and that's what we had in the initial setting
but that's roughly how you would determine it and so here now we can take out the tracking of these
and we can just simply set lr to be 10 to the negative one or
basically otherwise 0.1 as it was before and now we have some confidence that this is actually a fairly good learning
rate and so now we can do is we can crank up the iterations we can reset our optimization
and we can run for a pretty long time using this learning rate
oops and we don't want to print that's way too much printing so let me again reset
and run ten thousand stops
okay so we're 0.2 2.48 roughly let's run another 10 000 steps
2.46 and now let's do one learning rate decay what this means is we're going to take
our learning rate and we're going to 10x lower it and so we're at the late stages of training potentially and we may want
to go a bit slower let's do one more actually at 0.1 just to see if
we're making a dent here okay we're still making dent and by the way the bi-gram loss that we achieved last video
was 2.45 so we've already surpassed the bi-gram model and once i get a sense that this is
actually kind of starting to plateau off people like to do as i mentioned this learning rate decay so let's try to
decay the loss the learning rate i mean
and we achieve it about 2.3 now obviously this is janky and not exactly
how you would train it in production but this is roughly what you're going through you first find a decent learning
rate using the approach that i showed you then you start with that learning rate and you train for a while
and then at the end people like to do a learning rate decay where you decay the learning rate by say a factor of 10 and
you do a few more steps and then you get a trained network roughly speaking so we've achieved 2.3 and dramatically
improved on the bi-gram language model using this simple neural net as described here
using these 3 400 parameters now there's something we have to be careful with

当然可以！这一段讲的是 **如何找到一个合适的初始学习率（learning rate）**，我来帮你用中文详细解释：

---

## 📌 背景

* 前面我们训练模型时，随便用了一个学习率 0.1；

* 但是 learning rate 很重要，选得太大或太小都不行：

  \| 学习率太小 | 学习太慢，loss 下降非常慢 |
  \| 学习率太大 | loss 会震荡、发散，甚至崩掉 |

* 所以我们需要「试一试」看看哪个学习率效果最好。

---

## 🧠 方法：learning rate sweep（扫描）

* 思路是：**让 learning rate 从很小的值慢慢增大**，看 loss 的变化趋势；
* 最后画一个图，找到「比较合适」的 learning rate 区间。

---

### ✅ 1️⃣ 重新初始化参数

```python
# 重新初始化参数，回到初始状态，保证公平
```

---

### ✅ 2️⃣ 每次迭代使用不同的 learning rate

* 先准备一个「学习率列表」，从 0.001 到 1，指数间隔：

```python
# lre = torch.linspace(-3, 0, 1000)  # -3 到 0，1000 个点
# lrs = 10 ** lre  # 转换成真实 learning rate：10^-3 到 10^0
```

所以：

```text
learning rates = [0.001, 0.0012, 0.0015, ..., 0.1, 0.5, 1.0]
```

---

### ✅ 3️⃣ 用不同 learning rate 训练 1000 步

```python
for i in range(1000):
    # 当前用的 learning rate
    lr = lrs[i]

    # forward pass
    # backward pass
    # 更新参数：p.data -= lr * p.grad
```

---

### ✅ 4️⃣ 记录 learning rate 和 loss

```python
lri.append(lr)
lossi.append(loss.item())
```

这样就能画出「learning rate vs. loss」的图。

---

## 📊 观察学习率 vs 损失图

* learning rate 很小时，loss 基本不降，模型没学到东西；
* learning rate 增大到一定范围时，loss 快速下降 —— 最佳区间；
* learning rate 太大时，loss 会震荡、爆炸（变大）。

---

### 🎨 最佳学习率区间

通常看图是「U 型」或「L 型」：

```text
Learning Rate    Loss
  ↓↓↓            ↓↓↓↓↓
 too low → OK → best → too high
```

比如：

```text
0.001  → 太慢
0.01   → 开始有效
0.1    → 很快下降
1.0    → 爆炸
```

选 **0.1** 比较合理，接下去就用这个学习率来训练模型。

---

## 🎯 最佳实践：Learning Rate Decay

训练到后期，可以「降低学习率」，更精细地调整模型参数，减少振荡：

```python
lr = lr * 0.1  # learning rate decay
```

每次学习率减 10 倍，继续训练几轮，loss 会继续下降。

---

## 📈 本次实验结果

* bi-gram model loss ≈ 2.45；
* 这个 MLP 训练后，loss ≈ 2.3，效果超过了 bi-gram。

---

## ✅ 总结

| 步骤    | 说明                                                      |
| ----- | ------------------------------------------------------- |
| 为什么做  | 找到合适的学习率                                                |
| 怎么做   | 用 exponentially spaced learning rates 训练 1000 步，记录 loss |
| 如何选   | 看图，选最稳定下降区间                                             |
| 学习率衰减 | 后期降低学习率，fine-tune 参数                                    |
| 最终结果  | MLP 效果超越 bi-gram                                        |

---

如果你需要，我还可以：

1️⃣ 帮你画出「learning rate vs loss 曲线」图；
2️⃣ 解释 **Learning Rate Finder** 算法背后的理论；
3️⃣ 讲讲为什么「学习率」对训练速度影响那么大 🚀。

要不要继续？🌟

---

当然可以！

`torch.linspace` 是 PyTorch 里一个非常常用的函数，作用是：

> **在一个区间 \[start, end] 之间，等间距地生成若干个数字**，组成一个 tensor。

---

### 📐 函数格式

```python
torch.linspace(start, end, steps)
```

| 参数    | 说明      |
| ----- | ------- |
| start | 起始值（包含） |
| end   | 结束值（包含） |
| steps | 要生成多少个数 |

---

### 🧩 举例 1：简单例子

```python
torch.linspace(0, 1, 5)
```

输出：

```
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

解释：

* 从 0 到 1，总共生成 5 个数字，**等间距分布**。

---

### 🧩 举例 2：用于 Learning Rate Finder

Karpathy 在调学习率时：

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre
```

解释：

* `lre` 生成 1000 个点，均匀分布在 \[-3, 0] 区间；
* 再做 `10^lre`，就变成：

```text
10^-3 = 0.001
10^-2.5 = 0.003
10^-2 = 0.01
...
10^0 = 1
```

* 也就是 **学习率 0.001 \~ 1 之间，呈指数间隔**，用于扫学习率效果。

---

### ✅ 总结

\| 功能 | 生成一个在 \[start, end] 区间内，等间距的数字序列 |
\| 典型用法 | 调整学习率，画曲线，生成坐标 |
\| 举例 | `torch.linspace(0, 10, 11)`  → `[0, 1, 2, ..., 10]` |
\| 和 `torch.arange` 区别 | `arange` 是「固定步长」，`linspace` 是「固定数量」 |

---

如果你需要，我还可以：

1️⃣ 讲讲 `linspace` 和 `arange` 区别；
2️⃣ 讲讲为什么 Learning Rate Finder 里要用 log-space（指数区间）而不是 linear 区间？

要不要继续？🌟

---


# splitting up the dataset into train/val/test splits and why

i said that we have a better model because we are achieving a lower loss 2.3 much lower than 2.45 with the
bi-gram model previously now that's not exactly true and the reason that's not true is that
this is actually fairly small model but these models can get larger and larger if you keep adding neurons and
parameters so you can imagine that we don't potentially have a thousand parameters we could have 10 000 or 100
000 or millions of parameters and as the capacity of the neural network grows
it becomes more and more capable of overfitting your training set what that means is that the loss on the
training set on the data that you're training on will become very very low as low as zero
but all that the model is doing is memorizing your training set verbatim so if you take that model and it looks like
it's working really well but you try to sample from it you will basically only get examples exactly as they are in the
training set you won't get any new data in addition to that if you try to evaluate the loss on some withheld names
or other words you will actually see that the loss on those can be very high and so basically
it's not a good model so the standard in the field is to split up your data set into three splits as we
call them we have the training split the dev split or the validation split and the test split
so training split test or um sorry dev or validation split
and test split and typically this would be say eighty percent of your data set this could be ten percent and this ten
percent roughly so you have these three splits of the data now these eighty percent of your
trainings of the data set the training set is used to optimize the parameters of the model just like we're doing here
using gradient descent these 10 percent of the examples the dev or validation split
they're used for development over all the hyper parameters of your model so hyper parameters are for example the
size of this hidden layer the size of the embedding so this is a hundred or a two for us but we could try
different things the strength of the regularization which we aren't using yet so far
so there's lots of different hybrid parameters and settings that go into defining your neural net and you can try many different variations of them and
see whichever one works best on your validation split so this is used to train the parameters
this is used to train the hyperprimers and test split is used to evaluate
basically the performance of the model at the end so we're only evaluating the loss on the test plate very very sparingly and very
few times because every single time you evaluate your test loss and you learn something from it
you are basically starting to also train on the test split so you are only allowed to test the loss
on a test set very very few times otherwise you risk
overfitting to it as well as you experiment on your model so let's also split up our training data
into train dev and test and then we are going to train on train and only evaluate on tests very very
sparingly okay so here we go here is where we took all the words and
put them into x and y tensors so instead let me create a new cell here and let me just copy paste some code
here because i don't think it's that complex but
we're going to try to save a little bit of time i'm converting this to be a function now and this function takes some list of
words and builds the arrays x and y for those words only
and then here i am shuffling up all the words so these are the input words that we get
we are randomly shuffling them all up and then um we're going to
set n1 to be the number of examples that there's 80 of the words and n2 to be
90 of the way of the words so basically if len of words is 32 000 n1 is
well sorry i should probably run this n1 is 25 000 and n2 is 28 000.
and so here we see that i'm calling build data set to build the training set x and y
by indexing into up to and one so we're going to have only 25 000 training words
and then we're going to have roughly n2 minus n1
3 3 000 validation examples or dev examples and we're going to have
when of words basically minus and two or 3 204 examples
here for the test set so now we have x's and y's
for all those three splits
oh yeah i'm printing their size here inside the function as well
but here we don't have words but these are already the individual examples made from those words
so let's now scroll down here and the data set now for training is
more like this and then when we reset the network
when we're training we're only going to be training using x train
x train and y train so that's the only thing we're training
on
let's see where we are on the single batch let's now train maybe a few more steps
training neural networks can take a while usually you don't do it inline you launch a bunch of jobs and you wait for them to finish um can take in multiple
days and so on luckily this is a very small network
okay so the loss is pretty good oh we accidentally used a learning rate that is way too low
so let me actually come back we use the decay learning rate of 0.01
so this will train much faster and then here when we evaluate let's use the dep set here
xdev and ydev to evaluate the loss okay
and let's now decay the learning rate and only do say 10 000 examples
and let's evaluate the dev loss ones here okay so we're getting about 2.3 on dev
and so the neural network when it was training did not see these dev examples it hasn't optimized on them and yet
when we evaluate the loss on these dev we actually get a pretty decent loss and so we can also look at what the
loss is on all of training set oops and so we see that the training and the
dev loss are about equal so we're not over fitting um this model is not powerful enough to
just be purely memorizing the data and so far we are what's called underfitting
because the training loss and the dev or test losses are roughly equal so what that typically means is that our network
is very tiny very small and we expect to make performance improvements by scaling
up the size of this neural net so let's do that now so let's come over here and let's increase the size of the

# experiment: larger hidden layer

neural net the easiest way to do this is we can come here to the hidden layer which currently has 100 neurons and
let's just bump this up so let's do 300 neurons and then this is also 300 biases and
here we have 300 inputs into the final layer so
let's initialize our neural net we now have ten thousand ex ten thousand parameters instead of three thousand parameters
and then we're not using this and then here what i'd like to do is i'd like to actually uh keep track of uh
tap um okay let's just do this let's keep stats
again and here when we're keeping track of the
loss let's just also keep track of the steps and let's just have i here
and let's train on thirty thousand or rather say
okay let's try thirty thousand and we are at point one
and we should be able to run this and optimize the neural net
and then here basically i want to plt.plot the steps
against the loss
so these are the x's and y's and this is the loss function and how it's being
optimized now you see that there's quite a bit of thickness to this and that's because we are optimizing over these mini batches
and the mini batches create a little bit of noise in this uh where are we in the def set we are at
2.5 so we still haven't optimized this neural net very well and that's probably because we made it
bigger it might take longer for this neural net to converge um and so let's continue training
um yeah let's just continue training
one possibility is that the batch size is so low that uh we just have way too much noise
in the training and we may want to increase the batch size so that we have a bit more um correct gradient and we're
not thrashing too much and we can actually like optimize more properly
okay this will now become meaningless because we've reinitialized these so
yeah this looks not pleasing right now but there probably is like a tiny improvement but it's so hard
to tell let's go again 2.52
let's try to decrease the learning rate by factor two
okay we're at 2.32 let's continue training
we basically expect to see a lower loss than what we had before because now we have a much much bigger model and we
were under fitting so we'd expect that increasing the size of the model should help the neural net
2.32 okay so that's not happening too well now one other concern is that even
though we've made the 10h layer here or the hidden layer much much bigger it could be that the bottleneck of the
network right now are these embeddings that are two dimensional it can be that we're just cramming way too many
characters into just two dimensions and the neural net is not able to really use that space effectively and that that is
sort of like the bottleneck to our network's performance okay 2.23 so just by decreasing the
learning rate i was able to make quite a bit of progress let's run this one more time
and then evaluate the training and the dev loss
now one more thing after training that i'd like to do is i'd like to visualize the um
embedding vectors for these characters before we scale up the
embedding size from two because we'd like to make uh this bottleneck potentially go away
but once i make this greater than two we won't be able to visualize them so here okay we're at 2.23 and 2.24
so um we're not improving much more and maybe the bottleneck now is the character embedding size which is two

# visualizing the character embeddings

so here i have a bunch of code that will create a figure and then we're going to visualize
the embeddings that were trained by the neural net on these characters because right now the embedding has just two so we can
visualize all the characters with the x and the y coordinates as the two embedding locations for each of these
characters and so here are the x coordinates and the y coordinates which are the columns
of c and then for each one i also include the text of the little character
so here what we see is actually kind of interesting the network has basically learned to
separate out the characters and cluster them a little bit uh so for example you see how the vowels
a e i o u are clustered up here so that's telling us that is that the neural net treats these is very similar
right because when they feed into the neural net the embedding uh for all these characters is very similar and so the
neural net thinks that they're very similar and kind of like interchangeable if that makes sense um
then the the points that are like really far away are for example q q is kind of treated as an exception and q has a very
special embedding vector so to speak similarly dot which is a special character is all the way out here
and a lot of the other letters are sort of like clustered up here and so it's kind of interesting that there's a
little bit of structure here after the training and it's not definitely not random and
these embeddings make sense so we're now going to scale up the embedding size and won't be able to
visualize it directly but we expect that because we're under fitting and we made this layer much bigger and
did not sufficiently improve the loss we're thinking that the um constraint to better performance right
now could be these embedding pictures so let's make them bigger okay so let's scroll up here

# experiment: larger embedding size

and now we don't have two dimensional embeddings we are going to have say 10 dimensional embeddings for each
word then this layer will receive 3 times 10 so 30
inputs will go into the hidden layer
let's also make the hidden layer a bit smaller so instead of 300 let's just do 200 neurons in that hidden layer
so now the total number of elements will be slightly bigger at 11 000
and then here we have to be a bit careful because um okay the learning rate we set to 0.1
here we are hardcoded in six and obviously if you're working in production you don't wanna be hard-coding magic numbers but instead of
six this should now be thirty um and let's run for fifty thousand
iterations and let me split out the initialization here outside
so that when we run this cell multiple times it's not going to wipe out our loss
in addition to that here let's instead of logging lost.item let's
actually log the let's do log 10
i believe that's a function of the loss and i'll show you why in a second let's
optimize this basically i'd like to plot the log loss instead of the loss because when you
plot the loss many times it can have this hockey stick appearance and log squashes it in
uh so it just kind of like looks nicer so the x-axis is step i and the y-axis will be the loss i
and then here this is 30. ideally we wouldn't be hard-coding these
okay so let's look at the loss okay it's again very thick because the mini batch size is very small but the
total loss over the training set is 2.3 and the the tests or the def set is 2.38
as well so so far so good uh let's try to now decrease the learning rate by a factor of 10
and train for another 50 000 iterations
we'd hope that we would be able to beat uh 2.32
but again we're just kind of like doing this very haphazardly so i don't actually have confidence that our
learning rate is set very well that our learning rate decay which we just do at random is set very well
and um so the optimization here is kind of suspect to be honest and this is not how you would do it typically in
production in production you would create parameters or hyper parameters out of all these settings and then you
would run lots of experiments and see whichever ones are working well for you
okay so we have 2.17 now and 2.2 okay so you
see how the training and the validation performance are starting to slightly slowly depart
so maybe we're getting the sense that the neural net is getting good enough or
that number of parameters is large enough that we are slowly starting to overfit
let's maybe run one more iteration of this and see where we get
but yeah basically you would be running lots of experiments and then you are slowly scrutinizing whichever ones give you the best depth performance and then
once you find all the hyper parameters that make your dev performance good you take that model and
you evaluate the test set performance a single time and that's the number that you report in your paper or wherever
else you want to talk about and brag about your model
so let's then rerun the plot and rerun the train and death
and because we're getting lower loss now it is the case that the embedding size of these was holding us back very likely
okay so 2.162.19 is what we're roughly getting so there's many ways to go from many
ways to go from here we can continue tuning the optimization we can continue for example playing with
the sizes of the neural net or we can increase the number of uh words or characters in our case that we
are taking as an input so instead of just three characters we could be taking more characters as an input and that
could further improve the loss okay so i changed the code slightly so we have here 200 000 steps of the

# summary of our final code, conclusion

optimization and in the first 100 000 we're using a learning rate of 0.1 and then in the next 100 000 we're using a
learning rate of 0.01 this is the loss that i achieve and these are the performance on the
training and validation loss and in particular the best validation loss i've been able to obtain in the
last 30 minutes or so is 2.17 so now i invite you to beat this number
and you have quite a few knobs available to you to i think surpass this number so number one you can of course change
the number of neurons in the hidden layer of this model you can change the dimensionality of the embedding
lookup table you can change the number of characters that are feeding in as an input
as the context into this model and then of course you can change the details of the optimization how long are
we running what is the learning rate how does it change over time how does it decay
you can change the batch size and you may be able to actually achieve a much better convergence speed
in terms of how many seconds or minutes it takes to train the model and get
your result in terms of really good loss and then of course i actually invite you
to read this paper it is 19 pages but at this point you should actually be able to read a good chunk of this paper and
understand pretty good chunks of it and this paper also has quite a few ideas for improvements that you can play
with so all of those are not available to you and you should be able to beat this number i'm leaving that as an exercise
to the reader and that's it for now and i'll see you next time

# sampling from the model

before we wrap up i also wanted to show how you would sample from the model so we're going to generate 20 samples
at first we begin with all dots so that's the context and then until we generate
the zeroth character again we're going to embed the current context
using the embedding table c now usually uh here the first dimension was the size
of the training set but here we're only working with a single example that we're generating so this is just the mission
one just for simplicity and so this embedding then gets
projected into the end state you get the logits now we calculate the probabilities for
that you can use f.softmax of logits and that just basically
exponentiates the logits and makes them sum to one and similar to cross entropy it is careful that there's no overflows
once we have the probabilities we sample from them using torture multinomial to get our next index and then we shift the
context window to append the index and record it and then we can just
decode all the integers to strings and print them out and so these are some example samples
and you can see that the model now works much better so the words here are much more word like or name like so we have
things like ham joes
you know it's starting to sound a little bit more name-like so we're definitely making progress but we can still improve
on this model quite a lot okay sorry there's some bonus content i wanted to mention that i want to make

# google collab (new!!) notebook advertisement

these notebooks more accessible and so i don't want you to have to like install jupyter notebooks and torch and
everything else so i will be sharing a link to a google colab and google collab will look like a
notebook in your browser and you can just go to the url and you'll be able to execute all of the code that you saw in
the google collab and so this is me executing the code in this lecture and i shortened it a little bit but basically
you're able to train the exact same network and then plot and sample from the model and everything is ready for
you to like tinker with the numbers right there in your browser no installation necessary
so i just wanted to point that out and the link to this will be in the video description