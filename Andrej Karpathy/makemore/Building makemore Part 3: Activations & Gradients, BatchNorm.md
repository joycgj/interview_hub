We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization. Residual connections and the Adam optimizer remain notable todos for later video.

Links:
- makemore on github: https://github.com/karpathy/makemore
- jupyter notebook I built in this video: https://github.com/karpathy/nn-zero-t...
- collab notebook: https://colab.research.google.com/dri...
- my website: https://karpathy.ai
- my twitter:   / karpathy  
- Discord channel:   / discord  

Useful links:
- "Kaiming init" paper: https://arxiv.org/abs/1502.01852
- BatchNorm paper: https://arxiv.org/abs/1502.03167
- Bengio et al. 2003 MLP language model paper (pdf): https://www.jmlr.org/papers/volume3/b...
- Good paper illustrating some of the problems with batchnorm in practice: https://arxiv.org/abs/2105.07576

Exercises:
- E01: I did not get around to seeing what happens when you initialize all weights and biases to zero. Try this and train the neural net. You might think either that 1) the network trains just fine or 2) the network doesn't train at all, but actually it is 3) the network trains but only partially, and achieves a pretty bad final performance. Inspect the gradients and activations to figure out what is happening and why the network is only partially training, and what part is being trained exactly.
- E02: BatchNorm, unlike other normalization layers like LayerNorm/GroupNorm etc. has the big advantage that after training, the batchnorm gamma/beta can be "folded into" the weights of the preceeding Linear layers, effectively erasing the need to forward it at test time. Set up a small 3-layer MLP with batchnorms, train the network, then "fold" the batchnorm gamma/beta into the preceeding Linear layer's W,b by creating a new W2, b2 and erasing the batch norm. Verify that this gives the same forward pass during inference. i.e. we see that the batchnorm is there just for stabilizing the training, and can be thrown out after training is done! pretty cool.

Chapters:
- 00:00:00 intro
- 00:01:22 starter code
- 00:04:19 fixing the initial loss 
- 00:12:59 fixing the saturated tanh
- 00:27:53 calculating the init scale: “Kaiming init”
- 00:40:40 batch normalization
- 01:03:07 batch normalization: summary
- 01:04:50 real example: resnet50 walkthrough
- 01:14:10 summary of the lecture
- 01:18:35 just kidding: part2: PyTorch-ifying the code
- 01:26:51 viz #1: forward pass activations statistics
- 01:30:54 viz #2: backward pass gradient statistics
- 01:32:07 the fully linear case of no non-linearities
- 01:36:15 viz #3: parameter activation and gradient statistics
- 01:39:55 viz #4: update:data ratio over time
- 01:46:04 bringing back batchnorm, looking at the visualizations
- 01:51:34 summary of the lecture for real this time

当然可以，下面是这段介绍的中文翻译：

---

**Building makemore 第 3 部分：激活 & 梯度，BatchNorm**

我们深入探讨了多层 MLP（多层感知机）的内部机制，仔细分析了前向传播中的激活统计、反向传播中的梯度统计，以及当这些值缩放不当时会出现的一些问题。视频中还介绍了你通常会用到的诊断工具和可视化方法，帮助理解神经网络训练的健康状态。我们了解到为什么训练深度神经网络会比较脆弱，并介绍了第一个极大改善训练过程的现代技术：**批量归一化（Batch Normalization）**。
（残差连接（Residual connections）和 Adam 优化器将在后续视频中介绍。）

**相关链接：**

* makemore 代码仓库：[https://github.com/karpathy/makemore](https://github.com/karpathy/makemore)
* 本视频用到的 Jupyter notebook: [https://github.com/karpathy/nn-zero-t](https://github.com/karpathy/nn-zero-t)...
* Colab notebook: [https://colab.research.google.com/dri](https://colab.research.google.com/dri)...
* 我的个人网站：[https://karpathy.ai](https://karpathy.ai)
* 我的 Twitter： / karpathy
* Discord 频道： / discord

**参考论文：**

* “Kaiming 初始化” 论文：[https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
* BatchNorm 论文：[https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
* Bengio 等人 2003 年 MLP 语言模型论文（pdf）：[https://www.jmlr.org/papers/volume3/b](https://www.jmlr.org/papers/volume3/b)...
* 实际应用中 BatchNorm 存在的一些问题：[https://arxiv.org/abs/2105.07576](https://arxiv.org/abs/2105.07576)

**练习题：**

* **E01:** 如果将神经网络的所有权重和偏置都初始化为 0，训练效果会怎样？请尝试训练网络，你可能会猜测 1）网络正常训练，或 2）网络完全无法训练，但实际上是 3）网络部分训练，最终性能较差。分析梯度和激活，理解为何会这样，哪些部分被训练了，哪些没有。
* **E02:** BatchNorm 不同于 LayerNorm、GroupNorm 的一点是，在训练完成后，BatchNorm 中的 gamma 和 beta 参数可以“折叠”进前一层 Linear 层的权重和偏置中，这样在推理时就可以不用再单独计算 BatchNorm。试着构建一个 3 层 MLP，使用 BatchNorm，训练好之后把 BatchNorm “折叠”进 Linear 层，验证前向传播是否一致，从而证明 BatchNorm 主要作用在于训练时的稳定性，训练后可以丢弃，挺酷吧。

**视频章节：**
- 00:00:00 介绍
- 00:01:22 初始代码
- 00:04:19 修正初始损失
- 00:12:59 修正 tanh 饱和问题
- 00:27:53 计算初始化缩放比例（Kaiming init）
- 00:40:40 批量归一化（BatchNorm）
- 01:03:07 BatchNorm 总结
- 01:04:50 实例：ResNet50 结构解析
- 01:14:10 讲解总结
- 01:18:35 彩蛋：将代码改成 PyTorch 风格
- 01:26:51 可视化 #1：前向传播激活的统计信息
- 01:30:54 可视化 #2：反向传播梯度的统计信息
- 01:32:07 纯线性（无激活函数）情况分析
- 01:36:15 可视化 #3：参数激活值和梯度统计信息
- 01:39:55 可视化 #4：更新与数据比例随时间变化
- 01:46:04 重新引入 BatchNorm，观察可视化效果
- 01:51:34 这次是真正的讲解总结

---

如果你还需要，我可以帮你把重点内容再做个**简明版笔记**，方便复习 👌。要不要？

# intro

hi everyone today we are continuing our implementation of make more now in the last lecture we implemented the multier
perceptron along the lines of benj 2003 for character level language modeling so we followed this paper took in a few
characters in the past and used an MLP to predict the next character in a sequence so what we'd like to do now is
we'd like to move on to more complex and larger neural networks like recurrent neural networks and their variations
like the grw lstm and so on now before we do that though we have to stick around the level of malalia perception
on for a bit longer and I'd like to do this because I would like us to have a very good intuitive understanding of the
activations in the neural net during training and especially the gradients that are flowing backwards and how they
behave and what they look like and this is going to be very important to understand the history of the development of these architectures
because we'll see that recurr neural networks while they are very expressive in that they are a universal
approximator and can in principle Implement uh all the algorithms uh we'll see that they are not very easily
optimizable with the first order gradient based techniques that we have available to us and that we use all the time and the key to understanding why
they are not optimizable easily is to understand the the activations and the gradients and how they behave during
training and we'll see that a lot of the variants since recur neural networks have tried to improve that situation and
so that's the path that we have to take and uh let's get started so the starting code for this lecture is largely the

当然可以，下面是这段 **intro** 部分的中文解释：

---

大家好，今天我们继续来实现 *makemore*。
在上一讲中，我们按照 Bengio 2003 年的论文，实现了一个多层感知机（MLP），用于字符级语言建模：输入一串字符，用 MLP 预测下一个字符。

接下来，我们希望向更复杂、更强大的神经网络迈进，比如循环神经网络（RNN）及其各种变体（如 GRU、LSTM 等）。
不过，在正式进入 RNN 之前，我们还需要在 MLP 这个阶段多花一点时间，原因是：**我们需要对神经网络中的“激活值”（activations）和“反向传播中的梯度”（gradients）形成非常直观、深入的理解**。

了解这些内容非常重要，因为它们贯穿了神经网络架构的发展历程。
尽管 RNN 理论上非常强大，是**通用逼近器**，可以实现任何算法，但实际上它们用我们现在常用的**一阶梯度优化算法**并不好训练。

理解为什么 RNN 很难训练，关键就在于搞清楚**训练过程中激活值和梯度是如何表现的**。
你会看到，后来的很多 RNN 变种，正是为了改善这一问题而提出的。

所以今天我们就沿着这条路径继续深入，首先来看看代码吧。
这一讲用到的起始代码，基本和之前讲解的差不多。

---

如果需要，我也可以帮你归纳一个 **逻辑脉络图**，让学习路线更清晰，要不要？ 🌟


# starter code

code from before but I've cleaned it up a little bit so you'll see that we are importing all the torch and math plb utilities
we're reading in the words just like before these are eight example words there's a total of 32,000 of them here's
a vocabulary of all the lowercase letters and the special dot token here
we are reading the data set and processing it and um creating three
splits the train Dev and the test split now in MLP this is the identical same
MLP except you see that I removed a bunch of magic numbers that we had here and instead we have the dimensionality
of the embedding space of the characters and the number of hidden units in the hidden layer and so I've pulled them
outside here uh so that we don't have to go and change all these magic numbers all the time we have the same neural net
with 11,000 parameters that we optimize now over 200,000 steps with a batch size of 32 and you'll see that I refactor I
refactored the code here a little bit but there are no functional changes I just created a few extra variables a few
more comments and I removed all the magic numbers and otherwise is the exact same thing then when we optimize we saw
that our loss looked something like this we saw that the train and Val loss were about
2.16 and so on here I refactored the uh code a little bit for the evaluation of
arbitary splits so you pass in a string of which split you'd like to evaluate and then here depending on train Val or
test I index in and I get the correct split and then this is the forward pass of the network and evaluation of the
loss and printing it so just making that nicer uh one thing that you'll notice
here is I'm using a decorator torch. nograd which you can also um look up and
read the documentation of basically what this decorator does on top of a function is that whatever happens in this
function is assumed by uh torch to never require any gradients so it will not do
any of the bookkeeping that it does to keep track of all the gradients in anticipation of an eventual backward
pass it's it's almost as if all the tensors that get created here have a required grad of false and so it just
makes everything much more efficient because you're telling torch that I will not call that backward on any of this computation and you don't need to
maintain the graph under the hood so that's what this does and you can also
use a context manager uh with torch du nograd and you can look those
up then here we have the sampling from a model um just as before just a for
Passive neural nut getting the distribution sent from it adjusting the context window and repeating until we
get the special end token and we see that we are starting to get much nicer looking words simple from the model it's
still not amazing and they're still not fully name like uh but it's much better than what we had with the BAM
model so that's our starting point now the first thing I would like to scrutinize is the initialization I can tell that our

当然可以，下面是这一段 **starter code** 部分的中文解释：

---

这段代码其实是之前写的版本，我稍微做了一些整理和清理：
首先，和以前一样，我们导入了 torch、matplotlib、math 等工具库，
读取单词数据，例子里是 8 个单词，总共数据集中有 32,000 个单词，
词表（vocabulary）是小写字母 + 特殊的结束符号 `.`。

数据集读入后，我们做了数据处理，分成三份：**训练集（train）**、**开发集（dev）** 和 **测试集（test）**。

MLP 的结构和之前实现的一样，不过我把之前代码里的一些“魔法数字（magic numbers）”拿掉了，
比如字符的嵌入维度（embedding size）和隐藏层的隐藏单元数（hidden units），
这些参数我统一放到前面来定义，后面就不用到处改数字了，
这样结构更清晰，调参也方便。

整体上，神经网络的参数量大概 11,000 个，
我们优化 200,000 个步骤，batch size = 32。
我对代码做了一些重构，增加了一些变量、注释，去掉魔法数字，**功能上没变化**，只是更整洁了。

训练过程中，我们之前看到训练集和验证集的 loss 大概在 2.16 左右。

我还稍微整理了评估不同数据集（train/val/test）的时候用的代码，
现在可以传入一个字符串参数，指定要评估哪个数据集，
代码内部会自动选对应的数据，forward 一次网络，计算 loss 并打印。

还有一个细节：
你会注意到我用了 `@torch.no_grad()` 这个 **装饰器**，
它的作用是告诉 torch：**这个函数里的操作不需要计算梯度**，
这样 torch 不会在后台记录用于反向传播的计算图，能节省内存和计算资源。
其实和把所有 tensor 的 `requires_grad=False` 效果差不多。
当然也可以用 `with torch.no_grad():` 这个 **上下文管理器**，作用是一样的。

最后，采样代码（sample）还是和之前一样：
从神经网络里拿到分布，更新上下文窗口，循环直到遇到结束符号 `.`，
现在采样出来的单词，**比之前 Bigram 模型好很多**，虽然还不能算是真正像“名字”的单词，但效果明显提升了。

所以目前的起点是这样，接下来我们要先仔细分析一下 **初始化（initialization）**，
我可以告诉你，我们当前的初始化……

---

如果需要我还可以帮你整理 **这一段代码的关键改动点**，或者帮你写一份**中文注释版 starter code**，要不要？ 🚀

---

chars = sorted(list(set(''.join(words))))

解释一下

当然，来解释一下：

```python
chars = sorted(list(set(''.join(words))))
```

这一句代码的作用是 **提取出数据集中出现过的所有字符，放入一个有序的列表里**，详细拆解如下：

1. `''.join(words)`

   * 把 `words` 这个列表里所有的单词，拼成一个长字符串。
   * 例如：

     ```python
     words = ['hello', 'world']  
     ''.join(words) -> 'helloworld'
     ```

2. `set('helloworld')`

   * 把这个字符串转换成 **集合（set）**，集合的特点是 **去重**，所以只留下**出现过的不同字符**：

     ```python
     set('helloworld') -> {'h', 'e', 'l', 'o', 'w', 'r', 'd'}
     ```

3. `list(set(...))`

   * 把这个集合转回 **列表（list）**，因为后面我们通常需要用列表来索引字符。

4. `sorted(...)`

   * 最后对这个列表 **排序**，确保字符顺序固定（通常按字母表顺序排序），这样后续编码/解码字符时索引会稳定。

**最终结果**：`chars` 就是一个 **包含所有在单词列表里出现过的字符** 的 **有序列表**，
比如：

```python
chars = ['a', 'b', 'c', ..., 'z', '.']
```

这个 `chars` 后面就会用来建立 **字符到索引**、**索引到字符** 的映射，用于 embedding 层的输入处理。

---

如果你还想，我可以帮你画一个小图，直观展示这段代码的“流水线”过程，是否需要？ 🌈

---

```python
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y
```

解释一下

---

当然可以，来详细解释这一段 **build\_dataset** 代码的作用：

---

首先，这一段代码是为了 **把原始单词数据变成神经网络能用的训练数据 X 和标签 Y**。

### 关键变量：

```python
block_size = 3
```

* 意思是：**上下文窗口大小为 3**，也就是说，**用前 3 个字符来预测下一个字符**。
  （注意：如果单词不够长，窗口前面用 0 填充）

---

### 函数：

```python
def build_dataset(words):
    X, Y = [], []  # X 是输入，Y 是标签
```

---

### 主循环：

```python
for w in words:
    context = [0] * block_size  # 初始化 context，开始时 3 个 0
    for ch in w + '.':          # 每个单词末尾加一个 '.' 作为终止符
        ix = stoi[ch]           # 把字符 ch 转换成整数索引
        X.append(context)       # 把当前 context 加到 X
        Y.append(ix)            # 下一个要预测的字符索引 ix 加到 Y
        context = context[1:] + [ix]  # 更新 context：右移一格，加入当前字符
```

解释一下 **这一层 for 循环**发生了什么：

* 对于每个单词，初始 context 是 `[0, 0, 0]`
* 然后从单词的第一个字符开始，循环：

  * 当前 context 作为“输入” X
  * 当前字符（转成索引 ix）作为“标签” Y
  * context 窗口右移 1 格，加上当前字符，继续下一个字符

用一句话总结就是：**用过去 3 个字符，预测下一个字符**。

---

### 转成 Tensor：

```python
X = torch.tensor(X)
Y = torch.tensor(Y)
print(X.shape, Y.shape)
return X, Y
```

* 把 X、Y 转成 **torch tensor**，方便后续喂入模型训练。
* X.shape: 多少个样本 × 3（窗口大小）
* Y.shape: 多少个标签（和 X 的第一个维度相同）

---

### 一个简单例子：

假设 `block_size=3`，词是 `"abc"`，带上结束符 `"."`，变成 `"abc."`，过程是：

| context                      | 当前字符（ch） | 输出 X                         | 输出 Y（预测目标） |
| ---------------------------- | -------- | ---------------------------- | ---------- |
| \[0, 0, 0]                   | 'a'      | \[0, 0, 0]                   | ix('a')    |
| \[0, 0, ix('a')]             | 'b'      | \[0, 0, ix('a')]             | ix('b')    |
| \[0, ix('a'), ix('b')]       | 'c'      | \[0, ix('a'), ix('b')]       | ix('c')    |
| \[ix('a'), ix('b'), ix('c')] | '.'      | \[ix('a'), ix('b'), ix('c')] | ix('.')    |

---

总结：

* `build_dataset` 的目的是构造\*\*“前 3 个字符 -> 下一个字符”\*\*的训练数据对 `(X, Y)`
* 这样神经网络学到的就是：**根据过去 N 个字符，预测下一个字符的概率分布**

---

如果你需要的话，我也可以帮你画一个 **可视化流程图**，让这个过程更直观，是否需要？ 🌟

---

```python
# MLP revisited
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                        generator=g)
W2 = torch.randn((n_hidden, vocab_size),          generator=g)
b2 = torch.randn(vocab_size,                      generator=g)

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

解释一下

当然可以，来详细解释这一段 **MLP revisited** 代码的作用：

---

### 定义超参数：

```python
n_embd = 10  # 每个字符 embedding 向量的维度，10 维
n_hidden = 200  # MLP 隐藏层的神经元数量，200 个
```

解释：

* 每个字符（a-z 和 '.'）会被映射成一个 10 维的向量（embedding）
* MLP 的隐藏层有 200 个神经元

---

### 设置随机数种子：

```python
g = torch.Generator().manual_seed(2147483647)
```

* 用来确保 **结果可复现**，固定随机数种子，训练多次结果一致。

---

### 初始化模型参数：

```python
C  = torch.randn((vocab_size, n_embd), generator=g)
```

* `C` 是 **字符 embedding 矩阵**，大小 `[vocab_size, n_embd]`
* 每个字符都会映射成一个 n\_embd 维向量，初始化用高斯分布（标准正态）

---

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
```

* `W1` 是第一层全连接层（Linear Layer）的权重矩阵
* 输入维度是 `n_embd * block_size`

  * 因为我们用 `block_size` 个字符，每个字符是 n\_embd 维
  * 所以一共 `n_embd * block_size` 个输入
* `b1` 是第 1 层的 bias（偏置）

---

```python
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)
```

* `W2` 是第 2 层（隐藏层 -> 输出层） 的权重矩阵
* 输出层维度是 `vocab_size`，表示对每一个字符的概率预测
* `b2` 是输出层的偏置

---

### 汇总参数：

```python
parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters))
```

* 把所有模型参数放到 `parameters` 列表
* 计算一下总共有多少个参数（nelement 是元素个数）

---

### 设置 requires\_grad：

```python
for p in parameters:
    p.requires_grad = True
```

* 把所有参数设置为 `requires_grad=True`，这样后面可以用 **反向传播计算梯度**，做优化更新。

---

### 整体结构图：

```
字符 → embedding（10维） → 拼接成 block_size * n_embd →  
→ MLP 隐藏层（200 neurons） → 输出层（vocab_size 维，softmax，预测下一个字符）
```

---

### 总结：

这一段代码做了几件事：

1️⃣ 定义 embedding 大小 和 隐藏层大小
2️⃣ 初始化所有权重和偏置参数（正态分布）
3️⃣ 设置参数能求梯度（用于训练）
4️⃣ 打印总参数数量，方便了解模型大小

---

如果需要的话，我也可以帮你画一张 **模型结构图（MLP 版）**，或者写一个**中文注释版**的代码，是否需要？ 🚀

---

W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)

为什么是n_embd * block_size

---

这个问题问得非常好！👍
我们一步步来拆解 **为什么是 `n_embd * block_size`**：

---

### 1️⃣ 输入数据长什么样？

回忆一下前面我们 build\_dataset() 做了什么：

```python
block_size = 3
```

也就是说，我们的模型是用 **3 个字符的上下文（context）去预测下一个字符**，
例如：
输入：\[ 'a', 'b', 'c' ] → 预测：'d'？

---

### 2️⃣ 每个字符如何表示？

我们没有直接用 one-hot，
而是用了 **embedding 向量** 来表示每个字符：

```python
n_embd = 10  # 每个字符 -> 10 维向量
```

也就是说，每个字符变成一个 10 维向量，
如果 context 里有 3 个字符，
那总共就是 **3 个 embedding 向量**，也就是：

```
[10 维向量]， [10 维向量]， [10 维向量]  →  拼接成一个 30 维向量
```

---

### 3️⃣ 为什么拼接？

因为我们要把整个 context（3 个字符）作为一整个输入，
MLP 的输入是一个一维向量，
所以需要把这 3 个 10 维向量拼在一起，形成：

```
[10, 10, 10] → [30]
```

也就是：**n\_embd \* block\_size = 10 \* 3 = 30 维输入向量**

---

### 4️⃣ 为什么 W1 的形状是这样？

MLP 第一层是一个 **Linear 层**，
输入是 30 维，输出是隐藏层大小 n\_hidden=200，
所以权重矩阵 W1 应该是：

```
W1.shape = [输入维度, 输出维度]  
           = [n_embd * block_size, n_hidden]  
           = [30, 200]
```

这样矩阵乘法：

```
[batch_size, 30]  @  [30, 200]  →  [batch_size, 200]
```

---

### 5️⃣ 总结一句话：

`n_embd * block_size` 是因为：

* 每个字符 embedding 是 n\_embd 维
* context 里有 block\_size 个字符
* 要把这几个 embedding 拼接成一个长向量，作为 MLP 的输入

所以：

```python
输入维度 = n_embd * block_size
```

---

如果你想，我还可以给你画一张 **输入向量流向隐藏层的示意图**，是不是更直观？ 🚀✨

---

C: 27 * 10
tensor([[ 1.5328, -0.2343,  0.2040, -1.1125,  0.3263,  0.0561, -1.5299,  0.6566, 0.1745,  0.9202],
        [-0.4883,  0.8106, -0.4130, -0.3914,  1.3625,  2.8784,  1.5473, -1.6390, 0.6382, -0.8814],
        [ 0.9949, -0.1486, -1.4684,  0.4401, -0.0719,  2.5041,  2.4447, -0.6695, -1.2162,  0.3082],
        [-1.0945,  0.7357,  0.0343,  1.3114, -0.7974, -0.8282, -1.8212,  1.2437, -1.2517,  1.2082],
        [-0.9440, -0.2305, -0.3531,  0.3200, -1.3266,  1.1077,  0.5951,  0.4612, 0.0316, -1.7605],
        [ 0.0659,  0.7843,  0.5342, -1.1757,  0.1580, -0.9820,  0.4185,  0.4760, -0.9080, -0.7950],
        [ 0.1351, -0.2116, -1.0406, -1.5367,  0.9374, -0.8830,  1.7457,  2.1346, -0.8561,  0.5408],
        [ 0.6174,  1.5029, -1.0476, -0.6548, -0.7283,  1.7525,  0.1843,  0.9789, -0.6254,  0.0746],
        [-1.1798,  1.2718, -0.2672, -1.3653,  1.3153,  0.3396,  1.0084, -0.8621, -1.2118,  0.9619],
        [-1.3313,  0.1419,  0.1317, -1.5564, -0.7677,  1.1142, -0.9384, -0.3283, -1.3760,  0.1555],
        [-1.0212, -0.8457, -2.1222, -0.2275, -1.6920, -0.3452, -1.8101, -0.8370, -1.5620,  0.4710],
        [-0.5649,  0.5389,  0.1830, -2.2848, -0.7070, -0.2891, -2.1851,  0.0652, 0.9065,  1.2124],
        [-0.5383, -0.1118, -0.1771, -2.0607, -0.8119,  0.6571, -1.5029, -0.4848, 0.5506,  1.0067],
        [-0.3072, -0.3035, -1.7673, -1.6210,  0.3929,  0.4658,  1.5073,  0.3069, 1.0556,  2.4014],
        [ 0.4328, -0.6846,  0.5944, -0.6157,  0.4943,  1.2940,  0.2372, -0.3712, 0.0409, -0.5548],
        [ 0.5904,  0.4841, -1.3936,  1.6245,  0.1604,  1.0006,  0.5186,  1.0113, -1.9488, -0.4466],
        [-2.1396,  0.9569,  1.2400,  0.2532,  2.8215, -0.3417,  0.6906,  0.3958, -0.1928, -0.6994],
        [ 1.7039,  0.7420,  0.9737,  0.3003, -0.2897, -0.3157, -0.8790,  0.1066, 1.8598,  0.0558],
        [ 1.3152, -0.6235, -1.2270,  0.6756, -0.4293,  0.0408,  0.6072,  0.8683, 0.6256, -1.2261],
        [-0.1909,  0.4892,  0.3241,  1.9630, -0.2482, -0.8072,  0.2763, -1.1261, -0.4691, -0.4845],
        [ 1.2833,  1.4593,  1.1233,  0.9517, -1.2344,  0.7200,  0.8272, -0.7293, 0.5859, -1.4506],
        [-0.2477,  0.8786, -0.0705, -0.9768, -0.4685, -0.7242,  0.7069, -0.1953, 1.2007,  0.3565],
        [ 1.3223, -1.0930, -0.8457,  0.1900, -0.7271, -0.3932, -1.2494,  1.2130, -1.7152, -0.4757],
        [-1.0444, -0.1401, -0.2890,  1.1668, -0.6720,  0.6441,  0.5891,  1.9296, -0.3814,  0.0134],
        [ 0.4889,  0.7588, -1.6100, -0.1641,  1.3673, -0.2865,  0.2065,  0.9916, -0.5933, -0.4054],
        [-1.2319,  0.0912,  0.1056, -0.3980,  0.0822, -0.5721,  0.7985, -0.8651, -0.4032, -0.7815],
        [-0.9465, -0.1594, -0.1934, -0.3766, -0.0492,  0.0939, -0.6453,  1.2108, -0.7820,  0.3845]], requires_grad=True)

Xb: 32 * 3
tensor([[ 1,  1, 22],
        [ 0,  1, 19],
        [ 0,  0,  0],
        [13,  4,  9],
        [14, 20,  8],
        [13,  1, 18],
        [15, 21, 18],
        [ 0,  0, 13],
        [ 0,  0,  0],
        [ 1, 20,  5],
        [ 9, 12, 25],
        [ 0,  0,  3],
        [ 0,  0,  0],
        [ 0, 10,  1],
        [19, 19,  1],
        [ 0,  0,  0],
        [13,  3, 11],
        [18,  4,  8],
        [ 1,  9,  1],
        [ 9,  1,  8],
        [18,  5, 20],
        [ 0,  0,  1],
        [ 1, 10,  1],
        [ 0,  0,  0],
        [ 0,  0,  8],
        [ 0,  0,  9],
        [ 4,  1, 14],
        [ 8,  1, 21],
        [ 0,  0,  0],
        [18,  5, 25],
        [ 9, 10, 15],
        [19,  1, 13]])

Yb: 32
tensor([ 9,  8,  4,  0, 15, 25,  9,  1, 12, 13, 14,  1, 19,  3,  0,  2,  5,  1,
         0, 19, 26, 11, 14,  5,  1, 13,  9, 14,  1,  0, 14,  9])


emb = C[Xb] 是如何计算的?

你问得非常好！我们一步步拆开讲：

---

### 这一句代码：

```python
emb = C[Xb]
```

它的作用是：**把 batch 里的字符索引（Xb）转成对应的 embedding 向量**，
也就是 **“查表”操作**，查 embedding 矩阵 C。

---

### 先看 C 是什么？

```python
C = torch.randn((vocab_size, n_embd), generator=g)
```

* C 是一个 **\[vocab\_size, n\_embd]** 的矩阵
* **vocab\_size** 是字符表里有多少个不同字符（比如 27 个：a\~z + '.'）
* **n\_embd** 是每个字符的 embedding 维度（比如 10 维）

所以 C 的形状是：

```
[vocab_size, n_embd] → [27, 10]
```

意思是：

* 第 0 行：第 0 个字符（比如 'a'）的 embedding 向量 → \[10 维]
* 第 1 行：第 1 个字符（比如 'b'）的 embedding 向量 → \[10 维]
* ...
* 第 26 行：'.' 的 embedding 向量 → \[10 维]

---

### 再看 Xb 是什么？

```python
Xb = Xtr[ix]
```

* Xb 是一个 batch，shape 是：

```
[batch_size, block_size] → [32, 3]
```

* 里面存的是 **字符索引**（整数），
  比如：

```python
Xb =
[
  [5,  0, 12],  # 第一个样本的 3 个字符
  [3, 10,  2],  # 第二个样本的 3 个字符
  ...
  [9, 18,  4],  # 第 32 个样本
]
```

每个数字是字符的索引。

---

### C\[Xb] 是怎么计算的？

→ **查表操作（索引）**

```python
emb = C[Xb]  # shape: [batch_size, block_size, n_embd]
```

等效于对 Xb 里的每个索引去 C 里对应行取出 embedding 向量，
所以 emb 的 shape 变成：

```
[32, 3, 10]
```

举例：

假设：

```
Xb[0] = [5, 0, 12]
```

那么：

```python
emb[0] = [
  C[5],   # 第 5 个字符的 embedding 向量 (10 维)
  C[0],   # 第 0 个字符的 embedding 向量
  C[12],  # 第 12 个字符的 embedding 向量
]
```

---

### 图示：

```
Xb →  [batch_size, block_size]  →  [32, 3]
        ↓
C[Xb] → embedding 向量 → [32, 3, 10]
```

---

### 总结：

```python
emb = C[Xb]
```

\= 对 batch 里的每个字符索引
\= 在 C 里 **查 embedding 向量**
\= 变成 shape：\[batch\_size, block\_size, n\_embd]

---

**一句话总结：**
这一句就是“查 embedding 表，把索引变成向量”！ 🚀

---

如果你需要，我还可以帮你画一个**更完整的输入 → embedding → Linear 的图**，要不要？ 🌟



# fixing the initial loss

network is very improperly configured at initialization and there's multiple things wrong with it but let's just
start with the first one look here on the zeroth iteration the very first iteration we are recording a loss of 27
and this rapidly comes down to roughly one or two or so so I can tell that the initialization is all messed up because
this is way too high in training of neural Nets it is almost always the case that you will have a rough idea for what
loss to expect at initialization and that just depends on the loss function and the problem setup in this case I do
not expect 27 I expect a much lower number and we can calculate it together basically at initialization what we like
is that um there's 27 characters that could come next for any one training example at initialization we have no
reason to believe any characters to be much more likely than others and so we'd expect that the propy distribution that
comes out initially is a uniform distribution assigning about equal probability to all the 27
characters so basically what we' like is the probability for any character would
be roughly 1 over 20 7 that is the probability we should
record and then the loss is the negative log probability so let's wrap this in a
tensor and then then we can take the log of it and then the negative log probability is the loss we would expect
which is 3.29 much much lower than 27 and so what's happening right now is
that at initialization the neural nut is creating probity distributions that are all messed up some characters are very
confident and some characters are very not confident confident and then basically what's happening is that the network is very confidently wrong and uh
that that's what makes it um record very high loss so here's a smaller four-dimensional example of the issue
let's say we only have four characters and then we have logits that come out of the neural net and they are very very
close to zero then when we take the softmax of all zeros we get probabilities there are a diffused
distribution so sums to one and is exactly uniform and then in this case if the
label is say two it doesn't actually matter if this if the label is two or three or one or zero because it's a
uniform distribution we're recording the exact same loss in this case 1.38 so this is the loss we would expect for a
four-dimensional example and now you can see of course that as we start to manipulate these logits uh we're going
to be changing the law here so it could be that we lock out and by chance uh this could be a very high number like
you know five or something like that then case we'll record a very low loss because we're assigning the correct probability at initialization by chance
to the correct label much more likely it is that some other dimension will have a
high uh logit and then what will happen is we start to record much higher loss
and what can come what can happen is basically the logits come out like something like this you know and they
take on Extreme values and we record really high loss
um for example if we have to 4. random of four so these are uniform um sorry
these are normally distributed um numbers uh four of
them then here we can also print the logits probabilities that come out of it
and the loss and so because these logits are near zero for the most part the loss
that comes out is is okay uh but suppose this is like times 10
now you see how because these are more extreme values it's very unlikely that you're going to be guessing the correct
bucket and then you're confidently wrong and recording very high loss if your loes are coming out even more
extreme you might get extremely insane losses like infinity even at
initialization um so basically this is not good and we want the loges to be roughly zero um
when the network is initialized in fact the lits can don't have to be just zero they just have to be equal so for
example if all the logits are one then because of the normalization inside the softmax this will actually come out okay
but by symmetry we don't want it to be any arbitrary positive or negative number we just want it to be all zeros
and record the loss that we expect at initialization so let's now concretely see where things go wrong in our example
here we have the initialization let me reinitialize the neuronet and here let me break after the very first iteration
so we only see the initial loss which is 27 so that's way too high and intuitively
now we can expect the variables involved and we see that the logits here if we just print some of
these if we just print the first row we see that the Lo just take on quite extreme values and that's what's
creating the fake confidence in incorrect answers and makes the loss um
get very very high so these loes should be much much closer to zero so now let's
think through how we can achieve logits coming out of this neur not to be more closer to zero you see here that loes
are calculated as the hidden states multip by W2 plus B2 so first of all
currently we're initializing B2 as random values uh of the right size but
because we want roughly zero we don't actually want to be adding a bias of random numbers so in fact I'm going to add a times zero here to make sure that
B2 is just um basically zero at initialization and second this is H
multip by W2 so if we want logits to be very very small then we would be multiplying W2 and making that
smaller so for example if we scale down W2 by 0.1 all the elements then if I do
again just a very first iteration you see that we are getting much closer to what we expect so rough roughly what we
want is about 3.29 this is 4.2 I can make this maybe even
smaller 3.32 okay so we're getting closer and closer now you're probably
wondering can we just set this to zero then we get of course exactly what we're looking for um at
initialization and the reason I don't usually do this is because I'm I'm very nervous and I'll show you in a second
why you don't want to be setting W's or weights of a neural nut exactly to zero um you you usually want it to be small
numbers instead of exactly zero um for this output layer in this specific case
I think it would be fine but I'll show you in a second where things go wrong very quick quickly if you do that so
let's just go with 0.01 in that case our loss is close enough but has some entropy it's not
exactly zero it's got some little entropy and that's used for symmetry breaking as we'll see in a second the
logits are now coming out much closer to zero and everything is well and good so if I just erase these and I now take
away the break statement we can run the optimization with this new initialization and let's
just see what losses we record okay so I let it run and you see that we started off good
and then we came down a bit the plot of the loss uh now doesn't have this hockey shape appearance um
because basically what's happening in the hockey stick the very first few iterations of the loss what's happening
during the optimization is the optimization is just squashing down the logits and then it's rearranging the
logits so basically we took away this easy part of the loss function where just the the weights were just being
shrunk down and so therefore we're we don't we don't get these easy gains in the beginning and we're just getting
some of the hard gains of training the actual neural nut and so there's no hockey stick appearance so good things
are happening in that both number one losset initialization is what we expect and the the loss doesn't look like a
hockey stick and this is true for any neuron that you might train um and something to look out for and second the
loss that came out is actually quite a bit improved unfortunately I erased what we had here before I believe this was 2.
um2 and this was this was 2.16 so we get a slightly improved result and the
reason for that is uh because we're spending more Cycles more time optimizing the neuronet actually instead
of just uh spending the first several thousand iterations probably just squashing down the
weights because they are so way too high in the beginning in the initialization so something to look out for and uh
that's number one now let's look at the second problem let me reinitialize our neural net and let me reintroduce The

当然可以，下面是这一段 **Fixing the initial loss** 的中文解释，比较长我帮你分段讲：

---

### 🟢 1️⃣ **一开始的问题**

* 网络初始化之后，**第一次迭代（iteration 0）的 loss 高达 27**，非常不正常。
* 训练过程中 loss 很快下降到 1～2 左右。
* 为什么一开始这么高？→ **初始化有问题**。

**正常情况：**

* 对神经网络的 loss，**初始化时大概能估计个范围**，这个范围取决于 loss function 和任务类型。

* 这个任务是 **27 个字符的分类问题**，如果一开始没有任何偏好，模型输出应该是 **均匀分布**，即每个字符概率 ≈ 1/27。

* 对应的期望 loss 应该是：

```
loss = -log(1/27) ≈ 3.29  
```

---

### 🟢 2️⃣ **为什么会高达 27？**

* 因为现在模型的 logits（未归一化的分数）是\*\*“乱七八糟”\*\*的：

  * 有些 logits 很大（confident）
  * 有些 logits 很小

* 结果模型 “**非常自信但错了**” → **confidently wrong** → loss 爆炸到 27。

---

### 🟢 3️⃣ **举例说明**

* 假设我们有 4 类分类问题，初始 logits 都是 0，

  * softmax 后会得到 **均匀分布**，loss ≈ 1.38
* 如果 logits 很极端，比如：

```
logits = [10, -10, 5, -7]  
```

* 结果会出现 **错误类别概率极低，正确类别概率极低**，→ loss 很高。

→ 所以：**logits 初始应该接近 0**，分布越“温和”，loss 越正常。

---

### 🟢 4️⃣ **如何修正初始化？**

当前 logits 是：

```
logits = h @ W2 + b2  
```

* W2、b2 都是随机初始化，**b2 是不该有 bias**，所以：

```
b2 = 0  
```

* W2 太大 → 缩小 W2 比如 × 0.1
  → logits 更接近 0
  → 初始 loss 更接近 3.29

反复调整，最后用：

```
W2 *= 0.01  
```

* logits 接近 0，loss ≈ 3.3，符合预期！

---

### 🟢 5️⃣ **不要全 0 初始化！**

虽然你会想 “我是不是直接把 W2 初始化成 0 就好了？”

* 答案：不要！

* 因为如果所有权重都是 0，反向传播的时候 **所有神经元的梯度也都是一样的**，模型 **没法学习**，会出现所谓 **symmetry breaking** 的问题。

* 所以：**小随机值** 是更好的初始化。

---

### 🟢 6️⃣ **效果对比**

以前：

* loss 曲线是 hockey stick（冰球棒型），开始几千步只是在调整 logits
* 因为 logits 初始化太极端，训练一开始浪费了很多步来“压缩” logits

修正后：

* loss 一开始就合理
* 曲线变平滑，训练过程效率更高
* 最终 loss 也稍微下降了一点（比如 2.16 → 2.12）

---

### 🟢 7️⃣ **总结一句话**

这段讲的是：

- ✅ 如果 logits 初始化太极端，模型一开始就 “自信但错”，loss 非常大；
- ✅ 正确做法：初始化 logits ≈ 0，对应的 loss ≈ 期望值（3.29）；
- ✅ 要通过调整 W2 和 b2 的初始化，控制 logits 大小；
- ✅ 不要直接全 0，应该是小随机值，保持 symmetry breaking。
- ✅ 初始化调好了，训练更稳定，loss 更好，曲线也更平滑。

---

如果你需要，我还可以帮你画一个 **“logits 大小 vs loss”** 的对比图，或者整理一份 **初始化调优 checklist**，要不要？ 🌟

---

```python
# 4-dimensional example of the issue
logits = torch.tensor([0.0, 0.0, 5.0, 0.0])
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
probs, loss
```

解释一下

---

当然可以，这段代码是用一个 **简单 4 维的例子** 来演示 “logits 极端值会影响 loss” 的现象，咱们一步步拆开讲：

---

### 代码：

```python
logits = torch.tensor([0.0, 0.0, 5.0, 0.0])
probs = F.softmax(logits, dim=0)
loss = -probs[2].log()
probs, loss
```

---

### 一步步解释：

#### ① logits：

```python
logits = [0.0, 0.0, 5.0, 0.0]
```

* 4 维 logits，代表 4 个类别的未归一化分数
* 第 3 个类别 logits=5，其它是 0

#### ② softmax：

```python
probs = F.softmax(logits, dim=0)
```

* softmax 会把 logits 转成概率，**归一化到 0～1**，总和 = 1

计算过程：

```
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

实际算下来：

```
exp(0) = 1  
exp(5) ≈ 148.413  
```

所以：

```
sum = 1 + 1 + 148.413 + 1 ≈ 151.413

probs = [
    1 / 151.413 ≈ 0.0066,
    1 / 151.413 ≈ 0.0066,
    148.413 / 151.413 ≈ 0.9802,
    1 / 151.413 ≈ 0.0066
]
```

#### ③ loss：

```python
loss = -probs[2].log()
```

* 这里假设 “正确标签” 就是 index=2
* 那 loss 就是：

```
loss = -log( 0.9802 ) ≈ 0.0199
```

#### ④ 为什么要这样写？

目的是演示：

* 如果 logits 非常偏，比如某个值很大，softmax 就会让那个类别概率很高 → 网络 “非常自信”
* 如果 **预测对了**，loss 很小（ ≈ 0 ）
* 如果 **预测错了**，loss 会很大！

---

### 总结：

这一段代码用 **4 维 logits 的 toy example** 演示：

✅ 如果 logits 很 extreme（偏大），softmax 输出也会“非常偏”
✅ 如果恰好预测对了 → loss 很小
✅ 如果预测错了 → loss 会爆炸

---

一句话总结：

👉 logits 太极端 → 模型“过度自信”，如果错了，loss 会非常高！
👉 初始化 logits 时 **要接近 0**，softmax 才会合理分布。

---

如果需要我可以帮你画一个 **logits → softmax → loss 的小图**，要不要？ 🚀✨

---


# fixing the saturated tanh

Brak statement so we have a reasonable initial loss so even though everything is looking good on the level of the loss
and we get something that we expect there's still a deeper problem looking inside this neural net and its
initialization so the logits are now okay the problem now is with the values
of H the activations of the Hidden States now if we just visualize this
Vector sorry this tensor h it's kind of hard to see but the problem here roughly speaking is you see how many of the
elements are one or negative 1 now recall that torch. 10 the 10 function is
a squashing function it takes arbitrary numbers and it squashes them into a range of negative 1 and one and it does
so smoothly so let's look at the histogram of H to get a better idea of the distribution of the values inside
this tensor we can do this first well we can see that H is 32
examples and 200 activations in each example we can view it as1 to stretch it
out into one large vector and we can then call two list to
convert this into one large python list of floats and then we can pass this into
PLT doist for histogram and we say we want 50 bins and a semicolon to suppress
a bunch of output we don't want so we see this histogram and we see that most the values by far take on
value of netive one and one so this 10 H is very very active and we can also look
at basically why that is we can look at the pre activations that feed into the
10 and we can see that the distribution of the pre activations are is very very
broad these take numbers between -5 and 15 and that's why in a torure 10
everything is being squashed and capped to be in the range of negative 1 and one and lots of numbers here take on very
extreme values now if you are new to neural networks you might not actually see this as an issue but if you're well
vered in the dark arts of back propagation and then having an intuitive sense of how these gradients flow through a neural net you are looking at
your distribution of 10h activations here and you are sweating so let me show you why we have to keep in mind that
during back propagation just like we saw in microad we are doing backward passs starting at the loss and flowing through
the network backwards in particular we're going to back propagate through this torch. 10h and this layer here is made up of
200 neurons for each one of these examples and uh it implements an elementwise 10 so let's look at what
happens in 10h in the backward pass we can actually go back to our previous uh microgr code in the very first lecture
and see how we implemented 10 AG we saw that the input here was X and then we
calculate T which is the 10 age of X so that's T and T is between 1 and 1 it's
the output of the 10 H and then in the backward pass how do we back propagate through a 10 H we take out that grad um
and then we multiply it this is the chain rule with the local gradient which took the form of 1 - t ^2 so what
happens if the outputs of your t h are very close to1 or 1 if you plug in t one
here you're going to get a zero multiplying out. grad no matter what out. grad is we are killing the gradient
and we're stopping effectively the back propagation through this 10 unit similarly when t is1 this will again
become zero and out that grad just stops and intuitively this makes sense because
this is a 10h neuron and what's happening is if its output is very close to one then we are
in the tail of this 10 and so changing basically the
input is not going to impact the output of the 10 too much because it's it's so
it's in a flat region of the 10 H and so therefore there's no impact on the loss
and so so indeed the the weights and the biases along with the 10h neuron do not
impact the loss because the output of the 10 unit is in the flat region of the 10 and there's no influence we can we
can be changing them whatever we want however we want and the loss is not impacted that's so that's another way to
justify that indeed the gradient would be basically zero it vanishes indeed uh when T equals zero we
get one times out that grad so when the 10 h takes on exactly value of zero then
out grad is just passed through so basically what this is doing right is if
T is equal to zero then this the 10 unit is uh sort of inactive and uh gradient
just passes through but the more you are in the flat tails the more the gradient is squashed so in fact you'll see that
the the gradient flowing through 10 can only ever decrease and the amount that it decreases is um proportional through
a square here um depending on how far you are in the flat tail so this 10 H
and so that's kind of what's Happening Here and through this the concern here is that if all of these um outputs H are
in the flat regions of negative 1 and one then the gradients that are flowing through the network will just get
destroyed at this layer now there is some redeeming quality here and that we can actually
get a sense of the problem here as follows I wrote some code here and basically what we want to do here is we
want to take a look at H take the the absolute value and see how often it is
in the in a flat uh region so say greater than 099 and what you get is the following
and this is a Boolean tensor so uh in the Boolean tensor you get a white if
this is true and a black if this is false and so basically what we have here is the 32 examples and 200 hidden
neurons and we see that a lot of this is white and what that's telling us is that
all these 10h neurons were very very active and uh they're in a flat tail and
so in all these cases uh the back the backward gradient would get uh
destroyed now we would be in a lot of trouble if for for any one of these 200
neurons if it was the case that the entire column is white because in that
case we have what's called a dead neuron and this is could be a 10 neuron where the initialization of the weights and the biases could be such that no single
example ever activates uh this 10h in the um sort of active part of the 10age
if all the examples land in the tail then this neuron will never learn it is
a dead neuron and so just scrutinizing this and looking for Columns of
completely white uh we see that this is not the case so uh I don't see a single
neuron that is all of uh you know white and so therefore it is the case that for every one of these 10h neurons uh we do
have some examples that activate them in the uh active part of the 10 and so some gradients will flow through and this
neuron will learn and the neuron will change and it will move and it will do something but you can sometimes get get
yourself in cases where you have dead neurons and the way this manifests is that um for 10h neuron this would be
when no matter what inputs you plug in from your data set this 10h neuron always fir completely one or completely negative
one and then it will just not learn because all the gradients will be just zeroed out uh this is true not just for
10 but for a lot of other nonlinearities that people use in neural networks so we certainly used 10 a lot but sigmoid will
have the exact same issue because it is a squashing neuron and so the same will be true for sigmoid uh but um but um you
know um basically the same will actually apply to sigmoid the same will also apply to reu
so reu has a completely flat region here below zero so if you have a reu neuron
then it is a pass through um if it is positive and if it's if the preactivation is negative it will just
shut it off since the region here is completely flat then during back propagation uh this would be exactly
zeroing out the gradient um like all of the gradient would be set exactly to zero instead of just like a very very
small number depending on how positive or negative T is and so you can get for
example a dead reu neuron and a dead reu neuron would basically look like
basically what it is is if a neuron with a reu nonlinearity never activates so
for any examples that you plug in in the data set it never turns on it's always in this flat region then this re neuron
is a dead neuron its weights and bias will never learn they will never get a gradient because the neuron never
activated and this can sometimes happen at initialization uh because the way and a biases just make it so that by chance
some neurons are just forever dead but it can also happen during optimization if you have like a too high of learning
rate for example sometimes you have these neurons that get too much of a gradient and they get knocked out off
the data manifold and what happens is that from then on no example ever activates this
neuron so this neuron remains dead forever so it's kind of like a permanent brain damage in a in a mind of a network
and so sometimes what can happen is if your learning rate is very high for example and you have a neural net with neurons you train the neuron net and you
get some last loss but then actually what you do is you go through the entire training set and you forward um your
examples and you can find neurons that never activate they are dead neurons in your network and so those neurons will
will never turn on and usually what happens is that during training these Rel neurons are changing moving Etc and
then because of a high gradient somewhere by chance they get knocked off and then nothing ever activates them and
from then on they are just dead uh so that's kind of like a permanent brain damage that can happen to some of these
neurons these other nonlinearities like leyu will not suffer from this issue as much because you can see that it doesn't
have flat Tails you'll almost always get gradients and uh elu is also fairly uh
frequently used um it also might suffer from this issue because it has flat parts so that's just something to be
aware of and something to be concerned about and in this case we have way too many um activations AG that take on
Extreme values and because there's no column of white I think we will be okay
and indeed the network optimizes and gives us a pretty decent loss but it's just not optimal and this is not
something you want especially during initialization and so basically what's happening is that uh this H
preactivation that's floating to 10 H it's it's too extreme it's too large
it's creating very um it's creating a distribution that is too saturated in both sides of the 10 H and it's not
something you want because it means that there's less training uh for these neurons because they update um less
frequently so how do we fix this well H preactivation is MCAT which comes from C
so these are uniform gsan but then it's multiply by W1 plus B1 and H preact is
too far off from zero and that's causing the issue so we want this reactivation to be closer to zero very similar to
what we had with logits so here we want actually something very very
similar now it's okay to set the biases to very small number we can either multiply by 0 01 to get like a little
bit of entropy um I sometimes like to do that um just so that there's like a
little bit of variation and diversity in the original initialization of these 10 H neurons and I find in practice that
that can help optimization a little bit and then the weights we can also just like squash so let's multiply everything
by 0.1 let's rerun the first batch and now let's look at this and well first let's
look here you see now because we multiply dou by 0.1 we have a much better histogram
and that's because the pre activations are now between 1.5 and 1.5 and this we expect much much less white okay there's
no white so basically that's because there are no neurons that saturated
above 99 in either direction so this actually a pretty decent place to be um
maybe we can go up a little bit sorry am I am I changing W1 here so
maybe we can go to 0 2 okay so maybe something like this is
is a nice distribution so maybe this is what our initialization should be so let me now
erase these and let me starting with
initialization let me run the full optimization without the break and uh let's see what
we get okay so the optimization finished and I re the loss and this is the result that we get and then just as a reminder
I put down all the losses that we saw previously in this lecture so we see that we actually do get an improvement
here and just as a reminder we started off with a validation loss of 2.17 when we started by fixing the softmax being
confidently wrong we came down to 2.13 and by fixing the 10h layer being way too saturated we came down to 2.10
and the reason this is happening of course is because our initialization is better and so we're spending more time doing productive training instead of um
not very productive training because our gradients are set to zero and uh we have to learn very simple things like uh the
overconfidence of the softmax in the beginning and we're spending Cycles just like squashing down the weight Matrix so
this is illustrating um basically initialization and its impacts on performance uh just by being aware of
the internals of these neural net and their activations their gradients now we're working with a very small Network
this is just one layer multi-layer perception so because the network is so shallow the optimization problem is
actually quite easy and very forgiving so even though our initialization was terrible the network still learned
eventually it just got a bit worse result this is not the case in general though once we actually start um working
with much deeper networks that have say 50 layers uh things can get uh much more complicated and uh these problems stack
up and so you can actually get into a place where the network is basically not training at all if your initialization
is bad enough and the deeper your network is and the more complex it is the less forgiving it is to some of
these errors and so um something to definitely be aware of and uh something
to scrutinize something to plot and something to be careful with and um yeah

当然可以！我帮你把这一大段「fixing the saturated tanh」的内容整理成中文，按逻辑拆解讲，方便你理解：

---

### 🟢 1️⃣ 现在虽然 loss 看起来正常了，但还有隐藏问题：

* logits 修好了，loss ≈ 3.29，OK
* **但是隐藏层 H 的值有问题**

```
H = tanh(pre-activation)
```

* 我们发现 H 的很多值是 **-1 或 1**
* 这是因为 tanh 是一个「压缩函数」：输入很大或很小时，输出就被压扁到 -1 或 1

---

### 🟢 2️⃣ 画出 H 的 histogram

* 把 H 画出来发现：**大部分值都是 -1 或 1**

原因是：

```
pre-activation = embcat @ W1 + b1
```

* 现在 pre-activation 值分布很广（比如 -5 到 +15）
* 输入 tanh 后就大面积「饱和」（saturated）
* → H 很多都是 -1 或 1

---

### 🟢 3️⃣ 为什么 tanh 饱和不好？

你如果是新手，可能不觉得 tanh 饱和是个问题 —— 其实这是个「大坑」，因为：

1️⃣ 反向传播时，要计算：

```
grad = upstream_grad * (1 - tanh(x)^2)
```

如果 tanh(x) ≈ ±1，
→ (1 - tanh(x)^2) ≈ 0
→ grad ≈ 0，**梯度消失**！

2️⃣ 梯度消失 → **训练停滞**

* 这些神经元学不到东西
* 部分神经元变成「死神经元（dead neuron）」
  → 永远激活在 tanh 饱和区，不参与训练
  → 对 loss 没有贡献

---

### 🟢 4️⃣ 怎么观察 dead neuron？

他写了段代码：

```
abs(H) > 0.99
```

* 看哪些 neuron 的 H 输出 **绝对值接近 1**（饱和区）
* 结果发现很多 neuron 出现这种现象，虽然没有“整列”都白（整列白 → 完全 dead neuron）

---

### 🟢 5️⃣ 其他激活函数也会有类似问题：

* sigmoid 同样是压缩函数 → 也会有梯度消失
* ReLU 虽然在正区间不会饱和，但负区间会 **死掉**，也是 dead neuron（ReLU dead neuron）

  * 有些 neuron 训练过程中 ReLU 永远激活为 0 → 权重永远学不到 → 永远死 neuron
* ELU/Leaky ReLU 等可以减缓这个问题（平滑激活）

---

### 🟢 6️⃣ 为什么 H 饱和了？

归根结底：

```
pre-activation = embcat @ W1 + b1
```

* 现在 W1、b1 初始化太大了 → pre-activation 太大
* → 经过 tanh 被压到 ±1

---

### 🟢 7️⃣ 怎么修正？

和修 logits 一样：

- ✅ b1 → 初始化为 0 或小值
- ✅ W1 → 缩小初始化 scale，比如 × 0.1

然后重新观察：

* 现在 pre-activation 值在 -1.5 \~ 1.5 区间
* H 分布良好，不饱和
* 没有死 neuron
* 梯度可以正常传递

---

### 🟢 8️⃣ 优化后的结果？

* 以前 val loss ≈ 2.17
* 改 logits → val loss ≈ 2.13
* 改 tanh 饱和 → val loss ≈ 2.10

训练更高效、loss 更低
不会浪费很多步来「解饱和」「压 logits」

---

### 🟢 9️⃣ 总结：

- ✅ tanh 饱和会导致梯度消失 → 神经元死掉
- ✅ 初始 W1、b1 要调小，避免 pre-activation 过大
- ✅ 不只是 tanh，sigmoid、ReLU 都会有类似问题
- ✅ 越深的网络，初始化越重要（现在 MLP 1 层还好，50 层就 GG 了）
- ✅ 训练初期观察 H 的 histogram 是一个好习惯！

---

一句话总结：

- 👉 这段讲的是「**初始化太大 → tanh 饱和 → 梯度消失 → 训练失败**」的问题，
- 👉 修正方法是「**调小 W1、b1，防止饱和**」。

---

如果你需要我还可以帮你：

- ✅ 画一张「tanh 饱和 → 梯度消失」的曲线图
- ✅ 总结一份「初始化防饱和 checklist」

要不要？ 🚀✨


# calculating the init scale: “Kaiming init”

okay so that's great that that worked for us but what we have here now is all these magic numbers like0 2 like where
do I come up with this and how am I supposed to set these if I have a large neural net with lots and lots of layers
and so obviously no one does this by hand there's actually some relatively principled ways of setting these scales
um that I would like to introduce to you now so let me paste some code here that I prepared just to motivate the
discussion of this so what I'm doing here is we have some random input here x that is drawn
from a gan and there's 1,000 examples that are 10 dimensional and then we have a waiting layer here
that is also initialized using caution just like we did here and we these
neurons in the hidden layer look at 10 inputs and there are 200 neurons in this hidden layer and then we have here just
like here um in this case the multiplication X multip by W to get the pre activations of these
neurons and basically the analysis here looks at okay suppose these are uniform gion and these weights are uniform gion
if I do X W and we forget for now the bias and the
nonlinearity then what is the mean and the standard deviation of these gions so in the beginning here the input is uh
just a normal Gan distribution mean zero and the standard deviation is one and the standard deviation again is just the
measure of a spread of the gion but then once we multiply here and we look at the um histogram of Y we see
that the mean of course stays the same it's about zero because this is a symmetric operation but we see here that
the standard deviation has expanded to three so the input standard deviation was one but now we've grown to three and
so what you're seeing in the histogram is that this Gan is expanding and so um we're expanding this
Gan um from the input and we don't want that we want most of the neural net to have relatively similar activations uh
so unit gion roughly throughout the neural net and so the question is how do we scale these W's to preserve the uh um
to preserve this distribution to uh remain aan and so intuitively if I multiply
here uh these elements of w by a larger number let's say by
five then this gsan gross and gross in standard deviation so now we're at 15 so
basically these numbers here in the output y take on more and more extreme values but if we scale it down like .2
then conversely this Gan is getting smaller and smaller and it's shrinking
and you can see that the standard deviation is 6 and so the question is what do I multiply by here to exactly
preserve the standard deviation to be one and it turns out that the correct answer mathematically when you work out
through the variance of uh this multiplication here is that you are supposed to divide by the square root of
the fan in the fan in is the basically the uh number of input elements here 10
so we are supposed to divide by 10 square root and this is one way to do the square root you raise it to a power
of 0. five that's the same as doing a square root so when you divide by the um
square root of 10 then we see that the output caution it has exactly standard
deviation of one now unsurprisingly a number of papers have looked into how
but to best initialized neural networks and in the case of multilayer perceptrons we can have fairly deep
networks that have these nonlinearity in between and we want to make sure that the activations are well behaved and
they don't expand to infinity or Shrink all the way to zero and the question is how do we initialize the weights so that
these activations take on reasonable values throughout the network now one paper that has studied this in quite a
bit of detail that is often referenced is this paper by King hatal called delving deep into rectifiers now in this
case they actually study convolution neur neurals and they study especially the reu nonlinearity and the p
nonlinearity instead of a 10h nonlinearity but the analysis is very similar and um basically what happens
here is for them the the relu nonlinearity that they care about quite a bit here is a squashing function where
all the negative numbers are simply clamped to zero so the positive numbers are pass through but everything negative
is just set to zero and because uh you are basically throwing away half of the distribution they find in their analysis
of the forward activations in the neural that you have to compensate for that with a
gain and so here they find that basically when they initialize their
weights they have to do it with a zero mean Gan whose standard deviation is square < TK of 2 over the Fanon what we
have here is we are initializing gashin with the square root of Fanon this NL
here is the Fanon so what we have is sare root of one over the Fanon because
we have the division here now they have to add this factor of two because of the reu which basically
discards half of the distribution and clamps it at zero and so that's where you get an additional Factor now in
addition to that this paper also studies not just the uh sort of behavior of the activations in the forward pass of the
neural net but it also studies the back propagation and we have to make sure that the gradients also are well behaved
and so um because ultimately they end up updating our parameters and what they
find here through a lot of analysis that I invite you to read through but it's not exactly approachable what they find
is basically if you properly initialize the forward pass the backward pass is also approximately initialized up to a
constant factor that has to do with the size of the number of um hidden neurons
in an early and a late layer and uh but basically they find
empirically that this is not a choice that matters too much now this timing initialization is also implemented in
pytorch so if you go to torch. and then. init documentation you'll find climing normal and in my opinion this is
probably the most common way of initializing neural networks now and it takes a few keyword arguments here so
number one it wants to know the mode would you like to normalize the activations or would you like to
normalize the gradients to to be always uh gsh in with zero mean and a unit or
one standard deviation and because they find in the paper that this doesn't matter too much most of the people just
leave it as the default which is Fan in and then second passing the nonlinearity that you are using because depending on
the nonlinearity we need to calculate a slightly different gain and so if your nonlinearity is just um linear so
there's no nonlinearity then the gain here will be one and we have the exact same uh kind of formula that we've come
up here but if the nonlinearity is something else we're going to get a slightly different gain and so if we
come up here to the top we see that for example in the case of reu this gain is a square root of two and the reason it's
a square root because in this
paper you see how the two is inside of the square root so the gain is a square
root of two in the case of linear or identity we just get a gain of one in a
case of 10 H which is what we're using here the advised gain is a 5 over3 and intuitively why do we need a gain on top
of the initialization is because 10 just like reu is a contractive uh
transformation so that means is you're taking the output distribution from this matrix multiplication and then you are
squashing it in some way now reu squashes it by taking everything below zero and clamping it to zero 10 also
squashes it because it's a contractive operation it will take the Tails and it will squeeze them in and so in order to
fight the squeezing in we need to boost the weights a little bit so that we renormalize everything back to standard
unit standard deviation so that's why there's a little bit of a gain that comes out now I'm skipping through this
section A little bit quickly and I'm doing that actually intentionally and the reason for that is because about 7
years ago when this paper was written you had to actually be extremely careful with the activations and ingredients and
their ranges and their histograms and you had to be very careful with the precise setting of gains and the scrutinizing of the nonlinearities used
and so on and everything was very finicky and very fragile and to be very properly arranged for the neural nut to
train especially if your neural nut was very deep but there are a number of modern innovations that have made everything significantly more stable and
more well behaved and it's become less important to initialize these networks exactly right and some of those modern
Innovations for example are residual connections which we will cover in the future the use of a number of uh
normalization uh layers like for example batch normalization layer normalization group normalization we're going to go
into a lot of these as well and number three much better optimizers not just stochastic gradient descent the simple
Optimizer we're basically using here but a slightly more complex optimizers like ARS prop and especially Adam and so all
of these modern Innovations make it less important for you to precisely calibrate the neutralization of the neural net all
that being said in practice uh what should we do in practice when I initialize these neurals I basically
just uh normalize my weights by the square root of the Fanon uh so basically
uh roughly what we did here is what I do now if we want to be exactly accurate here we and go by um in it of uh timing
normal this is how it would implemented we want to set the standard deviation to be gain over the square root of fan in
right so to set the standard deviation of our weights we will proceed as
follows basically when we have a torch. Ranon and let's say I just create a th numbers we can look at the standard
deviation of this and of course that's one that's the amount of spread let's make this a bit bigger so it's closer to one so that's the spread of the Gan of
zero mean and unit standard deviation now basically when you take these and you multiply by
say2 that basically scales down the Gan and that makes it standard deviation 02
so basically the number that you multiply by here ends up being the standard deviation of this caution so
here this is a um standard deviation point2 caution here when we sample our
W1 but we want to set the standard deviation to gain over square root of
fan mode which is Fanon so in other words we want to mul mly by uh gain
which for 10 H is 5 over3 5 over3 is the gain and then
times
um or I guess sorry divide uh square root of the fan in and
in this example here the fan in was 10 and I just noticed that actually here the fan in for W1 is is actually an
embed times block size which as you all recall is actually 30 and that's because each character is 10 dimensional but
then we have three of them and we can catenate them so actually the fan in here was 30 and I should have used 30 here probably but basically we want 30
uh square root so this is the number this is what our standard deviation we want to be and this number turns out to
be3 whereas here just by fiddling with it and looking at the distribution and making sure it looks okay uh we came up
with 02 and so instead what we want to do here is we want to make the standard deviation b
um 5 over3 which is our gain divide this
amount times2 square root and these brackets here are not that uh necessary
but I'll just put them here for clarity this is basically what we want this is the timing in it in our case for a 10h
nonlinearity and this is how we would initialize the neural net and so we're multiplying by .3 instead of multiplying
by .2 and so we can we can initialize this
way and then we can train the neural net and see what we get okay so I trained the neural net and we end up in roughly
the same spot so looking at the validation loss we now get 2.10 and previously we also had 2.10 there's a
little bit of a difference but that's just the randomness of the process I suspect but the big deal of course is we get to the same spot but we did not have
to introduce any um magic numbers that we got from just looking at histograms
and guessing checking we have something that is semi- principled and will scale us to uh much bigger networks and uh
something that we can sort of use as a guide so I mentioned that the precise setting of these initializations is not

当然可以！我帮你把这一大段 **“calculating the init scale: Kaiming init”** 用中文分段讲解，逻辑更清楚：

---

### 🟢 1️⃣ 问题背景

* 前面我们发现：要把 W1 乘以 0.2 才能避免 tanh 饱和
* 但这个 0.2 是怎么来的？手动调？猜出来的？
* 如果是大网络（很多层），**没人会手动调 magic number**
* 于是就引出了一个系统性办法 → **Kaiming 初始化**

---

### 🟢 2️⃣ 例子演示

```python
x = 正态分布数据 (1000 个样本，10 维)
W = 正态分布权重矩阵 (10 输入 → 200 个神经元)
```

计算：

```python
y = x @ W
```

然后画出 y 的分布：

* 输入 x 的标准差是 1
* 但输出 y 的标准差变成了 3
  → 扩大了！

原因：

```
y = x_1 * W_1 + x_2 * W_2 + ... + x_10 * W_10
```

每一项的方差叠加，导致整体方差变大

---

### 🟢 3️⃣ 希望的效果

我们希望 **网络各层的激活分布“平稳”**，不要越往后越爆炸，或者缩小到 0

目标：

```
希望 y 的标准差 ≈ 1
```

---

### 🟢 4️⃣ 怎么办？调整 W 的 scale

理论推导：

```
W 应该除以 sqrt(fan_in)
```

fan\_in = 输入维度，比如这里是 10

所以：

```
W /= sqrt(10)
```

* 如果乘以 5，y 会爆炸
* 如果乘以 0.2，y 会缩小
* **除 sqrt(fan\_in) 恰好能保持 std=1**

---

### 🟢 5️⃣ 相关论文

引用了 **Kaiming He 2015 论文**（Delving Deep into Rectifiers）

* 研究了如何初始化 W，保证网络深层不会出现激活崩溃
* ReLU 激活函数要补偿 “砍掉一半分布” → 多一个 sqrt(2) 的 gain
* tanh 激活也是收缩型的函数 → 也需要一个 gain

---

### 🟢 6️⃣ PyTorch 实现

PyTorch 有：

```python
torch.nn.init.kaiming_normal
```

参数：

* mode: fan\_in / fan\_out
* nonlinearity: relu / tanh / linear 等

对不同激活函数，gain 不一样：

| 激活函数   | gain            |
| ------ | --------------- |
| linear | 1               |
| relu   | sqrt(2) ≈ 1.414 |
| tanh   | 5/3 ≈ 1.67      |

---

### 🟢 7️⃣ 为什么需要 gain？

因为：

* tanh / relu / sigmoid 都是 “压缩函数”
* 压缩了输入分布 → 需要提升 W 的初始化 scale，补偿压缩

---

### 🟢 8️⃣ 手动计算例子

```python
W1 = torch.randn(...) * (gain / sqrt(fan_in))
```

对于 tanh：

```python
gain = 5/3
fan_in = n_embd * block_size = 10 * 3 = 30
std = (5/3) / sqrt(30)
```

---

### 🟢 9️⃣ 实验结果

* 之前手动调 0.2，val\_loss ≈ 2.10
* 用 Kaiming 正确初始化，val\_loss 也是 ≈ 2.10

区别是：

- ✅ 之前需要反复看 histogram 猜 scale
- ✅ 现在 **理论上计算好**，不用猜，适合更大网络，通用性强！

---

### 🟢 10️⃣ 现代网络为什么更稳定？

以前（7 年前）：

* 初始化要求很严格
* 网络深容易不收敛

现在：

- ✅ ResNet（Residual 连接）
- ✅ BatchNorm / LayerNorm
- ✅ Adam 优化器

→ 网络更鲁棒，初始化也更容易设计

---

### 🟢 总结一句话：

👉 这一段讲的是：
**如何用 Kaiming 初始化法（gain / sqrt(fan\_in)）来系统设置 W1 的初始 scale，防止激活爆炸/消失，保证网络训练更稳定！**

---

如果你需要，我还可以：

- ✅ 帮你整理一个「Kaiming 初始化公式总结表」
- ✅ 帮你画「激活分布 vs 不同初始化 scale」的对比图

要不要？ 🚀✨


# batch normalization

as important today due to some Modern Innovations and I think now is a pretty good time to introduce one of those modern Innovations and that is batch
normalization so bat normalization came out in uh 2015 from a team at Google and
it was an extremely impact paper because it made it possible to train very deep neuron Nets quite reliably and uh it
basically just worked so here's what bash rization does and let's implement it
um basically we have these uh hidden States H preact right and we were
talking about how we don't want these uh these um preactivation states to be way
too small because the then the 10 H is not um doing anything but we don't want them to be too large because then the 10
H is saturated in fact we want them to be roughly roughly Gan so zero mean and
a unit or one standard deviation at least at initialization so the Insight from the
bachor liation paper is okay you have these hidden States and you'd like them to be roughly Gan then why not take the
hidden States and uh just normalize them to be Gan and it sounds kind of crazy but you
can just do that because uh standardizing hidden States so that
their unit caution is a perfect ly differentiable operation as we'll soon see and so that was kind of like the big Insight in this paper and when I first
read it my mind was blown because you can just normalize these hidden States and if you'd like unit Gan States in
your network uh at least initialization you can just normalize them to be unit gion so uh let's see how that works so
we're going to scroll to our preactivation here just before they enter into the 10h now the idea again is
remember we're trying to make these roughly Gan and that's because if these are way too small numbers then the 10 H
here is kind of inactive but if these are very large numbers then the 10 H is
way too saturated and gr is no flow so we'd like this to be roughly goshan so
the Insight in Bat normalization again is that we can just standardize these activations so they are exactly Gan so
here H preact has a shapee of 32 by 200 32
examples by 200 neurons in the hidden layer so basically what we can do is we can take H pract and we can just
calculate the mean um and the mean we want to calculate across the zero
Dimension and we want to also keep them as true so that we can easily broadcast
this so the shape of this is 1 by 200 in other words we are doing the mean over
all the uh elements in the batch and similarly we can calculate the
standard deviation of these activations and that will also be 1 by
200 now in this paper they have the uh sort of prescription here and see
here we are calculating the mean which is just taking uh the average
value of any neurons activation and then the standard deviation is basically kind
of like um this the measure of the spread that we've been using which is
the distance of every one of these values away from the mean and that
squared and averaged that's the that's the variance
and then if you want to take the standard deviation you would square root the variance to get the standard
deviation so these are the two that we're calculating and now we're going to normalize or standardize these X's by
subtracting the mean and um dividing by the standard deviation so basically
we're taking in pract and we subtract the mean
and then we divide by the standard deviation this is exactly what these two
STD and mean are calculating oops sorry this is the mean and this is
the variance you see how the sigma is a standard deviation usually so this is Sigma Square which the variance is the
square of the standard deviation so this is how you standardize these values and what this will do is
that every single neuron now and its firing rate will be exactly unit Gan on these 32 examples at least of this batch
that's why it's called batch normalization we are normalizing these batches and then we could in principle
train this notice that calculating the mean and your standard deviation these are just mathematical formulas they're
perfectly differentiable all of this is perfectly differentiable and we can just train this the problem is you actually
won't achieve a very good result with this and the reason for that
is we want these to be roughly Gan but only at initialization uh but we don't
want these be to be forced to be Garian always we we'd like to allow the neuron
net to move this around to potentially make it more diffuse to make it more sharp to make some 10 neurons maybe be
more trigger more trigger happy or less trigger happy so we'd like this distribution to move around and we'd
like the back propagation to tell us how the distribution should move around and so in addition to this idea of
standardizing the activations that any point in the network uh we have to also
introduce this additional component in the paper here described as scale and shift and so basically what we're doing
is we're taking these normalized inputs and we are additionally scaling them by some gain and offsetting them by some
bias to get our final output from this layer and so what that amounts to is the
following we are going to allow a batch normalization gain to be initialized at
just uh once and the ones will be in the shape of 1 by n
hidden and then we also will have a BN bias which will be torch. zeros and it
will also be of the shape n by 1 by n hidden and then here the BN gain will
multiply this and the BN bias will offset it here so because this is initialized to
one and this to zero at initialization each neurons firing values in this batch will be
exactly unit gion and will have nice numbers no matter what the distribution of the H pract is coming in coming out
it will be un Gan for each neuron and that's roughly what we want at least at initialization um and then during
optimization we'll be able to back propagate into BN gain and BM bias and change them so the network is given the
full ability to do with this whatever it wants uh internally here we just have to make
sure sure that we um include these in the parameters of the neural nut because
they will be trained with back propagation so let's initialize this and
then we should be able to
train and then we're going to also copy this line which is the batch
normalization layer here on a single line of code and we're going to swing down here and we're also going to do the
exact same thing at test time here so similar to train time we're
going to normalize uh and then scale and that's going to give us our train and validation
loss and we'll see in a second that we're actually going to change this a little bit but for now I'm going to keep it this way so I'm just going to wait
for this to converge okay so I allowed the neural nut to converge here and when we scroll down we see that our
validation loss here is 2.10 roughly which I wrote down here and we see that
this is actually kind of comparable to some of the results that we've achieved uh previously now I'm not actually
expecting an improvement in this case and that's because we are dealing with a very simple neural nut that has just a
single hidden layer so in fact in this very simple case of just one hidden layer we were able to actually calculate
what the scale of w should be to make these pre activations already have a roughly Gan shape so the bat
normalization is not doing much here but you might imagine that once you have a much deeper neural nut that has lots of
different types of operations and there's also for example residual connections which we'll cover and so on
it will become basically very very difficult to tune the scales of your weight matrices such that all the
activations throughout the neural nut are roughly gsen and so that's going to become very quickly intractable but
compared to that it's going to be much much easier to sprinkle batch normalization layers throughout the neural net so in particular it's common
to to look at every single linear layer like this one one this is a linear layer multiplying by a weight Matrix and adding a bias or for example
convolutions which we'll cover later and also perform basically a multiplication with a weight Matrix but in a more
spatially structured format it's custom it's customary to take these linear layer or convolutional layer and append
a b rization layer right after it to control the scale of these activations at every point in the neural nut so we'd
be adding these bom layers throughout the neural nut and then this controls the scale of these AC ations throughout
the neural net it doesn't require us to do uh perfect mathematics and care about the activation distributions uh for all
these different types of neural network uh Lego building blocks that you might want to introduce into your neural net
and it significantly stabilizes uh the training and that's why these uh layers are quite popular now the stability
offered by bash normalization actually comes at a terrible cost and that cost is that if you think about what's
Happening Here something something terribly strange and unnatural is happening it used to be that we have a
single example feeding into a neural nut and then uh we calculate its activations and its loits and this is a
deterministic sort of process so you arrive at some logits for this example and then because of efficiency of
training we suddenly started to use batches of examples but those batches of examples were processed independently
and it was just an efficiency thing but now suddenly in batch normalization because of the normalization through the
batch we are coupling these examples mathematically and in the forward pass and the backward pass of a neural l so
now the hidden State activations H pract in your log jits for any one input
example are not just a function of that example and its input but they're also a function of all the other examples that
happen to come for a ride in that batch and these examples are sampled randomly
and so what's happening is for example when you look at H pract that's going to feed into H the hidden State activations
for for example for for any one of these input examples is going to actually change slightly depending on what other
examples there are in a batch and and depending on what other examples happen to come for a ride H is going to change
subtly and it's going to like Jitter if you imagine sampling different examples because the the statistics of the mean
and the standard deviation are going to be impacted and so you'll get a Jitter for H and you'll get a Jitter for
loits and you think that this would be a bug uh or something undesirable but in a
very strange way this actually turns out to be good in your Network training and
as a side effect and the reason for that is that you can think of this as kind of like a regularizer because what's
happening is you have your input and you get your age and then depending on the other examples this is jittering a bit
and so what that does is that it's effectively padding out any one of these input examples and it's introducing a
little bit of entropy and um because of the padding out it's actually kind of like a form of a data augmentation which
we'll cover in the future and it's kind of like augmenting the input a little bit and jittering it and that makes it
harder for the neural nut to overfit to these concrete specific examples so by introducing all this noise it actually
like Pats out the examples and it regularizes the neural nut and that's one of the reasons why uh deceivingly as
a second order effect uh this is actually a regularizer and that has made it harder uh for us to remove the use of
batch normalization uh because basically no one likes this property that the the examples in the batch are coupled
mathematically and in the forward pass and at least all kinds of like strange uh results uh we'll go into some of that
in a second as well um and it leads to a lot of bugs and um and so on and so no
one likes this property uh and so people have tried to um deprecate the use of
bat normalization and move to other normalization techniques that do not couple the examples of a batch examples
are ler normalization instance normalization group normalization and so on and we'll come we'll come some these
uh later um but basically long story short bat normalization was the first
kind of normalization layer to be introduced it worked extremely well it happened to have this regularizing
effect it stabilized training and people have been trying to remove it and move
to some of the other normalization techniques uh but it's been hard because it it just works quite well and some of
the reason that it works quite well is again because of this regular rizing effect and because of the because it is
quite effective at um controlling the activations and their distributions uh so that's kind of like
the brief story of Bash normalization and I'd like to show you one of the other weird sort of outcomes of this
coupling so here's one of the strange outcomes that I only glossed over previously when I was evaluating the
loss on the validation set basically once we've trained a neural net we'd like to deploy it in some kind of a
setting and we'd like to be able to feed in a single individual example and get a prediction out from our neural net but
how do we do that when our neural net now in a forward pass estimates the statistics of the mean understand
deviation of a batch the neur lot expects batches as an input now so how do we feed in a single example and get
sensible results out and so the proposal in the batch normalization paper is the
following what we would like to do here is we would like to basically have a step after training that uh calculates
and sets the bach uh mean and standard deviation a single time over the training set and so I wrote this code
here in interest of time and we're going to call what's called calibrate the bachor statistics and basically what we
do is torch torch. nograd telling pytorch that none of this we will call
Dot backward on and it's going to be a bit more efficient we're going to take the training set get the pre activations
for every single training example and then one single time estimate the mean and standard deviation over the entire
training set and then we're going to get B and mean and B and standard deviation and now these are fixed numbers
estimating over the entire training set and here instead of estimating it
dynamically we are going to instead here use B and mean and here we're just going to use B
and standard deviation and so at test time we are going to fix these clamp them and use
them during inference and now you see that we get basically
identical result uh but the benefit that we've gained is that we can now also forward a single example because the
mean and standard deviation are now fixed uh sort of tensor that said nobody actually wants to
estimate this mean and standard deviation as a second stage after neural network training because everyone is
lazy and so this batch normalization paper actually introduced one more idea which is that we are can we can estimate
the mean and standard deviation in a running man running manner during training of the neuron nut and then we
can uh simply just have a single stage of training and on the side of that training we are estimating the running
mean and standard deviation so let's see what that would look like let me basically take the mean here that we are
estimating on the batch and let me call this B and mean on the I iteration um and then here this is BN
sdd um bnsd at I
okay uh and the mean comes here and the STD comes here so so far I've done
nothing I've just uh moved around and I created these EXT extra variables for the mean and standard deviation and I've
put them here so so far nothing has changed but what we're going to do now is we're going to keep running mean of
both of these values during training so let me swing up here and let me create a BN meanor running and I'm going to
initialize it at uh zeros and then BN STD running which I'll
initialize at once because um in the beginning because
of the way we ized W1 uh and B1 H pract will be roughly unit gion so the mean
will be roughly zero and a standard deviation roughly one so I'm going to initialize these that way but then here
I'm going to update these and in pytorch um uh these uh mean and standard
deviation that are running uh they're not actually part of the gradient based optimization we're never going to derive gradients with respect to them they're
they're updated on the side of training and so what we're going to do here is we're going to say with torch. nograd
telling pytorch that the update here is not supposed to be building out a graph because there will be no dot
backward but this running is basically going to be 0.99 uh9 times the current
Value Plus 0.001 times the um this value
this new mean and in the same way bnsd running will be mostly what it used to be
but it will receive a small update in the direction of what the current standard deviation
is and as you're seeing here this update is outside and on the side of the
gradient based optimization and it's simply being updated not using gradient scent it's just being updated using U
janky like Smooth um sort of uh running
mean Manner and so while the network is training and these pre activations are
sort of changing and shifting around during during back propagation we are keeping track of the typical mean and
standard deviation and we're estimating them once and when I run
this now I'm keeping track of this in the running Manner and what we're hoping for of course is that the me BN meore
running and BN meore STD are going to be very similar to the ones that we
calculated here before and that way we don't need a second stage because we've sort of combined the two stages and
we've put them on the side of each other if you want to look at it that way and this is how this is also implemented in
The Bash normalization uh layer impi torch so during training um the exact
same thing will happen and then later when you're using inference it will use the estimated running mean of both the
mean and standard deviation of those hidden States so let's wait for the optimization to converge and hopefully
the running mean and standard deviation are roughly equal to these two and then we can simply use it here and we don't
need this stage of explicit calibration at the end okay so the optimization finished I'll rerun the explicit
estimation and then the B and mean from the explicit estimation is here and B
and mean from the running estimation during the during the optimization you
can see is very very similar it's not identical but it's pretty close and the same way BN STD is this
and BN STD running is this and so you can see that once again they are fairly
similar values not identical but pretty close and so then here instead of being
mean we can use the BN mean running instead of bnsd we can use bnsd
running and uh hopefully the validation loss will not be impacted too much okay so it's basically identical
and this way we've eliminated the need for this explicit stage of calibration because we are doing it in line over
here okay so we're almost done with batch normalization there are only two more notes that I'd like to make number
one I've skipped a discussion over what is this plus Epsilon doing here this Epsilon is usually like some small fixed
number for example one5 by default and what it's doing is that it's basically preventing a division by zero in the
case that the variance over your batch is exactly zero in that case uh here we
normally have a division by zero but because of the plus Epsilon uh this is going to become a small number in the denominator instead and things will be
more well behaved so feel free to also add a plus Epsilon here of a very small number it doesn't actually substantially
change the result I'm going to skip it in our case just because uh this is unlikely to happen in our very simple example here and the second thing I want
you to notice is that we're being wasteful here and it's very subtle but right here where we are adding the bias
into H preact these biases now are actually useless because we're adding
them to the H preact but then we are calculating the mean for every one of these neurons and subtracting it so
whatever bias you add here is going to get subtracted right here and so these
biases are not doing anything in fact they're being subtracted out and they don't impact the rest of the calculation
so if you look at b1. grad it's actually going to be zero because it's being subtracted out and doesn't actually have
any effect and so whenever you're using bash normalization layers then if you have any weight layers before like a
linear or a c or something like that you're better off coming here and just like not using bias so you don't want to
use bias and then here you don't want to add it because it's that spirous instead
we have this B normalization bias here and that b rization bias is now in charge of the biasing of this
distribution instead of this B1 that we had here originally and so uh basically
bash normalization layer has its own bias and there's no need to have a bias in the layer before it because that bias
is going to be subtracted out anyway so that's the other small detail to be careful with sometimes it's not going to
do anything catastrophic this B1 will just be useless it will never get any gradient uh it will not learn it will
stay constant and it's just wasteful but it doesn't actually really uh impact anything otherwise okay so I rearranged

当然可以！我来帮你分段总结这段 **Batch Normalization** 部分的内容，方便你理解，结构清楚：

---

### 🟢 1️⃣ 背景

* 以前为了避免 tanh 饱和，我们手动调整 W1 scale
* 但是如果是「多层网络」，手动调很麻烦，难以维持激活分布

---

### 🟢 2️⃣ BatchNorm 是什么？

* 2015 年 Google 论文提出
* 目的是 **自动让激活分布「标准正态化」**（mean=0, std=1）
* 作用：

  * 避免 tanh 饱和 / 梯度消失
  * 训练变稳定
  * 可以训练更深的网络

---

### 🟢 3️⃣ BatchNorm 怎么做？

在 hidden 层：

```python
hpreact → (先算 mean, std)
        → (hpreact - mean) / std
        → (乘以 gain, 加上 bias)
        → 传入 tanh
```

---

### 🟢 4️⃣ 为什么加 gain + bias？

* 如果 **一直标准化到 mean=0, std=1**，网络没法灵活学习
* 加 gain（可学参数）+ bias（可学参数），允许网络 **微调激活分布**

---

### 🟢 5️⃣ BatchNorm 好处

✅ **自动调节激活分布**，避免饱和
✅ **无需手动调 W scale**
✅ **训练更稳定、更快**
✅ **间接起到正则化（regularization）作用**

---

### 🟢 6️⃣ BatchNorm 的副作用

* 以前「一个样本 → 一次 forward」
* BatchNorm 会在 **batch 内计算 mean/std → 不同 batch 结果不同**

→ 结果是：

```
同一个样本 → 不同 batch → 可能输出不同
```

→ 有点像 **数据增强**，也是一种正则化
→ 但副作用是，**样本之间「耦合」**，不再独立

---

### 🟢 7️⃣ 推理（test）怎么办？

训练时：

```
batch 内动态计算 mean/std
```

推理时：

```
只能传一个样本，batch mean/std 不存在
```

→ 解决方案：

✅ 训练时 **running mean/std**
✅ 推理用 running mean/std（固定值）

---

### 🟢 8️⃣ 实现细节

1️⃣ running mean/std 是用 moving average 算的：

```
new_mean = 0.999 * old_mean + 0.001 * current_batch_mean
```

2️⃣ Epsilon:

```
防止除 0，加一个小数 ε
```

---

### 🟢 9️⃣ Bias 的小细节

* 如果用了 BatchNorm，前一层（Linear / Conv）不需要 bias 了
* 因为：

```
(hpreact + b1 - mean) → b1 被减掉
```

* 结论：

```
用了 BatchNorm → Linear / Conv 不加 bias
```

---

### 🟢 10️⃣ 结果验证

实验结果：

* 用 BatchNorm 后 val loss ≈ 2.10，和手动调整 W scale 差不多
* 但是 BatchNorm **更稳，更自动，更好迁移到大网络**

---

### 🟢 11️⃣ 什么时候更有用？

* MLP 1 层效果差别不大
* 深层网络（ResNet、ConvNet）效果显著
* 自动解决激活分布问题，避免爆炸 / 消失

---

### 🟢 总结一句话：

👉 BatchNorm 的核心思想是：
**动态调整隐藏层激活 → 标准化 → 避免饱和 / 梯度消失 → 稳定训练 → 间接正则化**

---

如果你需要，我还可以帮你：

✅ 画一张「BatchNorm 流程图」
✅ 总结「BatchNorm vs 不加 BN 时激活分布对比图」

要不要？ 🚀✨


hpreact.mean(0, keepdim=True)

解释一下

当然可以，来详细解释：

```python
hpreact.mean(0, keepdim=True)
```

拆解一下：

---

### hpreact 是什么？

```python
hpreact = embcat @ W1 + b1
```

* 也就是 **隐藏层的 pre-activation 值**
* shape: `[batch_size, n_hidden]`，
  例子中就是 `[32, 200]` → 32 个样本，每个样本 200 个 neuron

---

### mean(0) 是什么意思？

```python
.mean(0)
```

* 表示对 **维度 0** 取平均
* 维度 0 是 batch 维度 → 也就是 **对 32 个样本求平均**

---

### keepdim=True 是什么意思？

* 如果不写 `keepdim=True`，求平均后会降维，shape 变成 `[200]`

* 如果加上 `keepdim=True`，shape 保持 `[1, 200]`

为什么要加 `keepdim=True`？
✅ 这样后面可以方便 **broadcast**（广播） 用来做：

```python
(hpreact - mean) / std
```

---

### 举个例子：

假设：

```
hpreact.shape = [32, 200]
```

执行：

```
hpreact.mean(0, keepdim=True).shape → [1, 200]
```

结果是：

```
[mean_of_neuron_1, mean_of_neuron_2, ..., mean_of_neuron_200]
```

→ 每个 neuron 在整个 batch 上的平均激活值

---

### 为什么要这么做？

✅ 这是 **BatchNorm 的核心步骤之一**：

```
hpreact_normalized = (hpreact - mean) / std
```

→ 把 neuron 的激活「标准化」，mean=0，std=1

---

### 总结一句话：

👉 `hpreact.mean(0, keepdim=True)` 是对 **每个 neuron**，在整个 batch 内 **求平均激活**，shape 保持 `[1, 200]`，用于 BatchNorm 归一化。

---

如果你需要，我还可以举一个 **具体例子（用小矩阵演示 mean(0) 的效果）**，要不要？ 🚀✨

需要

太好了！我来举个**具体小例子**，帮你完全搞懂 `hpreact.mean(0, keepdim=True)` 的效果：

---

假设：

```python
hpreact = torch.tensor([
    [1.0, 2.0, 3.0],   # 第 1 个样本
    [4.0, 5.0, 6.0],   # 第 2 个样本
    [7.0, 8.0, 9.0]    # 第 3 个样本
])
```

shape：

```
[3, 3] → 3 个样本，每个样本 3 个 neuron
```

---

### 如果我们执行：

```python
hpreact.mean(0)
```

→ 对 **dim=0** 取平均（跨 batch 取平均）

计算过程：

| neuron       | mean 计算               |
| ------------ | --------------------- |
| 第 1 个 neuron | (1 + 4 + 7) / 3 = 4.0 |
| 第 2 个 neuron | (2 + 5 + 8) / 3 = 5.0 |
| 第 3 个 neuron | (3 + 6 + 9) / 3 = 6.0 |

结果：

```python
tensor([4.0, 5.0, 6.0])  # shape: [3]
```

---

### 如果加上 keepdim=True：

```python
hpreact.mean(0, keepdim=True)
```

结果：

```python
tensor([[4.0, 5.0, 6.0]])  # shape: [1, 3]
```

---

### 为什么要 keepdim？

因为后面要做：

```python
hpreact - mean
```

而 hpreact shape 是 `[batch_size, n_hidden]`，
如果 mean 维度是 `[1, n_hidden]`，
broadcast 机制可以自动帮我们做 **按 neuron 减均值**，对 batch 中每个样本都生效。

---

### 最后总结：

```python
hpreact.mean(0, keepdim=True)
```

✅ 计算每个 neuron（200 个 neuron）在 batch 内的平均值
✅ shape 变成 `[1, 200]`，方便后面 broadcast
✅ 目的是标准化 neuron 的激活，达到 BatchNorm 的效果

---

如果你还想，我也可以写一段 PyTorch 代码 + 输出结果，直接跑给你看，效果更直观，要不要？ 🚀✨

```python
  with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```    

解释一下

当然可以，这一句是在「训练时动态更新 BatchNorm running mean/std」，我来拆开讲：

---

### 背景

* BatchNorm 推理时需要 **固定的 mean/std**
* 有两种做法：

✅ 训练完单独跑全训练集 → 校准 mean/std
✅ **训练过程中** 用「滑动平均（running average）」自动累计 → 更方便

---

### 这段代码就是第二种方法：**running mean/std**

---

### 逐行解释：

```python
with torch.no_grad():
```

✅ 说明这一段代码 **不参与反向传播**
✅ 只是更新统计量，不用求梯度

---

```python
bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
```

**逐项解释：**

* `bnmeani` 是当前 batch 的 mean
* `bnstdi` 是当前 batch 的 std

---

更新逻辑是：

```python
new_bnmean_running = 99.9% * 原值 + 0.1% * 当前 batch 值
```

\= **滑动平均**
\= 类似于「指数移动平均（EMA）」
\= 越训练到后面，`bnmean_running` 越接近整个训练集的「真实」 mean/std

---

### 为什么用 0.999 和 0.001？

* 0.999 → 保留之前累计的经验
* 0.001 → 缓慢吸收新 batch 信息
* 这样可以避免 batch 间差异大，统计量抖动太厉害
  → 训练更稳定

---

### 总结一句话：

👉 这一段代码就是：**训练过程中动态维护 BatchNorm 的 running mean/std，方便后面推理用**。

---

如果你需要，我可以帮你画一张「running mean/std 更新过程示意图」，要不要？ 🚀✨


# batch normalization: summary

the code a little bit with comments and I just wanted to give a very quick summary of The Bash normalization layer
we are using bash normalization to control the statistics of activations in the neural net it is common to sprinkle
bash normalization layer across the neural net and usually we will place it after layer that have multiplications
like for example a linear layer or convolutional layer which we may cover in the future now the bat normalization
internally has parameters for the gain and the bias and these are trained using
back propagation it also has two buffers the buffers are the mean and the
standard deviation the running mean and the running mean of the standard deviation and these are not trained
using back propagation these are trained using this uh janky update of kind of like a running mean
update so um these are sort of the parameters and the buffers of Bator layer and then
really what it's doing is it's calculating the mean and a standard deviation of the activations uh that are
feeding into the Bator layer over that batch then it's centering that batch to
be unit gion and then it's offsetting and scaling it by the Learned bias and
gain and then on top of that it's keeping track of the mean and standard deviation of the inputs and it's
maintaining this running mean and standard deviation and this will later be used at inference so that we don't
have to reestimate the mean stand deviation all the time and in addition that allows us to basically forward
individual examples at test time so that's the bash normalization layer it's a fairly complicated layer um but this
is what it's doing internally now I wanted to show you a little bit of a real example so you can search resnet

当然可以，来帮你总结这一段 **Batch Normalization: summary**，中文解释如下：

---

### 🟢 BatchNorm 的作用

✅ **控制神经网络内部激活值的分布**
✅ 让激活保持「标准正态分布」 → 均值 0，标准差 1

---

### 🟢 BatchNorm 通常放在哪里？

✅ 通常放在 **带乘法的层后面**，比如：

* Linear 层（全连接层）
* Conv 层（卷积层）
* 以后会讲 Residual、Conv

---

### 🟢 BatchNorm 里有哪些参数？

1️⃣ **可训练参数**（会反向传播学习）：

* gain（缩放因子，通常初始化为 1）
* bias（偏移量，通常初始化为 0）

---

2️⃣ **非训练参数**（不会反向传播）：

* running mean
* running std（标准差）

→ 用「滑动平均」动态更新（running mean update）

---

### 🟢 BatchNorm 做了什么操作？

* 对输入 activations 计算 batch 内 mean/std
* 把输入 **减去 mean，除以 std** → 变成标准正态分布
* 然后 **乘 gain，加 bias** → 让网络有自由调整空间

---

### 🟢 为什么保存 running mean/std？

✅ 推理时（inference）不希望每次动态算 mean/std

✅ 可以用 **running mean/std**，保证推理时：

```
- 可以单样本 forward
- 不受 batch 影响
- 结果稳定
```

---

### 🟢 总结一句话

👉 **BatchNorm 层的作用是标准化隐藏层激活，提升训练稳定性，帮助训练深层网络，并且通过 running mean/std 让推理过程稳定**。

---

如果你需要，我还可以整理一个「BatchNorm 内部计算流程图」或者「BatchNorm 参数和 buffer 列表」，要不要？ 🚀✨


# real example: resnet50 walkthrough

which is a residual neural network and these are common types of neural networks used for image
classification and of course we haven't come resnets in detail so I'm not going to explain all the pieces of it but for
now just note that the image feeds into a reset on the top here and there's many many layers with repeating structure all
the way to predictions of what's inside that image this repeating structure is made up of these blocks and these blocks
are just sequentially stacked up in this deep neural network now the code for
this uh the block basically that's used and repeated sequentially in series is
called this bottleneck block bottleneck block and there's a lot here this is all
pych and of course we haven't covered all of it but I want to point out some small pieces of it here in the init is
where we initialize the neuronet so this code of block here is basically the kind of stuff we're doing here we're initializing all the layers and in the
forward we are specifying how the neuron lot acts once you actually have the input so this code here is along the
lines of what we're doing here and now these blocks are replicated
and stacked up serially and that's what a residual Network would be and so
notice What's Happening Here com one um these are convolution layers and these
convolution layers basically they're the same thing as a linear layer except convolutional layers don't apply um
convolutional layers are used for images and so they have SP structure and basically this linear multiplication and
bias offset are done on patches instead of math instead of the full input so
because these images have structure spatial structure convolutions just basically do WX plus b but they do it on
overlapping patches of the input but otherwise it's WX plus P then we have the norm layer which by
default here is initialized to be a bash Norm in 2D so two- dimensional bash normalization layer and then we have a
nonlinearity like reu so instead of uh here they use reu we are using 10 in
this case but both both are just nonlinearities and you can just use them relatively interchangeably for very deep
networks re typically empirically work a bit better so see the motif that's being
repeated here we have convolution bat normalization reu convolution bat normalization re Etc and then here this
is residual connection that we haven't covered yet but basically that's the exact same pattern we have here with we
have a weight layer like a convolution or like a linear layer bash
normalization and then 10h which is nonlinearity but basically a weight
layer a normalization layer and nonlinearity and that's the motif that you would be stacking up when you create
these deep neural networks exactly as it's done here and one more thing I'd like you to notice is that here when
they are initializing the com layers like com 1 by one the depth for that is
right here and so it's initializing an nn. Tod which is a convolution layer in pytorch and there's a bunch of keyword
arguments here that I'm not going to explain yet but you see how there's bias equals false the bias equals false is
exactly for the same reason as bias is not used in our case you see how I eras
the use of bias and the use of bias is spous because after this weight layer there's a bash normalization and The
Bash normalization subtracts that bias and then has its own bias so there's no need to introduce these spous parameters
it wouldn't hurt performance it's just useless and so because they have this motif of C Bast umbrell they don't need
a bias here because there's a bias inside here so by the way this example
here is very easy to find just do resonet pie torch and uh it's this example here so
this is kind of like the stock implementation of a residual neural network in pytorch and uh you can find
that here but of course I haven't covered many of these parts yet and I would also like to briefly descend into
the definitions of these pytorch layers and the the parameters that they take now instead of a convolutional layer
we're going to look at a linear layer uh because that's the one that we're using here this is a linear layer and I
haven't cover covered convolutions yet but as I mentioned convolutions are basically linear layers except on
patches so a linear layer performs a WX plus b except here they're calling the W
A transpose um so to calcul WX plus b very much like we did here to initialize this
layer you need to know the fan in the fan out and that's so that they can
initialize this W this is the fan in and the fan out so they know how how big the
weight Matrix should be you need to also pass in whether you whether or not you want a bias and if you set it to false
then no bias will be uh inside this layer um and you may want to do that
exactly like in our case if your layer is followed by a normalization layer such as batch
Norm so this allows you to basically disable a bias now in terms of the initial ation if we swing down here this
is reporting the variables used inside this linear layer and our linear layer
here has two parameters the weight and the bias in the same way they have a weight and a bias and they're talking
about how they initialize it by default so by default P will initialize your weights by taking the
Fanon and then um doing one over fanin square root and then instead of a normal
distribution they are using a uniform distribution so it's very much the same thing but
they are using a one instead of 5 over three so there's no gain being calculated here the gain is just one but
otherwise is exactly one over the square root of fan in exactly as we have
here so one over the square root of K is the is the scale of the weights but when
they are drawing the numbers they're not using a gussion by default they're using a uniform distribution by default and so
they draw uniformly from negative of K to squ of K but it's the exact same thing and the
same motivation from for with respect to what we've seen in this lecture and the
reason they're doing this is if you have a roughly gsan input this will ensure that out of this layer you will have a
roughly Gan output and you you basically achieve that by scaling the weights by
one over the square root of fan in so that's what this is doing and then the second thing is the
bash normalization layer so let's look at what that looks like in pytorch so here we have a onedimensional b
normalization layer exactly as we are using here and there are a number of keyword arguments going into it as well
so we need to know the number of features uh for us that is 200 and that is needed so that we can initialize
these parameters here the gain the bias and the buffers for the running uh mean
and standard deviation then they need to know the value of Epsilon here and by default
this is one5 you don't typically change this too much then they need to know the momentum
and the momentum here as they explain is basically used for these uh running mean and running standard deviation so by
default the momentum here is 0.1 the momentum we are using here in this example is
0.001 and basically rough you may want to change this sometimes and roughly
speaking if you have a very large batch size then typically what you'll see is that when you estimate the mean and the
standard deviation for every single batch size if it's large enough you're going to get roughly the same result
and so therefore you can use slightly higher momentum like 0.1 but for a batch size as small as 32
the mean and standard deviation here might take on slightly different numbers because there's only 32 examples we are using to estimate the mean and standard
deviation so the value is changing around a lot and if your momentum is 0.1
that that might not be good enough for this value to settle and um converge to
the actual mean and standard deviation over the entire training set and so basically if your batch size is very
small uh momentum of 0.1 is potentially dangerous and it might make it so that the running uh mean and stand deviation
are is thrashing too much during training and it's not actually converging
properly uh aine equals true determines whether this batch normalization layer has these learnable Aline parameters the
uh the gain and the bias and this is almost always kept to true I'm not actually sure why you would want to
change this to false um then track running stats is determining
whether or not B rization layer of pytorch will be doing this and um one reason you may you may
want to skip the running stats is because you may want to for example estimate them at the end as a stage two
like this and in that case you don't want the bat normalization layer to be doing all this extra compute that you're not going to
use and uh finally we need to know which device we're going to run this bash normalization on a CPU or a GPU and what
the data type should be uh half Precision single Precision double precision and so on so that's the bat normalization layer
otherwise they link to the paper is the same formula we've implemented and everything is the same exactly as we've
done here okay so that's everything that I wanted to cover for this lecture really

# summary of the lecture

what I wanted to talk about is the importance of understanding the activations and the gradients and their statistics in neural networks and this
becomes increasingly important especially as you make your neural networks bigger larger and deeper we looked at the distributions basically
at the output layer and we saw that if you have two confident mispredictions because the activations are too messed
up at the last layer you can end up with these hockey stick losses and if you fix this you get a better loss at the end of
training because your training is not doing wasteful work then we also saw that we need to control the activations
we don't want them to uh you know squash to zero or explode to infinity and
because that you can run into a lot of trouble with all of these uh nonlinearities and these neural Nets and basically you want everything to be
fairly homogeneous throughout the neural net you want roughly goshan activations throughout the neural net let me talked
about okay if we want roughly Gan activations how do we scale these weight matrices and biases during
initialization of the neural nut so that we don't get um you know so everything is as controlled as
possible um so that give us a large boost in Improvement and then I talked
about how that strategy is not actually uh Poss for much much deeper neural nuts
because um when you have much deeper neural nuts with lots of different types of layers it becomes really really hard
to precisely set the weights and the biases in such a way that the activations are roughly uniform
throughout the neural nut so then I introduced the notion of a normalization layer now there are many normalization
layers that that people use in practice bat normalization layer normalization instance normalization group
normalization we haven't covered most of them but I've introduced the first one and also the one that I believe came out
first and that's called Bat normalization and we saw how bat normalization Works uh this is a layer
that you can sprinkle throughout your deep neural net and the basic idea is if you want roughly gsh in activations well
then take your activations and um take the mean and the standard deviation and Center your data and you can do that
because the centering operation is differentiable but and on top of that we
actually had to add a lot of bells and whistles and that gave you a sense of the complexities of the batch normalization layer because now we're
centering the data that's great but suddenly we need the gain and the bias and now those are
trainable and then because we are coupling all of the training examples now suddenly the question is how do you
do the inference where to do to do the inference we need to now estimate these
um mean and standard deviation once uh or the entire training set and then use
those at inference but then no one likes to do stage two so instead we fold everything everything into the bat
normalization later during training and try to estimate these in the running manner so that everything is a bit
simpler and that gives us the bat normalization layer um and as I
mentioned no one likes this layer it causes a huge amount of bugs um and
intuitively it's because it is coupling examples um in the for pass of a neural nut and uh I've shot myself in the foot
with this layer over and over again in my life and I don't want you to suffer the same uh so basically try to avoid it
as much as possible uh some of the other alternatives to these layers are for example group normalization or layer
normalization and those have become more common uh in more recent deep learning
uh but we haven't covered those yet uh but definitely bash normalization was very influential at the time when it
came out in roughly 2015 because it was kind of the first time that you could train reliably uh much deeper neural
nuts and fundamentally the reason for that is because this layer was very effective at controlling the statistics
of the activations in the neural nut so that's the story so far and um that's
all I wanted to cover and in the future lectures hopefully we can start going into recurrent R Nets and um recurring
neural Nets as we'll see are just very very deep networks because you uh you unroll the loop and uh when you actually
optimize these neurals and that's where a lot of this um analysis around the activation
statistics and all these normalization layers will become very very important for uh good performance so we'll see
that next time bye okay so I lied I would like us to do one more summary here as a bonus and I think it's useful

# just kidding: part2: PyTorch-ifying the code

as to have one more summary of everything I've presented in this lecture but also I would like us to start by torify our code a little bit so
it looks much more like what you would encounter in PCH so you'll see that I will structure our code into these
modules like a link uh module and a borm module and I'm
putting the code inside these modules so that we can construct neural networks very much like we would construct them in pytorch and I will go through this in
detail so we'll create our neural net then we will do the optimization loop as
we did before and then the one more thing that I want to do here is I want to look at the activation statistics
both in the forward pass and in the backward pass and then here we have the evaluation and sampling just like before
so let me rewind all the way up here and and go a little bit slower so here I creating a linear layer you'll notice
that torch.nn has lots of different types of layers and one of those layers is the linear layer torch. n. linear
takes a number of input features output features whether or not we should have a bias and then the device that we want to
place this layer on and the data type so I will emit these two but otherwise we
have the exact same thing we have the fan in which is the number of inputs fan out the number of outputs and whether or
not we want to use a bias and internally inside this layer there's a weight and a bias if you'd like it it
is typical to initialize the weight using um say random numbers drawn from
aashan and then here's the coming initialization um that we discussed already in this lecture and that's a
good default and also the default that I believe pytor chooses and by default the bias is usually initialized to zeros now
when you call this module uh this will basically calculate W * X plus b if you
have a b and then when you also call that parameters on this module it will return uh the tensors that are the
parameters of this layer now next we have the bash normalization layer so I've written that here and this is very
similar to pytorch nn. bashor 1D layer as shown here so I'm kind of um taking these
three parameters here the dimensionality the Epsilon that we will use in the division and the momentum that we will
use in keeping track of these running stats the running mean and the running variance um now py actually takes quite
a few more things but I'm assuming some of their settings so for us Aline will be true that means that we will be using
a gamma and beta after the normalization the track running stats will be true so we will be keeping track of the running
mean and the running variance in the in the bat Norm our device by default is the CPU and the data type by default is
uh float float 32 so those are the defaults otherwise
uh we are taking all the same parameters in this bachom layer so first I'm just saving them now here's something new
there's a doc training which by default is true and pytorch andn modules also have this attribute. training and that's
because many modules in borm is included in that have a different Behavior
whether you are training your interet and or whether you are running it in an evaluation mode and calculating your
evaluation loss or using it for inference on some test examples and bashor is an example of this because
when we are training we are going to be using the mean and the variance estimated from the current batch but
during inference we are using the running mean and running variance and so also if we are training we are updating
mean and variance but if we are testing then these are not being updated they're kept fixed and so this flag is necessary
and by default true just like in pytorch now the parameters of B 1D are the gamma and the beta
here and then the running mean and running variance are called buffers in pyto
nomenclature and these buffers are trained using exponential moving average
here explicitly and they are not part of the back propagation and stochastic radient descent so they are not sort of
like parameters of this layer and that's why when we C when we have a parameters here we only return gamma and beta we do
not return the mean and the variance this is trained sort of like internally here um every forward pass using
exponential moving average so that's the initialization now in a forward pass if
we are training then we use the mean and the variance estimated by the batch let me pull up the paper
here we calculate the mean and the variance now up above I was estimating
the standard deviation and keeping track of the standard deviation here in the running standard deviation instead of
running variance but let's follow the paper exactly here they calculate the variance which is the standard deviation
squared and that's what's get track of in a running variance instead of a running standard
deviation uh but those two would be very very similar I believe um if we are not training then
we use running mean and variance we normalize and then here I am calculating
the output of this layer and I'm also assigning it to an attribute called out now out is something that I'm using in
our modules here uh this is not what you would find in pytorch we are slightly deviating from it I'm creating a DOT out
because I would like to very easily um maintain all those variables so that we can create statistics of them and plot
them but pytorch and modules will not have a do out attribute and finally here
we are updating the buffers using again as I mentioned exponential moving average uh provide given the provided
momentum and importantly you'll notice that I'm using the torch. nogra context manager and I doing this because if we
don't use this then pytorch will start building out an entire computational graph out of these tensors because it is
expecting that we will eventually call Dot backward but we are never going to be calling dot backward on anything that
includes running mean and running variance so that's why we need to use this context manager so that we are not
um sort of maintaining them using all this additional memory um so this will make it more efficient and it's just
telling pyour that there will no backward we just have a bunch of tensors we want to update them that's it and
then we return okay now scrolling down we have the 10h layer this is very very similar
to uh torch. 10h and it doesn't do too much it just calculates 10 as you might
expect so uh that's torch. 10h and uh there's no parameters in this layer but
because these are layers um it now becomes very easy to sort of like stack them up into uh basically just a list um
and uh we can do all the initializations that we're used to so we have the initial sort of embedding Matrix we have
our layers and we can call them sequentially and then again with Tor no grb but there's some initializations
here so we want to make the output softmax a bit less confident like we saw and in addition to that because we are
using a six layer multi-layer percep on here so you see how I'm stacking linear 10age linear Tage Etc uh I'm going to be
using the gain here and I'm going to play with this in a second so you'll see how uh when we change this what happens
to the statistics finally the parameters are basically the embedding Matrix and all
the parameters in all the layers and notice here I'm using a double list apprehension if you want to call it that
but for every layer in layers and for every parameter in each of those layers we are just stacking up all those piece
uh all those parameters now in total we have 46,000 um
parameters and I'm telling P that all of them require
gradient then here uh we have everything here we are actually mostly used to uh
we are sampling a batch we are doing a forward pass the forward pass now is just the linear application of all the
layers in order followed by the cross entropy and then in the backward pass you'll notice that for every single
layer I now iterate over all the outputs and I'm telling pytorch to retain the gradient of them and then here we are
already used to uh all the all the gradient set To None do the backward to fill in the gradients uh do an update
using stochastic gradient sent and then uh track some statistics and then I am going to break after a single iteration

# viz #1: forward pass activations statistics

now here in this cell in this diagram I I'm visualizing the histogram the histograms of the for pass activations
and I'm specifically doing it at the 10 each layers so iterating over all the layers except for the very last one
which is basically just the U soft Max layer um if it is a 10h layer and I'm
using a 10h layer just because they have a finite output netive 1 to 1 and so it's very easy to visualize here so you
see 1 to one and it's a finite range and easy to work with I take the out tensor
from that layer into T and then I'm calculating the mean the standard deviation and the percent saturation of
T and the way I Define the percent saturation is that t. absolute value is greater than 97 so that means we are
here at the tals of the 10 H and remember that when we are in the tales of the 10 H that will actually stop
gradients so we don't want this to be too high now here I'm calling torch.
histogram and then I am plotting this histogram so basically what this is doing is that every different type of
layer and they have a different color we are looking at how many um values in
these tensors take on any of the values Below on this axis here so the first
layer is fairly saturated uh here at 20% so you can see that it's got Tails here
but then everything sort of stabilizes and if we had more layers here it would actually just stabilize at around the standard deviation of about 65 and the
saturation would be roughly 5% and the reason that the stabilizes and gives us a nice distribution here is because gain
is set to 5 over3 now here this gain you see that by
default we initialize with 1 /un of fan in but then here during initialization I
come in and I erator all the layers and if it's a linear layer I boost that by the gain now we saw that one so
basically if we just do not use a gain then what happens if I redraw this you
will see that the standard deviation is shrinking and the saturation is coming
to zero and basically what's happening is the first layer is you know pretty decent but then further layers are just
kind of like shrinking down to zero and it's happening slowly but it's shrinking to zero and the reason for that is when
you just have a sandwich of linear layers alone then a then initializing
our weights in this manner we saw previously would have conserved the standard deviation of one but because we
have this interspersed 10 in layers in there these 10h layers are squashing
functions and so they take your distribution and they slightly squash it and so some gain is necessary to keep
expanding it to fight the squashing so it just turns out that 5
over3 is a good value so if we have something too small like one we saw that things will come toward zero but if it's
something too high let's do two then here we see that um
well let me do something a bit more extreme because so it's a bit more visible let's try three okay so we see here that the
saturations are going to be way too large okay so three would create way too
saturated activations so 5 over3 is a good setting for a sandwich of linear
layers with 10h activations and it roughly stabilizes the standard deviation at a reasonable point now
honestly I have no idea where 5 over3 came from in pytorch um when we were looking at the coming initialization um
I see empirically that it stabilizes this sandwich of linear an 10age and that the saturation is in a good range
um but I don't actually know if this came out of some math formula I tried searching briefly for where this comes
from uh but I wasn't able to find anything uh but certainly we see that empirically these are very nice ranges
our saturation is roughly 5% which is a pretty good number and uh this is a good
setting of The gain in this context similarly we can do the exact same thing with the gradients so here is a very

# viz #2: backward pass gradient statistics

same Loop if it's a 10h but instead of taking a layer do out I'm taking the grad and then I'm also showing the mean
and the standard deviation and I'm plotting the histogram of these values and so you'll see that the gradient
distribution is uh fairly reasonable and in particular what we're looking for is that all the different layers in this
sandwich has roughly the same gradient things are not shrinking or exploding so
uh we can for example come here and we can take a look at what happens if this gain was way too small so this was
0.5 then you see the first of all the activations are shrinking to zero but also the gradients are doing something
weird the gradients started out here and then now they're like expanding out and similarly if we for example have
a too high of a gain so like three then we see that also the gradients have there's some asymmetry
going on where as you go into deeper and deeper layers the activation CS are changing and so that's not what we want
and in this case we saw that without the use of batro as we are going through right now we had to very carefully set
those gains to get nice activations in both the forward pass and the backward pass now before we move on to bat

# the fully linear case of no non-linearities

normalization I would also like to take a look at what happens when we have no 10h units here so erasing all the 10
nonlinearities but keeping the gain at 5 over3 we now have just a giant linear
sandwich so let's see what happens to the activations as we saw before the correct gain here
is one that is the standard deviation preserving gain so 1.66 7 is too high
and so what's going to happen now is the following uh I have to change this to be linear so we are because there's no more
10h layers and let me change this to linear as well so what we're seeing is um the
activations started out on the blue and have by layer four become very diffuse
so what's happening to the activations is this and with the gradients on the top layer the activation the gradient
statistics are the purple and then they diminish as you go down deeper in the layers and so basically you have an
asymmetry like in the neuron net and you might imagine that if you have very deep neural networks say like 50 layers or
something like that this just uh this is not a good place to be uh so that's why
before bash normalization this was incredibly tricky to to set in particular if this is too large of a
gain this happens and if it's too little of a gain then this happens so the opposite
of that basically happens here we have a um shrinking and a uh diffusion
depending on which direction you look at it from and so certainly this is not what you want and in this case the
correct setting of The gain is exactly one just like we're doing at initialization and then we see that the
uh statistics for the forward and a backward pass are well behaved and so the reason I want to show you this is
that basically like getting neural nness to train before these normalization layers and before the use of advanced
optimizers like adom which we still have to cover and residual connections and so on uh training neurs basically looked
like this it's like a total Balancing Act you have to make sure that everything is precisely orchestrated and
you have to care about the activations and the gradients and their statistics and then maybe you can train something uh but it was it was basically
impossible to train very deep networks and this is fundamentally the the reason for that you'd have to be very very
careful with your initialization um the other point here is you might be asking yourself by the
way I'm not sure if I covered this why do we need these 10h layers at all uh
why do we include them and then have to worry about the gain and uh the reason for that of course is that if you just have a stack of linear layers then
certainly we're getting very easily nice activations and so on uh but this is just massive linear sandwich and it
turns out that it collapses to a single linear layer in terms of its uh representation power so if you were to
plot the output as a function of the input you're just getting a linear function no matter how many linear layers you stack up you still just end
up with a linear transformation all the WX plus BS just collapse into a large WX
plus b with slightly different W's and slightly different B um but interestingly even though the forward
pass collapses to just a linear layer because of back propagation and uh the dynamics of the backward pass the
optimization natur is not identical you actually end up with uh all kinds of interesting um Dynamics in the backward
pass uh because of the uh the way the chain Ru is calculating it and so optimizing a linear layer by itself and
optimizing a sandwich of 10 linear layers in both cases those are just a linear transformation in the forward
pass but the training Dynamics would be different and there's entire papers that analyze in fact like infinitely layered
uh linear layers and and so on and so there's a lot of things to that you can play with there uh but basically the tal
linearities allow us to um turn this sandwich from just a
linear uh function into uh a neural network that can in principle um
approximate any arbitrary function okay so now I've reset the code to use the linear tanh sandwich like before and I

# viz #3: parameter activation and gradient statistics

reset everything so the gain is 5 over three uh we can run a single step of optimization and we can look at the
activation statistics of the forward pass and the backward pass but I've added one more plot here that I think is
really important to look at when you're training your neural nuts and to consider and ultimately what we're doing
is we're updating the parameters of the neural nut so we care about the parameters and their values and their
gradients so here what I'm doing is I'm actually iterating over all the parameters available and then I'm only
um restricting it to the two-dimensional parameters which are basically the weights of the linear layers and I'm
skipping the biases and I'm skipping the um gamas and the betas in the bom just
for Simplicity but you can also take a look at those as well but what's happening with the weights is um
instructive by itself so here we have all the different weights their shapes uh so this is the
embedding layer the first linear layer all the way to the very last linear layer and then we have the mean the
standard deviation of all these parameters the histogram and you can see that actually doesn't look that amazing
so there's some trouble in Paradise even though these gradients looked okay there's something weird going on here
I'll get to that in a second and the last thing here is the gradient to data ratio so sometimes I like to visualize
this as well because what this gives you a sense of is what is the scale of the gradient compared to the scale of the
actual values and this is important because we're going to end up taking a step update um that is the learning rate
times the gradient onto the data and so if the gradient has too large of magnitude if the numbers in there are
too large compared to the numbers in data then you'd be in trouble but in this case the gradient to data is our
low numbers so the values inside grad are 1,000 times smaller than the values
inside data in these weights most of them now notably that is not true about
the last layer and so the last layer actually here the output layer is a bit of a troublemaker in the way that this
is currently arranged because you can see that the um last layer here in pink
takes on values that are much larger than some of the values inside um inside
the neural nut so the standard deviations are roughly 1 and3 throughout except for the last last uh layer which
actually has roughly one -2 standard deviation of gradients and so the gradients on the last layer are
currently about 100 times greater sorry 10 times greater than all the other
weights inside the neural net and so that's problematic because in the simple stochastic rting theend setup you would
be training this last layer about 10 times faster than you would be training the other layers at
initialization now this actually like kind of fixes itself a little bit if you train for a bit longer so for example if
I greater than 1,000 only then do a break let me reinitialize and then let
me do it 1,000 steps and after 1,000 steps we can look at the forward pass
okay so you see how the neurons are a bit are saturating a bit and we can also look at the backward pass but otherwise
they look good they're about equal and there's no shrinking to zero or exploding to Infinities and you can see
that here in the weights uh things are also stabilizing a little bit so the Tails of the last pink layer are
actually coming coming in during the optimization but certainly this is like a little bit of troubling especially if
you are using a very simple update rule like stochastic gradient descent instead of a modern Optimizer like Adam now I'd

# viz #4: update:data ratio over time

like to show you one more plot that I usually look at when I train neural networks and basically the gradient to
data ratio is not actually that informative because what matters at the end is not the gradient to data ratio
but the update to the data ratio because that is the amount by which we will actually change the data in these
tensors so coming up here what I'd like to do is I'd like to introduce a new update to data uh ratio it's going to be
list and we're going to build it out every single iteration and here I'd like to keep track of basically the
ratio every single iteration so without any gradients I'm
comparing the update which is learning rate times the times the gradient that is the update that we're
going to apply to every parameter uh so see I'm iterating over all the parameters and then I'm taking
the basically standard deviation of the update we're going to apply and divided by the um actual content the data of of
that parameter and its standard deviation so this is the ratio of basically how great are the updates to
the values in these tensors then we're going to take a log of it and actually I'd like to take a log
10 um just so it's a nicer visualization um so we're going to be
basically looking at the exponents of uh the of this division here and then that
item to pop out the float and we're going to be keeping track of this for all the parameters and adding it to
these UD answer so now let me reinitialize and run a th iterations we
can look at the activations the gradients and the parameter gradients as
we did before but now I have one more plot here to introduce and what's Happening Here is
we're are interval parameters and I'm constraining it again like I did here to just the
weights so the number of dimensions in these sensors is two and then I'm basically plotting all of these um
update ratios over time so when I plot this I plot those ratios
and you can see that they evolve over time during initialization they take on certain values and then these updates s
of like start stabilizing usually during training then the other thing that I'm plotting here is I'm plotting here like
an approximate value that is a Rough Guide for what it roughly should be and it should be like roughly
one3 and so that means that basically there's some values in the tensor um and
they take on certain values and the updates to them at every iteration are no more than roughly 1,000th of the
actual like magnitude in those tensors uh if this was much larger like for example if this was um if the log of
this was like say negative 1 this is actually updating those values quite a lot they're undergoing a lot of change
but the reason that the final rate the final uh layer here is an outlier is because this layer was artificially
shrunk down to keep the soft Max um incom unconfident
so here you see how we multiplied The Weight by 0.1 uh in the initialization to make the
last layer prediction less confident that made that artificially made the
values inside that tensor way too low and that's why we're getting temporarily a very high ratio but you see that that
stabilizes over time once uh that weight starts to learn starts to learn but
basically I like to look at the evolution of this update ratio for all my parameters usually and I like to make
sure that it's not too much above onean neg3 roughly uh so around3 on this log
plot if it's below -3 usually that means that the parameters are not trained fast enough so if our learning rate was very
low let's do that experiment uh let's initialize and then let's actually do a learning rate of say
one3 here so 0.001 if your learning rate is way too
low this plot will typically reveal it so
you see how all of these updates are way too small so the size of the update is
uh basically uh 10,000 times um in magnitude to the size of the numbers in
that tensor in the first place so this is a symptom of training way too slow so this is another way to sometimes
set the learning rate and to get a sense of what that learning rate should be and ultimately this is something that you would uh keep track of
if anything the learning rate here is a little bit on the higher side uh because
you see that um we're above the black line of3 we're somewhere around -2.5
it's like okay and uh but everything is like somewhat stabilizing and so this looks like a pretty decent setting of of
um learning rates and so on but this is something to look at and when things are miscalibrated you will you will see very
quickly so for example everything looks pretty well behaved right but just as a comparison
when things are not properly calibrated what does that look like let me come up here and let's say that for example uh
what do we do let's say that we forgot to apply this a fan in normalization so
the weights inside the linear layers are just sampled from aaan and all the stages what happens to our how do we
notice that something's off well the activation plot will tell you whoa your neurons are way too saturated uh the
gradients are going to be all messed up uh the histogram for these weights are going to be all messed up as well and
there's a lot of asymmetry and then if we look here I suspect it's all going to be also pretty messed up so uh you see
there's a lot of uh discrepancy in how fast these layers are learning and some of them are learning way too fast so uh1
1.5 those are very large numbers in terms of this ratio again you should be somewhere around3 and not much more
about that um so this is how miscalibrations of your neuron nuts are going to manifest and these kinds of
plots here are a good way of um sort of bringing um those miscalibrations sort
of uh to your attention and so you can address them okay so so far we've seen

# bringing back batchnorm, looking at the visualizations

that when we have this linear tanh sandwich we can actually precisely calibrate the gains and make the
activations the gradients and the parameters and the updates all look pretty decent but it definitely feels a
little bit like balancing of a pencil on your finger and that's because this gain
has to be very precisely calibrated so now let's introduce bat normalization layers into the fix into the mix and
let's let's see how that helps fix the problem so here I'm going to take the bachom 1D
class and I'm going to start placing it inside and as I mentioned before the
standard typical place you would place it is between the linear layer so right after it but before the nonlinearity but
people have definitely played with that and uh in fact you can get very similar results even if you place it after the
nonlinearity um and the other thing that I wanted to mention is it's totally fine to also place it at the end uh after the
last linear layer and before the L function so this is potentially fine as well um and in this case this would be
output would be WAP size um now because the last layer is
Bash we would not be changing the weight to make the softmax less confident we'd be changing the gamma because gamma
remember in the bathroom is the variable that multiplicatively interacts with the output of that
normalization so we can initialize this sandwich now we can train and we can see
that the activations uh are going to of course look uh very good and they are going to necessarily look good because
now before every single 10h layer there is a normalization in the bashor so this
is unsurprisingly all uh looks pretty good it's going to be standard deviation of roughly 65 2% and roughly equal
standard deviation throughout the entire layers so everything looks very homogeneous the gradients look good the
weights look good and their distributions and then the
updates also look um pretty reasonable uh we are going above3 a little bit but
not by too much so all the parameters are training at roughly the same rate um
here but now what we've gained is um we are going to be slightly less
um brittle with respect to the gain of these so for example I can make the gain
be say2 here um which is much much much slower than what we had with the tan
H but as we'll see the activations will actually be exactly unaffected uh and that's because of again this explicit
normalization the gradients are going to look okay the weight gradients are going to look okay okay but actually the
updates will change and so even though the forward and backward pass to a very large extent
look okay because of the backward pass of the Bator and how the scale of the incoming activations interacts in the
Bator and its uh backward pass this is actually changing the um the scale of
the updates on these parameters so the grades on gradients of these weights are affected so we still don't get it
completely free pass to pass in arbitral um weights here but it everything else
is significantly more robust in terms of the forward backward and the weight
gradients it's just that you may have to retune your learning rate if you are changing sufficiently the the scale of
the activations that are coming into the batch Norms so here for example this um we changed the gains of these linear
layers to be greater and we're seeing that the updates are coming out lower as a
result and then finally we can also so if we are using borms we don't actually need to necessarily let me reset this to
one so there's no gain we don't necessarily even have to um normalize by fan in sometimes so if I take out the
fan in so these are just now uh random gsh in we'll see that because of borm
this will actually be relatively well behaved so the statistic look of course in the
forward pass look good the gradients look good the uh backward uh the weight
updates look okay A little bit of fat tails on some of the layers and uh this looks okay as well
but as you as you can see uh we're significantly below ne3 so we'd have to bump up the learning rate of this bachor
uh so that we are training more properly and in particular looking at this roughly looks like we have to 10x the
learning rate to get to about one3 so we' come here and we would
change this to be update of 1.0 and if I reinitialize
then we'll see that everything still of course looks good and now we are roughly here and we expect this to be an okay
training run so long story short we are significantly more robust to the gain of these linear layers whether or not we
have to apply the fan in and then we can change the gain uh but we actually do
have to worry a little bit about the update um scales and making sure that uh the learning rate is properly calibrated
here but this the activations of the forward backward pass and the updates are are looking significantly more well
behaved except for the global scale that is potentially being adjusted here okay

# summary of the lecture for real this time

so now let me summarize there are three things I was hoping to achieve with this section number one I wanted to introduce
you to bat normalization which is one of the first modern innovations that we're looking into that helped stabilize very
deep neural networks and their training and I hope you understand how the B normalization works and um how it would
be used in a neural network number two I was hoping to py torify some of our code
and wrap it up into these uh modules so like linear bash 1D 10h Etc these are
layers or modules and they can be stacked up into neural nuts like Lego building blocks and these layers
actually exist in pytorch and if you import torch NN then you can actually
the way I've constructed it you can simply just use pytorch by prepending n and Dot to all these different
layers and actually everything will just work because the API that I've developed
here is identical to the API that pytorch uses and the implementation also is basically as far as I'm Weare
identical to the one in pytorch and number three I tried to introduce you to the diagnostic tools that you would use
to understand whether your neural network is in a good State dynamically so we are looking at the statistics and
histograms and activation of the forward pass activ activations the backward pass gradients and then also we're looking at
the weights that are going to be updated as part of stochastic gradi in ascent and we're looking at their means standard deviations and also the ratio
of gradients to data or even better the updates to data and we saw that
typically we don't actually look at it as a single snapshot Frozen in time at some particular iteration typically
people look at this as a over time just like I've done here and they look at these update to data ratios and they
make sure everything looks okay and in particular I said said that um W3 or basically ne3 on the lock scale is
a good uh rough euristic for what you want this ratio to be and if it's way too high then probably the learning rate
or the updates are a little too too big and if it's way too small that the learning rate is probably too small so
that's just some of the things that you may want to play with when you try to get your neural network to uh work with
very well now there's a number of things I did not try to achieve I did not try to
beat our previous performance as an example by introducing using the bash layer actually I did try um and I found
the new I used the learning rate finding mechanism that I've described before I tried to train a borm layer a borm
neural nut and uh I actually ended up with results that are very very similar to what we've obtained before and that's
because our performance now is not bottlenecked by the optimization which is what borm is helping with the
performance at this stage is bottleneck by what I suspect is the context length of our context so currently we are
taking three characters to predict the fourth one and I think we need to go beyond that and we need to look at more powerful architectures like recurrent
neural networks and Transformers in order to further push um the lock probabilities that we're achieving on
this data set and I also did not try to have a full explanation of all of these
activations the gradients and the backward pass and the statistics of all these gradients and so you may have found some of the parts here un
intuitive and maybe you're slightly confused about okay if I change the uh gain here how come that we need a
different learning rate and I didn't go into the full detail because you'd have to actually look at the backward pass of all these different layers and get an
intuitive understanding of how that works and I did not go into that in this lecture the purpose really was just to
introduce you to the diagnostic tools and what they look like but there's still a lot of work remaining on the intuitive level to understand the
initialization the backward pass and how all of that interacts uh but you shouldn't feel too bad because honestly
we are getting to The Cutting Edge of where the field is we certainly haven't I would say soled
initialization and we haven't soled back propagation and these are still very much an active area of research people
are still trying to figure out what is the best way to initialize these networks what is the best update rule to use um and so on so none of this is
really solved and we don't really have all the answers to all the to you know all these cases but at least uh you know
we're making progress and at least we have some tools to tell us uh whether or not things are on the right track for
now so I think we've made positive progress in this lecture and I hope you enjoyed that
and I will see you next time