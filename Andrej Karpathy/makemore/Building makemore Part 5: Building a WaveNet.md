We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between jupyter notebooks and repository code, ...).

Links:
- makemore on github: https://github.com/karpathy/makemore
- jupyter notebook I built in this video: https://github.com/karpathy/nn-zero-t...
- collab notebook: https://colab.research.google.com/dri...
- my website: https://karpathy.ai
- my twitter:   / karpathy  
- our Discord channel:   / discord  

Supplementary links:
- WaveNet 2016 from DeepMind https://arxiv.org/abs/1609.03499
- Bengio et al. 2003 MLP LM https://www.jmlr.org/papers/volume3/b... 

```
Chapters:
intro
00:00:00 intro
00:01:40 starter code walkthrough
00:06:56 let’s fix the learning rate plot
00:09:16 pytorchifying our code: layers, containers, torch.nn, fun bugs
implementing wavenet
00:17:11 overview: WaveNet
00:19:33 dataset bump the context size to 8
00:19:55 re-running baseline code on block_size 8
00:21:36 implementing WaveNet
00:37:41 training the WaveNet: first pass
00:38:50 fixing batchnorm1d bug
00:45:21 re-training WaveNet with bug fix
00:46:07 scaling up our WaveNet
conclusions
00:46:58 experimental harness
00:47:44 WaveNet but with “dilated causal convolutions”
00:51:34 torch.nn
00:52:28 the development process of building deep neural nets
00:54:17 going forward
00:55:26 improve on my loss! how far can we improve a WaveNet on this data?
```

当然，以下是这段话的中文翻译：

---

我们将之前视频中的 2 层 MLP（多层感知机）做了“加深”，构建成树状结构，最终形成一种类似于 DeepMind 2016 年 WaveNet 论文中的卷积神经网络架构。在 WaveNet 论文中，同样的层次化结构通过\*\*因果扩张卷积（causal dilated convolutions）\*\*来更高效地实现（本视频暂未涉及该部分）。在这个过程中，你可以更好地理解 `torch.nn` 是什么，它背后的工作原理，以及一个典型的深度学习开发过程通常是什么样子（大量阅读文档、关注多维张量的 shape 变化、频繁切换 Jupyter notebook 和代码仓库等）。

**链接：**

* makemore 项目 GitHub: [https://github.com/karpathy/makemore](https://github.com/karpathy/makemore)
* 本视频对应的 Jupyter notebook: [https://github.com/karpathy/nn-zero-t](https://github.com/karpathy/nn-zero-t)...
* Colab notebook: [https://colab.research.google.com/dri](https://colab.research.google.com/dri)...
* 我的个人网站: [https://karpathy.ai](https://karpathy.ai)
* 我的推特:   / karpathy
* 我们的 Discord 频道:   / discord

**补充阅读：**

* DeepMind 2016 年 WaveNet 论文: [https://arxiv.org/abs/1609.03499](https://arxiv.org/abs/1609.03499)
* Bengio 等人 2003 年的 MLP 语言模型论文: [https://www.jmlr.org/papers/volume3/b](https://www.jmlr.org/papers/volume3/b)...

```
**视频章节：**
**intro**
00:00:00 介绍
00:01:40 起始代码讲解
00:06:56 修正学习率曲线的绘图
00:09:16 使用 PyTorch 改造代码：layers, containers, torch.nn，遇到的有趣小 bug

**实现 WaveNet**
00:17:11 WaveNet 总览
00:19:33 数据集：将 context size 增大到 8
00:19:55 使用 block\_size=8 重新运行 baseline 代码
00:21:36 实现 WaveNet 网络结构
00:37:41 第一次训练 WaveNet
00:38:50 修复 batchnorm1d 的 bug
00:45:21 修复后重新训练 WaveNet
00:46:07 扩展我们的 WaveNet 规模

**总结**
00:46:58 实验流程工具
00:47:44 使用“扩张因果卷积”的 WaveNet
00:51:34 进一步理解 torch.nn
00:52:28 构建深度神经网络的开发过程
00:54:17 展望未来
00:55:26 改进 loss！看看我们能把 WaveNet 在这个数据集上优化到什么程度？
```

---

如果你需要，我还可以帮你总结成更简单的中文，或者重点解释 "因果扩张卷积"、"WaveNet" 这些概念～要不要继续？ 🌟


# intro

hi everyone today we are continuing our implementation of make more our favorite character level language model
now you'll notice that the background behind me is different that's because I am in Kyoto and it is awesome so I'm in
a hotel room here now over the last few lectures we've built up to this architecture that is a
multi-layer perceptron character level language model so we see that it receives three previous characters and
tries to predict the fourth character in a sequence using a very simple multi perceptron using one hidden layer of
neurons with 10ational neuralities so we'd like to do now in this lecture is I'd like to complexify this
architecture in particular we would like to take more characters in a sequence as an input not just three and in addition
to that we don't just want to feed them all into a single hidden layer because that squashes too much information too
quickly instead we would like to make a deeper model that progressively fuses this information to make its guess about
the next character in a sequence and so we'll see that as we make this architecture more complex we're actually
going to arrive at something that looks very much like a wavenet the witness is this paper published by
the point in 2016 and it is also a language model basically but it tries to
predict audio sequences instead of character level sequences or Word level sequences but fundamentally the modeling
setup is identical it is an auto aggressive model and it tries to predict next character in a sequence and the
architecture actually takes this interesting hierarchical sort of approach to predicting the next
character in a sequence uh with the street-like structure and this is the architecture and we're going to
implement it in the course of this video so let's get started so the starter code for part five is very similar to where

当然，这一段我给你翻译一下：

---

大家好，今天我们继续实现我们的 `makemore`——我们最喜欢的**字符级语言模型**。
你可能注意到我身后的背景换了，因为我现在在京都，非常棒，我现在是在酒店房间里录的。

在前几节课里，我们已经搭建了这样一个架构：
一个**多层感知机（MLP）**的字符级语言模型。它的输入是**前面三个字符**，然后尝试预测**第四个字符**，模型的核心是一层隐藏层（hidden layer），里面有 10 个神经元（neurons）。

今天这节课我希望能**复杂化**这个架构，具体包括两点：

1. **输入更多的字符**，不仅仅是三个字符作为输入；
2. 不希望把这些字符一次性全送进一个隐藏层——因为那样会把信息压缩得太快、太狠。我们希望构建一个**更深的模型**，逐步融合信息，逐步做出对下一个字符的预测。

在这个过程中，我们会发现，随着架构的复杂化，最终我们做出来的网络会非常像 2016 年 DeepMind 发表的**WaveNet 论文**中的结构。
WaveNet 本质上也是一个语言模型，不过它是用来预测**音频序列**，而不是字符序列或者单词序列。
但其实这两者的建模方式是一样的，都是**自回归模型（autoregressive model）**，都是在预测序列中的下一个“元素”（字符或者音频采样点）。

WaveNet 的架构有一个很有意思的**分层（hierarchical）结构**，是树状（tree-like）的。
今天我们就要在这个视频里，手把手去实现这样一个类似 WaveNet 的模型架构。
好，那我们就开始吧～ 这节课用的 starter code，和我们之前课程结束时的代码非常相似。

---

如果需要的话，我还可以帮你总结出**这个模型的思路和演变过程**，这样你学习的时候会更清楚～ 要不要继续？🌟


# starter code walkthrough

we ended up in in part three recall that part four was the manual black replication exercise that is kind of an
aside so we are coming back to part three copy pasting chunks out of it and that is our starter code for part five
I've changed very few things otherwise so a lot of this should look familiar to if you've gone through part three so in
particular very briefly we are doing Imports we are reading our our data set of words and we are processing their set
of words into individual examples and none of this data generation code has changed and basically we have lots and
lots of examples in particular we have 182 000 examples of three characters try
to predict the fourth one and we've broken up every one of these words into little problems of given three
characters predict the fourth one so this is our data set and this is what we're trying to get the neural lot to do
now in part three we started to develop our code around these layer modules
um that are for example like class linear and we're doing this because we want to think of these modules as
building blocks and like a Lego building block bricks that we can sort of like stack up into neural networks and we can
feed data between these layers and stack them up into a sort of graphs now we also developed these layers to
have apis and signatures very similar to those that are found in pytorch so we
have torch.nn and it's got all these layer building blocks that you would use in practice and we were developing all
of these to mimic the apis of these so for example we have linear so there will also be a torch.nn.linear and its
signature will be very similar to our signature and the functionality will be also quite identical as far as I'm aware
so we have the linear layer with the Bass from 1D layer and the 10h layer that we developed previously
and linear just as a matrix multiply in the forward pass of this module batch
number of course is this crazy layer that we developed in the previous lecture and what's crazy about it is
well there's many things number one it has these running mean and variances that are trained outside of back
propagation they are trained using exponential moving average inside this
layer when we call the forward pass in addition to that there's this training plug because the
behavior of bathroom is different during train time and evaluation time and so suddenly we have to be very careful that bash form is in its correct state that
it's in the evaluation state or training state so that's something to now keep track of something that sometimes introduces bugs
uh because you forget to put it into the right mode and finally we saw that Bachelor couples the statistics or the
the activations across the examples in the batch so normally we thought of the bat as just an efficiency thing but now
we are coupling the computation across batch elements and it's done for the
purposes of controlling the automation statistics as we saw in the previous video so it's a very weird layer at least a
lot of bugs partly for example because you have to modulate the training in eval phase and
so on um in addition for example you have to wait for uh the mean and the variance to
settle and to actually reach a steady state and so um you have to make sure that you basically there's state in this
layer and state is harmful uh usually now I brought out the generator object
previously we had a generator equals g and so on inside these layers I've discarded that in favor of just
initializing the torch RNG outside here use it just once globally just for
Simplicity and then here we are starting to build out some of the neural network elements this should look very familiar we are we
have our embedding table C and then we have a list of players and uh it's a linear feeds to Bachelor feeds to 10h
and then a linear output layer and its weights are scaled down so we are not confidently wrong at the initialization
we see that this is about 12 000 parameters we're telling pytorch that the parameters require gradients
the optimization is as far as I'm aware identical and should look very very familiar
nothing changed here uh loss function looks very crazy we
should probably fix this and that's because 32 batch elements are too few and so you can get very lucky lucky or
unlucky in any one of these batches and it creates a very thick loss function um so we're going to fix that soon
now once we want to evaluate the trained neural network we need to remember because of the bathroom layers to set
all the layers to be training equals false so this only matters for the bathroom layer so far
and then we evaluate we see that currently we have validation
loss of 2.10 which is fairly good but there's still ways to go but even at
2.10 we see that when we sample from the model we actually get relatively name-like results that do not exist in a
training set so for example Yvonne kilo Pros
Alaia Etc so certainly not reasonable not unreasonable I would say
but not amazing and we can still push this validation loss even lower and get much better samples that are even more
name-like so let's improve this model okay first let's fix this graph because

当然，这一段我给你详细翻译一下：

---

**# starter code walkthrough（起始代码讲解）**

我们现在回到了**第三部分**结束时的代码（注意：**第四部分**是一次“手动反向传播”练习，算是个番外篇），所以这节课是从第三部分的代码中**复制部分代码**作为新的起点（starter code）开始的，整体变化不大。
如果你之前看过第三部分的内容，这些代码你应该会很熟悉。

大致流程是这样的：

* 先做 **import**，然后读取数据集（words），并把它们拆解成单个的训练样本（examples），这部分的数据处理逻辑没有变化。
* 数据量也挺大，总共生成了 **182,000** 个小样本，每个小样本是“给定 3 个字符，预测第 4 个字符”。
* 这个数据集就是我们希望用神经网络去学习的目标：**预测下一个字符**。

在 **第三部分**，我们引入了 “layer module” 这种设计方式：

* 例如写了一个 `class Linear`，因为我们想把这些模块（modules）像\*\*乐高积木（lego blocks）\*\*一样拼装起来，搭建成完整的神经网络，数据可以在不同层之间传递，形成一个计算图（graph）。
* 同时，我们写这些模块时，尽量让它们的 API 和 **PyTorch 的 torch.nn** 接口保持一致，这样以后可以无缝切换、学习 PyTorch 也更顺手。
* 举例来说，我们写的 `Linear` 层，其实功能和 `torch.nn.Linear` 差不多，接口签名（signature）也尽量对齐。

回顾之前开发的几种 layer：

* **Linear** 层：就是一层矩阵乘法（forward 过程就是 matmul）
* **BatchNorm1d** 层：这是之前开发过程中一个“神奇”的层，因为：

  1. 它内部维护了**均值（mean）和方差（variance）**，并且这些是通过\*\*指数滑动平均（EMA）\*\*来更新的，不是直接用反向传播学到的；
  2. 训练阶段（training）和评估阶段（evaluation）的行为不同，所以需要手动切换 `training=True/False`，否则容易出 bug；
  3. 这个层的计算是**跨 batch 的**，也就是不同 batch 元素之间会互相影响（因为需要计算整体的均值方差），不像普通的 Linear 层是独立计算的。

BatchNorm1d 是一个有“状态（state）”的层，这种 state 很容易引入 bug，比如需要等待均值方差收敛（settle），而且训练和推理阶段切换容易忘记。

以前我们把 `torch.Generator` 放到每层里，现在简化了，改成只在外面统一初始化 RNG（随机数生成器），整体代码更干净。

接下来我们搭建神经网络的“前半部分”：

* 有一个 **embedding table C**，
* 然后一层 `Linear` -> `BatchNorm1d` -> `Tanh` -> 输出层 `Linear`，
* 初始化时特意把输出层的权重缩小，防止一开始就预测得“信心过高但错误”（confidently wrong）。
* 目前模型大概有 **12,000 个参数**，我们告诉 PyTorch 这些参数需要 `requires_grad=True`，这样可以参与反向传播。

优化器部分（optimizer）和以前一样，没变。

**损失曲线（loss function）** 现在看起来很“跳”很“乱”，因为 batch size 太小（只有 32 个样本），容易出现 batch 的结果太好/太坏，导致 loss 曲线很厚重，不平滑，这个马上会调整。

训练完模型后，要做评估（evaluate）时要注意：

* 因为有 BatchNorm 层，评估前一定要把所有层设成 `training=False`，否则结果会不稳定。

当前模型的 **验证集 loss 大概是 2.10**，还可以继续优化。
尽管 loss 还不是很低，但已经能生成一些“像名字的”字符序列（虽然训练集中不存在），比如：
`Yvonne kilo Pros`
`Alaia` 等等。

目前来说，效果还不错，但也不是特别惊艳，我们希望继续把 loss 拉低，提升生成效果。
下一步，我们先来调整一下 loss 曲线的绘图逻辑——现在它太乱了。

---

如果你需要，我还可以帮你整理一下这个“starter code 讲解”的**核心重点**，方便你复习～要不要？ 🚀


# let’s fix the learning rate plot

it is daggers in my eyes and I just can't take it anymore um so last I if you recall is a python
list of floats so for example the first 10 elements
now what we'd like to do basically is we need to average up um some of these values to get a more
sort of Representative uh value along the way so one way to do this is the following
in part torch if I create for example a tensor of the first 10 numbers
then this is currently a one-dimensional array but recall that I can view this array as two-dimensional so for example
I can use it as a two by five array and this is a 2d tensor now two by five and
you see what petroch has done is that the first row of this tensor is the first five elements and the second row
is the second five elements I can also view it as a five by two as an example
and then recall that I can also use negative one in place of one of
these numbers and pytorch will calculate what that number must be in order to make the number of elements work out so this can
be this or like that but it will work of course this would not work
okay so this allows it to spread out some of the consecutive values into rows so that's very helpful because what we
can do now is first of all we're going to create a torshot tensor out of the a
list of floats and then we're going to view it as whatever it is but we're going to
stretch it out into rows of 1000 consecutive elements so the shape of this now becomes 200 by 1000. and each
row is one thousand um consecutive elements in this list so that's very helpful because now we
can do a mean along the rows and the shape of this will just be 200.
and so we've taken basically the mean on every row so plt.plot of that should be something nicer
much better so we see that we basically made a lot of progress and then here this is the
learning rate Decay so here we see that the learning rate Decay subtracted a ton of energy out of the system and allowed
us to settle into sort of the local minimum in this optimization so this is a much nicer plot let me come
up and delete the monster and we're going to be using this going forward now next up what I'm bothered by is that you

当然，这段我来帮你翻译解释一下：

---

**# 修正学习率曲线**

“这条曲线看得我眼睛都疼了，实在受不了了”，所以我们来修一下。

之前记录 loss 的变量 `lossi` 是一个 **Python 的 list**，里面存的是每次训练 step 的浮点数（float）。
比如前 10 个元素看起来像这样 `[2.3, 2.1, 2.05, ...]`。

现在我们希望对这些 loss 值**进行平滑处理**，也就是对它们做一下**平均**，让曲线更有代表性，不要太乱。

可以怎么做呢？
举例来说，在 PyTorch 里我可以把这个 list 转换成一个 tensor，变成一维的数组（1D array）。
然后，tensor 是可以\*\*reshape（重塑形状）\*\*的，比如我可以把它 reshape 成 2 行 5 列的二维 tensor（2x5），
PyTorch 会自动把前 5 个元素放到第一行，接下来的 5 个元素放到第二行。

还可以 reshape 成 5x2，当然，只要元素个数对得上就可以。
而且，PyTorch 的 reshape 里，shape 参数可以写 `-1`，这样 PyTorch 会自动帮你计算这一维的大小。
比如 `.view(-1, 1000)`，就会自动根据总元素个数算出行数。

这个特性很有用：
我们现在要把 `lossi` 这个 list 变成一个 PyTorch tensor，然后 reshape 成 “每行 1000 个元素”，
这样每行就代表**连续的 1000 个训练 step**。
比如 reshape 成 `200 x 1000`，表示总共有 200 组，每组 1000 个 loss 数据。

接下来就可以对每一行求平均（mean），也就是**每 1000 个 step 求一个平均 loss**，
这样画出来的曲线就不会那么乱了，变得平滑很多。

* 用 `plt.plot` 画出这个平滑后的曲线，效果就好多了！
* 曲线左侧部分是 loss 下降的过程，右侧你可以看到 learning rate 开始 decay（衰减），系统里的能量下降，优化器收敛到一个 local minimum。
* 这样画出来的 loss 曲线更清楚，整体趋势一目了然。

最后作者说：“我要把原来那条乱七八糟的曲线删掉，我们今后就用这种更好看的版本”。

---

如果你想要，我还可以帮你写一段**对应的 PyTorch 代码示例**，这样你可以直接参考或者用在你自己的 notebook 里，要不要？ 🌟


# pytorchifying our code: layers, containers, torch.nn, fun bugs

see our forward pass is a little bit gnarly and takes way too many lines of code
so in particular we see that we've organized some of the layers inside the layers list but not all of them uh for
no reason so in particular we see that we still have the embedding table a special case outside of the layers and
in addition to that the viewing operation here is also outside of our layers so let's create layers for these
and then we can add those layers to just our list so in particular the two things that we
need is here we have this embedding table and we are indexing at the integers inside uh the batch XB uh
inside the tensor xB so that's an embedding table lookup just done with indexing and then here we see
that we have this view operation which if you recall from the previous video Simply rearranges the character
embeddings and stretches them out into a row and effectively what print that does
is the concatenation operation basically except it's free because viewing is very cheap in pytorch no no memory is being
copied we're just re-representing how we view that tensor so let's create um
modules for both of these operations the embedding operation and flattening operation
so I actually wrote the code in just to save some time so we have a module embedding and a
module pattern and both of them simply do the indexing operation in the forward pass and the flattening operation here
and this C now will just become a salt dot weight inside an embedding module
and I'm calling these layers specifically embedding a platinum because it turns out that both of them actually exist in pi torch so in
phytorch we have n and Dot embedding and it also takes the number of embeddings and the dimensionality of the bedding
just like we have here but in addition python takes in a lot of other keyword arguments that we are not using for our
purposes yet and for flatten that also exists in pytorch and it also takes additional
keyword arguments that we are not using so we have a very simple platform but both of them exist in pytorch
they're just a bit more simpler and now that we have these we can simply take
out some of these special cased um things so instead of C we're just
going to have an embedding and of a cup size and N embed
and then after the embedding we are going to flatten so let's construct those modules and now
I can take out this the and here I don't have to special case anymore because now C is the embeddings
weight and it's inside layers so this should just work
and then here our forward pass simplifies substantially because we don't need to do these now outside of
these layer outside and explicitly they're now inside layers
so we can delete those but now to to kick things off we want this little X which in the beginning is
just XB uh the tensor of integers specifying the identities of these characters at the input
and so these characters can now directly feed into the first layer and this should just work so let me come here and insert a break
because I just want to make sure that the first iteration of this runs and then there's no mistake so that ran
properly and basically we substantially simplified the forward pass here okay I'm sorry I changed my microphone so
hopefully the audio is a little bit better now one more thing that I would like to do in order to pytortify our code even
further is that right now we are maintaining all of our modules in a naked list of layers and we can also
simplify this uh because we can introduce the concept of Pi torch containers so in tors.nn which we are
basically rebuilding from scratch here there's a concept of containers and these containers are basically a way of organizing layers into
lists or dicts and so on so in particular there's a sequential which
maintains a list of layers and is a module class in pytorch and it basically
just passes a given input through all the layers sequentially exactly as we are doing here
so let's write our own sequential I've written a code here and basically
the code for sequential is quite straightforward we pass in a list of layers which we keep here and then given
any input in a forward pass we just call all the layers sequentially and return the result in terms of the parameters
it's just all the parameters of the child modules so we can run this and we can again
simplify this substantially because we don't maintain this naked list of layers we now have a notion of a model which is
a module and in particular is a sequential of all these layers
and now parameters are simply just a model about parameters and so that list comprehension now lives
here and then here we are press here we are doing all the things we used to do
now here the code again simplifies substantially because we don't have to do this forwarding here instead of just
call the model on the input data and the input data here are the integers inside xB so we can simply do logits which are
the outputs of our model are simply the model called on xB
and then the cross entropy here takes the logits and the targets so this simplifies substantially
and then this looks good so let's just make sure this runs that looks good
now here we actually have some work to do still here but I'm going to come back later for now there's no more layers
there's a model that layers but it's not a to access attributes of these classes
directly so we'll come back and fix this later and then here of course this simplifies substantially as well because logits are
the model called on x and then these low Jets come here
so we can evaluate the train and validation loss which currently is terrible because we just initialized the
neural net and then we can also sample from the model and this simplifies dramatically as well because we just want to call the model
onto the context and outcome logits and these logits go into softmax and get
the probabilities Etc so we can sample from this model what did I screw up
okay so I fixed the issue and we now get the result that we expect which is gibberish because the model is not
trained because we re-initialize it from scratch the problem was that when I fixed this cell to be modeled out layers instead of
just layers I did not actually run the cell and so our neural net was in a training mode and what caused the issue
here is the bathroom layer as bathroom layer of the likes to do because Bachelor was in a training mode and here
we are passing in an input which is a batch of just a single example made up of the context
and so if you are trying to pass in a single example into a bash Norm that is in the training mode you're going to end
up estimating the variance using the input and the variance of a single number is is not a number because it is
a measure of a spread so for example the variance of just the single number five you can see is not a number and so
that's what happened in the master basically caused an issue and then that polluted all of the further processing
so all that we have to do was make sure that this runs and we basically made the
issue of again we didn't actually see the issue with the loss we could have evaluated
the loss but we got the wrong result because basharm was in the training mode and uh and so we still get a result it's
just the wrong result because it's using the uh sample statistics of the batch whereas we want to use the running mean
and running variants inside the bachelor and so again an example of introducing a bug
inline because we did not properly maintain the state of what is training or not okay so I Rewritten everything

当然，这段内容我来帮你详细翻译和解释：

---

**# Pytorch化代码：层、容器、torch.nn、遇到的小 bug**

现在我们神经网络的 forward 过程（前向传播）写得太啰嗦了，代码行数太多、结构不清晰。
目前有一部分 layer 被放在 `layers` 列表里管理，但还有一些没有放进去，比如：

* **embedding table C** 是单独写在外面的，
* **view（reshape）操作** 也是写在外面的，
  这其实是没有必要的。

所以现在的改进目标就是：
把这些“特例”操作也封装成 layer，然后统一管理，便于后面维护和扩展。

具体来说，当前需要处理的两个操作：

1. **embedding lookup**，也就是从 embedding table 查表，这个是通过 `C[xB]` 做的索引；
2. **view 操作**，也就是把 embedding 结果展平（flatten）成一行，相当于是“拼接”操作，不过 view 是“零成本”的（不复制内存，只是换一个 tensor 视图）。

于是作者提前写好两个模块：

* `Embedding` 模块：封装了 embedding lookup 的操作；
* `Flatten` 模块：封装了 flatten 操作。

PyTorch 里其实本来也有对应的模块：

* `torch.nn.Embedding`
* `torch.nn.Flatten`
  只不过 PyTorch 版本功能更丰富，参数更多，我们目前先实现一个简版的够用即可。

有了这两个模块之后，之前代码里单独处理 `C` 和 `view` 的地方就可以删掉，直接放入 layers 统一管理：

* `Embedding` -> `Flatten` -> 其他层，形成一个更干净的 forward 流程。

这样 forward 过程就变得更简单了，数据（xB）直接输入第一层 layer，后面层层传递。

---

接下来进一步“PyTorch 化”：
目前我们的 `layers` 还是裸 list，这样不好。
PyTorch 提供了“容器（containers）”，可以更好地管理 layer，比如：

* `torch.nn.Sequential` 就是一个常用的容器，内部是一个 layer 列表，forward 过程会**自动按顺序调用所有 layer**。

所以我们也写了一个简版的 `Sequential` 容器：

* 初始化时传入一个 layer list；
* forward 时依次调用这些 layer，最后返回结果；
* 同时可以很方便地收集所有的 parameters（模型参数）。

然后模型就升级为：

```
model = Sequential([...layers...])  
```

* 这样就不用单独管理 `layers`，也不用手动 forward 了，直接 `model(xB)` 就可以跑 forward，代码大大简化。

---

然后作者改写了 loss 计算和 sample 部分：

* logits = model(xB)
* loss = cross\_entropy(logits, yB)

评估（evaluate）和 sample 的流程也简化了，直接调用 model 即可，不用处理 layers 细节。

---

这里遇到了一个小 bug：
改完以后，跑 sample 结果出现问题，原因是：

* 改完 `model = Sequential(layers)` 后，没 rerun 这个 cell，导致 BatchNorm layer 还处于 training 模式；
* 结果在用单个 example 做 sample 时，BatchNorm 试图用一个 batch 的 statistics（均值方差），但是 batch size=1，方差无法计算（variance of single number = NaN）；
* 这就污染了后续计算，结果错误。

正确做法是 sample 时要把 model 设成 `eval()` 模式（training=False），BatchNorm 才会用 running mean/variance，结果才会正常。

---

总结：

1. 把 embedding / flatten 封装成 layer，forward 更干净；
2. 引入 Sequential 容器，自动管理 layer 列表和 forward 流程；
3. 遇到了 BatchNorm 训练/推理模式切换的小坑，要小心 state 的管理；
4. 整体代码 PyTorch 化程度更高，更接近实际工程写法了。

---

如果你需要的话，我也可以帮你写出这部分的 **关键代码示例**，方便你参考和自己练习～ 要不要？🌟


# overview: WaveNet

and here's where we are as a reminder we have the training loss of 2.05 and validation 2.10
now because these losses are very similar to each other we have a sense that we are not overfitting too much on
this task and we can make additional progress in our performance by scaling up the size of the neural network and
making everything bigger and deeper now currently we are using this architecture here where we are taking in
some number of characters going into a single hidden layer and then going to the prediction of the next character
the problem here is we don't have a naive way of making this bigger in a productive way we could of course use
our layers sort of building blocks and materials to introduce additional layers here and make the network deeper but it
is still the case that we are crushing all of the characters into a single layer all the way at the beginning
and even if we make this a bigger layer and add neurons it's still kind of like silly to squash all that information so
fast in a single step so we'd like to do instead is we'd like our Network to look a lot more like this
in the wavenet case so you see in the wavenet when we are trying to make the prediction for the next character in the
sequence it is a function of the previous characters that are feeding that feed in but not all of these
different characters are not just crushed to a single layer and then you have a sandwich they are crushed slowly
so in particular we take two characters and we fuse them into sort of like a diagram representation and we do that
for all these characters consecutively and then we take the bigrams and we fuse those into four character level chunks
and then we fuse that again and so we do that in this like tree-like hierarchical manner so we fuse the information from
the previous context slowly into the network as it gets deeper and so this is
the kind of architecture that we want to implement now in the wave Nets case this is a visualization of a stack of dilated
causal convolution layers and this makes it sound very scary but actually the idea is very simple and the fact that
it's a dilated causal convolution layer is really just an implementation detail to make everything fast we're going to
see that later but for now let's just keep the basic idea of it which is this Progressive Fusion so we want to make
the network deeper and at each level we want to fuse only two consecutive elements two characters then two bigrams
then two four grams and so on so let's unplant this okay so first up let me scroll to where we built the data set

当然，这段我帮你翻译解释一下：

---

**# 概述：WaveNet**

我们现在的训练 loss 是 **2.05**，验证集 loss 是 **2.10**。
这两个 loss 很接近，说明目前模型**没有严重 overfitting（过拟合）**，可以通过扩大模型规模（加大网络深度、宽度）进一步提升性能。

目前用的模型结构是这样的：

* 输入一串字符（某个 context size）
* 进到一个隐藏层
* 然后输出下一个字符的预测

但是这个架构有一个问题：

* 虽然可以简单通过加层、加神经元来“做大”模型，
* 但是它的本质是：**一开始就把所有输入字符直接“压缩”成一层的表示**，信息融合得太快了！
* 就算加大 hidden layer，信息的融合速度还是太快，这样其实不太合理，网络难以有效建模“长距离依赖”。

我们希望的架构是像 WaveNet 这样的，思路是：

* 预测下一个字符的时候，是所有前面字符的函数，
* 但这些字符不是一下子全“压”到一个层里，
* 而是通过**逐步融合（Progressive Fusion）**，信息逐步流入更深层。

举例来说，WaveNet 是这样做的：

1. 先把相邻两个字符融合成 bigram 的表示；
2. 然后把 bigram 融合成四个字符的表示（4-gram）；
3. 再继续融合... 形成一种**树状的层次结构**（tree-like hierarchical manner），
4. 每一层融合的是更大粒度的信息，直到最终做出预测。

WaveNet 论文图示是一个**堆叠的 dilated causal convolution layers（扩张型因果卷积层）**，
听起来很吓人，但其实核心思想很简单，就是为了**加快实现**，底层用了扩张卷积（dilated conv）这种技巧。
我们暂时不用管实现细节，重点是这个“信息逐层融合”的架构思想。

目标：

* 做一个**更深**的网络，
* 每一层只融合相邻两个元素（字符，bigram，4-gram，...），
* 逐层传递，逐层融合，最终做出下一个字符的预测。

---

作者说：“那我们来开始实现这个架构吧！”
首先要回去看看之前怎么构建的数据集，然后再往下改进模型。

---

如果你需要，我还可以帮你画一个简单的图，**MLP vs. WaveNet 的结构对比**，
这样你理解这个“逐步融合”思想会更直观～ 要不要？ 🚀


# dataset bump the context size to 8

and let's change the block size from 3 to 8. so we're going to be taking eight characters of context to predict the
ninth character so the data set now looks like this we have a lot more context feeding in to predict any next
character in a sequence and these eight characters are going to be processed in this tree like structure
now if we scroll here everything here should just be able to work so we should be able to redefine the network

当然，这段内容我来帮你翻译一下：

---

**# 数据集调整：把 context size 提升到 8**

我们把 `block_size`（上下文长度）从 3 改成 **8**，
也就是说：

* 现在输入是 8 个字符，
* 目标是预测**第 9 个字符**。

这样一来，数据集的格式就变成了：

* 输入更长的上下文（8 个字符）去预测序列中的下一个字符，
* 也就是说，每一个训练样本都会包含更多信息。

然后，这 8 个字符会在模型里通过**树状结构**（tree-like structure）来处理（之前讲过的 Progressive Fusion）。

接下来滚动到下面，可以看到其它代码基本不用大改，直接就能跑：

* 重新定义网络结构之后，整个训练流程应该是可以正常工作的。

---

简单说，就是“扩大了上下文窗口”，这样模型可以学习更复杂的序列依赖关系。

---

要不要我顺便也帮你整理一下：
1️⃣ **为什么提升 context size 有意义**
2️⃣ **会带来什么挑战**
这样你理解起来会更系统～ 🚀


# re-running baseline code on block_size 8

you see the number of parameters has increased by 10 000 and that's because the block size has grown so this first
linear layer is much much bigger our linear layer now takes eight characters into this middle layer so there's a lot
more parameters there but this should just run let me just break right after
the very first iteration so you see that this runs just fine it's just that this network doesn't make too much sense
we're crushing way too much information way too fast so let's now come in and see how we
could try to implement the hierarchical scheme now before we dive into the detail of the re-implementation here I
was just curious to actually run it and see where we are in terms of the Baseline performance of just lazily scaling up the context length so I'll
let it run we get a nice loss curve and then evaluating the loss we actually see quite a bit of improvement just from
increasing the context line length so I started a little bit of a performance log here and previously where we were is
we were getting a performance of 2.10 on the validation loss and now simply scaling up the contact length from 3 to
8 gives us a performance of 2.02 so quite a bit of an improvement here and
also when you sample from the model you see that the names are definitely improving qualitatively as well
so we could of course spend a lot of time here tuning um uh tuning things and making it even bigger and scaling up the network
further even with the simple um sort of setup here but let's continue
and let's Implement here model and treat this as just a rough Baseline performance but there's a lot of
optimization like left on the table in terms of some of the hyper parameters that you're hopefully getting a sense of
now okay so let's scroll up now and come back up and what I've done here

当然，这段内容我来帮你翻译解释一下：

---

**# 用 block\_size = 8 重新跑 baseline 代码**

现在把 `block_size` 提升到了 **8**，你可以看到模型的参数量增加了 **1 万个**，
原因是第一个 `Linear` 层的输入维度变大了（现在是 8 个字符 → 中间层），所以参数数量自然增多了。

不过代码是可以直接跑的。
作者这里加了一个 break point，确认第一轮训练跑得正常。
虽然这个网络“结构上”并不太合理（还是一下子把 8 个字符全压进一个大 hidden layer，信息压缩太快），
但它**能跑通**，可以先看看“单纯把上下文变大”带来的效果。

---

然后作者好奇，先不改 WaveNet 结构，直接“懒得动脑”跑一跑 baseline，看看 **仅仅把 context length 从 3 提升到 8，效果如何**。

* 跑完发现 loss 曲线（loss curve）还不错。
* 验证集 loss 从之前的 **2.10** 降到 **2.02**，光靠增加 context length 就带来了明显提升！

而且从模型 sample 出来的名字，质量也明显提升了，名字“看起来”更像真实名字了。

---

当然，作者说：

* 其实光用这个简单网络、调整一下超参数（hyperparameters）比如 learning rate、网络层数、hidden size，其实还能继续优化；
* 不过我们重点是要**改造模型结构**（实现层次结构的 WaveNet 风格），所以就暂时把这个结果当 baseline 记录下来，接下来继续往下实现更好的模型。

---

总结一下逻辑就是：
1️⃣ 仅仅增加 context size（3 → 8），模型效果已经明显提升；
2️⃣ 但是 MLP 结构本身“压缩太快”，还是有结构问题；
3️⃣ 我们目标是实现 WaveNet 样式的**逐层融合网络**，这才是更合理的架构；
4️⃣ baseline 先记录下来，之后可以对比效果。

---

如果你需要，我也可以帮你整理一份\*\*“WaveNet 架构 vs MLP baseline 架构的效果对比记录表”\*\*，这样以后你学习做实验也会更规范，要不要？🌟


# implementing WaveNet

is I've created a bit of a scratch space for us to just like look at the forward pass of the neural net and inspect the
shape of the tensor along the way as the neural net uh forwards so here I'm just
temporarily for debugging creating a batch of just say four examples so four random integers then I'm plucking out
those rows from our training set and then I'm passing into the model the input xB
now the shape of XB here because we have only four examples is four by eight and this eight is now the current block size
so uh inspecting XP we just see that we have four examples each one of them is a
row of xB and we have eight characters here and this integer tensor just contains the
identities of those characters so the first layer of our neural net is the embedding layer so passing XB this
integer tensor through the embedding layer creates an output that is four by eight by ten
so our embedding table has for each character a 10-dimensional vector that
we are trying to learn and so what the embedding layer does here is it plucks out the embedding
Vector for each one of these integers and organizes it all in a four by eight
by ten tensor now so all of these integers are translated into 10 dimensional vectors inside this
three-dimensional tensor now passing that through the flattened layer as you recall what this does is it views
this tensor as just a 4 by 80 tensor and what that effectively does is that all
these 10 dimensional embeddings for all these eight characters just end up being stretched out into a long row
and that looks kind of like a concatenation operation basically so by viewing the tensor differently we now
have a four by eighty and inside this 80 it's all the 10 dimensional uh vectors just uh concatenate next to each
other and then the linear layer of course takes uh 80 and creates 200 channels
just via matrix multiplication so so far so good now I'd like to show you something surprising
let's look at the insides of the linear layer and remind ourselves how it works
the linear layer here in the forward pass takes the input X multiplies it with a weight and then optionally adds
bias and the weight here is two-dimensional as defined here and the bias is one dimensional here
so effectively in terms of the shapes involved what's happening inside this linear layer looks like this right now
and I'm using random numbers here but I'm just illustrating the shapes and what happens
basically a 4 by 80 input comes into the linear layer that's multiplied by this 80 by 200 weight Matrix inside and
there's a plus 200 bias and the shape of the whole thing that comes out of the linear layer is four by two hundred as
we see here now notice here by the way that this here will create a 4x200 tensor and then
plus 200 there's a broadcasting happening here about 4 by 200 broadcasts with 200 uh so everything works here
so now the surprising thing that I'd like to show you that you may not expect is that this input here that is being
multiplied uh doesn't actually have to be two-dimensional this Matrix multiply
operator in pytorch is quite powerful and in fact you can actually pass in higher dimensional arrays or tensors and
everything works fine so for example this could be four by five by eighty and the result in that case will become four
by five by two hundred you can add as many dimensions as you like on the left here
and so effectively what's happening is that the matrix multiplication only works on the last Dimension and the
dimensions before it in the input tensor are left unchanged
so that is basically these um these dimensions on the left are all treated as just a batch Dimension so we can have
multiple batch dimensions and then in parallel over all those Dimensions we are doing the matrix multiplication on
the last dimension so this is quite convenient because we can use that in our Network now
because remember that we have these eight characters coming in and we don't want to now uh flatten all
of it out into a large eight-dimensional vector because we don't want to Matrix multiply
80. into a weight Matrix multiply immediately instead we want to group
these like this so every consecutive two elements
one two and three and four and five and six and seven and eight all of these should be now basically flattened out and multiplied
by weight Matrix but all of these four groups here we'd like to process in parallel so it's kind of like a batch
Dimension that we can introduce and then we can in parallel basically
process all of these uh bigram groups in the four batch dimensions of an
individual example and also over the actual batch dimension of the you know four examples in our example here so
let's see how that works effectively what we want is right now we take a 4 by 80
and multiply it by 80 by 200 to in the linear layer this is what happens
but instead what we want is we don't want 80 characters or 80 numbers to come in we only want two characters to come
in on the very first layer and those two characters should be fused so in other words we just want 20 to
come in right 20 numbers would come in and here we don't want a 4 by 80 to feed
into the linear layer we actually want these groups of two to feed in so instead of four by eighty we want this
to be a 4 by 4 by 20. so these are the four groups of two and
each one of them is ten dimensional vector so what we want is now is we need to change the flattened layer so it doesn't
output a four by eighty but it outputs a four by four by Twenty where basically these um
every two consecutive characters are uh packed in on the very last Dimension and
then these four is the first batch Dimension and this four is the second batch Dimension referring to the four
groups inside every one of these examples and then this will just multiply like this so this is what we want to get to
so we're going to have to change the linear layer in terms of how many inputs it expects it shouldn't expect 80 it
should just expect 20 numbers and we have to change our flattened layer so it doesn't just fully flatten out this
entire example it needs to create a 4x4 by 20 instead of four by eighty so let's
see how this could be implemented basically right now we have an input that is a four by eight by ten that
feeds into the flattened layer and currently the flattened layer just stretches it out so if you remember the
implementation of flatten it takes RX and it just views it as whatever the batch Dimension is and then
negative one so effectively what it does right now is it does e dot view of 4 negative one and
the shape of this of course is 4 by 80. so that's what currently happens and we
instead want this to be a four by four by Twenty where these consecutive ten-dimensional vectors get concatenated
so you know how in Python you can take a list of range of 10
so we have numbers from zero to nine and we can index like this to get all the
even parts and we can also index like starting at one and going in steps up two to get all
the odd parts so one way to implement this it would be as follows we can take e and we can
index into it for all the batch elements and then just even elements in this
Dimension so at indexes 0 2 4 and 8. and then all the parts here from this
last dimension and this gives us the even characters
and then here this gives us all the odd characters and basically what we want to do is we make
sure we want to make sure that these get concatenated in pi torch and then we want to concatenate these two tensors
along the second dimension so this and the shape of it would be
four by four by Twenty this is definitely the result we want we are explicitly grabbing the even parts and
the odd parts and we're arranging those four by four by ten right next to each other and concatenate
so this works but it turns out that what also works is you can simply use a view again and just request the right shape
and it just so happens that in this case those vectors will again end up being arranged in exactly the way we want so
in particular if we take e and we just view it as a four by four by Twenty which is what we want
we can check that this is exactly equal to but let me call this this is the explicit concatenation I suppose
um so explosives dot shape is 4x4 by 20. if you just view it as 4x4 by 20 you can
check that when you compare to explicit uh you got a big this is element wise
operation so making sure that all of them are true that is the truth so basically long story short we don't need
to make an explicit call to concatenate Etc we can simply take this input tensor
to flatten and we can just view it in whatever way we want and in particular you don't want to
stretch things out with negative one we want to actually create a three-dimensional array and depending on
how many vectors that are consecutive we want to um fuse like for example two then we can
just simply ask for this Dimension to be 20. and um use a negative 1 here and python will
figure out how many groups it needs to pack into this additional batch dimension so let's now go into flatten and
implement this okay so I scroll up here to flatten and what we'd like to do is we'd like to change it now so let me
create a Constructor and take the number of elements that are consecutive that we would like to concatenate now in the
last dimension of the output so here we're just going to remember solve.n equals n
and then I want to be careful here because pipe pytorch actually has a torch to flatten and its keyword
arguments are different and they kind of like function differently so R flatten is going to start to depart from patreon
flatten so let me call it flat flatten consecutive or something like that just to make sure that our apis are about
equal so this uh basically flattens only some n consecutive elements and puts them
into the last dimension now here the shape of X is B by T by C
so let me pop those out into variables and recall that in our example down below B was 4 T
was 8 and C was 10. now instead of doing x dot view of B by
negative one right this is what we had before
we want this to be B by um negative 1 by
and basically here we want c times n that's how many consecutive elements we
want and here instead of negative one I don't super love the use of negative one because I like to be very explicit so
that you get error messages when things don't go according to your expectation so what do we expect here we expect this
to become t divide n using integer division here so that's what I expect to happen
and then one more thing I want to do here is remember previously all the way in the beginning n was three and uh
basically we're concatenating um all the three characters that existed there so we basically are concatenated
everything and so sometimes I can create a spurious dimension of one here so if it is the
case that x dot shape at one is one then it's kind of like a spurious dimension
um so we don't want to return a three-dimensional tensor with a one here we just want to return a two-dimensional
tensor exactly as we did before so in this case basically we will just say x equals x dot squeeze that is a
pytorch function and squeeze takes a dimension that it
either squeezes out all the dimensions of a tensor that are one or you can specify the exact Dimension that you
want to be squeezed and again I like to be as explicit as possible always so I expect to squeeze out the First
Dimension only of this tensor this three-dimensional tensor and if
this Dimension here is one then I just want to return B by c times n and so self dot out will be X and then
we return salt dot out so that's the candidate implementation and of course this should be self.n
instead of just n so let's run and let's come here now
and take it for a spin so flatten consecutive
and in the beginning let's just use eight so this should recover the previous Behavior so flagging
consecutive of eight uh which is the current block size we can do this uh that should recover
the previous Behavior so we should be able to run the model and here we can inspect I have a little
code snippet here where I iterate over all the layers I print the name of this
class and the shape and so we see the shapes as we expect
them after every single layer in the top bit so now let's try to restructure it
using our flattened consecutive and do it hierarchically so in particular
we want to flatten consecutive not just not block size but just two and then we want to process this with
linear now then the number of inputs to this linear will not be an embed times block size it will now only be n embed
times two 20. this goes through the first layer and
now we can in principle just copy paste this now the next linear layer should expect and hidden times two
and the last piece of it should expect and it enters 2 again
so this is sort of like the naive version of it um so running this we now have a much much
bigger model and we should be able to basically just forward the model
and now we can inspect uh the numbers in between so four byte by 20
was Platinum consecutively into four by four by Twenty this was projected into four by four by
two hundred and then bash storm just worked out of the box we have to verify that bastron
does the correct thing even though it takes a three-dimensional impedance that are two dimensional input then we have 10h which is element wise
then we crushed it again so if we flatten consecutively and ended up with a four by two by 400 now
then linear brought it back down to 200 batch room 10h and lastly we get a 4 by
400 and we see that the flattened consecutive for the last flatten here uh it squeezed out that dimension of one so
we only ended up with four by four hundred and then linear Bachelor on 10h and uh the last linear layer to get our
logents and so The Lodges end up in the same shape as they were before but now we actually have a nice three layer
neural nut and it basically corresponds to whoops sorry it basically corresponds
exactly to this network now except only this piece here because we only have three layers whereas here in this
example there's uh four layers with the total receptive field size of 16
characters instead of just eight characters so the block size here is 16. so this piece of it's basically
implemented here um now we just have to kind of figure out some good Channel numbers to use
here now in particular I changed the number of hidden units to be 68 in this architecture because when I use 68 the

当然！这一段是**WaveNet 结构实现**的关键部分，内容很多，我帮你整理成**详细中文讲解**：

---

### **# 实现 WaveNet**

1️⃣ **建立调试区**
作者先做了一个“scratch space”（调试区）用来**观察 forward 过程中的张量 shape**变化。

* 随机选了 batch size=4 个样本（4 行数据），
* 每个样本里有 **8 个字符**（block\_size=8），
* 传入模型 `xB`，shape 是 **\[4, 8]**，每个元素是字符 ID（int tensor）。

---

2️⃣ **Embedding 层**

* 通过 embedding layer，把 int tensor 转成向量：
  `xB` → `embedding(xB)` → shape 变成 **\[4, 8, 10]**

  * 每个字符映射成 10 维向量（10-dimensional vector）

---

3️⃣ **Flatten 层**（传统 MLP 的 flatten）

* 之前 flatten 是**直接 view 成 \[4, 80]**，
  就是 8 个 10 维 embedding 展开拼接成一行：
  `4 (batch), 8 (tokens), 10 (embed_dim)` → flatten → `[4, 80]`

---

4️⃣ **Linear 层**

* 然后 Linear 层做矩阵乘法：
  `[4, 80] * [80, 200] → [4, 200]`
  也就是把 80 维向量映射成 200 维。

---

### **关键知识点：高维张量的矩阵乘法**

PyTorch 的 Linear 支持高维输入，比如 `[4, 5, 80]` \* `[80, 200]` → `[4, 5, 200]`

* 多出来的维度会被当作 batch 维度，矩阵乘法只作用在最后一维。
* 这个特性可以用来实现“并行处理 bigram”。

---

5️⃣ **实现 WaveNet 样式的逐步融合**

* WaveNet 不想把 8 个字符一次性 flatten，

  * 而是**成对融合** → 形成 bigram → 继续融合 → 形成 4-gram → 继续融合...

目标：

* 第一层：`[4, 4, 20]`

  * 每两个字符（10 + 10）拼接 → 20 维 → 每个样本有 4 个 bigram
* 第二层：`[4, 2, 40]`

  * 每两个 bigram 拼接 → 40 维
* 第三层：`[4, 1, 80]`

  * 最终拼接 → 80 维
  * squeeze 掉多余维度 → `[4, 80]`
* 然后输出层预测。

---

6️⃣ **Flatten consecutive 层实现**

* 写了一个新的 `FlattenConsecutive(n)` 层

  * 每 n 个连续向量拼接
  * 不全 flatten，只拼接局部
  * 输出 shape: `[B, T // n, C * n]`

---

7️⃣ **网络改写**
新的网络结构大致是：

```
Embedding  
→ FlattenConsecutive(2) → Linear → BatchNorm → Tanh  
→ FlattenConsecutive(2) → Linear → BatchNorm → Tanh  
→ FlattenConsecutive(2) → Linear → BatchNorm → Tanh  
→ Linear → Output
```

* 每一层是“成对融合”，逐步压缩 token 数。
* 最终输出的 logits shape 和以前一样 `[batch_size, vocab_size]`。

---

8️⃣ **测试 Forward 过程**

* 打印出每层的 tensor shape，确认 shape 变化是否符合预期。
* BatchNorm 也可以直接处理这种 3D tensor，没问题。
* 最终 logits 和原来一样。

---

### **结果**

* 现在实现了 3 层的“树状融合网络”，对应 WaveNet 里的前半段结构。
* 当前 block\_size=8，理论上可以扩展到 block\_size=16， receptive field 变大。
* 现在需要再调整 hidden channels 这些超参数，进一步优化效果。

---

### 总结一句话：

通过 `FlattenConsecutive + Linear`，就能实现 WaveNet 的**逐步融合**，比传统 MLP 一下子 flatten 全拼接更合理，效果更好。

---

如果你需要，我还可以帮你总结一个“**WaveNet 层级实现模板代码**”，你可以直接套用！要不要？🌟

```python
class FlattenConsecutive:

  def __init__(self, n):
    self.n = n

  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out

  def parameters(self):
    return []
```

解释一下

当然！这段是 `FlattenConsecutive` 的类定义，我们来一行一行讲解：

---

```python
class FlattenConsecutive:
```

定义一个**自定义的 PyTorch layer**，名字叫 `FlattenConsecutive`，
这个层的作用是：

✅ 把 **相邻 n 个 token 的 embedding 向量拼接**，
✅ 把 T 个 token → 拼成 T//n 个 token，
✅ 每个拼接后的 token 维度是 `C * n`。

---

```python
def __init__(self, n):
    self.n = n
```

* 初始化时传入参数 `n`：表示**多少个连续 token 拼成 1 个**。
  比如 `n=2`，就是 **2 个 token 拼成 1 个 bigram**。

---

```python
def __call__(self, x):
```

* 这里定义了 `__call__` 方法，
* 也就是说：**调用这个类实例的时候，就直接触发这个 forward 操作**，
  很像 PyTorch 的 forward 方法。

---

```python
B, T, C = x.shape
```

* 把输入张量 `x` 的 shape 解包成：

  * `B`: batch size
  * `T`: sequence length（token 数）
  * `C`: 每个 token 的 embedding 维度

---

```python
x = x.view(B, T//self.n, C*self.n)
```

* 关键操作！

* 通过 `.view()` 重塑 shape：

  * 原本有 `T` 个 token，每个是 `C` 维，
  * 现在变成 `T//n` 个 token，每个 token 是 `C*n` 维（拼接了 n 个原始 token）！

* 举例：

```
原 shape:  (B, 8, 10)    → 8 个 10 维 token
拼接 2 个: → (B, 4, 20)  → 4 个 20 维 token
```

---

```python
if x.shape[1] == 1:
    x = x.squeeze(1)
```

* 如果拼接完后，`T//n == 1`，
  也就是说 **只剩 1 个 token**，就多出来一个维度是 1，
  用 `.squeeze(1)` 去掉这一维，让输出 shape 更干净。

---

```python
self.out = x
return self.out
```

* 保存结果 `self.out`，同时返回输出。

---

```python
def parameters(self):
    return []
```

* 这是为了和其它 PyTorch layer 的接口保持一致：
  这个 layer **不包含参数**，所以 `parameters()` 返回空列表。

---

### 总结：

`FlattenConsecutive(n)` 的作用是：

✅ 输入形状 `(B, T, C)`
✅ 输出形状 `(B, T//n, C * n)`
✅ 把 **相邻 n 个 token 拼接成一个新的 token**，embedding 维度增加

---

### 举例：

```python
e = torch.randn(4, 8, 10)
flatten = FlattenConsecutive(2)
out = flatten(e)

# out.shape == (4, 4, 20)
```

---

如果你还想，我也可以帮你整理一份“FlattenConsecutive 在 WaveNet 结构中怎么用”的小例子代码，要不要？🌟



# training the WaveNet: first pass

number of parameters comes out to be 22 000 so that's exactly the same that we had before and we have the same amount
of capacity at this neural net in terms of the number of parameters but the question is whether we are utilizing
those parameters in a more efficient architecture so what I did then is I got rid of a lot of the debugging cells here
and I rerun the optimization and scrolling down to the result we see that we get the identical performance roughly
so our validation loss now is 2.029 and previously it was 2.027 so controlling
for the number of parameters changing from the flat to hierarchical is not giving us anything yet that said there are two things
um to point out number one we didn't really torture the um architecture here very much this is just my first guess
and there's a bunch of hyper parameters search that we could do in order in terms of how we allocate uh our budget
of parameters to what layers number two we still may have a bug inside the
bachelor 1D layer so let's take a look at um uh that because it runs but does it
do the right thing so I pulled up the layer inspector sort of that we have here and printed out the

当然，这段我来帮你翻译讲解：

---

**# 训练 WaveNet（第一轮）**

* 现在的模型参数总数大概 **22,000**，
  这个和之前的 MLP 结构是**一样的参数量**，所以**模型容量（capacity）是一样的**，
  问题是：

  * 通过 WaveNet 这种结构，**这些参数是否被用得更有效**？

---

* 然后，作者把前面的一些调试用的 cell 删掉，重新跑了训练过程。

* 训练结果：

  * 现在验证集 loss 大概是 **2.029**，
  * 之前的 baseline MLP 是 **2.027**，
    → 也就是说，**目前这轮 WaveNet 和 MLP 效果差不多**，没有明显提升。

---

作者总结两点原因：

1️⃣ **还没有调超参数**

* 目前的 WaveNet 架构只是一个初版“猜的”版本，
* 还没有细调 hidden size / channel 数量 / 层数 / learning rate 等等，
* 其实可以花点时间做超参数搜索（hyperparameter search），看看参数 budget 如何更好地分配到各层，可能能显著提升效果。

---

2️⃣ **BatchNorm1D 可能还有 bug**

* 虽然网络能跑，但是 BatchNorm1D 层的实现要确认一下，
* 是否在这种 3D tensor 场景下正确工作（之前改了层结构，BatchNorm 现在接的输入 tensor shape 变了），
* 需要仔细检查 BatchNorm 层，避免因为实现问题影响了网络效果。

---

接下来，作者就准备去看 BatchNorm 层的实现和行为了。

---

### 总结一句话：

**第一版 WaveNet 结构可以正常训练，参数量和 MLP 持平，初步效果持平，接下来需要调参 + 检查 BN 层实现，才可能释放 WaveNet 架构的优势。**

---

如果你需要，我可以帮你整理一份“WaveNet 结构 vs MLP baseline 对比表 + 下一步优化计划”，
方便你以后自己做实验时也可以参考。要不要？🌟


# fixing batchnorm1d bug

shape along the way and currently it looks like the batch form is receiving an input that is 32 by 4 by 68 right and
here on the right I have the current implementation of Bachelor that we have right now now this bachelor assumed in the way we
wrote it and at the time that X is two-dimensional so it was n by D where n
was the batch size so that's why we only reduced uh the mean and the variance over the zeroth dimension but now X will
basically become three-dimensional so what's happening inside the bachelor right now and how come it's working at all and not giving any errors the reason
for that is basically because everything broadcasts properly but the bachelor is not doing what we need what we wanted to
do so in particular let's basically think through what's happening inside the bathroom uh looking at what's what's do
What's Happening Here I have the code here so we're receiving an input of 32 by 4
by 68 and then we are doing uh here x dot mean here I have e instead of X but
we're doing the mean over zero and that's actually giving us 1 by 4 by 68. so we're doing the mean only over the
very first Dimension and it's giving us a mean and a variance that still maintain this Dimension here
so these means are only taking over 32 numbers in the First Dimension and then when we perform this everything
broadcasts correctly still but basically what ends up happening is
when we also look at the running mean
the shape of it so I'm looking at the model that layers at three which is the first bathroom layer and they're looking at whatever the running mean became and
its shape the shape of this running mean now is 1 by 4 by 68.
right instead of it being um you know just a size of dimension
because we have 68 channels we expect to have 68 means and variances that we're maintaining but actually we have an
array of 4 by 68 and so basically what this is telling us is this bash Norm is only
this bachelor is currently working in parallel over
4 times 68 instead of just 68 channels so basically we are maintaining
statistics for every one of these four positions individually and independently
and instead what we want to do is we want to treat this four as a batch Dimension just like the zeroth dimension
so as far as the bachelor is concerned it doesn't want to average we don't want
to average over 32 numbers we want to now average over 32 times four numbers for every single one of these 68
channels and uh so let me now remove this
it turns out that when you look at the documentation of torch.mean
so let's go to torch.me
in one of its signatures when we specify the dimension we see that the dimension here is not
just it can be in or it can also be a tuple of ins so we can reduce over
multiple integers at the same time over multiple Dimensions at the same time so instead of just reducing over zero we
can pass in a tuple 0 1. and here zero one as well and then what's going to happen is the output of
course is going to be the same but now what's going to happen is because we reduce over 0 and 1 if we
look at immin.shape we see that now we've reduced we took
the mean over both the zeroth and the First Dimension so we're just getting 68 numbers and a
bunch of spurious Dimensions here so now this becomes 1 by 1 by 68 and the
running mean and the running variance analogously will become one by one by 68. so even though there are the
spurious Dimensions uh the current the current the correct thing will happen in that we are only maintaining means and
variances for 64 sorry for 68 channels and we're not calculating the mean
variance across 32 times 4 dimensions so that's exactly what we want and let's
change the implementation of bash term 1D that we have so that it can take in two-dimensional or three-dimensional
inputs and perform accordingly so at the end of the day the fix is relatively straightforward basically the dimension
we want to reduce over is either 0 or the Tuple zero and one depending on the dimensionality of X so if x dot and dim
is two so it's a two dimensional tensor then Dimension we want to reduce over is just the integer zero
L if x dot ending is three so it's a three-dimensional tensor then the dims we're going to assume are zero and one
that we want to reduce over and then here we just pass in dim and if the dimensionality of X is
anything else we'll now get an error which is good um so that should be the fix now I want
to point out one more thing we're actually departing from the API of Pi torch here a little bit because when you
come to batch room 1D and pytorch you can scroll down and you can see that the input to this layer can either be n by C
where n is the batch size and C is the number of features or channels or it actually does accept three-dimensional
inputs but it expects it to be n by C by L where LSA like the sequence length or
something like that so um this is problem because you see how C is
nested here in the middle and so when it gets three-dimensional inputs this bash term layer will reduce over zero and two
instead of zero and one so it basically Pi torch batch number one D layer
assumes that c will always be the First Dimension whereas we'll we assume here
that c is the last Dimension and there are some number of batch Dimensions beforehand
um and so it expects n by C or M by C by all we
expect and by C or n by L by C and so it's a deviation
um I think it's okay I prefer it this way honestly so this is the way that we will
keep it for our purposes so I redefined the layers re-initialize the neural net and did a single forward pass with a break just for one step
looking at the shapes along the way they're of course identical all the shapes are the same but the way we see
that things are actually working as we want them to now is that when we look at the bathroom layer the running mean
shape is now one by one by 68. so we're only maintaining 68 means for every one
of our channels and we're treating both the zeroth and the First Dimension as a batch Dimension which is exactly what we
want so let me retrain the neural lot now okay so I retrained the neural net with the bug fix we get a nice curve and

当然！这一段是讲 **BatchNorm1D 层的 bug 修复**，内容很多，我帮你细细翻译讲解：

---

### 背景问题：

在改成 WaveNet 的树状网络后，中间层的张量 shape 变成了 **3 维**：

```
当前 BatchNorm 收到的输入是：(32, 4, 68)
```

解释：

* 32：batch size
* 4：一层里每个样本的 token 数（比如 bigram 后变成 4 个 token）
* 68：每个 token 的 channel 数（hidden dim）

---

而我们原来的 BatchNorm 层是手写的，默认只处理 **2D** 的输入：

```
(N, D) —> (batch_size, channel_dim)
```

原代码只对第 0 维（batch 维）求 mean 和 var：

```python
mean = x.mean(dim=0)
var  = x.var(dim=0)
```

---

但是现在 x 是 3 维的 `(32, 4, 68)`，
**BatchNorm 应该对 batch 维和 token 维（前两维）一起求 mean/var**！

---

### 目前发生了什么？

现在代码还能“跑”，是因为：

* PyTorch 广播机制自动帮忙扩展维度，代码不报错；
* 但是实际上，**计算出来的 running\_mean / running\_var 是错误的**！

比如：

```python
running_mean.shape → (1, 4, 68)   ❌
```

实际上我们只希望有：

```python
running_mean.shape → (1, 1, 68)   ✅
```

也就是说：

✅ 只对每个 channel 统计 68 个 mean/var，
不应该区分 token 位置（4），**位置不应该影响统计**，
应该把 batch 维和 token 维都当成 “batch 维度” 求平均。

---

### 如何修？

PyTorch `mean()` 支持多维同时 reduce，可以传 tuple：

```python
mean = x.mean(dim=(0,1))  # 对 batch 维 和 token 维 同时求 mean
```

同理：

```python
var = x.var(dim=(0,1))
```

这样 running\_mean / running\_var 就会是：

```python
shape → (1, 1, 68)
```

---

### 代码修正逻辑

```python
if x.ndim == 2:
    dim = 0
elif x.ndim == 3:
    dim = (0, 1)
else:
    raise Error
```

然后统一传 `dim` 给 mean/var 计算即可。

---

### PyTorch 官方 BatchNorm1D 差异：

PyTorch 的官方 BatchNorm1D 接受：

```
(N, C) 或 (N, C, L)
```

也就是说，**C 这个维度必须放在中间**，
我们自定义的版本，是：

```
(N, L, C)
```

也就是 **C 在最后一维**，这是 WaveNet 网络里更方便的格式，
虽然和官方 API 不一样，**但为了好用，继续用这个版本就好**！

---

### 结果验证：

重新定义 BatchNorm 层 → 重新初始化模型 → 只跑 1 个 step，检查：

```python
running_mean.shape == (1, 1, 68)  ✅
```

说明 bug 修好了，BatchNorm 层现在对 batch 和 token 维做了正确的归一化，
不会把 token 位置 (4) 也统计进去。

---

### 总结一句话：

**原先 BatchNorm1D 假定输入 2D，现在输入是 3D，必须改成同时对 (batch\_dim, token\_dim) 归一化，修完 bug 后就正常了。**

---

如果你需要，我也可以帮你写一份“**手写 BatchNorm1D 兼容 2D / 3D 版本**”的完整代码模板，
你以后可以直接套用！要不要？🌟


# re-training WaveNet with bug fix

when we look at the validation performance we do actually see a slight Improvement so we went from 2.029 to
2.022 so basically the bug inside the bathroom was holding up us back like a
little bit it looks like and we are getting a tiny Improvement now but it's not clear if this is statistical significant
um and the reason we slightly expect an improvement is because we're not maintaining so many different means and
variances that are only estimated using using 32 numbers effectively now we are estimating them using 32 times 4 numbers
so you just have a lot more numbers that go into any one estimate of the mean and variance and it allows things to be a
bit more stable and less Wiggly inside those estimates of those statistics so

当然，这一段我帮你翻译讲解：

---

**# 修好 BatchNorm bug 后重新训练 WaveNet**

修复了 BatchNorm 的 bug 之后，重新训练了一轮 WaveNet，
观察验证集的性能，确实看到了一点点提升：

✅ 验证 loss 从 **2.029 → 2.022**，有一个小幅度的改进。

---

说明：

* 之前的 bug 在 BatchNorm 层，**mean / var 统计不准确**，有一点影响性能；
* 修复后，效果略有提升，但目前还不确定这个提升是不是“统计显著的”——也有可能是随机波动。

---

为什么理论上**期望会有提升**？

✅ 修复前：

* BatchNorm 里在 `(32, 4, 68)` 这个张量上，
* 每个 token 位置单独维护 mean/var，
* 每组 mean/var 只有 32 个样本参与统计（每次只有 batch size 个数）

✅ 修复后：

* 现在把 `(batch 维 + token 维)` 一起当成 batch，
* 每个 channel 的 mean/var 是在 `32 * 4 = 128` 个样本上统计的，
* 数据量更大，估计更稳定，不容易抖动（less wiggly）。

---

### 总结一句话：

修好 BatchNorm 后，loss 有小幅度改善，
因为现在 mean/var 的估计更稳定，网络学得更扎实一点了！

---

如果你需要，我也可以帮你画个对比小图，**修复前后 loss 曲线对比**，要不要？🌟


# scaling up our WaveNet

pretty nice with this more General architecture in place we are now set up to push the performance further by
increasing the size of the network so for example I bumped up the number of embeddings to 24 instead of 10 and also
increased number of hidden units but using the exact same architecture we now have 76 000 parameters and the training
takes a lot longer but we do get a nice curve and then when you actually evaluate the performance we are now
getting validation performance of 1.993 so we've crossed over the 2.0 sort of
territory and right about 1.99 but we are starting to have to wait quite a bit
longer and we're a little bit in the dark with respect to the correct setting of the hyper parameters here and the
learning rates and so on because the experiments are starting to take longer to train and so we are missing sort of like an experimental harness on which we
could run a number of experiments and really tune this architecture very well so I'd like to conclude now with a few

当然，这段我来帮你翻译讲解：

---

**# 扩大 WaveNet 规模**

现在我们已经实现了这个**更通用的 WaveNet 架构**，
接下来就可以通过**扩大网络规模**，继续提升模型性能。

---

举例来说：

* 我把 embedding 维度从 10 提高到了 **24**，
* 同时增加了 hidden 层的通道数（hidden units），
* 但是整体架构**保持不变**。

---

调整后：

* 模型参数总数变成了 **76,000** 个（原来只有 22,000）。
* 训练时间明显变长了，
* 但得到了一条不错的 loss 曲线。

---

性能提升：

✅ 验证集 loss 现在达到了 **1.993**，
✅ 也就是说 loss 已经突破了 “2.0” 这个区间，达到 **1.99**，
✅ 网络规模变大确实能带来效果提升。

---

不过：

* 训练时间越来越长，
* 超参数（hyperparameters）、learning rate 等**还没有细调**，
* 训练一个实验结果需要时间，调参效率低，
* 现在**缺乏一个更好的实验框架**（experimental harness）来管理多个实验，自动调参会更高效。

---

### 总结一句话：

**扩大 WaveNet 规模 → loss 明显下降 → 但是训练慢 + 调参困难，需要更好的实验框架来优化整体效果**。

---

如果你需要，我也可以帮你整理一份**WaveNet 扩大规模时，超参数调整建议表**，
方便你以后练习时知道该调哪些参数，要不要？🌟


# experimental harness

notes we basically improved our performance from a starting of 2.1 down to 1.9 but I don't want that to be the
focus because honestly we're kind of in the dark we have no experimental harness we're just guessing and checking and
this whole thing is terrible we're just looking at the training loss normally you want to look at both the training and the validation loss together and the
whole thing looks different if you're actually trying to squeeze out numbers that said we did implement this
architecture from the wavenet paper but we did not implement this specific uh
forward pass of it where you have a more complicated a linear layer sort of that is this gated linear layer kind of and
there's residual connections and Skip connections and so on so we did not Implement that we just implemented this

当然，这一段我来帮你翻译讲解：

---

**# 实验框架（experimental harness）**

目前我们其实把模型的验证 loss 从最初的 **2.1** 提升到了 **1.9**，
✅ 看起来效果提升了，
❗️ 但作者说不希望大家太关注这个数字，为什么呢？

---

原因：

* **我们现在还没有一个好的实验框架**，
* 整个过程基本是“猜 + 试”，没有系统性，
* 也就是说，这种改进**缺乏科学性**，是靠运气的成分大，
* 目前训练过程中我们只看了 **training loss**，

  * 正确做法应该是同时看 **training loss + validation loss**，
  * 两者一起看，才能判断是否过拟合、是否真的泛化提升。

---

* 如果真的想要“榨干性能（squeeze out numbers）”，
  → 一定需要更系统的实验管理框架（harness），
  → 自动化跑各种超参数组合、自动记录结果。

---

另外：

* 虽然我们实现了 WaveNet 论文里的这种“树状”架构，
* **但还没有实现论文中的 gated linear unit (GLU) 结构**，

  * 论文里 forward pass 更复杂，包括：
    ✅ gated linear layers（门控线性层）
    ✅ residual connections（残差连接）
    ✅ skip connections（跳跃连接）

---

也就是说：

✅ 我们目前实现了 WaveNet 的基本思想结构，
❌ 但还没有完整实现论文里更强大的 forward 结构（带门控 + 残差的版本）。

---

### 总结一句话：

**目前提升到 1.9 的 loss，是靠手动调试得到的，缺乏系统的实验框架；
完整的 WaveNet forward 结构还有很多细节没实现，后续如果补上，效果应该还能进一步提升。**

---

如果你需要，我也可以帮你整理一份**完整的 WaveNet forward 结构组件清单**，
方便你以后按论文一步步实现，要不要？🌟


# WaveNet but with “dilated causal convolutions”

structure I would like to briefly hint or preview how what we've done here relates to convolutional neural networks
as used in the wavenet paper and basically the use of convolutions is strictly for efficiency it doesn't
actually change the model we've implemented so here for example let me look at a specific name to work
with an example so there's a name in our training set and it's DeAndre and it has
seven letters so that is eight independent examples in our model so all these rows here are independent examples
of the Android now you can forward of course any one of these rows independently so I can take
my model and call call it on any individual index notice by the way here
I'm being a little bit tricky the reason for this is that extra at seven that shape is just
um one dimensional array of eight so you can't actually call the model on it you're going to get an error because
there's no batch dimension so when you do extra at
a list of seven then the shape of this becomes one by eight so I get an extra batch dimension of one and then we can
forward the model so that forwards a single example and you
might imagine that you actually may want to forward all of these eight um at the same time
so pre-allocating some memory and then doing a for Loop eight times and forwarding all of those eight here will
give us all the logits in all these different cases now for us with the model as we've implemented it right now this is eight
independent calls to our model but what convolutions allow you to do is it allow you to basically slide this
model efficiently over the input sequence and so this for Loop can be
done not outside in Python but inside of kernels in Cuda and so this for Loop
gets hidden into the convolution so the convolution basically you can cover this it's a for Loop applying a
little linear filter over space of some input sequence and in our case the space
we're interested in is one dimensional and we're interested in sliding these filters over the input data
so this diagram actually is fairly good as well basically what we've done is here they
are highlighting in Black one individ one single sort of like tree of this calculation so just calculating the
single output example here um and so this is basically what we've
implemented here we've implemented a single this black structure we've implemented that and calculated a single
output like a single example but what collusions allow you to do is it allows you to take this black
structure and kind of like slide it over the input sequence here and calculate
all of these orange outputs at the same time or here that corresponds to
calculating all of these outputs of um at all the positions of DeAndre at
the same time and the reason that this is much more efficient is because number one as I
mentioned the for Loop is inside the Cuda kernels in the sliding so that
makes it efficient but number two notice the variable reuse here for example if we look at this circle this node here
this node here is the right child of this node but is also the left child of
the node here and so basically this node and its value is used twice
and so right now in this naive way we'd have to recalculate it but here we are
allowed to reuse it so in the convolutional neural network you think of these linear layers that we
have up above as filters and we take these filters and they're linear filters
and you slide them over input sequence and we calculate the first layer and then the second layer and then the third
layer and then the output layer of the sandwich and it's all done very efficiently using these convolutions
so we're going to cover that in a future video the second thing I hope you took away from this video is you've seen me

当然，这段我来帮你详细翻译讲解：

---

**# 带“扩张因果卷积”的 WaveNet**

作者在这里简单预告了一下 “**扩张因果卷积（dilated causal convolutions）**” 和我们目前实现的 WaveNet 有什么关系。

---

**核心观点**：

* 论文里用卷积（convolutional neural networks, CNN）其实**不是为了改模型结构**，
* 纯粹是**为了提高计算效率**，并不改变模型的表达能力。
* 换句话说：**我们现在的 WaveNet 结构，和用卷积实现出来的是“功能等效的”**。

---

### 举例：

训练集里有个名字 `DeAndre`，长度 7 个字母，
→ 对我们来说，这相当于 8 个样本（包括起始 token）。

目前我们实现的模型：

* 你可以手动 **单独 forward 这 8 个样本**，
* 比如：

```python
model(x[0])  
model(x[1])  
...
```

或者用 for-loop 批量 forward 8 个。

---

### 现在的问题：

**这种手动 for-loop 是低效的**！

* 每一个 forward 调用都是**独立计算**，
* 没有复用中间结果，
* 很多地方会重复计算，速度慢。

---

### 卷积能做什么？

**卷积 = 用一个 sliding window 滑动计算**，
本质上相当于：

```text
for i in range(len(x)):
    out[i] = linear_filter( x[i:i+window_size] )
```

**好处**：

1️⃣ **for-loop 是写在 Cuda kernel 里的，不是 Python 的循环，速度快**！
2️⃣ **中间结果可以复用**：

举例：

```
          parent
         /      \
    left_child  right_child
```

* 某个节点计算完后，左右 child 节点可以被多次复用，
* 而我们目前 naive 实现是“每次重复算一遍”，浪费时间。

---

### 图解：

论文里的图也很好说明了：

* 目前我们实现的是“黑色的树”：只计算一个 output，
* 卷积可以**平移这棵树**，滑动整个序列，
* 同时计算出“橙色的所有 output”。

举例：

```text
DeAndre

用目前 naive 实现：for 循环 8 次，每次算一个 output
用卷积实现：一遍滑过去，8 个 output 一次算出来，效率高
```

---

### 总结一句话：

✅ 我们目前实现的 WaveNet 树状结构 和 论文的 CNN 版是**同一个模型**，
✅ CNN 的作用只是 “优化计算速度 + 复用中间变量”，本质表达能力没变，
✅ 未来会用 dilated causal convolutions 来实现更高效版。

---

如果你需要，我也可以帮你整理一份\*\*“目前实现 vs 卷积实现的对比表”\*\*，
帮你更好理解为什么卷积会快很多，要不要？🌟


# torch.nn

basically Implement all of these layer Lego building blocks or module building
blocks and I'm implementing them over here and we've implemented a number of layers together and we've also
implemented these these containers and we've overall pytorchified our code
quite a bit more now basically what we're doing here is we're re-implementing torch.nn which is
the neural networks library on top of torch.tensor and it looks very much like
this except it is much better because because it's in pi torch instead of jingling my Jupiter notebook so I think
going forward I will probably have considered us having unlocked um torch.nn we understand roughly what's
in there how these modules work how they're nested and what they're doing on top of torture tensor so hopefully we'll
just uh we'll just switch over and continue and start using torch.net directly the next thing I hope you got a

# the development process of building deep neural nets

bit of a sense of is what the development process of building deep neural networks looks like which I think
was relatively representative to some extent so number one we are spending a lot of time in the documentation page of
pytorch and we're reading through all the layers looking at documentations where the shapes of the inputs what can
they be what does the layer do and so on unfortunately I have to say the patreon's documentation is not are very
good they spend a ton of time on Hardcore engineering of all kinds of distributed Primitives Etc but as far as
I can tell no one is maintaining any documentation it will lie to you it will be wrong it will be incomplete it will
be unclear so unfortunately it is what it is and you just kind of do your best
um with what they've given us um number two
uh the other thing that I hope you got a sense of is there's a ton of trying to make the shapes work and there's a lot
of gymnastics around these multi-dimensional arrays and are they two-dimensional three-dimensional four-dimensional uh what layers take
what shapes is it NCL or NLC and you're promoting and viewing and it just can
get pretty messy and so that brings me to number three I very often prototype these layers and implementations in
jupyter notebooks and make sure that all the shapes work out and I'm spending a lot of time basically babysitting the
shapes and making sure everything is correct and then once I'm satisfied with the functionality in the Jupiter notebook I will take that code and copy
paste it into my repository of actual code that I'm training with and so then
I'm working with vs code on the side so I usually have jupyter notebook and vs code I develop in Jupiter notebook I
paste into vs code and then I kick off experiments from from the reaper of course from the code repository so
that's roughly some notes on the development process of working with neurons lastly I think this lecture unlocks a lot of potential further

# going forward

lectures because number one we have to convert our neural network to actually use these dilated causal convolutional
layers so implementing the comnet number two potentially starting to get into
what this means whatever residual connections and Skip connections and why are they useful
number three we as I mentioned we don't have any experimental harness so right now I'm just guessing checking
everything this is not representative of typical deep learning workflows you have to set up your evaluation harness you
can kick off experiments you have lots of arguments that your script can take you're you're kicking off a lot of experimentation you're looking at a lot
of plots of training and validation losses and you're looking at what is working and what is not working and you're working on this like population
level and you're doing all these hyper parameter searches and so we've done none of that so far so how to set that
up and how to make it good I think as a whole another topic number three we
should probably cover recurring neural networks RNs lstm's grooves and of course Transformers so many uh places to
go and we'll cover that in the future for now bye sorry I forgot to say that

# improve on my loss! how far can we improve a WaveNet on this data?

if you are interested I think it is kind of interesting to try to beat this number 1.993 because I really haven't
tried a lot of experimentation here and there's quite a bit of fruit potentially to still purchase further so I haven't
tried any other ways of allocating these channels in this neural net maybe the number of dimensions for the embedding
is all wrong maybe it's possible to actually take the original network with just one hidden layer and make it big
enough and actually beat my fancy hierarchical Network it's not obvious that would be kind of embarrassing if
this did not do better even once you torture it a little bit maybe you can read the weight net paper and try to
figure out how some of these layers work and Implement them yourselves using what we have and of course you can always tune some
of the initialization or some of the optimization and see if you can improve it that way so I'd be curious if people
can come up with some ways to beat this and yeah that's it for now bye