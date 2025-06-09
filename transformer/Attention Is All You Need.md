# Attention Is All You Need

## 摘要

主流的序列转换模型主要基于复杂的循环神经网络（RNN）或卷积神经网络（CNN），这些模型通常包括一个编码器和一个解码器。性能最好的模型还通过注意力机制将编码器和解码器连接起来。我们提出了一种全新的、基于注意力机制的简单网络架构——Transformer，它完全摒弃了循环和卷积结构。

在两个机器翻译任务上的实验表明，该模型在翻译质量上更优，同时具有更强的并行能力，训练所需时间也大大减少。在WMT 2014英德翻译任务中，我们的模型取得了28.4的BLEU分数，比现有最好的结果（包括模型集成）提高了超过2个BLEU分。在WMT 2014英法翻译任务中，我们的模型在8块GPU上训练3.5天后，达到了41.8的单模型BLEU分数，创下了新的单模型最佳成绩，而训练成本只是文献中最优模型的一小部分。

我们还展示了Transformer在其他任务上的良好泛化能力，它在处理英语成分句法分析任务时也表现出色，无论是大规模训练数据还是数据有限的情况下都取得了成功。

---

<sup>*</sup>共同贡献，作者排序随机。Jakob 提出了用自注意力机制替代RNN的想法，并率先开展了对这一想法的验证工作。Ashish 与 Illia 一起设计并实现了最初的Transformer模型，并在本项工作的各个方面都发挥了关键作用。Noam 提出了缩放点积注意力机制、多头注意力机制以及无需参数的位置表示方法，是另一位几乎参与所有细节工作的作者。Niki 在我们的初始代码库和 tensor2tensor 中设计、实现、调试并评估了无数模型变体。Llion 也尝试了新颖的模型变体，负责最初的代码库开发、高效的推理实现以及可视化工作。Lukasz 和 Aidan 则花费了无数个日夜设计并实现了 tensor2tensor，替代了我们早期的代码库，极大地提升了实验结果并显著加速了我们的研究进展。

† 工作完成时供职于 Google Brain。

‡ 工作完成时供职于 Google Research。

发表于第31届神经信息处理系统大会（NIPS 2017），地点：美国加利福尼亚州长滩。

---

## 1 引言

这是《Attention Is All You Need》论文的第一部分，它主要讲了几个核心点：

---

### ✅ **背景是什么？**

传统上，处理语言（比如翻译、语言建模）的方法主要是用循环神经网络（RNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）。这些方法已经被广泛认为是效果最好的解决方案。很多研究也在不断改进这些模型。

---

### 🤔 **这些旧方法有什么问题？**

RNN类的模型有一个本质的问题：**它们是一步一步处理的**。也就是说，它只能先处理第一个词，再处理第二个词，然后第三个……所以整个训练过程不能并行，速度慢，特别是在句子很长的时候，更难搞，因为内存不够用，不能批量处理太多句子。

虽然有一些新技术（比如把模型分段计算或只在某些情况下激活部分模型）可以提高效率和效果，但**必须按顺序处理**这个限制还是没有解决。

---

### 🧠 **注意力机制（Attention）是个好东西**

注意力机制可以让模型直接关注输入中的关键部分，不管它们离得远不远。比如翻译时，句子开头的词可能跟句子结尾的词有关，注意力机制能捕捉到这种远距离的关系。

但之前大部分模型都是把注意力和RNN一起用的，还是受限于RNN的“顺序处理”。

---

### 💡 **我们做了啥创新？**

这篇论文提出了一个新模型叫 **Transformer**。它**完全不用RNN，也不用卷积神经网络**，**只靠注意力机制**！

好处是：

* 它可以并行处理所有词，更快；
* 效果还更好；
* 我们用8块P100显卡训练12小时，就在翻译任务上达到了最先进的水平。

---

### 总结一句话：

> 以前翻译模型都靠一步步来的RNN，我们这次干脆不用RNN了，直接用注意力机制来做，还做得更快更好！

---

当然可以！下面我来**逐句**用通俗易懂的语言帮你解释这段话：

---

### 原文：

> **Recurrent models typically factor computation along the symbol positions of the input and output sequences.**

✅ **解释：**
循环神经网络（RNN）这种模型在处理输入和输出的时候，通常是**一个字（或词）一个字地处理**。也就是说，它是**按顺序来处理句子中每一个位置的内容**。

---

“**factor computation**” 是一个在机器学习和数学中常见的术语，字面意思是“将计算进行因式分解”或“分步分解处理”。在这段论文中的语境下，它指的是：**将一个完整的计算任务拆解成若干个更小的、与输入位置相关的步骤来处理。**

我们来**详细拆解一下**这个术语在这里的含义：

---

### 📘 原句：

> "Recurrent models typically factor computation along the symbol positions of the input and output sequences."

### 🧠 翻译直白点是：

> 循环模型通常会**按照输入输出序列中每个符号（比如每个词或字符）的位置，把整个计算任务拆开来做**。

---

### ✅ 通俗解释：

我们以一句话翻译为例，比如英语句子 “I love cats.”：

循环神经网络会像这样做：

1. 第一步处理 “I” → 得到隐藏状态 $h_1$
2. 第二步处理 “love”，用 “love” + 上一步结果 $h_1$ → 得到 $h_2$
3. 第三步处理 “cats”，用 “cats” + $h_2$ → 得到 $h_3$

整个模型不是一下子处理完整句话，而是**把整个计算流程“拆开”（factor）成了每个词一步的计算**。这就是“factor computation along the symbol positions”：

👉 **每个词一个步骤，整个过程就是多个步骤的组合。**

---

### 📌 更技术一点的解释：

> "factor computation" 在深度学习中有时也意味着：**将一个复杂的函数拆分为多个子函数**，比如先做嵌入，再做编码，再做解码，每一层处理一个方面。
> 在RNN中，这种“因式分解”是**沿着时间维度（序列位置）来分解**的——每个时间点/序列位置独立完成一小步。

---

### 🔍 类比举个例子：

就好比你在做一道数学题，传统方法是一步到尾一口气做完。而RNN的思路是：

> 把这道题拆成 3 个步骤，每步都只算当前这一步，必须按顺序完成。
> 这就是 “factor computation along time steps”。

---

### ✍️ 总结一下：

* **factor computation** = 把整体计算任务拆成多个小步骤来做；
* 在 RNN 里，是 **按照词的位置/顺序** 来分步骤处理；
* 好处是结构清晰，但坏处是**不能并行处理**，速度慢。

---

### 原文：

> **Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$.**

✅ **解释：**
这个模型在每一步都会生成一个“隐藏状态”（可以理解为当前处理结果的中间记忆），叫做 $h_t$。这个状态是根据两个东西算出来的：

1. 上一步的隐藏状态 $h_{t-1}$
2. 当前这一步输入的词（比如第t个词）

简单说就是：**当前的处理结果，取决于你上一步的结果加上这一步的输入**。

---

详细解释一下 Aligning the positions to steps in computation time

非常好，这句看起来简单，其实挺关键的，尤其是在理解 RNN 的工作机制时。

---

### 📘 原句：

> **Aligning the positions to steps in computation time**

---

## ✅ 一句话解释：

这句话的意思是：

> **“把每个词（或字符）在序列中的位置，对应到一个具体的计算步骤。”**

也就是说，**模型处理第1个词就是第1步计算，处理第2个词就是第2步计算……每个位置都和计算时间一一对应**。

---

## 🔍 逐层拆解：

### 1. **position（位置）指的是啥？**

就是输入序列中每个词的顺序，比如这个句子：

```
I     love     cats
1      2        3
```

这些数字就是“位置”。

---

### 2. **computation time（计算时间）指的是啥？**

RNN 是**一步一步处理**的，每个时间步（timestep）模型都会执行一次计算。比如：

* 第1步 → 处理第1个词 “I”
* 第2步 → 处理 “love”
* 第3步 → 处理 “cats”

这个处理过程是有**顺序的时间步（time step）**的。

---

### 3. **Aligning… 是什么动作？**

“Aligning” 表示**对齐、对应、配对**。

这里的意思是：

> 把输入中的“第几个词”对齐（对应）到“第几步计算”。

所以：

| 输入位置（Position） | 计算时间（Time step） |
| -------------- | --------------- |
| 1（“I”）         | t = 1           |
| 2（“love”）      | t = 2           |
| 3（“cats”）      | t = 3           |

这就是“Aligning the positions to steps in computation time”。

---

## 🧠 更直白的说法：

RNN 的计算逻辑是：**一个词对应一步计算，不能跳着来，也不能一起算。**

就像你在数数，必须先数1，再数2，最后数3——这个过程不能并行。

---

## 📌 总结一下：

| 原词                           | 通俗解释         |
| ---------------------------- | ------------ |
| Aligning the positions       | 把每个词的位置      |
| to steps in computation time | 和模型的每一步计算对上号 |

➡️ 也就是：
**“模型第几步就处理第几个词，每个词和一个时间步绑定在一起，按顺序来。”**

---

![](../pictures/attention_1.png "")

这张图就是对“**Aligning the positions to steps in computation time**”的直观展示：

* 上面蓝色的是输入序列中每个词的位置（Position 1, 2, 3）。
* 下面绿色的是计算时间的步骤（t=1, t=2, t=3）。
* 每个箭头表示：第几个词对应第几步计算。

🔁 就是说：**模型第1步就处理第1个词，第2步处理第2个词……必须按顺序来，不能跳过，也不能一起算。**

这就是RNN的“顺序性”——它不能并行，只能一步一步来。这也正是Transformer想要突破的限制。需要我对比一下Transformer是怎么做的吗？

需要

![](../pictures/attention_1.png "")

这张图展示了 **Transformer 和 RNN 的核心区别**：

* 🔵 上方：每个词的位置（Position 1, 2, 3）
* 🟢 下方绿色区域：**Transformer 是“同时处理所有词”**，也就是所谓的“并行计算（Parallel Computation）”
* 每个箭头都指向同一个计算区域，表示这些词**可以一起被处理**

---

### 📌 总结对比：

| 模型          | 如何处理词序列        |
| ----------- | -------------- |
| RNN         | 一步一步来（不能并行）    |
| Transformer | 一次性处理所有词（可以并行） |

这正是 Transformer 更快、更高效的原因之一。
如果你还想了解“Transformer 是怎么做到并行”的，比如多头注意力或位置编码，我也可以继续讲解。

---

这里面的function是什么意思，详细解释一下

---

这个句子里提到的 **“as a function of...”** 是一个非常常见的数学表达方式，我们来一步一步解析。

---

## 📘 原句再读一遍：

> **“...they generate a sequence of hidden states $h_t$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$.”**

---

## ✅ 这句话大意是：

模型会生成一系列隐藏状态 $h_t$，
每一个 $h_t$ 是一个“函数”——它**依赖于**：

1. 上一步的隐藏状态 $h_{t-1}$
2. 当前时间步的输入 $x_t$

---

## 🔍 什么是 “function of”？

### 👉 通俗解释：

“Function of” 可以理解为：

> **某个东西是由其他因素通过某种“计算关系”产生的。**

举个生活中类比：

* 咖啡的味道是牛奶、咖啡粉、水、糖的 function（函数） → 味道是它们混合后的结果。
* 天气温度是阳光强度、风速、湿度的 function → 温度取决于这些因素。

---

### 👉 在神经网络里：

这个“函数”其实是一个计算过程，比如：

$$
h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)
$$

这就是：

* 当前隐藏状态 $h_t$
* 是由上一步状态 $h_{t-1}$ 和 当前输入 $x_t$
* **通过矩阵计算和激活函数组合出来的结果**

---

### ✅ 更口语化翻译原句：

“模型会依次生成隐藏状态 $h_t$，每一个都是根据上一个状态和当前输入计算出来的结果。”

---

### ✅ 更技术一点的说法：

“每一个隐藏状态 $h_t$，是一个函数：

$$
h_t = f(h_{t-1}, x_t)
$$

其中 $f$ 是神经网络内部定义好的计算公式（比如前馈层+激活函数）。”

---

## 📌 总结：

| 术语                  | 意思                                              |
| ------------------- | ----------------------------------------------- |
| Function of A and B | 由 A 和 B 经过某种规则/公式/运算 得出的结果                      |
| In this context     | 当前隐藏状态 $h_t$ 是前一个状态 $h_{t-1}$ 和当前输入 $x_t$ 的计算结果 |
| 实质上就是               | 神经网络的一个“计算步骤”                                   |

---

如果你想，我可以画出这个计算流程图（RNN的基本结构图）来帮助你更形象地理解这个“function”的含义，要不要？


---

### 原文：

> **This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.**

✅ **解释：**
因为这个模型必须**一步一步处理**（比如必须先处理第1个词，才能处理第2个词），所以它**不能并行处理同一个句子里的所有词**。
这在句子很长的时候就成了大问题，因为：

* 并行不了 → 慢；
* 每次只能处理有限数量的句子 → **内存不够，批处理数量受限**。

---

详细解释一下 sequential nature

---

当然可以！我们来详细剖析这句话里的 **“sequential nature”** 是什么，以及它为什么会成为问题。

---

## 📘 原句：

> **This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.**

---

## ✅ 什么是 "sequential nature"?

### ✅ 原词解释：

* **Sequential** = 按顺序，一个接一个
* **Nature** = 本质、特性

➡️ 所以 **sequential nature** 指的是：

> **模型天生就必须“按顺序”一步步来处理数据的特性。**

---

## 🔍 在RNN中，为什么是 sequential？

RNN 的基本工作方式是这样的：

$$
h_t = f(h_{t-1}, x_t)
$$

也就是说：

* 当前这一步的隐藏状态 $h_t$，**必须等上一步 $h_{t-1}$ 处理完之后才能计算出来**
* 所以你没法同时算第1步、第2步、第3步 → **只能一步一步来**

👉 这就是 RNN 的“顺序性本质”（**inherently sequential nature**）

---

## 💡 举个例子：

想象你在读一句话：“我 喜欢 猫。”

RNN 的思路是这样：

1. 先处理“我” → 得到状态 h1
2. 用 h1 和“喜欢” → 得到 h2
3. 用 h2 和“猫” → 得到 h3

你不能直接算 h3，必须等前面的都算完。这就是 **不能并行，只能按顺序来**。

---

## 🚫 为什么这个“顺序性”很糟糕？

### 原文说的是：

> **It precludes parallelization within training examples**
> 意思是：

* **在一个训练样本内部，没法并行处理**（因为你必须等前一步才能算下一步）

---

## 😓 那为什么“critical at longer sequence lengths”？

随着句子越来越长，比如：

> “在阳光明媚的午后，我一个人坐在图书馆的角落里阅读……”

* 步数变多（每个词都要单独处理一步）
* 总体计算时间变长
* **不能加快速度，因为不能并行**

同时，内存也吃不消，不能同时处理太多句子。

---

## 📌 总结一句话：

| 术语                    | 通俗解释                       |
| --------------------- | -------------------------- |
| **Sequential nature** | 模型必须一步一步、按顺序处理每个词          |
| **为什么是问题？**           | 没法同时处理多个词，训练慢，尤其是句子很长时     |
| **Transformer的突破？**   | 彻底打破这个顺序性 → 一次性处理整个句子（并行！） |

---

如果你想，我也可以画一张图来对比 RNN 的“顺序处理” 和 Transformer 的“并行处理”流程图，要看看吗？

---

across examples中的across是什么意思

---

很好，这是个非常常见但容易搞混的词。

---

## 📘 句子原文：

> **"...as memory constraints limit batching across examples."**

我们来重点解释这里的 **“across examples”**。

---

## ✅ across 的基本意思：

“**across**” 是一个介词，基本意思是：

> “横跨”、“跨越”、“覆盖多个”

所以 “across examples” 就是：

> **“在多个训练样本之间”**，或者说 **“跨样本”**

---

### 📍 举个例子：

比如你训练一个模型时，会一次性把很多句子打包成一个“batch”（批次），这样计算更高效。这个打包过程就叫 batching。

* 如果你每次训练处理10句话，就是在 **batching across 10 examples**。
* 如果你内存不够，只能装下3句话，那你只能 batching across 3 examples。

---

## ✅ 回到原句意思：

> **"as memory constraints limit batching across examples."**

翻译成大白话就是：

> “由于内存有限，不能一次性处理太多句子。”

也就是：

* 一边是：**模型内部（句子里）不能并行（因为顺序性）**
* 另一边是：**内存又限制了我们在多个句子之间的并行处理（across examples）**

➡️ **双重受限，训练效率低。**

---

## 📌 总结：

| 表达                       | 意思                  |
| ------------------------ | ------------------- |
| across examples          | 在多个训练样本之间           |
| batching across examples | 把多个训练样本（句子）一起打包处理   |
| 限制                       | 内存不足，就不能在多个样本之间并行训练 |

---

如果你还想知道“batching”是怎么操作的，也可以继续问，我可以画个流程图帮你直观理解。


---

### 原文：

> **Recent work has achieved significant improvements in computational efficiency through factorization tricks \[21] and conditional computation \[32], while also improving model performance in case of the latter.**

✅ **解释：**
最近有些研究尝试用一些“技巧”来让计算变快，比如：

* **分解计算**：把大的问题拆成多个小块来处理；
* **条件计算**：模型只激活一部分，而不是全部都跑一遍。

这些方法在提高运行效率的同时，有时还能让模型表现更好。

---

当然可以！我们来详细解释一下 **“in case of the latter”** 这个短语的意思和用法。

---

## 📘 原句（上下文）是这样的：

> "Recent work has achieved significant improvements in computational efficiency through factorization tricks \[21] and conditional computation \[32], **while also improving model performance in case of the latter.**"

---

## ✅ 分析结构

这个句子里有两个关键点（并列的）：

1. **factorization tricks**（分解技巧）
2. **conditional computation**（条件计算）

这两个是“最近提高效率的做法”。

---

详细解释一下 in case of the latter

---

## ✅ 什么是 "the latter"？

“**the latter**” 是个代词，意思是：

> **“后者”**，即上面列出的两个东西中的第二个。

👆 所以这里的 "the latter" 就是：

> **conditional computation**

---

## ✅ 什么是 "in case of..."？

这个短语可以理解为：

* **“在……的情况下”**
* **“就……而言”**

所以 **"in case of the latter"** 其实等于：

> **“在条件计算（conditional computation）的情况下”**
> **“就后者而言”**

---

## ✅ 整句话大白话翻译：

> “最近的一些研究通过分解技巧和条件计算显著提升了计算效率，
> 而**条件计算**在提升效率的同时，还能提升模型性能。”

---

## 📌 总结：

| 短语                    | 含义                                         |
| --------------------- | ------------------------------------------ |
| the latter            | 后者，这里指 conditional computation             |
| in case of the latter | 在后者（conditional computation）的情况下，也就是“说到后者” |

---

如果你还不确定“former”和“latter”这种结构的用法，我也可以给你列几个小例子来对比理解。需要吗？


---

### 原文：

> **The fundamental constraint of sequential computation, however, remains.**

✅ **解释：**
但**最大的问题还在**：这种“必须一步步来”的方式，还是没法解决。

---

### 总结一句话：

循环模型是“按顺序”一个一个词地处理的，这种处理方式限制了并行计算，尤其在长句子里效率低。虽然有些技巧能缓解这个问题，但**本质上的顺序限制还是存在**。

---

Attention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

逐句帮我解释清楚，需要用浅显易懂的话来解释。

---

当然可以！我们来一句一句地用**大白话**解释这段内容：

---

## 📘 原句 1：

> **"Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks..."**

✅ **通俗解释：**

注意力机制已经变成了各种“厉害的”序列建模和转换模型中的**重要组成部分**，几乎是“标配”了。

* “Attention mechanism”：注意力机制，让模型自动关注重要的信息；
* “Integral part”：不可或缺的一部分；
* “Sequence modeling and transduction”：处理序列（像语言、音频、翻译）；
* “Compelling”：效果很好的、有说服力的。

🎯 换句话说：

> 现在只要是处理语言等序列问题的好模型，基本上都会用上“注意力机制”。

---

详细解释一下 sequence modeling and transduction models

好问题！我们来详细解释这两个术语：

---

## ✅ 1. **Sequence Modeling（序列建模）**

### 📘 定义：

**Sequence modeling** 就是：

> **理解、预测或生成“序列”数据的模型。**

### 🔢 什么是“序列”？

序列就是有顺序的一串东西，比如：

* 一句话（词是有顺序的）👉 自然语言处理
* 一段语音（声音帧是按时间排的）👉 语音识别
* 股票价格（一天一天的数据）👉 时间序列分析

### 🎯 目的：

* 预测下一个词（语言模型）
* 理解整段话的意思（情感分析）
* 找出关键词（命名实体识别）

### 🧠 举例：

* RNN（循环神经网络）
* LSTM（长短期记忆网络）
* Transformer

---

## ✅ 2. **Transduction Models（转换模型 / 转导模型）**

### 📘 定义：

**Transduction** 是一个专业术语，意思是：

> **把一个序列“转换”成另一个序列。**

这种模型不仅要理解输入序列，还要**生成输出序列**。

### 🔁 举例任务：

| 输入序列 | 输出序列 | 任务名称   |
| ---- | ---- | ------ |
| 英文句子 | 中文句子 | 机器翻译   |
| 音频信号 | 英文文字 | 语音识别   |
| 问题   | 回答   | 问答系统   |
| 图像描述 | 文字   | 图像字幕生成 |

### 🧠 举例模型：

* 编码器-解码器（Encoder-Decoder）架构
* Seq2Seq 模型（sequence-to-sequence）
* Transformer（也是一种转导模型）

---

## 🔍 所以这两个的区别是：

| 类型                 | 核心任务         | 举个例子      |
| ------------------ | ------------ | --------- |
| Sequence Modeling  | 理解或预测一个序列    | 预测下一个词    |
| Transduction Model | 把一个序列变成另一个序列 | 英译中、语音转文字 |

---

## 📌 总结一句话：

> * **Sequence modeling** 是理解/建模一个序列；
> * **Transduction models** 是把一个序列变成另一个序列（比如翻译）。

---

如果你想，我也可以画张图，把“sequence modeling”和“transduction”在流程图上展示出来，直观对比。需要吗？


---

## 📘 原句 2：

> **"...allowing modeling of dependencies without regard to their distance in the input or output sequences \[2, 19]."**

✅ **通俗解释：**

注意力机制的厉害之处在于：

> 它可以让模型学习**词和词之间的关系（依赖）**，**不管它们在句子里离得有多远**！

比如：

* “今天的天气很冷，我**穿了厚外套**。” → “冷”和“外套”在句子中可能隔很远，但它们有关系，注意力机制能处理到这一点。

---

## 📘 原句 3：

> **"In all but a few cases \[27], however, such attention mechanisms are used in conjunction with a recurrent network."**

✅ **通俗解释：**

不过，要注意一点：

> **大多数情况下**，注意力机制是和 **循环神经网络（RNN）搭配使用**的，
> 只有极少数例外（比如第\[27]篇论文）。

📍 换句话说：

* 之前的模型：**RNN 是主力，注意力只是辅助**；
* Transformer 想做的就是：**把 RNN 完全换掉，只用注意力！**

---

## 📌 总结整段：

1. 注意力机制现在已经成了序列模型中非常重要的一部分；
2. 它可以帮助模型理解**任意距离的词之间的关系**；
3. 但几乎所有模型还是把它跟**RNN 结合着用**，很少单独使用注意力机制。

🎯 这就引出了 Transformer 的动机：我们能不能**只用注意力机制也能做好这些事**？答案就是——**Yes！Attention is all you need!**

需要我继续讲下一段吗？或者要不要我画个“远距离依赖 + 注意力”的小图？





## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.
We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.
The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

**Acknowledgements** We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

## 7 结论

在本研究中，我们提出了Transformer，这是第一个完全基于注意力机制的序列转换模型，它用多头自注意力机制取代了编码器-解码器架构中常见的循环层。

在翻译任务中，与基于循环或卷积层的架构相比，Transformer的训练速度显著更快。在WMT 2014英德翻译任务和WMT 2014英法翻译任务中，我们都取得了新的最先进成绩。在英德任务中，我们的最佳模型甚至超过了此前报道的所有模型集成结果。

我们对基于注意力机制的模型未来发展感到非常兴奋，并计划将其应用于更多任务。我们计划将Transformer扩展到处理输入和输出模态不限于文本的问题，并研究局部的、受限的注意力机制，以高效处理如图像、音频和视频等大规模输入输出数据。让生成过程更少依赖顺序性也是我们的研究目标之一。

我们用于训练和评估模型的代码已公开，地址为：https://github.com/tensorflow/tensor2tensor。

**致谢** 我们感谢Nal Kalchbrenner和Stephan Gouws提出的宝贵意见、修正建议以及给予我们的启发。