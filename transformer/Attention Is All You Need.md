# Attention Is All You Need

## 摘要

主流的序列转换模型主要基于复杂的循环神经网络（RNN）或卷积神经网络（CNN），这些模型通常包括一个编码器和一个解码器。性能最好的模型还通过注意力机制将编码器和解码器连接起来。我们提出了一种全新的、基于注意力机制的简单网络架构——Transformer，它完全摒弃了循环和卷积结构。

在两个机器翻译任务上的实验表明，该模型在翻译质量上更优，同时具有更强的并行能力，训练所需时间也大大减少。在WMT 2014英德翻译任务中，我们的模型取得了28.4的BLEU分数，比现有最好的结果（包括模型集成）提高了超过2个BLEU分。在WMT 2014英法翻译任务中，我们的模型在8块GPU上训练3.5天后，达到了41.8的单模型BLEU分数，创下了新的单模型最佳成绩，而训练成本只是文献中最优模型的一小部分。

我们还展示了Transformer在其他任务上的良好泛化能力，它在处理英语成分句法分析任务时也表现出色，无论是大规模训练数据还是数据有限的情况下都取得了成功。

> <sup>*</sup>共同贡献，作者排序随机。Jakob 提出了用自注意力机制替代RNN的想法，并率先开展了对这一想法的验证工作。Ashish 与 Illia 一起设计并实现了最初的Transformer模型，并在本项工作的各个方面都发挥了关键作用。Noam 提出了缩放点积注意力机制、多头注意力机制以及无需参数的位置表示方法，是另一位几乎参与所有细节工作的作者。Niki 在我们的初始代码库和 tensor2tensor 中设计、实现、调试并评估了无数模型变体。Llion 也尝试了新颖的模型变体，负责最初的代码库开发、高效的推理实现以及可视化工作。Lukasz 和 Aidan 则花费了无数个日夜设计并实现了 tensor2tensor，替代了我们早期的代码库，极大地提升了实验结果并显著加速了我们的研究进展。
> 
> † 工作完成时供职于 Google Brain。
> 
> ‡ 工作完成时供职于 Google Research。
> 
> 发表于第31届神经信息处理系统大会（NIPS 2017），地点：美国加利福尼亚州长滩。

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