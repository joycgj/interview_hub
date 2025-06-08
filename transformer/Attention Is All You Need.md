# Attention Is All You Need

- Ashish Vaswani<sup>∗</sup> Google Brain avaswani@google.com
- Noam Shazeer<sup>∗</sup> Google Brain noam@google.com
- Niki Parmar<sup>∗</sup> Google Research nikip@google.com
- Jakob Uszkoreit<sup>∗</sup> Google Research usz@google.com
- Llion Jones<sup>∗</sup> Google Research llion@google.com
- Aidan N. Gomez<sup>∗ †</sup> University of Toronto aidan@cs.toronto.edu
- Łukasz Kaiser<sup>∗</sup> Google Brain lukaszkaiser@google.com
- Illia Polosukhin<sup>∗ ‡</sup> illia.polosukhin@gmail.com

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

> <sup>∗</sup>Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.
†Work performed while at Google Brain. ‡Work performed while at Google Research.
>
> 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

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

## 1 Introduction

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.
Attention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.


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