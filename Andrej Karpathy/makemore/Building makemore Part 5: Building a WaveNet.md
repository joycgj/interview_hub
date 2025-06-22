We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between jupyter notebooks and repository code, ...).

Links:
makemore on github: https://github.com/karpathy/makemore
jupyter notebook I built in this video: https://github.com/karpathy/nn-zero-t...
collab notebook: https://colab.research.google.com/dri...
my website: https://karpathy.ai
my twitter:   / karpathy  
our Discord channel:   / discord  

Supplementary links:
WaveNet 2016 from DeepMind https://arxiv.org/abs/1609.03499
Bengio et al. 2003 MLP LM https://www.jmlr.org/papers/volume3/b... 

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

---

如果你需要，我还可以帮你总结成更简单的中文，或者重点解释 "因果扩张卷积"、"WaveNet" 这些概念～要不要继续？ 🌟
