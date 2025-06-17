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
