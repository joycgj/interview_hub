A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

当然，下面是这段英文的翻译，尽量保持原意清晰，同时通俗易懂：

---

一个**微型的自动求导引擎**（虽然小，但很强！🙂）。
它实现了 **反向传播（反向模式自动微分）**，基于动态构建的 **有向无环图（DAG）**，并在其基础上提供了一个小型的神经网络库，API 风格类似 PyTorch。

这两个部分都非常精简：

* 自动求导引擎大约只有 **100 行代码**；
* 神经网络库大约 **50 行代码**。

这个 DAG（计算图）只支持 **标量数值（单个数字）** 操作，比如我们会把每个神经元的运算拆解成多个**加法和乘法**的微小操作。
尽管如此，这已经足够用来构建完整的 **深度神经网络**，并实现 **二分类任务**，正如演示 notebook 中所展示的那样。

这个项目可能非常适合用作 **教学用途**。

---

如果你还想我帮你进一步分段讲解这段话的技术原理，比如 “反向模式自动微分” 或 “为什么只支持标量也能训练神经网络”，我也可以继续详细解释。
