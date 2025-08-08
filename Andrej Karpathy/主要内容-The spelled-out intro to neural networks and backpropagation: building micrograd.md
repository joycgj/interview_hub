It turns out that neural networks are just mathematical expressions just like this one but actually slightly bit less crazy even, neural networks are just a mathematical expression, they take the input data as an input and they take the weights of a neural network as an input and it's a mathematical expression, and the output are your predictions of your neural networks or the loss function, we'll see this in a bit, but basically neural networks just happen to be a certain class of mathematical expressions, but backpropagation is actually significantly more general, it doesn't actually care about neural networks at all, it only tells us about arbitrary mathematical expressions and then we happen to use that machinery for training of neural networks.

### ❓这些表达式是神经网络吗？

不是，这个例子只是一个人为构造的数学表达式，**用来展示 micrograd 支持的操作**，比如加减乘除、幂运算等。

但很重要的一点是：

> **神经网络本质上也是数学表达式，只是结构更规律一点。**

神经网络的输入是数据和权重，输出是预测或损失函数，而这中间只是由一堆数学操作构成的表达式图。因此：

> 反向传播其实**和神经网络无关**，它只是用来处理数学表达式的求导。


深度学习入门

由全部变量的偏导数汇总而成的向量称为梯度（gradient）。

### **核心思想**

1. **梯度的作用**：

   * 梯度表示了损失函数对某个参数的敏感度或变化率。具体来说，梯度指示了如果我们改变某个参数，损失函数会朝哪个方向变化，变化的速率有多大。
   * 我们希望通过调整输入（或网络中的参数），使得损失函数的值下降（更接近最小值）。因此，我们需要沿着梯度的方向调整输入值，朝着使损失函数减少的方向“走”。

2. **优化过程**：

   * 在这个示例中，我们通过“微调”输入值 `a`、`b`、`c` 和 `f` 来让损失函数 `l` 增加（即变得更少负）。调整的步骤是沿着梯度的方向进行的，称为 **梯度上升**，即如果我们想让 `l` 增加，我们就应该朝着梯度的方向调整输入。

   * 对于每个输入节点（`a`、`b`、`c` 和 `f`），我们根据它们的梯度值来调整它们的值。调整后的目标是使损失 `l` 增加（变得更不负），而不是下降。

   * 这就是一个简单的“优化步骤”，也就是一次 **梯度上升**，使得损失值从更负的数值（如 -8）变得稍微不那么负（如 -7）。

3. **预测的结果**：

   * 通过这个步骤，我们可以预期，经过一小步调整后，损失 `l` 会变得更加接近于目标值（例如 -6），因为我们沿着梯度方向对 `a`、`b`、`c` 和 `f` 进行了调整。
   * 然而，在实际执行时，经过一次优化步骤后，损失并未完全达到预期的结果，而是略微减小（变为 -7），这表明优化是一个逐步的过程，需要多个步骤才能真正达到预期目标。

### **总结**

* 通过对输入值进行微小的调整，并沿着梯度的方向进行更新，我们能够影响最终的损失函数。这个过程展示了梯度下降（或梯度上升）优化方法的一个步骤。虽然经过一次优化后，损失函数并没有完全达到目标值，但它的值已经变得稍微更“好”了。这是训练神经网络过程中逐步优化模型的一个典型示例。

