# intro
hello my name is andre and i've been training deep neural networks for a bit more than a decade and in this lecture i'd like to show you
what neural network training looks like under the hood so in particular we are going to start with a blank jupiter
notebook and by the end of this lecture we will define and train in neural net and you'll get to see everything that
goes on under the hood and exactly sort of how that works on an intuitive level now specifically what i would like to do
is i would like to take you through building of micrograd

你好，我叫 Andre，我从事深度神经网络训练已经十多年了。在这节课中，我想向你展示**神经网络训练背后到底发生了什么**。

我们将从一个空白的 Jupyter Notebook 开始，到课程结束时，我们会定义并训练一个神经网络，**你将看到背后的一切细节，以及它到底是如何在直观层面上运行的**。

具体来说，我想做的事情是：**带你一步步构建 micrograd**。


# micrograd overview
now micrograd is this library that i released on github about two years ago but at the time i only uploaded the source code and you'd
have to go in by yourself and really figure out how it works so in this lecture i will take you
through it step by step and kind of comment on all the pieces of it so what is micrograd and why is it interesting
good um micrograd is basically an autograd engine autograd is short for automatic
gradient and really what it does is it implements backpropagation now backpropagation is this algorithm that
allows you to efficiently evaluate the gradient of some kind of a loss function with
respect to the weights of a neural network and what that allows us to do then is we can iteratively tune the
weights of that neural network to minimize the loss function and therefore improve the accuracy of the network so
back propagation would be at the mathematical core of any modern deep neural network library like say pytorch
or jaxx so the functionality of microgrant is i think best illustrated by an example so
if we just scroll down here you'll see that micrograph basically allows you to build out mathematical
expressions and um here what we are doing is we have an expression that we're building out where you have two inputs a and b
and you'll see that a and b are negative four and two but we are wrapping those
values into this value object that we are going to build out as part of micrograd so this value object will wrap the
numbers themselves and then we are going to build out a mathematical expression here where a and b are transformed into c d and
eventually e f and g and i'm showing some of the functions some of the functionality of micrograph
and the operations that it supports so you can add two value objects you can multiply them you can raise them to a
constant power you can offset by one negate squash at zero
square divide by constant divide by it etc and so we're building out an expression
graph with with these two inputs a and b and we're creating an output value of g
and micrograd will in the background build out this entire mathematical expression so it will for example know
that c is also a value c was a result of an addition operation and the
child nodes of c are a and b because the and will maintain pointers to a and b
value objects so we'll basically know exactly how all of this is laid out and then not only can we do what we call
the forward pass where we actually look at the value of g of course that's pretty straightforward we will access
that using the dot data attribute and so the output of the forward pass the value
of g is 24.7 it turns out but the big deal is that we can also take this g
value object and we can call that backward and this will basically uh initialize back propagation at the node g
and what backpropagation is going to do is it's going to start at g and it's going to go backwards through that
expression graph and it's going to recursively apply the chain rule from calculus 

micrograd 是我大约两年前在 GitHub 上发布的一个库，但当时我只是上传了源代码，
你需要自己进去阅读代码，弄清楚它是如何运作的。

所以这节课里，我会一步步带你走过它的实现过程，并对每一部分都进行讲解。

那么，micrograd 到底是什么？为什么它有趣呢？

很好，micrograd 本质上是一个 autograd 引擎。
“autograd” 是 “automatic gradient” 的缩写，意思是“自动求梯度”。

它实现了的正是 反向传播算法（backpropagation）。

反向传播是一种算法，它能高效地计算损失函数（loss function）关于神经网络权重的梯度。
这样一来，我们就能迭代地调整神经网络的权重，以最小化损失函数，从而提高网络的准确率。

因此，反向传播几乎是所有现代深度学习框架（比如 PyTorch 或 JAX）背后的数学核心。

至于 micrograd 的功能，我认为通过一个例子可以更好地说明。

我们往下看：
micrograd 允许你构建数学表达式。

在这个例子里，我们构造了一个表达式，它有两个输入变量 a 和 b，
它们的值是 -4 和 2，但我们把它们封装成了 Value 对象 —— 这是 micrograd 的核心类。

这个 Value 类不只是保存一个数字，它还记录了这个值是如何通过计算得来的。

然后我们用 a 和 b 构建出中间变量 c、d，最后得到输出 g。
在这个过程中，我们展示了 micrograd 支持的各种操作，比如：

两个 Value 对象可以相加、相乘、

可以进行幂运算、加常数、取负号、取最大值为 0（ReLU）、

平方、除以常数、被常数除等等。

所以我们是从 a 和 b 出发，构建了一个表达式图，最终得到输出 g。

在后台，micrograd 会自动构建一张完整的计算图。
比如说，它会知道 c 是一个 Value，它是由 a + b 计算得来的，
而 a 和 b 是它的“子节点” —— 这些关系会在对象里通过引用记录下来。

这样我们不仅能进行所谓的“前向传播”（即计算 g 的值），
这个过程很简单：我们通过 .data 属性就能访问 g 的值。
在这个例子中，g 的值是 24.7。

但更重要的是，我们还可以对 g 调用 .backward() 方法！

这会从 g 节点开始，初始化反向传播：
micrograd 会从 g 开始，沿着计算图反向传播，
在每一个节点上递归应用微积分中的链式法则（chain rule）。


and what that allows us to do then is we're going to evaluate basically the derivative of g with respect to all the internal nodes
like e d and c but also with respect to the inputs a and b
and then we can actually query this derivative of g with respect to a for example that's a dot grad in this case
it happens to be 138 and the derivative of g with respect to b which also happens to be here 645
and this derivative we'll see soon is very important information because it's telling us how a and b are affecting g
through this mathematical expression so in particular a dot grad is 138 so if we slightly
nudge a and make it slightly larger 138 is telling us that g will grow and
the slope of that growth is going to be 138 and the slope of growth of b is going to be 645. so that's going to tell us about
how g will respond if a and b get tweaked a tiny amount in a positive direction
okay now you might be confused about what this expression is that we built out
here and this expression by the way is completely meaningless i just made it up i'm just flexing about the kinds of
operations that are supported by micrograd what we actually really care about are neural networks but it turns out that
neural networks are just mathematical expressions just like this one but actually slightly bit less crazy even
neural networks are just a mathematical expression they take the input data as an input and they take the weights of a
neural network as an input and it's a mathematical expression and the output are your predictions of your neural net
or the loss function we'll see this in a bit but basically neural networks just happen to be a certain class of
mathematical expressions but back propagation is actually significantly more general it doesn't
actually care about neural networks at all it only tells us about arbitrary mathematical expressions and then we
happen to use that machinery for training of neural networks now one more note i would like to make at this stage
is that as you see here micrograd is a scalar valued auto grant engine so it's working on the you know level of
individual scalars like negative four and two and we're taking neural nets and we're breaking them down all the way to
these atoms of individual scalars and all the little pluses and times and it's just excessive and so obviously you
would never be doing any of this in production it's really just put down for pedagogical reasons because it allows us
to not have to deal with these n-dimensional tensors that you would use in modern deep neural network library so
this is really done so that you understand and refactor out back propagation and chain rule and
understanding of neurologic training

当然，以下是这段英文内容的中文翻译：

---

这样我们就能够**计算出 `g` 对所有中间变量（如 `e`、`d`、`c`）以及输入变量（如 `a` 和 `b`）的导数**。

然后，我们实际上可以查询这些导数，比如：

* `g` 对 `a` 的导数是 `a.grad`，这个例子中是 138；
* `g` 对 `b` 的导数是 `b.grad`，这个例子中是 645。

---

这些导数其实非常重要，它们告诉我们：

> 在这个数学表达式中，`a` 和 `b` 是如何影响最终输出 `g` 的。

例如：

* `a.grad = 138` 意味着，如果我们稍微增大 `a`，那么 `g` 会以斜率为 **138** 的速度增长；
* 同理，`b.grad = 645` 表示 `b` 对 `g` 的影响更大，增长更快。

这些斜率（梯度）告诉我们：**如果我们让 `a` 和 `b` 稍微变大一点，`g` 会如何响应**。

---

现在你可能会疑惑：我们刚刚构造的这个表达式（涉及 `a`、`b`、`c`、`d` 等）到底是什么意思？

其实，这个表达式**完全是随便写的**，它没有任何实际意义。我只是为了**展示 micrograd 支持哪些操作**。

---

但我们真正关心的是**神经网络**。

而神经网络，其实也就是**像这样的数学表达式** —— 甚至没这么“疯狂”。

神经网络本质上只是：

* 一个函数，它接受**输入数据** 和 **神经网络的权重**，然后
* 通过一堆加法、乘法、非线性函数构成一个**数学表达式**，输出结果，比如预测值或 loss。

---

所以说，神经网络只是某一类数学表达式，而 **反向传播（backpropagation）** 是一种更通用的算法：

> 它不关心你是不是神经网络，它只是针对**任意的数学表达式**计算导数。

我们只是恰好利用这套机制来训练神经网络罢了。

---

在这里我还想补充一点：

你会注意到，**micrograd 是一个“标量级别”的自动求导引擎**，
也就是说，它是在处理每一个小数字（比如 -4 和 2）级别上运行的。

我们把神经网络拆解到最小的单位：每一个小的加法、乘法操作。

这当然在实际项目中**是完全不可行的**，太低效了。

我们这样做完全是为了**教学目的**，这样你就不需要接触现代深度学习框架中那些高维 tensor，
而是可以**真正理解反向传播算法和链式法则的原理**，也更容易理解神经网络训练背后的本质。

---

如果你希望，我可以用图或代码继续说明“g 对 a 的梯度 = 138”是怎么一步步计算出来的，或者展示神经网络如何由这些 Value 对象组成。是否继续？


and then if you actually want to train bigger networks you have to be using
these tensors but none of the math changes this is done purely for efficiency we are basically taking scale
value all the scale values we're packaging them up into tensors which are just arrays of these scalars and then because
we have these large arrays we're making operations on those large arrays that allows us to take advantage of the
parallelism in a computer and all those operations can be done in parallel and then the whole thing runs faster but
really none of the math changes and that's done purely for efficiency so i don't think that it's pedagogically useful to be dealing with tensors from
scratch uh and i think and that's why i fundamentally wrote micrograd because you can understand how things work uh at
the fundamental level and then you can speed it up later okay so here's the fun part my claim is that micrograd is what
you need to train your networks and everything else is just efficiency so you'd think that micrograd would be a
very complex piece of code and that turns out to not be the case so if we just go to micrograd
and you'll see that there's only two files here in micrograd this is the actual engine it doesn't know anything
about neural nuts and this is the entire neural nets library on top of micrograd so engine and nn.pi
so the actual backpropagation autograd engine that gives you the power of neural
networks is literally 100 lines of code of like very simple
python which we'll understand by the end of this lecture and then nn.pi
this neural network library built on top of the autograd engine um is like a joke it's like
we have to define what is a neuron and then we have to define what is the layer of neurons and then we define what is a
multi-layer perceptron which is just a sequence of layers of neurons and so it's just a total joke
so basically there's a lot of power that comes from only 150 lines of code
and that's all you need to understand to understand neural network training and everything else is just efficiency and
of course there's a lot to efficiency but fundamentally that's all that's happening okay so now let's dive right

# derivative of a simple function with one input

in and implement micrograph step by step the first thing i'd like to do is i'd like to make sure that you have a very good understanding intuitively of what a
derivative is and exactly what information it gives you so let's start with some basic imports that i copy
paste in every jupiter notebook always and let's define a function a scalar
valued function f of x as follows so i just make this up randomly i just
want to scale a valid function that takes a single scalar x and returns a single scalar y
and we can call this function of course so we can pass in say 3.0 and get 20 back
now we can also plot this function to get a sense of its shape you can tell from the mathematical expression that this is probably a parabola it's a
quadratic 

当然，以下是这段英文内容的中文翻译：

---

如果你真的想训练更大的神经网络，那你就**必须使用张量（tensors）**，
但其实**数学原理一点都没有变**，**这只是为了效率而做的改进**。

我们所做的事情是：
把一堆标量值（scalars）打包成一个张量（数组），
然后对整个数组执行操作 ——
这些操作可以并行进行，从而**利用计算机的并行性，加快运算速度**。

但本质上，**数学没有任何改变**，
这完全是出于效率的考虑。

---

所以我认为：
**从一开始就去接触高维张量，对学习并没有太大帮助**。
这也是我写 micrograd 的原因 ——
你可以**真正从根本层面理解神经网络的运作**，
等你理解透了，再去优化运行速度也不迟。

---

好了，下面是最有趣的部分：

我想告诉你：

> **训练神经网络，其实只需要 micrograd，其他的东西都是为了效率。**

你可能会以为 micrograd 是一个很复杂的系统，
但其实它**非常简单**。

我们打开 micrograd 这个项目，你会发现它只有两个文件：

* 一个是 `engine.py`：这是核心引擎，它**完全不知道什么是神经网络**；
* 另一个是 `nn.py`：是构建在 `engine.py` 基础上的神经网络库。

---

所以：

* 实现**反向传播和自动求导**的这个核心引擎，**只有大约 100 行 Python 代码**；
* 而整个神经网络库也只是：

  * 定义了“什么是神经元（neuron）”，
  * 然后定义“什么是神经层（layer）”，
  * 最后定义“什么是多层感知机（MLP）”，
  * 也就是多个层组成的网络。

所以这部分代码真的非常简单，**像个笑话一样（a joke）**。
**整个神经网络训练的逻辑，其实只靠 150 行代码就能搞定。**

剩下的，**全都是性能优化的问题**。

当然，**性能优化也是很重要的一部分**，但从本质上说，事情就是这么简单。

---

好了，现在我们真正开始深入学习吧。

我们会从最基础的入手：

> 求一个**只有一个输入变量的简单函数的导数**，并一步步实现 micrograd。

我想先确保你**真正直观地理解“导数”是什么，以及它能告诉我们什么信息**。

---

我们从一些基础的导入语句开始 —— 这些是我在所有 Jupyter Notebook 中都会复制粘贴的内容。

然后我们定义一个**标量函数** $f(x)$，它接收一个输入 `x`，返回一个输出 `y`。
我这个函数是**随便写的**，只是想要一个接收一个数、输出一个数的函数。

我们可以调用它，比如传入 `3.0`，它返回 `20`。

我们也可以把这个函数画出来，观察它的图像形状。
从数学表达式来看，这大概率是一个**抛物线**（也就是一个二次函数）。

---

如果你需要，我也可以用中文 Jupyter Notebook 模拟这段演示，包括导入、绘图和梯度可视化。是否继续？

---

什么是tensor

很好，这个问题非常关键！理解 **什么是 tensor（张量）**，是学习深度学习和 PyTorch 的第一步。

---

## ✅ 一句话定义：

> **Tensor（张量）是一个可以有任意维度的数值数组，是标量、向量、矩阵的推广。**

---

## 📐 举几个简单例子来理解：

| 类型     | 英文名称      | PyTorch 术语 | 示例                                        | 维度（rank） |
| ------ | --------- | ---------- | ----------------------------------------- | -------- |
| 单个数字   | Scalar    | 0D tensor  | `x = 3.14`                                | 0        |
| 一组数字   | Vector    | 1D tensor  | `[1.0, 2.0, 3.0]`                         | 1        |
| 表格（二维） | Matrix    | 2D tensor  | `[[1, 2], [3, 4]]`                        | 2        |
| 多张图片   | 3D tensor | 3D tensor  | `shape = (batch, height, width)`          | 3        |
| 彩色图片集  | 4D tensor | 4D tensor  | `shape = (batch, channel, height, width)` | 4        |

---

## 🔧 在 PyTorch 中：

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x)
print(x.shape)  # 输出：torch.Size([2, 2])
```

这是一个 **二维张量（2D tensor）**，也就是一个 **矩阵**。

---

## 🧠 更深层的理解：

* 张量就是多维数组；
* 它是**数据结构**，同时可以保存梯度信息（通过 `requires_grad=True`）；
* PyTorch 或 TensorFlow 都是基于张量来构建神经网络运算的；
* 张量支持各种数学操作：加法、乘法、转置、矩阵乘、卷积等。

---

## 📌 为什么要用 tensor 而不是只是 array？

1. **张量支持自动求导（autograd）**
2. **可以运行在 GPU 上加速训练**
3. **封装了 shape、dtype、device 等丰富属性**
4. **支持批处理、广播、复杂的运算规则**

---

## 🤖 总结一句话：

> **Tensor = 多维数组 + 自动微分能力 + 高效数学计算支持（可在 GPU 上并行）**

---

如果你想更具体看看张量的可视化（比如 3D 张量结构），或想知道它与 NumPy 的异同，我也可以继续讲解！



and so if we just uh create a set of um
um scale values that we can feed in using for example a range from negative five to five in steps of 0.25
so this is so axis is just from negative 5 to 5 not including 5 in steps of 0.25
and we can actually call this function on this numpy array as well so we get a set of y's if we call f on axis
and these y's are basically also applying a function on every one of
these elements independently and we can plot this using matplotlib so plt.plot x's and y's and we get a nice
parabola so previously here we fed in 3.0 somewhere here and we received 20
back which is here the y coordinate so now i'd like to think through what is the derivative
of this function at any single input point x right so what is the derivative at different points x of this function now
if you remember back to your calculus class you've probably derived derivatives so we take this mathematical expression 3x squared minus 4x plus 5
and you would write out on a piece of paper and you would you know apply the product rule and all the other rules and derive the mathematical expression of
the great derivative of the original function and then you could plug in different texts and see what the derivative is
we're not going to actually do that because no one in neural networks actually writes out the expression for
the neural net it would be a massive expression um it would be you know thousands tens of thousands of terms no
one actually derives the derivative of course and so we're not going to take this kind of like a symbolic approach
instead what i'd like to do is i'd like to look at the definition of derivative and just make sure that we really understand what derivative is measuring
what it's telling you about the function and so if we just look up derivative
we see that okay so this is not a very good definition of derivative this is a definition of what it means to be differentiable
but if you remember from your calculus it is the limit as h goes to zero of f of x plus h minus f of x over h so
basically what it's saying is if you slightly bump up you're at some point x
that you're interested in or a and if you slightly bump up you know you slightly increase it by
small number h how does the function respond with what sensitivity does it respond what is the
slope at that point does the function go up or does it go down and by how much and that's the slope of that function
the the slope of that response at that point and so we can basically evaluate
the derivative here numerically by taking a very small h of course the definition would ask us to take h to
zero we're just going to pick a very small h 0.001 and let's say we're interested in point 3.0 so we can look at f of x of course
as 20 and now f of x plus h so if we slightly nudge x in a positive
direction how is the function going to respond and just looking at this do you expect do you expect f of x plus h to be
slightly greater than 20 or do you expect to be slightly lower than 20
and since this 3 is here and this is 20 if we slightly go positively the function will respond positively so
you'd expect this to be slightly greater than 20. and now by how much it's telling you the
sort of the the strength of that slope right the the size of the slope so f of x plus h minus
f of x this is how much the function responded in the positive direction and we have to
normalize by the run so we have the rise over run to get the slope so this of course is just a
numerical approximation of the slope because we have to make age very very small to converge to the exact amount
now if i'm doing too many zeros at some point i'm gonna get an incorrect answer
because we're using floating point arithmetic and the representations of all these numbers in computer memory is
finite and at some point we get into trouble so we can converse towards the right answer with this approach
but basically um at 3 the slope is 14. and you can see that by taking 3x
squared minus 4x plus 5 and differentiating it in our head so 3x squared would be
6 x minus 4 and then we plug in x equals 3 so that's 18 minus 4 is 14. so this is correct
so that's at 3. now how about the slope at say negative 3
would you expect would you expect for the slope now telling the exact value is really hard but what is the sign of that slope
so at negative three if we slightly go in the positive direction at x the function would
actually go down and so that tells you that the slope would be negative so we'll get a slight number below
below 20. and so if we take the slope we expect something negative negative 22. okay
and at some point here of course the slope would be zero now for this specific function i looked it up
previously and it's at point two over three so at roughly two over three
uh that's somewhere here um this derivative be zero so basically at that precise point
yeah at that precise point if we nudge in a positive direction the function doesn't respond this stays the same almost and
so that's why the slope is zero 

当然，以下是这段英文内容的中文翻译：

---

我们先创建一组**标量值（scalar values）**，比如使用 `range` 从 -5 到 5，步长为 0.25。
也就是说，x 轴的值是从 -5 到 5（不包含 5），步长为 0.25。

我们可以把这些值组成一个 NumPy 数组，然后把这个数组输入我们定义好的函数 `f`，
这样我们就可以计算出对应的一组 y 值 —— 相当于把函数作用在每一个 x 值上。

我们可以用 matplotlib 来画出这组 `x` 和 `y`，就能看到一个很漂亮的抛物线图。

比如说，我们之前输入了 `x = 3.0`，函数输出是 `20`，
那我们在图上也能看到这个点 —— 横坐标 3.0、纵坐标 20。

---

现在我想带大家思考一个问题：

> 对于这个函数，在任意一个点 `x` 上，它的**导数（slope）是多少**？

换句话说，这个函数在不同的 `x` 点上，它的斜率是多少？

你可能还记得高中或大学微积分课学过的知识：
我们会把函数写出来，比如 `f(x) = 3x² - 4x + 5`，
然后在纸上手动计算它的导数，比如应用乘法法则、求导规则等，
最终得到它的导函数，比如 `f'(x) = 6x - 4`，
然后你就可以代入不同的 `x` 来得到具体的导数值。

---

但我们这里**不会这么做**，因为在神经网络中，没有人会把整个神经网络的数学表达式写出来 ——
那可能是成千上万个变量和操作的组合，非常庞大！

所以我们不采用“符号求导”（symbolic differentiation）的方法。

相反，我想让你通过**导数的定义**来真正理解：

> 导数到底表示什么？它告诉你函数的什么信息？

---

### 🔍 回顾导数的定义：

从微积分中我们知道，导数的定义是：

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

也就是说：

* 如果我们稍微把输入 `x` 增加一个非常小的数 `h`，
* 那么 `f(x+h)` 相对于 `f(x)` 增加了多少？
* 这个变化率就是导数，也就是**函数在 `x` 点的斜率**。

导数告诉我们：

> 如果你轻微改变输入，函数输出会如何、以多大幅度改变。

---

我们现在来用一个非常小的 `h = 0.001`，以 `x = 3.0` 为例进行数值估算：

* `f(x) = f(3.0) = 20`
* `f(x + h) = f(3.001)` 是什么？
* 它会比 20 稍微大一点，为什么？因为这个函数在 `x=3` 附近是上升的。

我们计算：

$$
\text{slope} ≈ \frac{f(3.001) - f(3.0)}{0.001}
$$

这就是一个**数值近似的导数**。虽然不是精确值，但很接近。

> 注意：`h` 不能太小，否则计算机会因为精度限制出错（浮点数精度有限）。

---

计算后我们发现：

* 在 `x=3` 的导数是 `14`。
* 如果你用脑算的方式对函数 `f(x) = 3x² - 4x + 5` 求导，得到：

  $$
  f'(x) = 6x - 4
  $$

  代入 `x = 3`：`f'(3) = 6×3 - 4 = 14`，刚好一致。

---

接着我们问：

> 那在 `x = -3` 呢？导数是正的还是负的？

* 我们从图上看，在 `x = -3` 附近，函数是向上的抛物线左侧部分，是下降趋势；
* 所以如果你从 -3 向右移动一点点（正方向），函数值会下降；
* 这意味着**导数是负的**，你会得到一个负数，比如 `-22`。

---

再往右移，函数的斜率逐渐变小；
在某个点上，斜率变成 0（也就是函数的最低点）—— 在这个例子中是 `x = 2/3 ≈ 0.666`：

* 在这个点附近，`f(x+h) ≈ f(x)`，几乎没有变化；
* 所以导数为 0；
* 表示函数在这一点**局部最平坦，不再上升或下降**。

---

## ✅ 总结：

* **导数 = 某一点的斜率 = 输出对输入的敏感程度**
* 数值导数 = $\frac{f(x+h) - f(x)}{h}$
* 在神经网络中，我们不会手动写出整个函数表达式；

  * 而是依靠自动微分（autograd）系统，根据定义反向传播自动求导。

---

如果你希望，我可以：

* 画出这段函数图像，标出斜率；
* 写一个 Python 函数帮你数值近似地估导数；
* 展示如何在 PyTorch 中用 `.backward()` 自动求导。

你想深入哪个方向？

derivative of a function with multiple inputs

okay now let's look at a bit more complex case so we're going to start you know
complexifying a bit so now we have a function here
with output variable d that is a function of three scalar inputs a b and c
so a b and c are some specific values three inputs into our expression graph and a single output d
and so if we just print d we get four and now what i have to do is i'd like to
again look at the derivatives of d with respect to a b and c and uh think through uh again just the
intuition of what this derivative is telling us so in order to evaluate this derivative we're going to get a bit hacky here
we're going to again have a very small value of h and then we're going to fix the inputs
at some values that we're interested in so these are the this is the point abc
at which we're going to be evaluating the the derivative of d with respect to all a b and c at that point
so there are the inputs and now we have d1 is that expression and then we're going to for example look
at the derivative of d with respect to a so we'll take a and we'll bump it by h and then we'll get d2 to be the exact
same function and now we're going to print um you know f1
d1 is d1 d2 is d2 and print slope
so the derivative or slope here will be um of course
d2 minus d1 divide h so d2 minus d1 is how much the function
increased uh when we bumped the uh the specific input that we're interested
in by a tiny amount and this is then normalized by h
to get the slope so um
yeah so this so if i just run this we're going to print
d1 which we know is four
now d2 will be bumped a will be bumped by h so let's just think through
a little bit uh what d2 will be uh printed out here
in particular d1 will be four will d2 be a number slightly greater
than four or slightly lower than four and that's going to tell us the sl the the sign of the derivative
so we're bumping a by h
b as minus three c is ten so you can just intuitively think through this derivative and what it's
doing a will be slightly more positive and but b is a negative number
so if a is slightly more positive because b is negative three
we're actually going to be adding less to d so you'd actually expect that the value
of the function will go down so let's just see this
yeah and so we went from 4 to 3.9996 and that tells you that the slope will
be negative and then uh will be a negative number because we went down
and then the exact number of slope will be exact amount of slope is negative 3.
and you can also convince yourself that negative 3 is the right answer mathematically and analytically because
if you have a times b plus c and you are you know you have calculus then differentiating a times b plus c with
respect to a gives you just b and indeed the value of b is negative 3 which is the derivative that we have so
you can tell that that's correct so now if we do this with b so if we bump b by a little bit in a
positive direction we'd get different slopes so what is the influence of b on the output d
so if we bump b by a tiny amount in a positive direction then because a is positive
we'll be adding more to d right so um and now what is the what is the
sensitivity what is the slope of that addition and it might not surprise you that this should be
2 and y is a 2 because d of d by db differentiating with respect to b
would be would give us a and the value of a is two so that's also working well
and then if c gets bumped a tiny amount in h by h then of course a times b is unaffected
and now c becomes slightly bit higher what does that do to the function it makes it slightly bit higher because
we're simply adding c and it makes it slightly bit higher by the exact same amount that we added to c
and so that tells you that the slope is one that will be the
the rate at which d will increase as we scale c
okay so we now have some intuitive sense of what this derivative is telling you about the function and we'd like to move to neural networks now as i mentioned

当然，以下是这段英文内容的中文翻译：

---

现在我们来看一个稍微复杂一点的例子，我们要开始让问题“更复杂”一些。

我们有一个函数，它的输出变量是 `d`，它是三个标量输入 `a`、`b` 和 `c` 的函数。
也就是说，`a`、`b`、`c` 是表达式图（expression graph）中的输入，`d` 是输出。

当我们打印 `d` 时，得到的值是 `4`。

接下来，我们还是想研究：

> **`d` 分别对 `a`、`b` 和 `c` 的导数是多少？**

并且我们要继续思考导数的**直觉含义**：
导数到底在告诉我们这个函数怎样的性质？

---

为了求这些导数，我们会用一种“数值估算”的方式：

* 设定一个非常小的 `h`（例如 0.0001）；
* 固定输入值 `a, b, c` 为我们关心的特定值；
* 然后我们先计算当前的 `d1 = f(a, b, c)`；
* 接着我们稍微“**扰动**”某一个输入，比如 `a = a + h`；
* 然后重新计算一次：`d2 = f(a+h, b, c)`；
* 最后我们计算导数近似为：

$$
\text{slope} \approx \frac{d2 - d1}{h}
$$

也就是说，我们把某个输入加了一点点，函数输出的变化除以这点变化，就是导数。

---

现在我们运行这个代码，会打印出：

* `d1 = 4`
* `d2` 是我们在 `a` 上扰动后的输出

我们来思考一下：

如果我们轻微增加 `a`，`d2` 会比 `d1` **大** 还是 **小**？

假设 `b = -3`，`c = 10`，我们要看 `a` 增大时对 `d = ab + c` 的影响。

由于 `b` 是负的，而 `a` 增加，那么 `ab` 会变得更负（负数变大了），
所以 `d = ab + c` 会变得更小。

所以我们**预计 `d2 < d1`，导数是负的**。

运行结果显示：

* `d1 = 4`
* `d2 ≈ 3.9996`
* 所以导数（斜率）约为 `-3`

---

我们可以验证这是对的：

原始函数是：

$$
d = ab + c
$$

对 `a` 求导是：

$$
\frac{∂d}{∂a} = b
$$

而 `b = -3`，所以导数确实是 `-3`，完美匹配！

---

接下来我们对 `b` 做同样的扰动：

* 如果我们让 `b` 增加一点点，而 `a = 2` 是正的，
* 那么 `ab` 会变大，也就是说 `d` 会变大。

导数是多少呢？

$$
\frac{∂d}{∂b} = a = 2
$$

也正是我们观察到的结果。

---

最后我们来看 `c`：

* 如果我们让 `c` 增加一点点，那么 `ab` 不变，`d = ab + c` 当然也就增加同样的量；
* 所以导数是：

$$
\frac{∂d}{∂c} = 1
$$

因为 `c` 在表达式中就是单独加进去的。

---

### ✅ 总结：

通过这种数值方式和直觉推理，我们得到了每个输入对输出的“敏感度”：

| 输入变量 | 导数（敏感度） | 解释                               |
| ---- | ------- | -------------------------------- |
| `a`  | `-3`    | 因为 `∂(ab + c)/∂a = b`，而 `b = -3` |
| `b`  | `2`     | 因为 `∂(ab + c)/∂b = a`，而 `a = 2`  |
| `c`  | `1`     | 因为 `∂(ab + c)/∂c = 1`            |

---

通过这个例子，我们对“导数告诉我们什么”有了直观理解 ——
也就是**输入变量每变化一点，输出会怎么变化、变化多快**。

接下来，我们就可以正式进入神经网络的部分了。正如我之前所说的……

（接下来就是进入网络结构本身的讲解）

---

如果你想我把这个例子用代码画出来、可视化 `d` 对 `a/b/c` 的曲线、或扩展为向量版本，也可以告诉我！


# starting the core Value object of micrograd and its visualization

neural networks will be pretty massive expressions mathematical expressions so we need some data structures that maintain these expressions and that's
what we're going to start to build out now so we're going to build out this value object that i
showed you in the readme page of micrograd so let me copy paste a skeleton of the
first very simple value object so class value takes a single
scalar value that it wraps and keeps track of and that's it so
we can for example do value of 2.0 and then we can get we can look at its content and
python will internally use the wrapper function to uh return
uh this string oops like that so this is a value object with data
equals two that we're creating here now we'd like to do is like we'd like to be able to
have not just like two values but we'd like to do a bluffy right we'd like to add them
so currently you would get an error because python doesn't know how to add two value objects so we have to tell it
so here's addition so you have to basically use these
special double underscore methods in python to define these operators for these objects so if we call um
the uh if we use this plus operator python will internally call a dot add of
b that's what will happen internally and so b will be the other and
self will be a and so we see that what we're going to return is a new value object and it's
just it's going to be wrapping the plus of their data
but remember now because data is the actual like numbered python number so this operator here is just the typical
floating point plus addition now it's not an addition of value objects and will return a new value so now a
plus b should work and it should print value of negative one because that's two plus minus three
there we go okay let's now implement multiply just so we can recreate this expression
here so multiply i think it won't surprise you will be fairly similar
so instead of add we're going to be using mul and then here of course we want to do times
and so now we can create a c value object which will be 10.0 and now we should be able to do a times b well
let's just do a times b first um [Music] that's value of negative six now
and by the way i skipped over this a little bit suppose that i didn't have the wrapper function here then it's just that you'll get some kind
of an ugly expression so what wrapper is doing is it's providing us a way to print out like a nicer looking
expression in python uh so we don't just have something cryptic we actually are you know it's
value of negative six so this gives us a times and then this we should now be able to
add c to it because we've defined and told the python how to do mul and add and so this will call this will
basically be equivalent to a dot small of b
and then this new value object will be dot add of c and so let's see if that worked
yep so that worked well that gave us four which is what we expect from before and i believe we can just call them
manually as well there we go so yeah okay so now what we are missing is the
connective tissue of this expression as i mentioned we want to keep these expression graphs so we need to know and
keep pointers about what values produce what other values so here for example we are going to
introduce a new variable which we'll call children and by default it will be an empty tuple and then we're actually going to keep a
slightly different variable in the class which we'll call underscore prev which will be the set of children
this is how i done i did it in the original micrograd looking at my code here i can't remember exactly the reason
i believe it was efficiency but this underscore children will be a tuple for convenience but then when we actually
maintain it in the class it will be just this set yeah i believe for efficiency
um so now when we are creating a value like this with a constructor children will be
empty and prep will be the empty set but when we're creating a value through addition or multiplication we're going
to feed in the children of this value which in this case is self and other
so those are the children here so now we can do d dot prev
and we'll see that the children of the we now know are this value of negative 6
and value of 10 and this of course is the value resulting from a times b and the c value which is 10.
now the last piece of information we don't know so we know that the children of every single value but we don't know
what operation created this value so we need one more element here let's call it underscore pop
and by default this is the empty set for leaves and then we'll just maintain it here
and now the operation will be just a simple string and in the case of addition it's plus in the case of
multiplication is times so now we not just have d dot pref we also have a
d dot up and we know that d was produced by an addition of those two values and so now
we have the full mathematical expression uh and we're building out this data structure and we know exactly how each value came to be
by word expression and from what other values now because these expressions are about
to get quite a bit larger we'd like a way to nicely visualize these expressions that we're building out so
for that i'm going to copy paste a bunch of slightly scary code that's going to visualize this these expression graphs
for us so here's the code and i'll explain it in a bit but first let me just show you what this code does
basically what it does is it creates a new function drawdot that we can call on some root node
and then it's going to visualize it so if we call drawdot on d which is this final value here that is a
times b plus c it creates something like this so this is d
and you see that this is a times b creating an integrated value plus c gives us this output node d
so that's dried out of d and i'm not going to go through this in complete detail you can take a look at
graphless and its api uh graphis is a open source graph visualization software
and what we're doing here is we're building out this graph and graphis api and
you can basically see that trace is this helper function that enumerates all of the nodes and edges in the graph
so that just builds a set of all the nodes and edges and then we iterate for all the nodes and we create special node
objects for them in using dot node
and then we also create edges using dot dot edge and the only thing that's like slightly tricky here is you'll notice that i
basically add these fake nodes which are these operation nodes so for example this node here is just like a plus node
and i create these special op nodes here
and i connect them accordingly so these nodes of course are not actual
nodes in the original graph they're not actually a value object the only value objects here are the things
in squares those are actual value objects or representations thereof and these op nodes are just created in this
drawdot routine so that it looks nice let's also add labels to these graphs
just so we know what variables are where so let's create a special underscore label
um or let's just do label equals empty by default and save it in
each node and then here we're going to do label as a
label is the label a c
and then let's create a special um e equals a times b
and e dot label will be e it's kind of naughty and e will be e plus c
and a d dot label will be d okay so nothing really changes i just
added this new e function a new e variable and then here when we are
printing this i'm going to print the label here so this will be a percent s
bar and this will be end.label
and so now we have the label on the left here so it says a b creating e and then e plus c
creates d just like we have it here and finally let's make this expression just one layer deeper
so d will not be the final output node instead after d we are going to create a
new value object called f we're going to start running out of variables soon f will be negative
2.0 and its label will of course just be f and then l capital l will be the output
of our graph and l will be p times f okay so l will be negative eight is the
output so now we don't just draw a d we draw l
okay and somehow the label of l was undefined oops all that label has
to be explicitly sort of given to it there we go so l is the output
so let's quickly recap what we've done so far we are able to build out mathematical expressions using only plus and times so
far they are scalar valued along the way and we can do this forward pass
and build out a mathematical expression so we have multiple inputs here a b c and f
going into a mathematical expression that produces a single output l and this here is visualizing the forward
pass so the output of the forward pass is negative eight that's the value
now what we'd like to do next is we'd like to run back propagation and in back propagation we are going to
start here at the end and we're going to reverse and calculate the gradient along along
all these intermediate values and really what we're computing for every single value here
um we're going to compute the derivative of that node with respect to l
so the derivative of l with respect to l is just uh one
and then we're going to derive what is the derivative of l with respect to f with respect to d with respect to c with
respect to e with respect to b and with respect to a and in the neural network setting you'd
be very interested in the derivative of basically this loss function l with respect to the weights of a neural
network and here of course we have just these variables a b c and f but some of these will eventually
represent the weights of a neural net and so we'll need to know how those weights are impacting
the loss function so we'll be interested basically in the derivative of the output with respect to some of its leaf
nodes and those leaf nodes will be the weights of the neural net and the other leaf nodes of course will be the data itself but usually we will
not want or use the derivative of the loss function with respect to data because the data is fixed but the
weights will be iterated on using the gradient information so next we are going to create a variable inside
the value class that maintains the derivative of l with respect to that
value and we will call this variable grad so there's a data and there's a
self.grad and initially it will be zero and remember that zero is basically means no
effect so at initialization we're assuming that every value does not impact does not affect the out the
output right because if the gradient is zero that means that changing this variable is not changing the loss function
so by default we assume that the gradient is zero and then
now that we have grad and it's 0.0
we are going to be able to visualize it here after data so here grad is 0.4 f
and this will be in that graph and now we are going to be showing both the data and the grad
initialized at zero and we are just about getting ready to calculate the back propagation
and of course this grad again as i mentioned is representing the derivative of the output in this case l with respect to this value so
with respect to so this is the derivative of l with respect to f with respect to d and so on so let's now fill

当然！以下是这段内容的中文翻译，非常详细，但保持原意清晰：

---

### 🚀 从 micrograd 的核心 Value 对象和它的可视化说起

神经网络将会变成非常庞大的**数学表达式**，
所以我们需要一种数据结构来**维护这些表达式**，
这正是我们现在要开始构建的 —— 也就是 **Value 对象**。

---

### ✅ 我们先定义一个最简单的 `Value` 类

它接受一个**标量数值（scalar value）**，把它包裹起来（封装），并记录这个值。
比如我们创建一个 `Value(2.0)`，然后我们可以打印它。

Python 会调用我们定义的 `__repr__()` 方法来返回字符串表示：
输出看起来像 `Value(data=2.0)`。

---

### ➕ 实现加法运算

我们希望 `Value` 不只是保存值，还能参与计算，比如：

```python
a = Value(2.0)
b = Value(-3.0)
d = a + b  # 应该输出 Value(-1.0)
```

一开始这会报错，因为 Python 不知道怎么把两个 `Value` 对象加起来。

我们需要定义 `__add__` 魔法方法，告诉 Python：

```python
def __add__(self, other):
    return Value(self.data + other.data)
```

这样以后，`a + b` 背后其实就是调用 `a.__add__(b)`，最终返回一个新的 `Value` 对象。

---

### ✖️ 实现乘法运算

我们也可以定义 `__mul__` 方法，来支持 `a * b`。
逻辑非常类似，只是用 `*` 而不是 `+`。

```python
def __mul__(self, other):
    return Value(self.data * other.data)
```

---

### 🖨️ 定义 **repr**：美化输出

如果不实现 `__repr__`，打印出来的对象会很丑：
像 `<__main__.Value object at 0x...>`。

我们定义一个 `__repr__()` 方法，
让打印变成更可读的形式：`Value(data=-6.0)`。

---

### 🌿 构建计算图：记录谁来自谁

我们不仅希望能算出结果，还希望能**追踪是谁“计算”出了这个结果**。

为此，我们在 `Value` 类中添加属性：

```python
self._prev = set()   # 存储这个值是由哪些值计算出来的
self._op = ''        # 是加法？乘法？其他操作？
```

例如：

```python
d = a * b + c
```

那么 `d._prev` 就是 `{e, c}`（其中 e 是 `a * b` 的结果），
`d._op` 是 `+`。

这样我们就能在计算图中回溯每个节点的来源。

---

### 🧾 可视化计算图

为了更好地理解这些表达式，我们引入一个函数 `draw_dot()` 来画出计算图。

它使用开源的 **Graphviz** 工具来画图。我们会：

* 遍历所有 `Value` 节点；
* 对每个 `Value` 创建一个图节点（方框）；
* 对每个操作（如加法）创建一个“虚拟节点”（圆形、带 `+` 或 `*`）；
* 添加边（edges）表示依赖关系。

最终图形展示类似：

```
 a     b     c
  \   /       \
   *     -->   +
    \         /
         d
```

你可以在图上看到变量 `a` 和 `b` 相乘生成 `e`，再加上 `c` 得到 `d`。

---

### 🧱 增加标签

我们给每个节点加上 `.label`，比如：

```python
a.label = "a"
b.label = "b"
e.label = "e"
d.label = "d"
```

这会在图中显示变量名，帮助识别。

---

### 🔁 更深一层

我们可以继续扩展表达式，让计算更复杂：

```python
f = Value(-2.0); f.label = "f"
l = d * f
l.label = "l"
```

现在输出不是 `d`，而是 `l = d * f`，值是 `-8`。我们也更新 `draw_dot(l)` 来可视化它。

---

### 📤 准备反向传播：引入 grad（梯度）

我们想做的不只是前向传播，还想做**反向传播（backpropagation）**。

所以我们要为每个 `Value` 增加 `.grad` 属性：

```python
self.grad = 0.0
```

这个值表示：

> **当前这个变量对最终输出（比如 `l`）的导数是多少？**

举例：

* `f.grad = ∂l/∂f`
* `a.grad = ∂l/∂a`

初始值设为 0，意味着“这个变量对最终输出还没有贡献”。

---

### ✅ 到目前为止，我们已经实现：

| 功能     | 实现内容                  |
| ------ | --------------------- |
| 前向传播   | 加法、乘法运算，封装在 Value 对象中 |
| 计算图构建  | 每个 Value 记录父节点和操作类型   |
| 可视化    | 用 Graphviz 显示表达式结构    |
| 准备反向传播 | 引入 `grad` 存储导数信息      |

---

接下来，我们会正式实现 `.backward()` 方法，
它会从 `l` 开始，反向走回去，**一步步计算梯度**（用链式法则）。

如果你想我继续帮你翻译或讲解 `.backward()` 的实现，或者帮你可视化反向传播过程，请告诉我！


# manual backpropagation example #1: simple expression

in those gradients and actually do back propagation manually so let's start filling in these gradients and start all
the way at the end as i mentioned here first we are interested to fill in this gradient here so what is the derivative
of l with respect to l in other words if i change l by a tiny amount of h
how much does l change it changes by h so it's proportional and
therefore derivative will be one we can of course measure these or estimate these numerical gradients
numerically just like we've seen before so if i take this expression and i create a def lol function here
and put this here now the reason i'm creating a gating function hello here is because i don't want to pollute or mess
up the global scope here this is just kind of like a little staging area and as you know in python all of these will
be local variables to this function so i'm not changing any of the global scope here
so here l1 will be l and then copy pasting this expression
we're going to add a small amount h in for example a
right and this would be measuring the derivative of l with respect to a so here this will be l2
and then we want to print this derivative so print l2 minus l1 which is how much l changed
and then normalize it by h so this is the rise over run and we have to be careful because l is a
value node so we actually want its data um so that these are floats dividing by h
and this should print the derivative of l with respect to a because a is the one that we bumped a little bit by h
so what is the derivative of l with respect to a it's six
okay and obviously if we change l by h
then that would be here effectively
this looks really awkward but changing l by h you see the derivative here is 1. um
that's kind of like the base case of what we are doing here so basically we cannot come up here and
we can manually set l.grad to one this is our manual back propagation
l dot grad is one and let's redraw and we'll see that we filled in grad as
1 for l we're now going to continue the back propagation so let's here look at the derivatives of l with respect to d and f
let's do a d first so what we are interested in if i create a markdown on here is we'd like to know
basically we have that l is d times f and we'd like to know what is uh d
l by d d what is that and if you know your calculus uh l is d
times f so what is d l by d d it would be f and if you don't believe me we can also
just derive it because the proof would be fairly straightforward uh we go to the
definition of the derivative which is f of x plus h minus f of x divide h
as a limit limit of h goes to zero of this kind of expression so when we have l is d times f
then increasing d by h would give us the output of b plus h
times f that's basically f of x plus h right
minus d times f and then divide h and symbolically
expanding out here we would have basically d times f plus h times f minus
t times f divide h and then you see how the df minus df cancels so you're left with h times f
divide h which is f so in the limit as h goes to zero of
you know derivative definition we just get f in the case of
d times f so symmetrically dl by d
f will just be d so what we have is that f dot grad
we see now is just the value of d which is 4.
and we see that d dot grad is just uh the value of f
and so the value of f is negative two so we'll set those manually
let me erase this markdown node and then let's redraw what we have
okay and let's just make sure that these were correct so we seem to think that dl by
dd is negative two so let's double check um let me erase this plus h from before
and now we want the derivative with respect to f so let's just come here when i create f and let's do a plus h here and this
should print the derivative of l with respect to f so we expect to see four
yeah and this is four up to floating point funkiness and then dl by dd
should be f which is negative two grad is negative two
so if we again come here and we change d
d dot data plus equals h right here so we expect so we've added a little h
and then we see how l changed and we expect to print uh negative two
there we go so we've numerically verified what we're doing here is what kind of like an
inline gradient check gradient check is when we are deriving this like back propagation
and getting the derivative with respect to all the intermediate results and then numerical gradient is just you know
estimating it using small step size now we're getting to the crux of backpropagation so this will be the most
important node to understand because if you understand the gradient for this node you understand all of back
propagation and all of training of neural nets basically so we need to derive dl by bc
in other words the derivative of l with respect to c because we've computed all these other gradients already
now we're coming here and we're continuing the back propagation manually so we want dl by dc and then we'll also
derive dl by de now here's the problem how do we derive dl
by dc we actually know the derivative l with respect to d so we know how l assessed
it to d but how is l sensitive to c so if we wiggle c how does that impact l
through d so we know dl by dc
and we also here know how c impacts d and so just very intuitively if you know the impact that c is having on d and the
impact that d is having on l then you should be able to somehow put that information together to figure out
how c impacts l and indeed this is what we can actually do so in particular we know just
concentrating on d first let's look at how what is the derivative basically of d with respect to c so in other words
what is dd by dc so here we know that d is c times c plus
e that's what we know and now we're interested in dd by dc if you just know your calculus again and
you remember that differentiating c plus e with respect to c you know that that gives you
1.0 and we can also go back to the basics and derive this because again we can go
to our f of x plus h minus f of x divide by h that's the definition of a derivative as
h goes to zero and so here focusing on c and its effect on d
we can basically do the f of x plus h will be c is incremented by h plus e
that's the first evaluation of our function minus c plus e
and then divide h and so what is this uh just expanding this out this will be
c plus h plus e minus c minus e divide h and then you see here how c
minus c cancels e minus e cancels we're left with h over h which is 1.0
and so by symmetry also d d by d
e will be 1.0 as well so basically the derivative of a sum
expression is very simple and and this is the local derivative so i call this the local derivative because we have the
final output value all the way at the end of this graph and we're now like a small node here and this is a little plus node
and it the little plus node doesn't know anything about the rest of the graph that it's embedded in all it knows is
that it did a plus it took a c and an e added them and created d and this plus note also knows the local
influence of c on d or rather rather the derivative of d with respect to c and it
also knows the derivative of d with respect to e but that's not what we want that's just a local derivative what we actually
want is d l by d c and l could l is here just one step away but in a general case
this little plus note is could be embedded in like a massive graph so
again we know how l impacts d and now we know how c and e impact d how do we put
that information together to write dl by dc and the answer of course is the chain rule in calculus
and so um i pulled up a chain rule here from kapedia
and i'm going to go through this very briefly so chain rule wikipedia sometimes can be very
confusing and calculus can can be very confusing like this is the way i
learned chain rule and it was very confusing like what is happening it's just complicated so i like this expression
much better if a variable z depends on a variable y which itself depends on the variable x
then z depends on x as well obviously through the intermediate variable y in this case the chain rule is expressed
as if you want dz by dx then you take the dz by dy and you
multiply it by d y by dx so the chain rule fundamentally is telling you
how we chain these uh derivatives together
correctly so to differentiate through a function composition we have to apply a multiplication
of those derivatives so that's really what chain rule is telling us
and there's a nice little intuitive explanation here which i also think is kind of cute the chain rule says that
knowing the instantaneous rate of change of z with respect to y and y relative to x allows one to calculate the instantaneous rate of change of z
relative to x as a product of those two rates of change simply the product of those two
so here's a good one if a car travels twice as fast as bicycle and the bicycle is four times as
fast as walking man then the car travels two times four eight times as fast as demand
and so this makes it very clear that the correct thing to do sort of is to multiply
so cars twice as fast as bicycle and bicycle is four times as fast as man
so the car will be eight times as fast as the man and so we can take these
intermediate rates of change if you will and multiply them together and that justifies the
chain rule intuitively so have a look at chain rule about here really what it means for us is there's a very simple
recipe for deriving what we want which is dl by dc and what we have so far
is we know want and we know
what is the impact of d on l so we know d l by
d d the derivative of l with respect to d d we know that that's negative two and now because of this local
reasoning that we've done here we know dd by d c
so how does c impact d and in particular this is a plus node so the
local derivative is simply 1.0 it's very simple and so the chain rule tells us that dl by dc
going through this intermediate variable will just be simply d l by
d times
dd by dc that's chain rule so this is identical to what's happening
here except z is rl y is our d and x is rc
so we literally just have to multiply these and because
these local derivatives like dd by dc are just one we basically just copy over dl by dd
because this is just times one so what does it do so because dl by dd is negative two what is dl by dc
well it's the local gradient 1.0 times dl by dd which is negative two
so literally what a plus node does you can look at it that way is it literally just routes the gradient
because the plus nodes local derivatives are just one and so in the chain rule one times
dl by dd is um is uh is just dl by dd and so that
derivative just gets routed to both c and to e in this case
so basically um we have that that grad or let's start with c since that's the
one we looked at is negative two times one
negative two and in the same way by symmetry e that grad will be negative two that's the
claim so we can set those we can redraw
and you see how we just assign negative to negative two so this backpropagating signal which is carrying the information
of like what is the derivative of l with respect to all the intermediate nodes we can imagine it almost like flowing
backwards through the graph and a plus node will simply distribute the derivative to all the leaf nodes sorry
to all the children nodes of it so this is the claim and now let's verify it so let me remove the plus h
here from before and now instead what we're going to do is we're going to increment c so c dot
data will be credited by h and when i run this we expect to see negative 2
negative 2. and then of course for e so e dot data plus equals h and we
expect to see negative 2. simple
so those are the derivatives of these internal nodes and now we're going to recurse our way
backwards again and we're again going to apply the chain rule so here we go our second
application of chain rule and we will apply it all the way through the graph we just happen to only have one more node remaining
we have that d l by d e as we have just calculated is negative two so we know that
so we know the derivative of l with respect to e and now we want dl
by da right and the chain rule is telling us that that's just dl by de
negative 2 times the local gradient so what is the local gradient basically d e
by d a we have to look at that so i'm a little times node
inside a massive graph and i only know that i did a times b and i produced an e
so now what is d e by d a and d e by d b that's the only thing that i sort of
know about that's my local gradient so because we have that e's a times b we're
asking what is d e by d a and of course we just did that here we
had a times so i'm not going to rederive it but if you want to differentiate this
with respect to a you'll just get b right the value of b which in this case is negative 3.0
so basically we have that dl by da well let me just do it right here we
have that a dot grad and we are applying chain rule here is d l by d e which we see here is
negative two times what is d e by d a
it's the value of b which is negative 3.
that's it and then we have b grad is again dl by
de which is negative 2 just the same way times what is d e by d
um db is the value of a which is 2.2.0
as the value of a so these are our claimed derivatives
let's redraw and we see here that a dot grad turns out to be 6 because
that is negative 2 times negative 3 and b dot grad is negative 4
times sorry is negative 2 times 2 which is negative 4. so those are our claims let's delete
this and let's verify them we have
a here a dot data plus equals h so the claim is that
a dot grad is six let's verify six
and we have beta data plus equals h so nudging b by h
and looking at what happens we claim it's negative four and indeed it's negative four plus minus
again float oddness um and uh
that's it this that was the manual back propagation
uh all the way from here to all the leaf nodes and we've done it piece by piece and really all we've done is as you saw
we iterated through all the nodes one by one and locally applied the chain rule we always know what is the derivative of
l with respect to this little output and then we look at how this output was produced this output was produced
through some operation and we have the pointers to the children nodes of this operation
and so in this little operation we know what the local derivatives are and we just multiply them onto the derivative
always so we just go through and recursively multiply on the local derivatives and
that's what back propagation is is just a recursive application of chain rule backwards through the computation graph
preview of a single optimization step
let's see this power in action just very briefly what we're going to do is we're going to
nudge our inputs to try to make l go up so in particular what we're doing is we
want a.data we're going to change it and if we want l to go up that means we just have to go in the direction of the
gradient so a should increase in the direction of gradient by like some small step amount
this is the step size and we don't just want this for ba but also for b
also for c also for f those are
leaf nodes which we usually have control over and if we nudge in direction of the
gradient we expect a positive influence on l so we expect l to go up
positively so it should become less negative it should go up to say negative you know
six or something like that uh it's hard to tell exactly and we'd have to rewrite the forward pass so let
me just um do that here um
this would be the forward pass f would be unchanged this is effectively the forward pass and now if we print l.data
we expect because we nudged all the values all the inputs in the rational gradient we expected a less negative l
we expect it to go up so maybe it's negative six or so let's see what happens
okay negative seven and uh this is basically one step of an
optimization that we'll end up running and really does gradient just give us some power because we know how to
influence the final outcome and this will be extremely useful for training knowledge as well as you'll see

当然可以，下面是这段内容的详细中文翻译，内容非常丰富，我也为你分段做了逻辑整理，方便理解：

---

## 📘 手动反向传播 示例一：简单表达式

现在我们来手动计算这些梯度，也就是手动执行反向传播。
我们从计算图的最后一层开始，一步步往前传递。

---

### ✅ 基础：`l` 对 `l` 的导数

我们要先填入 `l` 对自身的梯度 ∂l/∂l，这个值是多少？

* 直觉上：如果你改变 `l` 一点点（比如加上一个很小的 h），`l` 的值也会正好增加 h。
* 所以导数是 1。

```python
l.grad = 1
```

---

### 🧪 数值验证：l 对 a 的梯度（∂l/∂a）

我们也可以像之前那样数值计算：

```python
l1 = l.data
a.data += h
l2 = new_l.data
( l2 - l1 ) / h = 6
```

这个结果告诉我们：**l 对 a 的导数是 6**。

---

### 🔁 继续传播：l 对 d 和 f 的导数

已知：

```python
l = d * f
```

应用链式法则（chain rule）：

* ∂l/∂d = f
* ∂l/∂f = d

当前：

* d.data = 4
* f.data = -2

所以：

```python
d.grad = ∂l/∂d = -2
f.grad = ∂l/∂f = 4
```

可以用数值验证这两个值也没问题。

---

### 🧠 核心部分：l 对 c、e 的导数

我们现在来看表达式：

```python
d = e + c
l = d * f
```

我们已经有：

* ∂l/∂d = -2
* ∂d/∂c = 1
* ∂d/∂e = 1

（因为 `d = e + c`，所以每个输入的局部梯度是 1）

那么：

* ∂l/∂c = ∂l/∂d × ∂d/∂c = -2 × 1 = -2
* ∂l/∂e = ∂l/∂d × ∂d/∂e = -2 × 1 = -2

所以设置：

```python
c.grad = -2
e.grad = -2
```

---

### ⛓️ 链式法则直观理解

> 如果 z 依赖于 y，而 y 又依赖于 x，那么 z 也依赖于 x，
> 那么：
>
> ```
> ∂z/∂x = ∂z/∂y × ∂y/∂x
> ```

打个比方：

* 汽车速度是自行车的 2 倍
* 自行车速度是步行的 4 倍
  \=> 汽车是人的 2 × 4 = 8 倍

这个“倍数传递”就是链式法则的核心。

---

### 🔄 继续传播到 `a` 和 `b`

我们再看表达式：

```python
e = a * b
```

* ∂l/∂e = -2
* ∂e/∂a = b = -3
* ∂e/∂b = a = 2

所以：

* ∂l/∂a = ∂l/∂e × ∂e/∂a = -2 × (-3) = 6
* ∂l/∂b = ∂l/∂e × ∂e/∂b = -2 × 2 = -4

赋值：

```python
a.grad = 6
b.grad = -4
```

再用数值方法验证也都一致。

---

### ✅ 总结：反向传播是什么？

我们一步步回退：

* 对每个节点，我们知道 **输出对它的导数**
* 再根据这个节点是如何由子节点构建出来的，应用链式法则计算子节点的梯度
* 不断地把梯度“传播”回去

**这就是反向传播（Backpropagation）**：
它本质上是**递归地应用链式法则**，从输出回退到所有叶子节点（如参数 a、b、c、f）

---

## 🧪 梯度的力量：执行一次优化

接下来我们来感受下梯度的实际用途。

我们尝试“优化”这些输入值，让最终的输出 `l` 增大（也就是更接近 0，不那么负了）：

```python
# 模拟一个梯度下降步骤
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad
```

然后再重新执行 forward pass：

```python
e = a * b
d = e + c
l = d * f
print(l.data)  # 输出应该比之前更大（更不负了）
```

原来 `l = -8`，现在变成 `-7`，说明我们成功通过梯度**推动了输出向“更优方向”前进**。

---

### 🧠 总结核心概念

| 概念            | 解释                            |
| ------------- | ----------------------------- |
| forward pass  | 从输入出发，按表达式计算输出                |
| backward pass | 从输出反推每个中间节点的导数                |
| grad          | 当前节点对最终输出的导数                  |
| chain rule    | ∂z/∂x = ∂z/∂y × ∂y/∂x         |
| 反向传播          | 递归应用 chain rule，计算出每个输入变量的影响力 |
| 优化            | 顺着导数方向，微调输入或权重，让输出变得更优        |

---

如果你需要我继续帮你实现代码版本，或者基于这个手动反向传播继续深入神经网络梯度下降、训练过程的自动化版本（如微型的 `micrograd.backward()`），随时告诉我！


# manual backpropagation example #2: a neuron

so now i would like to do one more uh example of manual backpropagation using
a bit more complex and uh useful example we are going to back propagate through a
neuron so we want to eventually build up neural networks and in the simplest case these
are multilateral perceptrons as they're called so this is a two layer neural net and it's got these hidden layers made up
of neurons and these neurons are fully connected to each other now biologically neurons are very complicated devices but we have very
simple mathematical models of them and so this is a very simple mathematical model of a neuron you have
some inputs axis and then you have these synapses that have weights on them so
the w's are weights and then the synapse interacts with the input to
this neuron multiplicatively so what flows to the cell body of this neuron is w times x
but there's multiple inputs so there's many w times x's flowing into the cell body
the cell body then has also like some bias so this is kind of like the inert innate sort of trigger happiness
of this neuron so this bias can make it a bit more trigger happy or a bit less trigger happy regardless of the input
but basically we're taking all the w times x of all the inputs adding the bias and
then we take it through an activation function and this activation function is usually some kind of a squashing function
like a sigmoid or 10h or something like that so as an example we're going to use the 10h in this
example numpy has a np.10h so
we can call it on a range and we can plot it this is the 10h function and you see
that the inputs as they come in get squashed on the y coordinate here so
um right at zero we're going to get exactly zero and then as you go more positive in
the input then you'll see that the function will only go up to one and then plateau out
and so if you pass in very positive inputs we're gonna cap it smoothly at one and on the negative side we're gonna
cap it smoothly to negative one so that's 10h and that's the squashing function or an
activation function and what comes out of this neuron is just the activation function applied to the dot product of
the weights and the inputs so let's write one out
um i'm going to copy paste because
i don't want to type too much but okay so here we have the inputs x1 x2 so this is a two-dimensional
neuron so two inputs are going to come in these are thought out as the weights of this neuron
weights w1 w2 and these weights again are the synaptic strengths for each
input and this is the bias of the neuron b
and now we want to do is according to this model we need to multiply x1 times
w1 and x2 times w2 and then we need to add bias on top of
it and it gets a little messy here but all we are trying to do is x1 w1 plus x2 w2
plus b and these are multiply here except i'm doing it in small steps so
that we actually have pointers to all these intermediate nodes so we have x1 w1 variable x times x2 w2 variable and
i'm also labeling them so n is now the cell body raw
raw activation without the activation function for now
and this should be enough to basically plot it so draw dot of n
gives us x1 times w1 x2 times w2 being added
then the bias gets added on top of this and this n is this sum
so we're now going to take it through an activation function and let's say we use the 10h
so that we produce the output so what we'd like to do here is we'd like to do the output and i'll call it o
is um n dot 10h okay but we haven't yet written the 10h
now the reason that we need to implement another 10h function here is that tanh is a
hyperbolic function and we've only so far implemented a plus and the times and you can't make a 10h out of just pluses
and times you also need exponentiation so 10h is this kind of a formula here
you can use either one of these and you see that there's exponentiation involved which we have not implemented yet for
our low value node here so we're not going to be able to produce 10h yet and we have to go back up and implement something like it
now one option here is we could actually implement um
exponentiation right and we could return the x of a value instead of a 10h of a value
because if we had x then we have everything else that we need so um because we know how to add and we know
how to um we know how to add and we know how to multiply so we'd be able to create 10h
if we knew how to x but for the purposes of this example i specifically wanted to
show you that we don't necessarily need to have the most atomic pieces
in um in this value object we can actually like create functions at arbitrary
points of abstraction they can be complicated functions but they can be also very very simple functions like a plus and it's totally up to us the only
thing that matters is that we know how to differentiate through any one function so we take some inputs and we
make an output the only thing that matters it can be arbitrarily complex function as long as you know how to
create the local derivative if you know the local derivative of how the inputs impact the output then that's all you
need so we're going to cluster up all of this expression and we're not going to break it down to its atomic
pieces we're just going to directly implement tanh so let's do that depth nh
and then out will be a value of and we need this expression here so
um let me actually copy paste
let's grab n which is a cell.theta and then this i believe is the tan h
math.x of two no n
n minus one over two n plus one maybe i can call this x
just so that it matches exactly okay and now this will be t
and uh children of this node there's just one child and i'm wrapping it in a tuple so this
is a tuple of one object just self and here the name of this operation will be 10h
and we're going to return that okay
so now valley should be implementing 10h and now we can scroll all the way down here
and we can actually do n.10 h and that's going to return the tanhd output of n
and now we should be able to draw it out of o not of n so let's see how that worked
there we go n went through 10 h to produce this output
so now tan h is a sort of our little micro grad supported node
here as an operation and as long as we know the derivative of
10h then we'll be able to back propagate through it now let's see this 10h in action currently it's not squashing too
much because the input to it is pretty low so if the bias was increased to say
eight then we'll see that what's flowing into the 10h now is
two and 10h is squashing it to 0.96 so we're already hitting the tail of this 10h and
it will sort of smoothly go up to 1 and then plateau out over there okay so now i'm going to do something slightly strange i'm going to change
this bias from 8 to this number 6.88 etc
and i'm going to do this for specific reasons because we're about to start back propagation
and i want to make sure that our numbers come out nice they're not like very crazy numbers they're nice numbers that
we can sort of understand in our head let me also add a pose label o is short for output here
so that's zero okay so 0.88 flows into 10 h comes out 0.7 so on
so now we're going to do back propagation and we're going to fill in all the gradients so what is the derivative o with respect
to all the inputs here and of course in the typical neural network setting what we really
care about the most is the derivative of these neurons on the weights specifically the w2 and w1 because those
are the weights that we're going to be changing part of the optimization and the other thing that we have to remember is here we have only a single
neuron but in the neural natives typically have many neurons and they're connected so this is only like a one small neuron
a piece of a much bigger puzzle and eventually there's a loss function that sort of measures the accuracy of the neural net and we're back propagating
with respect to that accuracy and trying to increase it so let's start off by propagation here
in the end what is the derivative of o with respect to o the base case sort of we know
always is that the gradient is just 1.0 so let me fill it in
and then let me split out the drawing function
here and then here cell
clear this output here okay so now when we draw o we'll see that oh
that grad is one so now we're going to back propagate through the tan h so to back propagate through 10h we need
to know the local derivative of 10h so if we have that
o is 10 h of n then what is d o by d n
now what you could do is you could come here and you could take this expression and you could do your calculus derivative taking
um and that would work but we can also just scroll down wikipedia here into a section that hopefully tells us
that derivative uh d by dx of 10 h of x is any of these i like this one 1 minus 10
h square of x so this is 1 minus 10 h of x squared
so basically what this is saying is that d o by d n is
1 minus 10 h of n squared
and we already have 10 h of n that's just o so it's one minus o squared
so o is the output here so the output is this number
data is this number and then
what this is saying is that do by dn is 1 minus this squared so
one minus of that data squared is 0.5 conveniently
so the local derivative of this 10 h operation here is 0.5
and so that would be d o by d n so we can fill in that in that grad
is 0.5 we'll just fill in
so this is exactly 0.5 one half so now we're going to continue the back propagation
this is 0.5 and this is a plus node so how is backprop going to what is that
going to do here and if you remember our previous example a plus is just a distributor of gradient
so this gradient will simply flow to both of these equally and that's because the local derivative of this operation
is one for every one of its nodes so 1 times 0.5 is 0.5
so therefore we know that this node here which we called this
its grad is just 0.5 and we know that b dot grad is also 0.5
so let's set those and let's draw so 0.5
continuing we have another plus 0.5 again we'll just distribute it so 0.5 will flow to both of these
so we can set theirs
x2w2 as well that grad is 0.5 and let's redraw pluses are my favorite
uh operations to back propagate through because it's very simple so now it's flowing into these
expressions is 0.5 and so really again keep in mind what the derivative is telling us at every point in time along
here this is saying that if we want the output of this neuron to increase
then the influence on these expressions is positive on the output both of them are
positive contribution to the output
so now back propagating to x2 and w2 first this is a times node so we know that the
local derivative is you know the other term so if we want to calculate x2.grad
then can you think through what it's going to be
so x2.grad will be w2.data times this x2w2
by grad right and w2.grad will be
x2 that data times x2w2.grad
right so that's the local piece of chain rule
let's set them and let's redraw so here we see that the gradient on our weight 2 is 0 because x2 data was 0
right but x2 will have the gradient 0.5 because data here was 1. and so what's interesting here right is
because the input x2 was 0 then because of the way the times works
of course this gradient will be zero and think about intuitively why that is derivative always tells us the influence
of this on the final output if i wiggle w2 how is the output changing
it's not changing because we're multiplying by zero so because it's not changing there's no derivative and zero is the correct
answer because we're squashing it at zero and let's do it here point five should
come here and flow through this times and so we'll have that x1.grad is
can you think through a little bit what what this should be
the local derivative of times with respect to x1 is going to be w1
so w1 is data times x1 w1 dot grad
and w1.grad will be x1.data times x1 w2 w1 with graph
let's see what those came out to be so this is 0.5 so this would be negative 1.5 and this would be 1.
and we've back propagated through this expression these are the actual final derivatives so if we want this neuron's
output to increase we know that what's necessary is that
w2 we have no gradient w2 doesn't actually matter to this neuron right now but this neuron this weight should uh go
up so if this weight goes up then this neuron's output would have gone up and
proportionally because the gradient is one okay so doing the back propagation manually is obviously ridiculous so we
implementing the backward function for each operation
are now going to put an end to this suffering and we're going to see how we can implement uh the backward pass a bit
more automatically we're not going to be doing all of it manually out here it's now pretty obvious to us by example
how these pluses and times are back property ingredients so let's go up to the value
object and we're going to start codifying what we've seen in the examples below
so we're going to do this by storing a special cell dot backward
and underscore backward and this will be a function which is going to do that little piece of chain rule at each
little node that compute that took inputs and produced output uh we're going to store
how we are going to chain the the outputs gradient into the inputs gradients
so by default this will be a function that uh doesn't do anything
so um and you can also see that here in the value in micrograb
so with this backward function by default doesn't do anything
this is an empty function and that would be sort of the case for example for a leaf node for leaf node there's nothing to do
but now if when we're creating these out values these out values are an addition
of self and other and so we will want to sell set
outs backward to be the function that propagates the gradient
so let's define what should happen
and we're going to store it in a closure let's define what should happen when we call outs grad
for in addition our job is to take outs grad and propagate it into self's
grad and other grad so basically we want to sell self.grad to something
and we want to set others.grad to something okay
and the way we saw below how chain rule works we want to take the local derivative times
the sort of global derivative i should call it which is the derivative of the final output of the expression with respect to
out's data with respect to out so
the local derivative of self in an addition is 1.0 so it's just 1.0 times
outs grad that's the chain rule and others.grad will be 1.0 times
outgrad and what you basically what you're seeing here is that outscrad will simply be copied onto selfs grad
and others grad as we saw happens for an addition operation so we're going to later call this
function to propagate the gradient having done an addition let's now do multiplication we're going
to also define that backward and we're going to set its backward to
be backward and we want to chain outgrad into
self.grad and others.grad
and this will be a little piece of chain rule for multiplication so we'll have so what should this be
can you think through
so what is the local derivative here the local derivative was others.data
and then oops others.data and the times of that grad that's channel
and here we have self.data times of that grad that's what we've been doing
and finally here for 10 h left backward
and then we want to set out backwards to be just backward
and here we need to back propagate we have out that grad and we want to chain it into self.grad
and salt.grad will be the local derivative of this operation that we've done here which is 10h
and so we saw that the local the gradient is 1 minus the tan h of x squared which here is t
that's the local derivative because that's t is the output of this 10 h so 1 minus t squared is the local derivative
and then gradient um has to be multiplied because of the chain rule so outgrad is chained through the local
gradient into salt.grad and that should be basically it so we're
going to redefine our value node we're going to swing all the way down here
and we're going to redefine our expression make sure that all the grads are zero
okay but now we don't have to do this manually anymore we are going to basically be calling the
dot backward in the right order so first we want to call os
dot backwards
so o was the outcome of 10h right so calling all that those who's
backward will be this function this is what it will do
now we have to be careful because there's a times out.grad
and out.grad remember is initialized to zero
so here we see grad zero so as a base case we need to set both.grad to 1.0
to initialize this with 1
and then once this is 1 we can call oda backward and what that should do is it should
propagate this grad through 10h so the local derivative times
the global derivative which is initialized at one so this should
um a dope
so i thought about redoing it but i figured i should just leave the error in here because it's pretty funny why is
anti-object not callable uh it's because i screwed up we're trying to save these
functions so this is correct this here we don't want to call the function
because that returns none these functions return none we just want to store the function so let me redefine the value object
and then we're going to come back in redefine the expression draw a dot everything is great o dot grad is one
o dot grad is one and now now this should work of course
okay so all that backward should this grant should now be 0.5 if we redraw and if everything went correctly
0.5 yay okay so now we need to call ns.grad
and it's not awkward sorry ends backward so that seems to have worked
so instead backward routed the gradient to both of these so this is looking great
now we could of course called uh called b grad beat up backwards sorry
what's gonna happen well b doesn't have it backward b is backward
because b is a leaf node b's backward is by initialization the empty function
so nothing would happen but we can call call it on it but when we call
this one it's backward
then we expect this 0.5 to get further routed right so there we go 0.5.5
and then finally we want to call it here on x2 w2
and on x1 w1
do both of those and there we go so we get 0 0.5 negative 1.5 and 1
exactly as we did before but now we've done it through calling that backward um
sort of manually so we have the lamp one last piece to get rid of which is us calling

这段内容讲解了如何**手动实现反向传播（manual backpropagation）**，并以“一个神经元（a neuron）”为例，详细说明了如何通过计算图传播梯度，最终为训练神经网络做好准备。以下是翻译和总结：

---

### 一个神经元的手动反向传播示例

我们将手动对一个“神经元”进行反向传播，这个神经元是构建神经网络的基本单元。在最简单的形式中，它属于多层感知机（MLP）的一部分，通常会包含两个层，中间是隐藏层，隐藏层中包含若干个“神经元”。

#### 一个神经元的数学模型：

* 输入 $x_1, x_2$（输入特征）
* 权重 $w_1, w_2$（连接强度）
* 偏置 $b$（神经元自身的触发偏好）
* 输出公式为：

  $$
  o = \tanh(x_1 w_1 + x_2 w_2 + b)
  $$

其中 $\tanh$ 是一个常用的激活函数，用于“压缩”输入范围到 \[-1, 1]。

---

### 实现步骤：

1. **前向传播：**

   * 计算 $x_1 \cdot w_1$，$x_2 \cdot w_2$
   * 将它们与偏置 $b$ 相加，得出中间结果 $n$
   * 应用激活函数：$o = \tanh(n)$

2. **可视化计算图：**

   * 构建并显示从输入到输出的完整计算流程图

3. **手动反向传播：**

   * 从输出 $o$ 开始反向传播，计算每个变量对最终输出的梯度（即偏导数）
   * 使用链式法则（chain rule）：

     * 例如：$\frac{\partial o}{\partial w_1} = \frac{\partial o}{\partial n} \cdot \frac{\partial n}{\partial w_1}$
   * 对不同操作（加法、乘法、tanh）使用相应的导数公式

4. **自动构建 `.backward()` 函数：**

   * 不再手动赋值 `grad`
   * 每种操作在定义时就绑定一个反向传播函数
   * 最后，只需在输出节点调用 `.backward()`，整个图中所有变量的梯度都会被正确计算并填入

---

### 示例结论：

通过这种方式：

* 你可以对任意复杂表达式进行反向传播
* 权重的 `grad` 就表示了如果你想增加输出，应该如何调整这些参数
* 这就是神经网络训练的核心步骤：用反向传播算出每个权重的影响力，然后朝着“提升模型输出”的方向去微调这些权重

---

如果你想，我可以：

* 给你画出这整个神经元计算图并标注所有梯度
* 继续解释 `.backward()` 如何在整个网络中自动化
* 进一步讲解如何从一个神经元扩展到完整的网络

需要吗？


# implementing the backward function for a whole expression graph
underscore backward manually so let's think through what we are actually doing um
we've laid out a mathematical expression and now we're trying to go backwards through that expression
um so going backwards through the expression just means that we never want to call a dot backward for any node
before we've done a sort of um everything after it
so we have to do everything after it before we're ever going to call that backward on any one node we have to get all of its full dependencies everything
that it depends on has to propagate to it before we can continue back propagation so this ordering of
graphs can be achieved using something called topological sort so topological sort
is basically a laying out of a graph such that all the edges go only from left to right basically
so here we have a graph it's a directory a cyclic graph a dag and this is two different topological
orders of it i believe where basically you'll see that it's laying out of the notes such that all the edges go only
one way from left to right and implementing topological sort you can look in wikipedia and so on i'm not
going to go through it in detail but basically this is what builds a
topological graph we maintain a set of visited nodes and
then we are going through starting at some root node
which for us is o that's where we want to start the topological sort and starting at o we go through all of
its children and we need to lay them out from left to right and basically this starts at o
if it's not visited then it marks it as visited and then it iterates through all of its children
and calls build topological on them and then uh after it's gone through all
the children it adds itself so basically this node that we're going to call it on
like say o is only going to add itself to the topo list after all of the
children have been processed and that's how this function is guaranteeing that you're only going to be in the list
once all your children are in the list and that's the invariant that is being maintained so if we built upon o and
then inspect this list we're going to see that it ordered our
value objects and the last one is the value of 0.707 which is the
output so this is o and then this is n and then all the other nodes get laid
out before it so that builds the topological graph and really what we're doing now is we're
just calling dot underscore backward on all of the nodes in a topological order
so if we just reset the gradients they're all zero what did we do we started by
setting o dot grad to b1 that's the base case
then we built the topological order and then we went for node
in reversed of topo now
in in the reverse order because this list goes from you know we need to go through it in reversed order
so starting at o note that backward and this should be
it there we go those are the correct derivatives
finally we are going to hide this functionality so i'm going to copy this and we're going to hide it
inside the valley class because we don't want to have all that code lying around so instead of an underscore backward
we're now going to define an actual backward so that's backward without the underscore
and that's going to do all the stuff that we just arrived so let me just clean this up a little bit so
we're first going to build a topological graph
starting at self so build topo of self
will populate the topological order into the topo list which is a local variable
then we set self.grad to be one and then for each node in the reversed
list so starting at us and going to all the children underscore backward
and that should be it so save
come down here redefine [Music] okay all the grands are zero
and now what we can do is oh that backward without the underscore and
there we go and that's uh that's back propagation
place for one neuron now we shouldn't be too happy with ourselves actually because we have a bad

当然！以下是这段内容的中文翻译：

---

### 实现整个表达式图的反向传播（`backward`）函数

现在，我们来思考一下我们到底在做什么。
我们已经构建了一个数学表达式，现在我们要对这个表达式进行反向传播。

反向传播的过程就是：
我们从表达式的输出开始，逐步计算每一个中间节点的梯度，
并且在调用某个节点的 `.backward()` 之前，**必须先计算所有它依赖的节点**。
这意味着，在反向传播过程中，我们需要确保从输出开始，按照正确的顺序反向遍历整个计算图。

#### ✅ 解决依赖关系：拓扑排序（Topological Sort）

为了确保按照正确的顺序遍历节点，我们可以使用**拓扑排序**来安排节点的遍历顺序。
**拓扑排序**确保所有边只从左到右依赖，即它会安排图中的节点，使得每个节点在其所有依赖节点之后被访问。

这是一个\*\*有向无环图（DAG）\*\*的例子，展示了两个不同的拓扑排序结果。
拓扑排序的关键是从输出节点（例如 `o`）开始，递归访问它的所有子节点，直到所有节点都被访问并按顺序排列。

---

#### ✅ 实现拓扑排序

拓扑排序的实现步骤如下：

* 维护一个`visited`节点集合，表示哪些节点已经被访问过。
* 从一个根节点（比如 `o`）开始，递归遍历其所有子节点。
* 每次处理一个节点时，先处理它的子节点，再将当前节点加入拓扑排序列表。
* 最终，拓扑排序保证了每个节点都会在它依赖的节点之后被处理。

---

#### ✅ 完成反向传播

在拓扑排序完成后，我们会得到一个按照顺序排列的节点列表（`topo`）。
然后我们会从**后往前**遍历这些节点，并对每个节点调用 `.backward()`，这样就能确保梯度按正确顺序传播。

代码实现：

1. **初始化：**

   * 重置所有节点的 `grad` 为零。
   * 设置输出节点 `o.grad = 1`，即反向传播的起点。
2. **构建拓扑图：**

   * 使用 `build_topo(self)` 函数生成拓扑排序。
3. **反向传播：**

   * 从拓扑排序的末尾开始，逐个调用节点的 `.backward()`，最终完成反向传播。

---

#### ✅ 把反向传播封装进 `Value` 类

为了避免代码冗长，我们将反向传播的逻辑封装进 `Value` 类的 `.backward()` 方法中。
这个方法会：

* 调用 `build_topo(self)` 构建拓扑排序。
* 设置 `self.grad = 1`，作为反向传播的起始点。
* 遍历拓扑排序列表，调用每个节点的 `._backward()`，完成反向传播。

---

#### ✅ 最终的反向传播实现

在实际代码中：

```python
def backward(self):
    topo = build_topo(self)
    self.grad = 1
    for node in reversed(topo):
        node._backward()
```

现在，反向传播已经成功实现，可以在神经网络的训练过程中自动计算梯度。

---

#### ✅ 总结

通过拓扑排序和链式法则，我们完成了反向传播的整个过程：

1. **拓扑排序**确保了节点按正确顺序计算。
2. **反向传播**通过遍历拓扑图，逐个计算每个节点的梯度。

在神经网络的训练中，这个过程对于更新权重、优化模型至关重要。

---

如果你需要，我可以：

* 继续解释如何通过 `.backward()` 实现更复杂的网络结构；
* 进一步优化这个反向传播过程，或是介绍如何应用在多层网络中。
  是否继续深入呢？


# fixing a backprop bug when one node is used multiple times
bug um and we have not surfaced the bug because of some specific conditions that we are we have to think about right now
so here's the simplest case that shows the bug say i create a single node a
and then i create a b that is a plus a and then i called backward
so what's going to happen is a is 3 and then a b is a plus a so there's two
arrows on top of each other here then we can see that b is of course the
forward pass works b is just a plus a which is six but the gradient here is not actually
correct that we calculate it automatically and that's because
um of course uh just doing calculus in your head the
derivative of b with respect to a should be uh two
one plus one it's not one intuitively what's happening here right so b is the result of a plus a and then
we call backward on it so let's go up and see what that does
um b is a result of addition so out as b and then when we called backward what
happened is self.grad was set to one and then other that grad was set to one
but because we're doing a plus a self and other are actually the exact same object
so we are overriding the gradient we are setting it to one and then we are setting it again to one and that's why
it stays at one so that's a problem there's another way to see this in a
little bit more complicated expression
so here we have a and b and then uh d will be the multiplication
of the two and e will be the addition of the two and then we multiply e times d to get f and
then we called fda backward and these gradients if you check will be incorrect
so fundamentally what's happening here again is basically we're going to see an issue
anytime we use a variable more than once until now in these expressions above every variable is used exactly once so
we didn't see the issue but here if a variable is used more than once what's going to happen during backward pass we're backpropagating from
f to e to d so far so good but now equals it backward and it deposits its
gradients to a and b but then we come back to d and call backward and it overwrites
those gradients at a and b so that's obviously a problem
and the solution here if you look at the multivariate case of the chain rule
and its generalization there the solution there is basically that we have to accumulate these gradients these
gradients add and so instead of setting those gradients
we can simply do plus equals we need to accumulate those gradients plus equals plus equals
plus equals plus equals and this will be okay remember because
we are initializing them at zero so they start at zero and then any
contribution that flows backwards will simply add
so now if we redefine this one because the plus equals this now works
because a.grad started at zero and we called beta backward we deposit one and
then we deposit one again and now this is two which is correct and here this will also work and we'll
get correct gradients because when we call eta backward we will deposit the gradients from this branch and then we get to back into
detail backward it will deposit its own gradients and then those gradients simply add on top of each other and so
we just accumulate those gradients and that fixes the issue okay now before we move on let me actually do a bit of
cleanup here and delete some of these some of this intermediate work so
we're not gonna need any of this now that we've derived all of it um we are going to keep this because i want
to come back to it delete the 10h delete our morning example
delete the step delete this keep the code that draws
and then delete this example and leave behind only the definition of value

当然，以下是这段内容的中文翻译：

---

### 修复反向传播中的错误：当一个节点被多次使用时

我们现在遇到一个反向传播的错误，**这个错误只有在某些特定情况下才会出现**。
我们现在就来讨论如何修复这个错误。

#### ✅ 错误示例：节点多次使用

假设我们有一个简单的例子：

1. 创建一个节点 `a`，然后创建一个节点 `b = a + a`。
2. 接着我们调用 `backward()`。

在前向传播中：

* `a = 3`，所以 `b = a + a = 6`，计算结果是正确的。
* 但是在计算梯度时，结果是错误的。

因为根据**微积分**，`b` 对 `a` 的导数应该是 2（即 $\frac{d}{da} (a + a) = 2$），
但是我们得到的梯度只有 1，这是因为我们**覆盖了梯度**。

---

#### ✅ 问题分析

问题的原因是这样的：

* `b` 是 `a + a` 的结果，调用 `backward()` 时，**`a` 被使用了两次**。
* 因为我们在计算梯度时，对 `a` 的梯度进行了两次赋值：第一次设置为 1，然后又一次设置为 1，
  这导致我们没有正确累积这些梯度，最终梯度仍然是 1。

---

#### ✅ 解决方法：累积梯度

**解决方法是累积这些梯度，而不是直接覆盖它们**。
我们可以使用 `+=` 来累积梯度，而不是简单的赋值 `=`。

* 我们将 `a.grad += 1`，而不是 `a.grad = 1`。
* 这样，在多次计算梯度时，梯度就会被加到原本的值上，而不是被覆盖。

---

#### ✅ 更复杂的示例：多个节点

在一个更复杂的例子中：

* `d = a * b`
* `e = a + b`
* `f = e * d`

然后我们调用 `f.backward()` 来计算梯度。
在这种情况下，问题同样出现在节点被多次使用时：

1. 我们从 `f` 反向传播到 `e` 和 `d`，然后再传播到 `a` 和 `b`。
2. 但是，由于 `a` 和 `b` 被多次使用，我们会在梯度计算过程中**覆盖之前的梯度**，导致错误。

---

#### ✅ 正确的解决方法

解决方法还是一样：**梯度累加**。
我们通过使用 `+=` 来保证每次计算的梯度会累加，而不是覆盖。

例如：

```python
a.grad += 1
b.grad += 1
```

这样，每个节点的梯度就会被正确地累加，并且最终得出正确的结果。

---

#### ✅ 代码清理

在解决了这个问题后，我将清理掉一些中间步骤和不再需要的代码，只保留最终需要的部分。
我们将删除一些临时变量和示例，只保留核心代码和 `Value` 类的定义。

---

### ✅ 总结：

* **问题**：当一个节点被多次使用时，梯度计算会覆盖之前的值，导致错误。
* **解决方案**：使用 `+=` 累加梯度，而不是简单的覆盖。
* **结果**：修复了这个问题后，计算出的梯度会正确地累加，得到正确的反向传播结果。

---

如果你有其他问题，或者想深入了解如何改进反向传播代码，欢迎继续提问！

# breaking up a tanh, exercising with more operations
and now let's come back to this non-linearity here that we implemented the tanh now i told you that we could
have broken down 10h into its explicit atoms in terms of other expressions if
we had the x function so if you remember tan h is defined like this and we chose to develop tan h as a single function
and we can do that because we know its derivative and we can back propagate through it but we can also break down tan h into
and express it as a function of x and i would like to do that now because i want to prove to you that you get all the
same results and all those ingredients but also because it forces us to implement a few more expressions it
forces us to do exponentiation addition subtraction division and things like that and i think it's a good exercise to
go through a few more of these okay so let's scroll up to the definition of value
and here one thing that we currently can't do is we can do like a value of say 2.0
but we can't do you know here for example we want to add constant one and we can't do something like this
and we can't do it because it says object has no attribute data that's because a plus one comes right here to
add and then other is the integer one and then here python is trying to access
one.data and that's not a thing and that's because basically one is not a value object and we only have addition
for value objects so as a matter of convenience so that we can create expressions like this and make them make
sense we can simply do something like this basically
we let other alone if other is an instance of value but if it's not an instance of value we're going to assume
that it's a number like an integer float and we're going to simply wrap it in in value and then other will just become
value of other and then other will have a data attribute and this should work so if i just say this predefined value then
this should work there we go okay now let's do the exact same thing for multiply because we can't
do something like this again for the exact same reason so we just have to go to mole and if other is
not a value then let's wrap it in value let's redefine value and now this works
now here's a kind of unfortunate and not obvious part a times two works we saw that but two times a is that gonna work
you'd expect it to right but actually it will not and the reason it won't is because python doesn't know
like when when you do a times two basically um so a times two python will
go and it will basically do something like a dot mul of two that's basically what it will
call but to it 2 times a is the same as 2 dot mol of a
and it doesn't 2 can't multiply value and so it's really confused about that
so instead what happens is in python the way this works is you are free to define something called the r mold
and our mole is kind of like a fallback so if python can't do 2 times a it will check if um
if by any chance a knows how to multiply two and that will be called into our mole
so because python can't do two times a it will check is there an our mole in value and because there is it will now
call that and what we'll do here is we will swap the order of the operands so basically
two times a will redirect to armel and our mole will basically call a times two and that's how that will work
so redefining now with armor two times a becomes four okay now looking at the
other elements that we still need we need to know how to exponentiate and how to divide so let's first the explanation to the exponentiation part we're going
to introduce a single function x here and x is going to mirror 10h in the
sense that it's a simple single function that transforms a single scalar value and outputs a single scalar value
so we pop out the python number we use math.x to exponentiate it create a new value object
everything that we've seen before the tricky part of course is how do you propagate through e to the x
and so here you can potentially pause the video and think about what should go here
okay so basically we need to know what is the local derivative of e to the x so
d by d x of e to the x is famously just e to the x and we've already just calculated e to the x and it's inside
out that data so we can do up that data times and out that grad that's the chain rule
so we're just chaining on to the current running grad and this is what the expression looks like it looks a little confusing but
this is what it is and that's the exponentiation so redefining we should now be able to
call a.x and hopefully the backward pass works as well okay and the last thing we'd like
to do of course is we'd like to be able to divide now i actually will implement something slightly more powerful than division
because division is just a special case of something a bit more powerful so in particular just by rearranging
if we have some kind of a b equals value of 4.0 here we'd like to basically be able to do a divide b and we'd like
this to be able to give us 0.5 now division actually can be reshuffled
as follows if we have a divide b that's actually the same as a multiplying one over b
and that's the same as a multiplying b to the power of negative one and so what i'd like to do instead is i
basically like to implement the operation of x to the k for some constant uh k so it's an integer or a
float um and we would like to be able to differentiate this and then as a special case uh negative one will be division
and so i'm doing that just because uh it's more general and um yeah you might as well do it that way so basically what
i'm saying is we can redefine uh division which we will put here somewhere
yeah we can put it here somewhere what i'm saying is that we can redefine division so self-divide other
can actually be rewritten as self times other to the power of negative one and now
a value raised to the power of negative one we have now defined that so here's
so we need to implement the pow function where am i going to put the power function maybe here somewhere
this is the skeleton for it so this function will be called when we try to raise a value to some power and
other will be that power now i'd like to make sure that other is only an int or a float usually other is
some kind of a different value object but here other will be forced to be an end or a float otherwise the math
won't work for for or try to achieve in the specific case that would be a different derivative expression if we wanted other
to be a value so here we create the output value which is just uh you know this data raised to
the power of other and other here could be for example negative one that's what we are hoping to achieve
and then uh this is the backwards stub and this is the fun part which is what is the uh chain rule expression here for
back for um back propagating through the power function where the power is to the power
of some kind of a constant so this is the exercise and maybe pause the video here and see if you can figure it out yourself as to what we should put
here
okay so you can actually go here and look at derivative rules as an example and we
see lots of derivatives that you can hopefully know from calculus in particular what we're looking for is the power rule
because that's telling us that if we're trying to take d by dx of x to the n which is what we're doing here
then that is just n times x to the n minus 1 right okay
so that's telling us about the local derivative of this power operation
so all we want here basically n is now other and self.data is x
and so this now becomes other which is n times
self.data which is now a python in torah float it's not a valley object we're accessing
the data attribute raised to the power of other minus one or n
minus one i can put brackets around this but this doesn't matter because
power takes precedence over multiply and python so that would have been okay and that's the local derivative only but
now we have to chain it and we change just simply by multiplying by output grad that's chain rule
and this should technically work and we're going to find out soon but now
if we do this this should now work and we get 0.5 so the forward pass works
but does the backward pass work and i realize that we actually also have to know how to subtract so
right now a minus b will not work to make it work we need one more
piece of code here and basically this is the
subtraction and the way we're going to implement subtraction is we're going to implement it by addition of a negation
and then to implement negation we're gonna multiply by negative one so just again using the stuff we've already built and just um expressing it in terms
of what we have and a minus b is now working okay so now let's scroll again to this expression here for this neuron
and let's just compute the backward pass here once we've defined o and let's draw it
so here's the gradients for all these leaf nodes for this two-dimensional neuron that has a 10h that we've seen
before so now what i'd like to do is i'd like to break up this 10h into this expression here
so let me copy paste this here and now instead of we'll preserve the label
and we will change how we define o so in particular we're going to implement this formula here
so we need e to the 2x minus 1 over e to the x plus 1. so e to the 2x we need to take 2 times n and we
need to exponentiate it that's e to the two x and then because we're using it twice let's create an intermediate
variable e and then define o as e plus one over
e minus one over e plus one e minus one over e plus one
and that should be it and then we should be able to draw that of o so now before i run this what do we
expect to see number one we're expecting to see a much longer graph here because we've broken up 10h
into a bunch of other operations but those operations are mathematically equivalent and so what we're expecting
to see is number one the same result here so the forward pass works and number two because of that
mathematical equivalence we expect to see the same backward pass and the same gradients on these leaf nodes so these
gradients should be identical so let's run this so number one let's verify that instead
of a single 10h node we have now x and we have plus we have times negative one
uh this is the division and we end up with the same forward pass here and then the gradients we have to be
careful because they're in slightly different order potentially the gradients for w2x2 should be 0 and 0.5
w2 and x2 are 0 and 0.5 and w1 x1 are 1 and negative 1.5
1 and negative 1.5 so that means that both our forward passes and backward passes were correct
because this turned out to be equivalent to 10h before and so the reason i wanted to go through
this exercise is number one we got to practice a few more operations and uh writing more backwards passes and number
two i wanted to illustrate the point that the um the level at which you implement your
operations is totally up to you you can implement backward passes for tiny expressions like a single individual
plus or a single times or you can implement them for say 10h
which is a kind of a potentially you can see it as a composite operation because it's made up of all these more atomic
operations but really all of this is kind of like a fake concept all that matters is we have some kind of inputs
and some kind of an output and this output is a function of the inputs in some way and as long as you can do forward pass and the backward pass of
that little operation it doesn't matter what that operation is and how composite it is
if you can write the local gradients you can chain the gradient and you can continue back propagation so the design
of what those functions are is completely up to you so now i would like to show you how you

当然！以下是这段内容的中文翻译：

---

### 拆解 `tanh` 函数，练习更多操作

现在让我们回到之前实现的非线性激活函数——`tanh`，我曾告诉过你，我们可以把 `tanh` 拆解成更基础的表达式，并通过其他运算来实现它。
如果我们拥有指数函数（`x`），我们可以将 `tanh` 拆解开来。我现在想要做这个，因为我想证明通过拆解实现和之前的方法得到的结果是相同的，而且这个过程还可以让我们实现一些额外的表达式，比如指数、加法、减法、除法等。

---

### 🧮 修改 `Value` 类支持更多运算

我们首先回到 `Value` 类的定义，目前我们能够做像 `Value(2.0)` 这样的操作，但是我们还不能做像 `1 + a` 这样的加法操作。

#### 错误原因：

当执行 `1 + a` 时，Python 会尝试访问 `1.data`，但是 `1` 不是一个 `Value` 对象，因此会抛出错误。
为了修复这个问题，我们可以修改 `Value` 类，使得如果 `other` 不是 `Value` 对象时，我们将它包装成 `Value` 对象，这样 `1 + a` 就能正确工作了。

#### 代码修复：

```python
if not isinstance(other, Value):
    other = Value(other)
```

这样就能确保当 `other` 不是 `Value` 时，我们将其转换成一个 `Value` 对象，这样后续的操作就可以正确执行了。

---

### ✖️ 乘法的修复：`2 * a` 和 `a * 2`

`a * 2` 是可以工作的，因为 Python 会调用 `a.mul(2)`，但是如果我们执行 `2 * a`，它会出错。
这是因为 Python 不知道如何处理 `2 * a`，它会试图去调用 `2.mul(a)`，但 `2` 不是一个 `Value` 对象，所以会失败。

为了解决这个问题，我们可以定义一个特殊的方法 `__rmul__`，它是一个“反向乘法”运算符。当 Python 遇到 `2 * a` 时，它会检查 `a` 是否实现了 `__rmul__` 方法，从而调用它来进行正确的操作。

```python
def __rmul__(self, other):
    return self.mul(other)
```

这样，`2 * a` 就会通过反向乘法调用 `a * 2`，从而正确工作。

---

### 💡 实现指数运算（Exponentiation）

接下来，我们需要实现**指数运算**，这将帮助我们支持类似 `e^x` 的操作。

我们定义一个新的函数 `x` 来实现指数运算：

```python
import math

def exp(self):
    return Value(math.exp(self.data))
```

对于 `e^x`，它的局部导数（local derivative）就是 `e^x`，我们可以通过链式法则来传播梯度。

---

### 🧮 除法的实现

接下来，我们要实现除法。实际上，除法是乘法的特例，所以我们可以将除法表达为乘以一个数的倒数：

$$
a \div b = a \times b^{-1}
$$

我们将实现一个通用的 `x^k` 操作来支持除法，其中 `k = -1` 表示倒数（除法）。

```python
def __pow__(self, other):
    if isinstance(other, Value):
        other = other.data
    return Value(self.data ** other)
```

这样，`a / b` 就可以通过 `a * b^-1` 来实现。

---

### ✅ 完整的操作实现：加法、乘法、指数、除法

通过上述方法，我们实现了加法、乘法、指数运算和除法运算，并且每个运算都有了对应的反向传播。

### 🧠 `tanh` 的拆解

我们现在来拆解 `tanh` 函数，将它转换为一系列基础的运算。
`tan h(x)` 可以表示为：

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

因此，我们可以拆解 `tanh(x)` 为多个步骤：

1. 计算 $e^{2x}$
2. 计算 $e^{2x} - 1$
3. 计算 $e^{2x} + 1$
4. 计算最终结果

这就是我们拆解 `tanh` 后的过程，它会变得更加复杂，但数学上是等价的。

---

### 🔄 反向传播：确保正确性

我们已经拆解了 `tanh`，并且通过这种方式重新实现了前向传播。
由于我们是基于**基本的数学运算**实现的，所以可以确保前向传播的结果和之前完全一致。

反向传播也会保持一致，因为我们拆解后的每个操作都有正确的局部梯度，并且通过链式法则逐步传播。

---

### ✅ 总结：

* **自定义运算**：我们通过对 `Value` 类的扩展，实现了加法、乘法、指数、除法等基本操作。
* **拆解 `tanh`**：将复杂的激活函数 `tanh` 拆解为基本的运算，保证数学等价性，并确保反向传播正确。
* **反向传播的精细控制**：通过实现每个操作的反向传播，我们保证了梯度的正确计算，并且可以实现更加复杂的表达式。

---

如果你有其他问题，或者想深入了解如何优化反向传播的实现，请继续提问！

# doing the same thing but in PyTorch: comparison
can do the exact same thing by using a modern deep neural network library like for example pytorch which i've roughly
modeled micrograd by and so pytorch is something you would use in production and i'll show you how you can
do the exact same thing but in pytorch api so i'm just going to copy paste it in and walk you through it a little bit
this is what it looks like so we're going to import pi torch and then we need to define these
value objects like we have here now micrograd is a scalar valued
engine so we only have scalar values like 2.0 but in pi torch everything is
based around tensors and like i mentioned tensors are just n-dimensional arrays of scalars
so that's why things get a little bit more complicated here i just need a scalar value to tensor a tensor with
just a single element but by default when you work with pytorch you would use um
more complicated tensors like this so if i import pytorch
then i can create tensors like this and this tensor for example is a two by three array
of scalar scalars in a single compact representation so we
can check its shape we see that it's a two by three array and so on so this is usually what you would work
with um in the actual libraries so here i'm creating a tensor that has only a single element
2.0 and then i'm casting it to be double
because python is by default using double precision for its floating point numbers so i'd like everything to be
identical by default the data type of these tensors will be float32 so it's
only using a single precision float so i'm casting it to double so that we have float64 just like in
python so i'm casting to double and then we get something similar to value of two the
next thing i have to do is because these are leaf nodes by default pytorch assumes that they do not require gradients so i need to explicitly say
that all of these nodes require gradients okay so this is going to construct scalar valued one element tensors
make sure that fighters knows that they require gradients now by default these are set to false by the way because of
efficiency reasons because usually you would not want gradients for leaf nodes like the inputs to the network and this
is just trying to be efficient in the most common cases so once we've defined all of our values
in python we can perform arithmetic just like we can here in microgradlend so this will just work and then there's a
torch.10h also and when we get back is a tensor again and we can
just like in micrograd it's got a data attribute and it's got grant attributes so these tensor objects just like in
micrograd have a dot data and a dot grad and the only difference here is that we need
to call it that item because otherwise um pi torch
that item basically takes a single tensor of one element and it just returns that element stripping out
the tensor so let me just run this and hopefully we are going to get this is going to print
the forward pass which is 0.707 and this will be the gradients which
hopefully are 0.5 0 negative 1.5 and 1. so if we just run this
there we go 0.7 so the forward pass agrees and then point five zero negative one point five
and one so pi torch agrees with us and just to show you here basically o
here's a tensor with a single element and it's a double and we can call that item on it to just
get the single number out so that's what item does and o is a tensor object like i mentioned and it's
got a backward function just like we've implemented and then all of these also have a dot graph so like x2 for example in the grad
and it's a tensor and we can pop out the individual number with that actin
so basically torches torch can do what we did in micrograph is a special case when your
tensors are all single element tensors but the big deal with pytorch is that everything is significantly more
efficient because we are working with these tensor objects and we can do lots of operations in parallel on all of
these tensors but otherwise what we've built very much agrees with the api of pytorch

当然，以下是这段关于 PyTorch 与我们自己手动实现的微型自动微分库（micrograd）对比的中文翻译：

---

### 使用 PyTorch 实现相同的功能：对比分析

我们可以使用一个现代的深度神经网络库，比如 **PyTorch**，来完成完全相同的事情。实际上，micrograd 的设计就是参考 PyTorch 的思路构建的。

PyTorch 是一个可用于生产环境的深度学习库。下面我会展示如何使用 PyTorch API 实现我们之前在 micrograd 中做的同样事情。

以下是代码的样子：

1. 首先导入 PyTorch。
2. 然后我们定义“值对象”（即张量，tensor），就像我们在 micrograd 中定义的那样。

---

### 🔢 PyTorch 中的一切都是张量（Tensor）

在 micrograd 中，我们只处理 **标量值**（比如 `2.0`），但在 PyTorch 中，一切都是张量——也就是 *n* 维的标量数组。

这会让事情稍微复杂一些。例如我们可以这样创建一个张量：

```python
t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

这个 `t` 是一个 2 行 3 列的矩阵（二维张量）。我们可以查看它的形状（`shape`）来确认：

```python
t.shape  # 输出：torch.Size([2, 3])
```

在实际开发中，PyTorch 中的操作主要是围绕这些多维张量进行的。

---

### 🌱 创建标量张量

为了和 micrograd 保持一致，我们在这里创建**只包含一个元素的张量**，也就是标量张量：

```python
a = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
```

* `dtype=torch.double`：把数据类型设置为 double（float64），确保和 Python 默认的一致。
* `requires_grad=True`：告诉 PyTorch：这个张量**需要计算梯度**。

注意：默认情况下，PyTorch 创建的张量不会计算梯度（`requires_grad=False`），因为出于效率考虑，通常你不需要对输入进行反向传播。

---

### ➕ 一切操作都能自动求导

和 micrograd 一样，一旦你定义了张量（即“值”），你就可以对它们进行运算，比如加法、乘法、激活函数 `torch.tanh()` 等。

运算的结果仍然是一个张量（tensor）。这些张量具有 `.data` 属性（值），也有 `.grad` 属性（梯度）。

要从只有一个元素的张量中获取标量值，可以使用：

```python
o.item()
```

---

### ✅ 示例运行效果

我们运行这段 PyTorch 代码，输出会是：

* 正向传播结果：`0.707`（也就是 `tanh` 的输出）
* 梯度值：

  ```
  0.5
  0
  -1.5
  1
  ```

这和我们在 micrograd 中计算出来的完全一致。

---

### 🧠 结论：PyTorch 和 micrograd 对比

* **本质一致**：PyTorch 实际上做的就是我们在 micrograd 中手动实现的东西，只不过它使用的是**更高效的底层张量计算**。
* **效率更高**：PyTorch 利用了高性能的张量计算（如 GPU 加速），能处理大规模并行操作。
* **功能更全**：PyTorch 不仅支持自动求导，还支持神经网络、优化器、损失函数等完整深度学习工具链。

---

如果你已经理解了 micrograd 的反向传播原理，那么学习 PyTorch 会变得非常自然和简单！如果你还想看 PyTorch 中更复杂的网络构建方式，也可以继续问我。

# building out a neural net library (multi-layer perceptron) in micrograd
okay so now that we have some machinery to build out pretty complicated mathematical expressions we can also start building out neural nets and as i
mentioned neural nets are just a specific class of mathematical expressions so we're going to start building out a
neural net piece by piece and eventually we'll build out a two-layer multi-layer layer perceptron as it's called and i'll
show you exactly what that means let's start with a single individual neuron we've implemented one here but
here i'm going to implement one that also subscribes to the pytorch api in how it designs its neural network
modules so just like we saw that we can like match the api of pytorch on the auto grad side we're going to try
to do that on the neural network modules so here's class neuron and just for the sake of efficiency i'm
going to copy paste some sections that are relatively straightforward so the constructor will take
number of inputs to this neuron which is how many inputs come to a neuron so this
one for example has three inputs and then it's going to create a weight there is some random number between
negative one and one for every one of those inputs and a bias that controls the overall
trigger happiness of this neuron and then we're going to implement a def underscore underscore call
of self and x some input x and really what we don't do here is w times x plus b
where w times x here is a dot product specifically now if you haven't seen
call let me just return 0.0 here for now the way this works now is we can have an x
which is say like 2.0 3.0 then we can initialize a neuron that is two-dimensional
because these are two numbers and then we can feed those two numbers into that neuron to get an output
and so when you use this notation n of x python will use call
so currently call just return 0.0
now we'd like to actually do the forward pass of this neuron instead so we're going to do here first is we
need to basically multiply all of the elements of w with all of the elements of x pairwise we need to multiply them
so the first thing we're going to do is we're going to zip up celta w and x and in python zip takes two iterators
and it creates a new iterator that iterates over the tuples of the corresponding entries
so for example just to show you we can print this list and still return 0.0 here
sorry so we see that these w's are paired up
with the x's w with x
and now what we want to do is
for w i x i in we want to multiply w times
w wi times x i and then we want to sum all of that together to come up with an activation
and add also subnet b on top so that's the raw activation and then of course we need to pass that through a
non-linearity so what we're going to be returning is act.10h and here's out
so now we see that we are getting some outputs and we get a different output from a neuron each time because we are
initializing different weights and by and biases and then to be a bit more efficient here actually sum by the way takes a second
optional parameter which is the start and by default the start is zero so
these elements of this sum will be added on top of zero to begin with but actually we can just start with cell dot
b and then we just have an expression like this
and then the generator expression here must be parenthesized in python there we go
yep so now we can forward a single neuron next up we're going to define a layer of neurons so here we have a
schematic for a mlb so we see that these mlps each layer
this is one layer has actually a number of neurons and they're not connected to each other but all of them are fully connected to the input
so what is a layer of neurons it's just it's just a set of neurons evaluated independently
so in the interest of time i'm going to do something fairly straightforward here
it's um literally a layer is just a list of neurons
and then how many neurons do we have we take that as an input argument here how many neurons do you want in your layer
number of outputs in this layer and so we just initialize completely independent neurons with this given
dimensionality and when we call on it we just independently evaluate them so now instead of a neuron
we can make a layer of neurons they are two-dimensional neurons and let's have three of them and now we see that we have three
independent evaluations of three different neurons right okay finally let's complete this picture
and define an entire multi-layer perceptron or mlp and as we can see here in an mlp these
layers just feed into each other sequentially so let's come here and i'm just going to copy the code here in interest of time
so an mlp is very similar we're taking the number of inputs as before but now instead of taking a
single n out which is number of neurons in a single layer we're going to take a list of an outs and this list defines
the sizes of all the layers that we want in our mlp so here we just put them all together and then iterate over consecutive pairs
of these sizes and create layer objects for them and then in the call function we are just calling them sequentially so that's
an mlp really and let's actually re-implement this picture so we want three input neurons
and then two layers of four and an output unit so we want
a three-dimensional input say this is an example input we want three inputs into
two layers of four and one output and this of course is an mlp
and there we go that's a forward pass of an mlp to make this a little bit nicer you see how we have just a single element but
it's wrapped in a list because layer always returns lists circle for convenience
return outs at zero if len out is exactly a single element else return fullest
and this will allow us to just get a single value out at the last layer that only has a single neuron
and finally we should be able to draw dot of n of x and as you might imagine
these expressions are now getting relatively involved so this is an entire mlp that we're
defining now
all the way until a single output okay and so obviously you would never
differentiate on pen and paper these expressions but with micrograd we will be able to back propagate all the way
through this and back propagate into these weights of all these neurons so
let's see how that works okay so let's create ourselves a very simple example data set here

当然，以下是这段关于使用 micrograd 构建神经网络（多层感知机 Multi-Layer Perceptron, MLP）的内容翻译：

---

### 🧠 用 micrograd 构建神经网络库（多层感知机）

现在我们已经有了一套构建复杂数学表达式的工具，我们也可以开始一步步构建**神经网络**。神经网络其实就是一类特定形式的数学表达式。

我们最终要实现一个 **两层的多层感知机（MLP）**，现在我们一步步来实现它。

---

### ✅ 实现一个单独的神经元（Neuron）

我们之前已经实现了一个神经元，但这里我们将模仿 PyTorch 的 API 风格来实现它。

#### 🔧 `Neuron` 类的构造方法（`__init__`）：

* 接受一个参数 `nin`：代表这个神经元的输入维度（也就是输入的数量）。
* 为每个输入创建一个对应的**权重 `w`**（初始为 -1 到 1 的随机数）。
* 还创建一个**偏置 `b`**，控制这个神经元是否容易激活（“触发倾向”）。

#### 🚀 实现前向传播（`__call__` 方法）：

* 输入是 `x`，一个向量（列表）。
* 对每个 `w` 和 `x` 成对相乘（点积 dot product），然后加上偏置 `b`。
* 最后通过一个非线性函数（例如 `tanh`）进行激活，得到输出。

> 这样你就可以直接写 `n(x)` 来获得神经元的输出，就像在用 PyTorch 一样。

---

### 🔗 实现一层神经元（Layer）

在一个 MLP 中，每一层都由多个神经元组成，这些神经元彼此之间**没有连接**，但**都连接到输入**。

#### 🧱 `Layer` 类：

* 接收两个参数：输入维度 `nin` 和输出维度（神经元数量）`nout`。
* 本质上就是一个 `Neuron` 的列表。
* 调用时会对每个神经元独立进行前向计算，得到一组输出。

例如：

```python
layer = Layer(2, 3)
output = layer([1.0, 2.0])  # 输出是三个神经元的结果
```

---

### 🏗️ 实现多层感知机（MLP）

一个 **MLP** 由多层堆叠的 `Layer` 组成，层与层之间是顺序连接的。

#### 🧱 `MLP` 类：

* 接收输入维度 `nin`，以及一个列表 `nouts` 表示每一层的神经元数量。
* 通过成对读取这个列表，构建多个 `Layer` 对象。
* 前向传播时顺序调用每一层的输出作为下一层的输入。

例如：

```python
mlp = MLP(3, [4, 4, 1])  # 输入是 3 维，两层 4 个神经元，最后输出 1 个值
out = mlp([1.0, 2.0, 3.0])
```

为了方便，如果最后一层输出只有一个值，就直接返回这个值，而不是单元素列表。

---

### 📈 绘制神经网络的计算图

我们也可以用 `draw_dot()` 函数将整个神经网络表达式可视化，虽然图会很大。

这张图展示了：

* 所有神经元的权重和偏置；
* 所有层之间的数据流动；
* 最终的单个输出。

---

### 💡 总结

我们刚刚用 micrograd 实现了：

* 单个神经元；
* 神经元组成的层；
* 多层感知机（MLP）。

这些模块现在已经可以：

* 进行前向传播；
* 借助 micrograd 的自动求导系统，进行**反向传播**训练权重。

接下来就可以用这些模块构建一个完整的数据集进行训练了！

如需继续，可以问我如何用这个 MLP 训练模型、添加损失函数、做梯度下降等。

这一行代码的作用是：**为一个神经元创建多个随机初始化的权重参数**，每个权重是一个支持自动微分的 `Value` 对象。

---

我们来一步步解释：

```python
self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
```

### ✅ 含义解释

| 部分                           | 作用说明                                              |
| ---------------------------- | ------------------------------------------------- |
| `random.uniform(-1, 1)`      | 从区间 $[-1, 1]$ 中随机取一个小数，用作**初始化权重值**。              |
| `Value(...)`                 | 包装成一个 `Value` 类型对象，**支持自动求导**（autograd），能反向传播求梯度。 |
| `for _ in range(nin)`        | 循环 `nin` 次（即输入维度数），表示这个神经元有多少个输入，就要有多少个权重。        |
| `[ ... for _ in range(nin)]` | 生成一个列表，里面有 `nin` 个 `Value(...)` 权重。               |

例如：

```python
nin = 3
# 假设随机出来的是：0.6, -0.1, 0.8
self.w = [Value(0.6), Value(-0.1), Value(0.8)]
```

---

### 🧠 为什么要这样做？

这是在模拟一个**生物神经元的突触权重**（synaptic weights）：

* 如果神经元有 3 个输入（比如 x1, x2, x3），它就需要 3 个对应的权重 w1, w2, w3。
* 这些权重是训练中要优化的参数，因此要随机初始化，并且需要支持梯度传播。

---

### ➕ `self.b = Value(...)` 是什么？

这表示 **偏置项**（bias）：

* 它也是一个需要训练的参数。
* 它的作用是调整神经元的“激活倾向”，让输出不完全受输入控制。

---

如你需要，我可以画一个图展示这个神经元结构。是否需要？



# creating a tiny dataset, writing the loss function
so this data set has four examples and so we have four possible inputs into the neural net
and we have four desired targets so we'd like the neural net to assign
or output 1.0 when it's fed this example negative one when it's fed these examples and one when it's fed this
example so it's a very simple binary classifier neural net basically that we would like here
now let's think what the neural net currently thinks about these four examples we can just get their predictions
um basically we can just call n of x for x in axis and then we can
print so these are the outputs of the neural net on those four examples
so the first one is 0.91 but we'd like it to be one so we should push this one
higher this one we want to be higher this one says 0.88 and we want this to
be negative one this is 0.8 we want it to be negative one and this one is 0.8 we want it to be one
so how do we make the neural net and how do we tune the weights to better predict the desired targets
and the trick used in deep learning to achieve this is to calculate a single number that somehow
measures the total performance of your neural net and we call this single number the loss
so the loss first is is a single number that we're going to define that basically measures how
well the neural net is performing right now we have the intuitive sense that it's not performing very well because we're not very much close to this
so the loss will be high and we'll want to minimize the loss so in particular in this case what we're
going to do is we're going to implement the mean squared error loss so this is doing is we're going to
basically iterate um for y ground truth
and y output in zip of um wise and white red so we're going to
pair up the ground truths with the predictions and this zip iterates over tuples of
them and for each y ground truth and y output we're going
to subtract them and square them
so let's first see what these losses are these are individual loss components and so basically for each
one of the four we are taking the prediction and the ground truth we are subtracting them and
squaring them so because this one is so close to its target 0.91
is almost one subtracting them gives a very small number
so here we would get like a negative point one and then squaring it just makes sure
that regardless of whether we are more negative or more positive we always get a positive
number instead of squaring we should we could also take for example the absolute value we need to discard the sign
and so you see that the expression is ranged so that you only get zero exactly when y out is equal to y ground truth
when those two are equal so your prediction is exactly the target you are going to get zero and if your prediction is not the target
you are going to get some other number so here for example we are way off and so that's why the loss is quite high
and the more off we are the greater the loss will be so we don't want high loss we want low
loss and so the final loss here will be just the sum
of all of these numbers so you see that this should be zero roughly plus zero roughly
but plus seven so loss should be about seven
here and now we want to minimize the loss we want the loss to be low
because if loss is low then every one of the predictions is equal to its target
so the loss the lowest it can be is zero and the greater it is the worse off the
neural net is predicting so now of course if we do lost that backward
something magical happened when i hit enter and the magical thing of course that happened is that we can look at
end.layers.neuron and that layers at say like the the first layer that neurons at zero
because remember that mlp has the layers which is a list and each layer has a neurons which is a
list and that gives us an individual neuron and then it's got some weights and so we can for example look at the
weights at zero um
oops it's not called weights it's called w and that's a value but now this value
also has a groud because of the backward pass and so we see that because this gradient
here on this particular weight of this particular neuron of this particular layer is negative we see that its influence on the loss is
also negative so slightly increasing this particular weight of this neuron of this layer would make the loss go down
and we actually have this information for every single one of our neurons and all their parameters actually it's worth looking at also the draw dot loss by the
way so previously we looked at the draw dot of a single neural neuron forward pass and that was already a large expression
but what is this expression we actually forwarded every one of those four examples and
then we have the loss on top of them with the mean squared error and so this is a really massive graph
because this graph that we've built up now oh my gosh this graph that we've built
up now which is kind of excessive it's excessive because it has four forward passes of a neural net for every one of
the examples and then it has the loss on top and it ends with the value of the loss which was 7.12
and this loss will now back propagate through all the four forward passes all the way through just every single
intermediate value of the neural net all the way back to of course the parameters of the weights which are the
input so these weight parameters here are inputs to this neural net
and these numbers here these scalars are inputs to the neural net so if we went around here
we'll probably find some of these examples this 1.0 potentially maybe this 1.0 or you know
some of the others and you'll see that they all have gradients as well the thing is these gradients on the
input data are not that useful to us and that's because the input data seems
to be not changeable it's it's a given to the problem and so it's a fixed input we're not going to be changing it or
messing with it even though we do have gradients for it but some of these gradients here
will be for the neural network parameters the ws and the bs and those we of course we want to change
okay so now we're going to want some convenience code to gather up all of the parameters of the neural net so that we
collecting all of the parameters of the neural net
can operate on all of them simultaneously and every one of them we will nudge a tiny amount
based on the gradient information so let's collect the parameters of the neural net all in one array
so let's create a parameters of self that just returns celta w which is a list
concatenated with a list of self.b so this will just return a list
list plus list just you know gives you a list so that's parameters of neuron and i'm
calling it this way because also pi torch has a parameters on every single and in module
and uh it does exactly what we're doing here it just returns the parameter tensors for us as the
parameter scalars now layer is also a module so it will have parameters
itself and basically what we want to do here is something like this like
params is here and then for neuron in salt out neurons
we want to get neuron.parameters and we want to params.extend
right so these are the parameters of this neuron and then we want to put them on top of params so params dot extend
of peace and then we want to return brands so this is way too much code so actually
there's a way to simplify this which is return
p for neuron in self neurons
for p in neuron dot parameters
so it's a single list comprehension in python you can sort of nest them like this and you can um
then create uh the desired array so this is these are identical
we can take this out and then let's do the same here
def parameters self and return a parameter for layer in self dot layers
for p in layer dot parameters
and that should be good now let me pop out this so
we don't re-initialize our network because we need to re-initialize our
okay so unfortunately we will have to probably re-initialize the network because we just add functionality
because this class of course we i want to get all the and that parameters but that's not going to work because this is
the old class okay so unfortunately we do have to reinitialize the network which will
change some of the numbers but let me do that so that we pick up the new api we can now do in the
parameters and these are all the weights and biases inside the entire neural net
so in total this mlp has 41 parameters
and now we'll be able to change them if we recalculate the loss here we see

以下是这段代码的中文翻译：

---

### 创建一个简单的数据集，编写损失函数

我们现在有一个包含四个样本的数据集，这意味着我们有四个可能的输入和四个期望的目标输出。我们的目标是让神经网络在输入时输出以下目标：

* 当输入为第一个样本时，输出为 1.0；
* 当输入为第二和第三个样本时，输出为 -1.0；
* 当输入为第四个样本时，输出为 1.0。

这实际上是一个非常简单的二分类问题，我们希望神经网络能够预测这些目标。

接下来，让我们看看神经网络当前对这四个样本的预测。我们可以通过以下方式获取它们的预测结果：

```python
predictions = [n(x) for x in inputs]
```

然后，我们打印出神经网络对这些样本的预测结果。假设我们得到以下输出：

* 第一个预测为 0.91，但我们希望它是 1.0；
* 第二个预测为 0.88，我们希望它是 -1.0；
* 第三个预测为 0.8，我们希望它是 -1.0；
* 第四个预测为 0.8，我们希望它是 1.0。

### 计算损失函数

为了使神经网络的输出更接近目标，我们需要调整神经网络的权重。为此，我们需要计算一个度量网络性能的单一数值，这个数值叫做 **损失**。

**损失** 是用来衡量神经网络当前预测与目标值之间差距的数值。理想情况下，损失值应为 0，表示预测值完全等于目标值。

我们这里使用 **均方误差损失（Mean Squared Error, MSE）**。该损失函数通过以下步骤计算：

* 我们通过 `zip` 将目标值（ground truth）与预测值（output）配对；
* 对每对目标和预测，我们计算它们之间的差的平方。

具体实现如下：

```python
loss = sum([(y_true - y_pred) ** 2 for y_true, y_pred in zip(targets, predictions)])
```

这样，我们就得到了每个预测和目标之间的差异，并通过平方操作来消除正负号。

* 如果预测与目标完全一致，损失为 0；
* 如果预测与目标相差较大，损失值较高。

### 反向传播

为了调整网络的权重，使损失最小化，我们需要使用反向传播。通过调用 `.backward()`，我们能够看到每个神经元的梯度，这些梯度告诉我们每个权重如何影响最终的损失。

当我们调用反向传播后，神经网络的每个权重和偏置都会有一个与损失相关的梯度。这个梯度告诉我们如果稍微改变这个权重或偏置，损失会如何变化。

例如，如果某个权重的梯度是负数，那么我们知道增加该权重会导致损失减少。因此，我们可以通过调整这些权重来最小化损失。

### 参数收集与更新

为了高效地更新所有权重和偏置，我们需要收集神经网络中的所有参数。PyTorch 等框架通常会将这些参数封装在一个可迭代的对象中，方便我们进行统一操作。

我们可以使用如下方法收集所有参数：

```python
def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
```

对于多层感知机（MLP），我们可以为每一层收集它的参数，最后统一返回：

```python
def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

现在我们可以访问并更新所有神经网络中的权重和偏置。

### 重新初始化神经网络

由于我们对 `MLP` 类进行了修改，需要重新初始化网络。这样我们就可以使用新的 API 来访问和更新网络中的所有权重和偏置。

---

### 总结

* **损失函数**：我们计算均方误差（MSE），这是一个衡量神经网络输出与目标值差异的标准方法。
* **反向传播**：通过 `.backward()`，我们计算每个权重的梯度，以便更新它们，使损失最小化。
* **参数收集**：通过编写 `parameters()` 方法，我们将神经网络的所有参数收集到一个列表中，方便统一管理和更新。
* **权重更新**：通过计算损失的梯度，我们可以使用这些信息来调整神经网络的权重，以提高预测的准确性。

这样，我们就实现了一个简单的神经网络训练流程。

讲解一下python的List Comprehension

Python 中的 **List Comprehension（列表推导式）** 是一种简洁且高效的构建列表的方式。它可以用一行代码生成一个新的列表，代替传统的 `for` 循环和 `append()` 方法。列表推导式语法结构简洁，具有更高的可读性和执行效率。

### 基本语法

```python
[expression for item in iterable]
```

* **`expression`**：对每个 `item` 执行的操作，最终的结果会组成新的列表。
* **`item`**：遍历 `iterable` 中的每个元素。
* **`iterable`**：可以是任何可迭代对象，如列表、元组、字符串等。

### 示例 1：生成数字列表

传统方式：

```python
squares = []
for x in range(5):
    squares.append(x ** 2)
```

使用列表推导式：

```python
squares = [x ** 2 for x in range(5)]
```

这段代码会生成一个包含数字 0 到 4 的平方值的列表 `[0, 1, 4, 9, 16]`。

### 示例 2：过滤列表

列表推导式还可以结合 `if` 语句来过滤数据。

传统方式：

```python
even_numbers = []
for x in range(10):
    if x % 2 == 0:
        even_numbers.append(x)
```

使用列表推导式：

```python
even_numbers = [x for x in range(10) if x % 2 == 0]
```

这个例子会生成一个包含从 0 到 9 中所有偶数的列表 `[0, 2, 4, 6, 8]`。

### 示例 3：嵌套列表推导式

可以在列表推导式中使用嵌套循环。

传统方式：

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = []
for row in matrix:
    for num in row:
        flattened.append(num)
```

使用列表推导式：

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
```

此代码会将二维矩阵 `matrix` 扁平化为一维列表 `[1, 2, 3, 4, 5, 6, 7, 8, 9]`。

### 示例 4：应用函数

可以在列表推导式中使用函数来处理每个元素。

```python
words = ['apple', 'banana', 'cherry']
capitalized_words = [word.upper() for word in words]
```

此代码会将每个单词转换为大写字母，返回列表 `['APPLE', 'BANANA', 'CHERRY']`。

### 复杂用法：列表推导式的多条件

可以在列表推导式中使用多个 `if` 条件。

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
result = [x for x in numbers if x % 2 == 0 if x > 4]
```

这段代码会返回所有大于 4 且为偶数的数字 `[6, 8]`。

### 总结：

* **列表推导式** 提供了一种简洁的方式来创建新的列表，通常代替传统的 `for` 循环。
* 它的语法结构为 `[expression for item in iterable if condition]`，非常方便用来执行转换、过滤等操作。
* **嵌套循环**、**条件语句**等可以组合使用，使得其功能非常强大。
* 列表推导式通常比 `for` 循环更简洁且执行效率更高。


# doing gradient descent optimization manually, training the network
that unfortunately we have slightly different predictions and slightly different laws
but that's okay okay so we see that this neurons gradient is slightly negative we can
also look at its data right now which is 0.85 so this is the current
value of this neuron and this is its gradient on the loss so what we want to do now is we want to
iterate for every p in n dot parameters so for all the 41 parameters in this neural net
we actually want to change p data slightly according to the gradient information
okay so dot dot to do here but this will be basically a tiny update
in this gradient descent scheme in gradient descent we are thinking of the
gradient as a vector pointing in the direction of increased
loss and so in gradient descent we are modifying
p data by a small step size in the direction of the gradient so the step size as an
example could be like a very small number like 0.01 is the step size times p dot grad
right but we have to think through some of the signs here so uh
in particular working with this specific example here we see that if we just left it like this
then this neuron's value would be currently increased by a tiny amount of the gradient
the grain is negative so this value of this neuron would go slightly down it would become like 0.8 you know four or
something like that but if this neuron's value goes lower
that would actually increase the loss that's because
the derivative of this neuron is negative so increasing this makes the loss go down so
increasing it is what we want to do instead of decreasing it so basically what we're missing here is we're
actually missing a negative sign and again this other interpretation and that's because we want to minimize
the loss we don't want to maximize the loss we want to decrease it and the other interpretation as i mentioned is you can think of the
gradient vector so basically just the vector of all the gradients as pointing in the direction of
increasing the loss but then we want to decrease it so we actually want to go in the opposite direction
and so you can convince yourself that this sort of plug does the right thing here with the negative because we want to minimize the loss
so if we nudge all the parameters by tiny amount
then we'll see that this data will have changed a little bit so now this neuron
is a tiny amount greater value so 0.854 went to 0.857
and that's a good thing because slightly increasing this neuron uh
data makes the loss go down according to the gradient and so the correct thing has happened sign wise
and so now what we would expect of course is that because we've changed all these parameters we expect that the loss
should have gone down a bit so we want to re-evaluate the loss let me basically
this is just a data definition that hasn't changed but the forward pass here of the network we can recalculate
and actually let me do it outside here so that we can compare the two loss values so here if i recalculate the loss
we'd expect the new loss now to be slightly lower than this number so hopefully what we're getting now is a
tiny bit lower than 4.84 4.36 okay and remember the way we've arranged
this is that low loss means that our predictions are matching the targets so our predictions now are probably
slightly closer to the targets and now all we have to do is we
have to iterate this process so again um we've done the forward pass and this is the loss
now we can lost that backward let me take these out and we can do a step size
and now we should have a slightly lower loss 4.36 goes to 3.9
and okay so we've done the forward pass here's the backward pass nudge
and now the loss is 3.66 3.47
and you get the idea we just continue doing this and this is uh gradient descent we're just iteratively doing
forward pass backward pass update forward pass backward pass update and the neural net is improving its
predictions so here if we look at why pred now
like red we see that um this value should be getting closer to
one so this value should be getting more positive these should be getting more negative and this one should be also getting more positive so if we just
iterate this a few more times actually we may be able to afford go to
go a bit faster let's try a slightly higher learning rate
oops okay there we go so now we're at 0.31 if you go too fast by the way if you try
to make it too big of a step you may actually overstep
it's overconfidence because again remember we don't actually know exactly about the loss function the loss function has all kinds of structure and
we only know about the very local dependence of all these parameters on the loss but if we step too far
we may step into you know a part of the loss that is completely different and that can destabilize training and
make your loss actually blow up even so the loss is now 0.04 so actually the
predictions should be really quite close let's take a look so you see how this is almost one
almost negative one almost one we can continue going uh so
yep backward update oops there we go so we went way too fast
and um we actually overstepped so we got two uh too eager where are we
now oops okay seven e negative nine so this is very very low loss
and the predictions are basically perfect so somehow we
basically we were doing way too big updates and we briefly exploded but then somehow we ended up getting into a really good spot so usually this
learning rate and the tuning of it is a subtle art you want to set your learning rate if it's too low you're going to
take way too long to converge but if it's too high the whole thing gets unstable and you might actually even
explode the loss depending on your loss function so finding the step size to be just
right it's it's a pretty subtle art sometimes when you're using sort of vanilla gradient descent
but we happen to get into a good spot we can look at n-dot parameters
so this is the setting of weights and biases that makes our network
predict the desired targets very very close and
basically we've successfully trained neural net okay let's make this a tiny bit more respectable and implement an actual
training loop and what that looks like so this is the data definition that stays this is the forward pass
um so for uh k in range you know we're going to
take a bunch of steps first you do the forward pass
we validate the loss let's re-initialize the neural net from scratch
and here's the data and we first do before pass then we do
the backward pass
and then we do an update that's gradient descent
and then we should be able to iterate this and we should be able to print the current step the current loss um let's just print the
sort of number of the loss and that should be it
and then the learning rate 0.01 is a little too small 0.1 we saw is like a little bit dangerously too high let's go
somewhere in between and we'll optimize this for not 10 steps but let's go for say 20
steps let me erase all of this junk
and uh let's run the optimization and you see how we've actually converged
slower in a more controlled manner and got to a loss that is very low
so i expect white bread to be quite good there we go
um and that's it okay so this is kind of embarrassing but
we actually have a really terrible bug in here and it's a subtle bug and it's a
very common bug and i can't believe i've done it for the 20th time in my life
especially on camera and i could have reshot the whole thing but i think it's pretty funny and you know you get to
appreciate a bit what um working with neural nets maybe is like sometimes
we are guilty of come bug i've actually tweeted
the most common neural net mistakes a long time ago now uh and
i'm not really gonna explain any of these except for we are guilty of number three you forgot to
zero grad before that backward what is that
basically what's happening and it's a subtle bug and i'm not sure if you saw it is that all of these
weights here have a dot data and a dot grad and that grad starts at zero
and then we do backward and we fill in the gradients and then we do an update on the data but
we don't flush the grad it stays there so when we do the second
forward pass and we do backward again remember that all the backward operations do a plus equals on the grad
and so these gradients just add up and they never get reset to zero
so basically we didn't zero grad so here's how we zero grad before
backward we need to iterate over all the parameters and we need to make sure that p dot grad
is set to zero we need to reset it to zero just like it is in the constructor
so remember all the way here for all these value nodes grad is reset to zero and then all these backward passes do a
plus equals from that grad but we need to make sure that we reset these graphs to zero so that
when we do backward all of them start at zero and the actual backward pass accumulates um
the loss derivatives into the grads so this is zero grad in pytorch
and uh we will slightly get we'll get a slightly different optimization let's reset the neural net
the data is the same this is now i think correct and we get a much more
you know we get a much more slower descent we still end up with pretty good results
and we can continue this a bit more to get down lower and lower
and lower yeah so the only reason that the previous
thing worked it's extremely buggy um the only reason that worked is that
this is a very very simple problem and it's very easy for this neural net to fit this data
and so the grads ended up accumulating and it effectively gave us a massive step size and it made us converge
extremely fast but basically now we have to do more steps to get to very low values of loss
and get wipe red to be really good we can try to step a bit greater
yeah we're gonna get closer and closer to one minus one and one so
working with neural nets is sometimes tricky because uh
you may have lots of bugs in the code and uh your network might actually work just like ours worked
but chances are is that if we had a more complex problem then actually this bug would have made us not optimize the loss
very well and we were only able to get away with it because the problem is very simple

以下是这段代码的中文翻译：

---

### 手动进行梯度下降优化，训练网络

我们注意到，预测结果和损失值稍有不同，但这没有关系。

首先，我们看到了这个神经元的梯度是负的。我们也可以查看它的当前数据值，0.85，这就是当前神经元的值，而这个值的梯度是关于损失的。接下来，我们希望遍历所有参数，并根据梯度信息稍微调整每个参数的数据。

```python
for p in n.parameters():
    p.data -= step_size * p.grad
```

在梯度下降法中，梯度向量指向损失增加的方向。因此，在梯度下降中，我们会稍微按照梯度的方向修改参数数据。比如，步长（step size）可以是一个很小的数值，比如 `0.01`，然后乘以 `p.grad`。

需要注意的是，当前这个神经元的值是0.85，而它的梯度是负数。如果我们直接根据这个梯度调整，它的值会稍微下降，这可能使得损失增加。因此我们实际上需要在更新时加上一个负号，这样才能减少损失。我们想要的是最小化损失，而不是最大化损失。

因此，我们在调整权重时会加上负号，因为我们希望减少损失。稍微调整这些参数后，神经元的值将会增加，变为 0.857，这使得损失降低。

接下来，我们希望通过重新计算损失来验证这一点。我们可以重新执行前向传播并验证损失值是否降低了。比如，新的损失值应该略低于 4.84（之前的损失值），比如 4.36。

### 训练过程中的迭代

在梯度下降过程中，我们会不断地进行前向传播、反向传播和更新。每次更新后，损失值应该会逐渐减小。例如，经过一次更新后，损失从 4.84 降到 4.36，之后再降到 3.9，最终接近 0。

```python
# 假设我们设置了合适的步长
step_size = 0.01
```

### 学习率的选择

学习率的选择非常重要。如果步长太小，优化会非常缓慢；如果步长太大，可能会导致训练不稳定，甚至损失值爆炸。为了避免这个问题，我们需要找到一个合适的学习率。

例如，使用较高的学习率会导致我们快速收敛，但如果设置得过高，可能会导致梯度更新过大，反而使训练不稳定，损失值可能爆炸。我们发现，如果过度更新，损失可能会从极低值（0.04）变得非常高，这表明训练过程出现了不稳定的情况。

### 解决梯度累积问题

有一个常见的 bug，通常是由于在反向传播之前没有清空梯度（`zero_grad`）。如果不清除梯度，梯度会在每次反向传播时累积，导致梯度值不正确。

```python
# 在每次反向传播之前，我们必须手动清空梯度
for p in self.parameters():
    p.grad = 0
```

这种错误的原因是每次调用 `.backward()` 时，梯度会加到之前的梯度上，而没有重置它们。这就是我们需要在每次反向传播之前清除旧的梯度。

### 正确的训练过程

在修复了这个 bug 后，我们可以重新初始化神经网络并使用更合理的学习率进行训练。随着训练的进行，损失逐渐下降，最终我们能够训练出一个性能不错的神经网络。

例如，通过调整学习率和步骤数，我们能够将损失降到一个非常低的值，接近 0，这时我们的预测结果也非常接近目标值。

### 总结

* **梯度下降**：通过计算梯度并根据梯度调整神经网络的权重，最小化损失。
* **学习率**：合适的学习率对于稳定的训练至关重要，步长过大或过小都会影响训练效果。
* **梯度累积问题**：在每次反向传播之前清除梯度，避免梯度累积导致的问题。
* **训练过程的迭代**：通过不断迭代前向传播、反向传播和更新，逐步优化神经网络的参数。

这个过程虽然看起来简单，但在实际应用中，优化和调试神经网络时可能会遇到许多挑战，尤其是处理更复杂的任务时。

# summary of what we learned, how to go towards modern neural nets
so let's now bring everything together and summarize what we learned what are neural nets neural nets are
these mathematical expressions fairly simple mathematical expressions in the case of multi-layer perceptron
that take input as the data and they take input the weights and the parameters of the
neural net mathematical expression for the forward pass followed by a loss function and the loss function tries to
measure the accuracy of the predictions and usually the loss will be low when your predictions are matching your
targets or where the network is basically behaving well so we we manipulate the loss function so that
when the loss is low the network is doing what you want it to do on your problem
and then we backward the loss use backpropagation to get the gradient and then we know how to tune all the
parameters to decrease the loss locally but then we have to iterate that process many times in what's called the gradient
descent so we simply follow the gradient information and that minimizes the loss
and the loss is arranged so that when the loss is minimized the network is doing what you want it to do
and yeah so we just have a blob of neural stuff and we can make it do arbitrary things and that's what gives
neural nets their power um it's you know this is a very tiny network with 41 parameters
but you can build significantly more complicated neural nets with billions at this point almost trillions of
parameters and it's a massive blob of neural tissue simulated neural tissue
roughly speaking and you can make it do extremely complex problems and these neurons then have all
kinds of very fascinating emergent properties in when you try to make them do
significantly hard problems as in the case of gpt for example we have massive amounts of text from the
internet and we're trying to get a neural net to predict to take like a few words and try to predict the next word
in a sequence that's the learning problem and it turns out that when you train this on all of internet the neural net
actually has like really remarkable emergent properties but that neural net would have hundreds of billions of parameters
but it works on fundamentally the exact same principles the neural net of course will be a bit more complex but otherwise the
value in the gradient is there and would be identical and the gradient descent would be there and would be
basically identical but people usually use slightly different updates this is a very simple stochastic gradient descent
update um and the loss function would not be mean squared error they would be using something called the cross-entropy loss
for predicting the next token so there's a few more details but fundamentally the neural network setup and neural network
training is identical and pervasive and now you understand intuitively
how that works under the hood in the beginning of this video i told you that by the end of it you would understand everything in micrograd and then we'd
walkthrough of the full code of micrograd on github
slowly build it up let me briefly prove that to you so i'm going to step through all the code that is in micrograd as of today
actually potentially some of the code will change by the time you watch this video because i intend to continue developing micrograd
but let's look at what we have so far at least init.pi is empty when you go to engine.pi that has the value
everything here you should mostly recognize so we have the data.grad attributes we have the backward function
uh we have the previous set of children and the operation that produced this value we have addition multiplication and
raising to a scalar power we have the relu non-linearity which is slightly different type of nonlinearity
than 10h that we used in this video both of them are non-linearities and notably 10h is not actually present in
micrograd as of right now but i intend to add it later with the backward which is identical and
then all of these other operations which are built up on top of operations here so values should be very recognizable
except for the non-linearity used in this video um there's no massive difference between relu and 10h and sigmoid and these other
non-linearities they're all roughly equivalent and can be used in mlps so i use 10h because it's a bit smoother and
because it's a little bit more complicated than relu and therefore it's stressed a little bit more the
local gradients and working with those derivatives which i thought would be useful and then that pi is the neural networks
library as i mentioned so you should recognize identical implementation of neuron layer and mlp
notably or not so much we have a class module here there is a parent class of all these modules i did
that because there's an nn.module class in pytorch and so this exactly matches that api and end.module and pytorch has
also a zero grad which i've refactored out here so that's the end of micrograd really
then there's a test which you'll see basically creates two chunks of code one in micrograd and
one in pi torch and we'll make sure that the forward and the backward pass agree identically for a slightly less complicated
expression a slightly more complicated expression everything agrees so we agree with pytorch on all
of these operations and finally there's a demo.ipymb here and it's a bit more complicated binary
classification demo than the one i covered in this lecture so we only had a tiny data set of four examples um here
we have a bit more complicated example with lots of blue points and lots of red points and we're trying to again build a
binary classifier to distinguish uh two dimensional points as red or blue it's a bit more complicated mlp here
with it's a bigger mlp the loss is a bit more complicated because it supports batches
so because our dataset was so tiny we always did a forward pass on the entire data set of four examples but when your
data set is like a million examples what we usually do in practice is we chair we basically pick out some random subset we
call that a batch and then we only process the batch forward backward and update so we don't have to forward the
entire training set so this supports batching because there's a lot more examples here
we do a forward pass the loss is slightly more different this is a max margin loss that i implement here
the one that we used was the mean squared error loss because it's the simplest one there's also the binary cross entropy
loss all of them can be used for binary classification and don't make too much of a difference in the simple examples
that we looked at so far there's something called l2 regularization used here this has to do
with generalization of the neural net and controls the overfitting in machine learning setting but i did not cover
these concepts and concepts in this video potentially later and the training loop you should recognize so forward backward with zero
grad and update and so on you'll notice that in the update here the learning rate is
scaled as a function of number of iterations and it shrinks and this is something called learning
rate decay so in the beginning you have a high learning rate and as the network sort of stabilizes near the end you
bring down the learning rate to get some of the fine details in the end and in the end we see the decision
surface of the neural net and we see that it learns to separate out the red and the blue area based on the data
points so that's the slightly more complicated example and then we'll demo that hyper ymb that you're free to go over
but yeah as of today that is micrograd i also wanted to show you a little bit of real stuff so that you get to see how
real stuff: diving into PyTorch, finding their backward pass for tanh
this is actually implemented in production grade library like by torch uh so in particular i wanted to show i
wanted to find and show you the backward pass for 10h in pytorch so here in
micrograd we see that the backward password 10h is one minus t square where t is the output of the tanh of x
times of that grad which is the chain rule so we're looking for something that looks like this now
i went to pytorch um which has an open source github codebase and uh i looked
through a lot of its code and honestly i i i spent about 15 minutes and i couldn't find 10h
and that's because these libraries unfortunately they grow in size and entropy and if you just search for 10h
you get apparently 2 800 results and 400 and 406 files so i don't know what these
files are doing honestly and why there are so many mentions of
10h but unfortunately these libraries are quite complex they're meant to be used not really inspected um
eventually i did stumble on someone who tries to change the 10 h backward
code for some reason and someone here pointed to the cpu kernel and the kuda kernel for 10 inch
backward so this so basically depends on if you're using pi torch on a cpu device or
on a gpu which these are different devices and i haven't covered this but this is the 10 h backwards kernel
for uh cpu and the reason it's so large is that
number one this is like if you're using a complex type which we haven't even talked about if you're using a specific data type of b-float 16 which we haven't
talked about and then if you're not then this is the kernel and deep here we see something
that resembles our backward pass so they have a times one minus b square uh so this b
b here must be the output of the 10h and this is the health.grad so here we found it
uh deep inside pi torch from this location for some reason inside binaryops kernel when 10h
is not actually a binary op and then this is the gpu kernel
we're not complex we're here and here we go with one line of code
so we did find it but basically unfortunately these codepieces are very large and
micrograd is very very simple but if you actually want to use real stuff uh finding the code for it you'll actually
find that difficult i also wanted to show you a little example here where pytorch is showing
you how can you can register a new type of function that you want to add to pytorch as a lego building block
so here if you want to for example add a gender polynomial 3
here's how you could do it you will register it as a class that subclasses storage.org that function
and then you have to tell pytorch how to forward your new function and how to backward through it
so as long as you can do the forward pass of this little function piece that you want to add and as long as you know the the local derivative the local
gradients which are implemented in the backward pi torch will be able to back propagate through your function and then
you can use this as a lego block in a larger lego castle of all the different lego blocks that pytorch already has
and so that's the only thing you have to tell pytorch and everything would just work and you can register new types of functions
in this way following this example and that is everything that i wanted to cover in this lecture so i hope you enjoyed building out
conclusion
micrograd with me i hope you find it interesting insightful and
yeah i will post a lot of the links that are related to this video in the video description below i will also
probably post a link to a discussion forum or discussion group where you can ask questions related to this video and then
i can answer or someone else can answer your questions and i may also do a follow-up video that answers some of the
most common questions but for now that's it i hope you enjoyed it if you did then please like and
subscribe so that youtube knows to feature this video to more people and that's it for now i'll see you later
outtakes :)
now here's the problem we know dl by
wait what is the problem and that's everything i wanted to cover in this lecture
so i hope you enjoyed us building up microcraft micro crab
okay now let's do the exact same thing for multiply because we can't do something like a times two
oops i know what happened there

在本讲中，我们总结了学到的内容以及如何向现代神经网络进发。简要来说：

### 1. **神经网络是什么**

神经网络是通过数学表达式构建的，在多层感知机（MLP）的情况下，这些表达式相对简单。神经网络的输入通常是数据和神经网络的权重与参数。我们通过前向传播计算输出，然后使用损失函数来衡量预测的准确度，目标是使得损失值尽可能低，以便让网络的预测尽量接近真实目标。

### 2. **损失函数和反向传播**

损失函数衡量网络输出与真实目标之间的差距。我们通过计算损失函数的梯度（即反向传播）来更新网络的参数，使得损失最小化。通过反向传播，我们可以计算每个参数的梯度，并使用梯度下降法来调整这些参数，从而使得损失最小化。

### 3. **梯度下降法**

梯度下降法是一种优化算法，它通过不断更新神经网络的参数来最小化损失函数。每次更新时，参数朝着损失函数的梯度方向调整，步骤通常非常小。随着不断的前向和反向传播，神经网络的性能逐步提升。

### 4. **神经网络的复杂性**

我们学习了如何使用一个小的神经网络（41个参数）来做二分类任务，但现代神经网络通常会有数百万到数十亿个参数。例如，像GPT这种大规模的语言模型，拥有数百亿的参数，依然遵循相同的训练原理，只是在实现上有所不同。

### 5. **现代神经网络的实现**

现代神经网络库（例如PyTorch）也遵循相同的原理，但是它们具有更高效的实现方式，支持并行计算和批处理。PyTorch等库有许多优化和工具来支持大规模数据和复杂模型的训练。

### 6. **代码讲解**

我们通过一个简单的示例逐步构建了一个小型神经网络，并介绍了如何实现前向传播、反向传播以及梯度更新。然后，我们展示了如何利用梯度下降法来优化神经网络的参数，最终使网络能够更准确地预测目标值。

### 7. **PyTorch与Micrograd**

我们还展示了如何用PyTorch实现同样的功能。尽管PyTorch的实现更加高效和复杂，但底层原理是相同的。通过学习这些基本原理，我们能够理解更复杂的神经网络库的实现。

### 8. **实际应用**

通过实现梯度下降优化算法，我们可以训练神经网络并观察其逐步提高的表现。最后，我们讨论了如何使用学习率衰减等技巧来进一步优化网络训练过程。

### 结论

我们通过讲解 `micrograd` 和相关的神经网络原理，让大家了解了神经网络是如何工作的，并为深入学习现代神经网络架构打下了基础。

