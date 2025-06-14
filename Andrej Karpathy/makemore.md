# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
$ python makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
$ python makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### License

MIT

# makemore

makemore 接受一个文本文件作为输入，其中每一行被视为一个训练样本，并生成类似的内容。在后台，它是一个自回归的字符级语言模型，支持从二元组到Transformer（正如在GPT中看到的）等多种模型。例如，我们可以给它一个名字的数据库，makemore 将生成听起来像名字的酷婴儿名字建议，但这些名字并不是已存在的名字。或者，如果我们给它一个公司名称的数据库，它就能生成新的公司名称创意。或者我们可以给它有效的拼字游戏单词，makemore 将生成类似英语的胡言乱语。

这不是一个复杂的库，没有亿万个开关和按钮。它只是一个可以修改的文件，主要用于教育目的。唯一的依赖是 PyTorch。

当前的实现参考了几篇关键论文：

* Bigram（一个字符预测下一个字符，通过查找计数表）
* MLP，参考 Bengio 等人 2003
* CNN，参考 DeepMind WaveNet 2016（进行中...）
* RNN，参考 Mikolov 等人 2010
* LSTM，参考 Graves 等人 2014
* GRU，参考 Kyunghyun Cho 等人 2014
* Transformer，参考 Vaswani 等人 2017

### 使用方法

所包含的 `names.txt` 数据集作为示例，包含了来自 ssa.gov 的2018年最常见的32K个名字，格式如下：

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

我们可以这样运行脚本：

```
$ python makemore.py -i names.txt -o names
```

训练进度、日志和模型将会保存到工作目录 `names` 中。默认模型是一个非常小的200K参数的Transformer；更多的训练配置可用——请查看 argparse 并阅读代码。训练不需要任何特殊硬件，它可以在我的 Macbook Air 上运行，也可以在其他任何设备上运行，但如果有 GPU，训练会更快。随着训练的进行，脚本会定期打印一些样本。如果你想手动采样，可以使用 `--sample-only` 标志，例如，在一个单独的终端中执行：

```
$ python makemore.py -i names.txt -o names --sample-only
```

这将加载到目前为止表现最好的模型，并按需打印更多样本。以下是一些在当前默认设置下最终生成的独特婴儿名字（测试对数概率约为1.92，尽管通过调整超参数可以达到更低的对数概率）：

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

玩得开心！

### 许可证

MIT

# makemore.py introduction

you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.

你给这个脚本一些单词（每行一个），它将生成更多类似的内容。
使用最先进的 Transformer AI 技术。
这段代码旨在非常易于修改。根据你的需求进行调整。

与 minGPT 的变化：

* 我移除了 `from_pretrained` 函数，原本用于初始化 GPT2 权重。
* 我移除了 dropout 层，因为我们在这里训练的模型很小，在这个阶段和规模下不需要。
* 我移除了权重衰减以及关于哪些参数需要衰减、哪些不需要的所有复杂性。我认为在我们操作的规模下，这不会产生巨大差异。

