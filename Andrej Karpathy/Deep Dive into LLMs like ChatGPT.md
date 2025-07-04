This is a general audience deep dive into the Large Language Model (LLM) AI technology that powers ChatGPT and related products. It is covers the full training stack of how the models are developed, along with mental models of how to think about their "psychology", and how to get the best use them in practical applications. I have one "Intro to LLMs" video already from ~year ago, but that is just a re-recording of a random talk, so I wanted to loop around and do a lot more comprehensive version.

Instructor
Andrej was a founding member at OpenAI (2015) and then Sr. Director of AI at Tesla (2017-2022), and is now a founder at Eureka Labs, which is building an AI-native school. His goal in this video is to raise knowledge and understanding of the state of the art in AI, and empower people to effectively use the latest and greatest in their work.
Find more at https://karpathy.ai/ and https://x.com/karpathy

```
Chapters
00:00:00 introduction
00:01:00 pretraining data (internet)
00:07:47 tokenization
00:14:27 neural network I/O
00:20:11 neural network internals
00:26:01 inference
00:31:09 GPT-2: training and inference
00:42:52 Llama 3.1 base model inference
00:59:23 pretraining to post-training
01:01:06 post-training data (conversations)
01:20:32 hallucinations, tool use, knowledge/working memory
01:41:46 knowledge of self
01:46:56 models need tokens to think
02:01:11 tokenization revisited: models struggle with spelling
02:04:53 jagged intelligence
02:07:28 supervised finetuning to reinforcement learning
02:14:42 reinforcement learning
02:27:47 DeepSeek-R1
02:42:07 AlphaGo
02:48:26 reinforcement learning from human feedback (RLHF)
03:09:39 preview of things to come
03:15:15 keeping track of LLMs
03:18:34 where to find LLMs
03:21:46 grand summary
```

Links
- ChatGPT https://chatgpt.com/
- FineWeb (pretraining dataset): https://huggingface.co/spaces/Hugging...
- Tiktokenizer: https://tiktokenizer.vercel.app/
- Transformer Neural Net 3D visualizer: https://bbycroft.net/llm
- llm.c Let's Reproduce GPT-2 https://github.com/karpathy/llm.c/dis...
- Llama 3 paper from Meta: https://arxiv.org/abs/2407.21783
- Hyperbolic, for inference of base model: https://app.hyperbolic.xyz/
- InstructGPT paper on SFT: https://arxiv.org/abs/2203.02155
- HuggingFace inference playground: https://huggingface.co/spaces/hugging...
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- TogetherAI Playground for open model inference: https://api.together.xyz/playground
- AlphaGo paper (PDF): https://discovery.ucl.ac.uk/id/eprint...
- AlphaGo Move 37 video:    • Lee Sedol vs AlphaGo  Move 37 reactions an...  
- LM Arena for model rankings: https://lmarena.ai/
- AI News Newsletter: https://buttondown.com/ainews
- LMStudio for local inference https://lmstudio.ai/

- The visualization UI I was using in the video: https://excalidraw.com/
- The specific file of Excalidraw we built up: https://drive.google.com/file/d/1EZh5...
- Discord channel for Eureka Labs and this video:   / discord  

Educational Use Licensing
This video is freely available for educational and internal training purposes. Educators, students, schools, universities, nonprofit institutions, businesses, and individual learners may use this content freely for lessons, courses, internal training, and learning activities, provided they do not engage in commercial resale, redistribution, external commercial use, or modify content to misrepresent its intent.

这是一个面向一般观众的深度解析，介绍了支持ChatGPT及相关产品的大型语言模型（LLM）AI技术。视频全面讲解了模型的开发过程，包括从数据预训练到神经网络架构的各个方面，探讨了如何理解它们的“心理”，以及如何在实际应用中高效使用这些技术。虽然我大约一年前已经发布了一个“LLM入门”视频，但那只是一次随机演讲的重录，因此我希望通过这次视频进行更加全面的讲解。

**讲师：**
Andrej是OpenAI的创始成员之一（2015年），曾担任特斯拉的AI高级总监（2017-2022），现为Eureka Labs的创始人，该公司致力于构建AI本土化的学校。他在这个视频中的目标是提升人们对AI技术最前沿的了解，并帮助他们在工作中高效利用最新的AI成果。

**更多信息**：
- [Andrej的个人网站](https://karpathy.ai/)
- [Andrej的X主页](https://x.com/karpathy)

```
**视频章节：**
00:00:00 引言
00:01:00 预训练数据（互联网）
00:07:47 分词
00:14:27 神经网络输入输出
00:20:11 神经网络内部结构
00:26:01 推理过程
00:31:09 GPT-2：训练与推理
00:42:52 Llama 3.1基础模型推理
00:59:23 从预训练到后训练
01:01:06 后训练数据（对话）
01:20:32 幻觉、工具使用、知识与工作记忆
01:41:46 自我认知
01:46:56 模型需要令牌才能思考
02:01:11 分词再探：模型在拼写上的困扰
02:04:53 锯齿型智能
02:07:28 监督微调到强化学习
02:14:42 强化学习
02:27:47 DeepSeek-R1
02:42:07 AlphaGo
02:48:26 来自人类反馈的强化学习（RLHF）
03:09:39 未来展望
03:15:15 跟踪LLM的发展
03:18:34 如何找到LLM
03:21:46 总结
```

**相关链接：**

* [ChatGPT](https://chatgpt.com/)
* [FineWeb（预训练数据集）](https://huggingface.co/spaces/Hugging...)
* [Tiktokenizer](https://tiktokenizer.vercel.app/)
* [Transformer神经网络3D可视化器](https://bbycroft.net/llm)
* [llm.c：重现GPT-2](https://github.com/karpathy/llm.c/dis...)
* [Meta的Llama 3论文](https://arxiv.org/abs/2407.21783)
* [Hyperbolic，基础模型推理](https://app.hyperbolic.xyz/)
* [InstructGPT论文：SFT](https://arxiv.org/abs/2203.02155)
* [HuggingFace推理平台](https://huggingface.co/spaces/hugging...)
* [DeepSeek-R1论文](https://arxiv.org/abs/2501.12948)
* [TogetherAI开放模型推理平台](https://api.together.xyz/playground)
* [AlphaGo论文（PDF）](https://discovery.ucl.ac.uk/id/eprint...)
* \[AlphaGo第37步视频]\(• Lee Sedol vs AlphaGo  Move 37 reactions an...)
* [LM Arena：模型排名](https://lmarena.ai/)
* [AI新闻通讯](https://buttondown.com/ainews)
* [LMStudio本地推理平台](https://lmstudio.ai/)

**视频中使用的可视化UI**：
- [Excalidraw](https://excalidraw.com/)
- [我们在Excalidraw中构建的具体文件](https://drive.google.com/file/d/1EZh5...)
- [Eureka Labs和本视频的Discord频道]

**教育用途许可：**
该视频可自由用于教育和内部培训目的。教育工作者、学生、学校、大学、非营利机构、企业和个人学习者可以自由使用此内容用于课程、教学、内部培训和学习活动，前提是不得进行商业转售、再分发、外部商业用途，或修改内容以误导其原意。
