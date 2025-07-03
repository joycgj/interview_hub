This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What they are, where they are headed, comparisons and analogies to present-day operating systems, and some of the security-related challenges of this new computing paradigm.
As of November 2023 (this field moves fast!).

Context: This video is based on the slides of a talk I gave recently at the AI Security Summit. The talk was not recorded but a lot of people came to me after and told me they liked it. Seeing as I had already put in one long weekend of work to make the slides, I decided to just tune them a bit, record this round 2 of the talk and upload it here on YouTube. Pardon the random background, that's my hotel room during the thanksgiving break.

- Slides as PDF: https://drive.google.com/file/d/1pxx_... (42MB)
- Slides. as Keynote: https://drive.google.com/file/d/1FPUp... (140MB)

Few things I wish I said (I'll add items here as they come up):
- The dreams and hallucinations do not get fixed with finetuning. Finetuning just "directs" the dreams into "helpful - assistant dreams". Always be careful with what LLMs tell you, especially if they are telling you something from - memory alone. That said, similar to a human, if the LLM used browsing or retrieval and the answer made its way into - the "working memory" of its context window, you can trust the LLM a bit more to process that information into the - final answer. But TLDR right now, do not trust what LLMs say or do. For example, in the tools section, I'd always - recommend double-checking the math/code the LLM did.
- How does the LLM use a tool like the browser? It emits special words, e.g. |BROWSER|. When the code "above" that is - inferencing the LLM detects these words it captures the output that follows, sends it off to a tool, comes back with - the result and continues the generation. How does the LLM know to emit these special words? Finetuning datasets teach - it how and when to browse, by example. And/or the instructions for tool use can also be automatically placed in the - context window (in the “system message”).
- You might also enjoy my 2015 blog post "Unreasonable Effectiveness of Recurrent Neural Networks". The way we obtain - base models today is pretty much identical on a high level, except the RNN is swapped for a Transformer. http://- karpathy.github.io/2015/05/21/...
- What is in the run.c file? A bit more full-featured 1000-line version hre: https://github.com/karpathy/llama2.c/...

Chapters:

```
Part 1: LLMs
00:00:00 Intro: Large Language Model (LLM) talk
00:00:20 LLM Inference
00:04:17 LLM Training
00:08:58 LLM dreams
00:11:22 How do they work?
00:14:14 Finetuning into an Assistant
00:17:52 Summary so far
00:21:05 Appendix: Comparisons, Labeling docs, RLHF, Synthetic data, Leaderboard
Part 2: Future of LLMs
00:25:43 LLM Scaling Laws
00:27:43 Tool Use (Browser, Calculator, Interpreter, DALL-E)
00:33:32 Multimodality (Vision, Audio)
00:35:00 Thinking, System 1/2
00:38:02 Self-improvement, LLM AlphaGo
00:40:45 LLM Customization, GPTs store
00:42:15 LLM OS
Part 3: LLM Security
00:45:43 LLM Security Intro
00:46:14 Jailbreaks
00:51:30 Prompt Injection
00:56:23 Data poisoning
00:58:37 LLM Security conclusions
End
00:59:23 Outro
```

Educational Use Licensing
This video is freely available for educational and internal training purposes. Educators, students, schools, universities, nonprofit institutions, businesses, and individual learners may use this content freely for lessons, courses, internal training, and learning activities, provided they do not engage in commercial resale, redistribution, external commercial use, or modify content to misrepresent its intent.

这是一个为普通观众设计的关于大型语言模型（LLM）的1小时入门介绍，讲解了LLM的核心技术组件，这些技术构成了像ChatGPT、Claude和Bard这样的系统。内容包括LLM是什么、它们的未来发展趋势、与现有操作系统的比较及类比，以及这种新型计算范式所面临的一些安全挑战。

截至2023年11月（这个领域发展迅速！）。

背景：本视频基于我最近在AI安全峰会上进行的演讲幻灯片。由于演讲没有录制下来，但很多与会者会后告诉我他们很喜欢这次演讲。所以我决定稍微调整一下幻灯片，录制这次演讲的第二轮，并上传到YouTube。请原谅我背景的随机性，那是我在感恩节假期时住的酒店房间。

幻灯片PDF版：[链接](https://drive.google.com/file/d/1pxx_...)（42MB）
幻灯片Keynote版：[链接](https://drive.google.com/file/d/1FPUp...)（140MB）

几点我希望自己说的（会根据需要在这里更新）：

1. 梦境和幻觉并不会通过微调解决。微调只是“引导”梦境进入“有帮助的助手梦境”。始终要小心LLM告诉你的内容，特别是当它们仅仅从记忆中告诉你某些事情时。话虽如此，类似于人类，如果LLM使用了浏览或检索，并且答案已经进入了它的“工作记忆”或上下文窗口，你可以稍微相信LLM会将信息处理成最终答案。但总的来说，现在的结论是，不要完全相信LLM说的任何事情。例如，在工具部分，我总是建议重新检查LLM做的数学/代码。
2. LLM如何使用像浏览器这样的工具？它会发出特殊的词，例如|BROWSER|。当“上层”代码推理LLM并检测到这些词时，它会捕获后续输出，发送到工具，工具返回结果后，继续生成。LLM如何知道发出这些特殊的词？微调数据集通过示例教它如何以及何时浏览，或者工具使用的指令也可以自动放置在上下文窗口中（在“系统消息”中）。
3. 你可能会喜欢我2015年的博客文章《循环神经网络的非理性有效性》。今天我们获得基础模型的方式在高层次上几乎是相同的，只不过RNN被Transformer取代了。[博客链接](http://karpathy.github.io/2015/05/21/)
4. 在run.c文件中是什么？这里有一个功能更全的1000行版本：[GitHub链接](https://github.com/karpathy/llama2.c/...)

章节：
**第1部分：LLM**
```
00:00:00 介绍：大型语言模型（LLM）讲座
00:00:20 LLM推理
00:04:17 LLM训练
00:08:58 LLM梦境
00:11:22 它们是如何工作的？
00:14:14 微调为助手
00:17:52 到目前为止的总结
00:21:05 附录：比较、标签文档、强化学习与人类反馈（RLHF）、合成数据、排行榜
```

**第2部分：LLM的未来**
```
00:25:43 LLM扩展法则
00:27:43 工具使用（浏览器、计算器、解释器、DALL-E）
00:33:32 多模态（视觉、音频）
00:35:00 思维，系统1/2
00:38:02 自我改进，LLM AlphaGo
00:40:45 LLM定制化，GPT商店
00:42:15 LLM操作系统
```

**第3部分：LLM安全性**
```
00:45:43 LLM安全性介绍
00:46:14 越狱
00:51:30 提示注入
00:56:23 数据中毒
00:58:37 LLM安全性结论
结束
00:59:23 结尾
```

**教育用途许可**
此视频可免费用于教育和内部培训目的。教育者、学生、学校、大学、非营利组织、企业和个人学习者可以自由使用此内容进行课程、内部培训和学习活动，前提是不得进行商业转售、重新分发、外部商业使用或修改内容以误导其原意。
