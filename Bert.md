# Bert

##基本概念

### language representation model

作者认为，确实存在通用的语言模型，先用文章预训练通用模型，然后再根据具体应用，用 supervised 训练数据，精加工（fine tuning）模型，使之适用于具体应用。**为了区别于针对语言生成的 Language Model，作者给通用的语言模型，取了一个名字，叫语言表征模型 Language Representation Model**。

###预测方式：unidirectional-bi-directional-deep bi-directional；实现方法：transformer
能实现语言表征目标的模型，可能会有很多种，具体用哪一种呢？作者提议，用 Deep Bidirectional Transformers 模型。假如给一个句子 “能实现语言表征[mask]的模型”，遮盖住其中“目标”一词。从前往后预测[mask]，也就是用“能/实现/语言/表征”，来预测[mask]；或者，从后往前预测[mask]，也就是用“模型/的”，来预测[mask]，称之为单向预测 unidirectional。单向预测，不能完整地理解整个语句的语义。于是研究者们尝试双向预测。把从前往后，与从后往前的两个预测，拼接在一起 [mask1/mask2]，这就是双向预测 bi-directional。细节参阅《Neural Machine Translation by Jointly Learning to Align and Translate》。

BERT 的作者认为，bi-directional 仍然不能完整地理解整个语句的语义，更好的办法是用上下文全向来预测[mask]，也就是用 “能/实现/语言/表征/../的/模型”，来预测[mask]。BERT 作者把上下文全向的预测方法，称之为 deep bi-directional。如何来实现上下文全向预测呢？BERT 的作者建议使用 Transformer 模型。这个模型由《Attention Is All You Need》一文发明。

###具体训练任务
BERT 用了两个步骤，试图去正确地训练模型的参数。第一个步骤是把一篇文章中，15% 的词汇遮盖，让模型根据上下文全向地预测被遮盖的词。假如有 1 万篇文章，每篇文章平均有 100 个词汇，随机遮盖 15% 的词汇，模型的任务是正确地预测这 15 万个被遮盖的词汇。通过全向预测被遮盖住的词汇，来初步训练 Transformer 模型的参数。然后，用第二个步骤继续训练模型的参数。譬如从上述 1 万篇文章中，挑选 20 万对语句，总共 40 万条语句。挑选语句对的时候，其中 2*10 万对语句，是连续的两条上下文语句，另外 2*10 万对语句，不是连续的语句。然后让 Transformer 模型来识别这 20 万对语句，哪些是连续的，哪些不连续。

**这两步训练合在一起，称为预训练 pre-training。训练结束后的 Transformer 模型，包括它的参数，是作者期待的通用的语言表征模型。**


##输入表征

![img](picture/bert.jpg)图1 输入表征

图1是输入表征。从图中看，输入序列由三部分相加而成：词嵌入(Token Embedding)、分隔嵌入(Segment Embedding)、位置嵌入(Position embeddings)。

Token Embedding**使用的是Wordpiece模型训练的词向量**；

因为要做NSP任务，后续同时也要完成QA任务，需要将问题和答案放在一起调优，所以添加特殊分隔符[SEP]和Segment Embedding来区别不同的句子。Segment Embedding有两个--Embedding A和Embedding B。当然后面也有单句的调优就直接使用Embedding A。

Position embeddings和self-attention中用不一样，**self-attention中使用的是三角函数，这里通过模型学习得到。**

[CLS]表示的是特殊分类嵌入，它是Transformer的输出。对于句子级分类任务，[CLS]就是输入序列的固定维度的表示(就像词向量直接拼成句向量再输入模型中一样)。对于非分类的任务，则忽略此向量。

##模型架构

![img](picture/bert1.jpg)图2 模型框架

图2是BERT、GPT、EMLo三者的模型架构。从图中可以看到：BERT每一层都使用了双向编码，GPT只使用了从左到右的编码，ELMo独立训练从左到右的编码和从右到左的编码，然后联合起来。**难道直接使用双向的不行吗？**简单地说语言模型要做的是：计算当前面k个词出现时，第k+1个词出现的概率。直接双向模型不就知道了吗还计算预测个什么？原文模型之所以可以同时双向是因为它利用了MLM任务，下文介绍。

原文使用的模型是基于self-attention的编码器部分。如图3所示：

![img](picture/bert2.jpg)

**图3是我直接从self-attention的网络结构中直接摘出来的，应该再加上Segment Embedding就是原文所用的Transformer了。图2中BERT里每一个Trm基本上就是图3的结构。**

## 任务微调

（a）句对关系判断，第一个起始符号[CLS]经过Transformer编码器后，增加简单的Softmax层，即可用于分类；

（b）单句分类任务，具体实现同（a）一样；

（c）问答类任务，譬如SQuAD v1.1，问答系统输入文本序列的question和包含answer的段落，并在序列中标记answer，让BERT模型学习标记answer开始和结束的向量来训练模型；

（d）序列标准任务，譬如命名实体标注NER，识别系统输入标记好实体类别（人、组织、位置、其他无名实体）的文本序列进行微调训练，识别实体类别时，将序列的每个Token向量送到预测NER标签的分类层进行识别。

![](picture/bert3.jpg)

## Reference

- [bert](https://zhuanlan.zhihu.com/p/46887114)
- [nlp新秀](https://www.jiqizhixin.com/articles/2019-02-18-12)