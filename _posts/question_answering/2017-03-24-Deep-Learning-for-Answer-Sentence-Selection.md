---
layout: default
---
### Deep Learning based approaches
1. [APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK](https://arxiv.org/pdf/1508.01585.pdf) [IBM Watson]
We apply a general deep learning framework to address the non-factoid question answering task. Our approach does not rely on any linguistic tools and can be applied to different languages or domains...the top-1 accuracy can reach up to 65.3% on a test set, which indicates a great potential for practical use. 
We treat the QA from a text matching and selection perspective.  
From the definition(From answer selection's perspective), the QA problem can be regarded as a binary classification problem. For each question, for each answer candidate, it may be appropriate or not. In order to find the best pair, we need a metric to measure the matching degree of each QA pair so that the QA pair with highest metric value will be chosen.   
Section 4 "4. RESULTS AND DISCUSSIONS" needs carefully attention
2. [A Neural Network for Factoid Question Answering over Paragraphs](https://www.semanticscholar.org/paper/A-Neural-Network-for-Factoid-Question-Answering-Iyyer-Boyd-Graber/0ec80f1e6f7dfbbbcc97459d4c5ae13be1cadc7b) 
3. [Deep Learning for Answer Sentence Selection](https://www.semanticscholar.org/paper/Deep-Learning-for-Answer-Sentence-Selection-Yu-Hermann/a62b58c267fddfa06545a7fc63a3c62ef7dc9e15)
4. [A Deep Architecture for Matching Short Texts](https://www.semanticscholar.org/paper/A-Deep-Architecture-for-Matching-Short-Texts-Lu-Li/4aba54ea82bf99ed4690d45051f1b25d8b9554b5)
However, before learning can take place, such pairs needs to be mapped from the original space of symbolic words into some feature space encoding various aspects of their relatedness, e.g. lexical, syntactic and semantic.  
While pairwise and listwise approaches claim to yield better performance, they are more complicated to implement and less effective train.
To train
the embeddings we use the skipgram model with window size 5
and filtering words with frequency less than 5. The resulting model
contains 50-dimensional vectors for about 3.5 million words. 
. Embeddings
for words not present in the word2vec model are randomly
initialized with each component sampled from the uniform
distribution U[−0.25, 0.25].
Additionally, even for the words found in the word matrix, as
noted in [38], one of the weaknesses of approaches relying on dis-tributional word vectors is their inability to deal with numbers and
proper nouns. This is especially important for factoid question answering,
where most of the questions are of type what, when, who
that are looking for answers containing numbers or proper nouns
5. [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](https://www.semanticscholar.org/paper/Dynamic-Pooling-and-Unfolding-Recursive-Socher-Huang/167abf2c9eda9ce21907fcc188d2e41da37d9f0b)
6. [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](https://www.semanticscholar.org/paper/Pairwise-Word-Interaction-Modeling-with-Deep-He-Lin/0476b7d387d2a6381a784b2b89ccf7baef098f5e) [... a novel neural network architecture that demonstrates state-of-the-art accuracy on three SemEval tasks and two answer selection tasks.]
7. [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](https://www.semanticscholar.org/paper/Learning-to-Rank-Short-Text-Pairs-with-Severyn-Moschitti/73d826d4c2363701b88e3e234fe3b8756c0f9671)
8. [Multitask Learning with Deep Neural Networks for Community Question Answering](https://www.semanticscholar.org/paper/Multitask-Learning-with-Deep-Neural-Networks-for-Bonadiman-Uva/ad21f9672634fe1ef2048b58e09d6f85529dfd81)
9. [Inner Attention based Recurrent Neural Networks for Answer Selection](https://www.semanticscholar.org/paper/Inner-Attention-based-Recurrent-Neural-Networks-Wang-Liu/52956422f86722aca6becb67ea4c3ad61f0c1aea)
10. [LSTM-based Deep Learning Models for non-factoid answer selection](https://www.semanticscholar.org/paper/LSTM-based-Deep-Learning-Models-for-non-factoid-Tan-Xiang/24b4746688c15c8580d1000d4f4ac63f5eb79561)

### Undefined

1. [A survey on question answering systems with classification](http://ac.els-cdn.com/S1319157815000890/1-s2.0-S1319157815000890-main.pdf?_tid=51d952d0-1063-11e7-8285-00000aacb35d&acdnat=1490340621_39f523cb0a867f3bb6b3c1235d7709b9)
2. [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.semanticscholar.org/paper/WikiQA-A-Challenge-Dataset-for-Open-Domain-Yang-Yih/03fe39386ce90e10ec87f10e00532c5cf30b244f)
3. [Automatic Feature Engineering for Answer Selection and Extraction](https://www.semanticscholar.org/paper/Automatic-Feature-Engineering-for-Answer-Selection-Severyn-Moschitti/7aa63f414a4d7c6e4369a15a04dc5d3eb5da2b0e)
4. [Question Answering Using Enhanced Lexical Semantic Models](https://www.semanticscholar.org/paper/Question-Answering-Using-Enhanced-Lexical-Semantic-Yih-Chang/3e393df4a5731fb7b49cf2f527fed1ee4e6e6942)
5. [A Joint Model for Answer Sentence Ranking and Answer Extraction](https://www.semanticscholar.org/paper/A-Joint-Model-for-Answer-Sentence-Ranking-and-Sultan-Castelli/59018eb4d0a5161a12cdca42cbcb6bf78d73612f)
6. [Learning concept importance using a weighted dependence model]()
7. [Parameterized concept weighting in verbose queries]() [6 and 7 are mentioned as state-of-the-art in Waston lab's paper]

### Questions (q,q)
1. 关键词
宝宝能吃辣椒吗？ 跟宝宝能吃鱼吗？相似度高
2. 肯定，否定
3. 语义
4. 相似问题不同答案
5. 歧义
宝宝多大能睡枕头。宝宝能睡多大的枕头
6. 时间／数量词
7个月宝宝能吃鱼吗？    9个月宝宝能吃鱼吗？
7. 疑问词
问的是怎么办，回答的是为什么
8. 分词粒度
大三阳和小三阳的分词错误
9. 缩写
孕前和怀孕前期； 唐筛和唐氏筛查


（1）关键词的识别：识别词的重要程度，除了BM25之外，还有搜索引擎中term出现的频率；
“孕妇怎么补锌？” vs “孕妇怎么补钙？”

（2）附加词的去除：
比如“速度有多快”里面的“快”，“炎症有哪些症状”里面的“症状”，省略词的复原类似；

（3）属性词的识别：
急性、慢性

（4）性别词的识别：
前列腺、经期

（5）缩写词的还原：
大学三年级 vs 大三
唐筛 vs 唐氏筛查

（6）数量词的计算：
3周半、7个月

（7）分词的粒度：
小／三／阳、大三／阳

（8）疑问词的识别：疑问句有不同的类别之分
哪些？多少？

（9）时间词的识别：
天、周、月、岁

（10）限定语的去除：
怀孕早期能不能喝蜂蜜
怀孕晚期能不能喝蜂蜜

（11）歧义的消解（焦点的识别）
婴儿多大能用枕头？
婴儿能用多大枕头？

（12）否定词的识别？
“孩子白天不爱睡觉怎么办？” vs “孩子白天爱睡觉怎么办？”
“我这个结果是正常的吗？” vs “我这个结果有问题吗？”

训练数据 粒度。 分词 embedding  抽特征    
1. paragraph 第一句和最后一句  
2. 同一个问题的不同答案的第一句和最后一句的聚合做data augment  
3. 问题分类，看一下error case  
4. 提取问题的关键词，焦点， 意图， 不一定需要做full parsing  

反馈： 1. 直接问他喜不喜欢。2. 点进去相关问题。 3. 点进去后like还是dislike