---
layout: default
---

# Document Based Question Answering Heuristics
The task of Document Based Question Answering(DBQA) is to answer questions by selecting one or multiple sentences from **a set of unstructured documents** as answers. Formally, given an utterance $$Q$$ and a document set $$D$$, the document-based chatbot engine retrieves response $$R$$ based on the following three steps:

  1. **Response Retrieval**, which retrieves response candidates $$C$$ from $$D$$ based on $$Q$$:  
$$C = Retrieve(Q, D)$$  
Each S ∈ C is a sentence existing in D.
  2. **Response Ranking**, which ranks all response candidates in **C** and selects the most possible response candidate as $$\hat S$$:  
$$\hat S = argmax_{S∈C}Rank(S, Q)$$
  3. **Response Triggering**, which decides whether it is confident enough to response $$Q$$ using $$\hat{S}$$:  
$$I = T rigger(\hat S, Q)$$  
where $$I$$ is a binary value. When $$I$$ equals to true, let the response $$R = \hat S$$ and output $$R$$; otherwise, output nothing.


## Response Retrieval
### Search Engine 101
#### Index
The purpose of storing an index is to optimize speed and performance in finding relevant documents for a search query. Without an index, the search engine would scan every document in the corpus, which would require considerable time and computing power.  
The most commonly used indexing method is ***Inverted Indexing***. An inverted index consists of a list of all the unique words that appear in any document, and for each word, we attach a list of the documents in which it appears.

![inverted index](/assets/img/question_answering/inverted_index.png)

#### Search
Given user query $$q$$, we first tokenize $$q$$, such that $$q = (w_{1,q}, w_{2, q}, ..., w_{n, q})$$, for each word $$w_{k, q}$$, we can read their corresponding inverted index and get a list of documents $$d_{k}$$ contain the word $$w_{k,q}$$. The final candidate document set is $$(d_{1}, d_{2}, ..., d_{n})$$.  
We could have more constrains on the candidate document set, for example, a binary boolean operator(*should*, *must*) could be specified on each word to indicate this world *must* be contained by each of the final candidate documents or not. The operator and the word constitute a boolean clause, clause can be integrated into a more complex clause.

#### Ranking
The candidate document set could be large as 10,000, we need a facility to score these documents to find out the most relevance ones.

##### Vector Space Model and TF-IDF statistics
Say we have a vocabulary represents the uniq terms in the corpus(all documents in search engine). Each query and document can be represented as a vocab-size vector, such that we can calculate the similarity score between $$q$$ and a candidate document $$\hat d$$  
$$cosine\_similarity(q, \hat d) = \dfrac{q \cdot \hat d}{||q||\cdot||\hat d||}$$  

Each dimension $$i$$ of the vector is the weight of $$i$$-th term in the vocabulary. The most simple weighting strategy is *Boolean weighting*, if term $$i$$ appear in document, so set $$\hat d[i] = 1$$. Documents share more common terms with th query have higher score.

Boolean weighting has its own weakness. Terms in query are not identical important. TF-IDF(Term Frequency–Inverse Document Frequency) weighting believes that the more some term $$t$$ appears in the document, and the less $$t$$ is contained by other document, the more important $$t$$ is.  
$$
\begin{align}
TF(t, d) &= 0.5 + 0.5 * \dfrac{f_{t, d}}{max\{f'_{t', d}:t' \in d\}}\\
IDF(t) &= log \dfrac{|corpus|}{|d \in D, t \in d|} \\
TF-IDF(t, d) &= TF(t, d) * IDF(t)
\end{align}
$$

## Response Ranking
Given a user utterance Q and a response candidate $$S$$, the ranking function $$Rank(S, Q)$$ is designed as an ensemble of individual matching features:   
$$Rank(S, Q) = \sum_k λ_k · h_k(S, Q)$$  
where $$h_k(·)$$ denotes the $$k$$-th feature function, $$λ_k$$ denotes $$h_k(·)$$’s corresponding weight.

### Framework
We need to design multi-level(Paraphrase, Causality, Discourse Relationship, etc.) semantic and syntactic features to rank $$<Q, S>$$ pair precisely, which is often laborious. Deep learning approaches have gained a lot of attention from the research community and industry for their ability to automatically learn optimal feature representation for a given task, while claiming state-of-the-art performance in many tasks.

<img src="/assets/img/question_answering/DQA.jpeg" alt="Deep QA" width="100%"/>

### Data
#### 正例
1. Q:吃 了 罗 红霉素 分散 片 可以 怀孕 吗 ？  
A: 怀孕 三十八 周 可以 口服 罗 红霉素 分散 片 的 ， 但 在 怀孕 期间 不 能 盲目 用药 ， 盲目 用药 是 会 影响 胎儿 发育 的 ， 建议 你 在 医生 的 指导 下用药 为宜 。
2. Q:拍 胸片 后 多久 可以 怀孕 ？  
A:因此 , 妇女 平时 应 尽量 减少 x光 的 照射 机会 , 怀孕 前 4 周 内 必须 禁忌 照射 X光
3. Q:剖腹产 后 多久 可以 做 瑜伽 ？  
A:那剖腹产 后 多久 可以 做 瑜伽 ？

正例噪声很大， QA库里面的答案不一定是全部跟问题相关。
   
#### 负例
  1. Q：孕前 优生优育 如何 检查 ？  
A: 宝宝 补 锌 吃 什么 好 ？
  2. Q: 哺乳期 用 什么 护肤品 好 ？  
A: 头 围 是 指 绕 胎 头 一 周 的 最 大 长度 。  

负例太负，跟实际的数据分布不一致。实际数据是一篇篇的文章，这些文章里面的句子都是相关的。我们实际需要的是一些看起来相近的负例。比如问题是问因果，答案是说为什么。

### Demo
[DEMO](http://10.15.26.26:8080/zd-docchat-platform-0.0.1-SNAPSHOT/docchat)

Cases:
1. 怀孕可以锻炼吗 （G）
2. 宝宝吃手指（G）
3. 宝宝吃手指怎么办（G）
3. 宝宝发烧身上出红疹子怎么办 (N)
4. 宝宝晚上不睡觉 (N)
5. 宝宝感冒怎么办（B）


## Reference
1. [A first take at building an inverted index](https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html)
2. [Search engine indexing](https://en.wikipedia.org/wiki/Search_engine_indexing)
3. [Vector Space Model](https://en.wikipedia.org/wiki/Vector_space_model)
4. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
5. [DocChat: An Information Retrieval Approach for Chatbot Engines
Using Unstructured Documents](http://aclweb.org/anthology/P16-1049)
6. [Learning to Rank Short Text Pairs with Convolutional Deep
Neural Networks](https://pdfs.semanticscholar.org/73d8/26d4c2363701b88e3e234fe3b8756c0f9671.pdf)