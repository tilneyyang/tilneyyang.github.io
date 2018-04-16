---
layout: default
---
# Document-based Question Answering: A Survey

## Deep Learning based approaches
1. [APPLYING DEEP LEARNING TO ANSWER SELECTION: A STUDY AND AN OPEN TASK](https://arxiv.org/pdf/1508.01585.pdf) [IBM Watson]
We apply a general deep learning framework to address the
non-factoid question answering task. Our approach does not
rely on any linguistic tools and can be applied to different languages
or domains...the top-1 accuracy can reach up to 65.3% on a test set, which indicates a great potential for practical use.  
We treat the QA from a text matching and selection perspective.  
From the definition(From answer selection's perspective), the QA problem
can be regarded as a binary classification problem. For each question, for each answer candidate, it may be appropriate or not.  
In order to find the best pair, we need a metric to measure the matching degree of each QA pair so that the QA pair with highest metric value will be chosen.
Section 4 "4. RESULTS AND DISCUSSIONS" need carefully attention
2. [A Neural Network for Factoid Question Answering over Paragraphs](https://www.semanticscholar.org/paper/A-Neural-Network-for-Factoid-Question-Answering-Iyyer-Boyd-Graber/0ec80f1e6f7dfbbbcc97459d4c5ae13be1cadc7b) 
Focused on *Quiz Bowl* domain, where the task is to match multi sentences(paragraph-length text) to entities. Our model improves upon the existing dt-rnn model by **jointly learning answer and question representations
in the same vector space rather than learning them separately.**(IBM paper showed a similar result, use the same filter to both question and answer get the best result. Their embeddings are pretrained via word2vec, both question and answer share the same embedding matrix. They did not conduct experiment on using different embedding matrix for questions and ansers.).  
这个里面的loss是不是有问题？理论上讲，如果理论上讲，正确答案离 $$ h_s $$ 更近的话， loss应该是0，但是目前这个不是。

3. [Deep Learning for Answer Sentence Selection](https://www.semanticscholar.org/paper/Deep-Learning-for-Answer-Sentence-Selection-Yu-Hermann/a62b58c267fddfa06545a7fc63a3c62ef7dc9e15) **[It's almost a survey]**
Answer sentence selection is the task of identifying sentences that contain the
answer to a given question. We propose a novel approach to solving this task via means of distributed representations, and learn to match questions with answers by considering their semantic encoding. **This contrasts prior work on this task, which typically relies on classifiers with large numbers of hand-crafted syntactic and semantic features and various external resources.**.  
Question answering can broadly be divided into *two categories*. One approach focuses on semantic parsing, where answers are retrieved by turning a question into a database query and subsequently applying that query to an existing knowledge base. The other category is open domain question answering, which is more closely related to the field of information retrieval.  
In this paper, we focus on answer sentence selection, the task that selects
the correct sentences answering a factual question from a set of candidate sentences.  
Another line of work—closely related to the model presented here—is the application of recursive neural networks to factoid question answering over paragraphs. A key difference to our approach is that this model, given a question, selects answers from a relatively small fixed set of candidates encountered during training. On the other hand, the task of answer sentence selection that we address here, requires picking an answer from among a set of candidate sentences not encountered during training. In addition, each question has different numbers of candidate sentences.  
**Then, what does our data look like?**. 
Answer sentence selection requires both **semantic and syntactic information** in order to establish both what information the question seeks to answer, as well as whether a given candidate contains the required information, with current state-of-the-art approaches mostly focusing on syntactic matching (a number of paper has been published based on tree-edit feature) between questions and answers.  
Our solution to this problem assumes that correct answers have high semantic similarity to questions. Unlike previous work, which measured the similarity mainly using syntactic information and handcrafted semantic resources, we model questions and answers as vectors, and evaluate the relatedness of each QA pair in a shared vector space.  
Answer candidates were chosen using a combination of overlapping non-stop word counts and pattern matching.  
The task is to rank the candidate answers based on their relatedness to the question, and is thus measured in Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR), which are standard metrics in Information Retrieval and Question Answering. Whereas MRR measures the rank of any correct answer, MAP examines the ranks of all the correct answers. In general, MRR is slightly
higher than MAP on the same list of ranked outputs, except that they are the same in the case where each question has exactly one correct answer. 
**One weakness of the distributed approach** is that—unlike symbolic approaches—distributed representations are not very well equipped for dealing with cardinal numbers and proper nouns, especially considering the small dataset. They solve the by adding a words co-occurring count(boolean count and tf*idf count).

4. [A Deep Architecture for Matching Short Texts](https://www.semanticscholar.org/paper/A-Deep-Architecture-for-Matching-Short-Texts-Lu-Li/4aba54ea82bf99ed4690d45051f1b25d8b9554b5).    
（given embeddings of sentences, how can we calculate similarity more concisely）
inner-product cannot sufficiently take into account the complicated interaction between components within the objects, often in a rather nonlinear manner.
5. [Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection](https://www.semanticscholar.org/paper/Dynamic-Pooling-and-Unfolding-Recursive-Socher-Huang/167abf2c9eda9ce21907fcc188d2e41da37d9f0b)
6. [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement](https://www.semanticscholar.org/paper/Pairwise-Word-Interaction-Modeling-with-Deep-He-Lin/0476b7d387d2a6381a784b2b89ccf7baef098f5e) [... a novel neural network architecture that demonstrates state-of-the-art accuracy on three SemEval tasks and two answer selection tasks.]  
most previous neural network approaches are based on sentence modeling, which
first maps each input sentence into a fixed-length vector and then performs comparisons on these representations.
7. [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](https://www.semanticscholar.org/paper/Learning-to-Rank-Short-Text-Pairs-with-Severyn-Moschitti/73d826d4c2363701b88e3e234fe3b8756c0f9671)[模型的架构可以参考] 
Before learning can take place, such pairs(question question pair or query document pair) needs to be mapped from the original space of symbolic words into some feature space encoding various aspects of their relatedness, e.g. lexical, syntactic and semantic. Feature engineering is often a laborious task and may require external knowledge sources that are not always available or difficult to obtain. Recently, deep learning approaches have gained a lot of attention from the research community and industry for their ability to automatically learn optimal feature representation for a given task, while claiming state-of-the-art performance in many tasks.  
In this paper, we present a convolutional neural network architecture for reranking pairs of short texts, where we learn the optimal representation of text pairs and a similarity function to relate them in a supervised way from the available training data.  
Our model demonstrates strong performance on the first task(TREC Question Answering task) beating previous state-of-the-art systems by about 3% absolute points in both MAP and MRR.
8. [Multitask Learning with Deep Neural Networks for Community Question Answering](https://www.semanticscholar.org/paper/Multitask-Learning-with-Deep-Neural-Networks-for-Bonadiman-Uva/ad21f9672634fe1ef2048b58e09d6f85529dfd81)
9. [Inner Attention based Recurrent Neural Networks for Answer Selection](https://www.semanticscholar.org/paper/Inner-Attention-based-Recurrent-Neural-Networks-Wang-Liu/52956422f86722aca6becb67ea4c3ad61f0c1aea)
10. [LSTM-based Deep Learning Models for non-factoid answer selection](https://www.semanticscholar.org/paper/LSTM-based-Deep-Learning-Models-for-non-factoid-Tan-Xiang/24b4746688c15c8580d1000d4f4ac63f5eb79561)

## Undefined

1. [A survey on question answering systems with classification](http://ac.els-cdn.com/S1319157815000890/1-s2.0-S1319157815000890-main.pdf?_tid=51d952d0-1063-11e7-8285-00000aacb35d&acdnat=1490340621_39f523cb0a867f3bb6b3c1235d7709b9)  
Provide a generalized architecture of (retrieval-based)question answering system.
2. [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.semanticscholar.org/paper/WikiQA-A-Challenge-Dataset-for-Open-Domain-Yang-Yih/03fe39386ce90e10ec87f10e00532c5cf30b244f)
3. [Automatic Feature Engineering for Answer Selection and Extraction](https://www.semanticscholar.org/paper/Automatic-Feature-Engineering-for-Answer-Selection-Severyn-Moschitti/7aa63f414a4d7c6e4369a15a04dc5d3eb5da2b0e)
4. [Question Answering Using Enhanced Lexical Semantic Models](https://www.semanticscholar.org/paper/Question-Answering-Using-Enhanced-Lexical-Semantic-Yih-Chang/3e393df4a5731fb7b49cf2f527fed1ee4e6e6942)
5. [A Joint Model for Answer Sentence Ranking and Answer Extraction](https://www.semanticscholar.org/paper/A-Joint-Model-for-Answer-Sentence-Ranking-and-Sultan-Castelli/59018eb4d0a5161a12cdca42cbcb6bf78d73612f)
6. [Learning concept importance using a weighted dependence model]()
7. [Parameterized concept weighting in verbose queries]() [6 and 7 are mentioned as state-of-the-art in Waston lab's paper]