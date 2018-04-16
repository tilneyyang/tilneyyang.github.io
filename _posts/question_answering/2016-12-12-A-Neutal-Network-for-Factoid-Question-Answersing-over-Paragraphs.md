---
layout: default
---
# A Neural Network Factoid Question Answering over Paragraphs
[[paper](https://cs.umd.edu/~miyyer/pubs/2014_qb_rnn.pdf)]

`This paper note is for group sharing.`

## Novelty and Contribution

 * Text classification based methods based on manually defined string matching or bag of words representations are ineffective on questions with very few individual words(eg. named entities), this paper introduce a dependency tree  recursive neural network(DT-RNN) model that can reason over such questions by modeling textual compositionality;
 * Unlike recurrent neural network,  DT-RNNs are capable of learning word and phrase-level representation and thus are robust to similar sentences with slightly different syntax, which is ideal for the problem domain.

## Model details

<center><img alt="dependency parse of a sentence from a question about Sparta" src="/assets/img/question_answering/dt--setence-on-aparta.png"></center>
<center>Figure 1  dependency parse of a sentence from a question about Sparta</center>

We will show how DT-RNN works through the dependency parse of a sentence from a question about Sparto showed above.

### Some prerequisites
 1. For each word in our vocabulary, we associate it with a vector representation $$x_m \in R^d $$, These representations are stored as columns in an $$d*V$$ dimensional word embedding matrix $$W_e$$, where $$V$$ is the size of the vocabulary;
 2. Each node $$n$$ in the parse tree is associated with a word $$w$$, a word vector $$x_w$$, and a hidden vector $$h_n \in R^d$$, **For internal nodes, this vector is a phrase-level representation, while at leaf nodes it is word vector $$x_w$$ mapped into the hidden space**, because unlike consituency trees where all words reside at the leaf level, internal nodes of dependency trees are associated with words, the DT-RNN has to combine current node's word vector its children's hidden vectors to form $$h_n$$, this process continues recursively up to the root, which represents the entire sentence.
 3. A $$d*d$$ matrix $$W_v$$ to incorporate the word vector $$w_x$$ into $$h_n$$;
 4. For each dependency relation(the DET, POSSESSIVE and POSS symbol in Figure 1) $$r$$, we associate a separate $$d*d$$ matrix $$W_r$$.
 
### Model
Given parse tree(Figure 1), 

we first compute leaf representations. For example, the hidden representation $$h_{helots}$$ is
$$
h_{helots}=f(W_v\cdot{x_b} + b) \tag{1}
$$
where f is a non-linear activation such as $$tanh$$ and $$b$$ is the bias term.

Once all leaves are finished, we move to the interior nodes. Continuing from "helots" to its parent, "called", we compute:
$$
h_{called} = f(W_{DOBJ}\cdot{h_{helots}} + W_v \cdot {x_{called}} + b) \tag{2}
$$

Repeat this process up to the root, which is
$$
h_{depended} = f(W_{NSUBJ} \cdot {h_{economy}} + W_{PREP} \cdot {h_{on}} + W_v \cdot x_{depended}) \tag{3}
$$
and  this is the representation of the entire sentence.

The composition equation for any node $$n$$ with children $$K(n)$$ and word vector $$x_w$$ is 
$$
h(n) = f(W_v \cdot x_w + b + \sum_{k \in K(n)}W_{R(n,k)}\cdot h_k) \tag{4}
$$
where $$R(n,k)$$ is the dependency relation between node $$n$$ and child node $$k$$.

### Training
Intuitively, we want to encourage the vectors of questions to be near their correct answers and far away from incorrect answers. This goal is accomplished with a **contrasive [max-margin object](https://en.wikipedia.org/wiki/Hinge_loss) function**.

For the problem domain, answers themselves are words in other questions, so the answer can be trained in the same feature space as the question text, enabling us to model relationships between answers instead of assuming incorrectly that all answers are independent
(I don't think it's neccesary to model this kind of relation between answers, because as the paper quoted, they don't need a ranked list of answer, any answer that does't match the predefined answer list is treated wrong. And we can't even be sure what will be captured in the embedding vector. However this maybe helpful in other domain where we model the output as independent catogories while they are not.)

 * Given a sentence paired with its correct answer $$c$$, we randomly select $$j$$ incorrect answers from the set of all incorrect answers and denote this subset as $$Z$$. 
 * since $$c$$ is part of vocabulary, it has a word vector $$x_c \in W_e$$, and incorrect answer $$z \in Z$$ is also associated with a vector $$x_z \in W_e$$.
 * we define $$S$$ to be the set of all nodes in the sentence's dependency tree, where an individual node $$s \in S$$ is associated with the hidden vector $$h_s$$.

The error for the sentence is 
$$
C(S,\theta) = \sum_{s \in S}\sum_{z \in Z}L(rank(c, s, Z))max(0, 1 - x_c \cdot h_s + x_z \cdot h_s) \tag{5}
$$
where $$rank(c,s,z)$$ provides the rank of the correct answer with respect to the incorrect answers $$Z$$. we transform this rank into a loss function to optimize the top of the ranked list,$$L(r)=\sum_{i=1}^{r}1/i$$, 

Since $$rank(c, s, Z)$$ is expensive to compute. We approximate it by randomly sampling $$K$$ in correct answers until a violation is observed(x_c \cdot h_s < 1 + x_z \cdot h_s) and set $$rank(c,s,z) = (|Z| - 1)/K$$.

The model minimized the sum of the error over all sentences $$T$$ normalized by the number of nodes $$N$$ in the training set,
$$
J(\theta)=\dfrac{1}{N}\sum_{t \in T} \dfrac{\partial J(t)}{\partial \theta} \tag{6}
$$
### From Sentences to Questions
The model we have just described considers each sentence in a quiz bowl question independently. However, previously-heard sentences within the same question contain useful information that we do not want our model to ignore. While past work on rnn models have been restricted to the sentential and sub-sentential levels, we show that sentence-level representations can be easily combined to generate useful representations at the larger paragraph level.

The simplest and best aggregation method is just to average the representations of each sentence seen so far in a particular question. This method is very powerful and performs better than most of our baselines. We call this averaged DT-RNN model QANTA: a question answering neural network with trans-sentential averaging. 

### Decoding
To obtain the final answer prediction given a partial question, we first generate a feature representation for each sentence within that partial question. This representation is computed by concatenating together the word embeddings and hidden representations averaged over all nodes in the tree as well as the root node’s hidden vector. Finally, we send the average of all of the individual sentence features as input to a logistic regression classifier for answer prediction. **The paper didn't mention how this classifier is trained**.
