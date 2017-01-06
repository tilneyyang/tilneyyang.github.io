---
layout: default
---
# Recurrent Neural Networks
Compared to vanilla neural networks, there are two things we need to know first about rnn:

1. A recurrent neural network is a neural network that is specialized for processing a sequence of values $$x^{(1)}.....x^{(t)}$$. Recurrent networks can scale to much longer sequences.

2. Most recurrent networks can also process sequences of variable length.

## Overview: A Simple RNN And The Computational Graph
Essentially, any function involving recurrence can be considered a recurrent neural network. Many recurrent neural networks use Eq. 1 or similar equation to define the values of their hidden units.

$$
{h^{(t)} = f(h^{(t-1)}, x^{(t)}; \theta)} \tag{1}
$$

<center><img alt="rnn with no out puts" src="/assets/img/rnn_noout.png"></center>
<center>Figure 0: A recurrent network with no outputs. This recurrent network just processes information from the input x by incorporating it into the state h that is passed forward through time.
 (Left) Circuit diagram. The black square indicates a delay of 1 time step. 
(Right) The same network seen as an unfolded computational graph, where each node is now associated with one particular time instance.</center>

Most RNN may have a output layer,
$$
y^{(t)} = g(h^{(t)};\theta)
$$
where $$g$$ could be the softmax function for classification tasks.

## The Challenges of Long-Term Dependencies


## The Vanishing Gradient problems
Let disscuss this problem under a more realistic RNN formulation:

$$
\begin{align}
h_t &= Wf(h_{t-1}) + W^{(hs)}x_{[t]} \\
\hat{y}_t &= W^{(S)}f(h_t)
\end{align}
$$

where $$W$$ is the shared hidden layer parameter, $$W^{(hs)}$$ is the input transformation paramter, and the $$W^{(S)}$$ is the output softmax parameter.

Let $$E$$ be the total error, and $$E_t$$ be the error at a specified time step $$t$$. Let $$T$$ be the total time step, then

$$
\frac{\partial{E}}{\partial{W}} = \sum^T_{t=1}\frac{\partial{E_t}}{\partial{W}}
$$

Apply chain rule to $$\frac{\partial{E_T}}{\partial{W}}$$,

$$
\frac{\partial{E_T}}{\partial{W}} = \sum^t_{k=1} \frac{\partial{E_t}}{\partial{y_t}} \frac{\partial{y_t}}{\partial{h_t}} \frac{\partial{h_t}}{\partial{h_k}} \frac{\partial{h_k}}{\partial{W}}
$$

Consider $$\frac{\partial{h_t}}{\partial{h_k}}$$,

$$
\frac{\partial{h_t}}{\partial{h_k}} = \Pi^t_{j=k+1} \frac {\partial h_j}{\partial h_{j-1}}
$$ 

$$\frac {\partial h_j}{\partial h_{j-1}}$$ is the Jacobian Matrix of $$h_t = Wf(h_{t-1}) + W^{(hs)}x_{[t]}$$.



The basic problem is that gradients propagated over many stages tend to either vanish \(most of the time\) or explode \(rarely, but with much damage to the optimization\). 
Even if we assume that the parameters are such that the recurrent network is stable \(can store memories, with gradients not exploding\), the difficulty with long-term 
dependencies arises from the exponentially smaller weights given to long-term interactions \(involving the multiplication of many Jacobians\) compared to short-term ones.

Consider a simple RNN lacking inputs $$x$$ and nonlinear behavior.

$$
h^{(t)} = W^Th^{(t-1)}
$$

which could simplified to

$$
h^{(t)} = (W^t)^Th^{(0)}
$$

and if $$W$$ admits an eigendecomposition

$$
W=Q{\Lambda}Q^T
$$

then recurrence may be simplified further to:

$$
h^{(t)} = Q^T{\Lambda}^{t}Qh^{(0)}
$$

The eigenvalues are raised to the power of $$t$$ causing eigenvalues with magnitude
less than one to decay to zero and eigenvalues with magnitude greater than one to
explode. Any component of $$h^{(0)}$$ that is not   aligned with the largest eigenvector will eventually be discarded.

Below we will talk about some methods to overcome the difficulty of learning long-term dependencies.

### Multiple Time Scales

One way to deal with long-term dependencies is to design a model that operates at multiple time scales, so that some parts of the model operate at fine-grained time scales and can handle small details, while other parts operate at coarse time scales and transfer information from the distant past to the present more efficiently.

1. ** Adding Skip Connections through Time: **add direct connections from variables in the distant past to variables in the present.

2. ** Leaky Units**

When we accumulate a running average $$μ^(t)$$ of some value $$v^(t)$$ by applying the update $$μ^{(t)} ← αμ^{(t−1)} + (1 − α)v^{(t)}$$ the $$α$$ parameter is an example of a linear self-connection from $$μ^{(t−1)}$$ to $$μ^{(t)}$$. Leaky units are hidden units with linear self-connections. A leak unit in rnn could be represented as $$h_{t,i} = α_ih_{t−1,i} + (1 − α_i)F_i(h_{t−1}, x_{t})$$.

The standard RNN corresponds to $$α_i = 0$$, while here different values of $$α_i$$ were randomly sampled from $$(0.02, 0.2)$$, allowing some units to react quickly while others are forced to change slowly, but also propagate signals and gradients further in time. Note that because $$α < 1$$, the vanishing effect is still present (and gradients can still explode via $$F$$), but the time-scale of the vanishing effect can be expanded.

3. ** Remove Connections: ** remove length-one connections and replace them with longer connections

 ### LSTM

Like leaky units, gated RNNs(long short-term memory and networks based on the gated recurrent unit etc.) are based on the idea of creating paths through time that have derivatives that neither vanish nor explode. Leaky units did this with connection weights that were either manually chosen constants or were parameters. Gated RNNs generalize this to connection weights that may change at each time step(conditioned on the context).

![vallina rnn](images/vallina_rnn.png)



![LSTM](images/lstm.png)

self loop: $$s_i^{(t)} = f_i^{(t)}s_i^{(t-1)} + g_i^{(t)}{\sigma(b_i + W_i[h^{(t-1)}, x^{(t)}])}

$$

output state: $$h_i^{(t)} = tanh(s_i^{(t)})q_i^{(t)} $$

forget gate: $$f_i^{(t)} = \sigma(b_i^f + W_i^f[h^{(t-1)}, x^{(t)}])$$



external input gate: $$g_i^{(t)} = \sigma(b_i^g + W_i^g[h^{(t-1)}, x^{(t)}])$$



output gate: $$q_i^{(t)} = \sigma(b_i^o + W_i^o[h^{(t-1)}, x^{(t)}]) $$

注:

1. 上面公式中$$[a, b]$$代表连接$$a, b $$两个列表， $$[(1,2), (2,3)] = (1,2,2,3)$$

2. deep learning一书中并不是将$$h^{(t-1)}$$和$$x^{(t)}$$做连接，而是对他们两个各有一套参数

### GRU
dl book P414 下方三个公式

### Explicit Memory

![key-values memory network](images/kvmn.png)

MemNNs are class of models which have four component networks \(which may or may not have shared parameters\):

I: \(input feature map\) convert incoming data to the internal feature representation.

G: \(generalization\) update memories given new input.

O: produce new output \(in feature representation space\) given the memories.

R: \(response\) convert output O into a response seen by the outside world.



## Reference



1. Bengio et al. Deep Learning, chapter 10 Sequence Modeling: Recurrent and Recursive Nets

2. Bengio et al. [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

3. Bengio [Advances in optimizing recurrent networks](http://120.52.73.81/arxiv.org/pdf/1212.0901.pdf)

4. Mikolov et al. [Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)

5. Socher [cs224d classnotes](http://cs224d.stanford.edu/lectures/CS224d-Lecture8.pdf)

6. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

7.




