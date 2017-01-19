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

$$
\dfrac {\partial h_j}{\partial h_{j-1}} = 
\begin{bmatrix}
\dfrac {\partial h_j}{\partial h_{j-1}^0} & \dfrac {\partial h_j}{\partial h_{j-1}^1} & \dots & \dfrac {\partial h_j}{\partial h_{j-1}^n}
\end{bmatrix}
= 
\begin{bmatrix}
\dfrac {\partial h_j^0}{\partial h_{j-1}^0} & \dfrac {\partial h_j^0}{\partial h_{j-1}^1} & \dots & \dfrac {\partial h_j^0}{\partial h_{j-1}^n} \\
\dfrac {\partial h_j^1}{\partial h_{j-1}^0} & \dfrac {\partial h_j^1}{\partial h_{j-1}^1} & \dots & \dfrac {\partial h_j^1}{\partial h_{j-1}^n} \\
\vdots & \vdots & \ddots & \vdots \\
\dfrac {\partial h_j^n}{\partial h_{j-1}^0} & \dfrac {\partial h_j^n}{\partial h_{j-1}^1} & \dots & \dfrac {\partial h_j^n}{\partial h_{j-1}^n}
\end{bmatrix}
$$

$$h_{j}^l$$ means the $$l$$-th element in vector $$h_j$$. Note, according to the definition of Jacobian, Given vector $$f$$ with length $$m$$ and $$x$$ with length $$n$$, the Jacobian
 $$\dfrac{\partial \textbf{f}}{\partial \textbf{x}}$$ is a $$m \times n$$ matrix.

Now consider the derivative of each element of the matrix $$\dfrac {\partial h_{j, m}}{\partial h_{j-1, n}}$$, because 

$$
h_j^m = h_j^{[m,1]} = \sum_k W^{[m,k]}*f(h_{j-1})^{[k,1]}
$$

where $$A^{[m,n]}$$ means the element at the $$m$$-th raw, $$n$$-th column of matrix $$A$$. The shapes of $$h_j \text{, } W \text{, } h_{j-1}$$ are $$m \times 1 \text{, } m \times m \text{, } m \times 1$$ respectively.

$$
\begin{align}
\dfrac {\partial h_{j, m}}{\partial h_{j-1, n}} &= \dfrac{\partial}{\partial h_{j-1, n}} \sum_k W^{[m,k]}*f(h_{j-1}])^{[k,1]} \\ 
&= W^{[m,k]}*f^{\prime}(h_{j-1})^{[k,1]} \\
&= W^{[m,k]}*f^{\prime}(h_{j-1}^{[k,1]}) \\
\end{align}
$$

Note, $$f(\bullet)$$ is a pointwise non-linearty. Finally,

$$
\dfrac{\partial h_{j}}{\partial h_{j-1}} = 
\begin{bmatrix}
W^{[0,0]}*f^{\prime}(h_{j-1}^{[0,1]}) & W^{[0,1]}*f^{\prime}(h_{j-1}^{[1,1]}) & \dots & W^{[0,m]}*f^{\prime}(h_{j-1}^{[m,1]}) \\
W^{[1,0]}*f^{\prime}(h_{j-1}^{[0,1]}) & W^{[1,1]}*f^{\prime}(h_{j-1}^{[1,1]}) & \dots & W^{[1,m]}*f^{\prime}(h_{j-1}^{[m,1]}) \\
\vdots & \vdots & \ddots & \vdots \\
W^{[m,0]}*f^{\prime}(h_{j-1}^{[0,1]}) & W^{[m,1]}*f^{\prime}(h_{j-1}^{[1,1]}) & \dots & W^{[m,m]}*f^{\prime}(h_{j-1}^{[m,1]})
\end{bmatrix} 
= Wdiag[f^{\prime}(h_{j-1})]
$$

where $$diag(z) =
\begin{pmatrix}
    z_0                                    \\
      & z_1             &   & \text{0}\\
      &               & \ddots                \\
      & \text{0} &   & z_{n-1}            \\
      &               &   &   & z_n
\end{pmatrix}
$$

Analyzing the norm of the Jacobians,

$$
\| \dfrac{\partial h_j}{\partial h_{j-1}} \| \leq \|W\|\|diag[f^\prime (h_{j-1})] \leq \beta_W\beta_h
$$

where $$\beta_W \text{, } \beta_h$$ are the upper bounds of the norms of $$W \text{and } diag[f^\prime]$$. So,

$$
\|\dfrac {\partial h_t} {\partial h_k}\| = \| \Pi_{j=k+1}^{t} \dfrac {\partial h_j}{\partial h_{j-1}} \| \leq (\beta_W\beta_h)^{t-k}
$$

which could easily become very large or very small.

Vanishing gradient could actually do harm to learning process. The error at a time step ideally can tell a previous time step from
 many steps away to change during backprop. When vanishing gradient happens, information from time steps far away are not taken into 
consideration.In language modelling training, given sentence:
```
Jane walked into the room. John walked in too. It was late in the day. Jane said hi to (?)
```
We can easily find out that the qustion mark should be `Jane`, however this `Jane` depends on the word 16-time-step away. It is hard for
the error signal to flow that long range of  time steps.

 

### Multiple Time Scales

One way to deal with long-term dependencies is to design a model that operates at multiple time scales, so that some parts of the model operate at fine-grained time scales and can handle small details, while other parts operate at coarse time scales and transfer information from the distant past to the present more efficiently.

1. **Adding Skip Connections through Time:** add direct connections from variables in the distant past to variables in the present.

2. **Leaky Units:**
When we accumulate a running average $$μ_t$$ of some value $$v_t$$ by applying the update $$μ_t ← αμ_{t−1} + (1 − α)v_t$$ the $$α$$ parameter is an example of a linear self-connection from $$μ_{t−1}$$ 
to $$μ_{t}$$. Leaky units are hidden units with linear self-connections. A leak unit in rnn could be represented as $$h_{t,i} = α_ih_{t−1,i} + (1 − α_i)F_i(h_{t−1}, x_{t})$$
The standard RNN corresponds to $$α_i = 0$$, while here different values of $$α_i$$ were randomly sampled from $$(0.02, 0.2)$$, allowing some units to react quickly while others 
are forced to change slowly, but also propagate signals and gradients further in time. Note that because $$α < 1$$, the vanishing effect is still present (and gradients can still explode via $$F$$), 
but the time-scale of the vanishing effect can be expanded.

3. **Remove Connections:** remove length-one connections and replace them with longer connections

### LSTM

Like leaky units, gated RNNs(long short-term memory and networks based on the gated recurrent unit etc.) are based on the idea of creating paths through time that have derivatives that
 neither vanish nor explode. Leaky units did this with connection weights that were either manually chosen constants or were parameters. Gated RNNs generalize this to connection weights 
that may change at each time step(conditioned on the context).

A LSTM unit consists of a memory cell $$c_t$$, an *input gate* $$i_t$$, a *forget gate* $$f_t$$, and an *output gate* $$o_t$$. The memory cell caries the memory content of a LSTM unit, while 
the gates control the amount of changes to and exposure of the memory content. The content of the memory cell $$C_t$$ at time-step $$t$$ is update similar to the form of a gated
 leaky neuron.

$$
C_t = f_tC_{t-1} + i_t\tilde{C}_t
$$

where $$\tilde{C}$$ is the candidate memory:

$$
\tilde{C_t} = tanh(Wh_{t-1} + W_{[x]}x_t)
$$

Gates are all sigmoid functions of affined transformation on current input $$x_t$$ and last hidden state $$h_{t-1}$$

$$
\begin{align}
f_t &= \sigma(W_{f}h_{t-1} + U_fx_t + b_f) \\
i_t &= \sigma(W_ih_{t-1} + U_ix_t + b_i) \\
\end{align}
$$

The final hidden state of $$h_t$$ could be computed as

$$
h_t = o_ttanh(C_t)
$$

where $$o_t = \sigma(W_oh_{t-1} + U_ox_t + b_o)$$

### GRU
GRU is more like a leaky unit, except that $$\alpha$$ now becomes  trainable and depend on the context.

$$
h_t = (1-z_t)h_{t-1} + z_t\tilde{h_t}
$$

where $$\tilde{h_t}$$ is the new candidate memory 

$$ 
\tilde{h_t} = tanh(r_t \odot Wh_{t-1} + W_{[x]}x_t)
$$

$$z_t$$ are called *update gate*, $$r_t$$ are called *reset gate*.

$$
\begin{align}
z_t &= \sigma(W_zh_{t-1} + U_zx_t + b_z) \\
r_t &= \sigma(W_rh_{t-1} + u_rx_t + b_r) \\
\end{align}
$$


### Trick for exploding gradient: clipping trick
The intuition is simple, we set a *threshold* to the gradient, make the absolute value of gradient no larger than the threshold.
```python
if math.abs(gradient) > threshold:
    gradient = threshold if gradient > 0 else -threshold
```





## Reference

1. Bengio et al. Deep Learning, chapter 10 Sequence Modeling: Recurrent and Recursive Nets
2. Bengio [Advances in optimizing recurrent networks](http://120.52.73.81/arxiv.org/pdf/1212.0901.pdf)
3. Socher [cs224d classnotes](http://cs224d.stanford.edu/lectures/CS224d-Lecture8.pdf)
4. Bengio et al. [Gated Feedback Recurrent Neural Networks](http://arxiv.org/pdf/1502.02367v3.pdf)
