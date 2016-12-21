---
layout: default
---
# Neural Network tips and practices

This post aims at collecting reasons behind the common practices of training neural networks, as well as maintaining a list of reference on how to train them.

## Notes of CS224d lecture<sup><a href="#ref3">[1]</a></sup>
### BackPropagation
In backpropagation, we just want to comput the gradient of example-wise loss with respect to paramters. The main piece is to apply the derivative chain rule wisely. A common property in bp is if
computing the loss(example, paramters) is $$O(n)$$ computation, then so is computing the gradient.

<center><img alter="back propagation" src="/assets/img/back-propagation.png"> </center>
<center><p>Figure 1. back propagation</p></center>

 * **Forword Passing**: visit nodes in topo-sort order, compute value of node given predecessors
 * **Back Propagation**: visit nodes in reverse order, compute gradient with respect to node using gradient with respect to successor

$$
{y_1, y_2, ..., y_n} = successors\:of\:x \\
\dfrac{\partial z}{\partial x} = \sum_{i=1}^{n} \dfrac{\partial z}{\partial y_i} \dfrac{\partial y_i}{\partial x}
$$

Both operations are visualized in Figure 1. 

### Multi-task learing/weight sharing
We already knew that word vectors can be share cross tasks. The main idea of multi-task learning is instead of only word vectors, the hidden layer weights can be shared too. Only the final softmax weights
are different. The cost function of multi-task learning is the sum of all the cross entropy errors.

### General Strategy for Successful NNets
#### Select network structure  appropriate for problem
##### Structure
  * Model: bag of words, recursive vs. recurrent,CNN 
  * Model on: Single words, fixed windows, sentence based, or document level;
  
#### Nonlinearity
<center><img src="/assets/img/nolinearty.png" alter="activation function"></center>
<center><p>Figure 2. Different types of nolinearties.(Left): hard tanh; (center): soft sign, tanh and softmax; (right):relu </p></center>

#### Check for implementation bugs with gradient checks
Gradient check allows you to know that whether there are bugs in your neural network implementation or not. just follow the steps bellow:
1. implement your gradient
2. Implement a finite difference computation by looping through the parameters of your network, adding and subtracting a small epsilon($$∼10^{-4}$$) and estimate derivatives   
$$
f^{\prime}(\theta) \approx \frac{J(\theta^{i+}) - J(\theta^{i-})}{2\epsilon} \qquad\qquad \theta^{i+} = \theta + \epsilon \times e_i
$$

3. compare  the two and make sure they are almost the same
When work with tensorflow, this part could be skipped.

#### Parameter initialization
Before initialize paramter, make sure the input data are normalized to zero mean with small deviation. Some common practices on weight initializing:
 1. initialize hidden layer bias to 0
 2. initialize weights by sampling from $$Unifor(-r, r)$$, where $$r$$ inversely propotional to fan-in(previous layer size) and fan-out(next layer size). Normally, $$\sqrt{\frac{6}{fan-in + fan-out}}$$ for sigmoid, $$4\sqrt{\frac{6}{fan-in + fan-out}}$$ for tanh.

#### Optimization tricks
Gradient descent uses total gradient over all examples per update, it is very slow and should never be used. Most commly used now is Mini-batch Stochastic Gradient Descent with a mini batch size between
 $$[20, 100]$$, the update rule is

$$
\theta^{new} = \theta^{old} - \alpha \nabla_{\theta}J_{t:t+B}(\theta)
$$

where $$B$$ is the mini batch size.

##### Momentum
The idea of momentum is adding a fraction $$v$$ of previous update to current one.

$$
\begin{align}
v &= \mu v - \alpha\nabla_\theta J_t(\theta) \\
\theta^{new} &= \theta^{old} + v

\end{align}
$$

where $$v$$ is initialized to $$0$$.

When the gradient keeps pointing in the same direction, this will increase the size of the steps taken towards the minimum.

<center><img src="/assets/img/single-update-momentum.png" alt="single sgd update with momentum"></center>
<center><p>Figure 3. Single SGD update with Momentum</p></center>
<center><img src="/assets/img/momentum-sgd.png" alt="simple converx function optimization dynamics"></center>
<center><p>Figure 4. Simple convex function optimization dynamics: without momentum(left), with momentum(right)</p></center>
However, when using a lot of momentum, the global learning rate should be reduced. 

##### Learning Rates
The simpleest recipe is to keep it fixed and use the same for all parameters. Better results show by allowing learning rates to decrease Options:
  * reduce by $$0.5$$ when validation error stops improving
  * keep the learning rate constant for the first $$\tau$$ steps, and then reduce by $$O(1/t)$$
  * better to handle learning rate via AdaGrad

##### AdaGrad
AdaGrad adapts differently for each parameter and rare parameters get larger uodates than frequently occuring parameters. Let

$$
\begin{align}
g_{t,i} &= \frac{\partial}{\partial \theta_i^t} \\
\theta_{t,i} &= \theta_{t-1,i} - \frac{\alpha}{\sqrt{\sum_{\tau=1}^tg_{\tau,i}^2}}g_{t,i} \\
\end{align}
$$


#### Check if the model is powerful enough to overfit
  * If not, change model structure or make model "larger"
  * If you can overfit: Regularize

##### Regularize
  * Simple first step: reduce model size by lowering number of units and layers and other parameters
  * Standard L1 or L2 regularization on weights
  * Early stopping: Use parameters that gave best validation error
  * Sparsity constrains on hidden activation, eg, add $$KL(1/N\sum_{n=1}{N}a_i^n\|0.0001)$$ to cost
  * dropout: randomly set $$50%$$ of the inputs to each neuron to $$0$$, and halve the model weights at test time. This prevent feature co-adaption, which means a feature can be useful even 
when other features do not present


## Reference
1. [Practical recommendations for gradient-based training of deep architectures](http://arxiv.org/abs/1206.5533)
2. [UFLDL page on gradient checking](UFLDL page on gradient checking)
3. [CS224d Lecture: Practical tips: gradient checks, overfitting, regularization, activation functions, details](http://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf)<a name="ref3"></a>
