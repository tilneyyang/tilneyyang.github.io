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
When work with tensorflow, this part could be skipped.

#### Parameter initialization

#### Optimization tricks

#### Check if the model is powerful enough to overfit
  * If not, change model structure or make model "larger"
  * If you can overfit: Regularize

## Reference
1. [Practical recommendations for gradient-based training of deep architectures](http://arxiv.org/abs/1206.5533)
2. [UFLDL page on gradient checking](UFLDL page on gradient checking)
3. [CS224d Lecture: Practical tips: gradient checks, overfitting, regularization, activation functions, details](http://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf)<a name="ref3"></a>
