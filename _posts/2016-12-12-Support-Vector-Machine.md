---
layout: default
---
# Support Vector Machine
The SVM is a decision machine and so does not provide posterior probabilities.

## Max Margin Classifier
We will begin by two-class classification problem using linear models of the form. 

$$
y=w^T\phi(x) + b \tag{1}
$$

where $$\phi(x)$$ denotes a fixed feature-space transformation, and $$b$$ is the bias. The training data set comprises N input vectors $$x_1, . . . , x_N$$ , with corresponding target values $$t_1,...,t_N$$ where $$t_n$$ ∈{−1,1}, and new data points $$x$$ are classified according to the *sign* of $$y(x)$$.

We currently assume that the dataset is linear separable, so that, their must exist at least one set of paramter $$w$$ and $$b$$ that a function of form $$(1)$$ satisfies $$y(x_n) > 0$$ for points having $$t_n = +1$$ and $$y(x_n) < 0$$ for points having $$t_n =−1$$, so that $$t_ny(x_n)>0$$ for the whole dataset.

### Differences between SVM and other solutions
There may of course exist many such solutions(perceptron, LR) that separate the classes exactly. Solutions find b

### Model Formalization

## Dual Representation

### linear regression

### svm

### gradient computing

## kenel tricks

## Reference
1. Pattern Recognition and Machine Learning, chapter 6, 7 and Appendix E
2. [Gradient and Lagrange](http://ee263.stanford.edu/notes/gradient-lagrange.pdf)




