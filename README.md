# Creating a neural network with Numpy

A simple neural network project with a single hidden layer built from scratch with NumPy, trained using backpropagation on binary classification datasets. The hidden layer uses ReLU activation, and the output layer uses sigmoid activation.

## Project structure

- **src/nn.py**: core neural network implementation built with Numpy.
- **src/train.py**: training script that uses the network implementation.
- **data/basic_dataset.csv**: small dataset for quick tests, size = (12, 2).
- **data/large_dataset.csv**: larger dataset for more complete training runs, size (2000, 2).

### Installation

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy
```

## Usage

Run the training script from the project root:

```bash
python src/train.py --data-path data/basic_dataset.csv --hidden-size 5 --learning-rate 0.01 --epochs 5000 --test-size 0.2 --seed 42
```

- `--data-path`: Path to CSV dataset with the data (default: `data/basic_dataset.csv`).
- `--hidden-size`: Number of hidden units in the neural network (default: `5`).
- `--learning-rate`: Learning rate (default: `0.01`).
- `--epochs`: Training epochs (default: `5000`).
- `--test-size`: Test split ratio (0-1) (default: `0.2`).
- `--seed`: Random seed for reproducibility (default: `42`).

## Backpropagation algorithm

### 1. Notation and definitions

In this section we will explain the backpropagation algorithm used to train the neural network. We will use the following notation

* $n_l$ is the number of neurons in layer $l$.

* $X \in M_{n_0 \times m}(\mathbb{R})$ is the input data matrix, where each column represents an example ($n_0$ examples) and each row represents a feature ($m$ features).

* $W^{[l]} \in M_{n_l \times n_{l-1}}(\mathbb{R})$ denotes the weight matrix connecting layer $l - 1$ to layer $l$, where each element $w_{jk}^{[l]}$ represents the weight of the connection between neuron $j$ of layer $l$ and neuron $k$ of layer $l - 1$.

* $b^{[l]} \in M_{n_l \times 1}(\mathbb{R})$ is a vector denoting the bias of layer $l$, where $b_j^{[l]}$ represents the bias corresponding to neuron $j$ of layer $l$

* $B^{[l]} \in M_{n_l \times m}(\mathbb{R})$ is a matrix formed by repeating the column vector $b^{[l]}$ $m$ times.

* $z^{[l]} \in M_{n_l \times 1}(\mathbb{R})$ denotes the linear combination of the input to layer $l$ with the parameters $W^{[l]}$ and $b^{[l]}$.

* $Z^{[l]} = [z^{[l](1)}, z^{[l](2)}, \dots, z^{[l](m)}] \in M_{n_l \times m}(\mathbb{R})$ is the matrix formed by stacking the linear combinations of all $m$ examples.

* $a^{[l]} \in M_{n_l \times 1}(\mathbb{R})$ denotes the output vector of layer $l$ of the neural network. In particular, $a^{[L]}$ denotes the output of the neural network and $a^{[0]}$ denotes the of input features to the neural network.

* $A^{[l]} = [a^{[l](1)}, a^{[l](2)}, \dots, a^{[l](m)}] \in M_{n_l \times m}(\mathbb{R})$ is the matrix formed by stacking the activation vectors of all $m$ examples. By convention, $A^{[0]} = X$.

* $g : \mathbb{R} \to \mathbb{R}$ is a non-linear function (such as ReLU or sigmoid). If $M \in M_{c \times d}(\mathbb{R})$ is a matrix (or a vector), we denote $g(M) \in M_{c \times d}(\mathbb{R})$ as the matrix (or vector) obtained by applying the function $g$ to each coordinate of $M$.


* We will denote matrix multiplication with the symbol $*$, scalar multiplication with the symbol $\cdot$, and the element-wise product with the symbol $\odot$.

### 2. Forward propagation

Using the matrix convention defined above, the forward propagation equations for any layer $l$ are written as:

$$Z^{[l]} = W^{[l]} * A^{[l-1]} + B^{[l]}$$
$$A^{[l]} = g(Z^{[l]})$$

In this form, $A^{[L]}$ denotes the final output of the network, where each column corresponds to the prediction for each individual example.

In our specific case $g$ corresponds to the ReLU activation function in the hidden layer and to the sigmoid activation function in the output layer.

### 3. Global Cost Function

Since we are dealing with a binary classification problem we use the Binary Cross-Entropy for the Loss function $\mathcal{L}$:

$$\mathcal{L}(a^{[L]}, y) = - \left( y \cdot \log(a^{[L]}) + (1 - y) \cdot \log(1 - a^{[L]}) \right)$$

When we have $m$ examples, the Global Cost Function $J$ is defined as the average of the loss functions for each individual error:

$$J = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(a^{[L](i)}, y^{(i)})$$

### 4. Backpropagation

Given $J$, our objective is to calculate:
$$\frac{\partial J}{\partial W^{[l]}},\quad \frac{\partial J}{\partial b^{[l]}} \quad \forall l$$

First we calculate the derivatives of the activation functions because we will need them later.

Sigmoid, $\sigma(z)=\frac{1}{1+e^{-z}}$:

$$
\sigma'(z)=\frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right)= \frac{1}{1 + e^{-z}} \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right)=\sigma(z)(1-\sigma(z))
$$

ReLU, $g(z)=\max(0,z)$:

$$
g'(z)=
\begin{cases}
1, & z>0\\
0, & z\le 0
\end{cases}
$$

Since $J$ is the average loss:

$$
\frac{\partial J}{\partial W^{[l]}}
=
\frac{1}{m}
\sum_{i=1}^{m}
\frac{\partial \mathcal{L}^{(i)}}{\partial W^{[l]}}
$$

For a single example $i$, applying the chain rule:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial W^{[l]}}
=
\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[l](i)}}
\cdot
\frac{\partial  A^{[l](i)}}{\partial Z^{[l](i)}}
\cdot
\frac{\partial Z^{[l](i)}}{\partial W^{[l]}}
$$

We can calculate:

$$\frac{\partial  A^{[l](i)}}{\partial Z^{[l](i)}}=g'(Z^{[l](i)})$$
$$\frac{\partial Z^{[l](i)}}{\partial W^{[l]}}=A^{[l-1](i)}$$
Define:
$$
dZ^{[l](i)} =
\frac{\partial \mathcal{L}^{(i)}}{\partial Z^{[l](i)}}
=
\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[l](i)}}
\odot
g'(Z^{[l](i)})
$$

Then:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial W^{[l]}}
=

dZ^{[l](i)}
\cdot
A^{[l-1](i)}

$$

Now we just need:
$$\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[l](i)}}$$

For one example $i$ in the final layer $L$, the loss is:

$$
\mathcal{L}^{(i)}
=
- \left(
y^{(i)} \log\left(a^{[L](i)}\right)
+
(1 - y^{(i)}) \log\left(1 - a^{[L](i)}\right)
\right)
$$

We differentiate with respect to $a^{[L](i)}$:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial a^{[L](i)}}
=
- \left(
y^{(i)} \frac{1}{a^{[L](i)}}
+
(1 - y^{(i)}) \frac{\partial}{\partial a^{[L](i)}} \log(1 - a^{[L](i)})
\right)
$$
Since:
$$
\frac{\partial}{\partial a} \log(1 - a)
=
-\frac{1}{1 - a}
$$

we obtain

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial a^{[L](i)}}
=
- \frac{y^{(i)}}{a^{[L](i)}}
+
\frac{1 - y^{(i)}}{1 - a^{[L](i)}}

=
\frac{a^{[L](i)} - y^{(i)}}{
a^{[L](i)} \left(1 - a^{[L](i)}\right)
}
$$

And finally:

$$
\boxed{
dZ^{[L](i)} = \frac{\partial \mathcal{L}^{(i)}}{\partial a^{[L](i)}} \cdot \sigma'(z^{[L](i)}) 
= \frac{a^{[L](i)} - y^{(i)}}{a^{[L](i)}(1 - a^{[L](i)})} \cdot a^{[L](i)}(1 - a^{[L](i)}) 
= a^{[L](i)} - y^{(i)}
}
$$

For the hidden layer, the dependency path is:

$$
A^{[1](i)} \longrightarrow Z^{[L](i)} \longrightarrow A^{[L](i)} \longrightarrow \mathcal{L}^{(i)}
$$

Thus:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[1](i)}}
=
\frac{\partial \mathcal{L}^{(i)}}{\partial Z^{[L](i)}}
\cdot
\frac{\partial Z^{[L](i)}}{\partial A^{[1](i)}}
$$

Since

$$
Z^{[L](i)} = W^{[L]}A^{[1](i)} + b^{[L]}
$$

we have:

$$
\frac{\partial Z^{[L](i)}}{\partial A^{[1](i)}} = W^{[L]}
$$

Then:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[1](i)}}
=
\left(W^{[L]}\right)^T dZ^{[L](i)}
$$

And finally

$$
\boxed{
dZ^{[L](i)} =
\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[L](i)}}
\cdot
\mathbf{1}\{ Z^{[L](i)} > 0 \}
=
\left(W^{[L]}\right)^T dZ^{[L](i)}
\cdot
\mathbf{1}\{ Z^{[L](i)} > 0 \}
}
$$

where $\mathbf{1}\{ Z^{[L](i)} > 0 \}$ is the derivative of ReLU.

For the bias terms, we consider the dependency path:

$$
b^{[l]} \longrightarrow Z^{[l]} \longrightarrow A^{[l]} \longrightarrow \mathcal{L} \longrightarrow J
$$

Since $J$ is the average loss:

$$
\frac{\partial J}{\partial b^{[l]}}
=
\frac{1}{m}
\sum_{i=1}^{m}
\frac{\partial \mathcal{L}^{(i)}}{\partial b^{[l]}}
$$

For a single example $i$, applying the chain rule:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial b^{[l]}}
=
\frac{\partial \mathcal{L}^{(i)}}{\partial A^{[l](i)}}
\cdot
\frac{\partial A^{[l](i)}}{\partial Z^{[l](i)}}
\cdot
\frac{\partial Z^{[l](i)}}{\partial b^{[l]}}
$$


Recall that for one example $i$:

$$
Z^{[l](i)} = W^{[l]} A^{[l-1](i)} + b^{[l]}
$$

In components:

$$
Z^{[l](i)}_j
=
\sum_{k=1}^{n_{l-1}} w^{[l]}_{jk} A^{[l-1](i)}_k
+
b^{[l]}_j
$$

Therefore, for each neuron $j$:

$$
\frac{\partial Z^{[l](i)}_j}{\partial b^{[l]}_j} = 1
\qquad
\text{and}
\qquad
\frac{\partial Z^{[l](i)}_j}{\partial b^{[l]}_{j'}} = 0 \ \ \text{if } j' \neq j
$$

Hence:

$$
\frac{\partial Z^{[l](i)}}{\partial b^{[l]}} = I

$$

Then:

$$
\frac{\partial \mathcal{L}^{(i)}}{\partial b^{[l]}}
=
dZ^{[l](i)}
$$

Finally, averaging over the $m$ examples:

$$
\boxed{
\frac{\partial J}{\partial b^{[l]}}
=
\frac{1}{m}
\sum_{i=1}^{m}
dZ^{[l](i)}
}
$$


## License

This project is licensed under the MIT License.

