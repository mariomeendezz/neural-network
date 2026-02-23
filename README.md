# Creating a neural network with Numpy

A simple neural network project with a single hidden layer built from scratch with NumPy, trained using backpropagation on binary classification datasets. The hidden layer uses ReLU activation, and the output layer uses sigmoid activation.

## Project structure

- **src/nn.py**: core neural network implementation built with Numpy.
- **src/train.py**: training script that uses the network implementation.
- **data/basic_dataset.csv**: small dataset for quick tests, size = (12, 2).
- **data/large_dataset.csv**: larger dataset for more complete training runs, size (2000, 2).
- **/docs/backpropagation.html**: mathematical derivation of the algorithm.

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

Full mathematical derivation available in /docs/backpropagation.html

## License

This project is licensed under the MIT License.

