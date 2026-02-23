import argparse
import numpy as np
from nn import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simpleneural network")
    parser.add_argument("--data-path", type=str, default="data/basic_dataset.csv", help="Path to CSV dataset.")
    parser.add_argument("--hidden-size", type=int, default=5, help="Number of hidden units.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load data
    data = np.loadtxt(args.data_path, delimiter=",", skiprows=1)
    X = data[:, :-1].T
    y = data[:, -1].reshape(1, -1)

    # Train/test split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(X.shape[1])
    test_count = max(1, int(X.shape[1] * args.test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    X_train, y_train = X[:, train_idx], y[:, train_idx]
    X_test, y_test = X[:, test_idx], y[:, test_idx]

    # Hyperparameters
    input_size = X_train.shape[0]
    hidden_size = args.hidden_size
    output_size = y_train.shape[0]
    learning_rate = args.learning_rate
    epochs = args.epochs

    # Initialize neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(learning_rate, X_train, y_train, epochs)

    # Evaluate on train
    train_predictions = nn.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test
    predictions = nn.predict(X_test)
    test_accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
