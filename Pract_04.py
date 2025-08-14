def mcculloch_pitts_and_not(a,b):
  w1 = 1
  w2 = -1

  theta = 1

  weighted_sum = a * w1 + b * w2
  if weighted_sum > theta:
    return 1
  else:
    return 0

print("A B | A AND NOT B")
for a in [0,1]:
  for b in [0,1]:
    print(f"{a} {b} | {mcculloch_pitts_and_not(a,b)}")

________________________________________________________________________________________________________________________



import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR)
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output labels
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed for reproducibility
np.random.seed(42)

# Network architecture
input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1

# Weights and biases
weights_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Store loss for plotting
losses = []

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(x, weights_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_input = np.dot(hidden_layer_output, weights_output) + bias_output
    final_output = sigmoid(final_input)

    # Loss (Mean Squared Error)
    error = y - final_output
    loss = np.mean(np.square(error))
    losses.append(loss)

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(weights_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += x.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Final predictions
print("Final predictions after training:")
print(np.round(final_output))

# Plot loss curve
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()


____________________________________________________________________________________________________________________

