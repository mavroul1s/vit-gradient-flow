import torch
import torch.nn as nn
import torch.optim as optim

print("--- Simple PyTorch Linear Regression ---")

# 1. THE DATA
# Let's create 100 random input numbers (x)
# The true rule we want the network to learn is: y = 2x + 1
x_data = torch.randn(100, 1)
y_data = 2 * x_data + 1

# 2. THE MODEL
# nn.Linear(1, 1) is a single neuron. It takes 1 input and produces 1 output.
# Under the hood, it represents the equation: Output = Weight * Input + Bias
model = nn.Linear(1, 1)

# Let's see what the untrained model guesses the weight and bias are
initial_weight = model.weight.item()
initial_bias = model.bias.item()
print(f"Untrained randomly initialized Weight: {initial_weight:.4f}")
print(f"Untrained randomly initialized Bias: {initial_bias:.4f}\n")

# 3. THE LOSS AND OPTIMIZER
# Mean Squared Error (how far off the guess is from the true y)
criterion = nn.MSELoss()
# Stochastic Gradient Descent (the algorithm that updates the weight/bias)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. THE TRAINING LOOP
epochs = 100
for epoch in range(epochs):
    
    # Step A: Clear old gradients
    optimizer.zero_grad()
    
    # Step B: Forward pass (make a guess)
    predictions = model(x_data)
    
    # Step C: Calculate the error
    loss = criterion(predictions, y_data)
    
    # Step D: Backward pass (calculate the derivatives automatically)
    loss.backward()
    
    # Step E: Update the weight and bias based on the derivatives
    optimizer.step()
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3} | Loss: {loss.item():.6f}")

# 5. THE RESULTS
print("\n--- Training Complete ---")
final_weight = model.weight.item()
final_bias = model.bias.item()

print(f"Target Weight: 2.0  | Model Learned: {final_weight:.4f}")
print(f"Target Bias:   1.0  | Model Learned: {final_bias:.4f}")