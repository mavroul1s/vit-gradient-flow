# learning_scripts/05_simple_relu_network.jl
using LinearAlgebra

# 1. Define the Activation Function
# The dot (.) tells Julia to apply the max function to every single element in the matrix independently.
relu(x) = max.(0.0, x)

# 2. Define the Forward Pass
function forward_pass(X, W1, b1, W2, b2)
    println("--- Starting Forward Pass ---")
    
    # Layer 1: Linear transformation followed by ReLU activation
    # Z1 shape will be (batch_size, hidden_neurons)
    Z1 = X * W1 .+ b1' 
    H1 = relu(Z1)
    println("Hidden Layer (H1) computed. Shape: ", size(H1))
    
    # Layer 2: Linear transformation to the final output
    # Z2 shape will be (batch_size, output_classes)
    Z2 = H1 * W2 .+ b2'
    println("Output Layer (Z2) computed. Shape: ", size(Z2))
    
    return Z2
end

# 3. Initialize Network Dimensions
batch_size = 3      # Number of samples processing at once
input_features = 4  # e.g., length of a flattened patch vector
hidden_neurons = 5  # Size of the intermediate representations
output_classes = 2  # Final prediction categories

# 4. Generate Dummy Data and Parameters
# Random input matrix (X)
X_input = randn(batch_size, input_features) 

# Layer 1 weights and biases
W1 = randn(input_features, hidden_neurons)
b1 = zeros(hidden_neurons) # Initialize biases as zeros

# Layer 2 weights and biases
W2 = randn(hidden_neurons, output_classes)
b2 = zeros(output_classes)

# 5. Execute the Network
final_predictions = forward_pass(X_input, W1, b1, W2, b2)

println("\nFinal Output Matrix:")
display(final_predictions)
println("\nExecution complete.")