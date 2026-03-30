# learning_scripts/08_trainable_network.jl
using LinearAlgebra
using Statistics

println("Initializing Neural Network...")

# 1. Activation Function & Its Derivative
sigmoid(x) = 1.0 / (1.0 + exp(-x))
sigmoid_derivative(output) = output * (1.0 - output)

# Wrap everything in a function to fix variable scoping and improve speed
function train_network()
    # 2. The Dataset (XOR Logic Gate)
    X = [0.0 0.0;
         0.0 1.0;
         1.0 0.0;
         1.0 1.0]

    Y = [0.0; 1.0; 1.0; 0.0]

    # 3. Network Architecture
    input_size = 2
    hidden_size = 4
    output_size = 1

    W1 = randn(input_size, hidden_size)
    b1 = zeros(1, hidden_size)
    W2 = randn(hidden_size, output_size)
    b2 = zeros(1, output_size)

    learning_rate = 0.5
    epochs = 5000

    println("Starting Training Loop for $epochs epochs...")

    # Declare a variable to hold our final answer
    local final_predictions 

    # 4. The Training Loop
    for epoch in 1:epochs
        # --- FORWARD PASS ---
        Z1 = X * W1 .+ b1
        A1 = sigmoid.(Z1)
        
        Z2 = A1 * W2 .+ b2
        A2 = sigmoid.(Z2)
        
        # --- CALCULATE LOSS ---
        error = Y .- A2
        loss = mean(error.^2)
        
        # --- BACKWARD PASS ---
        dZ2 = error .* sigmoid_derivative.(A2)
        dW2 = A1' * dZ2
        db2 = sum(dZ2, dims=1)
        
        dZ1 = (dZ2 * W2') .* sigmoid_derivative.(A1)
        dW1 = X' * dZ1
        db1 = sum(dZ1, dims=1)
        
        # --- UPDATE WEIGHTS ---
        W2 .+= learning_rate .* dW2
        b2 .+= learning_rate .* db2
        W1 .+= learning_rate .* dW1
        b1 .+= learning_rate .* db1
        
        # Print progress
        if epoch % 1000 == 0
            println("Epoch $epoch | Loss: ", round(loss, digits=6))
        end
        
        # Save the very last A2 before the loop ends
        if epoch == epochs
            final_predictions = A2
        end
    end

    println("\nTraining Complete!")
    println("Target Outputs:     ", Y')
    println("Network Predictions:", round.(final_predictions', digits=4))
end

# Execute the function
train_network()