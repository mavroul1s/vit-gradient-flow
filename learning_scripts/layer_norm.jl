# learning_scripts/layer_norm.jl
using Statistics

function layer_norm(X, gamma, beta; epsilon=1e-5)
    println("--- Applying Layer Normalization ---")
    
    # X shape: (sequence_length, embedding_dim)
    # We calculate the mean and variance across the embedding dimension (dims=2)
    mu = mean(X, dims=2)
    
    # corrected=false ensures we divide by N, not N-1, matching standard neural net math
    variance = var(X, dims=2, corrected=false) 
    
    # 1. Normalize the input (subtract mean, divide by standard deviation)
    # epsilon prevents division by zero if the variance is incredibly small
    X_normalized = (X .- mu) ./ sqrt.(variance .+ epsilon)
    
    # 2. Scale and shift using the learned parameters
    # The ' operator transposes the vectors so they broadcast correctly across the matrix rows
    output = (X_normalized .* gamma') .+ beta'
    
    return output
end

# --- Test the Function ---

# Define dimensions (4 image patches, embedding dimension of 6)
seq_len = 4
embed_dim = 6

# Let's simulate some heavily distorted outputs from a previous deep layer
# Multiplying by 100 and adding 50 makes the data wildly unstable
X_messy = randn(seq_len, embed_dim) .* 100 .+ 50

# Initialize scale (gamma) to 1.0 and shift (beta) to 0.0
gamma_init = ones(embed_dim)
beta_init = zeros(embed_dim)

println("Original Messy Data (Notice the huge, unstable numbers):")
display(round.(X_messy, digits=2))

# Run the forward pass through our normalization layer
X_clean = layer_norm(X_messy, gamma_init, beta_init)

println("\nNormalized Data (Notice how everything is perfectly tamed around 0):")
display(round.(X_clean, digits=2))

# Let's prove the math worked by checking the new mean and variance of the first row
println("\nProof of Normalization (Row 1):")
println("New Mean (should be ~0.0): ", round(mean(X_clean[1, :]), digits=4))
println("New Variance (should be ~1.0): ", round(var(X_clean[1, :], corrected=false), digits=4))