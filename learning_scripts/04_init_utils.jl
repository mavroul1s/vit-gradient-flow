# src/init_utils.jl
using LinearAlgebra

function generate_orthogonal_matrix(rows, cols)
    println("--- Generating Orthogonal Matrix ---")
    println("Requested dimensions: ", rows, "x", cols)
    
    # 1. Start with a standard random normal matrix
    # We use max(rows, cols) to ensure we can create a proper square or wide matrix for QR
    dim = max(rows, cols)
    A = randn(dim, dim)
    
    # 2. Perform QR Decomposition
    # Q is the orthogonal matrix, R is the upper triangular matrix
    Q, R = qr(A)
    
    # 3. Extract the exact dimensions we need
    # If the requested matrix is not square, we slice the Q matrix
    W = Matrix(Q)[1:rows, 1:cols]
    
    return W
end

# --- Test the Function ---

# Define dimensions for a sample weight matrix (e.g., input embedding to hidden state)
dim_in = 5
dim_out = 5

# Generate the weights
W_ortho = generate_orthogonal_matrix(dim_in, dim_out)

println("\nGenerated Matrix W:")
display(round.(W_ortho, digits=4))

# Verify the mathematical property: W^T * W = I
# We use I to represent the Identity matrix. The off-diagonals should be extremely close to 0.
identity_check = W_ortho' * W_ortho

println("\nVerifying W^T * W (Should approximate the Identity Matrix):")
display(round.(identity_check, digits=4))

# Check the determinant (should be 1 or -1 for orthogonal matrices)
det_W = det(W_ortho)
println("\nDeterminant of W: ", round(det_W, digits=4))