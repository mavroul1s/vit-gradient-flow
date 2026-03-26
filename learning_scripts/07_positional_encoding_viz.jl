# learning_scripts/07_positional_encoding_viz.jl
using Plots

println("Calculating Positional Encodings...")

function generate_positional_encoding(seq_len, d_model)
    # Initialize an empty matrix of zeros
    pe = zeros(seq_len, d_model)
    
    # Iterate through every patch position and every embedding dimension
    for pos in 1:seq_len
        for i in 0:2:(d_model-1)
            # The scaling factor that controls the frequency of the waves
            div_term = exp(i * -log(10000.0) / d_model)
            
            # Even dimensions get the Sine wave
            pe[pos, i+1] = sin(pos * div_term)
            
            # Odd dimensions get the Cosine wave
            if i+2 <= d_model
                pe[pos, i+2] = cos(pos * div_term)
            end
        end
    end
    return pe
end

# Define parameters: 100 image patches, each with a vector size of 128
sequence_length = 100
embedding_dimension = 128

# Generate the matrix
pe_matrix = generate_positional_encoding(sequence_length, embedding_dimension)

println("Rendering heatmap...")

# Plot the matrix as a heatmap
pe_plot = heatmap(
    pe_matrix,
    title="Vision Transformer Positional Encodings",
    xlabel="Embedding Dimension (Depth)",
    ylabel="Sequence Position (Patch Index)",
    color=:viridis, # A standard, colorblind-friendly scientific colormap
    size=(800, 500),
    yflip=true # Flips the Y-axis so patch 1 is at the top
)

display(pe_plot)
println("Plot generated successfully!")