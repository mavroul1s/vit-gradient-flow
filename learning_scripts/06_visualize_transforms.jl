# learning_scripts/06_visualize_transforms.jl
using Plots
using LinearAlgebra

println("Loading plotting engine...")

# 1. Create a "dataset" of 2D points that form a perfect circle
theta = range(0, 2pi, length=100)
x = cos.(theta)
y = sin.(theta)
# Combine into a 2x100 matrix (2 features, 100 samples)
circle_points = hcat(x, y)' 

# 2. Define our two different Weight Matrices
# A standard random matrix (how normal neural networks start)
W_random = randn(2, 2)

# An orthogonal matrix (a 45-degree rotation matrix)
angle = pi / 4 
W_ortho = [cos(angle) -sin(angle);
           sin(angle)  cos(angle)]

# 3. Pass the data through the "layers" (Matrix Multiplication)
random_transformed = W_random * circle_points
ortho_transformed = W_ortho * circle_points

# 4. Draw the plots
# aspect_ratio=:equal ensures circles don't look like ovals
p1 = plot(circle_points[1,:], circle_points[2,:], 
          label="Input", title="Original Data", 
          aspect_ratio=:equal, xlims=(-3,3), ylims=(-3,3), lw=2)

p2 = plot(random_transformed[1,:], random_transformed[2,:], 
          label="Output", title="Random Weights\n(Distorted)", 
          aspect_ratio=:equal, xlims=(-3,3), ylims=(-3,3), color=:red, lw=2)

p3 = plot(ortho_transformed[1,:], ortho_transformed[2,:], 
          label="Output", title="Orthogonal Weights\n(Preserved)", 
          aspect_ratio=:equal, xlims=(-3,3), ylims=(-3,3), color=:green, lw=2)

# 5. Combine them into one dashboard
final_dashboard = plot(p1, p2, p3, layout=(1,3), size=(900, 350))

# Display the plot
display(final_dashboard)
println("Plot successfully generated!")