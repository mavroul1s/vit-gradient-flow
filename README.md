# Investigating Gradient Flow and Sensitivity in Vision Transformers

**Author:** Nikolaos Mavros  
**Institution:** Department of Electrical and Computer Engineering, University of Thessaly  
**Course:** ECE452 Special Topics in Applied Mathematics  

## Project Overview
The stability and convergence of deep neural networks depend strongly on the properties of their weight matrices and the propagation of gradients through hidden layers. This project investigates how **Orthogonal Initialization** ($W^TW = I$) influences gradient propagation and sensitivity in Vision Transformers (ViTs).

Modern architectures like Transformers rely heavily on automatic differentiation. However, the exact dynamics of gradient flow through the Self-Attention mechanism can be formally analyzed using matrix calculus. By mathematically deriving and computationally analyzing the Jacobian matrices of attention layers, this project aims to understand how initialization strategies affect gradient stability (preventing vanishing or exploding gradients) and model interpretability during training.

## Core Objectives
1. **Theoretical Derivation:** Applying matrix differential rules to derive the explicit Jacobian of the self-attention mechanism.
2. **Computational Verification:** Using Julia's automatic differentiation engine to empirically validate theoretical matrix calculus derivations.
3. **Sensitivity Analysis:** Comparing standard Xavier/Glorot initialization against Orthogonal initialization to evaluate Jacobian conditioning and visualize feature saliency maps.

## Technologies Used
* **Language:** Julia
* **Calculus & Autograd:** `LinearAlgebra`, `ForwardDiff.jl`
* **Environment:** Jupyter Notebooks (`IJulia`)
* **Data:** MNIST (formatted as sequential patches for ViT processing via `MLDatasets.jl`)

## Repository Structure
* `/src`: Core Julia modules containing the self-attention mechanism, initialization logic, and training loops.
* `/learning_scripts`: Sandbox environment for testing Julia matrix operations, broadcasting, and differential syntax.
* `/results`: Generated outputs, including saliency maps, Jacobian condition number plots, and gradient flow visualizations.
* `/paper`: LaTeX source code for the final ECE452 research paper.

## Getting Started
To run the notebooks and scripts locally:
1. Clone this repository.
2. Open the Julia REPL in the project root and instantiate the environment:
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()