# Investigating Gradient Flow and Sensitivity in Vision Transformers

[cite_start]**Author:** Nikolaos Mavros [cite: 2]
[cite_start]**Institution:** Department of Electrical and Computer Engineering, University of Thessaly [cite: 3, 4]
**Course:** ECE452 Special Topics in Applied Mathematics

## Project Overview
[cite_start]This project investigates how Orthogonal Initialization influences gradient propagation and sensitivity in Vision Transformers (ViTs)[cite: 11]. [cite_start]Modern architectures rely on automatic differentiation and backpropagation, which can be formally analyzed using matrix calculus[cite: 8]. [cite_start]By analyzing the Jacobian matrices of attention layers, this project aims to understand how initialization affects gradient stability and interpretability during training[cite: 12].

## Repository Structure
* `/notebooks`: Interactive Julia notebooks verifying mathematical derivations.
* `/src`: Core Julia modules for the self-attention mechanism and weight initialization.
* `/learning_scripts`: Small Julia scripts for practicing matrix operations.
* `/results`: Generated saliency maps and gradient flow visualizations.
* `/paper`: LaTeX source code for the final research paper.