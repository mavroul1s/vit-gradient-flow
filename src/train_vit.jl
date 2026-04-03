# =============================================================================
# train_vit.jl — Lightweight Vision Transformer Training on MNIST
# =============================================================================
# Project : Investigating Gradient Flow and Sensitivity in Vision Transformers
#           via Matrix Calculus and Orthogonal Initialization
# Author  : Nikolaos Mavros — University of Thessaly, ECE452
# =============================================================================
#
# This script trains two small ViT models on MNIST:
#   1. Baseline  → Xavier/Glorot initialization
#   2. Experiment → Orthogonal initialization (WᵀW ≈ I)
#
# It logs per-epoch: loss, accuracy, gradient norms per layer, and periodically
# computes Jacobian condition numbers through the attention layers.
# Weights are saved via JLD2 for later inference and saliency analysis.
#
# Target hardware: 2024 ThinkPad — Ryzen 5, 16 GB RAM, CPU-only.
# =============================================================================

using Pkg

# ── Ensure dependencies ──────────────────────────────────────────────────────
const REQUIRED_PKGS = [
    "Flux", "Zygote", "MLDatasets", "JLD2", "ProgressMeter", "Random", "Statistics"
]
installed = keys(Pkg.project().dependencies)
for p in REQUIRED_PKGS
    if !(p in installed)
        @info "Installing $p …"
        Pkg.add(p)
    end
end

using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, DataLoader
using Zygote
using MLDatasets
using LinearAlgebra
using Statistics
using Random
using JLD2
using ProgressMeter

# ── Reproducibility ──────────────────────────────────────────────────────────
Random.seed!(42)

# ═════════════════════════════════════════════════════════════════════════════
# §1  HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
const IMG_SIZE      = 28          # MNIST is 28×28
const PATCH_SIZE    = 7           # 7×7 patches → 4×4 grid = 16 patches
const NUM_PATCHES   = (IMG_SIZE ÷ PATCH_SIZE)^2   # 16
const PATCH_DIM     = PATCH_SIZE * PATCH_SIZE      # 49 (flattened patch)
const D_MODEL       = 64          # embedding dimension
const NUM_HEADS     = 4           # attention heads
const HEAD_DIM      = D_MODEL ÷ NUM_HEADS          # 16
const MLP_HIDDEN    = 128         # feed-forward hidden dim
const NUM_BLOCKS    = 2           # transformer encoder blocks
const NUM_CLASSES   = 10          # digits 0–9
const DROPOUT_RATE  = 0.1

const BATCH_SIZE    = 128
const EPOCHS        = 15
const LR            = 1e-3
const WEIGHT_DECAY  = 1e-4

# How often (in epochs) to compute Jacobian condition numbers
const JACOBIAN_EVERY = 5

# Output directory for weights and logs
const OUT_DIR = "results"
mkpath(OUT_DIR)

# ═════════════════════════════════════════════════════════════════════════════
# §2  INITIALIZATION STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════
"""
    xavier_init(dims...) → Matrix{Float32}

Standard Xavier/Glorot uniform initialization.
"""
function xavier_init(rng::AbstractRNG, dims...)
    fan_in  = dims[end-1] isa Integer ? dims[end-1] : prod(dims[1:end-1])
    fan_out = dims[end]
    limit   = Float32(sqrt(6.0 / (fan_in + fan_out)))
    return (rand(rng, Float32, dims...) .- 0.5f0) .* 2.0f0 .* limit
end
xavier_init(dims...) = xavier_init(Random.default_rng(), dims...)

"""
    orthogonal_init(dims...) → Matrix{Float32}

Orthogonal initialization via QR decomposition.
For non-square matrices, pads to square, computes QR, then truncates.
Guarantees singular values = 1 (preserves gradient norms).
"""
function orthogonal_init(rng::AbstractRNG, dims...)
    # For weight matrices we expect exactly 2 dims
    rows, cols = dims[1], dims[2]
    n = max(rows, cols)
    A = randn(rng, Float32, n, n)
    Q, _ = qr(A)
    Q_mat = Matrix{Float32}(Q)
    return Q_mat[1:rows, 1:cols]
end
orthogonal_init(dims...) = orthogonal_init(Random.default_rng(), dims...)

# ═════════════════════════════════════════════════════════════════════════════
# §3  MODEL COMPONENTS (manual, for full gradient visibility)
# ═════════════════════════════════════════════════════════════════════════════

# ── 3.1 Patch Embedding ──────────────────────────────────────────────────────
"""
Linearly project flattened patches into d_model dims and prepend a CLS token.
Adds learnable positional embeddings.
"""
struct PatchEmbedding
    proj::Dense                               # PATCH_DIM → D_MODEL
    cls_token::AbstractArray{Float32, 2}      # (D_MODEL, 1)
    pos_embed::AbstractArray{Float32, 2}      # (D_MODEL, NUM_PATCHES + 1)
end

Flux.@layer PatchEmbedding

function PatchEmbedding(init_fn)
    # FIXED: Flipped PATCH_DIM and D_MODEL to match (out, in)
    proj = Dense(init_fn(D_MODEL, PATCH_DIM), zeros(Float32, D_MODEL))
    cls  = randn(Float32, D_MODEL, 1) .* 0.02f0
    pos  = randn(Float32, D_MODEL, NUM_PATCHES + 1) .* 0.02f0
    return PatchEmbedding(proj, cls, pos)
end

function (pe::PatchEmbedding)(patches)
    # patches: (PATCH_DIM, NUM_PATCHES, batch)
    B = size(patches, 3)
    x = pe.proj(reshape(patches, PATCH_DIM, :))          # (D_MODEL, NUM_PATCHES * B)
    x = reshape(x, D_MODEL, NUM_PATCHES, B)              # (D_MODEL, NUM_PATCHES, B)
    cls_expanded = repeat(pe.cls_token, 1, 1, B)          # (D_MODEL, 1, B)
    x = cat(cls_expanded, x; dims=2)                      # (D_MODEL, NUM_PATCHES+1, B)
    x = x .+ pe.pos_embed                                 # broadcast over batch dim
    return x
end

# ── 3.2 Multi-Head Self-Attention ────────────────────────────────────────────
"""
Explicit multi-head self-attention.
Stores WQ, WK, WV, WO as plain Dense layers for easy gradient inspection.
"""
struct MultiHeadAttention
    wq::Dense
    wk::Dense
    wv::Dense
    wo::Dense
end

Flux.@layer MultiHeadAttention

function MultiHeadAttention(init_fn)
    wq = Dense(init_fn(D_MODEL, D_MODEL), zeros(Float32, D_MODEL))
    wk = Dense(init_fn(D_MODEL, D_MODEL), zeros(Float32, D_MODEL))
    wv = Dense(init_fn(D_MODEL, D_MODEL), zeros(Float32, D_MODEL))
    wo = Dense(init_fn(D_MODEL, D_MODEL), zeros(Float32, D_MODEL))
    return MultiHeadAttention(wq, wk, wv, wo)
end

function (mha::MultiHeadAttention)(x)
    # x: (D_MODEL, seq_len, batch)
    D, S, B = size(x)

    # Linear projections
    Q = mha.wq(reshape(x, D, :))   # (D_MODEL, S*B)
    K = mha.wk(reshape(x, D, :))
    V = mha.wv(reshape(x, D, :))

    # Reshape to (HEAD_DIM, NUM_HEADS, seq_len, batch)
    Q = reshape(Q, HEAD_DIM, NUM_HEADS, S, B)
    K = reshape(K, HEAD_DIM, NUM_HEADS, S, B)
    V = reshape(V, HEAD_DIM, NUM_HEADS, S, B)

    # Scaled dot-product attention per head
    # scores: (S, S, NUM_HEADS, B)
    scale = Float32(1.0 / sqrt(HEAD_DIM))

    # Permute to (HEAD_DIM, S, NUM_HEADS, B) for batched matmul
    Q = permutedims(Q, (1, 3, 2, 4))   # (HEAD_DIM, S, NUM_HEADS, B)
    K = permutedims(K, (1, 3, 2, 4))
    V = permutedims(V, (1, 3, 2, 4))

    # Compute attention scores via einsum-style batched matmul
    # QᵀK: for each (head, batch), compute (S, S) attention matrix
    # We'll use a reshape trick: merge head and batch dims
    Q_flat = reshape(Q, HEAD_DIM, S, NUM_HEADS * B)   # (HEAD_DIM, S, H*B)
    K_flat = reshape(K, HEAD_DIM, S, NUM_HEADS * B)
    V_flat = reshape(V, HEAD_DIM, S, NUM_HEADS * B)

    # Batched matmul: Qᵀ · K → (S, S, H*B)
    scores = batched_mul(
        permutedims(Q_flat, (2, 1, 3)),   # (S, HEAD_DIM, H*B)
        K_flat                             # (HEAD_DIM, S, H*B)
    ) .* scale                             # (S, S, H*B)

    # Softmax along the key dimension (dim 2)
    attn_weights = softmax(scores; dims=2)    # (S, S, H*B)

    # Weighted sum of values: (HEAD_DIM, S, H*B)
    out = batched_mul(V_flat, permutedims(attn_weights, (2, 1, 3)))  # (HEAD_DIM, S, H*B)

    # Reshape back to (D_MODEL, S, B)
    out = reshape(out, HEAD_DIM, S, NUM_HEADS, B)
    out = permutedims(out, (1, 3, 2, 4))         # (HEAD_DIM, NUM_HEADS, S, B)
    out = reshape(out, D_MODEL, S, B)

    # Output projection
    out_flat = mha.wo(reshape(out, D_MODEL, :))   # (D_MODEL, S*B)
    return reshape(out_flat, D_MODEL, S, B)
end

# Helper: batched matrix multiply
function batched_mul(A, B)
    # A: (m, k, batch), B: (k, n, batch) → (m, n, batch)
    Flux.batched_mul(A, B)
end

# ── 3.3 Feed-Forward Network ────────────────────────────────────────────────
struct FeedForward
    fc1::Dense
    fc2::Dense
end

Flux.@layer FeedForward

function FeedForward(init_fn)
    # FIXED: Flipped input and output dimensions for both layers
    fc1 = Dense(init_fn(MLP_HIDDEN, D_MODEL), zeros(Float32, MLP_HIDDEN), gelu)
    fc2 = Dense(init_fn(D_MODEL, MLP_HIDDEN), zeros(Float32, D_MODEL))
    return FeedForward(fc1, fc2)
end

function (ff::FeedForward)(x)
    # x: (D_MODEL, S, B)
    D, S, B = size(x)
    h = ff.fc1(reshape(x, D, :))     # (MLP_HIDDEN, S*B)
    h = ff.fc2(h)                     # (D_MODEL, S*B)
    return reshape(h, D, S, B)
end

# ── 3.4 Transformer Block ───────────────────────────────────────────────────
struct TransformerBlock
    ln1::LayerNorm
    attn::MultiHeadAttention
    ln2::LayerNorm
    ff::FeedForward
end

Flux.@layer TransformerBlock

function TransformerBlock(init_fn)
    ln1  = LayerNorm(D_MODEL)
    attn = MultiHeadAttention(init_fn)
    ln2  = LayerNorm(D_MODEL)
    ff   = FeedForward(init_fn)
    return TransformerBlock(ln1, attn, ln2, ff)
end

function (tb::TransformerBlock)(x)
    # Pre-norm residual connections
    # x: (D_MODEL, seq_len, batch)
    D, S, B = size(x)

    # Self-attention with residual
    x_norm = tb.ln1(reshape(x, D, :))
    x_norm = reshape(x_norm, D, S, B)
    x = x .+ tb.attn(x_norm)

    # Feed-forward with residual
    x_norm2 = tb.ln2(reshape(x, D, :))
    x_norm2 = reshape(x_norm2, D, S, B)
    x = x .+ tb.ff(x_norm2)

    return x
end

# ── 3.5 Full Vision Transformer ─────────────────────────────────────────────
struct ViT
    patch_embed::PatchEmbedding
    blocks::Vector{TransformerBlock}
    ln_final::LayerNorm
    head::Dense
end

Flux.@layer ViT

function ViT(init_fn)
    pe     = PatchEmbedding(init_fn)
    blocks = [TransformerBlock(init_fn) for _ in 1:NUM_BLOCKS]
    ln     = LayerNorm(D_MODEL)
    # FIXED: Flipped NUM_CLASSES and D_MODEL
    head   = Dense(init_fn(NUM_CLASSES, D_MODEL), zeros(Float32, NUM_CLASSES))
    return ViT(pe, blocks, ln, head)
end

function (model::ViT)(patches)
    # patches: (PATCH_DIM, NUM_PATCHES, batch)
    x = model.patch_embed(patches)           # (D_MODEL, NUM_PATCHES+1, B)
    for block in model.blocks
        x = block(x)
    end
    D, S, B = size(x)
    # Take CLS token (position 1)
    cls = x[:, 1, :]                         # (D_MODEL, B)
    cls = model.ln_final(cls)                # (D_MODEL, B)
    logits = model.head(cls)                 # (NUM_CLASSES, B)
    return logits
end

# ═════════════════════════════════════════════════════════════════════════════
# §4  DATA LOADING — MNIST as patch sequences
# ═════════════════════════════════════════════════════════════════════════════
"""
Convert a batch of MNIST images (28×28) into patch sequences.
Returns (PATCH_DIM, NUM_PATCHES, batch_size).
"""
function images_to_patches(images)
    # images: (28, 28, batch) already Float32
    B = size(images, 3)
    n_per_side = IMG_SIZE ÷ PATCH_SIZE   # 4
    patches = zeros(Float32, PATCH_DIM, NUM_PATCHES, B)

    idx = 1
    for row in 1:n_per_side
        for col in 1:n_per_side
            r_start = (row - 1) * PATCH_SIZE + 1
            c_start = (col - 1) * PATCH_SIZE + 1
            patch = @view images[r_start:r_start+PATCH_SIZE-1,
                                 c_start:c_start+PATCH_SIZE-1, :]
            patches[:, idx, :] .= reshape(patch, PATCH_DIM, B)
            idx += 1
        end
    end
    return patches
end

function load_mnist()
    @info "Loading MNIST …"
    train_x, train_y = MLDatasets.MNIST(split=:train)[:]
    test_x,  test_y  = MLDatasets.MNIST(split=:test)[:]

    # Normalise to [0, 1] Float32
    train_x = Float32.(train_x)
    test_x  = Float32.(test_x)

    # One-hot encode labels
    train_labels = onehotbatch(train_y, 0:9)
    test_labels  = onehotbatch(test_y,  0:9)

    # Convert to patches
    train_patches = images_to_patches(train_x)
    test_patches  = images_to_patches(test_x)

    @info "Train patches: $(size(train_patches)), Test patches: $(size(test_patches))"

    train_loader = DataLoader(
        (train_patches, train_labels);
        batchsize=BATCH_SIZE, shuffle=true, partial=false
    )
    test_loader = DataLoader(
        (test_patches, test_labels);
        batchsize=BATCH_SIZE, shuffle=false, partial=true
    )

    return train_loader, test_loader
end

# ═════════════════════════════════════════════════════════════════════════════
# §5  GRADIENT FLOW DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════
"""
    collect_gradient_norms(grads, model) → Dict{String, Float64}

Walk the model parameter tree and record the L2 norm of each gradient tensor.
"""
function collect_gradient_norms(grads, model)
    norms = Dict{String, Float64}()
    for (name, param) in pairs(Flux.trainable(model))
        g = grads[param]
        if g !== nothing
            norms[string(name)] = Float64(norm(vec(g)))
        end
    end
    return norms
end

"""
    compute_jacobian_condition(model, sample_patches) → Vector{Float64}

For a single sample, compute the Jacobian condition number of each
transformer block's attention output w.r.t. its input (via finite differences).
This is the key metric: orthogonal init should yield condition numbers ≈ 1.
"""
function compute_jacobian_condition(model, sample_patches)
    # sample_patches: (PATCH_DIM, NUM_PATCHES, 1) — single sample
    cond_numbers = Float64[]

    # Forward through patch embedding
    x = model.patch_embed(sample_patches)   # (D_MODEL, S, 1)

    for (i, block) in enumerate(model.blocks)
        # Compute Jacobian of the block output (flattened) w.r.t. input (flattened)
        x_vec = vec(x)
        n = length(x_vec)

        # For memory, subsample if the Jacobian is too large
        # Full Jacobian: n × n. With D_MODEL=64, S=17 → n=1088. That's ~1088² ≈ 1.2M entries → fine.
        J = Zygote.jacobian(z -> vec(block(reshape(z, size(x)))), x_vec)[1]

        if J !== nothing && !isempty(J)
            sv = svdvals(J)
            sv_pos = filter(s -> s > 1e-10, sv)
            if length(sv_pos) >= 2
                κ = sv_pos[1] / sv_pos[end]
                push!(cond_numbers, Float64(κ))
            else
                push!(cond_numbers, NaN)
            end
        else
            push!(cond_numbers, NaN)
        end

        # Advance x through this block for the next iteration
        x = block(x)
    end

    return cond_numbers
end

# ═════════════════════════════════════════════════════════════════════════════
# §6  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
function accuracy(model, loader)
    correct = 0
    total   = 0
    for (x, y) in loader
        ŷ = model(x)
        correct += sum(onecold(ŷ) .== onecold(y))
        total   += size(y, 2)
    end
    return correct / total
end

"""
    train_model!(model, train_loader, test_loader, tag::String) → training_log

Trains the given ViT model with AdamW, logging metrics each epoch.
`tag` is "xavier" or "orthogonal" — used for saving weights.
"""
function train_model!(model, train_loader, test_loader, tag::String)
    opt_state = Flux.setup(AdamW(LR, (0.9, 0.999), WEIGHT_DECAY), model)

    log = Dict{String, Vector}(
        "epoch"          => Int[],
        "train_loss"     => Float64[],
        "train_acc"      => Float64[],
        "test_acc"       => Float64[],
        "grad_norms"     => Dict{String, Float64}[],
        "jacobian_conds" => Vector{Float64}[],
    )

    # Grab a fixed sample for Jacobian analysis
    sample_x, _ = first(train_loader)
    jac_sample   = sample_x[:, :, 1:1]   # single sample, keep 3D

    for epoch in 1:EPOCHS
        epoch_loss  = 0.0
        num_batches = 0

        last_grad_norms = Dict{String, Float64}()

        @info "[$tag] Epoch $epoch / $EPOCHS"
        p = Progress(length(train_loader); desc="  Training: ")

        for (x, y) in train_loader
            loss_val, grads = Flux.withgradient(model) do m
                logitcrossentropy(m(x), y)
            end
            Flux.update!(opt_state, model, grads[1])

            epoch_loss  += loss_val
            num_batches += 1

            # Record gradient norms from the last batch of this epoch
            last_grad_norms = collect_gradient_norms(grads[1], model)

            next!(p)
        end

        avg_loss  = epoch_loss / num_batches
        train_acc = accuracy(model, train_loader)
        test_acc  = accuracy(model, test_loader)

        # Jacobian condition numbers (expensive — do periodically)
        jac_conds = Float64[]
        if epoch == 1 || epoch % JACOBIAN_EVERY == 0 || epoch == EPOCHS
            @info "  Computing Jacobian condition numbers …"
            jac_conds = compute_jacobian_condition(model, jac_sample)
            for (i, κ) in enumerate(jac_conds)
                @info "    Block $i condition number: $(round(κ; digits=2))"
            end
        end

        @info "  Loss: $(round(avg_loss; digits=4))  " *
              "Train acc: $(round(100*train_acc; digits=2))%  " *
              "Test acc: $(round(100*test_acc; digits=2))%"

        # Log
        push!(log["epoch"],          epoch)
        push!(log["train_loss"],     avg_loss)
        push!(log["train_acc"],      train_acc)
        push!(log["test_acc"],       test_acc)
        push!(log["grad_norms"],     last_grad_norms)
        push!(log["jacobian_conds"], jac_conds)
    end

    # ── Save weights ─────────────────────────────────────────────────────
    weight_path = joinpath(OUT_DIR, "vit_weights_$(tag).jld2")
    model_state = Flux.state(model)
    @save weight_path model_state
    @info "[$tag] Weights saved to $weight_path"

    # ── Save training log ────────────────────────────────────────────────
    log_path = joinpath(OUT_DIR, "training_log_$(tag).jld2")
    @save log_path log
    @info "[$tag] Training log saved to $log_path"

    return log
end

# ═════════════════════════════════════════════════════════════════════════════
# §7  MAIN — Run both experiments
# ═════════════════════════════════════════════════════════════════════════════
function main()
    train_loader, test_loader = load_mnist()

    # ── Experiment 1: Xavier/Glorot ──────────────────────────────────────
    @info "═══ Building ViT with Xavier/Glorot initialization ═══"
    model_xavier = ViT(xavier_init)
    n_params = sum(length, Flux.params(model_xavier))
    @info "Total parameters: $n_params"

    log_xavier = train_model!(model_xavier, train_loader, test_loader, "xavier")

    # ── Experiment 2: Orthogonal ─────────────────────────────────────────
    @info "═══ Building ViT with Orthogonal initialization ═══"
    Random.seed!(42)   # reset seed for fair comparison
    model_ortho = ViT(orthogonal_init)

    log_ortho = train_model!(model_ortho, train_loader, test_loader, "orthogonal")

    # ── Summary comparison ───────────────────────────────────────────────
    println("\n" * "="^70)
    println("  TRAINING SUMMARY")
    println("="^70)
    println()

    for (tag, log) in [("Xavier", log_xavier), ("Orthogonal", log_ortho)]
        final_train = round(100 * log["train_acc"][end]; digits=2)
        final_test  = round(100 * log["test_acc"][end];  digits=2)
        final_loss  = round(log["train_loss"][end]; digits=4)
        println("  [$tag]  Final loss: $final_loss  |  Train: $final_train%  |  Test: $final_test%")

        # Print last Jacobian conds if available
        jc = log["jacobian_conds"][end]
        if !isempty(jc)
            for (i, κ) in enumerate(jc)
                println("           Block $i Jacobian κ = $(round(κ; digits=2))")
            end
        end
        println()
    end

    println("="^70)
    println("  Weights saved in:  $(OUT_DIR)/")
    println("    • vit_weights_xavier.jld2")
    println("    • vit_weights_orthogonal.jld2")
    println("  Logs saved in:")
    println("    • training_log_xavier.jld2")
    println("    • training_log_orthogonal.jld2")
    println("="^70)
end

# ── Entry point ──────────────────────────────────────────────────────────────
main()
