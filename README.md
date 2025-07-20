# Differential Vision Transformer (Diff-ViT)

This repository contains a PyTorch implementation of a Vision Transformer (ViT) that incorporates the **Differential Attention** mechanism, as proposed in the ICLR 2025 paper, "[Differential Transformer](https://arxiv.org/abs/2410.05258)". This implementation adapts the original concept from the language domain to computer vision tasks.

## The Problem with Standard Attention

Standard Vision Transformers have revolutionized computer vision, but they are not without limitations. A key issue, identified in recent research, is that the standard self-attention mechanism tends to **over-allocate attention to irrelevant context**. In the vision domain, this means the model might pay significant attention to background patches or uninformative textures, effectively "drowning out" the signal from the most critical parts of the image.

This phenomenon, which we can call **"attention noise,"** can lead to suboptimal performance, especially in complex scenes where identifying the key subject is crucial. The model wastes capacity on irrelevant features, hindering its ability to learn a truly robust and focused representation of the image content.

## Solution: Differential Attention

To address this, we replace the standard Multi-Head Self-Attention (MHSA) layer in the Vision Transformer with `MultiheadDiffAttn` module introduced by Ye *et al.*. The core idea is inspired by noise-canceling headphones and differential amplifiers in electrical engineering.

Instead of calculating a single attention map, Differential Attention computes **two separate attention maps (A1 and A2)** from two different sets of queries and keys (Q1/K1 and Q2/K2). The final attention distribution is then calculated as the **difference between these two maps**, modulated by a learnable scalar, `lambda`.

**`Final_Attention = softmax(A1) - λ * softmax(A2)`**

This subtraction has a powerful effect:
-   **Noise Cancellation:** Features common to both attention maps (i.e., the "common-mode noise" or attention to irrelevant background) are canceled out.
-   **Signal Amplification:** Features that are unique or stronger in the primary attention map are amplified, forcing the model to focus more sharply on the most salient information.
-   **Sparsity:** This process naturally encourages sparse attention patterns, where only the most critical image patches receive high attention scores.

This leads to a more efficient, focused, and powerful attention mechanism that learns to distinguish between the subject and the background more effectively.

## Key Differences: Vanilla ViT vs. Differential ViT

| Feature                 | Standard Vision Transformer (Vanilla ViT)                               | Differential Vision Transformer (This Repo)                                                              |
| ----------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Attention Mechanism** | Standard Multi-Head Self-Attention (MHSA).                              | **Multi-Head Differential Attention (`MultiheadDiffAttn`)**.                                             |
| **Attention Calculation** | A single attention map is computed using `softmax(Q * K^T)`.            | **Two attention maps** are computed, and their **difference** is used as the final attention score.      |
| **Focus & Sparsity**    | Attention can be dense, often assigning non-trivial scores to irrelevant background patches. | Promotes **sparse attention**, actively canceling out "attention noise" to focus on salient features.    |
| **Parameters**          | Projects to Q, K, V for each head.                                      | Projects to two sets of queries and keys (Q1, K1, Q2, K2) and one set of values (V) for each head.         |
| **Learnable Components**| The projection matrices (Wq, Wk, Wv, Wo) are the main learnable parts.    | Includes the projection matrices plus a **learnable scalar `lambda`** for each layer to balance the subtraction. |
| **Normalization**       | Typically uses LayerNorm after the attention block.                     | Employs **Headwise Normalization** (`RMSNorm` in our implementation) before the final projection to stabilize training, as the differential mechanism leads to more diverse statistics between heads. |

## Repository Structure

-   `DiffAttention.py`: Contains the core implementation of the `MultiheadDiffAttn` module. This is the heart of the novel mechanism.
-   `ViT.py`: Defines the `VisionTransformer` architecture, integrating `MultiheadDiffAttn` into the encoder blocks.
-   `DataLoader.py`: Utility for loading image datasets.
-   `ViT_train.py`: Script for training the model on a single GPU.
-   `ViT_Train_Dist.py`: Script for distributed training across multiple GPUs.

## How to Use

To train the Differential Vision Transformer, you can use the `ViT_train.py` script. Make sure to adjust the parameters according to your dataset and environment.

### Training Command Example:

```bash
python ViT_train.py \
    --data-dir /path/to/your/dataset \
    --num-classes 10 \
    --image-size 224 \
    --patch-size 16 \
    --embed-dim 768 \
    --depth 12 \
    --num-heads 12 \
    --mlp-dim 3072 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --epochs 100 \
    --output-dir ./checkpoints
```
## To DO:
Due to lack of computational resources we are not able to train the model. Pretraining on ImageNet is needed before fine tuning it. 
Use CKA to analyse and compare the internal representations of Diff_ViT and Vanilla ViT.
## Citation

This work is an implementation based on the concepts from the following paper. Please consider citing it if you find this repository useful.

```
@article{ye2024differential,
  title={Differential Transformer},
  author={Ye, Tianzhu and Dong, Li and others},
  journal={arXiv preprint arXiv:2410.05258},
  year={2025}
}
```
