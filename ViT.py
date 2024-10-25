import torch
import torch.nn.functional as F
from torch import nn
from DiffAttention import MultiheadDiffAttn
try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from torch.nn import LayerNorm as RMSNorm
from torch.autograd import gradcheck
from torchinfo import summary

#Embedding creation
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, channels, embed_dim):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.proj(x)  # [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x

#Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadDiffAttn(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_residual = x
        x = self.norm1(x)
        attn_output = self.attn(x)
        x = x_residual + self.dropout1(attn_output)

        x_residual = x
        x = self.norm2(x)
        x = x_residual + self.mlp(x)
        return x


#Complete Impementation
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        channels=3,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                depth=layer_idx,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for layer_idx in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.zeros_(self.cls_token)

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.trunc_normal_(module.weight, std=0.02)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.LayerNorm) or isinstance(module, RMSNorm):
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #         if module.weight is not None:
    #             nn.init.ones_(module.weight)

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        x = self.patch_embedding(x)  # [batch_size, num_patches, embed_dim]

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]
        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits



# Unit Test:
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        channels=3,
        dropout=0.1,
    )
    
    model.to(device)
    # Create a random input tensor (batch_size, channels, height, width)
    
    x = torch.randn(1, 3, 224, 224).to(device)
    logits = model(x)
    target = torch.randn(1, 1000).to(device)
    
    loss_function = nn.MSELoss()
    loss = loss_function(logits, target)


    loss.backward()
    
    # Forward pass
    
    print(logits.shape)  # Output: torch.Size([1, 1000])

    # Gradients with respect to the input tensor
    print("Gradients of the input tensor:")
    print(x.grad)
    summary(model, input_size=(1, 3, 224, 224))

    # # Gradients with respect to the model's parameters
    # print("\nGradients of the model's parameters:")
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Parameter: {name}, Gradient:\n{param.grad}")
    #     else:
    #         print(f"Parameter: {name} has no gradient.")

    # # Optionally, you can check if gradients contain NaNs or Infs
    # print("\nChecking for NaNs or Infs in gradients:")
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #             print(f"Parameter: {name} has NaN or Inf in gradients.")
    #         else:
    #             print(f"Parameter: {name} gradients are OK.")
    