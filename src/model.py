import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class ClassStyleConditionedUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_styles: int = 4,
        image_size: int = 96,
        base_channels: int = 64,
        cond_dim: int = 256,  # size of conditioning vector
    ):
        super().__init__()

        # UNet with cross-attention for conditioning
        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(base_channels, base_channels*2, base_channels*4),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=cond_dim,
        )

        self.num_classes = num_classes
        self.num_styles = num_styles
        self.cond_dim = cond_dim

        # embeddings for class + style
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.style_emb = nn.Embedding(num_styles, cond_dim)

        # project combined embedding to cross-attn input
        self.norm = nn.LayerNorm(cond_dim)

    def get_conditioning(self, class_labels: torch.LongTensor, style_labels: torch.LongTensor):
        """
        Returns encoder_hidden_states of shape [B, 1, cond_dim]
        suitable for UNet2DConditionModel.
        """
        c_emb = self.class_emb(class_labels)   # [B, cond_dim]
        s_emb = self.style_emb(style_labels)   # [B, cond_dim]
        cond = c_emb + s_emb                   # simple sum (you can experiment)
        cond = self.norm(cond)
        cond = cond.unsqueeze(1)               # [B, 1, cond_dim]
        return cond

    def forward(self, x, timesteps, class_labels, style_labels):
        encoder_hidden_states = self.get_conditioning(class_labels, style_labels)
        out = self.unet(
            sample=x,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample  # diffusers UNet returns a struct
        return out
