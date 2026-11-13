import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ViTBackbone


def pad_images_to_batch(
    images,
    pad_size_multiple=1,
):
    if torch.is_tensor(images):
        return images, None
    if not isinstance(images, (list, tuple)) or len(images) == 0:
        raise TypeError("images must be a Tensor or a non-empty list/tuple of Tensors")
    if any((not torch.is_tensor(im) or im.ndim != 3) for im in images):
        raise ValueError("each image must be a (C,H,W) Tensor")

    max_h = max(int(im.shape[-2]) for im in images)
    max_w = max(int(im.shape[-1]) for im in images)
    pad_size_multiple = int(pad_size_multiple)
    if pad_size_multiple < 1:
        raise ValueError("pad_size_multiple must be >= 1")
    if pad_size_multiple > 1:
        max_h = int(math.ceil(max_h / pad_size_multiple) * pad_size_multiple)
        max_w = int(math.ceil(max_w / pad_size_multiple) * pad_size_multiple)
    batch_size = len(images)

    c = int(images[0].shape[0])
    dtype = images[0].dtype
    device = images[0].device

    batch = torch.zeros((batch_size, c, max_h, max_w), dtype=dtype, device=device)
    mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool, device=device)
    for i, im in enumerate(images):
        _, h, w = im.shape
        batch[i, :, :h, :w] = im
        mask[i, :h, :w] = False

    return batch, mask


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x, mask):
        b, _, h, w = x.shape
        if mask is None:
            not_mask = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
        else:
            not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos


class SimpleDETR(nn.Module):
    def __init__(
        self,
        num_classes=20,
        backbone="facebook/dinov2-with-registers-small",
        num_queries=25,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = ViTBackbone(name=backbone)
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.num_queries = num_queries
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d_model, kernel_size=1)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, images):
        input_size_multiple = getattr(self.backbone, "input_size_multiple", 1)
        x, mask = pad_images_to_batch(images, pad_size_multiple=input_size_multiple)
        features = self.backbone(x)
        src = self.input_proj(features)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=src.shape[-2:]).to(dtype=torch.bool).squeeze(1)
        pos = self.position_embedding(src, mask)
        B, C, H, W = src.shape
        src = src.flatten(2).permute(0, 2, 1)
        pos = pos.flatten(2).permute(0, 2, 1)
        mask = mask.flatten(1)
        src_with_pos = src + pos
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(src=src_with_pos, tgt=tgt + query_embed, src_key_padding_mask=mask)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
