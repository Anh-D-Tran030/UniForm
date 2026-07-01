from pathlib import Path
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Model


CHECKPOINT_DIR = Path(r"A:\RealForm\outputs\layoutlmv3_projection_commonforms_masked_split_10ep_rerun_285trainretrieval")
OUTPUT_PATH = Path(r"A:\RealForm\outputs\odc_projection_scripted.pt")
EXPORT_DEVICE = "cpu"


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size, projection_dim, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, pooled):
        return F.normalize(self.network(pooled), p=2, dim=-1)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin, scale):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)


class ProjectionModel(nn.Module):
    def __init__(self, model_name, projection_dim, pooling, num_train_classes, arcface_margin, arcface_scale):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(model_name)
        self.pooling = pooling
        hidden_size = self.backbone.config.hidden_size
        self.projection_head = ProjectionHead(hidden_size, projection_dim)
        self.arcface_head = ArcFaceHead(projection_dim, num_train_classes, arcface_margin, arcface_scale)

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )

        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.projection_head(pooled)


def main():
    config = json.loads((CHECKPOINT_DIR / "training_config.json").read_text())
    checkpoint = torch.load(
        CHECKPOINT_DIR / "best_projection_model.pt",
        map_location=EXPORT_DEVICE,
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]
    num_train_classes = state_dict["arcface_head.weight"].shape[0]

    model = ProjectionModel(
        model_name=config["model_name"],
        projection_dim=int(config["projection_dim"]),
        pooling=config["pooling"],
        num_train_classes=int(num_train_classes),
        arcface_margin=float(config["arcface_margin"]),
        arcface_scale=float(config["arcface_scale"]),
    )

    model.load_state_dict(state_dict)
    model.to(EXPORT_DEVICE)
    model.eval()

    example_input_ids = torch.ones((1, 512), dtype=torch.long, device=EXPORT_DEVICE)
    example_attention_mask = torch.ones((1, 512), dtype=torch.long, device=EXPORT_DEVICE)
    example_bbox = torch.zeros((1, 512, 4), dtype=torch.long, device=EXPORT_DEVICE)
    example_pixel_values = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=EXPORT_DEVICE)

    traced = torch.jit.trace(
        model,
        (
            example_input_ids,
            example_attention_mask,
            example_bbox,
            example_pixel_values,
        ),
        strict=False,
    )

    traced.save(str(OUTPUT_PATH))
    print(f"Saved TorchScript model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
