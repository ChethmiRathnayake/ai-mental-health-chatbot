import torch
import torch.nn as nn


class CognitiveLoadTransformer(nn.Module):
    def __init__(self, input_dim=11, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.classifier = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.classifier(x)

def load_artifact(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    feature_order = ckpt["feature_order"]
    label_map = ckpt["label_map"]
    scaler = ckpt["scaler"]
    inv_label_map = {v: k for k, v in label_map.items()}

    model = CognitiveLoadTransformer(input_dim=len(feature_order))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, scaler, feature_order, label_map, inv_label_map
