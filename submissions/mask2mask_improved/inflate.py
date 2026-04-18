import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import av
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frame_utils import camera_size

# --- Architecture (must match train_improved.py) ---
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, input_tensor, cur_state):
        if cur_state is None:
            cur_state = torch.zeros(input_tensor.size(0), self.hidden_dim, input_tensor.size(2), input_tensor.size(3), device=input_tensor.device)
        combined = torch.cat([input_tensor, cur_state], dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        combined_reset = torch.cat([input_tensor, reset * cur_state], dim=1)
        candidate = torch.tanh(self.out_gate(combined_reset))
        next_h = (1 - update) * cur_state + update * candidate
        return next_h

class TemporalGenerator(nn.Module):
    def __init__(self, num_classes=5, features=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, features)
        self.enc = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.GroupNorm(8, features), nn.ReLU(inplace=True),
            nn.Conv2d(features, features * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, features * 2), nn.ReLU(inplace=True)
        )
        self.gru = ConvGRUCell(features * 2, features * 2, 3, True)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, 4, stride=2, padding=1),
            nn.GroupNorm(8, features), nn.ReLU(inplace=True),
            nn.Conv2d(features, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, masks, h=None):
        batch, seq, H, W = masks.shape
        outputs = []
        for t in range(seq):
            x = self.embedding(masks[:, t])
            x = einops.rearrange(x, 'b h w c -> b c h w')
            x = self.enc(x)
            h = self.gru(x, h)
            outputs.append(self.dec(h))
        return torch.stack(outputs, dim=1), h

def inflate(data_dir, output_dir, file_list_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGenerator().to(device)
    sd = torch.load(data_dir / 'model.pt', map_location=device)
    if any(v.dtype == torch.float16 for v in sd.values()):
        sd = {k: v.float() for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    mask_path = data_dir / 'mask.mp4'
    container = av.open(str(mask_path))
    
    all_masks = []
    for frame in container.decode(video=0):
        img = np.frombuffer(frame.planes[0], np.uint8).reshape(192, 256)
        all_masks.append(torch.from_numpy(img.copy() // 63).to(torch.uint8))
    container.close()
    all_masks = torch.stack(all_masks).to(device)

    target_w, target_h = camera_size
    files = [line.strip() for line in open(file_list_path) if line.strip()]
    
    cursor = 0
    with torch.no_grad():
        for file_name in tqdm(files, desc="Inflating videos"):
            base_name = os.path.splitext(file_name)[0]
            dst_path = Path(output_dir) / f"{base_name}.raw"
            
            video_masks = all_masks[cursor : cursor + 1200]
            cursor += 1200
            
            with open(dst_path, 'wb') as f_out:
                h = None
                seq_len = 10
                for i in range(0, len(video_masks), seq_len):
                    batch_masks = video_masks[i : i + seq_len].unsqueeze(0).long()
                    pred, h = model(batch_masks, h)
                    
                    pred_up = F.interpolate(
                        einops.rearrange(pred, 'b t c h w -> (b t) c h w'),
                        size=(target_h, target_w),
                        mode='bicubic',
                        align_corners=False
                    )
                    
                    output_bytes = (pred_up.clamp(0, 1) * 255.0).round().to(torch.uint8)
                    output_bytes = einops.rearrange(output_bytes, 'n c h w -> n h w c')
                    f_out.write(output_bytes.cpu().numpy().tobytes())

if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list = sys.argv[3]
    inflate(data_dir, output_dir, file_list)
