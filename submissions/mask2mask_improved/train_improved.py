import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import av
from pathlib import Path
import sys
from tqdm import tqdm
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frame_utils import yuv420_to_rgb
from modules import SegNet, PoseNet

# --- Architecture ---
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

# --- Dataset ---
class QuickDataset(torch.utils.data.Dataset):
    def __init__(self, mask_path, video_path, seq_len=8):
        self.masks = self.load_masks(mask_path)
        self.frames = self.load_frames(video_path)
        self.seq_len = seq_len

    def load_masks(self, path):
        container = av.open(str(path))
        masks = []
        for frame in container.decode(video=0):
            img = np.frombuffer(frame.planes[0], np.uint8).reshape(192, 256)
            masks.append(torch.from_numpy(img.copy() // 63).to(torch.uint8))
        container.close()
        return torch.stack(masks)

    def load_frames(self, path):
        container = av.open(str(path))
        frames = []
        for frame in container.decode(video=0):
            rgb = yuv420_to_rgb(frame)
            frames.append(rgb.permute(2, 0, 1)) # uint8
        container.close()
        return torch.stack(frames)

    def __len__(self):
        return len(self.masks) - self.seq_len

    def __getitem__(self, idx):
        return self.masks[idx:idx+self.seq_len], self.frames[idx:idx+self.seq_len]

# --- Training ---
def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    print(f"=== Full Training Initialization ===")
    print(f"Device: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    
    model = TemporalGenerator().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TemporalGenerator with {n_params:,} parameters")

    print(f"Loading Models (SegNet, PoseNet)...")
    segnet = SegNet().to(device).eval()
    segnet.load_state_dict(load_file(ROOT / 'models/segnet.safetensors', device=str(device)))
    posenet = PoseNet().to(device).eval()
    posenet.load_state_dict(load_file(ROOT / 'models/posenet.safetensors', device=str(device)))
    

    
    print(f"Loading Full Dataset (1200 frames)...")
    ds = QuickDataset(ROOT / 'submissions/mask2mask_improved/mask.mp4', ROOT / 'videos/0.mkv', seq_len=8)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    epochs = 50
    best_loss = float('inf')
    ckpt_path = 'submissions/mask2mask_improved/model.pt'
    if Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, map_location=device)
        sd = {k: v.float() for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Starting Full Training ({epochs} epochs)...")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Rec':>8} | {'Seg':>8} | {'Pose':>8}")
    print("-" * 40)

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch}", leave=False)
        epoch_loss, epoch_rec, epoch_seg, epoch_pose = 0, 0, 0, 0
        
        for m, f in pbar:
            m = m.to(device, non_blocking=True).long()
            f = f.to(device, non_blocking=True).float() / 255.0
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                pred, _ = model(m)
                
                f_down = F.interpolate(
                    einops.rearrange(f, 'b t c h w -> (b t) c h w'),
                    size=(384, 512),
                    mode='bilinear',
                    align_corners=False
                )
                f_down = einops.rearrange(f_down, '(b t) c h w -> b t c h w', b=f.shape[0])
                
                loss_rec = F.l1_loss(pred, f_down)
                
                loss_diff = F.l1_loss(pred[:, 1:] - pred[:, :-1], f_down[:, 1:] - f_down[:, :-1])
                
                seg_in_pred = segnet.preprocess_input(pred * 255.0)
                with torch.no_grad():
                    seg_in_gt = segnet.preprocess_input(f_down * 255.0)
                    seg_out_gt = segnet(seg_in_gt)
                seg_out_pred = segnet(seg_in_pred)
                
                loss_seg = F.kl_div(
                    F.log_softmax(seg_out_pred, dim=1), 
                    F.softmax(seg_out_gt, dim=1), 
                    reduction='batchmean'
                )
                
                B, T = pred.shape[:2]
                num_pairs = T - 1
                
                all_pred_pairs = einops.rearrange(torch.stack([pred[:, t:t+2] for t in range(num_pairs)], dim=0), 'n b t c h w -> (n b) t c h w')
                all_gt_pairs = einops.rearrange(torch.stack([f_down[:, t:t+2] for t in range(num_pairs)], dim=0), 'n b t c h w -> (n b) t c h w')
                
                posenet_in_pred = posenet.preprocess_input(all_pred_pairs * 255.0)
                with torch.no_grad():
                    posenet_in_gt = posenet.preprocess_input(all_gt_pairs * 255.0)
                    posenet_out_gt = posenet(posenet_in_gt)
                posenet_out_pred = posenet(posenet_in_pred)
                
                loss_pose = sum(
                    F.mse_loss(posenet_out_pred[h.name], posenet_out_gt[h.name])
                    for h in posenet.hydra.heads
                )
                
                loss = loss_rec + 0.1 * loss_seg + 1.0 * loss_pose + 0.5 * loss_diff
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                epoch_rec += loss_rec.item()
                epoch_seg += loss_seg.item()
                epoch_pose += loss_pose.item()
                pbar.set_postfix(L=f"{loss.item():.3f}", R=f"{loss_rec.item():.3f}", P=f"{loss_pose.item():.3f}")
        
        avg_loss = epoch_loss / len(dl)
        avg_rec = epoch_rec / len(dl)
        avg_seg = epoch_seg / len(dl)
        avg_pose = epoch_pose / len(dl)
        
        status = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            sd = {k: v.half() for k, v in model.state_dict().items()}
            torch.save(sd, 'submissions/mask2mask_improved/model.pt')
            status = " (Best)"
            
        print(f"{epoch:5d} | {avg_loss:8.4f} | {avg_rec:8.4f} | {avg_seg:8.4f} | {avg_pose:8.4f}{status}")
        
        import gc
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("-" * 40)
    print(f"Training complete. Best model saved to: submissions/mask2mask_improved/model.pt")

if __name__ == "__main__":
    train()
