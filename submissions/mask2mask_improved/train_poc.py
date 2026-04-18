import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules import SegNet, PoseNet

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=self.bias)
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=self.bias)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=self.bias)

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
    def __init__(self, num_classes=5, features=64):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, features)
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.down = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True)
        )
        
        # Temporal Recurrent Bottleneck
        self.gru = ConvGRUCell(features * 2, features * 2, 3, True)
        
        # Decoder
        self.up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, features, 4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(features, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, masks, h=None):
        # masks shape: (batch, seq, H, W)
        batch, seq, H, W = masks.shape
        outputs = []
        
        for t in range(seq):
            m_t = masks[:, t]
            x = self.embedding(m_t) # (B, H, W, C)
            x = einops.rearrange(x, 'b h w c -> b c h w')
            
            e1 = self.enc1(x)
            d1 = self.down(e1)
            
            h = self.gru(d1, h)
            
            u1 = self.up(h)
            out = self.final(u1)
            outputs.append(out)
            
        return torch.stack(outputs, dim=1), h

def train_step(model, masks, gt_frames, optimizer, segnet, posenet):
    # masks: (B, T, H, W) long
    # gt_frames: (B, T, 3, H, W) float [0, 1]
    
    model.train()
    optimizer.zero_grad()
    
    pred_frames, _ = model(masks) # (B, T, 3, H, W)
    
    # 1. Reconstruction Loss
    loss_rec = F.l1_loss(pred_frames, gt_frames)
    
    # 2. SegNet Loss (Semantic)
    # SegNet expects (B, 1, 3, H, W) - using last frame of sequence
    # Note: modules.py SegNet.preprocess_input takes (B, T, C, H, W)
    seg_in_pred = segnet.preprocess_input(pred_frames)
    seg_in_gt = segnet.preprocess_input(gt_frames)
    
    with torch.no_grad():
        seg_out_gt = segnet(seg_in_gt)
    seg_out_pred = segnet(seg_in_pred)
    
    loss_seg = F.kl_div(
        F.log_softmax(seg_out_pred, dim=1),
        F.softmax(seg_out_gt, dim=1),
        reduction='batchmean'
    )
    
    # 3. PoseNet Loss (Temporal)
    # PoseNet expects (B, 2, 3, H, W) 
    # We take consecutive pairs from the sequence
    loss_pose = 0
    for t in range(pred_frames.shape[1] - 1):
        pair_pred = pred_frames[:, t:t+2]
        pair_gt = gt_frames[:, t:t+2]
        
        posenet_in_pred = posenet.preprocess_input(pair_pred)
        with torch.no_grad():
            posenet_in_gt = posenet.preprocess_input(pair_gt)
            posenet_out_gt = posenet(posenet_in_gt)
        posenet_out_pred = posenet(posenet_in_pred)
        
        loss_pose += sum(
            F.mse_loss(posenet_out_pred[h.name], posenet_out_gt[h.name])
            for h in posenet.hydra.heads
        )
    loss_pose /= (pred_frames.shape[1] - 1)

    # Combined Loss (Weights tuned for challenge formula)
    loss = loss_rec + 0.1 * loss_seg + 0.01 * loss_pose
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), loss_rec.item(), loss_seg.item(), loss_pose.item()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalGenerator().to(device)
    segnet = SegNet().to(device).eval()
    posenet = PoseNet().to(device).eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Proof of Concept Initialized.")
    print(f"Generator Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy data for demonstration
    B, T, H, W = 2, 4, 128, 128
    dummy_masks = torch.randint(0, 5, (B, T, H, W)).to(device)
    dummy_frames = torch.rand(B, T, 3, H, W).to(device)
    
    l, rec, seg, pose = train_step(model, dummy_masks, dummy_frames, optimizer, segnet, posenet)
    print(f"Step Loss: {l:.4f} [Rec: {rec:.4f}, Seg: {seg:.4f}, Pose: {pose:.4f}]")
