import torch
import av
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from frame_utils import yuv420_to_rgb, camera_size
from modules import SegNet
from safetensors.torch import load_file

def generate_masks(video_path, output_path, device):
    segnet = SegNet().to(device).eval()
    segnet.load_state_dict(load_file(ROOT / 'models/segnet.safetensors', device=str(device)))
    
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    
    # We will save masks as a grayscale mp4 for compression
    output_container = av.open(str(output_path), mode='w')
    # Use a small resolution for the masks to save space
    out_stream = output_container.add_stream('libx264', rate=20)
    out_stream.width = 512
    out_stream.height = 384
    out_stream.pix_fmt = 'yuv420p'
    
    with torch.no_grad():
        for frame in tqdm(container.decode(stream), total=1200):
            rgb = yuv420_to_rgb(frame)
            # Preprocess for SegNet (expects (B, T, C, H, W))
            inp = rgb.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float().to(device)
            seg_in = segnet.preprocess_input(inp)
            out = segnet(seg_in)
            mask = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8) # (H, W) classes 0-4
            
            # Map classes 0-4 to 0-255 for visibility/compression
            # 0->0, 1->63, 2->127, 3->191, 4->255
            mask_vis = (mask * 63).astype(np.uint8)
            
            # Create a YUV frame for encoding
            # We put the mask in the Y channel, and 128 in U,V (grayscale)
            frame_out = av.VideoFrame(512, 384, 'gray8')
            frame_out.planes[0].update(mask_vis.tobytes())
            
            for packet in out_stream.encode(frame_out):
                output_container.mux(packet)
                
    for packet in out_stream.encode():
        output_container.mux(packet)
    output_container.close()
    container.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_masks(ROOT / 'videos/0.mkv', 'submissions/mask2mask_improved/mask.mp4', device)
