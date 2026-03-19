# comma2k25 compression challenge 🤏

<p align="center">
<img height="300" alt="image" src="https://github.com/user-attachments/assets/f72fae51-96bd-47c6-b4ff-6e17e79220cb" />
</p>

 `./test_videos/b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc` is a 1 minute long driving video of size 37.5 MB. Make it as small as possible while preserving semantic content (evaluated by a segmentation model) and temporal dynamics (evaluated by an egomotion relative pose model).

- distortion:
  - SegNet distortion: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
  - PoseNet distortion: MSE of the outputs of a PoseNet (x,y,z velocities and roll,pitch,yaw rates) evaluated on original vs. reconstructed seq_len frames
- rate
  - the size of the compressed archive devided by the size of the original archive
- score: a weighted average of the different components of the distortion and the rate

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/e4421c23-8fbe-4293-b8de-9a77e6d568ab"/>
</p>

## Quickstart
```
# clone the repo
git clone https://github.com/commaai/comma2k25_compression_challenge.git
cd comma2k25_compression_challenge

# git lfs
git lfs install
git lfs pull

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick one: cu126/ cu128 / cu130 / cpu
uv sync --group cu128

# activate
source .venv/bin/activate

# test dataloaders
python frame_utils.py

# test models
python modules.py

# naively recompress
bash examples/baseline_fast.sh --in-dir test_videos/ --jobs 1 --video-names-file public_test_video_names.txt --out-dir ./submission/

# evaluate the naive recompression strategy
torchrun --nproc-per-node 1 evaluate.py  --dataloader examples/baseline_dataloader_fast.py --compressed-dir ./submission
```

If everything worked as expected, this should producce a `report.txt` file with this content:

```
=== Evaluation config ===
  batch_size: 16
  compressed_dir: submission
  dataloader: examples/baseline_dataloader_fast.py
  device: None
  num_threads: 2
  prefetch_queue_depth: 4
  report: report.txt
  seed: 1234
  uncompressed_dir: test_videos
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.06430182
  Average SegNet Distortion: 0.00381109
  Submission file size (deflated): 14428760.00000000 bytes
  Original uncompressed size (deflated): 37533786.00000000 bytes
  Compression Rate (deflated): 0.38442059
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 10.79350826
```

## submission format and rules

`DatasetClass` should be a subclass of `torch.utils.data.IterableDataset` defined in a python file, with the following signature and methods.

The dataset should yield batches of decompressed video frames of `file_names` as torch tensors, which will be fed to the distortion evaluation models.

The dataset can consume anything in `args.compressed_dir` to produce the batches, all files in `args.compressed_dir` will be used to compute the compressed size for the rate term of the score.

External libraries and tools can be used and won't be counted towards the compressed size, unless they use large artifacts (such as neural networks, meshes, point clouds, etc.).

The dataset should not consume anything outside of `args.compressed_dir`.

You can use anything for compression, including the models and the original uncompressed videos. Include your compression code if you wish but it won't be used for evaluation.

```python

class DatasetClass(torch.utils.data.IterableDataset):
  def __init__(
      self,
      file_names: List[str],
      batch_size: int,
      device: torch.device,
      ...
      ):

  def prepare_data(self):
    # called by all ranks
    # e.g. do something with if needed submission.zip
    ...

  def __iter__(self):
    for file_name in self.file_names:
      # yield name, idx, batch

      # name: segment_id
      # idx: batch index within the segment
      # batch: torch.Tensor with shape (batch_size, seq_len, camera_size[1], camera_size[0], 3), dtype=uint8
      ...

```

## going further

You can use [test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip), which is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset, to test your compression strategy on more samples.

The evaluation script and the dataloader are designed to be scalable and can handle different batch sizes, sequence lengths, and video resolutions. You can modify them to fit your needs.
