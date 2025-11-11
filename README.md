# comma2k19 compression challenge 🤏

<p align="center">
<img height="300" alt="image" src="https://github.com/user-attachments/assets/f72fae51-96bd-47c6-b4ff-6e17e79220cb" />
</p>

[test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip) is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset. Make it as small as possible while preserving semantic content (evaluated by a segmentation model) and temporal dynamics (evaluated by an egomotion relative pose model).

- distortion:
  - SegNet distortion: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
  - PoseNet distortion: MSE of the outputs of a PoseNet (x,y,z velocities and roll,pitch,yaw rates) evaluated on original vs. reconstructed seq_len frames
- rate
  - the size of the compressed archive devided by the size of the original archive
- score: a weighted average of the different components of the distortion and the rate

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

- scoring window: The score is computed using 64 videos from `Chunk_1` of `comma2k19` listed in `test_video_names.txt`. We use the largest prefix that fits an integer number of batches of seq_len sized sequences. The score is computed using a batch size of 32, meaning that for a video of 1200 frames, the first 1152 frames are used for scoring (18 batches of 2 frames)

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4dc4d758-230f-4af7-9d5a-06230c48c274" />
</p>

## submission format
```
comma2k19_submission/
  comma2k19_submission.zip      # all data needed to reconstruct (this is the archive)
  your_dataset.py               # defines YourSubmissionDataset
  README.md                     # optional notes

```

`YourSubmissionDataset` should be a subclass of `torch.utils.data.IterableDataset` with the following signature and methods

```python

class YourSubmissionDataset(torch.utils.data.IterableDataset):
  def __init__(
      self,
      file_names: list[str],
      archive_path: Path,
      batch_size: int,
      device_id: int,
      **kwargs,
  ):
      self.file_names = file_names
      self.archive_path = archive_path

  def prepare_data(self):
    # called by all ranks
    # e.g. do something with comma2k19_submission.zip
    ...

  def __iter__(self):
    for file_name in self.file_names:
      # yield torch.Tensor with shape (batch_size, seq_len, camera_size[1], camera_size[0], 3), dtype=uint8
      ...

```


