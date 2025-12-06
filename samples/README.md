# Sample Test Data

This directory contains minimal synthetic test data for the 4DGS MV pipeline.

## Files

- `tiny_sample.mp4` - A minimal test video (placeholder - generate with ffmpeg from frames)
- `synthetic_test_frames/` - 5 synthetic test frames (640x480 PNG)

## Generating the test video

If `tiny_sample.mp4` is not present, generate it from frames:

```bash
ffmpeg -framerate 5 -i synthetic_test_frames/frame_%06d.png \
  -vframes 5 -c:v libx264 -pix_fmt yuv420p -crf 23 \
  samples/tiny_sample.mp4
```

## Usage in tests

These samples are used by the unit tests to verify pipeline functionality without requiring large video files.

The synthetic frames show a simple moving red rectangle on a gray background, simulating basic motion for testing frame extraction, segmentation, and COLMAP fallback.
