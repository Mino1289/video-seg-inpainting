# Real-time Video Segmentation and Inpainting

This project implements real-time person removal from video streams using YOLO11 segmentation models and background inpainting techniques. The system can process live webcam feeds or video files, automatically detecting and removing people while reconstructing the background.

Future:

- Update the YOLO version or use another segmentation model.
- Create a model for real-time pixels generation / small part of an image.

## Features

- **Real-time Processing**: Process video streams with minimal latency
- **Person Detection & Removal**: Uses YOLO11 segmentation models to detect and remove people from frames
- **Background Reconstruction**: Dynamically builds and maintains a background model for inpainting
- **Multiple Input Sources**: Supports webcam, video files, and other video sources
- **OpenVINO Optimization**: Includes model quantization for improved performance on Intel hardware
- **Flexible Output**: View results in real-time or save processed videos

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- OpenVINO (for optimized inference)
- NumPy

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd video-seg-inpainting
```

2. Install pytorch for your hardware:
    * [Classic devices](https://pytorch.org/get-started/locally/)
    * [XPU device](https://docs.pytorch.org/docs/2.7/notes/get_start_xpu.html)

3. Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

4. Download or convert YOLO models (see Model Setup section)

## Model Setup

### Option 1: Use Pre-trained Models

The project uses YOLO11 segmentation models. Available models:

- `yolo11n-seg.pt` (nano - fastest)
- `yolo11s-seg.pt` (small)
- `yolo11m-seg.pt` (medium)
- `yolo11l-seg.pt` (large)
- `yolo11x-seg.pt` (extra large - most accurate)

### Option 2: Convert to OpenVINO for Better Performance

For Intel hardware optimization, convert models to OpenVINO format:

```bash
python quant.py --model-name yolo11n-seg --openvino --quantize
```

### Option 3: Quantize for Even Better Performance

For maximum performance on Intel hardware, use INT8 quantization:

```bash
python quant.py --model-name yolo11n-seg --openvino --quantize
```

**Note**: Quantization requires downloading the COCO validation dataset (~1GB) and may take some time.

## Usage

### Basic Usage

Run the demo with default settings (uses webcam):

```bash
python demo.py
```

### Process Video File

Edit the `source` variable in `demo.py` to point to your video file:

```python
source = "./data/your_video.mp4"  # Use video file
# source = 0  # Use webcam
```


### Interactive Controls

While the video is playing:

- **ESC** or **Q**: Quit the application
- **R**: Reset the background model (useful if the scene changes significantly)

## How It Works

1. **Initialization**: Captures an initial frame to establish the background model
2. **Person Detection**: Uses YOLO11 segmentation to identify people in each frame
3. **Mask Processing**: Creates and processes segmentation masks with morphological operations
4. **Background Update**: Continuously updates the background model using areas without people
5. **Inpainting**: Replaces detected people with the reconstructed background
6. **Output**: Displays or/and saves the processed video

## Performance

Using a laptop with latest Intel Core Ultra 9 285H on Ubuntu 24, I got 60+ FPS using GPU, 50+ using NPU and 40+ using CPU.

- Use quantized OpenVINO models for better performance on Intel hardware
- Reduce `video_width` for faster processing
- Use GPU device if available
- For real-time applications, consider using the nano model (`yolo11n-seg`)

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure YOLO models are downloaded to the `models/` directory
2. **OpenVINO errors**: Make sure OpenVINO is properly installed and compatible with your hardware
3. **Performance issues**: Try using a smaller model or reducing video resolution
4. **Memory issues**: Reduce batch size or use model quantization

### Performance Optimization

- For Intel CPUs: Use OpenVINO with CPU device
- For Intel GPUs: Use OpenVINO with GPU device
- For NVIDIA GPUs: Use CUDA-enabled PyTorch installation

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.