**Default Device**
This repo is **not Jetson-specific by default**. It is a Python/OpenCV/ONNX Runtime project, so the easiest default target is:

- Linux desktop/laptop
- x86_64 CPU
- NVIDIA GPU optional through `onnxruntime-gpu`
- Webcam or video file through OpenCV

The README says it supports webcam/video inference and uses SCRFD + ArcFace. It also lists ONNX weights for SCRFD detection and ArcFace recognition. Source: [repo README](https://github.com/Kumar2421/scrfd_arcface_facerecognition).

Important: this is **not ready-made Jetson Nano software**. Jetson Nano is ARM/aarch64 with JetPack 4.6.x, CUDA 10.2, TensorRT 8.2.1, and older Python. NVIDIA confirms JetPack 4.6.4 is the current production release for Nano-class JetPack 4.x, with CUDA 10.2 and TensorRT 8.2.1. Source: [NVIDIA JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464).

---

## 1. RTX 3070 Setup

This is the easier setup. I would use Ubuntu Linux, not Windows, for the first proof of concept.

### Step 1: Check GPU driver

```bash
nvidia-smi
```

You should see the RTX 3070. If this command fails, install/update the NVIDIA driver first.

### Step 2: Clone repo

```bash
git clone https://github.com/Kumar2421/scrfd_arcface_facerecognition.git
cd scrfd_arcface_facerecognition
```

The README says `cd face-reidentification`, but the actual repo folder name will be `scrfd_arcface_facerecognition`.

### Step 3: Create Python environment

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev ffmpeg libgl1 libglib2.0-0 v4l-utils

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install minimal dependencies

Do **not** blindly install the full `requirements.txt` first. It includes both `opencv-python` and `opencv-python-headless`, which can break `cv2.imshow()` GUI display.

Install this cleaner set:

```bash
pip install "numpy<2" opencv-python scikit-image onnx onnxruntime-gpu
```

ONNX Runtime’s docs say the default `onnxruntime-gpu` PyPI package targets CUDA 12.x since version 1.19. Source: [ONNX Runtime install docs](https://onnxruntime.ai/docs/install/).

Check providers:

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

You want to see:

```text
CUDAExecutionProvider
CPUExecutionProvider
```

If you only see `CPUExecutionProvider`, the app will run but not use the RTX 3070.

### Step 5: Download weights

```bash
mkdir -p weights
sh download.sh
```

The README lists these weights:

- `det_500m.onnx` - lightweight SCRFD detector
- `det_2.5g.onnx`
- `det_10g.onnx` - heavier detector
- `w600k_mbf.onnx` - lightweight ArcFace/MobileFace recognition
- `w600k_r50.onnx` - heavier ArcFace ResNet-50 recognition

For RTX 3070 testing, start with:

```text
det_500m.onnx
w600k_mbf.onnx
```

Then test heavier models later.

### Step 6: Add known faces

Create a `faces` folder:

```bash
mkdir -p faces
```

Put one clear face image per person:

```text
faces/sameed.jpg
faces/ali.jpg
faces/person_name.jpg
```

The filename becomes the displayed identity.

### Step 7: Fix source handling bug

The README says you can run with `--source`, but current `main.py` hardcodes webcam:

```python
# cap = cv2.VideoCapture(params.source)
cap = cv2.VideoCapture(0)
```

Change it to:

```python
cap = cv2.VideoCapture(params.source)
```

Otherwise `--source assets/in_video.mp4` will be ignored.

### Step 8: Force GPU provider

In `models/scrfd.py` and `models/arcface.py`, find the `onnxruntime.InferenceSession(...)` calls.

Use this pattern:

```python
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    else ["CPUExecutionProvider"]
)

self.session = onnxruntime.InferenceSession(model_file, providers=providers)
```

This makes the RTX 3070 path explicit.

### Step 9: Run webcam

```bash
python main.py \
  --source 0 \
  --det-weight weights/det_500m.onnx \
  --rec-weight weights/w600k_mbf.onnx \
  --confidence-thresh 0.5 \
  --similarity-thresh 0.4
```

### Step 10: Run video file

```bash
python main.py \
  --source assets/in_video.mp4 \
  --det-weight weights/det_500m.onnx \
  --rec-weight weights/w600k_mbf.onnx
```

### RTX Difficulty

RTX 3070 setup difficulty: **3/10 to 4/10**.

Main changes:

- clean dependency install,
- fix `--source`,
- explicitly enable CUDA provider,
- avoid headless OpenCV.

---

## 2. Jetson Nano Setup

This is harder. Jetson Nano is not just a smaller RTX GPU. It is ARM/aarch64, usually Ubuntu 18.04, Python 3.6, CUDA 10.2, and much less memory.

### Recommended Jetson strategy

Use the lightest models:

```text
det_500m.onnx
w600k_mbf.onnx
```

Avoid these at first:

```text
det_10g.onnx
w600k_r50.onnx
```

Those are too heavy for Nano real-time work.

### Step 1: Flash JetPack

Use JetPack 4.6.4 / L4T 32.7.4 for Jetson Nano.

NVIDIA JetPack page: [JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464)

After boot:

```bash
cat /etc/nv_tegra_release
python3 --version
```

You will likely see Python 3.6.x.

### Step 2: Set power mode

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Step 3: Add swap

Nano can run out of RAM during installs.

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Step 4: Install system dependencies

```bash
sudo apt update
sudo apt install -y \
  git \
  python3-pip \
  python3-dev \
  python3-opencv \
  python3-numpy \
  python3-scipy \
  python3-skimage \
  libopenblas-base \
  libomp-dev \
  ffmpeg
```

Do **not** install `opencv-python` or `opencv-python-headless` from pip on Nano first. Use JetPack’s OpenCV package.

### Step 5: Prepare pip carefully

Python 3.6 cannot use many modern packages.

```bash
python3 -m pip install --upgrade "pip<22" "setuptools<60" wheel
```

### Step 6: Install ONNX Runtime for Jetson

This is the hardest part.

Do **not** use normal:

```bash
pip3 install onnxruntime-gpu
```

That is mainly for desktop CUDA packages. ONNX Runtime’s official install matrix points Jetson TensorRT usage toward Jetson Zoo/NVIDIA-managed packages. Source: [ONNX Runtime install docs](https://onnxruntime.ai/docs/install/).

NVIDIA has a Jetson ONNX Runtime path through Jetson Zoo. Source: [NVIDIA ONNX Runtime for Jetson blog](https://developer.nvidia.com/blog/announcing-onnx-runtime-for-jetson/).

You need a wheel matching:

- Jetson Nano
- aarch64
- Python 3.6
- JetPack 4.x

Example pattern:

```bash
wget <jetson-zoo-onnxruntime-wheel-url> -O onnxruntime_gpu-*-cp36-cp36m-linux_aarch64.whl
python3 -m pip install ./onnxruntime_gpu-*-cp36-cp36m-linux_aarch64.whl
```

Then check:

```bash
python3 - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
PY
```

Best case:

```text
CUDAExecutionProvider
CPUExecutionProvider
```

If you only get CPU, it may still run, but likely not real-time on Nano.

### Step 7: Clone repo on Jetson

```bash
git clone https://github.com/Kumar2421/scrfd_arcface_facerecognition.git
cd scrfd_arcface_facerecognition
```

### Step 8: Install only missing Python deps

Try not to use the full `requirements.txt`.

```bash
python3 -m pip install pillow requests
```

If `skimage` import fails, use apt package first:

```bash
sudo apt install -y python3-skimage
```

### Step 9: Download weights

```bash
mkdir -p weights
sh download.sh
```

If `download.sh` fails, download manually from the README links.

Use only:

```text
weights/det_500m.onnx
weights/w600k_mbf.onnx
```

### Step 10: Add known faces

```bash
mkdir -p faces
```

Example:

```text
faces/sameed.jpg
faces/ali.jpg
```

### Step 11: Apply same code fixes

In `main.py`, change:

```python
cap = cv2.VideoCapture(0)
```

to:

```python
cap = cv2.VideoCapture(params.source)
```

In `models/scrfd.py` and `models/arcface.py`, explicitly set ONNX Runtime providers as shown in the RTX section.

### Step 12: Run USB webcam

```bash
python3 main.py \
  --source 0 \
  --det-weight weights/det_500m.onnx \
  --rec-weight weights/w600k_mbf.onnx \
  --confidence-thresh 0.5 \
  --similarity-thresh 0.4 \
  --max-num 5
```

### Step 13: Run CSI camera

For a Raspberry Pi camera / CSI camera, OpenCV usually needs a GStreamer pipeline, not just `--source 0`.

You would need to modify `main.py` like this:

```python
if params.source == "csi":
    gst = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(params.source)
```

Then run:

```bash
python3 main.py \
  --source csi \
  --det-weight weights/det_500m.onnx \
  --rec-weight weights/w600k_mbf.onnx
```

---

## Jetson Performance Reality

This repo does detection + recognition every frame. On Jetson Nano that may be slow.

For real-time Nano performance, expect to add optimizations:

1. Use `det_500m.onnx`.
2. Use `w600k_mbf.onnx`.
3. Lower detector input size from `640x640` to `320x320`.
4. Resize camera frames to `640x480`.
5. Run recognition every N frames, not every frame.
6. Add tracking so recognition is not repeated constantly.
7. Consider TensorRT conversion if ONNX Runtime GPU is not fast enough.

Jetson Nano setup difficulty: **7/10**.

The main difficulty is not the face-recognition logic. It is the Jetson dependency stack: ARM wheels, old Python, CUDA 10.2, ONNX Runtime compatibility, and performance tuning.