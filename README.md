# RDF-YOLO: Remote Sensing Small Object Detection Based on YOLOv11

RDF-YOLO is an improved object detection model tailored for small object detection in aerial imagery. Built upon the YOLOv11 framework, it integrates a lightweight RepViT module and a dynamic fusion module (DFM) to enhance global feature modeling and multi-scale feature interaction.

## 🔍 Key Features

- **RepViT Module**: Lightweight transformer-based encoder for robust semantic modeling under noisy or low-resolution conditions.
- **DFM (Dynamic Fusion Module)**: Channel-adaptive, spatially-aware feature fusion design for effective multi-scale context integration.
- **SOTA Performance on AI-TOD**: Achieves high accuracy on very small objects while maintaining real-time inference capability.
- **Fully compatible with Ultralytics YOLOv11**.

## 🗂 Project Structure

. ├── datasets/ # Dataset config files (e.g. AITOD.yaml) ├── yolo11m_CF.yaml # Custom model configuration file ├── train.py # Training entry script ├── val.py # Validation/evaluation script (with APvt, APt, APs metrics) └── ultralytics/nn/conv.py # Modified YOLOv11 backbone containing RepViT and DFM


> This repository is a fork of [YOLOv11](https://github.com/ultralytics/ultralytics) with custom modules added to support RDF-YOLO.

## ⚙️ Environment

- Python 3.8+
- PyTorch 2.4.1
- CUDA 12.4
- torch-pruning 1.5.2
- torchvision 0.19.1

Install dependencies with:

```bash
pip install -r requirements.txt

🚀 Getting Started
Training

python train.py --model yolo11m_CF.yaml --data datasets/AITOD.yaml --epochs 600 --batch 16


📄 Citation

If you find this project useful, please consider citing the original YOLOv11 paper and our work (citation to be added).
📜 License

This project inherits the original YOLOv11 AGPL-3.0 License.

RDF-YOLO was developed for research on lightweight and high-precision small object detection in aerial remote sensing. Feel free to open issues or submit pull requests for improvement!