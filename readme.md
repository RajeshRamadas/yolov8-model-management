# YOLOv8 Model Repository

A comprehensive repository for managing YOLOv8 models with structured performance documentation.

## Overview

This repository provides an organized structure for:
- Storing YOLOv8 models for various tasks (detection, segmentation, classification)
- Documenting model performance metrics
- Comparing model variants
- Tracking datasets used for training and evaluation
- Managing model configuration files

## Repository Structure

The repository is organized as follows:

```
yolov8-models/
├── models/                  # Model files
├── datasets/                # Dataset information
├── performance/             # Performance documentation
├── scripts/                 # Utility scripts
├── configs/                 # Configuration files
└── docs/                    # Documentation
```

See the [full structure documentation](./docs/repository_structure.md) for details.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Ultralytics package

```bash
pip install ultralytics
```

### Adding a New Model

1. Place the model weights in the appropriate directory under `models/`:

```bash
# Example for a detection model
mkdir -p models/detection/my_yolov8n/weights
cp path/to/your/model.pt models/detection/my_yolov8n/weights/best.pt
```

2. Create a configuration file:

```bash
# Create model configuration
cp configs/templates/model_config_template.yaml models/detection/my_yolov8n/config.yaml
# Edit the configuration file with your model details
```

3. Run evaluation and generate documentation:

```bash
python scripts/evaluate.py \
  --model models/detection/my_yolov8n/weights/best.pt \
  --data datasets/your_dataset/dataset_info.yaml \
  --task detect \
  --variant n \
  --device 0 \
  --benchmark \
  --generate-report
```
Evaluate All Models in a Directory:

```bash
python enhanced_script.py --models-dir C:\path\to\models_folder --data C:\path\to\data.yaml --task detect --device cpu

```
Evaluate a Single Model:
```bash
python enhanced_script.py --model C:\path\to\specific\model.pt --data C:\path\to\data.yaml --task detect --device cpu

```

4. Additional Options:
	1. --non-recursive: Don't search subdirectories for models
	2. --skip-benchmark: Skip speed benchmarking (speeds up evaluation)
	3. --batch-size: Set the batch size for evaluation
	4. --output-dir: Specify the output directory for reports
	
## Performance Documentation

Models are documented in several formats:

1. **Model Cards**: Detailed information about each model, including performance metrics, training details, and use cases.
2. **Benchmark CSVs**: Structured data for comparing models across various metrics.
3. **Comparison Reports**: Analysis of model trade-offs between accuracy and speed.

## Example Usage

### Evaluating a Model

```python
from ultralytics import YOLO

# Load a model
model = YOLO('models/detection/yolov8n/weights/best.pt')

# Evaluate on a dataset
results = model.val(data='datasets/coco/dataset_info.yaml')

# Print metrics
print(results.box.map)    # mAP@0.5:0.95
print(results.box.map50)  # mAP@0.5
```

### Using a Model for Inference

```python
from ultralytics import YOLO

# Load a model
model = YOLO('models/detection/yolov8n/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for r in results:
    boxes = r.boxes  # Bounding boxes
    masks = r.masks  # Segmentation masks
    probs = r.probs  # Classification probabilities
```


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
