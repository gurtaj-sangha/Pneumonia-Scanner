# Pneumonia-Scanner

A web-based pneumonia classification tool that analyzes chest X-ray images and provides real-time diagnostic predictions using deep learning.

## Features

- **Data Organization**: Automatically splits pneumonia images by type (bacterial/viral)
- **Preprocessing Pipeline**: Handles image loading, normalization, and augmentation
- **Class Balance**: Calculates class weights for imbalanced datasets
- **Visualization**: Generates sample images and dataset statistics
- **Extensible**: Ready for model training and web interface integration

**Model Building**: Created a ResNet50-based classification model
**Training Pipeline**: Implemented training with proper validation
**Web Interface**: Create Flask/FastAPI backend and HTML frontend

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Organize Your Data
Run the splitter script to organize pneumonia images by type:
```bash
python src/splitter.py
```

### 3. Test Your Setup
Verify everything is working:
```bash
python src/test_setup.py
```

### 4. Explore Your Dataset
Get detailed statistics about your data:
```bash
python src/data_explorer.py
```

### 5. Test the Data Pipeline
Test the preprocessing pipeline:
```bash
python src/data_pipeline.py
```

## Data Structure

After running the splitter, your data should be organized as:
```
data/
├── train/
│   ├── NORMAL/
│   ├── bacterial_pneumonia/
│   └── viral_pneumonia/
├── val/
│   ├── NORMAL/
│   ├── bacterial_pneumonia/
│   └── viral_pneumonia/
└── test/
    ├── NORMAL/
    ├── bacterial_pneumonia/
    └── viral_pneumonia/
```

## Project Structure

```
pneumonia-scanner/
├── src/
│   ├── config.py                 # Configuration settings
│   ├── splitter.py              # Data organization script
│   ├── data_explorer.py         # Dataset analysis
│   ├── data_pipeline.py         # Preprocessing pipeline
│   └── test_setup.py            # Setup verification
├── data/                        # Dataset (excluded from git)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```



## Credits

Dataset: Kermany, Daniel; Zhang, Kang; Goldbaum, Michael(2018), "Labeled Optical COherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, v2

Source: http://cell.com/cell/fulltext/S0092-8674(18)30154-5
