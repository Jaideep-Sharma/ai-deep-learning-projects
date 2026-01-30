# Churn Prediction using ANN

This project implements an end-to-end churn prediction system using a feedforward Artificial Neural Network (ANN) to predict customer churn based on historical data.

## Features
- Proper preprocessing & scaling with StandardScaler
- ANN with dropout regularization to prevent overfitting
- Early stopping & learning-rate scheduling for optimal training
- ROC-AUC based evaluation metrics
- Model + scaler persistence for production deployment
- Batch prediction on new customer data

## Project Structure
```
churn-prediction/
├── data/
│   ├── raw/                  # Raw customer data
│   ├── processed/            # Preprocessed train/test splits
│   └── prediction/           # New customer data for predictions
├── models/
│   └── v1/                   # Saved model, scaler, and metadata
├── notebook/                 # Jupyter notebooks for exploration
├── src/                      # Source code modules
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── build_model.py        # Model architecture definition
│   ├── train.py              # Training logic
│   ├── evaluate.py           # Evaluation metrics
│   └── inference.py          # Prediction pipeline
├── run.py                    # Main CLI for training pipeline
├── prediction.py             # CLI for batch predictions
└── requirement.txt           # Python dependencies
```

## Installation

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

## Usage

### Full Pipeline (Preprocess + Train + Evaluate)

#### Step 1: Preprocess Data
Prepare raw data for training by splitting into train/test sets and creating a scaler.
```bash
python run.py preprocess
```

**With custom paths:**
```bash
python run.py preprocess --raw-data data/raw/synthetic_customer_churn_100k.csv --processed-dir data/processed --model-dir models/v1
```

#### Step 2: Train Model
Train the ANN model with the preprocessed data.
```bash
python run.py train
```

**With custom paths:**
```bash
python run.py train --processed-dir data/processed --model-dir models/v1
```

#### Step 3: Evaluate Model
Evaluate the trained model on test data and generate metrics.
```bash
python run.py evaluate
```

**With custom paths:**
```bash
python run.py evaluate --processed-dir data/processed --model-dir models/v1
```

### Quick Start (All-in-One)
Run all steps sequentially:
```bash
python run.py preprocess && python run.py train && python run.py evaluate
```

### Making Predictions on New Data

#### Predict Churn for New Customers
Use the trained model to predict churn probability for new customers.
```bash
python prediction.py --input-csv data/prediction/new_customers.csv
```

**With custom output and model directory:**
```bash
python prediction.py --input-csv data/prediction/new_customers.csv --model-dir models/v1 --output-csv results/predictions.csv
```

**Parameters:**
- `--input-csv`: Path to CSV file with new customer data (required)
- `--model-dir`: Directory containing trained model artifacts (default: `models/v1`)
- `--output-csv`: Path to save prediction results (default: `predictions.csv`)

## Command Reference

### run.py Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `preprocess` | Preprocess raw data, split into train/test, save scaler | `python run.py preprocess` |
| `train` | Train ANN model on preprocessed data | `python run.py train` |
| `evaluate` | Evaluate trained model and display metrics | `python run.py evaluate` |

**Common Arguments:**
- `--raw-data`: Path to raw CSV data (default: `data/raw/synthetic_customer_churn_100k.csv`)
- `--processed-dir`: Directory for processed datasets (default: `data/processed`)
- `--model-dir`: Directory for model artifacts (default: `models/v1`)

### prediction.py Command

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--input-csv` | Path to new customer data CSV | Yes | - |
| `--model-dir` | Directory with trained model | No | `models/v1` |
| `--output-csv` | Path to save predictions | No | `predictions.csv` |

## Model Architecture
- Input layer: Based on feature count after preprocessing
- Hidden layers: Dense layers with ReLU activation
- Dropout layers: For regularization (prevents overfitting)
- Output layer: Single neuron with sigmoid activation (binary classification)
- Optimizer: Adam with learning rate scheduling
- Loss function: Binary crossentropy
- Callbacks: Early stopping, learning rate reduction

## Output Files

### After Preprocessing
- `data/processed/X_train.csv` - Training features
- `data/processed/X_test.csv` - Test features
- `data/processed/y_train.csv` - Training labels
- `data/processed/y_test.csv` - Test labels
- `models/v1/scaler.pkl` - Fitted StandardScaler

### After Training
- `models/v1/model.keras` - Trained Keras model
- `models/v1/metadata.json` - Model metadata (feature names, etc.)

### After Prediction
- `predictions.csv` (or custom name) - Original data + churn_probability + churn_prediction columns

## Evaluation Metrics
The model is evaluated using:
- **ROC-AUC Score**: Area under the ROC curve
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- tensorflow
- joblib

## Tips
1. Always run `preprocess` before `train` on first run
2. Model artifacts are saved in `models/v1/` by default
3. Use same `--model-dir` for train, evaluate, and prediction
4. Input CSV for predictions must have same features as training data
5. Activate virtual environment before running any commands

## Troubleshooting

### Issue: Module not found
**Solution:** Ensure all dependencies are installed:
```bash
pip install -r requirement.txt
```

### Issue: File not found error
**Solution:** Check that data files exist in correct directories or specify custom paths with arguments.

### Issue: Model not found during prediction
**Solution:** Ensure model is trained first and `--model-dir` points to correct location:
```bash
python run.py train --model-dir models/v1
python prediction.py --input-csv data/prediction/new_customers.csv --model-dir models/v1
```

## License
See LICENSE file for details.

## Contributing
Feel free to open issues or submit pull requests for improvements.
