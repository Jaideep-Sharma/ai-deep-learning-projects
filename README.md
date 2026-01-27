# AI Deep Learning Projects

This repository contains a collection of **hands-on deep learning projects** built using **TensorFlow** and **Keras**. The goal of this repo is to practice, apply, and document core deep learning concepts across different problem types using **ANN, CNN, and RNN** models.

The focus is on building models end to end — from data preparation to training, evaluation, and interpretation — while keeping the structure clean and easy to navigate.

---

## Purpose of this repository

This repository is intended to:
- Practice building deep learning models on real datasets
- Understand when to use **ANN, CNN, or RNN** based on the data
- Apply standard training and validation techniques
- Keep all projects organized in a single place for reference

Each project is self-contained and documented separately.

---

## Repository structure

```
ai-deep-learning-projects/
│
├── README.md                  # Repository overview and usage guide
│
├── ann/                       # ANN projects (tabular data)
│   ├── churn-prediction/
│   │   └── README.md
│   └── house-price-regression/
│       └── README.md
│
├── cnn/                       # CNN projects (image data)
│   └── image-classification/
│       └── README.md
│
├── rnn/                       # RNN projects (sequential data)
│   └── time-series-forecasting/
│       └── README.md
│
├── common/                    # Shared utilities
│   ├── preprocessing.py
│   ├── metrics.py
│   └── utils.py
│
├── requirements.txt           # Project dependencies
└── LICENSE
```

---

## How to use this repository

### 1. Clone the repository
```bash
git clone https://github.com/Jaideep-Sharma/ai-deep-learning-projects.git
cd ai-deep-learning-projects
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\\Scripts\\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a project
Each project folder contains its own **README.md** with instructions on:
- Dataset setup
- Model training
- Evaluation steps

Navigate into the project you want to run, for example:
```bash
cd ann/churn-prediction
```

---

## Project categories

### ANN (Artificial Neural Networks)
Used mainly for **tabular datasets**, such as:
- Customer churn prediction
- Regression problems

### CNN (Convolutional Neural Networks)
Used for **image-based tasks**, such as:
- Image classification

### RNN (Recurrent Neural Networks)
Used for **sequential data**, such as:
- Time series forecasting

---

## Training and evaluation approach

Across projects, the following practices are generally used:
- Train / validation / test split
- Monitoring validation loss to check generalization
- Early stopping to avoid overfitting
- Metrics chosen based on the problem type
  - Classification: accuracy, precision, recall, ROC-AUC
  - Regression: MSE, RMSE, MAE, R²

Exact details and results are documented in each project README.

---

## Datasets

Datasets are not committed directly to the repository. Instead:
- Dataset sources are linked in project READMEs
- Instructions are provided to download the data

This keeps the repository lightweight and easy to clone.

---

## Tools and libraries

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Future work

- Add more datasets and experiments
- Compare different model architectures
- Experiment with hyperparameter tuning

---

## License

This repository uses the **MIT License**.

---

## Notes

This repository is meant to evolve over time as new projects are added and existing ones are improved.

