# 🌌 GalaxyVision AI

An end-to-end Deep Learning project for **Galaxy Morphology Classification** using the Galaxy Zoo dataset.

The model classifies galaxies into scientifically meaningful morphology categories and provides explainable predictions using Grad-CAM.

---

## 🎯 Project Goal

Build a computer vision system that can classify galaxy images into morphology classes such as:

- Round Elliptical
- Intermediate Elliptical
- Edge-On Disk
- Barred Spiral
- Unbarred Spiral
- Irregular / Merger

The project follows a real-world ML workflow:

1. Dataset Understanding
2. Label Engineering
3. Data Cleaning
4. Model Training
5. Explainability (Grad-CAM)
6. Web Deployment (Streamlit)

---

## 📂 Dataset

Dataset: Galaxy Zoo

The original dataset contains galaxy images along with volunteer vote distributions for multiple morphology-related questions.

Instead of using predefined labels, morphology labels were engineered from Galaxy Zoo's hierarchical decision tree.

---

## 🏗️ Morphology Classes

| Class | Description |
|---------|------------|
| Round_Elliptical | Smooth and round galaxy |
| Intermediate_Elliptical | Smooth but elongated galaxy |
| EdgeOn_Disk | Disk galaxy viewed from the side |
| Barred_Spiral | Spiral galaxy with central bar |
| Unbarred_Spiral | Spiral galaxy without central bar |
| Irregular_Merger | Disturbed or merging galaxy |

---

## 🔬 Label Engineering

Galaxy Zoo provides probabilities instead of labels.

Morphology scores were generated using the Galaxy Zoo decision hierarchy.

Example:

```python
Round_Elliptical = Class1.1 * Class7.1

Barred_Spiral = (
    Class1.2 *
    Class4.1 *
    Class3.1
)
```

The morphology with the highest score becomes the final label.

---

## 🧹 Dataset Cleaning

### Confidence-Based Filtering

Each galaxy receives a confidence score:

```python
Confidence = max(morphology_scores)
```

Only galaxies with:

```python
Confidence >= 0.4
```

are retained.

This removes ambiguous samples and improves label quality.

---

## 📊 Dataset Statistics

Initial Dataset: ~61,000 galaxies

Final Clean Dataset:

- Total Samples: 26,281
- Classes: 6

Class Distribution:

- Round Elliptical
- Intermediate Elliptical
- Edge-On Disk
- Unbarred Spiral
- Barred Spiral
- Irregular / Merger

---

## Project Progress

| Day | Work Completed | Result |
|-----|---------------|---------|
| Day 1 | Data Cleaning & Morphology Label Creation | 26k high-confidence galaxy samples |
| Day 2 | Baseline CNN Development | Validation Accuracy: 73.4% |
| Day 3 | Transfer Learning with ResNet50 | Accuracy: 86.45%, Macro F1: 0.8225 |
| Day 4 | Grad-CAM & Error Analysis | Interpretable predictions and failure analysis |

---

## Key Metrics

| Metric | Value |
|----------|---------|
| Original Samples | 61,578 |
| Clean Samples | 26,281 |
| Number of Classes | 7 |
| Model | Custom CNN |
| Epochs | 20 |
| Validation Accuracy | 73.4% |
| Framework | PyTorch |
| Hardware | Tesla T4 GPU |

The baseline CNN achieved 73.4% validation accuracy with a weighted F1 score of 71.3%. The lower macro F1 score (62.1%) indicates reduced performance on minority morphology classes, motivating the use of transfer learning with ResNet50 in the next phase of the project.

---

## Skills Acquired

| Day 1 | Day 2 |
|---------|---------|
| Dataset Cleaning | CNN Architecture |
| Label Engineering | PyTorch Dataset & DataLoader |
| Confidence Thresholding | GPU Training |
| Data Splitting | Cross Entropy Loss |
| Class Distribution Analysis | Model Evaluation |

---

### Confusion Matrix Insights

The ResNet50 model achieved strong class separation, with most predictions concentrated along the diagonal of the confusion matrix.

Major confusion pairs included:

- Barred Spiral ↔ Unbarred Spiral
- Round Elliptical ↔ Intermediate Elliptical
- Unbarred Spiral ↔ Irregular Merger

These errors are scientifically reasonable because the paired classes share similar visual morphology. The model demonstrated excellent performance on Edge-On Disk and Round Elliptical galaxies, while Irregular Merger remained the most challenging class due to its highly variable structure.

---

## Day 4: Explainability & Error Analysis

| Task | Outcome |
|--------|---------|
| Implemented Grad-CAM | Visualized regions used by ResNet50 for predictions |
| Generated heatmaps for all galaxy classes | Verified model focuses on galaxy structures |
| Performed error analysis | Investigated common misclassification patterns |
| Analyzed prediction confidence | Evaluated model certainty across classes |

### Key Learnings

- Grad-CAM helps interpret CNN predictions by highlighting important image regions.
- The model primarily focused on galaxy cores and structural features.
- Background stars and empty regions contributed minimally to predictions.
- Most classification errors occurred between visually similar galaxy morphologies.
- Explainability is essential for validating model reliability in scientific applications.

### Observations

- Spiral galaxies showed strong activation around central bulges and surrounding arm structures.
- Elliptical galaxies exhibited broader activation across the galaxy body.
- Irregular and merger galaxies activated around asymmetric and distorted regions.
- Misclassified samples often contained ambiguous morphology or weak structural features.

### Grad-CAM Examples

| Example | Screenshot |
|----------|-----------|
| Unbarred Spiral | Insert Image |
| Round Elliptical | Insert Image |
| Irregular Merger | Insert Image |
| Misclassified Galaxy | Insert Image |

---

## 🛠️ Tech Stack

### Data Processing
- Python
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Deep Learning
- PyTorch
- Torchvision

### Explainability
- Grad-CAM

### Deployment
- Streamlit

### Development
- Google Colab

---

## 📈 Future Enhancements

- Hierarchical Classification
- Redshift Estimation
- Galaxy Similarity Search
- Anomaly Detection
- Galaxy-to-Pokémon Mapping

---
