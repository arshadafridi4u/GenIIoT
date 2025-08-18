### Title
Predictive Maintenance AI Application: Class-Imbalance Handling with SMOTE and Baseline ML Models

### Description
This repository contains code and analysis for an AI Application study on predictive maintenance using the AI4I machine condition monitoring data. We evaluate classic machine learning models for binary failure prediction under severe class imbalance and explore class-balancing strategies (SMOTE oversampling and simple resampling). The work is intended to accompany a manuscript submission to PeerJ Computer Science under the AI Application article type.

### Repository layout
- `Models with SMOTE.py`: Loads the AI4I dataset, applies SMOTE to the training split, and trains/evaluates Decision Tree, SVM (linear kernel), and KNN classifiers. Generates classification metrics, confusion matrices, and example precision–recall visualizations.
- `Predictive Models.py`: Baseline experiments on the AI4I dataset, including downsampling for speed, a constrained Decision Tree, and robustness stress tests (label noise). Also trains SVM and KNN and reports metrics with confusion matrices and PR curves.
- `models on GANs.py`: Experiments using a second CSV (`fake_data.csv`) with manual resampling to create a more balanced subset; trains Decision Tree, SVM, and KNN and reports metrics. Includes an initial Linear Regression baseline (for reference only; classification metrics use the classifiers).
- `*.ipynb`: Notebooks with related experiments (e.g., GAN-based augmentation investigations) and dataset-specific workflows.
- `GenIIoT_Peerj_Computer_Science (1).pdf`: The manuscript (research paper) associated with this codebase.

### Dataset information
- **Primary dataset**: AI4I 2020 Predictive Maintenance Dataset (commonly distributed via the UCI Machine Learning Repository and other sources).
- **Core features used**: `UDI`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, and failure subtype indicators `TWF`, `HDF`, `RNF`, `OSF`, `PWF`.
- **Target**: `Machine failure` (binary: 1 = failure, 0 = non-failure).
- **Expected file names**: `pdm.csv` (same schema as AI4I). Some experiments in `models on GANs.py` also reference `fake_data.csv` for synthetic/alternative data.
- **Imbalance**: Minority failure rate is low, which motivates SMOTE and resampling strategies.

How to obtain data:
- Download AI4I 2020 Predictive Maintenance Dataset and save as `data/pdm.csv` (recommended) or update the script paths to your local copy.
- If providing data as supplementary material to PeerJ, include `pdm.csv` (if licensing permits) or a link and checksum.

### Code information
- **Feature engineering/preprocessing**: Column selection; optional casting to integers for consistency; optional downsampling of rows (for faster iteration) in `Predictive Models.py`.
- **Balancing strategies**:
  - SMOTE applied only to the training split (`Models with SMOTE.py`) using `imblearn.SMOTE`.
  - Manual resampling to match class counts (`models on GANs.py`) via row sampling.
- **Models**: Decision Tree (`sklearn.tree.DecisionTreeClassifier`), Support Vector Machine (`sklearn.svm.SVC`, linear kernel), and K-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`). A Linear Regression baseline appears in one script for a quick numeric baseline, but classification conclusions rely on the classifiers.
- **Metrics/plots**: Accuracy, Precision, Recall, F1, Classification Report, Confusion Matrix; example Precision–Recall curves for imbalanced evaluation.

File-specific notes:
- `Models with SMOTE.py`
  - Loads `pdm.csv`, splits into train/test, applies SMOTE to the training set, trains DT/SVM/KNN, and evaluates on the untouched test set.
  - Example PR plots are currently generated with randomly sampled values for illustration; see "Reproducibility and correctness" below to plot PR using actual model scores.
- `Predictive Models.py`
  - Includes an example downsampling routine to create a `pdm_downsampled.csv` and exploratory robustness tests (e.g., shallow trees and injected label noise to simulate annotation errors).
- `models on GANs.py`
  - Uses `fake_data.csv` and simple majority/minority sampling to construct a working dataset (`selected_data`) for baseline comparisons; evaluates DT/SVM/KNN.

### Usage instructions
1) Set up environment
- Python 3.9+ recommended.
- Install dependencies:
```
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
# Optional (imported in some files, not required unless extending NN/GAN experiments):
pip install tensorflow keras
```

2) Place data
- Create a `data/` folder and copy `pdm.csv` there. If you have a synthetic/alternative file, place it as `data/fake_data.csv`.
- Update the file paths in scripts from absolute Windows paths (e.g., `C:\\Users\\...\\pdm.csv`) to your local path. For example, change to `data/pdm.csv`.

3) Run experiments
- SMOTE-based baselines:
```
python "Models with SMOTE.py"
```
- Baselines and robustness (downsampling/noise):
```
python "Predictive Models.py"
```
- Manual resampling baselines with alternative data:
```
python "models on GANs.py"
```

4) Recommended corrections for PR curves
- Replace placeholder PR calculations (currently using randomly sampled arrays) with actual model scores:
  - For SVM: use `decision_function` or `predict_proba` (if configured) to obtain continuous scores.
  - For DT/KNN: use `predict_proba(X_test)[:, 1]`.
- Example snippet:
```
from sklearn.metrics import precision_recall_curve
scores = model.predict_proba(X_test)[:, 1]  # or decision_function(...) for linear SVM
precision, recall, thresholds = precision_recall_curve(y_test, scores)
```

### Requirements
- **Python**: 3.9+ (tested on Windows 10)
- **Libraries**:
  - numpy, pandas
  - scikit-learn
  - imbalanced-learn (for SMOTE)
  - matplotlib, seaborn
  - Optional: tensorflow/keras (imported but not required for the baseline ML scripts)

### Methodology
- **Problem**: Binary predictive maintenance — predict `Machine failure` from sensor and operational features.
- **Preprocessing**:
  - Select relevant features from the AI4I schema.
  - Optionally cast selected columns to integers for consistency.
  - Optional row downsampling in `Predictive Models.py` when exploring quickly.
- **Handling class imbalance**:
  - SMOTE applied exclusively to the training split to avoid leakage.
  - Manual resampling comparisons using random undersampling/oversampling on copies of the data.
- **Models and configuration**:
  - Decision Tree with default or shallow depth (for interpretability and stress testing).
  - Linear-kernel SVM.
  - KNN with configurable neighbors (k = 6 in provided examples).
- **Evaluation method (procedure)**:
  - Hold-out validation via `train_test_split` (e.g., 80/20 or 70/30 as configured per script) with a fixed `random_state` for reproducibility.
  - For SMOTE runs, oversampling is performed on the training split only; the test split remains untouched to simulate real-world class imbalance.
  - Metrics computed on the held-out test split: Accuracy, Precision, Recall, F1; confusion matrices visualized with seaborn.
  - Imbalanced-performance visualization via Precision–Recall curves (see "Recommended corrections" to compute using actual scores).
  - Comparative analysis across balancing strategies (no balancing vs. SMOTE vs. manual resampling) and models (DT vs. SVM vs. KNN).
  - Optional robustness checks: label-noise injection in `Predictive Models.py` to assess model sensitivity to annotation errors.
- **Recommended extensions (if needed by reviewers)**:
  - Stratified k-fold cross-validation (e.g., 5-fold) with SMOTE inside each training fold to report mean/std metrics.
  - Ablation study isolating the impact of each balancing strategy and feature subset.
  - Calibrated probability estimates and threshold selection analysis for operational deployment.

### Reproducibility
- Use fixed seeds (`random_state=42` or as specified) for data splits and SMOTE.
- Keep the test set completely separate and unmodified by oversampling or noise procedures.
- Make file paths relative (e.g., `data/pdm.csv`) instead of absolute local Windows paths.
- Record the exact library versions with `pip freeze > requirements.txt` if submitting as supplemental material.

### Materials & Methods (computing infrastructure)
- **Operating system**: Tested on Windows 10 (should also work on Linux/macOS with minor path changes).
- **Hardware**: CPU-only is sufficient for all included scripts; typical laptop (e.g., 4–8 cores, 8–16 GB RAM). GPU not required unless running optional deep-learning notebooks.
- **Software**: Python 3.9+, packages listed above.

### Citations
- AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. Include URL/DOI as appropriate for your submission context.
- Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. Journal of Machine Learning Research.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
- If you use or extend GAN-based augmentation in notebooks, cite the specific GAN method(s) and implementations used.

### License & contribution
- **License**: to be specified by the authors. If submitting to PeerJ, include the desired code/data license in the supplemental materials.
- **Contributions**: Please open issues or pull requests describing proposed changes and their motivation.

### Known limitations and notes
- Some PR curve sections in the scripts currently use randomly generated arrays for illustration; replace with actual model scores for publication-quality plots (see instructions above).
- Several scripts contain absolute Windows paths; convert to relative paths under a `data/` directory for portability.
- The `models on GANs.py` file name references GANs, but the provided script implements classical ML baselines; GAN-based augmentation is explored in notebooks, not in this script.

### Contact
For questions or to request the exact environment specification used for the manuscript, please contact the authors listed in `GenIIoT_Peerj_Computer_Science (1).pdf`.
