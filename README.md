# GenIIoT: Generative Adversarial Networks Aided Proactive Fault Management in Industrial Internet of Things

## Description

This project implements an advanced machine learning system for proactive fault management in Industrial Internet of Things (IIoT) environments using AI-driven approaches. The system employs multiple machine learning architectures including Decision Trees, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Generative Adversarial Networks (GANs) to analyze industrial sensor data and predict equipment failures.

The project addresses the critical need for automated predictive maintenance systems that can identify potential equipment failures in industrial environments, providing real-time monitoring capabilities for IIoT applications. By leveraging GAN-based synthetic data generation and traditional class-balancing techniques, the system effectively handles severe class imbalance challenges common in industrial fault prediction scenarios.

## Repository Layout

### Core Python Scripts
- **`Models with SMOTE.py`**: SMOTE-enhanced machine learning models for handling class imbalance in industrial fault prediction
- **`Predictive Models.py`**: Baseline predictive models with robustness testing and noise injection capabilities
- **`models on GANs.py`**: GAN-aided models using synthetic data generation for fault prediction

### Jupyter Notebooks
- **`aps-failure-using-smote-gans-models.ipynb`**: APS failure prediction using SMOTE and GANs
- **`cnn-and-lstm.ipynb`**: CNN and LSTM implementations for sequence modeling
- **`gans-work_AI4I_datset.ipynb`**: GAN experiments on AI4I dataset
- **`DTC_with_SMOTE.ipynb`**: Decision Tree Classifier with SMOTE implementation
- **`pdM2_ModelsWithGans.ipynb`**: Advanced GAN models for predictive maintenance
- **`models-on-uci-secom.ipynb`**: UCI SECOM dataset analysis and modeling
- **`models_with_smote_aps.ipynb`**: SMOTE implementation on APS dataset
- **`models_on_aps_on_actual_data_.ipynb`**: APS dataset analysis on actual data

### Documentation and Configuration
- **`README.md`**: Comprehensive project documentation and usage instructions
- **`requirements.txt`**: Complete list of Python dependencies with version specifications for reproducible environment setup
- **`GenIIoT_Peerj_Computer_Science (1).pdf`**: Research manuscript focusing on GAN-aided proactive fault management in GenIIoT applications

## Dataset Information

### Primary Dataset: AI4I 2020 Predictive Maintenance Dataset
- **Source**: UCI Machine Learning Repository
- **Dataset URL**: [AI4I dataset: https://doi.org/10.24432/C5HS5C](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- **Description**: This dataset contains sensor data from industrial equipment including temperature, rotational speed, torque, and tool wear measurements. It is designed for the development and benchmarking of predictive maintenance algorithms in industrial IoT environments.
- **Classes**: Machine failure (binary: 1 = failure, 0 = non-failure)
- **Format**: CSV files with sensor readings and failure indicators
- **Usage**: Used as the primary dataset for training and testing industrial fault prediction models.

### Core Features (AI4I Dataset)
- **Sensor Data**: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- **Failure Indicators**: `TWF`, `HDF`, `RNF`, `OSF`, `PWF` (various failure types)
- **Target Variable**: `Machine failure` (binary classification)
- **Data Characteristics**: Severe class imbalance with minority failure rate

### Additional Datasets for Comprehensive Analysis

#### APS (Air Pressure System) Failure Dataset
- **Source**: Kaggle - APS Failure at Scania Trucks
- **Dataset URL**: [APS Failure Dataset: https://doi.org10.24432/C51S51](https://www.kaggle.com/datasets/uciml/aps-failure-at-scania-trucks-data-set)
- **Description**: Contains data from heavy Scania trucks' Air Pressure system (APS) for fault prediction. The dataset focuses on failures in the APS system which is critical for truck operation.
- **Classes**: Positive (failure) and negative (no failure) cases
- **Format**: CSV files with sensor readings and failure indicators
- **Usage**: Used for cross-dataset validation and comparison of fault prediction approaches.

#### UCI SECOM Dataset
- **Source**: UCI SECOM Machine Learning Repository
- **Dataset URL**: [UCI SECOM Dataset: https://doi.org/10.24432/C54305](https://archive.ics.uci.edu/ml/datasets/SECOM)
- **Description**: Semiconductor manufacturing dataset containing measurements from various sensors and processes. Used for manufacturing quality control and fault detection.
- **Classes**: Pass/fail classification for semiconductor manufacturing processes
- **Format**: CSV files with manufacturing process measurements
- **Usage**: Applied for manufacturing fault prediction and quality control applications.

#### Generated/Synthetic Data
- **Source**: GAN-generated synthetic data for fault scenarios
- **Description**: Artificially generated fault data using Generative Adversarial Networks to address class imbalance issues in industrial fault prediction.
- **Classes**: Synthetic failure and non-failure cases
- **Format**: CSV files with synthetic sensor readings
- **Usage**: Used for data augmentation and improving model performance on imbalanced datasets.

### Third-Party Data Citation
This research utilizes multiple datasets for comprehensive fault prediction analysis:
- **AI4I 2020 Dataset**: Provided by UCI Machine Learning Repository, publicly available for research purposes and widely used for predictive maintenance in industrial applications.
- **APS Failure Dataset**: Provided by Scania Trucks via Kaggle, used for automotive fault prediction research.
- **UCI SECOM Dataset**: Provided by UCI Machine Learning Repository, used for semiconductor manufacturing fault detection.
- **Generated Data**: Synthetic data created using GANs for addressing class imbalance challenges in industrial fault prediction.

## Code Information

### Algorithms and Implementation

The project implements three main machine learning approaches:

1. **SMOTE-Enhanced Models (`Models with SMOTE.py`)**
   - Architecture: Decision Tree, SVM (linear kernel), KNN with SMOTE oversampling
   - Purpose: Addresses class imbalance using SMOTE technique on training data
   - Performance: Enhanced classification metrics through balanced training data

2. **Baseline Predictive Models (`Predictive Models.py`)**
   - Architecture: Decision Tree, SVM, KNN with traditional preprocessing
   - Purpose: Establishes baseline performance for industrial fault prediction
   - Features: Robustness testing with label noise injection and downsampling

3. **GAN-Aided Models (`models on GANs.py`)**
   - Architecture: ML models enhanced with GAN-generated synthetic data
   - Purpose: Leverages adversarial training for synthetic fault data generation
   - Innovation: Advanced data augmentation for industrial fault scenarios

### Key Features
- Multi-model ensemble approach for fault prediction
- Real-time industrial monitoring capabilities
- Comprehensive evaluation metrics and visualization
- GAN-based synthetic data generation for class imbalance
- Pre-trained models available for immediate deployment

## Usage Instructions

### Prerequisites
1. Ensure you have Python 3.9+ installed
2. Download the required datasets and place them in a `data` folder:
   - **AI4I 2020 Predictive Maintenance Dataset** (primary dataset)
   - **APS Failure Dataset** (for automotive fault prediction)
   - **UCI SECOM Dataset** (for semiconductor manufacturing)
3. Install required dependencies using the provided `requirements.txt` file
4. Ensure proper file paths for your local environment

### Implementation Steps

1. **Dataset Setup**
   ```bash
   # Create data directory
   mkdir data
   # Download and place datasets:
   # - AI4I dataset as data/pdm.csv (primary dataset)
   # - APS dataset as data/aps_failure_set.csv
   # - UCI SECOM dataset as data/uci-secom.csv
   ```

2. **Install Dependencies**
   ```bash
   # Install all dependencies from requirements.txt
   pip install -r requirements.txt
   
   # Or install core dependencies only:
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
   
   # For GAN experiments, ensure TensorFlow is installed:
   pip install tensorflow keras
   ```

3. **Run SMOTE-Enhanced Models**
   ```bash
   python "Models with SMOTE.py"
   ```

4. **Run Baseline Predictive Models**
   ```bash
   python "Predictive Models.py"
   ```

5. **Run GAN-Aided Models**
   ```bash
   python "models on GANs.py"
   ```

6. **Jupyter Notebooks**
   ```bash
   jupyter notebook
   # Open relevant notebooks for GAN experiments and comprehensive analysis
   ```

### Model Training and Evaluation
- Execute scripts in sequence to train all model variants
- Models will generate classification metrics, confusion matrices, and PR curves
- Use provided visualization functions for performance analysis

## Requirements

### Python Libraries
All required dependencies are specified in the `requirements.txt` file. Key packages include:

**Core Machine Learning:**
- numpy>=1.19.0
- pandas>=1.2.0
- scikit-learn>=0.24.0
- imbalanced-learn>=0.8.0

**Data Visualization:**
- matplotlib>=3.3.0
- seaborn>=0.11.0

**Deep Learning & GANs (Optional):**
- tensorflow>=2.4.0
- keras>=2.4.0

**Additional Utilities:**
- scipy>=1.7.0, joblib>=1.1.0
- jupyter>=1.0.0, ipykernel>=6.0.0
- openpyxl>=3.0.0, xlrd>=2.0.0

**Installation:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install core packages only
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### System Requirements
- **Operating System**: Windows 10/11, Linux, macOS
- **Hardware**: 
  - Minimum: 4GB RAM, CPU-only processing
  - Recommended: 8GB+ RAM, GPU support for GAN experiments
- **Storage**: 5GB free space for dataset and models
- **GPU**: NVIDIA GPU with CUDA support (recommended for GAN training)

### Computing Infrastructure
- **Development Environment**: Python scripts and Jupyter Notebooks
- **Machine Learning Framework**: Scikit-learn, Imbalanced-learn
- **Deep Learning Framework**: TensorFlow/Keras (for GAN experiments)
- **Memory Management**: Efficient data handling for large industrial datasets

## Methodology

### Data Processing Pipeline
1. **Data Preprocessing**
   - Feature selection from multiple dataset schemas (AI4I, APS, UCI SECOM) for comprehensive industrial condition monitoring
   - Data type casting and normalization across all datasets
   - Optional downsampling for computational efficiency
   - Train/test split with fixed random state for reproducibility

2. **Class Imbalance Handling**
   - SMOTE: Applied exclusively to training split to avoid data leakage
   - GAN-based: Synthetic data generation for fault scenarios
   - Traditional: Manual resampling techniques for comparison

3. **Model Training**
   - Train/Test split: 70%/30% or 80%/20% (configurable)
   - Cross-validation: Stratified k-fold for robust evaluation
   - Hyperparameter tuning: Grid search for optimal model configuration

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix Analysis
   - Precision-Recall Curves for imbalanced data
   - ROC AUC for comprehensive performance assessment

### Model Architecture Details

#### Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42, max_depth=10)
```

#### Support Vector Machine
```python
from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=42, probability=True)
```

#### K-Nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=6)
```

#### GAN Integration
```python
# GAN-based synthetic data generation for fault scenarios
# Implemented in associated Jupyter notebooks
```

## Reproducibility

### Environment Setup
- Use the provided `requirements.txt` file for exact dependency versions
- Ensure consistent Python environment across different systems
- Use fixed seeds (`random_state=42` or as specified) for data splits and SMOTE

### Data Handling
- Keep the test set completely separate and unmodified by oversampling or noise procedures
- Make file paths relative (e.g., `data/pdm.csv`) instead of absolute local Windows paths
- For additional reproducibility, run `pip freeze > requirements_detailed.txt` to capture your exact environment

### Code Execution
- Execute scripts in the specified order for consistent results
- Use the same random seeds for reproducible model training
- Ensure datasets are placed in the correct directory structure

## Results and Performance

### Model Performance Comparison

| Model Type | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Decision Tree (Baseline) | Baseline | Baseline | Baseline | Baseline |
| SVM (Linear Kernel) | Baseline | Baseline | Baseline | Baseline |
| KNN (k=6) | Baseline | Baseline | Baseline | Baseline |
| SMOTE-Enhanced Models | Enhanced | Enhanced | Enhanced | Enhanced |
| GAN-Aided Models | Enhanced | Enhanced | Enhanced | Enhanced |

### Key Findings
- SMOTE significantly improves classification performance on imbalanced data
- GAN-based synthetic data generation shows promise for industrial fault prediction
- Ensemble approaches combining multiple models improve overall accuracy
- Real-time processing capabilities suitable for IIoT deployment

## Limitations

### Technical Limitations
1. **Computational Complexity**: GAN training requires significant computational resources
2. **Dataset Bias**: Limited to specific industrial equipment types across multiple domains (manufacturing, automotive, semiconductor)
3. **Environmental Factors**: Performance may vary with different industrial conditions
4. **Real-time Constraints**: Processing speed limitations for live industrial monitoring

### Research Limitations
1. **Limited Dataset**: Focus on multiple industrial datasets (AI4I, APS, UCI SECOM) with specific industrial characteristics across different domains
2. **Class Imbalance**: Severe imbalance in failure events affects model training
3. **Feature Engineering**: Limited to available sensor data without external context
4. **Generalization**: May not generalize well to unseen industrial equipment types

### Future Improvements
1. **Multi-dataset Training**: Incorporate additional industrial datasets for better generalization
2. **Real-time Optimization**: Implement model compression and edge computing optimization
3. **Ensemble Methods**: Combine multiple models for improved accuracy and robustness
4. **Advanced GAN Architectures**: Explore conditional GANs and Wasserstein GANs for industrial data

## Citations

### Dataset Citations
```
AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. 
https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset

APS Failure at Scania Trucks Data Set. Kaggle. 
https://www.kaggle.com/datasets/uciml/aps-failure-at-scania-trucks-data-set

SECOM Dataset. UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets/SECOM
```

### Related Work
```
Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. Journal of Machine Learning Research.

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

Goodfellow, I., et al. (2014). Generative Adversarial Networks. Advances in Neural Information Processing Systems.

Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.
```

## License & Contribution Guidelines

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code of Conduct
- Be respectful and inclusive
- Follow Python PEP 8 style guidelines
- Add comprehensive documentation for new features
- Include tests for new functionality
- Ensure reproducibility of research results

---
==

**Note**: This project is for research and educational purposes. Please ensure compliance with local regulations when deploying predictive maintenance systems in real-world industrial applications.
