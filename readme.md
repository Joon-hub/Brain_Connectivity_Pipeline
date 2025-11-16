## Brain Connectivity Classification

A machine learning framework for identifying brain regions from functional connectivity patterns and detecting functional reorganization during cognitive tasks.

## Project Overview

This project develops a novel machine learning approach to study brain organization by:

1. **Training classifiers on resting-state fMRI data** to learn normal connectivity patterns for each of 232 brain regions
2. **Applying these classifiers to task fMRI data** (Gender Stroop task) 
3. **Analyzing misclassification patterns** to reveal functional reorganization and task-specific neural engagement

### Key Innovation
Classification errors are not random failures—they reveal meaningful brain organization. Regions that are misclassified together show functional similarity, and systematic errors during tasks indicate altered connectivity patterns specific to cognitive demands.

### Datasets
- **PIOP-2 (Training):** Resting-state fMRI, 224 subjects, Amsterdam Open MRI Collection
- **PIOP-1 (Testing):** Gender Stroop task fMRI, 200 subjects, Amsterdam Open MRI Collection
- **Brain Parcellation:** 232 regions total
  - **Cortical:** 200 regions (Schaefer atlas, 17 networks)
  - **Subcortical:** 32 regions (Tian atlas)

## Current Performance
### Classification Accuracy

- **Validation Accuracy:** 73-84% (depending on diagonal imputation strategy)
- **Baseline (Random Chance):** 0.43%
- **Improvement over Chance:** 168-196x
- **Task Data Accuracy:** 70-82%

### Best Configuration

```yaml
Model: Multinomial Logistic Regression
Diagonal Strategy: sample_from_matrix
Regularization (C): 0.01
Cross-Validation: 5-fold (subject-wise)
Random Seed: 123
```

## Scientific Findings
### Hemisphere Confusion Pattern

The classifier successfully identifies functional network types but struggles with left vs. right hemisphere assignment. This is **neuroscientifically meaningful** because mirror regions have nearly identical connectivity patterns, demonstrating the brain's bilateral symmetry.

### Task-Specific Reorganization

Analysis reveals systematic connectivity changes during the Gender Stroop task:

- **Motor regions:** High misclassification during button-press responses
- **Attention networks:** Increased activation in salience and dorsal attention areas
- **Cognitive control:** FrontoParietal network shows altered connectivity patterns

## Project Structure

```
brain_connectivity_classifier/
├── src/                          # Core modules 
│   ├── data.py                   # Data loading utilities
│   ├── features.py               # Connectivity preprocessing
│   ├── model.py                  # Classifier implementation
│   ├── evaluate.py               # Performance metrics
│   ├── visualize.py              # Publication-quality plots
│   ├── utils.py                  # Helper functions
│   └── __init__.py
│
├── AdvanceAnalysis/              # Advanced analysis scripts
│   ├── 01_atlas_performace_analysis.py
│   ├── 02_atlas_comparison.py
│   └── 03_connectivity_analysis.py
│
├── data/
│   ├── raw/                      # Original PIOP datasets
│   └── processed/                # Preprocessed features & predictions
│
├── reports/
│   ├── tables/                   
│   │   ├── basic_analysis/
│   │   ├── atlas_analysis/
│   │   ├── confusion_matrix/
│   │   └── connectivity_analysis/
│   └── figures/                  
│       ├── basic_analysis/
│       ├── atlas_analysis/
│       └── connectivity_analysis/
│
├── models/                       # Trained classifier (.joblib)
├── logs/                         # HTCondor execution logs
├── sh_files/                     # Bash scripts for cluster
│
├── run.py                        # Main pipeline script
├── config.yaml                   # Central configuration
├── run_brain_pipeline.sub        # HTCondor submit file
└── README.md
```

## Quick Start
#### Installation

```bash
# Clone repository
cd /home/sjoon/projects/brain_connectivity_classifier

# Activate virtual environment
source masterthesis_venv2/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt 
```

#### Running the Pipeline

```bash
# Full pipeline with default config
python run.py --config config.yaml

# Quick test with sample data (10 subjects)
python run.py --sample

# Custom configuration
python run.py --diagonal region_mean --C 0.01
```

#### HTCondor Cluster Execution

```bash
# Submit job to cluster
condor_submit run_brain_pipeline.sub

# Monitor job
condor_q

# View logs
tail -f logs/*.out
```
## Pipeline Workflow

### Step 1: Load Training Data (PIOP-2 Resting State)

- Loads 224 subjects with 26,796 connections per subject
- Extracts 232 brain regions from connectivity matrix
- Validates data integrity

### Step 2: Train Classifier (Leak-Free Cross-Validation)

**CRITICAL:** All preprocessing happens **inside** each CV fold to prevent data leakage:

```python
# CORRECT: Statistics computed only on training fold
for train_idx, val_idx in kfold.split(subjects):
    X_train_fold = X[train_idx]
    scaler = StandardScaler().fit(X_train_fold)  # Fit on train only
    X_train_scaled = scaler.transform(X_train_fold)
    X_val_scaled = scaler.transform(X[val_idx])
```

**Cross-Validation Settings:**
- 5-fold GroupKFold (subject-wise splitting)
- Prevents data leakage across subjects
- Reports both training and validation accuracy

### Step 3: Diagonal Imputation Strategies

Connectivity matrices have diagonal elements (self-connections) that must be handled:

1. **`random`**: Random values from normal distribution
2. **`region_mean`**: Mean connectivity to the same region across subjects  
3. **`sample_from_matrix`**: Sample from the region's existing connectivity distribution  **BEST**
4. **`network_mean`**: Mean connectivity to regions in same functional network (neuroscientifically grounded)

### Step 4: Evaluate on Training Data

- Calculate error maps showing misclassification rates per region
- Generate confusion matrices (raw counts and normalized)
- Save predictions to CSV

### Step 5: Apply to Task Data (PIOP-1 Gender Stroop)

- Load 200 subjects from task dataset
- Apply trained classifier without retraining
- Compare rest vs. task error patterns
- Identify task-specific connectivity changes

### Step 6: Generate Visualizations

Creates publication-quality figures:

- Error maps (rest and task)
- Rest vs. task comparison plots
- Confusion matrices at multiple hierarchical levels
- Network-level connectivity analysis

## Configuration

### Main Configuration File: `config.yaml`

```yaml
# Data paths
data:
  piop2_file: "data/raw/PIOP2_restingstate.csv"
  piop1_file: "data/raw/PIOP1_gstroop.csv"

# Preprocessing
preprocessing:
  diagonal_strategy: "random"  # values to (1 to -1)
  scaling_method: "standard"

# Model
model:
  classifier: "logistic"
  C: 0.01                      
  max_iter: 1000
  multi_class: "multinomial"

# Cross-validation
cv:
  n_folds: 5
  strategy: "subject_wise"     # GroupKFold by subject

# Output directories
output_dirs:
  models: "models"
  processed: "data/processed"
  tables: "reports/tables/basic_analysis"
  figures: "reports/figures/basic_analysis"

# Reproducibility
random_seed: 123
```
## Advanced Analysis Scripts
### 1. Atlas Performance Analysis

Compares classification performance across different atlas resolutions:

```bash
python AdvanceAnalysis/01_atlas_performace_analysis.py --config config.yaml
```

**Analyzes:**
- Schaefer N7 (7 cortical networks)
- Schaefer N17 (17 cortical networks)  
- Tian Scale I (8 subcortical regions)
- Tian Scale II (16 subcortical regions)
- Combined N7 + Tian I

### 2. Atlas Comparison

Direct statistical comparison between atlas resolutions:

```bash
python AdvanceAnalysis/02_atlas_comparison.py --config config.yaml
```

**Outputs:**
- Accuracy comparison tables
- Statistical significance tests
- Visualization of performance differences

### 3. Connectivity Analysis

Network-level connectivity patterns and task-induced changes:

```bash
python AdvanceAnalysis/03_connectivity_analysis.py --config config.yaml
```

**Analyzes:**
- Inter-network connectivity matrices
- Task vs. rest connectivity differences
- Top changing connections
- Cortical-subcortical integration

---


### Neuroscientific Insights

1. **Symmetric matrix handling:** Connectivity matrices are symmetric; only upper triangle needed
2. **Self-connections:** Diagonal elements must be removed or imputed
3. **Network-aware preprocessing:** Imputation strategies should respect functional networks
4. **Hemisphere symmetry:** Mirror regions have nearly identical connectivity patterns

### Long-Term Vision (Clinical Translation)

**Four-Phase Roadmap:**

1. **Phase 1 - Method Validation:** Extend to HCP dataset, validate robustness
2. **Phase 2 - Disease Datasets:** Apply to Alzheimer's, stroke, psychosis cohorts
3. **Phase 3 - Biomarker Development:** Identify disease-specific connectivity signatures
4. **Phase 4 - Clinical Tool:** Deploy as diagnostic/prognostic tool