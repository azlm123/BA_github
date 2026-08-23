# Domain Decomposition HDG ROM & Neural Operators

This repository implements a **Domain Decomposition Hybridizable Discontinuous Galerkin (DD-HDG)** framework accelerated by **Reduced-Order Modeling (ROM)** using **Proper Orthogonal Decomposition (POD)/Singular Value Decomposition (SVD)** and **Deep Neural Operators**.

The repository provides a complete workflow covering:

* POD/SVD basis generation and snapshot compression
* DD-HDG subdomain and interface data preparation
* Hyperparameter studies for neural operators
* Training with two operator decompositions
* Full-domain iterative inference
* Multiple face and corner initialization strategies
* First-layer weight audits
* Input-block sensitivity experiments
* A spatial U-Net benchmark workflow

The main DD-HDG neural-operator framework is available in two variants:

* **6-operator setup:** 2 solution operators and 4 shared internal flux operators
* **10-operator setup:** separate internal and boundary operators for both solution and directional flux prediction

---

## 📁 Repository Structure

```text id="zh97jk"
.
├── Bases/                          # ROM bases, mean vectors, datasets, and analysis results
├── Plots/                          # Training, reconstruction, sensitivity, and audit plots
├── Tests/                          # Additional test scripts and experiments
├── input_block_sensitivity_results/ # Detailed input-block sensitivity CSV results
├── trained_operators/              # Trained models and preprocessing/scaling objects
│
├── dd_hdg_SVD.py                   # POD/SVD basis generation and snapshot compression
├── dd_hdg_trainingdata.py          # Tabular DD-HDG operator dataset preparation
├── dd_hdg_hyperparameterstudy.py   # Hyperparameter grid-search study
│
├── dd_hdg_training6.py             # Training for the 6-operator configuration
├── dd_hdg_training8.py             # Training for the 10-operator configuration
│
├── inference6.py                   # Full-domain inference with 6 operators
├── inference8.py                   # Full-domain inference with 10 operators
│
├── first_layer_audit6.py           # First-layer weight audit for the 6-operator setup
├── first_layer_audit8.py           # First-layer weight audit for the 10-operator setup
│
├── sensitivity_test6.py            # Input-block sensitivity experiment for 6 operators
├── sensitivity_test8.py            # Input-block sensitivity experiment for 10 operators
│
├── unet_model.py                   # Spatial U-Net architecture
├── unet_train.py                   # U-Net benchmark training
├── unet_predict.py                 # U-Net prediction and visualization
└── unet_predict_statistics.py      # U-Net reconstruction and statistical utilities
```

---

# 🚀 Complete Workflow

The repository can be used through the following workflow:

1. **POD/SVD basis generation**
2. **DD-HDG operator dataset preparation**
3. **Optional hyperparameter study**
4. **Neural-operator training**

   * 6-operator setup, or
   * 10-operator setup
5. **Full-domain DD-HDG inference**
6. **Optional first-layer weight audit**
7. **Optional input-block sensitivity analysis**
8. **Optional spatial U-Net benchmark**

The recommended execution order is:

```text id="7uzsmr"
Snapshot Data
     │
     ▼
POD/SVD Compression
     │
     ▼
Tabular DD-HDG Dataset Generation
     │
     ├──────────────────────────────► Optional Hyperparameter Study
     │                                      │
     │                                      ▼
     │                               Hyperparameter Selection
     │
     ▼
Neural-Operator Training
     │
     ├── 6-Operator Setup
     │
     └── 10-Operator Setup
     │
     ▼
Full-Domain Iterative Inference
     │
     ├──────────────────┬───────────────────────┐
     ▼                  ▼                       ▼
First-Layer Audit   Input Sensitivity      Physical Field
                                              Evaluation
     │
     ▼
Optional Spatial U-Net Benchmark
```

---

# 1. POD/SVD Basis Generation

Run `dd_hdg_SVD.py` to construct the reduced-order representation from the available DD-HDG snapshot data.

The script:

* Computes POD/SVD bases for solution and flux quantities.
* Computes the corresponding mean vectors.
* Compresses high-dimensional snapshots into reduced-order coordinates.
* Generates reduced representations for subdomain and interface quantities.
* Evaluates reconstruction errors for different truncation ranks.

### Run

```bash id="s7ab25"
python dd_hdg_SVD.py
```

### Generated Outputs

The generated ROM data are stored in `Bases/`.

The outputs include:

* Dataset-specific `.npz` files containing train, validation, and test data.
* `hdg_rom_bases.npz`, containing the consolidated POD/SVD basis matrices and mean vectors.

SVD truncation plots are stored in `Plots/`.

These plots show the relative reconstruction error as a function of the truncation rank.

> **Note:** Boundary face values that are identically zero and scalar corner quantities do not require the same SVD truncation analysis as high-dimensional solution and flux fields.

---

# 2. DD-HDG Training-Dataset Preparation

Run `dd_hdg_trainingdata.py` to transform the reduced-order DD-HDG data into tabular datasets suitable for neural-operator training.

### Run

```bash id="h4fh68"
python dd_hdg_trainingdata.py
```

### Generated Outputs

The script generates train, validation, and test datasets for:

* **Internal subdomains**
* **Boundary subdomains**

The resulting CSV files are stored in:

```text id="p2ikgq"
Bases/
├── dataset_operator_internal_train.csv
├── dataset_operator_internal_val.csv
├── dataset_operator_internal_test.csv
│
├── dataset_operator_boundary_train.csv
├── dataset_operator_boundary_val.csv
└── dataset_operator_boundary_test.csv
```

The datasets contain the reduced-order input blocks required by the neural operators, including:

* `F_sub` reduced forcing/subdomain features
* `U_face` interface-face information
* `U_corners` corner information
* Boundary-condition flags where applicable

These datasets are used by the training, hyperparameter, inference, and sensitivity-analysis scripts.

---

# 3. Optional Hyperparameter Study

Before training the final neural operators, an optional hyperparameter study can be performed using:

```bash id="wnfwc3"
python dd_hdg_hyperparameterstudy.py
```

The study evaluates different neural-network configurations for the available operator setups.

The study can be run for:

* `training6`
* `training8`

The script evaluates the relevant representative solution and flux operators for each configuration and ranks the tested configurations using both:

* **Test Mean Squared Error (MSE)**
* **Relative reconstruction error**

The relative reconstruction error is computed after mapping the predicted reduced-order coefficients back into the physical field using the corresponding ROM basis.

---

## Hyperparameter Search Space

The current grid search explores the following hyperparameters:

| Hyperparameter            | Explored Values     |
| ------------------------- | ------------------- |
| Hidden-layer architecture | `(64, 32)`          |
|                           | `(64, 128, 64, 32)` |
|                           | `(128, 128, 128)`   |
| Learning rate             | `1e-2`, `1e-3`      |
| Batch size                | `32`, `64`          |
| Activation function       | `SiLU`, `ReLU`      |
| Weight decay              | `5e-4`              |

This gives:

```text id="p4wh4h"
3 architectures
× 2 learning rates
× 2 batch sizes
× 2 activation functions
× 1 weight-decay value
= 24 configurations per operator
```

For each tested configuration, the study records:

* Operator name
* Hidden-layer architecture
* Learning rate
* Batch size
* Activation function
* Test MSE
* Relative reconstruction error

The configuration with the lowest relative reconstruction error is identified for each evaluated operator.

---

## Hyperparameter Study Outputs

Detailed results are saved in `Bases/`:

```text id="op7p50"
Bases/
├── hyperparameter_study_training6.csv
└── hyperparameter_study_training8.csv
```

Summary plots are saved in `Plots/`:

```text id="8b4pxd"
Plots/
├── hyperparameter_study_training6_best_summary.pdf
└── hyperparameter_study_training8_best_summary.pdf
```

The summary plots compare the best relative reconstruction errors obtained during the hyperparameter search.

> **Note:** The hyperparameter study is optional. It is intended for architecture and training-parameter selection before final model training.

---

# 4. Neural-Operator Training

Two operator decompositions are available.

---

## Option A — 6-Operator Setup

Run:

```bash id="rqn9bc"
python dd_hdg_training6.py
```

The 6-operator setup consists of:

### Solution Operators

* 1 internal solution operator
* 1 boundary solution operator

### Flux Operators

* 1 bottom-direction flux operator
* 1 right-direction flux operator
* 1 top-direction flux operator
* 1 left-direction flux operator

The directional flux operators are shared according to the 6-operator decomposition strategy.

This setup provides a more compact neural-operator architecture.

---

## Option B — 10-Operator Setup

Run:

```bash id="4t1n4l"
python dd_hdg_training8.py
```

The 10-operator setup consists of:

### Internal Operators

* 1 internal solution operator
* 1 internal bottom flux operator
* 1 internal right flux operator
* 1 internal top flux operator
* 1 internal left flux operator

### Boundary Operators

* 1 boundary solution operator
* 1 boundary bottom flux operator
* 1 boundary right flux operator
* 1 boundary top flux operator
* 1 boundary left flux operator

This setup explicitly separates internal and boundary operator behavior.

---

## Training Outputs

The trained models and scaling objects are stored in:

```text id="i5p05w"
trained_operators/
├── *_model.pth
├── *_scalers.pkl
├── unet_benchmark_model.pth
└── unet_scalers.pkl
```

Training and evaluation plots are stored in `Plots/`.

---

# 5. Full-Domain DD-HDG Inference

After training, the neural operators are coupled through an iterative full-domain inference procedure.

### 6-Operator Inference

```bash id="hqqc3l"
python inference6.py
```

### 10-Operator Inference

```bash id="2n4cc9"
python inference8.py
```

The inference procedure:

1. Selects a test sample.
2. Loads the trained neural operators and their feature scalers.
3. Initializes interior face and corner quantities.
4. Iteratively updates interface quantities using the predicted directional fluxes.
5. Evaluates the internal and boundary solution operators.
6. Collects the predicted reduced-order solution modes.
7. Reconstructs the complete physical solution using the ROM bases.
8. Computes global physical-field errors.

The inferred subdomain patches are assembled into a global field by averaging contributions at overlapping patch locations.

---

## Inference Initialization Options

The inference scripts support several initialization strategies for interior face and corner quantities.

The initialization is controlled by:

```python id="6pe8d4"
interior_init_mode = "nearest_match"
```

The available options are:

### `nearest_match`

The interface values are initialized from the **globally closest-matching training sample** based on the forcing/subdomain information.

This is useful when a training sample with similar global forcing characteristics provides a physically meaningful initial interface state.

---

### `per_subdomain_match`

Each subdomain receives its initial interface values from the training subdomain with the closest matching `F_sub` features at the **same spatial subdomain location**.

This provides a more localized initialization strategy.

---

### `zeros`

All interior face and corner quantities are initialized to zero.

This provides a simple baseline and allows the convergence behavior to be evaluated without a data-driven initialization.

---

### `constant`

All interior interface quantities are initialized with:

```python id="txex2g"
interior_init_constant = 0.5
```

The constant can be changed by the user.

---

### `random`

Interior face and corner quantities are initialized using independent Gaussian random values.

The random initialization is controlled through:

```python id="3s4w28"
interior_init_random_mean = 0.0
interior_init_random_std = 1.0
interior_init_seed = 42
```

This option can be used to test the robustness of the iterative inference procedure with respect to the initial interface state.

---

## Inference Configuration

Additional important parameters include:

```python id="wfz68b"
sample_idx = 10
run_optimization = True

n_subx, n_suby = 8, 8
```

`run_optimization` controls whether the optimization/update stage is executed after the initial inference state is constructed.

---

## Inference Evaluation

The inference scripts evaluate the reconstructed physical field using the global relative error:

```text id="zzx57q"
||U_pred - U_true|| / ||U_true||
```

The inference workflow also supports diagnostics of:

* Predicted versus target ROM modes
* Per-mode relative errors
* Spatial error maps
* Reconstructed physical solution
* Absolute physical-field error

When `nearest_match` initialization is used, the inference scripts additionally compare:

* The nearest training sample against the true test solution
* The final prediction against the nearest training sample

This helps distinguish the quality of the data-driven initialization from the improvement achieved by the iterative neural-operator inference.

---

# 6. First-Layer Weight Audit

The repository includes first-layer audits for both operator configurations.

### 6-Operator Audit

```bash id="o5e0wc"
python first_layer_audit6.py
```

### 10-Operator Audit

```bash id="u1xtzq"
python first_layer_audit8.py
```

The purpose of the first-layer audit is to examine how strongly each trained operator weights the two main groups of input features:

* **`F_sub` features**
* **Trace and corner features**, including `U_face`, `U_corners`, and applicable boundary flags

The first linear-layer weight matrix is divided into:

```text id="u2zqv3"
W^(1) = [W_F | W_U]
```

where:

* `W_F` corresponds to the `F_sub` feature block
* `W_U` corresponds to the interface, corner, and boundary-information block

---

## Audit Metrics

For each trained model, the scripts compute:

### Frobenius Norm

The total weight magnitude associated with each feature block:

```text id="fjz8gq"
||W_F||_F
```

and:

```text id="6iq9ye"
||W_U||_F
```

### Mean Column L2 Norm

The average weight magnitude per input feature:

```text id="zv8qsn"
mean_j ||W_F[:, j]||_2
```

and:

```text id="h7q30m"
mean_j ||W_U[:, j]||_2
```

This is particularly useful because the input blocks can have different numbers of features.

### Per-Feature Weight Ratio

The primary audit ratio is:

```text id="36bymt"
mean_column_norm(F_sub)
-----------------------
mean_column_norm(U_features)
```

A ratio:

* **greater than 1** suggests stronger first-layer weighting of `F_sub` features
* **less than 1** suggests stronger first-layer weighting of trace and corner features
* **close to 1** indicates more balanced first-layer weighting

> **Important:** This audit measures learned first-layer weight structure. It does not directly prove functional input sensitivity. For that reason, the repository also includes explicit input-block sensitivity experiments.

---

## First-Layer Audit Outputs

For the 6-operator setup, the detailed table is saved in:

```text id="yd1xha"
Bases/first_layer_weight_audit6.csv
```

The plots are saved in:

```text id="7g0dp9"
Plots/
├── first_layer_weight_audit_mean_col6.pdf
└── first_layer_weight_ratio6.pdf
```

For the 10-operator setup, the plots are saved in:

```text id="q8mv0l"
Plots/
├── first_layer_weight_audit_mean_col8.pdf
└── first_layer_weight_ratio8.pdf
```

The plots show:

1. Mean first-layer column norms for the two input blocks.
2. The relative `F_sub`-to-interface sensitivity ratio based on learned first-layer weights.

---

# 7. Input-Block Sensitivity Analysis

The first-layer audit examines the learned network weights, whereas the input-block sensitivity tests directly examine **how the model output changes when one block of input information is replaced**.

Separate scripts are provided for both architectures.

### 6-Operator Setup

```bash id="psiwbu"
python sensitivity_test6.py
```

### 10-Operator Setup

```bash id="pmrjvj"
python sensitivity_test8.py
```

---

## Sensitivity Experiment

Each experiment compares two controlled input modifications.

### Experiment A — `F_sub` Swap

The trace and corner information is kept fixed:

```text id="hv7sb3"
U_face + U_corners + boundary flags = fixed
```

while the `F_sub` block is replaced using a donor sample:

```text id="qtd4yp"
F_sub(base sample) → F_sub(donor sample)
```

This measures how strongly the operator output responds to changes in the forcing/subdomain-information block.

---

### Experiment B — `U` Swap

The `F_sub` block is kept fixed:

```text id="5a0dph"
F_sub = fixed
```

while the interface-related block is replaced:

```text id="cgy9o1"
U_face + U_corners + boundary flags
```

This measures how strongly the operator output responds to changes in interface and corner information.

---

# Input-Block Sensitivity Results: How to Interpret Them

For every base sample and donor sample, the scripts compute three quantities.

---

## 1. Relative Input Change

The size of the applied perturbation is:

```text id="krrghj"
input_rel_change
=
||X_donor,block - X_base,block||
--------------------------------
||X_base,block||
```

This measures how strongly the selected input block was changed.

---

## 2. Relative Output-Coefficient Change

The resulting change in the predicted ROM coefficients is:

```text id="jpwvv8"
output_coeff_rel_change
=
||y_modified - y_baseline||
---------------------------
||y_baseline||
```

This is the direct response of the neural operator in reduced-order output space.

A larger value means that swapping the selected input block produces a larger change in the predicted reduced-order coefficients.

---

## 3. Relative Physical-Field Change

When the corresponding ROM basis is available, the predicted outputs are reconstructed into physical fields.

The sensitivity is then also evaluated as:

```text id="yz56jm"
output_field_rel_change
=
||U_modified - U_baseline||
---------------------------
||U_baseline||
```

This provides an interpretable measure of how much the actual reconstructed physical field changes.

---

## Normalized Input-Block Sensitivity

The experiments also compute:

```text id="18q8si"
sensitivity_ratio
=
output_coeff_rel_change
-----------------------
input_rel_change
```

This is important because the raw swaps can have different input magnitudes.

The normalized sensitivity measures:

> **How much the operator output changes per unit of relative change in the selected input block.**

Therefore, when comparing the two input blocks:

* A larger normalized `F_sub` sensitivity means the model responds more strongly to changes in `F_sub`.
* A larger normalized `U` sensitivity means the model responds more strongly to changes in face/corner/interface information.
* Similar sensitivities indicate that both input blocks contribute comparably to the model output.

This distinction is more informative than comparing only the absolute output changes because it accounts for the magnitude of the perturbation introduced into each block.

---

## Sensitivity Result Plots

The scripts generate comparison plots showing:

### Raw Output Sensitivity

```text id="cnz2ra"
Mean relative change in predicted output
```

for:

* `F_sub` swap with interface information fixed
* Interface/corner swap with `F_sub` fixed

### Normalized Sensitivity

```text id="hmgmhi"
Output relative change
----------------------
Input relative change
```

This indicates the output response per unit of relative input perturbation.

For the 6-operator configuration, plots are saved with names including:

```text id="mxvptw"
Plots/
├── input_block_sensitivity_comparison6.pdf
└── input_block_sensitivity_normalized6.pdf
```

The detailed experiment results are stored in:

```text id="1zv8yd"
input_block_sensitivity_results/
```

with separate CSV files for each model and swapped input block.

The 10-operator sensitivity script performs the corresponding analysis for the separate internal and boundary operator decomposition.

---

# 8. U-Net Benchmark Workflow

In addition to the DD-HDG operator-based models, the repository contains a spatial **U-Net benchmark**.

The U-Net is used as an alternative model that directly learns the mapping:

```text id="anljvr"
Spatial grid of F_sub ROM modes
            │
            ▼
         U-Net
            │
            ▼
Spatial grid of U_sub ROM modes
```

Rather than predicting each subdomain through separately coupled neural operators, the U-Net processes the complete `8 × 8` subdomain grid simultaneously.

---

## U-Net Data Representation

The internal and boundary datasets are combined.

For each sample:

* `F_sub` reduced modes become the input channels.
* `U_sub` reduced modes become the output channels.
* The subdomains are arranged on an `8 × 8` spatial grid.
* Each sample therefore becomes a tensor with the form:

```text id="tlbkrd"
Input:
[number of F_sub modes, 8, 8]

Output:
[number of U_sub modes, 8, 8]
```

The training data are normalized using per-channel standard scaling.

The scaling statistics are saved in:

```text id="g3cyon"
trained_operators/unet_scalers.pkl
```

---

## U-Net Training

Run:

```bash id="imqv72"
python unet_train.py
```

The U-Net training workflow uses:

* Batch size: `16`
* Optimizer: `AdamW`
* Learning rate: `1e-3`
* Weight decay: `1e-4`
* Loss function: Mean Squared Error
* Learning-rate scheduler: Cosine Annealing
* Training epochs: `300`

The trained model is saved as:

```text id="yrc3t8"
trained_operators/unet_benchmark_model.pth
```

---

## U-Net Prediction

Run:

```bash id="0n4goc"
python unet_predict.py
```

The prediction workflow:

1. Selects a test sample.
2. Reconstructs the spatial `F_sub` input grid.
3. Applies the saved input scaling.
4. Loads the trained U-Net.
5. Predicts the spatial grid of reduced-order solution modes.
6. Applies the inverse output scaling.
7. Converts the spatial output back into per-subdomain ROM coordinates.
8. Reconstructs the complete physical field using the ROM bases.
9. Computes the global physical-field relative error.

The U-Net prediction script displays:

* The true physical solution
* The U-Net prediction
* The absolute error field

The reported relative error is:

```text id="8yc12v"
||U_pred - U_true|| / ||U_true||
```

This provides a direct benchmark against the DD-HDG neural-operator approach.

---

# 📊 Analysis Workflow Summary

After training either the 6- or 10-operator setup, the repository supports two complementary analyses.

```text id="evqlb0"
Trained Neural Operators
          │
          ├───────────────────────────────┐
          ▼                               ▼
First-Layer Weight Audit         Input-Block Sensitivity Test
          │                               │
          ▼                               ▼
What did the network learn?      What does the network actually
                                 respond to when inputs change?
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Interpretable Input Analysis
```

### First-Layer Audit

Answers:

> **Which feature groups receive stronger learned weights in the first network layer?**

### Input-Block Sensitivity

Answers:

> **Which feature groups actually cause larger output changes when they are perturbed?**

Using both analyses provides a more complete interpretation of the trained operators.

---

# 📦 Requirements

The repository requires:

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# 🔄 Recommended End-to-End Execution

## Standard DD-HDG Workflow

```bash id="cugybi"
# Step 1 — Generate POD/SVD bases and compress snapshots
python dd_hdg_SVD.py

# Step 2 — Generate tabular DD-HDG datasets
python dd_hdg_trainingdata.py

# Step 3 — Optional hyperparameter study
python dd_hdg_hyperparameterstudy.py

# Step 4 — Train one operator configuration

# 6-operator setup
python dd_hdg_training6.py

# OR 10-operator setup
python dd_hdg_training8.py

# Step 5 — Run full-domain inference

# For the 6-operator setup
python inference6.py

# OR for the 10-operator setup
python inference8.py
```

---

## Optional Model Analysis

After training:

```bash id="usq9a3"
# First-layer audit for the 6-operator setup
python first_layer_audit6.py

# First-layer audit for the 10-operator setup
python first_layer_audit8.py

# Input-block sensitivity for the 6-operator setup
python sensitivity_test6.py

# Input-block sensitivity for the 10-operator setup
python sensitivity_test8.py
```

---

## Optional U-Net Benchmark

```bash id="29tt1c"
# Train the spatial U-Net benchmark
python unet_train.py

# Run prediction and physical-field reconstruction
python unet_predict.py
```

---

# Notes

* The **hyperparameter study**, **first-layer audits**, **input-block sensitivity tests**, and **U-Net benchmark** are optional analysis or comparison workflows.
* The 6- and 10-operator configurations should be used consistently between training and inference.
* The inference initialization strategy can significantly affect the starting point of the iterative coupling procedure and can be changed through `interior_init_mode`.
* First-layer weight analysis and functional sensitivity analysis measure different aspects of the trained models and are therefore best interpreted together.
