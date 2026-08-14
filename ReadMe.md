# Domain Decomposition HDG ROM & Neural Operators

This repository implements a **Domain Decomposition Hybridizable Discontinuous Galerkin (DD-HDG)** framework accelerated by **Reduced-Order Modeling (ROM)** using **Proper Orthogonal Decomposition (POD)/Singular Value Decomposition (SVD)** and **Deep Neural Operators**.

The workflow covers the complete pipeline, from snapshot compression and reduced-order basis construction to dataset preparation, neural-operator training, and full-domain inference across multiple subdomains.

## 📁 Repository Structure

```text
.
├── Bases/                  # SVD bases, mean vectors (.npz), and tabular datasets (.csv)
├── Plots/                  # SVD truncation plots, training-loss curves, and evaluation charts (.pdf)
├── trained_models/         # Trained neural-operator weights (.pth) and feature scalers (.pkl)
│
├── dd_hdg_SVD.py           # Step 1: SVD basis construction and rank truncation
├── dd_hdg_trainingdata.py  # Step 2: Training-dataset generation for subelements
├── dd_hdg_training6.py     # Step 3a: Neural-operator training (6-operator setup)
├── dd_hdg_training8.py     # Step 3b: Neural-operator training (10-operator setup)
├── dd_hdg_inference6.py    # Step 4a: Full-domain inference using 6 operators
└── dd_hdg_inference8.py    # Step 4b: Full-domain inference using 10 operators
```

---

## 🚀 Workflow

The complete pipeline consists of four steps:

1. **POD/SVD basis generation and snapshot compression**
2. **Tabular training-dataset preparation**
3. **Neural-operator training**
4. **Full-domain inference**

---

## 1. POD/SVD Basis Generation

Run `dd_hdg_SVD.py` to construct the reduced-order representation from the available snapshot data.

The script:

* Computes SVD/POD bases for the relevant solution and flux fields.
* Computes the corresponding mean vectors.
* Compresses the solution and flux snapshots.
* Generates reduced-order coordinates for the interface-skeleton features.
* Evaluates reconstruction errors for different truncation ranks.

### Run

```bash
python dd_hdg_SVD.py
```

### Generated outputs

The script stores the resulting data in `Bases/`.

This includes:

* **6 dataset-specific `.npz` files** containing the train, validation, and test data for internal and boundary subdomains.
* `hdg_rom_bases.npz`, a consolidated file containing the computed SVD basis matrices and mean vectors.

SVD truncation plots are stored in `Plots/`. These plots show the **relative reconstruction error on the test data as a function of the truncation rank (k)**.

> **Note:** `U_face,bnd` and `U_corners` do not have SVD truncation plots because the boundary-face values are identically zero and the corner quantities are scalar-valued.

---

## 2. Training-Dataset Preparation

Run `dd_hdg_trainingdata.py` to transform the reduced-order data into tabular datasets suitable for neural-operator training.

### Run

```bash
python dd_hdg_trainingdata.py
```

### Generated outputs

Six CSV files are generated in `Bases/`:

* Training, validation, and test datasets for **internal elements**
* Training, validation, and test datasets for **boundary elements**

These datasets provide the input-output pairs required by the neural operators in the next stage.

---

## 3. Neural-Operator Training

Two training configurations are available.

### Option A — 6-Operator Setup

Run:

```bash
python dd_hdg_training6.py
```

This configuration trains **6 neural operators**:

#### Solution operators

* 1 internal solution operator
* 1 boundary solution operator

#### Flux operators

* 1 top flux operator
* 1 bottom flux operator
* 1 left flux operator
* 1 right flux operator

This configuration provides a more compact operator decomposition.

---

### Option B — 10-Operator Setup

Run:

```bash
python dd_hdg_training8.py
```

This configuration trains **10 neural operators**, separating the internal and boundary predictions for both solution and flux quantities.

#### Internal operators

* 1 solution operator
* 1 top flux operator
* 1 bottom flux operator
* 1 left flux operator
* 1 right flux operator

#### Boundary operators

* 1 solution operator
* 1 top flux operator
* 1 bottom flux operator
* 1 left flux operator
* 1 right flux operator

This configuration provides a more granular decomposition between internal and boundary elements.

### Generated outputs

Training results are stored in `Plots/`, including:

* Training and validation loss curves for each operator.
* Test **Mean Squared Error (MSE)** results.
* **Relative reconstruction error** for each operator.

The trained models and preprocessing objects are stored in `trained_models/`:

* `.pth` files containing neural-network parameters/weights.
* `.pkl` files containing the feature scalers used by each operator.

---

## 4. Full-Domain Inference

After training, run the inference script corresponding to the selected operator configuration.

### 6-Operator configuration

```bash
python dd_hdg_inference6.py
```

### 10-Operator configuration

```bash
python dd_hdg_inference8.py
```

The inference stage evaluates the **coupled DD-HDG solution across the complete computational domain**, using the trained neural operators for the individual subdomains.

---

## 📦 Requirements

The project requires:

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 🔄 End-to-End Execution

For a complete run, execute the scripts in the following order:

```bash
# Step 1 — Construct reduced-order bases
python dd_hdg_SVD.py

# Step 2 — Generate tabular training datasets
python dd_hdg_trainingdata.py

# Step 3 — Train neural operators
python dd_hdg_training6.py
# or
python dd_hdg_training8.py

# Step 4 — Perform full-domain inference
python dd_hdg_inference6.py
# or
python dd_hdg_inference8.py
```

Choose the **6-operator** or **10-operator** configuration consistently between the training and inference stages.
