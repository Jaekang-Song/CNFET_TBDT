# CNFET TBDT

Code, pretrained models, and analysis notebooks for carbon nanotube field-effect transistor (CNFET) gas sensing using transient and bias-dependent tomography (TBDT).

**Four pretrained models are included. You can evaluate them without training.**

## Get started

### 1. Download the files

Download this repository using **Code → Download ZIP**, then extract it. If you received a code archive with the paper, extract that archive instead.

Obtain `TBDT_Dataset.zip` using the **data availability statement in the paper**. The Figshare link will be added here upon publication. Extract it into the code folder so that `README.md` and `TBDT_Dataset/` are next to each other:

```text
CNFET_TBDT/
├── README.md
├── environment.yml
├── ML_test.ipynb
└── TBDT_Dataset/
    ├── 0_chemical_gas/
    ├── 1_solvent/
    └── 2_bacteria/
```

Keep both `training/` and `heldout_test/` inside each dataset folder. Both are required for data loading, including when evaluating a pretrained model.

### 2. Install and open Jupyter

Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Open Anaconda Prompt on Windows, or a Conda-enabled terminal on Linux, and enter the folder containing `README.md`:

```text
cd "path/to/CNFET_TBDT"
conda env create --file environment.yml
conda activate cnfet_tbdt
python -m ipykernel install --user --name cnfet_tbdt --display-name "Python (CNFET TBDT)"
python -m jupyter lab
```

Replace the example path with your actual folder. Select the **Python (CNFET TBDT)** kernel in Jupyter. The environment includes PyTorch and the CUDA 11.8 runtime for Windows/Linux 64-bit. The code uses a compatible NVIDIA GPU when available and otherwise uses the CPU. **A GPU is optional but highly recommended for training.** [Installation details](docs/installation.md)

In a notebook cell, run `import torch; print(torch.cuda.is_available())`. `True` means the selected kernel can use the GPU; `False` means it will use the CPU. GPU use requires a compatible NVIDIA driver and, on a cluster, a GPU allocation.

### 3. Evaluate a model

Open [ML_test.ipynb](ML_test.ipynb). The default evaluates the chemical dry model:

```python
CONFIG_NAME = "Config_TBDT_Chem_dry_pretrained"
RUN_FOLDER_NAME = "pretrained"
```

Run all cells. The notebook prints held-out accuracy, displays the confusion matrix, and saves it under `figures/confusion_matrices/`.

To evaluate another model, change only `CONFIG_NAME`:

| Task | CONFIG_NAME | Held-out accuracy |
| --- | --- | ---: |
| Chemical dry | `Config_TBDT_Chem_dry_pretrained` | 96.6165% |
| Chemical humid | `Config_TBDT_Chem_humid_pretrained` | 93.6476% |
| Solvent vapors | `Config_TBDT_Solvent_pretrained` | 95.9766% |
| Microbial headspace | `Config_TBDT_Bacteria_pretrained` | 97.4941% |

## Other notebooks

| Notebook | Purpose |
| --- | --- |
| [ML_training.ipynb](ML_training.ipynb) | Train a new model using one of the four task configs. |
| [Analysis.ipynb](Analysis.ipynb) | Compute dSNR and MI for the chemical-humidity experiment and display the saved SHAP results. |
| [Single_Vg_training.ipynb](Single_Vg_training.ipynb) | View the saved single-gate-voltage results with `RUN_TRAINING = False`; no dataset is needed for this mode. |

For analysis, keep `RUN_FOLDER_NAME = "pretrained"`.

## Further details

- [Dataset layout and filename examples](docs/dataset.md)
- [Analysis filtering and exact MI coordinates](docs/analysis_protocol.md)
- [Single-Vg parameter grid and saved results](docs/single_vg_training.md)

To check downloaded data and model files, run `python scripts/verify_files.py` from the code folder.
