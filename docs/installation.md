# Installation

[Back to the quick start](../README.md#get-started)


Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and open a Conda-enabled terminal in the repository root. If you prefer Git to downloading the code ZIP:

```text
git clone https://github.com/Jaekang-Song/CNFET_TBDT.git
cd CNFET_TBDT
```

### Create the environment

First, extract the supplied code archive. To obtain the code from GitHub instead, open the [repository](https://github.com/Jaekang-Song/CNFET_TBDT) and select **Code > Download ZIP**, or clone it as shown above. The archive includes all code and model files needed for these steps. [environment.yml](../environment.yml) is included beside `README.md`; no separate download of that file is needed. It installs Python, the scientific-computing packages, Jupyter, and PyTorch 2.4.1 with the CUDA 11.8 runtime on Windows/Linux 64-bit.

Open Anaconda Prompt on Windows, or a Conda-enabled terminal on Linux. Use `cd` to enter the local folder containing `README.md` and `environment.yml`. Replace the example path below with your actual extracted or cloned repository folder:

```text
cd "path/to/CNFET_TBDT"
conda env create --file environment.yml
conda activate cnfet_tbdt
```

### CPU and GPU use

**A GPU is optional but highly recommended for training.** The environment already includes the CUDA runtime needed by PyTorch; a separate CUDA Toolkit installation is not required for these notebooks. GPU execution requires a compatible NVIDIA GPU and driver. On a cluster, start Jupyter in an allocation that includes a GPU.

Training and inference automatically use CUDA when available and otherwise use the CPU. See the [official PyTorch 2.4.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v241) for the supported builds.

### Check the installation

```text
python -c "import sys, torch, numpy, pandas, sklearn, matplotlib, seaborn, natsort, tqdm, shap, IPython, scipy; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Start Jupyter

From the activated environment and repository root:

```text
python -m ipykernel install --user --name cnfet_tbdt --display-name "Python (CNFET TBDT)"
python -m jupyter lab
```

For the Jupyter Notebook interface, run `python -m jupyter notebook` instead of `python -m jupyter lab`. Both interfaces open the same `.ipynb` files.

Select the **Python (CNFET TBDT)** kernel in each notebook.

To check the selected notebook kernel, run:

```python
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Compute device:", "cuda" if torch.cuda.is_available() else "cpu")
```

The CUDA runtime should be `11.8`. If you expect GPU execution but see `cpu`, check the selected kernel, NVIDIA driver, and GPU allocation. After changing the environment or kernel, restart the kernel and run the notebook from the beginning.

