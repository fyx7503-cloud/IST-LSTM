# IST-LSTM

PyTorch implementation of **IST-LSTM**: a spatio-temporal heterogeneous feature decoupling method for traffic flow prediction.

## Project Structure

```
IST-LSTM/
├── configs/              # Configuration files
├── data/                 # Datasets
│   ├── hangzhou/         # Hangzhou dataset
│   ├── nanjing/          # Nanjing Metro AFC dataset
│   └── shanghai/         # Shanghai dataset
├── models/               # Core model code
│   ├── ist_lstm_model.py # IST-LSTM network architecture
│   ├── ist_lstm_engine.py# Training / evaluation engine
│   └── ist_lstm_util.py  # Data loading and utility functions
├── scripts/              # Data processing scripts
│   ├── mdb_to_csv.py     # MDB to CSV converter
│   ├── physicgraph.py    # Physical graph construction
│   └── process_njmetro_afc.py  # Nanjing Metro AFC data processing
├── tools/                # Visualization and auxiliary tools
│   ├── plotline.py       # Line chart plotting
│   ├── zhuzhuangtu.py    # Bar chart plotting
│   ├── 创建物理图.py      # Physical network graph plotting
│   └── ...
├── train.py              # Training entry point
└── README.md
```

## Environment Setup

### 1. Install Python

Python 3.8 is recommended (the project's `__pycache__` is `cpython-38`).

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy pandas matplotlib seaborn scipy networkx pyodbc
```

> If you have an NVIDIA GPU with CUDA installed, replace `torch` with the corresponding CUDA version, e.g., `torch==1.9.0+cu111`.

## Data Preparation and Processing

### Full Dataset Download

The raw Nanjing Metro AFC dataset (`4.2 NJAFC.rar`, ~6.9GB) is not hosted on GitHub due to its size. Please download it from Baidu Netdisk:

- **Link**: https://pan.baidu.com/s/1csGXgEbPAnqrUb_5IbSmpA?pwd=7890
- **Extraction Code**: `7890`
- **File Name**: `4.2 NJAFC.rar`

After downloading, extract the archive and place the raw data file (e.g., `0916.mdb`) in the project root directory or any path you specify, then preprocess it using `scripts/mdb_to_csv.py`.

### Datasets Already in the Repository

The repository already contains partially preprocessed data:

- `data/hangzhou/`: complete data (train / val / test / graph)
- `data/shanghai/`: all data except `train.pkl` (`train.pkl` is ~144MB, exceeding GitHub's single-file limit, and needs to be prepared separately)
- `data/nanjing/`: only contains `graph_nj_conn.pkl`; the complete data needs to be generated from `0916.mdb`

### Nanjing Metro AFC Data Processing Pipeline

#### Step 1: Convert MDB to CSV (if needed)

If the raw data is in Microsoft Access `.mdb` format, you can convert it to CSV first:

```bash
python scripts/mdb_to_csv.py 0916.mdb --output-dir ./csv_output
```

This script will automatically detect and try the following strategies:
1. `pyodbc` (recommended on Windows)
2. `mdbtools` (recommended on Linux / macOS)
3. `win32com` (Windows COM interface)

#### Step 2: Generate Model-Ready Data

```bash
python scripts/process_njmetro_afc.py --input 0916.mdb --output data/nanjing --interval 5
```

Main arguments:

| Argument | Description | Default |
|---|---|---|
| `--input` | Input file path (`.mdb` or `.csv`) | `0916.mdb` |
| `--output` | Output directory | `njmetro_data` |
| `--interval` | Time aggregation window in minutes, options: 5/15/30 | `5` |
| `--num_nodes` | Number of stations, automatically inferred by default | Auto |

After processing, the following files will be generated under `data/nanjing/`:

```
data/nanjing/
├── train.pkl
├── val.pkl
├── test.pkl
└── graph_nj_*.pkl
```

### Data File Description

| File | Description |
|---|---|
| `train.pkl` / `val.pkl` / `test.pkl` | Training, validation, and test sets |
| `graph_*_conn.pkl` | Connectivity adjacency matrix |
| `graph_*_cor.pkl` | Correlation adjacency matrix |
| `graph_*_sml.pkl` | Similarity adjacency matrix |

## Training

### Hangzhou Dataset

```bash
python train.py --data hangzhou --device cuda:0
```

### Shanghai Dataset

> Make sure `data/shanghai/train.pkl` is placed in the corresponding directory first.

```bash
python train.py --data shanghai --device cuda:0
```

### Nanjing Dataset

> Make sure `data/nanjing/train.pkl` has been generated via `scripts/process_njmetro_afc.py` first.

```bash
python train.py --data nanjing --device cuda:0
```

### Common Arguments

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset name: `hangzhou` / `shanghai` / `nanjing` | `hangzhou` |
| `--device` | Training device: `cuda:0` / `cpu` | `cuda:0` |
| `--epochs` | Number of training epochs | `200` |
| `--batch_size` | Batch size | `64` |
| `--learning_rate` | Learning rate | `0.001` |
| `--hidden_dim` | Hidden dimension | `64` |
| `--num_layers_main` | Number of main-branch LSTM layers | `3` |
| `--num_layers_sub` | Number of sub-branch IST-LSTM layers | `2` |
| `--num_heads` | Number of attention heads | `4` |
| `--window_size` | Swin Transformer window size | `2` |
| `--seq_length` | Input time steps | `4` |
| `--pred_steps` | Prediction time steps | `4` |
| `--use_dcg` | Whether to use DCG graph structure | Disabled |
| `--save` | Model save path | `./checkpoints/ist_lstm` |

### CPU Training

```bash
python train.py --data hangzhou --device cpu
```

## Testing

Currently, `train.py` will automatically evaluate the best model on the test set after training, outputting MAE / MAPE / RMSE for each prediction horizon and the average:

```text
Evaluate best model on test data for horizon 1, Test MAE: x.xxxx, Test MAPE: x.xxxx, Test RMSE: x.xxxx
...
On average over 4 horizons, Test MAE: x.xxxx, Test MAPE: x.xxxx, Test RMSE: x.xxxx
```

Model weights for each epoch will be saved in the directory specified by `--save`.

## Visualization Tools

The project provides some simple plotting scripts under `tools/`:

- `tools/plotline.py`: line chart plotting
- `tools/zhuzhuangtu.py`: bar chart plotting
- `tools/创建物理图.py`: physical network topology plotting

Run directly:

```bash
python tools/plotline.py
```

## Notes

1. **Large file limits**: `0916.mdb` (~380MB), `data/nanjing/4.2 NJAFC.rar` (~6.9GB), and `data/shanghai/train.pkl` (~144MB) are not included in the repository because they exceed GitHub's 100MB single-file limit. Please prepare them separately.
2. **Data paths**: Before running training, make sure the files for the corresponding dataset are complete under `data/`.
3. **CUDA version**: Please select the appropriate PyTorch CUDA version according to your local GPU driver.

## Citation

This project corresponds to the paper *Research on Traffic Flow Prediction Method Based on Spatio-Temporal Heterogeneous Feature Decoupling*.
