# IST-LSTM

基于时空异质特征解耦的交通流量预测方法（IST-LSTM）的 PyTorch 实现。

## 项目结构

```
IST-LSTM/
├── configs/              # 配置文件
├── data/                 # 数据集
│   ├── hangzhou/         # 杭州数据集
│   ├── nanjing/          # 南京地铁 AFC 数据集
│   └── shanghai/         # 上海数据集
├── models/               # 核心模型代码
│   ├── ist_lstm_model.py # IST-LSTM 网络结构
│   ├── ist_lstm_engine.py# 训练/评估引擎
│   └── ist_lstm_util.py  # 数据加载与工具函数
├── scripts/              # 数据处理脚本
│   ├── mdb_to_csv.py     # MDB 转 CSV 工具
│   ├── physicgraph.py    # 物理图构建
│   └── process_njmetro_afc.py  # 南京地铁 AFC 数据处理
├── tests/                # 单元测试
│   └── test_mdb_to_csv.py
├── tools/                # 可视化与辅助工具
│   ├── plotline.py       # 折线图绘制
│   ├── zhuzhuangtu.py    # 柱状图绘制
│   ├── 创建物理图.py      # 物理网络图绘制
│   └── ...
├── train.py              # 训练入口
└── README.md
```

## 环境配置

### 1. 安装 Python

建议使用 Python 3.8（项目中的 `__pycache__` 为 `cpython-38`）。

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy pandas matplotlib seaborn scipy networkx pyodbc
```

> 如果你有 NVIDIA GPU 且已安装 CUDA，可将 `torch` 替换为对应的 CUDA 版本，例如 `torch==1.9.0+cu111`。

## 数据准备及处理

### 已有数据集

仓库中已包含部分预处理后的数据：

- `data/hangzhou/`：完整数据（train / val / test / graph）
- `data/shanghai/`：除 `train.pkl` 外的数据（`train.pkl` 约 144MB，超过 GitHub 单文件限制，需单独准备）
- `data/nanjing/`：仅含 `graph_nj_conn.pkl`，完整数据需从 `0916.mdb` 处理得到

### 南京地铁 AFC 数据处理流程

#### 步骤 1：MDB 转 CSV（如需要）

如果原始数据为 Microsoft Access `.mdb` 格式，可先转换为 CSV：

```bash
python scripts/mdb_to_csv.py 0916.mdb --output-dir ./csv_output
```

该脚本会自动检测并尝试以下策略：
1. `pyodbc`（推荐 Windows）
2. `mdbtools`（推荐 Linux / macOS）
3. `win32com`（Windows COM 接口）

#### 步骤 2：生成模型可用数据

```bash
python scripts/process_njmetro_afc.py --input 0916.mdb --output data/nanjing --interval 5
```

主要参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--input` | 输入文件路径（`.mdb` 或 `.csv`） | `0916.mdb` |
| `--output` | 输出目录 | `njmetro_data` |
| `--interval` | 时间聚合窗口（分钟），可选 5/15/30 | `5` |
| `--num_nodes` | 站点数量，默认自动推断 | 自动 |

处理完成后，`data/nanjing/` 目录下会生成：

```
data/nanjing/
├── train.pkl
├── val.pkl
├── test.pkl
└── graph_nj_*.pkl
```

### 数据文件说明

| 文件 | 说明 |
|---|---|
| `train.pkl` / `val.pkl` / `test.pkl` | 训练、验证、测试集 |
| `graph_*_conn.pkl` | 连通性邻接矩阵 |
| `graph_*_cor.pkl` | 相关性邻接矩阵 |
| `graph_*_sml.pkl` | 相似性邻接矩阵 |

## 运行训练

### 杭州数据集

```bash
python train.py --data hangzhou --device cuda:0
```

### 上海数据集

> 需先确保 `data/shanghai/train.pkl` 已放入对应目录。

```bash
python train.py --data shanghai --device cuda:0
```

### 南京数据集

> 需先通过 `scripts/process_njmetro_afc.py` 生成 `data/nanjing/train.pkl`。

```bash
python train.py --data nanjing --device cuda:0
```

### 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--data` | 数据集名称：`hangzhou` / `shanghai` / `nanjing` | `hangzhou` |
| `--device` | 训练设备：`cuda:0` / `cpu` | `cuda:0` |
| `--epochs` | 训练轮数 | `200` |
| `--batch_size` | 批次大小 | `64` |
| `--learning_rate` | 学习率 | `0.001` |
| `--hidden_dim` | 隐藏层维度 | `64` |
| `--num_layers_main` | 主分支 LSTM 层数 | `3` |
| `--num_layers_sub` | 子分支 IST-LSTM 层数 | `2` |
| `--num_heads` | 注意力头数 | `4` |
| `--window_size` | Swin Transformer 窗口大小 | `2` |
| `--seq_length` | 输入时间步 | `4` |
| `--pred_steps` | 预测时间步 | `4` |
| `--use_dcg` | 是否使用 DCG 图结构 | 不启用 |
| `--save` | 模型保存路径 | `./checkpoints/ist_lstm` |

### CPU 训练

```bash
python train.py --data hangzhou --device cpu
```

## 测试流程

当前 `train.py` 在训练结束后会自动在测试集上评估最优模型，输出每个预测步及平均的 MAE / MAPE / RMSE：

```text
Evaluate best model on test data for horizon 1, Test MAE: x.xxxx, Test MAPE: x.xxxx, Test RMSE: x.xxxx
...
On average over 4 horizons, Test MAE: x.xxxx, Test MAPE: x.xxxx, Test RMSE: x.xxxx
```

训练过程中每个 epoch 的模型权重会保存在 `--save` 指定的目录中。

## 单元测试

```bash
python -m unittest tests.test_mdb_to_csv -v
```

或：

```bash
cd tests
python test_mdb_to_csv.py
```

## 可视化工具

项目提供了一些简单的绘图脚本，位于 `tools/` 目录：

- `tools/plotline.py`：折线图绘制
- `tools/zhuzhuangtu.py`：柱状图绘制
- `tools/创建物理图.py`：物理网络拓扑图绘制

可以直接运行：

```bash
python tools/plotline.py
```

## 注意事项

1. **大文件限制**：`0916.mdb`（约 380MB）、`data/nanjing/4.2 NJAFC.rar`（约 6.9GB）和 `data/shanghai/train.pkl`（约 144MB）因超过 GitHub 单文件 100MB 限制，未包含在仓库中，需要自行准备。
2. **数据路径**：运行训练前请确保 `data/` 下对应数据集的文件完整。
3. **CUDA 版本**：请根据本地 GPU 驱动选择合适的 PyTorch CUDA 版本。

## 引用

本项目对应论文《基于时空异质特征解耦的交通流量预测方法研究》。
