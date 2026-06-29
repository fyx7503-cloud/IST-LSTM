"""
NJMetro AFC原始数据处理脚本
================================
按照论文《基于时空异质特征解耦的交通流量预测方法研究》中NJMetro数据集的统计方式，
将AFC打卡原始数据处理为模型可用的训练数据。

处理流程:
1. 读取AFC原始数据 (.mdb 或 CSV格式)
2. 数据清洗与预处理
3. 按时间窗口聚合 (5min/15min/30min)
4. 生成站点级客流数据 (进站/出站/断面)
5. 生成OD流量矩阵
6. 构造滑动窗口样本
7. 划分训练集/验证集/测试集

Usage:
    # 直接读取mdb文件 (需要安装Access驱动或mdbtools)
    python process_njmetro_afc.py --input 0916.mdb --output njmetro_data --interval 5
    
    # 读取CSV文件 (如果mdb无法读取，先用Access导出为CSV)
    python process_njmetro_afc.py --input 0916.csv --output njmetro_data --interval 5
    
    # 指定站点数量 (默认自动推断)
    python process_njmetro_afc.py --input 0916.mdb --output njmetro_data --num_nodes 159
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='NJMetro AFC Data Processing')
    parser.add_argument('--input', type=str, default='0916.mdb', help='输入文件路径 (.mdb 或 .csv)')
    parser.add_argument('--output', type=str, default='njmetro_data', help='输出目录')
    parser.add_argument('--interval', type=int, default=5, choices=[5, 15, 30], 
                        help='时间聚合间隔 (分钟): 5, 15, 30')
    parser.add_argument('--num_nodes', type=int, default=None, help='站点数量 (默认自动推断)')
    parser.add_argument('--seq_length', type=int, default=12, help='输入历史时间步长')
    parser.add_argument('--pred_length', type=int, default=4, help='预测未来时间步长')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--start_date', type=str, default='2021-09-16', help='数据起始日期')
    parser.add_argument('--end_date', type=str, default='2021-10-29', help='数据结束日期')
    return parser.parse_args()


# ============================================================
# 1. 数据读取模块
# ============================================================

def read_mdb_pandas_access(filepath):
    """尝试使用pandas_access读取mdb文件"""
    try:
        import pandas_access as mdb
        tables = mdb.list_tables(filepath)
        print(f"[INFO] MDB tables found: {tables}")
        
        # 通常AFC数据表名可能是 'AFC', 'Transaction', 'Records', '刷卡记录' 等
        for tbl in tables:
            tbl_lower = tbl.lower()
            if any(k in tbl_lower for k in ['afc', 'trans', 'record', '刷卡', '交易', 'data']):
                df = mdb.read_table(filepath, tbl)
                print(f"[INFO] Read table '{tbl}', shape: {df.shape}")
                return df
        
        # 如果没有匹配的表名，读取第一个表
        if tables:
            df = mdb.read_table(filepath, tables[0])
            print(f"[INFO] Read first table '{tables[0]}', shape: {df.shape}")
            return df
    except Exception as e:
        print(f"[WARN] pandas_access failed: {e}")
    return None


def read_mdb_pyodbc(filepath):
    """尝试使用pyodbc读取mdb文件"""
    try:
        import pyodbc
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={filepath};'
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        tables = [t.table_name for t in cursor.tables(tableType='TABLE')]
        print(f"[INFO] MDB tables found: {tables}")
        
        for tbl in tables:
            tbl_lower = tbl.lower()
            if any(k in tbl_lower for k in ['afc', 'trans', 'record', '刷卡', '交易', 'data']):
                df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
                print(f"[INFO] Read table '{tbl}', shape: {df.shape}")
                conn.close()
                return df
        
        if tables:
            df = pd.read_sql(f"SELECT * FROM [{tables[0]}]", conn)
            print(f"[INFO] Read first table '{tables[0]}', shape: {df.shape}")
            conn.close()
            return df
        conn.close()
    except Exception as e:
        print(f"[WARN] pyodbc failed: {e}")
    return None


def read_mdb_pypyodbc(filepath):
    """尝试使用pypyodbc读取mdb文件"""
    try:
        import pypyodbc
        conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={filepath};'
        conn = pypyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        tables = [t.table_name for t in cursor.tables(tableType='TABLE')]
        print(f"[INFO] MDB tables found: {tables}")
        
        for tbl in tables:
            tbl_lower = tbl.lower()
            if any(k in tbl_lower for k in ['afc', 'trans', 'record', '刷卡', '交易', 'data']):
                df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
                print(f"[INFO] Read table '{tbl}', shape: {df.shape}")
                conn.close()
                return df
        
        if tables:
            df = pd.read_sql(f"SELECT * FROM [{tables[0]}]", conn)
            print(f"[INFO] Read first table '{tables[0]}', shape: {df.shape}")
            conn.close()
            return df
        conn.close()
    except Exception as e:
        print(f"[WARN] pypyodbc failed: {e}")
    return None


def read_mdb_meza(filepath):
    """尝试使用meza读取mdb文件"""
    try:
        from meza import io
        records = list(io.read_mdb(filepath))
        if records:
            df = pd.DataFrame(records)
            print(f"[INFO] Read via meza, shape: {df.shape}")
            return df
    except Exception as e:
        print(f"[WARN] meza failed: {e}")
    return None


def read_mdb(filepath):
    """
    尝试多种方式读取mdb文件
    如果都失败，提示用户导出为CSV
    """
    print(f"[INFO] Attempting to read MDB file: {filepath}")
    
    # 尝试各种读取方式
    df = read_mdb_pandas_access(filepath)
    if df is not None:
        return df
    
    df = read_mdb_pyodbc(filepath)
    if df is not None:
        return df
    
    df = read_mdb_pypyodbc(filepath)
    if df is not None:
        return df
    
    df = read_mdb_meza(filepath)
    if df is not None:
        return df
    
    # 所有方式都失败
    print("\n" + "="*70)
    print("[ERROR] 无法直接读取 .mdb 文件。可能的原因:")
    print("  1. 系统未安装 Microsoft Access Database Engine")
    print("  2. 文件格式不兼容或已损坏")
    print("  3. 文件需要密码")
    print("\n[建议] 请使用以下方法之一导出数据:")
    print("  方法A: 用 Microsoft Access 打开 0916.mdb，将主数据表导出为 CSV")
    print("  方法B: 安装 mdbtools (http://sourceforge.net/projects/mdbtools/)")
    print("  方法C: 安装 Access Database Engine (Microsoft 官网下载)")
    print("\n导出CSV后，运行:")
    print(f"  python process_njmetro_afc.py --input 0916.csv --output njmetro_data --interval 5")
    print("="*70 + "\n")
    
    csv_path = filepath.replace('.mdb', '.csv')
    if os.path.exists(csv_path):
        print(f"[INFO] 检测到同名CSV文件: {csv_path}，尝试读取...")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[INFO] Read CSV, shape: {df.shape}")
        return df
    
    sys.exit(1)


def read_data(filepath):
    """通用数据读取入口"""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.mdb':
        return read_mdb(filepath)
    elif ext == '.csv':
        print(f"[INFO] Reading CSV file: {filepath}")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"[INFO] Read CSV, shape: {df.shape}")
        return df
    elif ext in ['.xls', '.xlsx']:
        print(f"[INFO] Reading Excel file: {filepath}")
        df = pd.read_excel(filepath)
        print(f"[INFO] Read Excel, shape: {df.shape}")
        return df
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ============================================================
# 2. 数据预处理模块
# ============================================================

def infer_columns(df):
    """
    推断AFC数据的字段含义
    常见的AFC字段命名:
    - 进站站点: entry_station, origin, from_station, 进站, 上车站, o_station
    - 出站站点: exit_station, destination, to_station, 出站, 下车站, d_station
    - 进站时间: entry_time, origin_time, in_time, 进站时间, 上车时间
    - 出站时间: exit_time, destination_time, out_time, 出站时间, 下车时间
    - 卡号/交易号: card_id, transaction_id, 卡号, 交易号
    """
    cols = {c.lower(): c for c in df.columns}
    
    mapping = {}
    
    # 进站站点
    for key in ['entry_station', 'origin', 'from_station', '进站', '上车站', 
                'o_station', 'start_station', 'source', 'start', 'origin_station',
                'enter_station', 'in_station', '上车', '始发站']:
        if key in cols:
            mapping['entry_station'] = cols[key]
            break
    
    # 出站站点
    for key in ['exit_station', 'destination', 'to_station', '出站', '下车站',
                'd_station', 'end_station', 'target', 'end', 'dest_station',
                'leave_station', 'out_station', '下车', '终点站']:
        if key in cols:
            mapping['exit_station'] = cols[key]
            break
    
    # 进站时间
    for key in ['entry_time', 'origin_time', 'in_time', '进站时间', '上车时间',
                'start_time', 'enter_time', 'o_time', '刷卡时间', 'transaction_time',
                '交易时间', 'time', 'datetime', 'date_time']:
        if key in cols:
            mapping['entry_time'] = cols[key]
            break
    
    # 出站时间
    for key in ['exit_time', 'destination_time', 'out_time', '出站时间', '下车时间',
                'end_time', 'leave_time', 'd_time']:
        if key in cols:
            mapping['exit_time'] = cols[key]
            break
    
    # 卡号
    for key in ['card_id', 'transaction_id', '卡号', '交易号', 'id', '票号',
                'ticket_id', 'card_no', 'user_id', '乘客号']:
        if key in cols:
            mapping['card_id'] = cols[key]
            break
    
    # 如果没有找到出站时间，但有进站时间和停留时间/行程时间，可以计算
    if 'exit_time' not in mapping and 'entry_time' in mapping:
        for key in ['travel_time', 'duration', '行程时间', '停留时间', 'time_diff']:
            if key in cols:
                mapping['duration'] = cols[key]
                break
    
    print("[INFO] Inferred column mapping:")
    for k, v in mapping.items():
        print(f"  {k}: '{v}'")
    
    # 如果没有推断出关键字段，打印所有列名供参考
    if 'entry_station' not in mapping or 'entry_time' not in mapping:
        print("\n[WARN] 未能自动推断所有关键字段，可用的列名:")
        for c in df.columns:
            print(f"  - {c}")
        print("\n请根据上述列名修改脚本中的 infer_columns 函数。")
    
    return mapping


def clean_data(df, col_map):
    """
    数据清洗:
    1. 去除重复记录
    2. 处理缺失值
    3. 过滤异常时间
    4. 解析时间格式
    """
    print("\n[INFO] Starting data cleaning...")
    
    # 提取关键列
    required_cols = []
    if 'entry_station' in col_map:
        required_cols.append(col_map['entry_station'])
    if 'exit_station' in col_map:
        required_cols.append(col_map['exit_station'])
    if 'entry_time' in col_map:
        required_cols.append(col_map['entry_time'])
    
    if len(required_cols) < 2:
        raise ValueError("关键字段缺失，无法继续处理")
    
    df_clean = df[required_cols].copy()
    
    # 去除完全重复的记录
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    n_after = len(df_clean)
    print(f"[INFO] Removed {n_before - n_after} duplicate records")
    
    # 去除含缺失值的记录
    df_clean = df_clean.dropna()
    print(f"[INFO] After dropping NA: {len(df_clean)} records")
    
    # 解析时间
    time_col = col_map['entry_time']
    
    # 尝试多种时间格式
    def parse_time(x):
        if isinstance(x, pd.Timestamp):
            return x
        if pd.isna(x):
            return pd.NaT
        x = str(x).strip()
        for fmt in [
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M',
            '%d-%m-%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
            '%Y%m%d%H%M%S', '%Y%m%d %H%M%S',
            '%H:%M:%S', '%H:%M'
        ]:
            try:
                return pd.to_datetime(x, format=fmt)
            except:
                continue
        try:
            return pd.to_datetime(x)
        except:
            return pd.NaT
    
    df_clean['parsed_time'] = df_clean[time_col].apply(parse_time)
    df_clean = df_clean.dropna(subset=['parsed_time'])
    print(f"[INFO] After time parsing: {len(df_clean)} records")
    
    # 过滤运营时间外的数据 (假设地铁运营时间 05:00 - 23:30)
    df_clean['hour'] = df_clean['parsed_time'].dt.hour
    df_clean = df_clean[(df_clean['hour'] >= 5) & (df_clean['hour'] <= 23)]
    print(f"[INFO] After filtering operating hours: {len(df_clean)} records")
    
    # 标准化站点编码为整数
    entry_col = col_map['entry_station']
    exit_col = col_map.get('exit_station', entry_col)
    
    df_clean['entry_station'] = pd.to_numeric(df_clean[entry_col], errors='coerce').astype(int)
    if exit_col in df_clean.columns:
        df_clean['exit_station'] = pd.to_numeric(df_clean[exit_col], errors='coerce').astype(int)
    else:
        df_clean['exit_station'] = df_clean['entry_station']
    
    df_clean = df_clean.dropna(subset=['entry_station', 'exit_station'])
    print(f"[INFO] After station cleaning: {len(df_clean)} records")
    
    # 保留需要的列
    df_clean = df_clean[['parsed_time', 'entry_station', 'exit_station']].copy()
    df_clean.columns = ['timestamp', 'entry_station', 'exit_station']
    
    return df_clean


# ============================================================
# 3. 客流聚合模块
# ============================================================

def aggregate_flow(df_clean, interval=5, num_nodes=None):
    """
    按时间窗口聚合客流数据
    
    返回:
    - inflow: (num_timesteps, num_nodes) 各站点进站客流
    - outflow: (num_timesteps, num_nodes) 各站点出站客流
    - od_matrix: (num_timesteps, num_nodes, num_nodes) OD流量矩阵
    """
    print(f"\n[INFO] Aggregating flow with {interval}-minute intervals...")
    
    # 创建时间分桶
    df_clean['time_bucket'] = df_clean['timestamp'].dt.floor(f'{interval}min')
    
    # 确定站点数量
    all_stations = set(df_clean['entry_station'].unique()) | set(df_clean['exit_station'].unique())
    if num_nodes is None:
        num_nodes = int(max(all_stations)) + 1
        # 如果站点编号从1开始
        if min(all_stations) >= 1:
            num_nodes = int(max(all_stations)) + 1
    
    print(f"[INFO] Number of stations: {num_nodes}")
    print(f"[INFO] Date range: {df_clean['timestamp'].min()} ~ {df_clean['timestamp'].max()}")
    
    # 创建完整的时间索引
    start_time = df_clean['timestamp'].min().normalize() + timedelta(hours=5)
    end_time = df_clean['timestamp'].max().normalize() + timedelta(hours=23, minutes=30)
    time_index = pd.date_range(start=start_time, end=end_time, freq=f'{interval}min')
    n_timesteps = len(time_index)
    print(f"[INFO] Time steps: {n_timesteps} ({interval}min intervals)")
    
    # 初始化流量矩阵
    inflow = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
    outflow = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
    od_matrix = np.zeros((n_timesteps, num_nodes, num_nodes), dtype=np.float32)
    
    # 按时间分桶聚合
    for time_bucket, group in df_clean.groupby('time_bucket'):
        if time_bucket not in time_index:
            continue
        t_idx = time_index.get_loc(time_bucket)
        
        # 进站客流
        entry_counts = group['entry_station'].value_counts()
        for station, count in entry_counts.items():
            if 0 <= station < num_nodes:
                inflow[t_idx, station] += count
        
        # 出站客流
        exit_counts = group['exit_station'].value_counts()
        for station, count in exit_counts.items():
            if 0 <= station < num_nodes:
                outflow[t_idx, station] += count
        
        # OD矩阵
        od_counts = group.groupby(['entry_station', 'exit_station']).size()
        for (o, d), count in od_counts.items():
            if 0 <= o < num_nodes and 0 <= d < num_nodes:
                od_matrix[t_idx, o, d] += count
    
    print(f"[INFO] Aggregation complete. Total inflow: {inflow.sum():.0f}, outflow: {outflow.sum():.0f}")
    
    return inflow, outflow, od_matrix, time_index


def build_graph_matrices(inflow, outflow, od_matrix, num_nodes):
    """
    构建论文中的多种图结构矩阵:
    1. 深度关系矩阵 (基于物理拓扑)
    2. 旅行距离权重矩阵 (基于高斯核)
    3. 乘客流量权重矩阵 (基于OD数据)
    """
    print("\n[INFO] Building graph matrices...")
    
    # 1. 连通性矩阵 (基于是否有OD流量)
    total_od = od_matrix.sum(axis=0)  # (N, N)
    adj_conn = (total_od > 0).astype(np.float32)
    
    # 2. 相关性矩阵 (基于站点流量时间序列的皮尔逊相关系数)
    # 使用进站流量计算站点间相关性
    inflow_corr = np.corrcoef(inflow.T)
    inflow_corr = np.nan_to_num(inflow_corr, nan=0.0)
    adj_cor = np.abs(inflow_corr).astype(np.float32)
    np.fill_diagonal(adj_cor, 0)
    
    # 3. 相似性矩阵 (基于余弦相似度)
    from numpy.linalg import norm
    inflow_norm = inflow.T  # (N, T)
    norms = norm(inflow_norm, axis=1, keepdims=True)
    norms[norms == 0] = 1
    inflow_normalized = inflow_norm / norms
    adj_sml = np.dot(inflow_normalized, inflow_normalized.T).astype(np.float32)
    np.fill_diagonal(adj_sml, 0)
    
    # 4. OD流量权重矩阵 (论文中的F矩阵)
    od_total = od_matrix.sum(axis=0)
    row_sum = od_total.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    adj_od = od_total / row_sum
    adj_od = adj_od.astype(np.float32)
    
    print("[INFO] Graph matrices built:")
    print(f"  adj_conn: {adj_conn.shape}, nonzeros: {np.count_nonzero(adj_conn)}")
    print(f"  adj_cor:  {adj_cor.shape}")
    print(f"  adj_sml:  {adj_sml.shape}")
    print(f"  adj_od:   {adj_od.shape}")
    
    return adj_conn, adj_cor, adj_sml, adj_od


# ============================================================
# 4. 样本构造模块
# ============================================================

def build_samples(flow_data, seq_length=12, pred_length=4):
    """
    使用滑动窗口构造训练样本
    
    flow_data: (T, N, C)  C=2 [inflow, outflow]
    return: x, y
    x: (num_samples, seq_length, N, C)
    y: (num_samples, pred_length, N, C)
    """
    T, N, C = flow_data.shape
    num_samples = T - seq_length - pred_length + 1
    
    if num_samples <= 0:
        raise ValueError(f"时间步不足: T={T}, seq={seq_length}, pred={pred_length}")
    
    x = np.zeros((num_samples, seq_length, N, C), dtype=np.float32)
    y = np.zeros((num_samples, pred_length, N, C), dtype=np.float32)
    
    for i in range(num_samples):
        x[i] = flow_data[i : i + seq_length]
        y[i] = flow_data[i + seq_length : i + seq_length + pred_length]
    
    return x, y


def split_dataset(x, y, train_ratio=0.6, val_ratio=0.2):
    """按时间顺序划分数据集"""
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    x_test, y_test = x[n_train+n_val:], y[n_train+n_val:]
    
    print(f"\n[INFO] Dataset split:")
    print(f"  Train: {len(x_train)} samples")
    print(f"  Val:   {len(x_val)} samples")
    print(f"  Test:  {len(x_test)} samples")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ============================================================
# 5. 主流程
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 读取原始数据
    df_raw = read_data(args.input)
    
    # 2. 推断字段映射
    col_map = infer_columns(df_raw)
    
    # 3. 数据清洗
    df_clean = clean_data(df_raw, col_map)
    
    # 4. 按时间窗口聚合
    inflow, outflow, od_matrix, time_index = aggregate_flow(
        df_clean, interval=args.interval, num_nodes=args.num_nodes
    )
    
    # 保存时间索引
    with open(os.path.join(args.output, 'time_index.pkl'), 'wb') as f:
        pickle.dump(time_index, f)
    
    # 5. 构建图结构矩阵
    adj_conn, adj_cor, adj_sml, adj_od = build_graph_matrices(
        inflow, outflow, od_matrix, inflow.shape[1]
    )
    
    # 保存图矩阵
    with open(os.path.join(args.output, 'graph_nj_conn.pkl'), 'wb') as f:
        pickle.dump(adj_conn, f)
    with open(os.path.join(args.output, 'graph_nj_cor.pkl'), 'wb') as f:
        pickle.dump(adj_cor, f)
    with open(os.path.join(args.output, 'graph_nj_sml.pkl'), 'wb') as f:
        pickle.dump(adj_sml, f)
    with open(os.path.join(args.output, 'graph_nj_od.pkl'), 'wb') as f:
        pickle.dump(adj_od, f)
    
    # 6. 构造流量张量
    # flow_data: (T, N, 2) = [inflow, outflow]
    flow_data = np.stack([inflow, outflow], axis=-1).astype(np.float32)
    
    # 7. 构造滑动窗口样本
    x, y = build_samples(flow_data, args.seq_length, args.pred_length)
    
    # 8. 划分数据集
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
        x, y, args.train_ratio, args.val_ratio
    )
    
    # 9. 保存为pickle格式
    for split_name, x_data, y_data in [
        ('train', x_train, y_train),
        ('val', x_val, y_val),
        ('test', x_test, y_test)
    ]:
        save_dict = {
            'x': x_data,
            'y': y_data,
            # 保存对应的时间戳信息 (用于后续分析)
            'xtime': np.array([t.strftime('%Y-%m-%d %H:%M') for t in time_index[:len(x_data)]]),
            'ytime': np.array([t.strftime('%Y-%m-%d %H:%M') for t in time_index[:len(y_data)]])
        }
        filepath = os.path.join(args.output, f'{split_name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"[INFO] Saved {split_name}: {x_data.shape} -> {filepath}")
    
    # 10. 保存数据统计信息
    stats = {
        'num_nodes': inflow.shape[1],
        'num_timesteps': inflow.shape[0],
        'interval_minutes': args.interval,
        'seq_length': args.seq_length,
        'pred_length': args.pred_length,
        'train_samples': len(x_train),
        'val_samples': len(x_val),
        'test_samples': len(x_test),
        'mean_inflow': float(inflow.mean()),
        'std_inflow': float(inflow.std()),
        'mean_outflow': float(outflow.mean()),
        'std_outflow': float(outflow.std()),
    }
    
    with open(os.path.join(args.output, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\n{'='*70}")
    print("[SUCCESS] NJMetro data processing complete!")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print(f"Files generated:")
    print(f"  - train.pkl, val.pkl, test.pkl  (模型输入数据)")
    print(f"  - graph_nj_conn.pkl             (连通性矩阵)")
    print(f"  - graph_nj_cor.pkl              (相关性矩阵)")
    print(f"  - graph_nj_sml.pkl              (相似性矩阵)")
    print(f"  - graph_nj_od.pkl               (OD流量矩阵)")
    print(f"  - time_index.pkl                (时间索引)")
    print(f"  - stats.pkl                     (数据统计信息)")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
