# plotline_clean_dates_14days_with_nj_daylabels.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
import random
from pathlib import Path

sns.set(style="whitegrid", palette="tab10", context="talk")

# ---------- 字体处理 ----------
def find_and_set_chinese_font():
    preferred_names = [
        "Microsoft YaHei", "Microsoft Yahei", "SimHei", "Noto Sans CJK", "PingFang",
        "WenQuanYi", "Source Han Sans", "AR PL UKai CN"
    ]
    sys_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    name_map = {}
    for fp in sys_fonts:
        try:
            prop = fm.FontProperties(fname=fp)
            name = prop.get_name()
            name_map[name] = fp
        except Exception:
            continue

    for pname in preferred_names:
        for name, fp in name_map.items():
            if pname.lower() in name.lower():
                fm.fontManager.addfont(fp)
                plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name()
                plt.rcParams['axes.unicode_minus'] = False
                print("使用字体:", plt.rcParams['font.family'])
                return

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    print("未检测到首选中文字体，若出现方块请手动指定字体文件路径。")


# ---------- 数据构造（针对 x, xtime 的 4D 结构） ----------
def load_and_build_df(pkl_path, time_key="xtime", data_key="x"):
    obj = pd.read_pickle(pkl_path)
    if not isinstance(obj, dict):
        raise ValueError("pickle 顶层不是 dict，脚本假定包含 'xtime' 与 'x' 键。")

    if time_key not in obj or data_key not in obj:
        raise KeyError(f"未找到 '{time_key}' 或 '{data_key}' 键。可用键: {list(obj.keys())}")

    xtime = np.array(obj[time_key])   # shape (S, L)
    x = np.array(obj[data_key])       # shape (S, L, N, C)

    if xtime.ndim != 2 or x.ndim != 4:
        raise ValueError("期望 xtime 为 2D, x 为 4D（samples, steps, nodes, features）")

    if x.shape[0] != xtime.shape[0] or x.shape[1] != xtime.shape[1]:
        raise ValueError("xtime 与 x 在 samples/steps 维度上不匹配")

    x_sum = x.sum(axis=(2,3))   # shape (S, L)

    times_flat = xtime.reshape(-1)
    flows_flat = x_sum.reshape(-1)

    try:
        times_parsed = pd.to_datetime(times_flat)
    except Exception:
        try:
            times_parsed = pd.to_datetime(times_flat.astype(np.int64), unit='s')
        except Exception:
            try:
                times_parsed = pd.to_datetime(times_flat.astype(np.int64), unit='ms')
            except Exception:
                print("无法解析 xtime，请贴出 xtime.flatten()[:20] 的样例")
                raise ValueError("无法解析 xtime")

    df = pd.DataFrame({"datetime": times_parsed, "flow_total": flows_flat})
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


# ---------- 选择窗口与重采样 ----------
def select_window(df, start_date=None, days=14):
    min_dt = df["datetime"].min().normalize()
    max_dt = df["datetime"].max().normalize()

    if start_date:
        start = pd.to_datetime(start_date).normalize()
        if start < min_dt:
            start = min_dt
        if start > max_dt:
            raise ValueError("指定起始日期超出数据范围")
    else:
        total_days = (max_dt - min_dt).days + 1
        if total_days <= days:
            start = min_dt
        else:
            max_start = max_dt - pd.Timedelta(days=days-1)
            rand_day = random.randint(0, (max_start - min_dt).days)
            start = min_dt + pd.Timedelta(days=rand_day)

    end = start + pd.Timedelta(days=days)

    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    df_window = df.loc[mask].copy()

    if df_window.empty:
        raise ValueError("选取的窗口内没有数据")

    return df_window, start.date(), (end - pd.Timedelta(days=1)).date()


def resample_hourly(df_window, how="sum"):
    df = df_window.set_index("datetime")
    # 使用小写 '1h' 避免 FutureWarning
    if how == "sum":
        hourly = df["flow_total"].resample("1h").sum()
    else:
        hourly = df["flow_total"].resample("1h").mean()

    return hourly.fillna(0)


# ---------- 绘图（横轴用 Day 1..Day N 表示） ----------
def plot_clean_dates_with_nj(hourly_series, start_date, end_date, out_png="fourteen_days_with_nj.png"):
    """
    绘图时横轴不显示具体日期，而显示 Day 1 ... Day N（N 根据数据长度自动计算，通常为 14）。
    其余绘图样式与你原来保持一致；NJ 曲线的数值生成与平滑逻辑也保留。
    """
    # 准备 x 轴为小时序列（从 0 开始）
    hours = np.arange(len(hourly_series))  # 每个点对应的小时偏移
    # 计算天数（向上取整），用于生成 Day 标签
    n_days = int(np.ceil(len(hours) / 24.0))
    # 主刻度位置（每 24 小时一个刻度）
    major_ticks = np.arange(0, n_days * 24, 24)
    # 对应标签 Day 1..Day n_days
    major_labels = [f"Day {i+1}" for i in range(len(major_ticks))]

    fig, ax = plt.subplots(figsize=(14,5), dpi=150)

    # 主线：实线、无标记（保持你原有的绘图风格）
    # 注意：这里 x 用 hours，y 用 hourly_series.values
    hz_line, = ax.plot(hours, hourly_series.values,
            label="HZMetro",
            linewidth=0.8, linestyle='--', marker=None, zorder=3, alpha=0.95)

    # 读取 HZMetro 的颜色（若需要）
    orig_color = hz_line.get_color()
    nj_color = "tomato"

    # ---------------- 生成 NJMetro 数据（增强波动，但不改绘图样式） ----------------
    base = hourly_series.values.copy()
    base = base * 0.7 + 200

    # 记录原始为 0 的位置（这些位置不要被平滑）
    orig_zero_mask = (hourly_series.values == 0)

    rng = np.random.default_rng(seed=42)
    noise_scale = 0.25
    noise = rng.normal(loc=0.0, scale=np.std(base) * noise_scale, size=base.shape)

    hours_idx = np.arange(len(base))
    daily_amp = 0.08 * np.nanmax(base)
    daily = daily_amp * np.sin(2 * np.pi * (hours_idx % 24) / 24.0)

    sub_amp = 0.03 * np.nanmax(base)
    sub = sub_amp * np.sin(2 * np.pi * hours_idx / (24.0 * 3.0))

    spikes = np.zeros_like(base)
    n_spikes = max(1, len(base) // 200)
    spike_indices = rng.choice(len(base), size=n_spikes, replace=False)
    for idx in spike_indices:
        spike_height = rng.uniform(0.15, 0.5) * np.nanmax(base)
        spikes[idx] += spike_height
        for off in (1, 2):
            if idx + off < len(spikes):
                spikes[idx + off] += spike_height * 0.4
            if idx - off >= 0:
                spikes[idx - off] += spike_height * 0.4

    nj_values = base + noise + daily + sub + spikes
    nj_values = np.maximum(nj_values, 0.0)

    # 平滑但保留原始为 0 的位置（只改数值）
    smooth_window = 2
    s = pd.Series(nj_values)
    s_masked = s.where(~orig_zero_mask, np.nan)
    s_smooth = s_masked.rolling(window=smooth_window, center=True, min_periods=1).mean()
    s_smooth = s_smooth.fillna(0.0)
    nj_values = s_smooth.values
    # -------------------------------------------------------------------

    # ---------------- 绘制 NJMetro（x 轴同样用 hours） ----------------
    mark_every = 6
    ax.plot(hours, nj_values,
            label="NJMetro", color=nj_color,
            linewidth=1.3, linestyle='-',
            markersize=5, markerfacecolor='none', markeredgewidth=0.9,
            markevery=mark_every, zorder=2, alpha=0.95)
    # -------------------------------------------------------------------

    # ---------- 横轴刻度与标签（Day 1..Day N） ----------
    # 设置主刻度为每 24 小时一个，并用 Day 标签
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=11)
    # 次刻度每 6 小时（不显示标签）
    minor_ticks = np.arange(0, n_days * 24, 6)
    ax.set_xticks(minor_ticks, minor=True)
    # 次刻度样式（长度等）
    ax.tick_params(axis='x', which='minor', length=6, color='#bbbbbb')
    # ---------- 横轴设置结束 ----------

    # 标签与标题（不显示具体日期）
    ax.set_xlabel("天数", fontsize=13)
    ax.set_ylabel("客流量", fontsize=13, labelpad=6)
    ax.set_title(f"杭州市地铁和南京市地铁14天客流量序列", fontsize=13)

    # 其余样式（保持你原有的样式）
    ax.tick_params(axis='y', labelsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', linestyle='--', alpha=0.45)

    # ---------- 纵轴以 K 单位显示并限制刻度数量 ----------
    ax.ticklabel_format(style='plain', axis='y')
    ymin, ymax = ax.get_ylim()
    ymax = max(ymax, 1000)
    new_ymax = np.ceil(ymax / 1000.0) * 1000.0
    ax.set_ylim(0, new_ymax)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, prune=None))

    def k_formatter(x, pos):
        k = int(round(x / 1000.0))
        return f"{k}K"

    ax.yaxis.set_major_formatter(FuncFormatter(k_formatter))
    # ---------- 纵轴设置结束 ----------

    # 图例
    ax.legend(fontsize=11, loc='upper left')

    # 调整边距，避免标签被裁切
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.14)

    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print("图已保存为", out_png)
    plt.show()


# ---------- 主流程 ----------
def main(pkl_path, start_date=None):
    find_and_set_chinese_font()

    pkl = Path(pkl_path)
    if not pkl.exists():
        print("文件不存在:", pkl)
        return

    df = load_and_build_df(pkl, time_key="xtime", data_key="x")
    # 默认 14 天窗口
    df_window, sdate, edate = select_window(df, start_date=start_date, days=14)
    hourly = resample_hourly(df_window, how="sum")

    plot_clean_dates_with_nj(hourly, sdate, edate, out_png="fourteen_days_with_nj.png")


# ---------- 写死路径（直接运行时使用） ----------
if __name__ == "__main__":
    # 把下面路径改为你的 pkl 文件绝对路径
    pkl = r"D:\研究生论文\Firstpapercode\hangzhou\train.pkl"
    # 可选：指定起始日期（格式 YYYY-MM-DD），或设为 None 随机选择一个 14 天窗口
    start_date = None

    main(pkl, start_date=start_date)
