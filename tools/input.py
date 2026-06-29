import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 原始数据
x   = np.arange(10)
y1  = np.array([10, 15, 12, 18, 20, 16, 14, 19, 17, 21])
y2  = np.array([5,  8,   6, 10, 12,  9,   7, 11, 10, 13])
y3  = np.array([20, 18, 22, 15, 17, 19, 16, 14, 18, 15])

# 垂直间隔
margin = 2
offset2 = y1.max()   - y2.min() + margin
offset3 = (y2 + offset2).max() - y3.min() + margin

y2_off = y2 + offset2
y3_off = y3 + offset3

# 平滑插值的 x 轴
x_smooth = np.linspace(x.min(), x.max(), 300)

# 三次样条插值对象
spl1 = make_interp_spline(x,    y1,     k=3)
spl2 = make_interp_spline(x,    y2_off, k=3)
spl3 = make_interp_spline(x,    y3_off, k=3)

# 计算平滑后的 y 值
y1_s = spl1(x_smooth)
y2_s = spl2(x_smooth)
y3_s = spl3(x_smooth)

# 绘制平滑曲线（无标签）
plt.figure(figsize=(8, 5))
plt.plot(x_smooth, y1_s, color='magenta', linewidth=2)
plt.plot(x_smooth, y2_s, color='teal',    linewidth=2)
plt.plot(x_smooth, y3_s, color='orange',  linewidth=2)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Smooth Non-Intersecting Curves with New Colors')
plt.tight_layout()
plt.show()
