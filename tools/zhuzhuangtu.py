import numpy as np
from matplotlib import pyplot as plt
from pylab import mpl
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16  # 设置字体大小
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 设置每个时间点的人流量数据，符合早晚高峰特征
time_hours = np.arange(5, 24)  # 每小时一个数据点
passenger_counts = [0.2, 2, 12, 18, 14, 6, 5, 4, 3, 4, 8, 12, 17, 14.5, 8.2, 5, 3.5, 1, 0.5]  # 手动设置的人流量符合早晚高峰特征

# 创建图像并设置尺寸和分辨率
plt.figure(figsize=(13.28, 5.31), dpi=96)

# 绘制柱状图
plt.bar(time_hours, passenger_counts, color='darkgrey', edgecolor='white')

# 设置标题和标签
plt.suptitle("数据统计", y=-0.1)  # 使用suptitle并调整位置

plt.xlabel("出发时间")
plt.ylabel("乘客数量/万")
# plt.xlabel("Departure time")
# plt.ylabel("Number of passengers/10k")

# 设置x轴刻度值和刻度标签
time_ticks = np.arange(5, 24, 2)  # 每2小时一个刻度
time_labels = [f"{t}:00" for t in time_ticks]
plt.xticks(time_ticks, time_labels)

# 设置x轴范围
plt.xlim([5, 23])

# 设置y轴刻度
plt.yticks([10, 20])

# 只去掉顶部和右侧的轴线
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 显示图表
plt.show()
