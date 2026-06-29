import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5]
y = [10, 15, 13, 18, 16]

# 绘制折线图并用三角符号表示数据点
plt.plot(x, y, marker='^', linestyle='-', color='b',markerfacecolor='none',linewidth=1)

# 添加标题和标签
plt.title('折线图示例')
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 显示图表
plt.show()
