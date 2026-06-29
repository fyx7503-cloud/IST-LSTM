import numpy as np

# 加载.npz文件
data = np.load(r"E:\code\GCN_predict-Pytorch-main\PeMS_04\PeMS04.npz")

# 1. 查看文件中有哪些数组
print("文件中的数组名称:", data.files)  # 注意应该是 data.files，这里有一个拼写错误

# 2. 打印每个数组的详细信息（假设文件中有'data', 'indices', 'values'等数组）
for array_name in data.files:
    print("\n===== 数组名称:", array_name, "=====")
    array_data = data[array_name]
    print("形状:", array_data.shape)
    print("数据类型:", array_data.dtype)
    print("前几个元素示例:\n", array_data[:2] if len(array_data) > 0 else "空数组")  # 避免打印过多数据

# 关闭文件（可选，但推荐）
data.close()