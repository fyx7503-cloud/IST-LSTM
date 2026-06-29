import pickle
import numpy as np

# 打开pkl文件
with open(r'D:\Firstpapercode\graph_nj_conn.pkl', 'rb') as f:
    conn = pickle.load(f)

with open('D:\Firstpapercode\hangzhou\graph_hz_cor.pkl', 'rb') as f:
    cor = pickle.load(f)

with open('D:\Firstpapercode\hangzhou\graph_hz_sml.pkl', 'rb') as f:
    sml = pickle.load(f)


# 查看数据
np.set_printoptions(threshold=np.inf)

# print(conn)
# for key in conn.keys():
#     print(f"{key}:{conn[key][0]}")
#     print(f"{key}:{conn[key][1]}")
#     print(f"{key}:{conn[key][2]}")
#     print(f"{key}:{conn[key][3]}")