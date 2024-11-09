import numpy as np
from scipy import linalg

# 输入个数, (无限维空间划分)
x = np.array([2, 2, 4, 8, 4])
y = np.array([2, 6, 6, 8, 8])
# x - x_mean
x = x - np.mean(x)
y = y - np.mean(y)
A = np.stack((x, y))
# 协方差
c = np.cov(x, y)
# 拿到特征值, 特征向量
value, vector = linalg.eig(c)
# 样本矩阵为
S = np.stack((x, y))
# 转置
vector_t = vector.T
# 求出协方差矩阵在各个轴上的强度
res = (vector_t@c@vector)/(len(x) - 1)
# 选择强度最大的作为主轴, 强度第二大的作为副轴
new_XY = (vector_t@A)
print(new_XY)