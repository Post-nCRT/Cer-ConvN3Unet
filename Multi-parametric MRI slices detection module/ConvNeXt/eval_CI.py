import math

# 输入数据
AUC = 0.9989
n = 2678

# 计算标准误差
SE_AUC = math.sqrt(AUC * (1 - AUC) / n)

# 计算置信区间
lower_bound = AUC - 1.96 * SE_AUC
upper_bound = AUC + 1.96 * SE_AUC

print(f'AUC: {AUC:.4f}')
print(f'95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]')
