import pandas as pd
from scipy.stats import norm

# 读取CSV文件
df = pd.read_csv('./metrics.csv')

# 提取第一列的百分数数据
percentages = df.iloc[:, 0]

# 计算每个百分数的95%置信区间
confidence_intervals = []

for percentage in percentages:
    # 将百分数转换为小数
    value = float(percentage.strip('%')) / 100.0

    # 计算标准误差
    standard_error = norm.ppf(0.975) * (value * (1 - value) / 2678) ** 0.5

    # 计算95%置信区间
    lower_bound = max(0, value - standard_error)
    upper_bound = min(1, value + standard_error)

    confidence_intervals.append("(95% CI: {:.5}%, {:.5}%)".format(str(lower_bound * 100), str(upper_bound * 100)))

# 将置信区间添加到DataFrame的第二列
df['95% CI'] = confidence_intervals

# 将DataFrame保存到原始CSV文件
df.to_csv('your_file.csv', index=False)
