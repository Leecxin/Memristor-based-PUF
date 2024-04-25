import itertools

import numpy as np

def pro(list):
    # list=[1,3,5,6,7,8] templist=10*[0]
    templist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in list:
        templist[i]=1
    return templist

digits_1 = list(range(25))
digits_2 = list(range(16))

# 枚举所有可能的6个数字的组合

digit_combinations_1 = itertools.combinations(digits_1, 4)
digit_combinations_2 = itertools.combinations(digits_2, 2)

# 将每个组合转换为字符串形式，用逗号分隔

digit_combinations_str_1 = [",".join(str(d) for d in comb) for comb in digit_combinations_1]
digit_combinations_str_2 = [",".join(str(d) for d in comb) for comb in digit_combinations_2]

print(digit_combinations_str_1, digit_combinations_str_2)

print(digit_combinations_str_1[0],digit_combinations_str_2[0])
# 输出所有可能的6个数字的组合

import torch
import torch.nn.functional as F
import torch.distributions as dist
import pandas as pd
import openpyxl

import random
import itertools


# 设置随机种子，以保证结果可复现
random.seed(42)
torch.manual_seed(42)


# 构造初始矩阵
# 创建一个10x10的矩阵，初始每个单元格全为35
matrix = torch.full((25, 16), 35.0)

# 生成高斯分布的随机数，均值为0，标准差为0.2
random_numbers = torch.randn_like(matrix) * 0.2

# 将随机数加到矩阵上
matrix += random_numbers
# print(matrix)
# 绘制初始矩阵的数值分布直方图
matrix_flattened = matrix.flatten()
matrix_flattened_np = matrix_flattened.numpy()

pd.DataFrame(matrix_flattened_np).hist()
# 构建保存结果的DataFrame
result_df = pd.DataFrame(columns=["Selected Rows", "Unselected Rows", "Selected Cols",
                                  "Left 3 Cols Sum", "Right 3 Cols Sum", "Output"])
i = 0
temp = torch.Tensor(np.zeros_like(matrix))

for selected_rows in digit_combinations_str_1:

    selected_rows = list(map(int, selected_rows.split(',')))

    temp[selected_rows] = matrix[selected_rows]*0.4


    # 未被选择的行设为0
    unselected_rows = list(set(range(25)) - set(selected_rows))
    temp[unselected_rows] = 0

    # 枚举所有可能的选择情况
    possible_choices = list(itertools.combinations(range(25), 4))

    # 随机选择2列
    for selected_cols in digit_combinations_str_2:
        selected_cols = list(map(int, selected_cols.split(',')))
        col_sums =  temp[:, selected_cols].sum(dim=0)
        # 将6列分为左3列和右3列，计算它们的数值总和
        left_cols_sum = col_sums[:1].sum()
        right_cols_sum = col_sums[1:].sum()

        # 比较左3列和右3列总和大小，输出结果为1或0
        output = 1 if right_cols_sum > left_cols_sum else 0

        # 将结果保存到DataFrame中
        selected_cols2 = pro(selected_cols)
        selected_rows2 = pro(selected_rows)
        result_df.loc[i] = [selected_rows2, unselected_rows, selected_cols2, left_cols_sum, right_cols_sum, output]
        i += 1
        print(i)
        # 将结果保存到Excel表格中
result_df.to_excel("result.xlsx", index=False)

