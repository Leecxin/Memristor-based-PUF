import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os


# 读取 Excel 文件
df = pd.read_csv('output.csv', header=None)[2:]

print('The number of Responses is:',df.shape[0])
responses_array = np.array(df[0].tolist())
num_columns = 32

num_rows = len(responses_array) // num_columns

print('32-bit Responses length is:', num_rows )
reshaped_responses = responses_array[:num_rows*num_columns].reshape((num_rows, num_columns))
reshaped_responses = reshaped_responses.astype(int)

print('reshaped_responses is:',reshaped_responses)

def hamming_weight(arr):
    """Normalized Hamming weight"""
    #dividing the sum of non-zero elements by the total number of elements in the sequence.
    return np.sum(arr)/len(arr)
  #sum求非0数总和，len求arr的total number

def calculate_uniformity(responses):
    hamming_weights = [hamming_weight(response) for response in responses]

    uniformity = sum(hamming_weights) / len(responses)

    return uniformity, hamming_weights

# ======================================= 计算均匀性指标 ======================================
uniformity, all_hm_w = calculate_uniformity(reshaped_responses[:31250])
print(f"Uniformity: {uniformity:.4f}")
# print(f"Hamming Weights in all responses : {all_hm_w}")

mean_hw_d = np.mean(all_hm_w)
variance_hw_d = np.var(all_hm_w)

print(f"Mean of Hamming weights: {mean_hw_d:.4f}")
print(f"Variance of Hamming weights: {variance_hw_d}")



file_name = f"./figures/{df[:12]}_Uniformity.png"

# Create a histogram for all_hm_d
plt.figure()  # Add this line
plt.hist(all_hm_w, bins=80,  color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Hamming weights')
plt.ylabel('Frequency')
plt.title('Uniformity')
os.makedirs("Figures", exist_ok=True)
plt.savefig(file_name)


plt.show()
# 计算每一行的平均值
# row_means = df.mean(axis=1)
# # 计算平均值的平均值和方差
# average_of_means = row_means.mean()
# variance_of_means = row_means.var()
#
# print(f"每一行的平均值：\n{row_means}")
# print(f"\n平均值的平均值：{average_of_means}")
# print(f"平均值的方差：{variance_of_means}")
#
# # 绘制数据分布直方图
# plt.hist(row_means, bins=20, edgecolor='black')
# plt.title('数据分布直方图')
# plt.xlabel('Fractional Hamming Weight')
# plt.ylabel('Normalized count (%)')
# plt.show()
