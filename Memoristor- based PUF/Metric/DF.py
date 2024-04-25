import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os


df = pd.read_csv('output.csv', header=None)[2:]
responses_array = np.array(df[0].tolist())
num_columns = 32

num_rows = len(responses_array) // num_columns #1000/32=31

print('32-bit Responses length is:', num_rows )
reshaped_responses = responses_array[:num_rows*num_columns].reshape((num_rows, num_columns)) #31*32
print(reshaped_responses.shape)
print('reshaped_responses is:',reshaped_responses)
def hamming_distance(arr1, arr2):
    """Normalized Hamming distance"""
    assert len(arr1) == len(arr2), "The input arrays must have the same length"
    return np.sum(arr1 != arr2) / len(arr1)

def calculate_diffuseness(responses):
    num_responses = len(responses)
    pairwise_distances = []

    for i in range(num_responses):
        print(i)

        for j in range(i+1, num_responses):
            pairwise_distances.append(hamming_distance(responses[i], responses[j]))

    diffuseness = sum(pairwise_distances) / len(pairwise_distances)

    return diffuseness, pairwise_distances

# ====================================== 计算散度指标 ================================
diffuseness, all_hm_d = calculate_diffuseness(reshaped_responses[:31250:10])
print(f"Diffuseness: {diffuseness:.4f}")
# print(f"Hamming Distances in all responses : {all_hm_d}")

mean_hm_d = np.mean(all_hm_d)
variance_hm_d = np.var(all_hm_d)


print(np.array(all_hm_d).shape)
print(f"Mean of Hamming Distances: {mean_hm_d:.4f}")
print(f"Variance of Hamming Distances: {variance_hm_d}")

# Create a histogram for all_hm_d

file_name = f"./figures/{df[:12]}_Diffuseness.png"

plt.figure()  # Add this line
plt.hist(all_hm_d[::49], bins=80, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Hamming Distance')
plt.ylabel('Frequency')
plt.title('Diffuseness')
os.makedirs("Figures", exist_ok=True)
plt.savefig(file_name)

# Show the plot
plt.show()