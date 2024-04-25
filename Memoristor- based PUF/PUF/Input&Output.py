import pandas as pd
import numpy as np

df = pd.read_csv('result.csv', usecols=[0], header=None)
df_1 = pd.read_csv('result.csv', usecols=[1], header=None)
df_2 = pd.read_csv('result.csv', usecols=[4], header=None)

elements = df
elements_1 = df_1
print(elements)
# print(elements.shape)

ele2 = []
ele3 = []

ele2.append([list(map(int, e[1:-1].split(' '))) for e in elements[0][1:]])
ele3.append([list(map(int, e[1:-1].split(' '))) for e in elements_1[1:][1]])

ele2 = np.array(ele2)
ele2 = ele2.reshape(ele2.shape[1], -1)
print(ele2.shape)

ele3 = np.array(ele3)
ele3 = ele3.reshape(ele3.shape[1], -1)
print(ele3.shape)

df = pd.DataFrame(ele2)
df_1 = pd.DataFrame(ele3)

# Save the DataFrames to CSV files
df.to_csv("input1.csv", index=False)
df_1.to_csv("input2.csv", index=False)
df_2.to_csv("output.csv", index=False)

df3 = pd.read_csv('input1.csv')
df4 = pd.read_csv('input2.csv')


result_df = pd.concat([df3, df4], axis=1)

# Save the result DataFrame to a new CSV file
result_df.to_csv('input.csv', index=False)