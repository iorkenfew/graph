import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "C:/Users/zheny/Downloads/yacht_hydrodynamics.data"
names = [
    "Longitudinal",
    "Prismatic coefficient",
    "Length-displacement",
    "Beam-draught",
    "Length-beam",
    "Froude number",
    "Residuary",
]

dataset = pd.read_csv(url, names=names)
array = dataset.values

X = array[:,0:8]
Y = array[:,6]
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)
j = 0

for i in names:
    print('%-15s' % (i), sep=':', end = '')
print("\n")
for j in range(30):
    for k in range(7):
        print('%-15s' % (str(X_test[j,k])), sep=':', end = '')
    print(Y_test[j], end = "\n")


print(dataset.shape)

sns.set(style="whitegrid", context="notebook")
sns.pairplot(dataset[names], height=1, hue = "Prismatic coefficient")
dataset[names[0:9]].hist()
plt.show()

names = ["Prismatic coefficient", "Beam-draught", "Froude number", "Longitudinal", "Length-displacement"]
cm = np.corrcoef(dataset[names].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(
    cm,
    cbar=False,
    annot=True,
    square=False,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=names,
    xticklabels=names,
)
plt.show()

