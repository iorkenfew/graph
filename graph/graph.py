import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "C:/Users/zheny/Downloads/Dry_Bean_Dataset.arff"
names = ["Area", 
         "Perimeter", 
         "Major axis length", 
         "Minor axis length", 
         "Aspect ratio", 
         "Eccentricity", 
         "Convex area",
         "Equivalent diameter",
         "Extent",
         "Solidity",
         "Roundness",
         "Compactness",
         "ShapeFactor1",
         "ShapeFactor2",
         "ShapeFactor3",
         "ShapeFactor4",
         "Class"]

dataset = pd.read_csv(url, names=names)
array = dataset.values

X = array[:,0:17]
Y = array[:,16]
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)
j = 0

for i in names:
    print('%-15s' % (i), sep=':', end = '')
print("\n")
for j in range(30):
    for k in range(17):
        print('%-15s' % (str(X_test[j,k])), sep=':', end = '')
    print(Y_test[j], end = "\n")

print(dataset.shape)

sns.set(style="whitegrid", context="notebook")
sns.pairplot(dataset[names], height=1, hue = "Area")
dataset[names[0:17]].hist()
plt.show()

names = ["Area", "Perimeter", "Major axis length", "Minor axis length", "Aspect ratio", "Eccentricity", "Convex area", "Equivalent diameter", "Extent", "Solidity", "Roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4", "Class"]
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

