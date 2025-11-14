import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

#iris_dataset = load_dataset("sasageooo/iris-flower-classification") #loading dataset from hugging face

#iris_dataset['train'].to_csv("iris_dataset.csv") #saving dataset as CSV file

df = pd.read_csv("dataset/iris_dataset.csv")
#print(df.head()) 
#print(df.info()) 
#print(df.describe()) 

plot1 = sns.pairplot(df, hue = "species")
print(plot1)
plt.show()


