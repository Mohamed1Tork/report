# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# importing libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

#read data
dataset=pd.read_csv("Wuzzuf_Jobs.csv")
dataset.describe()


#clean duplicates
dataset.sort_values("Title", inplace = True)
dataset.drop_duplicates(subset ="Title",keep = "first", inplace = True)
#dataset.sort_values("Skills", inplace = True)
#dataset.drop_duplicates(subset ="Skills",keep = "first", inplace = True)
#dataset.sort_values("Country", inplace = True)
#dataset.drop_duplicates(subset ="Country",keep = "first", inplace = True)
#dataset.sort_values("YearsExp", inplace = True)
#dataset.drop_duplicates(subset ="YearsExp",keep = "first", inplace = True)
#dataset.sort_values("Level", inplace = True)
#dataset.drop_duplicates(subset ="Level",keep = "first", inplace = True)
#dataset.sort_values("Type", inplace = True)
#dataset.drop_duplicates(subset ="Type",keep = "first", inplace = True)
#dataset.drop_duplicates(keep = "first", inplace = True)


#clean missing
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer = imputer.fit(dataset)
cl_data = imputer.transform(dataset)
print(cl_data)

#company pie chart
x=dataset['Company'].value_counts()
plt.pie(x)

#title bar chart
t=dataset['Title'].value_counts()
titels=list(t.keys())
plt.xlabel("Jobs")
plt.ylabel("No. of titles")
plt.title("The most pouplar jobs ")
plt.bar(titels,t,color="red",width=0.1)

#location bar chart
l=dataset['Location'].value_counts()
locations=list(l.keys())
plt.xlabel("Area")
plt.ylabel("No. of titles")
plt.title("The most pouplar areas ")
plt.bar(locations,l,color="blue",width=0.2)

#important Skills
dataset.sort_values("Skills", inplace = True)
s=dataset["Skills"].value_counts()
print(s.keys())



