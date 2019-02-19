import numpy as np
import pandas as pd
import os
print(os.listdir("datasets/employee-reviews"))

data = pd.read_csv('datasets/employee-reviews/employee_reviews.csv')

print(data['overall-ratings'].unique())
print(data['company'].unique())

data["Liked"] = [1 if i >= 3 else 0 for i in data['overall-ratings']] # Simple liked/dislike mechanism for now
liked = data.iloc[:,-1:]
pros = data.iloc[:,6:7]
cons = data.iloc[:,7:8]
pros_cons_result = pd.concat([pros,cons,liked], axis = 1, ignore_index = True)
pros_cons_result.columns = ['Pros', 'Cons', 'Liked']
print(pros_cons_result)
