import os
import json
import pandas as pd
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# pip install networkx==2.6.3 #because other versions of networkx cause issues
# provide file name manually in file

file_name = 'office.json'
current_directory = os.getcwd()
summer_path = os.path.join(current_directory, r'summer.json')
christmas_path = os.path.join(current_directory, r'christmas.json')

data = []
with open(file_name) as f:
    for l in f:
        data.append(json.loads(l.strip()))

# first row of the list
#print("first row of the data", data[0])

# convert list into pandas dataframe
df = pd.DataFrame.from_dict(data)
print("Number of total reviews: ", len(df))

# 1. convert reviewtime to rank-orderable number (by month)
# MMDYYYY - > MMDDYYYY
t1 = time.time()

for i, row in df.iterrows():
    split_str = row[-1].split()
    daystring = split_str[1].replace(',',"") if len(split_str[1]) == 3 else "0" + split_str[1].replace(',',"") # MMDYYYY - > MMDDYYYY
    new_format = split_str[0] + daystring + split_str[2]
    df.at[i, 'reviewTime'] = new_format

t2 = time.time()
print("reformatting dates took {:.2f}s".format(t2-t1))

# 2. Make different dataframes for different periods of time
df_christmas = [] # Rest december, January, February
df_summer = [] # July, August

for i, row in df.iterrows():
    month_as_int = int(row[-1]) #last column is reviewtime
    row = row.rename(None).fillna(0) #some reviewers are NaN
    if (12251996 <= month_as_int <= 12312018) or (1011996 <= month_as_int <= 2282018): # whatever leap years
        df_christmas.append(row)
    if 6011996 <= month_as_int <= 8312018:
        df_summer.append(row)

t3 = time.time()

print("Created summer and christmas files in {:.2f}s".format(t3-t2))

df_summer = pd.DataFrame(df_summer)
df_christmas = pd.DataFrame(df_christmas)

unique_reviewers = df_summer.reviewerID.unique().tolist()
unique_products = df_summer.asin.unique().tolist()
reviewer_by_product = np.zeros([len(unique_reviewers), len(unique_products)])

for i, row in df_summer.iterrows():
    current_item = row.loc['asin']
    current_reviewer = row.loc['reviewerID']
    reviewer_by_product[unique_reviewers.index(current_reviewer), unique_products.index(current_item)] = 1

co_purchase_matrix = np.zeros([len(unique_products), len(unique_products)])
for row in reviewer_by_product:
    indices = np.where(row == 1)[0]
    for val in indices:
        for val_ in indices:
            if val != val_:
                co_purchase_matrix[val, val_] += 1
                
#unique, counts = np.unique(co_purchase_matrix, return_counts=True)
#for un, cn in zip(unique, counts):
    #print("Value: {} Count: {}".format(un, cn))

G_co_purchase = nx.from_numpy_matrix(co_purchase_matrix)
nx.draw(G_co_purchase)
plt.show()



    




 





