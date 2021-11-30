# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

# read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

# find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

# remove nan (empty) values by using:
itemset.remove(np.nan)

# To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
# Example:

# Bread Wine Eggs
# 1     0    1
# 0     1    1
# 1     1    1

# To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
# and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
# -->add your python code below

encoded_vals = []
vCounter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for index, row in df.iterrows():
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 7):
        if row[i] == "Bread":
            v[0] = 1
            vCounter[0] += 1
        if row[i] == "Wine":
            v[1] = 1
            vCounter[1] += 1
        if row[i] == "Eggs":
            v[2] = 1
            vCounter[2] += 1
        if row[i] == "Meat":
            v[3] = 1
            vCounter[3] += 1
        if row[i] == "Cheese":
            v[4] = 1
            vCounter[4] += 1
        if row[i] == "Pencil":
            v[5] = 1
            vCounter[5] += 1
        if row[i] == "Diaper":
            v[6] = 1
            vCounter[6] += 1
        if row[i] == "Bagel":
            v[7] = 1
            vCounter[7] += 1
        if row[i] == "Milk":
            v[8] = 1
            vCounter[8] += 1

    labels = {'Bread': v[0], 'Wine': v[1], 'Eggs': v[2],
              'Meat': v[3], 'Cheese': v[4], 'Pencil': v[5],
              'Diaper': v[6], 'Bagel': v[7], 'Milk': v[8]}

    encoded_vals.append(labels)

# adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

# calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# iterate the rules data frame and print the apriori algorithm results by using the following format:

# Meat, Cheese -> Eggs
# Support: 0.21587301587301588
# Confidence: 0.6666666666666666
# Prior: 0.4380952380952381
# Gain in Confidence: 52.17391304347825
# -->add your python code below

for index, row in rules.iterrows():
    supportCount = 0
    for x in row['antecedents']:
        print(str(x) + " ", end="")
    print("-> ", end="")
    for x in row['consequents']:
        print(str(x) + " ", end="")
        if x == "Bread":
            supportCount += vCounter[0]
        if x == "Wine":
            supportCount += vCounter[1]
        if x == "Eggs":
            supportCount += vCounter[2]
        if x == "Meat":
            supportCount += vCounter[3]
        if x == "Cheese":
            supportCount += vCounter[4]
        if x == "Pencil":
            supportCount += vCounter[5]
        if x == "Diaper":
            supportCount += vCounter[6]
        if x == "Bagel":
            supportCount += vCounter[7]
        if x == "Milk":
            supportCount += vCounter[8]

    print()
    print("Support: " + str(row['support']))  # support
    print("Confidence: " + str(row['confidence']))  # confidence

# To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
# use the gain formula provided right after.
# prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
# print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
# -->add your python code below

prior = supportCount / len(encoded_vals)  # -> encoded_vals is the number of transactions
print("Prior: " + str(prior))
print("Gain in Confidence: " + str(100 * (row['confidence'] - prior) / prior))

# Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
