#Title: Visualization of the Brain Structure in Schizophrenia using Dimension Reduction Analysis
#By: Shaun Chen

import pandas
df = pandas.read_csv('book-schizo.csv').to_numpy() #read csv file and convert to numpy array
# print(df['x'][0])
# print(df.loc[0])
# print(type(df))
print(df[0][1])