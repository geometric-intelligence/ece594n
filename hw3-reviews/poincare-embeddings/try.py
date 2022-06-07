import csv
import os
import pandas as pd
# os.getcwd()+"/wordnet/mammal_closure_noweights.csv"

class PoincareModel():
    def __init__(self, datasetFilePath):
        self.datasetFilePath = datasetFilePath
        # self.keyedVectors = 
        # self.keyedVectorSize =  

    def parse_dataset(self):
        results = pd.read_csv(self.datasetFilePath)
        print(results)
        # with open(self.datasetFilePath) as csvfile:
        #     relationReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #     for row in relationReader:
        #         print(row)
        #         print(', '.join(row))

if __name__ == "__main__":
    test = PoincareModel(os.getcwd()+"/wordnet/mammal_closure_noweights.csv")
    test.parse_dataset()