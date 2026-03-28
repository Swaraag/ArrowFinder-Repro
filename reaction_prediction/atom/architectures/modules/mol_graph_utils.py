# Utils file for converting raw CSV file into a list of molecule graphs
import csv
from torch_geometric.data import Data

class CSVToGraphs:
    def __init__(self):
        pass
    def reaction_to_graph_data(self, row):
        return row
    def process_csv(self, csv_path):
        # this list is gonna contain both atom information (torch_geometric.data.Data.x) 
        # and atom labels (torch_geometric.data.Data.y)
        all_data_objs = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                all_data_objs.extend(self.reaction_to_graph_data(row))
        return all_data_objs


if __name__ == "__main__":
    mol_to_graph = CSVToGraphs()

    csv_path = "data/mc_train_fold0/reformatted/train.txt"