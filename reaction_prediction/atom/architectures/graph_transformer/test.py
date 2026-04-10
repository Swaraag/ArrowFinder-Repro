import csv
from reaction_prediction.atom.utils import label_each_atom
from rdkit import Chem
import torch

def main():
    objs = torch.load("/tmp/smoke.pt", weights_only=False)
    for i, obj in enumerate(objs):
        if obj.num_nodes == 1:
            print(f"Object {i}: num_nodes=1, random_walk={obj.random_walk}, has_nan={obj.random_walk.isnan().any()}")


if __name__ == "__main__":
    main()