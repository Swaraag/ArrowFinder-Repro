import csv
from reaction_prediction.atom.utils import label_each_atom
from rdkit import Chem

def main():
    train_path = "data/mc_train_fold0/reformatted/train.txt"

    with open(train_path, 'r') as f:
        reader = csv.reader(f)

        unique_smi = set()
        count = 0
        for i, row in enumerate(reader):
            if i % 100 == 0:
                print(f"{i} reactions processed.")
            reaction, arrows, source_atom, sink_atom = row
            reactants = reaction.split(">>")[0]
            for mol_smi in reactants.split("."):
                try:
                    atom_smis = label_each_atom(mol_smi)
                except:
                    continue
                for smi in atom_smis:
                    canon_mol = Chem.MolFromSmiles(smi)
                    count += 1
                    unique_smi.add(Chem.MolToSmiles(canon_mol))
        print(len(unique_smi))
        print(f"There were {len(unique_smi)} unique canon smiles, out of {count} total smiles.")
        print(f"Ratio of {len(unique_smi)/count}")


if __name__ == "__main__":
    main()