import multiprocessing as mp 
import time
import rdkit
import numpy as np
import h5py
from rdkit import Chem
import sys, os
import csv
import argparse

from reaction_prediction.ranker.modules.feature_extraction import ReactionFeatureExtraction as RFE
from reaction_prediction.ranker.utils import *
from reaction_prediction.atom.utils import *
from reaction_prediction.ranker.modules.simple_orbpair_object import SimpleOrbPairObject as SOO
from reaction_prediction.atom.modules.simple_atom_object import SimpleAtomObject as SAO
from rpCHEM.Common.Util import smi_to_unique_smi_fast, smi_to_unique_smi_map
from functools import partial

# Negative sample generators
negative_generators = [
    source_to_non_reactive_inter,
    sink_to_non_reactive_inter,
    non_reactive_pairs_inter,
    non_reactive_pairs_intra,
    source_to_non_reactive_intra,
    sink_to_non_reactive_intra
]

sample_dict = dict(zip(negative_generators, [5 for _ in negative_generators]))

# Global variable to hold allid file (set by argparse in main)
ALLID_FILE = None


def process_reaction(row, ALLID_FILE, MAXORBS):
    try:
        rxn, arrow, source, sink = row
    except Exception:
        return None  

    rxn_fe = RFE(morgan_radi=2, morgan_bits=2048, atom_length=3, allid_file=ALLID_FILE)

    try:
        pos_fv = rxn_fe.extract_rxn_rep(rxn, source, sink)
    except Exception:
        return None

    try:
        reactants, _ = rxn.split(">>")
        atoms = SAO.atomObjFromReactantSmi(reactants)
    except Exception:
        return None

    try:
        all_ops = SOO.orbPairObjectsFromAtoms_bounded(atoms, atoms, max_orbs=MAXORBS, radical=False)
    except Exception:
        return None

    negative_ops = []
    for generator, count in sample_dict.items():
        try:
            negative_ops.extend(generator(all_ops, source, sink, count))
        except Exception:
            continue

    results = []
    for neg in negative_ops:
        neg_rxn = (neg.reactionSmiles, neg.srcAtom.connectedSmiles, neg.sinkAtom.connectedSmiles)
        try:
            neg_fv = rxn_fe.extract_rxn_rep(neg_rxn[0], neg_rxn[1], neg_rxn[2])
            results.append((pos_fv, neg_fv, 1.0))
        except Exception:
            continue

    return results


def main(args):
    global ALLID_FILE
    ALLID_FILE = args.allid
    global MAXORBS
    MAXORBS = args.maxorbs

    positive_features_all = []
    negative_features_all = []
    targets_all = []

    with open(args.input, newline="") as f:
        reader = list(csv.reader(f))

    pool = mp.Pool(processes=mp.cpu_count())
    
    start_time = time.time()
    counter = 0  
    process_reaction_partial = partial(process_reaction, ALLID_FILE=ALLID_FILE, MAXORBS=MAXORBS)
    for res in pool.imap(process_reaction_partial, reader):
        counter += 1
        if counter % 100 == 0:
            elapsed = time.time() - start_time
            print(f"{counter} reactions processed... Elapsed time: {elapsed:.2f} seconds", flush=True)
        if res:
            for pos_fv, neg_fv, targ in res:
                positive_features_all.append(pos_fv)
                negative_features_all.append(neg_fv)
                targets_all.append(targ)
    
    pool.close()
    pool.join()

    positive_features_all, negative_features_all = shuffle_two_lists(
        positive_features_all, negative_features_all
    )

    positive_features_all = np.array(positive_features_all, dtype='float32')
    negative_features_all = np.array(negative_features_all, dtype='float32')
    targets_all = np.array(targets_all, dtype='float32')

    with h5py.File(args.output, 'w', libver='latest') as hf:
        hf.create_dataset('pos_features', data=positive_features_all)
        hf.create_dataset('neg_features', data=negative_features_all)
        hf.create_dataset('targets', data=targets_all)
        print("Data has been written to the HDF5 file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate positive/negative features from reactions"
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output HDF5 file"
    )
    parser.add_argument(
        "--allid", required=True, help="Path to allid JSON file"
    )
    parser.add_argument(
        "--maxorbs", type=int, default=5000, help="Path to allid JSON file"
    )
    args = parser.parse_args()

    main(args)