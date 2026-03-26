#!/usr/bin/env python3
# encoding: utf-8
"""
Evaluate the ranker given trained Keras models.

Outputs in --output:
  - preds.csv           top-k predictions (ascending score)
  - bad.csv             rows that failed, with exception info

For every input row, exactly one output line is written to preds_ascending.csv.
On failures, a BLANK line is written to preds_ascending.csv, and a row is appended to bad.csv.
"""

import os, sys, csv, h5py, traceback, argparse
import numpy as np

from tensorflow.keras.models import load_model

import rdkit
from rdkit import Chem
from openeye import oechem

from reaction_prediction.atom.modules.path_extractor import PathExtractor as PE
from reaction_prediction.atom.modules.feature_extraction import FeatureExtraction as FE
from reaction_prediction.ranker.modules.feature_extraction import ReactionFeatureExtraction as RFE
from reaction_prediction.ranker.modules.simple_orbpair_object import SimpleOrbPairObject as SOO
from reaction_prediction.atom.modules.simple_atom_object import SimpleAtomObject as SAO

# ---------------------------
# Utilities
# ---------------------------

def patch_keras_h5_training_config(h5_path: str):
    """Fix older TF/Keras models that stored 'learning_rate' in H5 training_config."""
    with h5py.File(h5_path, 'r+') as f:
        if 'training_config' in f.attrs:
            data = f.attrs['training_config']
            if isinstance(data, bytes) and b'learning_rate' in data:
                f.attrs['training_config'] = data.decode().replace("learning_rate", "lr").encode()

def mol_with_hydrogens(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return Chem.MolFromSmiles('')
    m = Chem.AddHs(m)
    return m

def write_rdkit_style(smi: str) -> str:
    m = mol_with_hydrogens(smi)
    return Chem.MolToSmiles(m)

def extract_atom_fv_to_numpy_array(atom_list, allid_file):
    fv_list = []
    pe = PE(3)
    fe = FE(allid_file, pe)
    for atom in atom_list:
        fv, _ = fe.atom_feat_vec(atom)
        fv_list.append(fv)
    return np.array(fv_list)

def extract_single_op_fv_to_list(op, allid_file):
    rfe = RFE(morgan_radi=2, morgan_bits=2048, atom_length=3, allid_file=allid_file)
    return rfe.extract_rxn_rep(op.reactionSmiles, op.srcAtom.connectedSmiles, op.sinkAtom.connectedSmiles)

# ---------------------------
# Core eval
# ---------------------------

def run_eval(
    source_model_path: str,
    sink_model_path: str,
    ranker_model_path: str,
    input_file: str,
    allid_file: str,
    out_dir: str,
    max_orbs: int,
    top_k: int,
    threshold: float,
    ranker_inner_index: int
):
    os.makedirs(out_dir, exist_ok=True)

    # Patch & load models
    for p in (source_model_path, sink_model_path, ranker_model_path):
        patch_keras_h5_training_config(p)

    source_model = load_model(source_model_path)
    sink_model   = load_model(sink_model_path)

    # If your ranker is a Siamese wrapper, this picks its inner ranker by index.
    siamese_model = load_model(ranker_model_path)
    try:
        ranker_model  = siamese_model.layers[ranker_inner_index]
    except Exception:
        # Fallback: if loading a plain ranker model
        ranker_model = siamese_model

    preds_path     = os.path.join(out_dir, "preds_ascending.csv")
    bad_path       = os.path.join(out_dir, "bad.csv")

    num_rows = 0
    num_bad  = 0

    # Open outputs
    with open(preds_path, "w", newline='') as preds_f, \
         open(bad_path, "w", newline='') as bad_f, \
         open(input_file, "r") as in_f:

        preds_writer     = csv.writer(preds_f)
        bad_writer       = csv.writer(bad_f)
        bad_writer.writerow(["index", "reaction_line", "error_type", "error_message"])  # header

        reader = csv.reader(in_f)
        for i, row in enumerate(reader):
            num_rows += 1
            predictions_asc = []

            try:
                # Expect the full reaction string in the first column
                rxn_full = row[0]
                reactants = rxn_full.split(">>")[0]
                rxn = rxn_full.split(" ")[0]
                # arrows part is not used downstream, but keep the parsing in case needed later
                # arrows = rxn_full.split(" ")[1].strip().rstrip(",")

                # Preprocess atoms and features
                atoms = SAO.atomObjFromReactantSmi(reactants)
                atoms_oesmiles = [atom.connectedSmiles for atom in atoms]
                atoms_feature_array = extract_atom_fv_to_numpy_array(atoms_oesmiles, allid_file)

                # Predict source/sink scores
                source_scores = list(source_model.predict(atoms_feature_array, verbose=0))
                sink_scores   = list(sink_model.predict(atoms_feature_array,   verbose=0))

                source_score_dict = dict(zip(atoms, [float(j) for j in source_scores]))
                sink_score_dict   = dict(zip(atoms, [float(j) for j in sink_scores]))

                source_sorted = sorted(source_score_dict.items(), key=lambda x: x[1], reverse=True)
                sink_sorted   = sorted(sink_score_dict.items(),   key=lambda x: x[1], reverse=True)

                source_list = [a for a, s in source_sorted if s > threshold] or [source_sorted[0][0]]
                sink_list   = [a for a, s in sink_sorted   if s > threshold] or [sink_sorted[0][0]]

                # Build orbital pairs and rank
                ops = SOO.orbPairObjectsFromAtoms_bounded(source_list, sink_list, max_orbs=max_orbs)

                #print("source list was ",  source_list)
                #print("sink list was ",    sink_list)

                ops_feature_rows, valid_ops = [], []
                for op in ops:
                    try:
                        fv = extract_single_op_fv_to_list(op, allid_file)
                        ops_feature_rows.append(fv)
                        valid_ops.append(op)
                    except Exception as e:
                        # Skip only this op, keep processing others
                        #print("Error while processing op:", op)
                        #print("rxn was ", rxn_full)
                        #traceback.print_exc()  # prints the full traceback
                        continue

                if not ops_feature_rows:
                    #sys.exit()
                    raise RuntimeError("No valid orbital pairs extracted.")

                ops_scores = list(ranker_model.predict(np.array(ops_feature_rows), verbose=0))
                ops_score_dict = dict(zip(valid_ops, [float(j) for j in ops_scores]))

                ops_sorted_asc  = sorted(ops_score_dict.items(), key=lambda x: x[1])           # low→high

                for op, score in ops_sorted_asc[:top_k]:
                    predictions_asc.append(f"{op.reactionSmiles} {op.arrowCodes}")

                # Write predictions
                preds_writer.writerow(predictions_asc)

            except Exception as e:
                # On any failure: write BLANK lines to preds & preds_reversed,
                # and log the failure to bad.csv with error info
                preds_writer.writerow([])        # blank line
                num_bad += 1

                err_type = type(e).__name__
                # Keep message concise; if you want full traceback, swap to traceback.format_exc()
                err_msg  = str(e)
                # Save the original line (if any)
                rxn_line = row[0] if row else ""
                bad_writer.writerow([i, rxn_line, err_type, err_msg])

            if i and (i % 100 == 0):
                print(f"{i} reactions processed...")

    print("Done.")
    print("Input rows:         ", num_rows)
    print("Failures (bad.csv): ", num_bad)
    print("Outputs in:", out_dir)
    print(" -", preds_path)
    print(" -", bad_path)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ranker models on reaction inputs.")
    # Required (your preferred names)
    p.add_argument("--input",  required=True, help="Path to input reactions CSV/TXT")
    p.add_argument("--output", required=True, help="Output directory for results")
    p.add_argument("--allid",  required=True, help="Path to allids JSON")

    # Required models
    p.add_argument("--source_model", required=True, help="Path to source atom model (.h5)")
    p.add_argument("--sink_model",   required=True, help="Path to sink atom model (.h5)")
    p.add_argument("--ranker_model", required=True, help="Path to Siamese/Ranker model (.h5)")

    # Optional knobs
    p.add_argument("--max_orbs", type=int, default=128, help="Max orbital pairs to consider")
    p.add_argument("--top_k",    type=int, default=10,  help="Top-K predictions to emit")
    p.add_argument("--threshold", type=float, default=0.18, help="Score threshold for source/sink selection")
    p.add_argument("--ranker_inner_index", type=int, default=2,
                   help="Index to extract the shared/inner ranker from a Siamese wrapper")
    return p.parse_args()

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_eval(
        source_model_path=args.source_model,
        sink_model_path=args.sink_model,
        ranker_model_path=args.ranker_model,
        input_file=args.input,
        allid_file=args.allid,
        out_dir=args.output,
        max_orbs=args.max_orbs,
        top_k=args.top_k,
        threshold=args.threshold,
        ranker_inner_index=args.ranker_inner_index
    ))
