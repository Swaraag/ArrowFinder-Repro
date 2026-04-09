# ArrowFinder-Repro: Full Project Context

## Project Overview
This is a fork of ArrowFinder (rjmille3/ArrowFinder), a two-stage pipeline for polar organic reaction mechanism prediction, described in a JACS 2025 paper by Miller et al. The active repo is Swaraag/ArrowFinder-Repro on the `experiments` branch.

**Researcher:** Swaraag Sistla, undergraduate researcher in Prof. Pierre Baldi's lab at UC Irvine, supervised by PhD student Ryan Miller.

## Goal of Experiment

Ryan Miller (supervisor) asked Swaraag to investigate alternative architectures to the MLP for Stage 1 (source/sink atom prediction) to see if prediction accuracy can be improved. The MLP baseline achieves top-1 = 55.6% and top-10 = 83.9% on the end-to-end two-stage pipeline.

Swaraag selected Graph Transformers (GPS architecture) as the replacement after surveying 8 GNN architectures. The hypothesis is:

- **Hypothesis 1:** The GT atom models achieve higher validation accuracy (measured by top-k accuracy on isolated atom prediction) than the MLP baseline atom models.
- **Hypothesis 2:** If H1 holds, this translates to higher end-to-end top-k accuracy on the two-stage pipeline. If H1 holds but H2 fails, the bottleneck is in OrbChain or the Siamese ranker, not the atom models.

The experiment involves: (1) implementing a custom GPS Graph Transformer in PyTorch/PyG, (2) writing a data pipeline to convert reaction SMILES to per-molecule homogeneous graphs, (3) training source and sink models, (4) integrating the GT into the full two-stage inference pipeline, and (5) evaluating atom-level and end-to-end accuracy against the MLP baseline.

**Current status:** H1 appears false based on current eval numbers (GT source top-1 ~55% vs MLP ~86%), but the validity of this comparison is in question because the two eval scripts were written independently and may not be measuring the same thing. Resolving whether the gap is real or an artifact of eval differences is the immediate priority before reporting results to Ryan or deciding on next steps (e.g. hyperparameter tuning, architecture changes, or abandoning the GT approach).

**Why this matters:** Ryan specifically asked for atom-level top-k statistics. There was no pre-existing script to compute these — both eval scripts were written during this experiment. If either eval is flawed, the conclusion about whether the GT is better or worse than the MLP would be wrong, which would mislead the research direction.

---

## Pipeline Overview

### Stage 1: Atom Models
Given a reaction's reactant SMILES, score every atom for likelihood of being the source (electron donor) or sink (electron acceptor).

**MLP baseline:** Each atom is represented as a ~1500-dimensional path-based feature vector (PathExtractor length=3, FeatSel top 1500). A 6-layer MLP with hidden_dim=540 predicts a score per atom. Symmetric atoms are deduplicated via SimpleAtomObject (SAO).

**GT experiment:** Each reactant molecule is converted to a PyG homogeneous graph (one graph per molecule). A GPS Graph Transformer (GPSConv with GINEConv local conv + RWSE positional encoding, walk_length=20) predicts a score per node (atom). 47 node features: 5 scalar atom properties + 16-dim atom type OHE + 6-dim hybridization OHE + 20-dim RWSE.

### Stage 2: Ranker
Top-scoring source/sink atoms are used to generate orbital pairs (via OrbChain/OpenEye). A Siamese ranker scores each orbital pair. The pipeline outputs the top-k most likely reaction mechanisms.

---

## Key Files

### Data Generation
- `reaction_prediction/atom/architectures/graph_transformer/make_atom_graph_data.py` — converts raw CSV to PyG Data objects (one graph per reactant molecule). Stores `reaction_id` (CSV row index), `x`, `y`, `edge_index`, `edge_attr`, `random_walk`.
- `reaction_prediction/atom/modules/feature_extraction.py` — MLP feature extraction (path-based)
- `reaction_prediction/atom/modules/simple_atom_object.py` — SAO class; `atomObjFromReactantSmi` deduplicates symmetric atoms via OpenEye symmetry perception

### Training
- `reaction_prediction/atom/architectures/graph_transformer/train_graph_transformer.py` — GT training script (PyTorch, AdamW, BCEWithLogitsLoss with dynamic pos_weight, ReduceLROnPlateau)
- `reaction_prediction/atom/modules/train_atom_model.py` — MLP training script (Keras/TF, balanced class weights)

### Inference
- `reaction_prediction/atom/architectures/graph_transformer/gt_generate_predictions.py` — GT end-to-end inference (builds graphs on-the-fly from reactant SMILES, maps scores back to SAO atoms via RDKit canonicalization)
- `reaction_prediction/eval/generate_predictions.py` — MLP end-to-end inference

### Evaluation
- `reaction_prediction/atom/architectures/graph_transformer/gt_eval_atom_models.py` — GT atom model top-k eval (uses test .pt files)
- `reaction_prediction/eval/eval_atom_model.py` — MLP atom model top-k eval (uses reformatted test CSV)
- `reaction_prediction/eval/compute_topk_accuracy.py` — end-to-end top-k eval (compares predicted product SMILES to ground truth)

### Model Architecture
- `reaction_prediction/atom/architectures/graph_transformer/graph_transformer.py` — CustomGPS class (GPSConv, GINEConv, RWSE)

---

## Data Formats

### Input CSV formats
Three formats exist for the test set:
- `data/mc_train_fold0/raw/test.txt` — `reaction_smiles arrows` (space separated, one quoted field)
- `data/mc_train_fold0/with_quotes/test.txt` — same but with outer quotes
- `data/mc_train_fold0/reformatted/test.txt` — 4 columns: `reaction_smiles, arrows, source_smiles, sink_smiles`

The reformatted format is used for MLP atom eval. The with_quotes format is used for end-to-end MLP inference.

### GT test .pt files
- `output/mc_train_fold0/graph_data/source_unfiltered/test.pt` — list of PyG Data objects for source model
- `output/mc_train_fold0/graph_data/sink_unfiltered/test.pt` — list of PyG Data objects for sink model

Each Data object contains: `x` (node features, 47-dim), `y` (binary labels per atom), `edge_index`, `edge_attr` (3-dim: bond type, ring, aromaticity), `random_walk` (20-dim RWSE), `reaction_id` (int, CSV row index), `num_nodes`.

### MLP training data
- `output/mc_train_fold0/features/source/train.hdf5` etc. — HDF5 files with shape (273042, 6358) for source train

---

## Training Details

### GT Models (current best, unfiltered)
**Source model:**
- Training data: 24672 graphs, 7537 positive / 427534 negative atoms
- Best val loss: 0.1737, val AUROC: 0.994
- Early stopped at epoch 82/200

**Sink model:**
- Training data: 24678 graphs, 4405 positive / 430870 negative atoms  
- Best val loss: 0.1131, val AUROC: 0.997
- Early stopped at epoch 76/200

**Final GT config:**
```json
{
    "num_hidden_layers": 2,
    "hidden_dim": 64,
    "dropout": 0.1,
    "lr": 5e-5,
    "weight_decay": 0.03,
    "epochs": 200,
    "batch_size": 64,
    "walk_length": 20,
    "heads": 4,
    "num_node_features": 47,
    "num_edge_features": 3,
    "patience": 20
}
```

**pos_weight** is computed dynamically as `neg_sum / pos_sum` (~57:1 for source, ~98:1 for sink).

### Previous GT Models (filtered, y.sum()==0 removed from training)
**Source:** best val loss 0.1325, AUROC 0.997 (epoch 146)
**Sink:** best val loss 0.1199, AUROC 0.997 (epoch 125)
These were trained on ~7.2k source / ~3.6k sink graphs (much smaller, only molecules containing the true source/sink).

### MLP Models
- Source train: (273042, 6358) features, 10388 pos / 262654 neg
- 6-layer MLP, hidden_dim=540, balanced class weights
- Trained with Keras/TF

---

## Critical Data Pipeline History

### The y.sum()==0 filter issue
Originally, `make_atom_graph_data.py` filtered out molecules where `y.sum()==0` (no true source/sink atom) during both training AND test data generation. This caused:
- Training: model never saw negative-only molecules, so it had no concept of "this molecule doesn't contain the source/sink"
- Test: artificially small test sets (~460 source, ~900 sink graphs) with inflated accuracy (~86% source top-1, ~90% sink top-1)

Fix: filter is removed so all molecules per reaction are included.

After fix: test sets grew to ~2700 graphs each, accuracy dropped to realistic values.

### Current test set sizes
- Source: 2686 graphs across 1336 reactions
- Sink: 2697 graphs across 1336 reactions (1337 for MLP since one reaction fails differently)

---

## Current Eval Results

### GT atom model eval (gt_eval_atom_models.py, unfiltered test .pt)
```
SOURCE: Top-1: 55.2%, Top-5: 68.9%, Top-10: 69.6%
SINK:   Top-1: 31.8%, Top-5: 35.0%, Top-10: 35.1%
```

### MLP atom model eval (eval_atom_model.py, reformatted test CSV)
```
SOURCE: Top-1: 85.9%, Top-5: 98.6%, Top-10: 98.7%
SINK:   Top-1: 89.8%, Top-5: 99.3%, Top-10: 99.4%
```

### End-to-end pipeline eval (compute_topk_accuracy.py)
- MLP baseline: top-1 = 55.6%, top-10 = 83.9% (1367 rows)
- GT pipeline: top-1 = 24.5%, top-10 = 34.0% (1337 rows)

---

## Known Issues and Open Questions

### Eval comparison validity (PRIMARY INVESTIGATION TARGET)
The two atom model evals may not be measuring the same thing:

1. **Candidate pool size:** GT eval ranks all atoms across all molecules in the reaction (including H atoms, all heavy atoms). MLP eval ranks SAO atoms, which are deduplicated by symmetry class via OpenEye. GT has ~1.5-2x more candidates per reaction than MLP.
   - GT candidate counts for first 10 reactions: 46, 20, 25, 25, 55, 26, 55, 87, 64, 55
   - MLP candidate counts for first 10 reactions: 25, 10, 17, 11, 40, 16, 31, 53, 39, 42

2. **Ground truth matching:** GT eval uses `data_obj.y.argmax()` to find correct atom index (only valid when `y.sum() > 0`). MLP eval uses `source[0].connectedNonMappedSmiles == clearAtomMapsSmiStr(true_source)` — OpenEye canonicalization on both sides. Is this matching correct and complete?

3. **Hydrogen inclusion:** GT graphs include explicit hydrogens (via `mol_with_hydrogens`). MLP SAO atoms include hydrogens as separate candidates. Both include H, but GT includes them as graph nodes while MLP includes them as separate SAO objects.

4. **The MLP numbers seem suspiciously high** (85-99%). Ryan's original paper never reported atom-level top-k separately — only end-to-end. If atom models are truly 85-99% accurate, the end-to-end accuracy of 55.6% seems surprisingly low. This suggests either: (a) the MLP eval is inflated somehow, (b) the bottleneck is in the ranker/OrbChain, or (c) both.

### GT create_y correctness
`create_y` in `make_atom_graph_data.py` compares `Chem.MolToSmiles(mol_with_atom_tagged_:1)` to `canon_special_atom` (built from the source/sink SMILES in the CSV). This was verified to match correctly for full molecules (not just neighborhoods) — the reformatted CSV stores the full connected component SMILES of the source/sink molecule, not just the atom neighborhood.

### Known bugs already fixed
1. RWSE computed but not used (fixed: concatenated to x in forward method)
2. `min_lr=1**(-6)` evaluated to 1 not 1e-6 (fixed)
3. `weight_decay` not passed to AdamW (fixed)
4. Per-atom graph construction (fixed: now per-molecule)
5. OpenEye/RDKit SMILES canonicalization mismatch in inference (fixed)
6. Atom map numbers polluting canonical comparison in inference (fixed)

---

## Hypotheses About the Accuracy Gap

1. **Feature richness:** MLP uses 1500 path-based features encoding specific 3-bond chemical substructures. GT uses 47 generic atom property features + RWSE topology encoding. The path features may be more informative for reactivity prediction.

2. **Eval non-comparability:** GT ranks ~2x more candidates per reaction. Even a perfect model would have lower top-1 accuracy with more candidates.

3. **Training distribution mismatch:** GT was trained on unfiltered data (all molecules including non-source/sink ones), but class imbalance is severe (~57:1 and ~98:1). pos_weight scaling may not be sufficient.

4. **Architecture genuinely inferior for this task:** At this dataset scale (~7k reactions), simpler models may outperform graph transformers.

---

## Environment
- Python 3.9 (local), 3.12 (Colab)
- PyTorch, PyTorch Geometric (GPSConv, GINEConv, AddRandomWalkPE)
- TensorFlow/Keras (MLP)
- RDKit, OpenEye/OrbChain
- Google Colab Pro (A100/V100 for training)
- Local: M3 MacBook Air

## Repo
- GitHub: Swaraag/ArrowFinder-Repro, branch: `experiments`
- Data: Google Drive at MyDrive/BaldiLab/ArrowFinder-Experiments/graph_data/