from tensorflow.keras.models import load_model
from reaction_prediction.eval.generate_predictions import patch_keras_h5_training_config, extract_atom_fv_to_numpy_array
import csv

from reaction_prediction.atom.modules.simple_atom_object import SimpleAtomObject as SAO

from openeye.oechem import OEAssignAromaticFlags, OEAroModelMMFF, OEPerceiveChiral
from rpCHEM.Common.Util import molBySmiles
from rpCHEM.Common.CanonicalAtomMapSmiles import canonicalizeAtomMapSmiString, createCanonicalAtomMapSmiString
from rpCHEM.Common.MolExt import setSingleExplicitHydrogens
from rpCHEM.Common.MolExt import removeNonsenseStereo

def main(source_model_path, sink_model_path, input_file, allid_file, threshold, topk):

    for p in (source_model_path, sink_model_path):
        patch_keras_h5_training_config(p)

    source_model = load_model(source_model_path)
    sink_model = load_model(sink_model_path)
    num_correct_topk_source = {k: 0 for k in range(1, topk+1)}
    num_correct_topk_sink = {k: 0 for k in range(1, topk+1)}

    total_source = 0
    total_sink = 0
    successful_rows = 0

    with open(input_file, 'r') as in_f:
        reader = csv.reader(in_f)
        for i, row in enumerate(reader):
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

                # these lines below practically jus copy what SAO.atomObjFromReactantSmi does to ensure its in the same format
                # as the SAO objects to maximize their comparability
                true_source = molBySmiles(row[2])
                true_sink = molBySmiles(row[3])
                setSingleExplicitHydrogens(true_source)
                setSingleExplicitHydrogens(true_sink)
                OEAssignAromaticFlags(true_source, OEAroModelMMFF)
                OEAssignAromaticFlags(true_sink, OEAroModelMMFF)
                OEPerceiveChiral(true_source)
                removeNonsenseStereo(true_source)
                OEPerceiveChiral(true_sink)
                removeNonsenseStereo(true_sink)
                true_source = createCanonicalAtomMapSmiString(true_source)
                true_source = canonicalizeAtomMapSmiString(true_source)
                true_sink = createCanonicalAtomMapSmiString(true_sink)
                true_sink = canonicalizeAtomMapSmiString(true_sink)

                total_source += len(source_sorted)
                total_sink += len(sink_sorted)

                for index, source in enumerate(source_sorted):
                    # print(source[0].connectedSmiles)
                    # print(canonicalizeAtomMapSmiString(true_source))
                    if source[0].connectedSmiles == true_source:
                        for topk_index in range(index+1, topk+1):
                            num_correct_topk_source[topk_index] += 1
                        break

                for index, sink in enumerate(sink_sorted):
                    if sink[0].connectedSmiles == true_sink:
                        for topk_index in range(index+1, topk+1):
                            num_correct_topk_sink[topk_index] += 1
                        break
                successful_rows += 1
            except Exception as e:
                print(e)

            if i and (i % 100 == 0):
                print(f"{i} reactions processed...")
    return num_correct_topk_source, num_correct_topk_sink, successful_rows

if __name__ == "__main__":
    source_model_path = "output/mc_train_fold0/models/atom/source.h5"
    sink_model_path = "output/mc_train_fold0/models/atom/sink.h5"
    input_file = "data/mc_train_fold0/reformatted/test.txt"
    allid_file = "output/mc_train_fold0/allid/train_val_combined_allids.json"
    threshold = 0.5
    topk = 10
    num_correct_topk_source, num_correct_topk_sink, len_reactions = main(source_model_path, sink_model_path, input_file, allid_file, threshold, topk)
    print(len_reactions)
    for k, k_accuracy in num_correct_topk_source.items():
        print(f"SOURCE: Top-{k} accuracy is {k_accuracy/len_reactions}")

    for k, k_accuracy in num_correct_topk_sink.items():
        print(f"SINK: Top-{k} accuracy is {k_accuracy/len_reactions}")