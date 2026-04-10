import torch
import json
from reaction_prediction.atom.architectures.graph_transformer.graph_transformer import CustomGPS

def create_test_reactions(test_objs):
    test_reactions = dict()
    for obj in test_objs:
        if test_reactions.get(obj.reaction_id) is None:
            test_reactions[obj.reaction_id] = [obj]
        else:
            test_reactions[obj.reaction_id].append(obj)
    return test_reactions
            
def main(source_test_file_path, sink_test_file_path, source_model_path, sink_model_path, gt_source_hparams_path, gt_sink_hparams_path, topk):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    source_test_objs = torch.load(source_test_file_path, weights_only=False)
    sink_test_objs = torch.load(sink_test_file_path, weights_only=False)
    #print(source_test_objs)

    with open(gt_source_hparams_path, "r") as source_file, open(gt_sink_hparams_path, "r") as sink_file:
        source_hparams = json.load(source_file)
        sink_hparams = json.load(sink_file)

    source_model = CustomGPS(source_hparams).to(device)
    source_model.load_state_dict(torch.load(source_model_path, weights_only=False, map_location=device))
    source_model.eval()

    sink_model = CustomGPS(sink_hparams).to(device)
    sink_model.load_state_dict(torch.load(sink_model_path, weights_only=False, map_location=device))
    sink_model.eval()

    source_test_reactions = create_test_reactions(source_test_objs)
    sink_test_reactions = create_test_reactions(sink_test_objs)
    
    print(len(source_test_reactions))
    num_correct_topk_source = create_topk(source_test_reactions, source_model, topk, device)
    for k, k_accuracy in num_correct_topk_source.items():
        print(f"SOURCE: Top-{k} accuracy is {k_accuracy/len(source_test_reactions)}")

    print(len(sink_test_reactions))
    num_correct_topk_sink = create_topk(sink_test_reactions, sink_model, topk, device)
    for k, k_accuracy in num_correct_topk_sink.items():
        print(f"SINK: Top-{k} accuracy is {k_accuracy/len(sink_test_reactions)}")

def create_topk(test_reactions, model, topk, device):
    num_correct_topk = {k: 0 for k in range(1, topk+1)}
    for i, objs in test_reactions.items():
        max_sym_class_score = {}
        is_reactive = {}

        for obj_index, data_obj in enumerate(objs):
            data_obj = data_obj.to(device)
            with torch.inference_mode():
                model_outputs = model(x=data_obj.x,
                                    edge_index=data_obj.edge_index, 
                                    batch=torch.zeros(len(data_obj.x), dtype=torch.long).to(device), 
                                    edge_attr=data_obj.edge_attr,
                                    random_walk=data_obj.random_walk)
                scores = [torch.sigmoid(output).item() for output in model_outputs]
            for score, sym, y_label in zip(scores, data_obj.sym_class.tolist(), data_obj.y.tolist()):
                # unique key because the same sym class can be two diff atoms in two diff molecules
                key = (obj_index, sym)
                # updating the newest max symmetry class prediction score
                if key not in max_sym_class_score or max_sym_class_score[key] < score:
                    max_sym_class_score[key] = score
                # if the y label was 1 for any of the values in the sym class, it should be correct
                if y_label == 1:
                    is_reactive[key] = True
        correct_pairs = []
        for key, score in max_sym_class_score.items():
            pred_result = False if is_reactive.get(key) is None else True
            correct_pairs.append((score, pred_result))

        # correct atom is going to be based on reaction_real
        correct_pairs_sorted = sorted(correct_pairs, reverse=True, key=lambda x: x[0])
        #print(f"For idx {i}, there are {len(correct_pairs_sorted)} correct pairs sorted preds.")
        for index, score_pair in enumerate(correct_pairs_sorted[:topk]):
            if score_pair[1]:
                for topk_index in range(index+1, topk+1):
                    num_correct_topk[topk_index] += 1
                break
    return num_correct_topk


if __name__ == "__main__":
    source_test_file_path = "output/mc_train_fold0/graph_data/source_unfiltered/test.pt"
    sink_test_file_path = "output/mc_train_fold0/graph_data/sink_unfiltered/test.pt"
    source_model_path = "output/mc_train_fold0/models/graph_transformer_unfiltered/source.pt"
    sink_model_path = "output/mc_train_fold0/models/graph_transformer_unfiltered/sink.pt"
    gt_source_hparams_path = "output/mc_train_fold0/models/graph_transformer_unfiltered/source_plots/config_used.json"
    gt_sink_hparams_path = "output/mc_train_fold0/models/graph_transformer_unfiltered/sink_plots/config_used.json"
    topk = 10
    main(source_test_file_path, sink_test_file_path, source_model_path, sink_model_path, gt_source_hparams_path, gt_sink_hparams_path, topk)
