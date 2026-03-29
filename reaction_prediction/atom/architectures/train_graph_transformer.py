from graph_transformer import CustomGPS
import json
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC
import os
import argparse

def calculate_pos_neg_sum(data_objs):
    pos_sum = 0
    neg_sum = 0
    for obj in data_objs:
        pos_sum += obj.y.sum()
        neg_sum += (1 - obj.y).sum()
    return pos_sum, neg_sum

def create_checkpoint_callback(best_monitor, cur_monitor, output_file_path, gps_model):
    # best_monitor is a variable for tracking the current best monitor value
    if cur_monitor < best_monitor:
        best_monitor = cur_monitor
        torch.save(gps_model.state_dict(), output_file_path)
    return best_monitor


def run_val_loop(val_loop, gps_model, running_val_loss, bce, epoch, hparams, val_AUROC_metric, device):
    # switch to .eval() mode 
    gps_model.eval()
    with torch.no_grad():
        for batch in val_loop:
            batch = batch.to(device)
            val_outputs = gps_model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)

            running_val_loss.append(bce(val_outputs, torch.reshape(batch.y, (val_outputs.shape[0], 1))).item())
            val_AUROC_metric(torch.squeeze(val_outputs), batch.y.long())
            val_AUROC = val_AUROC_metric.compute().item()

            val_loop.set_description(f"Epoch (validation) [{epoch+1}/{hparams['epochs']}]")
            val_loop.set_postfix(val_loss=sum(running_val_loss)/len(running_val_loss),
                                 val_AUROC=val_AUROC)
    return running_val_loss, val_AUROC

def run_training_loop(training_loop, optimizer, gps_model, bce, epoch, hparams, device):
    # switch to .train() mode
    running_train_loss = 0.0
    gps_model.train()
    for batch in training_loop:
        batch = batch.to(device)
        optimizer.zero_grad()
        train_outputs = gps_model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
        # shapes are slightly mismatched, so reshape needed to adjust
        train_loss = bce(train_outputs, torch.reshape(batch.y, (train_outputs.shape[0], 1)))
        train_loss.backward()
        optimizer.step()

        training_loop.set_description(f"Epoch (train) [{epoch+1}/{hparams['epochs']}]")
        running_train_loss += train_loss.item()
        training_loop.set_postfix(loss=train_loss.item())
    return running_train_loss
    
def main(model_config_file_path, train_file_path, val_file_path, history_output_dir, model_output_file_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    with open(model_config_file_path, "r") as f:
        hparams = json.load(f)

    train_data_objs = torch.load(train_file_path, weights_only=False)
    val_data_objs   = torch.load(val_file_path, weights_only=False)

    print(f"{len(train_data_objs) + len(val_data_objs)} data objects successfully loaded.")
    
    train_pos_neg = calculate_pos_neg_sum(train_data_objs)
    val_pos_neg = calculate_pos_neg_sum(val_data_objs)
    print(f"TRAIN: There are {train_pos_neg[0]} positive targets and {train_pos_neg[1]} negative targets.")
    print(f"VAL: There are {val_pos_neg[0]} positive targets and {val_pos_neg[1]} negative targets.")

    train_data_loader = DataLoader(train_data_objs, batch_size=hparams["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_data_loader   = DataLoader(val_data_objs, batch_size=hparams["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    gps_model = CustomGPS(hparams=hparams).to(device)
    # optimizer also has a weight decay parameter to possibly tune. hparams has hparams["weight_decay"] set at default to 0.01
    optimizer = torch.optim.AdamW(gps_model.parameters(), hparams["lr"])

    # pos_weight set to neg_sum / pos_sum (inverse of frequency)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_pos_neg[1]/train_pos_neg[0]])).to(device)

    train_losses = []
    val_losses = []
    val_AUROCs = []
    early_stopping_patience = hparams["patience"]
    best_val_loss = float('inf')

    # EPOCH TRAINING LOOP
    for epoch in range(hparams["epochs"]):
        training_loop = tqdm(train_data_loader, total=len(train_data_loader), leave=True)

        # core loop for feed forward and backprop
        running_train_loss = run_training_loop(training_loop, optimizer, gps_model, bce, epoch, hparams, device)
        train_losses.append(running_train_loss/len(train_data_loader))
        
        val_AUROC_metric = BinaryAUROC()
        val_loop = tqdm(val_data_loader, total=len(val_data_loader), leave=True)
        # tracking as list since i need average incrementally in run_val_loop
        running_val_loss = []
        # core loop for validation dataset
        running_val_loss, val_AUROC = run_val_loop(val_loop, gps_model, running_val_loss, bce, epoch, hparams, val_AUROC_metric, device)
        # update history lists
        val_loss = sum(running_val_loss)/len(running_val_loss)
        val_losses.append(val_loss)
        val_AUROCs.append(val_AUROC)

        # EARLY STOPPING CALLBACK
        if val_loss < best_val_loss:
            # best val loss isn't getting updated inside here because checkpoint callback still needs to run
            early_stopping_patience = hparams["patience"]
        elif early_stopping_patience <= 0:
            print(f"The hyperparam patience of {hparams['patience']} has run out. Training has ended.")
            break
        else:
            early_stopping_patience -= 1

        # OMDEL CHECKPOINT CALLBACK
        # function saves the checkpoint weights, and then updates best_monitor
        best_val_loss = create_checkpoint_callback(best_val_loss, val_loss, model_output_file_path, gps_model)
    
    with open(os.path.join(history_output_dir, "history.json"), "w") as file:
        json.dump({"train_loss": train_losses, "val_loss": val_losses, "val_auroc": val_AUROCs}, file, indent=2)
    with open(os.path.join(history_output_dir, "config_used.json"), "w") as file:
        json.dump(hparams, file, indent=2)

def parse_args():
    # model_config_file_path = "model_configs/gt_config.json"
    # train_file_path = "output/mc_train_fold0/graph_data/sink/train.pt"
    # val_file_path   = "output/mc_train_fold0/graph_data/sink/val.pt"
    # history_output_dir = "output/mc_train_fold0/models/graph_transformer/sink_plots"
    # model_output_file_path = "output/mc_train_fold0/models/graph_transformer/sink.pt"
    parser = argparse.ArgumentParser(
        description="Train a custom Graph Transformer on data objects (stored in .pt) containing atom graphs."
    )
    parser.add_argument(
        "--model_config_file_path", "-mod_config",
        required=True,
        help="Path to model configs that will be loaded as the model's hyperparameters.",
    )
    parser.add_argument(
        "--train_file_path", "-train",
        required=True,
        help="Path to an input file containing the train portion of the split.",
    )
    parser.add_argument(
        "--val_file_path", "-val",
        required=True,
        help="Path to an input file containing the validation portion of the split.",
    )
    parser.add_argument(
        "--history_output_dir", "-history",
        required=True,
        help="Path to where history (val_loss, accuracy,, and AUROC) is saved as a .json.",
    )
    parser.add_argument(
        "--model_output_file_path", "-model",
        required=True,
        help="Path to where model checkpoints and final model state dict are saved.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.model_config_file_path, args.train_file_path, args.val_file_path, 
         args.history_output_dir, args.model_output_file_path)
    print("Files have been successfully saved.")