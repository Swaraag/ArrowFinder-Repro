from graph_transformer import CustomGPS
import json
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC

def calculate_pos_neg_sum(data_objs):
    pos_sum = 0
    neg_sum = 0
    for obj in data_objs:
        pos_sum += obj.y.sum()
        neg_sum += (1 - obj.y).sum()
    return pos_sum, neg_sum

def main():
    config_path = "model_configs/gt_config.json"
    with open(config_path, "r") as f:
        hparams = json.load(f)

    # example file path. reality, you'd have one for sink one for source running as shell scripts
    train_file_path = "output/mc_train_fold0/graph_data/sink/train.pt"
    val_file_path   = "output/mc_train_fold0/graph_data/sink/val.pt"

    train_data_objs = torch.load(train_file_path, weights_only=False)
    val_data_objs   = torch.load(val_file_path, weights_only=False)

    print("Data objects successfully loaded.")
    
    train_pos_neg = calculate_pos_neg_sum(train_data_objs)
    val_pos_neg = calculate_pos_neg_sum(val_data_objs)
    print(f"TRAIN: There are {train_pos_neg[0]} positive targets and {train_pos_neg[1]} negative targets.")
    print(f"VAL: There are {val_pos_neg[0]} positive targets and {val_pos_neg[1]} negative targets.")

    train_data_loader = DataLoader(train_data_objs, batch_size=hparams["batch_size"], shuffle=True)
    val_data_loader   = DataLoader(val_data_objs, batch_size=hparams["batch_size"], shuffle=True)

    gps_model = CustomGPS("source", hparams=hparams)
    # optimizer also has a weight decay parameter to possibly tune. hparams has hparams["weight_decay"] set at default to 0.01
    optimizer = torch.optim.AdamW(gps_model.parameters(), hparams["lr"])

    # pos_weight set to neg_sum / pos_sum (inverse of frequency)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_pos_neg[1]/train_pos_neg[0]]))

    train_losses = []
    val_losses = []
    val_AUROC = []

    for epoch in range(hparams["epochs"]):
        training_loop = tqdm(train_data_loader, total=len(train_data_loader), leave=True)
        running_train_loss = 0.0

        # core loop for feed forward and backprop
        running_train_loss = run_training_loop(training_loop, optimizer, gps_model, bce, epoch, hparams)
        train_losses.append(running_train_loss/len(train_data_loader))
        
        val_auroc_metric = BinaryAUROC(pos_label=1)
        val_loop = tqdm(val_data_loader, total=len(val_data_loader), leave=True)
        # tracking as list since i need average incrementally in run_val_loop
        running_val_loss = []
        # core loop for validation dataset
        running_val_loss = run_val_loop(val_loop, gps_model, running_val_loss, bce, epoch, hparams, val_auroc_metric)
        # update history lists
        val_losses.append(sum(running_val_loss)/len(running_val_loss))
        val_AUROC.append(val_auroc_metric.compute())


def run_val_loop(val_loop, gps_model, running_val_loss, bce, epoch, hparams, val_auroc_metric):
    # switch to .eval() mode 
    gps_model.eval()
    with torch.no_grad():
        for batch in val_loop:
            val_outputs = gps_model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)

            running_val_loss.append(bce(val_outputs, torch.reshape(batch.y, (val_outputs.shape[0], 1))).item())
            val_auroc_metric(val_outputs, torch.reshape(batch.y, (val_outputs.shape[0], 1)))

            val_loop.set_description(f"Train Epoch [{epoch+1}/{hparams['epochs']}]")
            val_loop.set_postfix(val_loss=sum(running_val_loss)/len(running_val_loss))
    return running_val_loss

def run_training_loop(training_loop, optimizer, gps_model, bce, epoch, hparams):
    # switch to .train() mode
    gps_model.train()
    for batch in training_loop:
        optimizer.zero_grad()
        train_outputs = gps_model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
        # shapes are slightly mismatched, so reshape needed to adjust
        train_loss = bce(train_outputs, torch.reshape(batch.y, (train_outputs.shape[0], 1)))
        train_loss.backward()
        optimizer.step()

        training_loop.set_description(f"Val Epoch [{epoch+1}/{hparams['epochs']}]")
        running_train_loss += train_loss.item()
        training_loop.set_postfix(loss=train_loss.item())
    return running_train_loss
    

if __name__ == "__main__":
    main()