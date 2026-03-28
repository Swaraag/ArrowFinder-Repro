from graph_transformer import CustomGPS
import json
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn

def main():
    config_path = "model_configs/gt_config.json"
    with open(config_path, "r") as f:
        hparams = json.load(f)

    
    # example file path. reality, you'd have all 4 running as shell scripts
    file_path = "output/mc_train_fold0/graph_data/sink/train.pt"

    data_objs = torch.load(file_path)
    pos_sum = 0
    neg_sum = 0
    for obj in data_objs:
        pos_sum += obj.y.sum()
        neg_sum += (1 - obj.y).sum()

    data_loader = DataLoader(data_objs, batch_size=hparams["batch_size"], shuffle=True)
    gps_model = CustomGPS("source", hparams=hparams)
    # optimizer also has a weight decay parameter to possibly tune. hparams has hparams["weight_decay"] set at default to 0.01
    optimizer = torch.optim.AdamW(gps_model.parameters(), hparams["lr"])

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_sum/pos_sum]))
    for epoch in range(hparams["epochs"]):
        print(f"Starting epoch {epoch}/{hparams["epochs"]}")
        for batch in data_loader:
            optimizer.zero_grad()
            outputs = gps_model(batch)
            loss = bce(outputs, batch.y)
            loss.backward()
            optimizer.step()



if __name__ == "__main__":
    main()