# src/training.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import torch
import psutil
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.retrieval import RetrievalMRR, RetrievalHitRate
from src.data_utils import KnowledgeGraphDataset, create_dataloader
from src.model import GraphSAGE



def train(model, 
          dataset, 
          dataloader, 
          device, 
          optimizer, 
          epochs):
    
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for step, data in enumerate(dataloader):
            # Data preparation
            compacted_seeds = data.compacted_seeds.T.to(device)
            labels = data.labels.to(device)
            x = data.node_features["feat"].to(device)
            blocks = [b.to(device) for b in data.blocks]

            # Forward
            y = model(blocks, x)
            src_embed = y[compacted_seeds[0].long()]
            dst_embed = y[compacted_seeds[1].long()]
            logits = model.predictor(src_embed * dst_embed).squeeze()
            
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # You can add print(f"Epoch {epoch} loss: {total_loss}") here

@torch.no_grad()
def prediction_model(model, 
                     dataset, 
                     test_dataloader, 
                     device):
    
    model.eval()
    logits_list, labels_list, indexes_list = [], [], []
    for data in test_dataloader:
        index = data.indexes.long().to(device)
        compacted_seeds = data.compacted_seeds.T.to(device)
        label = data.labels.to(device)
        x = data.node_features["feat"].to(device)
        blocks = [b.to(device) for b in data.blocks]

        # Get final embedding from the model
        y = model(blocks, x)

        # Compute prediction
        src = compacted_seeds[0].long()
        dst = compacted_seeds[1].long()
        logit = model.predictor(y[src] * y[dst]).squeeze().detach()
        logits_list.append(logit)
        labels_list.append(label)
        indexes_list.append(index)

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0), torch.cat(indexes_list, dim=0)



def get_current_ram_usage():
    """Returns the RAM usage of the current process in gigabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def run_experiment(dataset_name,
                   file_path,
                   fanouts_train,
                   fanouts_test,
                   batch_size_train,
                   batch_size_test,
                   neg_ratio_train,
                   neg_ratio_hitk,
                   neg_ratio_mrr,
                   metric,
                   learning_rate,
                   epochs,
                   result=None):
    
    if result is None:
        result = {}

    # Record memory status at the start
    mem_start = get_current_ram_usage()
   
    dataset = KnowledgeGraphDataset(dataset_name, file_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Training setup and training
    start_train_time = time.time()
    train_dataloader = create_dataloader( dataset=dataset, 
                                          device=device, 
                                          is_train=True, 
                                          batch_size=batch_size_train, 
                                          fanouts=fanouts_train, 
                                          negative_ratio=neg_ratio_train)

    in_size = dataset.feature.size("node", None, "feat")[0]
    model = GraphSAGE(in_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train(model, dataset, train_dataloader, device, optimizer, epochs)
    end_train_time = time.time()
    
    # Testing and evaluation
    start_test_time = time.time()
    negative_ratio = neg_ratio_hitk if metric == "Hit@10" else neg_ratio_mrr
    test_dataloader = create_dataloader( dataset=dataset, 
                                         device=device, 
                                         is_train=False, 
                                         batch_size=batch_size_test, 
                                         fanouts=fanouts_test, 
                                         negative_ratio=negative_ratio )

    logits, labels, indexes = prediction_model(model, dataset, test_dataloader, device)
    
    # Metric calculations
    auc = roc_auc_score(labels.detach().cpu().numpy(), logits.detach().cpu().numpy())
    mrr_metric = RetrievalMRR()
    mrr_score = mrr_metric(logits, labels, indexes).item()
    
    auc_pr = average_precision_score(labels.detach().cpu().numpy(), logits.detach().cpu().numpy()) if metric == "AUC-PR" else 0.0
    hit_score = 0.0
    if metric == "Hit@10":
        hit_rate = RetrievalHitRate(top_k=10)
        hit_score = hit_rate(logits, labels, indexes).item()

    end_test_time = time.time()
    mem_end = get_current_ram_usage()
    
    # Calculate average RAM usage during this process
    avg_ram_usage = (mem_start + mem_end) / 2
    result[dataset_name] = {
        "auc": auc,
        "auc_pr": auc_pr,
        "mrr": mrr_score,
        "hit_rate_10": hit_score,
        "train_time": end_train_time - start_train_time,
        "test_time": end_test_time - start_test_time,
        "ram_gb": avg_ram_usage # Store RAM usage 
    }
    
    return result