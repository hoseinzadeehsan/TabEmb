import copy
import os
import time
import warnings
from os.path import join

import configargparse
import dgl
import lightning as L
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset_llm import GroupedTableDataset
from model_llm import GAT_Multitask as GAT, GCN, GGNN
from util import get_base_dir

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp\\.autocast.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*autocast_mode\\._cast.*deprecated.*",
    category=FutureWarning,
)


def load_data(data_type, args):
    data_task = "cpa" if task == "tta" else task
    filename = (
        f"{data_task}_{args.data_name}_{data_type}_{args.row_examples}_"
        f"{args.model_name.split('/')[-1]}_{args.quantized}_{args.mean_pooling}_"
        f"{args.layer_offset}_{args.layer_pooling}.pkl"
    )
    filename = join(base_dir, "data", "representation", "webtables", "cta", filename)
    print(f"Embedding already exist {filename}")
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data


def collate(graphs_data):
    graphs, labels = [], []
    for item in graphs_data:
        graphs.append(item[0])
        labels.append(item[1])
    graphs = dgl.batch(graphs)
    labels = torch.stack(labels)
    return graphs, labels


def create_dataloaders(df, text_column, label_column, batch_size):
    L.seed_everything(seed=252)
    data_dataset = GroupedTableDataset(
        args,
        texts=list(df[text_column]),
        targets=list(df[label_column]),
        table_ids=list(df["table_id"]),
        max_columns=args.max_col,
    )
    data_dataloader = DataLoader(
        dataset=data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8,
    )
    return data_dataloader


def shutdown_dataloader(dataloader):
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    dataloader._iterator = None


def train_model_multitask(
    model,
    optimizer,
    lr_scheduler,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    device,
    num_epochs,
    accumulation_steps,
):
    scaler = GradScaler()
    criteria = 0
    saved_model_state = None
    saved_epoch = None

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        node_loss = 0
        edge_loss = 0
        graph_loss = 0

        for batch_idx, graph_data in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            graph_batch, graph_labels = graph_data
            graph_batch = graph_batch.to(device)
            graph_labels = graph_labels.to(device)

            targets = {}
            if args.λ_node:
                targets["node_targets"] = graph_batch.ndata["label"].view(-1)
            if args.λ_edge:
                targets["edge_targets"] = graph_batch.edata["label"].view(-1)
            if args.λ_graph:
                targets["graph_target"] = graph_labels

            with autocast("cuda"):
                logits = model(graph_batch, graph_batch.ndata["feat"])

                loss = 0
                _node_loss = 0
                _edge_loss = 0
                _graph_loss = 0
                if args.λ_edge:
                    edge_mask = graph_batch.edata["mask"]

                if args.λ_node:
                    _node_loss = F.cross_entropy(
                        logits["node_logits"], targets["node_targets"]
                    )
                    loss += args.λ_node * _node_loss
                if args.λ_edge:
                    _edge_loss = F.cross_entropy(
                        logits["edge_logits"][edge_mask],
                        targets["edge_targets"][edge_mask],
                    )
                    loss += args.λ_edge * _edge_loss
                if args.λ_graph:
                    _graph_loss = F.cross_entropy(
                        logits["graph_logits"], targets["graph_target"]
                    )
                    loss += args.λ_graph * _graph_loss

            total_loss += loss.item()
            if args.λ_node:
                node_loss += _node_loss.item()
            if args.λ_edge:
                edge_loss += _edge_loss.item()
            if args.λ_graph:
                graph_loss += _graph_loss.item()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_loss /= len(train_dataloader)
        valid_results = calc_accuracy_nodes_edges(model, device, valid_dataloader)
        print(
            f"Epoch: {epoch + 1} / {num_epochs}"
            f"| Valid Total Loss: {valid_results['total_loss']:.4f}"
        )

        if valid_results[component + "_micro_f1"] > criteria:
            criteria = valid_results[component + "_micro_f1"]
            saved_model_state = copy.deepcopy(model.state_dict())
            saved_epoch = epoch
            print("Saving best model in Epoch", saved_epoch + 1)

            test_results = calc_accuracy_nodes_edges(model, device, test_dataloader)
            msg_parts = []
            if args.λ_node:
                msg_parts.append(f"Node {test_results['node_micro_f1']:.4f}")
            if args.λ_edge:
                msg_parts.append(f"Edge {test_results['edge_micro_f1']:.4f}")
            if args.λ_graph:
                msg_parts.append(f"Graph {test_results['graph_micro_f1']:.4f}")
            msg = "| Test Micro F1: " + "\t\t".join(msg_parts)
            print(
                "Saving best model state ("
                + args.model_name
                + ") in Epoch"
                + str(saved_epoch + 1)
                + "\t"
                + msg
            )
            print("Test Performance:")

            if args.λ_node:
                print(
                    f"NODE >> "
                    f"| Test Loss: {test_results['node_loss']:.4f}"
                    f"| Test Micro F1: {test_results['node_micro_f1']:.4f}"
                )
            if args.λ_edge:
                print(
                    f"EDGE >> "
                    f"| Test Loss: {test_results['edge_loss']:.4f}"
                    f"| Test Micro F1: {test_results['edge_micro_f1']:.4f}"
                )
            if args.λ_graph:
                print(
                    f"GRPH >> "
                    f"| Test Loss: {test_results['graph_loss']:.4f}"
                    f"| Test Micro F1: {test_results['graph_micro_f1']:.4f}"
                )

    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"Total training time: {training_time:.2f} min")

    return saved_model_state


def calc_accuracy_nodes_edges(model, device, dataloader):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        node_loss = 0
        edge_loss = 0
        graph_loss = 0
        node_pred_scores = []
        node_actual_scores = []
        edge_pred_scores = []
        edge_actual_scores = []
        graph_pred_scores = []
        graph_actual_scores = []
        for graph_data in dataloader:
            graph_batch, graph_labels = graph_data
            graph_batch = graph_batch.to(device)
            graph_labels = graph_labels.to(device)
            if args.λ_node:
                node_targets = graph_batch.ndata["label"].view(-1)
            if args.λ_edge:
                edge_targets = graph_batch.edata["label"].view(-1)
                edge_mask = graph_batch.edata["mask"]

            with autocast("cuda"):
                logits = model(graph_batch, graph_batch.ndata["feat"])

                _node_loss = 0
                _edge_loss = 0
                _graph_loss = 0
                if args.λ_node:
                    _node_loss = F.cross_entropy(logits["node_logits"], node_targets)
                    node_loss += _node_loss.item()
                    _, pred = torch.max(logits["node_logits"], dim=1)
                    node_pred_scores.extend(pred.cpu().numpy())
                    node_actual_scores.extend(node_targets.cpu().numpy())
                if args.λ_edge:
                    _edge_loss = F.cross_entropy(
                        logits["edge_logits"][edge_mask], edge_targets[edge_mask]
                    )
                    edge_loss += _edge_loss.item()
                    _, pred = torch.max(logits["edge_logits"], dim=1)
                    edge_pred_scores.extend(pred[edge_mask].cpu().numpy())
                    edge_actual_scores.extend(edge_targets[edge_mask].cpu().numpy())
                if args.λ_graph:
                    _graph_loss = F.cross_entropy(
                        logits["graph_logits"], graph_labels.to(device)
                    )
                    graph_loss += _graph_loss.item()
                    _, pred = torch.max(logits["graph_logits"], dim=1)
                    graph_pred_scores.extend(pred.cpu().numpy())
                    graph_actual_scores.extend(graph_labels.cpu().numpy())

                loss = (
                    args.λ_node * _node_loss
                    + args.λ_edge * _edge_loss
                    + args.λ_graph * _graph_loss
                )
                total_loss += loss.item()

        node_accuracy = 0
        node_micro_f1 = 0
        if args.λ_node:
            node_accuracy = accuracy_score(node_actual_scores, node_pred_scores)
            node_micro_f1 = f1_score(node_actual_scores, node_pred_scores, average="micro")

        edge_accuracy = 0
        edge_micro_f1 = 0
        if args.λ_edge:
            edge_accuracy = accuracy_score(edge_actual_scores, edge_pred_scores)
            edge_micro_f1 = f1_score(edge_actual_scores, edge_pred_scores, average="micro")

        graph_accuracy = 0
        graph_micro_f1 = 0
        if args.λ_graph:
            graph_accuracy = accuracy_score(graph_actual_scores, graph_pred_scores)
            graph_micro_f1 = f1_score(graph_actual_scores, graph_pred_scores, average="micro")

        return {
            "node_acc": node_accuracy,
            "node_micro_f1": node_micro_f1,
            "edge_acc": edge_accuracy,
            "edge_micro_f1": edge_micro_f1,
            "graph_acc": graph_accuracy,
            "graph_micro_f1": graph_micro_f1,
            "total_loss": total_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "graph_loss": graph_loss,
        }


parser = configargparse.ArgParser()
parser.add_argument("--gpu", type=str, default="cuda:0", help="Which GPU to use")
parser.add_argument(
    "--row_examples",
    type=int,
    default=25,
    help="batch size used for training, validation and test",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="batch size used for training, validation and test",
)
parser.add_argument(
    "--accumulation_steps",
    type=int,
    default=1,
    help="batch size used for training, validation and test",
)
parser.add_argument(
    "--data-name",
    type=str,
    default="webtables",
    help="for loading dataset",
)
parser.add_argument("--mode", type=str, default="train", help="train or eval")
parser.add_argument(
    "--model-name",
    type=str,
    default="mistral",
    help="name of the llm",
)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--diff_lr", type=float, default=0.00001, help="learning rate")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
parser.add_argument(
    "--num_epochs", type=int, default=100, help="number of training epochs"
)
parser.add_argument("--classes", type=int, default=0, help="Number of final classes")
parser.add_argument(
    "--quantized",
    action="store_true",
    default=True,
    help="Whether to quantize the model or not",
)
parser.add_argument("--max_col", type=int, default=25, help="Max Columns in a table")
parser.add_argument("--num-heads", type=int, default=1, help="hidden attention heads")
parser.add_argument(
    "--num-out-heads", type=int, default=4, help="output attention heads"
)
parser.add_argument("--num-layers", type=int, default=2, help="hidden layers")
parser.add_argument(
    "--num-hidden", nargs="+", type=int, default=[1024], help="hidden units"
)
parser.add_argument("--residual", action="store_true", default=True)
parser.add_argument("--self-loop", action="store_true", default=True)
parser.add_argument("--in-drop", type=float, default=0.2, help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.2, help="attention dropout")
parser.add_argument("--alpha", type=float, default=0.2, help="negative slop of leaky relu")
parser.add_argument(
    "--mean-pooling",
    action="store_true",
    default=True,
    help="Whether to mean-pool embeddings",
)
parser.add_argument(
    "--layer-offset",
    type=int,
    default=1,
    help="how many layers from the top to use. 1 -> last layer, 2 -> second last.",
)
parser.add_argument(
    "--layer-pooling",
    type=str,
    choices=["none", "mean", "max", "min"],
    default="none",
    help="How to aggregate top layers",
)
parser.add_argument(
    "--GNN",
    type=str,
    choices=["GAT", "GCN", "GGNN"],
    default="GAT",
    help="GNN model type to use: GAT, GCN, or GGNN",
)

totaltime = time.time()
args = parser.parse_args()
base_dir = get_base_dir()
device = torch.device(args.gpu)
print(args)

task = "cta"
args.task = task
args.data_name = "webtables"
args.data_dir_name = "webtables"

stage = "training"
args.λ_node = 1
args.λ_edge = 0
args.λ_graph = 0

component = "node"

graph_task = "cpa" if task == "tta" else task
if args.λ_graph == 1:
    graph_labels_path = os.path.join(
        base_dir,
        "data",
        "cpadata",
        "graph_labels",
        f"graph_labels_{graph_task}_training.csv",
    )
    args.graph_labels = pd.read_csv(graph_labels_path)
    args.num_classes_graph = args.graph_labels["label"].unique().shape[0]
else:
    args.graph_labels = None
    args.num_classes_graph = 0

text_column = "embedding"
label_column = "label"

all_fold_results = []

for fold in range(5):
    print(f"\n===== Fold {fold + 1} =====")
    data_type = "msato"

    test_embedding = load_data(f"{data_type}{fold}", args)

    train_embedding = None
    for i in range(5):
        if i != fold:
            data = load_data(f"{data_type}{i}", args)
            if train_embedding is None:
                train_embedding = data
            else:
                for key in train_embedding.keys():
                    train_embedding[key] = np.concatenate(
                        [train_embedding[key], data[key]], axis=0
                    )

    train_dataloader = create_dataloaders(
        train_embedding,
        text_column,
        label_column,
        args.batch_size * args.accumulation_steps,
    )
    test_dataloader = create_dataloaders(
        test_embedding,
        text_column,
        label_column,
        args.batch_size * args.accumulation_steps,
    )

    # args.classes = len(np.unique(list(train_embedding[label_column])))
    args.classes = 78

    num_feats = len(train_embedding[text_column][0])
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    args.num_hidden = args.num_hidden * args.num_layers

    if args.GNN == "GAT":
        model = GAT(
            args.num_layers,
            num_feats,
            args.num_hidden,
            args.classes,
            args.num_classes_graph,
            heads,
            F.gelu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual,
        )
    elif args.GNN == "GCN":
        model = GCN(
            num_feats,
            args.num_hidden,
            args.classes,
            num_classes_graph=args.num_classes_graph,
            n_layers=args.num_layers,
            activation=F.gelu,
            residual=True,
            batchnorm=True,
            dropout=0.2,
        )
    elif args.GNN == "GGNN":
        model = GGNN(
            num_feats,
            args.num_hidden,
            args.classes,
            args.num_layers,
            num_classes_graph=args.num_classes_graph,
            dropout=0,
            shared_weight=False,
        )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    best_model_state = train_model_multitask(
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
        test_dataloader,
        device,
        args.num_epochs,
        args.accumulation_steps,
    )

    model.load_state_dict(best_model_state)
    test_results = calc_accuracy_nodes_edges(model, device, test_dataloader)
    print(
        f"Fold {fold + 1} Test Micro F1: {test_results['node_micro_f1']:.4f}"
    )
    all_fold_results.append(test_results["node_micro_f1"])

    shutdown_dataloader(train_dataloader)
    shutdown_dataloader(test_dataloader)

avg_micro_f1 = np.mean(all_fold_results)
print("\n===== 5-Fold Cross-Validation Results =====")
print(f"Average Test Micro F1: {avg_micro_f1:.4f}")
print(f"Total Time: {((time.time() - totaltime) / 60):2f} minutes")
