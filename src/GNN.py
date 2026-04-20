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

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# Data loading
def load_data(data_type, args):
    data_task = 'cpa' if task == 'tta' else task
    #filename = f'{args.data_name}_{data_type}_{args.row_examples}_{args.model_name.split("/")[-1]}_{args.quantized}_{args.mean_pooling}.pkl'
    filename = f'{data_task}_{args.data_name}_{data_type}_{args.row_examples}_{args.model_name.split("/")[-1]}_{args.quantized}_{args.mean_pooling}_{args.layer_offset}_{args.layer_pooling}.pkl'
    # filename = join('/mnt/datadisk/ar', 'representation', args.data_dir_name, data_task, filename)
    filename = join(base_dir, 'data', 'representation', args.data_dir_name, data_task, filename)
    
    print(f"Embedding already exist {filename}")
    with open(filename, 'rb') as file:
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

# DataLoader creation
def create_dataloaders(df, text_column, label_column, batch_size, data_type):
    L.seed_everything(seed=252)
    # Create train dataset and dataloader
    
    data_dataset = GroupedTableDataset(
        args,
        texts=list(df[text_column]),
        targets=list(df[label_column]),
        table_ids=list(df['table_id']),
        max_columns=args.max_col
    )
    data_dataloader = DataLoader(
        dataset=data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
        num_workers=8
    )

    return data_dataloader


def shutdown_dataloader(dataloader):
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    dataloader._iterator = None



# Training function
def train_model_multitask(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, test_dataloader, device, num_epochs, accumulation_steps):
    scaler = GradScaler()
    criteria = 0 # float('inf')
    saved_model_state = None
    saved_epoch = None

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        node_loss = 0
        edge_loss = 0
        graph_loss = 0

        for batch_idx, graph_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            graph_batch, graph_labels = graph_data
            graph_batch = graph_batch.to(device)
            graph_labels = graph_labels.to(device)

            ##################################################
            targets = {}
            if args.λ_node:
                targets["node_targets"] = graph_batch.ndata['label'].view(-1)
            if args.λ_edge:
                targets["edge_targets"] = graph_batch.edata['label'].view(-1)
            if args.λ_graph:
                targets["graph_target"] = graph_labels  # graph_batch.gdata['graph_label']
            ##################################################

            # Perform forward pass with autocast for mixed precision training
            with autocast('cuda'):
                logits = model(graph_batch, graph_batch.ndata['feat'])

                # loss = F.cross_entropy(logits, targets) / accumulation_steps
                loss = 0
                _node_loss = 0
                _edge_loss = 0
                _graph_loss = 0
                if args.λ_edge:
                    edge_mask = graph_batch.edata['mask']
                #node_mask = None # Not required currently, as graph nodes are created on available columns only

                #print(targets["edge_targets"][edge_mask].min(), targets["edge_targets"][edge_mask].max(), args.classes)


                #print(min(targets["edge_targets"][edge_mask]), max(targets["edge_targets"][edge_mask]))


                if args.λ_node:
                    _node_loss = F.cross_entropy(logits["node_logits"], targets["node_targets"])
                    loss += args.λ_node * _node_loss
                if args.λ_edge:
                    _edge_loss = F.cross_entropy(logits["edge_logits"][edge_mask], targets["edge_targets"][edge_mask])
                    loss += args.λ_edge * _edge_loss
                if args.λ_graph:
                    _graph_loss = F.cross_entropy(logits["graph_logits"], targets["graph_target"])
                    loss += args.λ_graph * _graph_loss
 
            total_loss += loss.item()
            if args.λ_node:
                node_loss += _node_loss.item()
            if args.λ_edge:
                edge_loss += _edge_loss.item()
            if args.λ_graph:
                graph_loss += _graph_loss.item()
            scaler.scale(loss).backward()

            # Backward pass, optimization step, and learning rate adjustment
            # if not batch_idx % accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_loss /= len(train_dataloader)
        #train_results = calc_accuracy_nodes_edges(model, device, train_dataloader)
        valid_results = calc_accuracy_nodes_edges(model, device, valid_dataloader) 
        
        # Train and Valid results
        print(
                    f'Epoch: {epoch + 1} / {num_epochs}', 
                    #f'| Train Total Loss: {train_results["total_loss"]:.4f}'
                    f'| Valid Total Loss: {valid_results["total_loss"]:.4f}'
                )
        '''
        print(
                    f'NODE >> '
                    f'Train Loss: {train_results["node_loss"]:.4f}'
                    f'| Valid Loss: {valid_results["node_loss"]:.4f}'
                    #f'| Train Accuracy: {train_results["node_acc"]:.4f}'
                    #f'| Test Accuracy: {valid_results["node_acc"]:.4f}'
                    f'| Train Micro F1: {train_results["node_micro_f1"]:.4f}'
                    f'| Valid Micro F1: {valid_results["node_micro_f1"]:.4f}'
                )
        print(
                    f'EDGE >> '
                    f'Train Loss: {train_results["edge_loss"]:.4f}'
                    f'| Valid Loss: {valid_results["edge_loss"]:.4f}'
                    #f'| Train Accuracy: {train_results["edge_acc"]:.4f}'
                    #f'| Test Accuracy: {valid_results["edge_acc"]:.4f}'
                    f'| Train Micro F1: {train_results["edge_micro_f1"]:.4f}'
                    f'| Valid Micro F1: {valid_results["edge_micro_f1"]:.4f}'
                )
        print(
                    f'GRPH >> '
                    f'Train Loss: {train_results["graph_loss"]:.4f}'
                    f'| Valid Loss: {valid_results["graph_loss"]:.4f}'
                    #f'| Train Accuracy: {train_results["graph_acc"]:.4f}'
                    #f'| Test Accuracy: {valid_results["graph_acc"]:.4f}'
                    f'| Train Micro F1: {train_results["graph_micro_f1"]:.4f}'
                    f'| Valid Micro F1: {valid_results["graph_micro_f1"]:.4f}'
                )'''

        if valid_results[component + '_micro_f1'] > criteria:
            criteria = valid_results[component + '_micro_f1']
            saved_model_state = copy.deepcopy(model.state_dict()) 
            saved_epoch = epoch
            print("Saving best model in Epoch", saved_epoch+1)

            # Test result for best validation found so far
            test_results = calc_accuracy_nodes_edges(model, device, test_dataloader)
            msg_parts = []
            if args.λ_node:
                msg_parts.append(f'Node {test_results["node_micro_f1"]:.4f}')
            if args.λ_edge:
                msg_parts.append(f'Edge {test_results["edge_micro_f1"]:.4f}')
            if args.λ_graph:
                msg_parts.append(f'Graph {test_results["graph_micro_f1"]:.4f}')
            msg = '| Test Micro F1: ' + '\t\t'.join(msg_parts)
            print("Saving best model state ("+args.model_name+") in Epoch" + str(saved_epoch+1) + "\t" + msg)
            print(  
                    f'Test Performance:'  # , f'| Test Total Loss: {test_results["total_loss"]:.4f}'
                )

            if args.λ_node:
                print(
                        f'NODE >> '
                        f'| Test Loss: {test_results["node_loss"]:.4f}'
                        f'| Test Micro F1: {test_results["node_micro_f1"]:.4f}'
                    )
            if args.λ_edge:
                print(
                        f'EDGE >> '
                        f'| Test Loss: {test_results["edge_loss"]:.4f}'
                        f'| Test Micro F1: {test_results["edge_micro_f1"]:.4f}'
                    )
            if args.λ_graph:
                print(
                        f'GRPH >> '
                        f'| Test Loss: {test_results["graph_loss"]:.4f}'
                        f'| Test Micro F1: {test_results["graph_micro_f1"]:.4f}'
                    )
        
        

    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f'Total training time: {training_time:.2f} min')

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
        # for batch in tqdm(dataloader, total=len(dataloader), desc=f'Calc {type} accuracy'):
        for graph_data in dataloader:
            #graph_batch = graph_batch.to(device)
            graph_batch, graph_labels = graph_data
            graph_batch = graph_batch.to(device)
            graph_labels = graph_labels.to(device)
            # logits = model(graph_batch, graph_batch.ndata['feat'])
            if args.λ_node:
                node_targets = graph_batch.ndata['label'].view(-1)
            if args.λ_edge:
                edge_targets = graph_batch.edata['label'].view(-1)
                edge_mask = graph_batch.edata['mask']

            with autocast('cuda'):
                logits = model(graph_batch, graph_batch.ndata['feat'])

                # Loss
                _node_loss = 0
                _edge_loss = 0
                _graph_loss = 0
                if args.λ_node:
                    _node_loss = F.cross_entropy(logits["node_logits"], node_targets)
                    node_loss += _node_loss.item()
                    # Nodes
                    _, pred = torch.max(logits["node_logits"], dim=1)
                    node_pred_scores.extend(pred.cpu().numpy())
                    node_actual_scores.extend(node_targets.cpu().numpy())
                if args.λ_edge:
                    _edge_loss = F.cross_entropy(logits["edge_logits"][edge_mask], edge_targets[edge_mask])
                    edge_loss += _edge_loss.item()
                    # Edges
                    _, pred = torch.max(logits["edge_logits"], dim=1)
                    edge_pred_scores.extend(pred[edge_mask].cpu().numpy())
                    edge_actual_scores.extend(edge_targets[edge_mask].cpu().numpy())
                if args.λ_graph:
                    _graph_loss = F.cross_entropy(logits["graph_logits"], graph_labels.to(device))
                    graph_loss += _graph_loss.item()
                    # Graph
                    _, pred = torch.max(logits["graph_logits"], dim=1)
                    graph_pred_scores.extend(pred.cpu().numpy())
                    graph_actual_scores.extend(graph_labels.cpu().numpy())

                loss = args.λ_node * _node_loss + args.λ_edge * _edge_loss + args.λ_graph * _graph_loss

                total_loss += loss.item()
                # pred_score = F.softmax(logits, dim=-1).argmax(dim=-1).cpu().detach().numpy().tolist()
                # pred_scores.extend(pred_score)
                # actual_scores.extend(targets.numpy().tolist())

        # pred_scores = np.array(pred_scores)
        node_accuracy = 0
        node_micro_f1 = 0
        if args.λ_node:
            node_accuracy = accuracy_score(node_actual_scores, node_pred_scores)
            node_micro_f1 = f1_score(node_actual_scores, node_pred_scores, average='micro')

        edge_accuracy = 0
        edge_micro_f1 = 0
        if args.λ_edge:
            edge_accuracy = accuracy_score(edge_actual_scores, edge_pred_scores)
            edge_micro_f1 = f1_score(edge_actual_scores, edge_pred_scores, average='micro')

        #actual = [int(x) for x in graph_actual_scores]
        #print(actual, len(actual))
        #print(np.unique(actual), len(np.unique(actual)))
        #pred = [int(x) for x in graph_pred_scores]
        #print(pred, len(pred))
        #print(np.unique(pred), len(np.unique(pred)))

        #uactual, upred = np.unique(actual), np.unique(pred)
        #print("Intersection", set(uactual) - set(upred))

        graph_accuracy = 0
        graph_micro_f1 = 0
        if args.λ_graph:
            graph_accuracy = accuracy_score(graph_actual_scores, graph_pred_scores)
            graph_micro_f1 = f1_score(graph_actual_scores, graph_pred_scores, average='micro')

        return {"node_acc": node_accuracy, "node_micro_f1": node_micro_f1, 
                "edge_acc": edge_accuracy, "edge_micro_f1": edge_micro_f1,  
                "graph_acc": graph_accuracy, "graph_micro_f1": graph_micro_f1,
                "total_loss": total_loss,
                "node_loss": node_loss, 
                "edge_loss": edge_loss, 
                "graph_loss": graph_loss
                }



    

parser = configargparse.ArgParser()

parser.add_argument("--gpu", type=str, default="cuda:0",
                        help="Which GPU to use")
parser.add_argument('--row_examples', type=int, default=25,
                        help="batch size used for training, validation and test")
parser.add_argument('--batch-size', type=int, default=128,
                        help="batch size used for training, validation and test")
parser.add_argument('--accumulation_steps', type=int, default=1,
                        help="batch size used for training, validation and test")
parser.add_argument('--data-name', type=str, default='',
                    help="for loading dataset")
parser.add_argument(
    "--data-dir-name",
    type=str,
    default="",
    help="Directory name under data/representation and data/graph_labels. Defaults to --data-name.",
)
parser.add_argument('--mode', type=str, default='train',
                    help="train or eval")
parser.add_argument('--train-valid-ratio', type=float, default=1,
                    help="ratio for dividing train data between train and validation")
parser.add_argument('--model-name', type=str, default='mistral', # mistral, Qwen7, llama3, mixtral, tinyllama #default='mistralai/Mistral-7B-v0.1',
                    help="name of the llm")
parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
parser.add_argument("--diff_lr", type=float, default=0.00001,
                        help="learning rate")
parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of training epochs")
parser.add_argument("--classes", type=int, default=0,
                    help="Number of final classes. Semtab:275, Webtables: 78")
parser.add_argument("--quantized", action="store_true", default=True,
                        help="Whether to quantize the model or not")
parser.add_argument("--max_col", type=int, default=25,
                        help="Max Columns in a table")
parser.add_argument("--num-heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=4,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", nargs='+', type=int, default=[1024],
                    help="number of hidden units")
parser.add_argument("--residual", action="store_true", default=True,
                    help="use residual connection")
parser.add_argument("--self-loop", action="store_true", default=True,
                    help="Whether to use self loop")
parser.add_argument("--in-drop", type=float, default=0.2,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.2,
                    help="attention dropout")
parser.add_argument('--alpha', type=float, default=0.2,
                    help="the negative slop of leaky relu")
parser.add_argument("--description", action="store_true", default=False,
                        help="Whether to quantize the model or not")
parser.add_argument("--mean-pooling", action="store_true", default=True,
                        help="Whether to quantize the model or not")
parser.add_argument('--layer-offset', type=int, default=1,
                    help="how many layers from the top to use. 1 -> last layer, 2 -> second last.")
parser.add_argument('--layer-pooling', type=str, choices=['none', 'mean', 'max', 'min'], default='none',
                    help="How to aggregate top layers: 'none' (single), 'mean', 'max', or 'min'")
parser.add_argument('--GNN', type=str, choices=['GAT', 'GCN', 'GGNN'], default='GAT',
                    help='GNN model type to use: GAT, GCN, or GGNN')
parser.add_argument('--task', type=str, choices=['cta', 'cpa', 'tta', 'multitask'], default='cta',
                    help="Task to run: cta, cpa, tta, or multitask")

args = parser.parse_args()
base_dir = get_base_dir()
device = torch.device(args.gpu)
print(args)


# models = {0: "mistral", 1: "Qwen7", 2: "mixtral", 3: "llama3", 4: "tinyllama", 5: "llama2", 6: "bert"}
dataset_aliases = {
    'dataset-sota-large': ('dataset-sota-large', 'sotabv2'),
    'dataset-sota-small': ('dataset-sota-small', 'sotabv2'),
    'dataset-sota-dbpedia': ('dataset-sota-dbpedia', 'dataset-sota-dbpedia'),
    'dataset-t2d': ('dataset-t2d', 'dataset-t2d'),
    'wikitable': ('wikitable', 'wikitable'),
}

task = args.task
if not args.data_name:
    raise ValueError(
        f"--data-name is required. Choose one of: {', '.join(dataset_aliases.keys())}"
    )

if args.data_name in dataset_aliases:
    resolved_data_name, default_data_dir_name = dataset_aliases[args.data_name]
else:
    resolved_data_name = args.data_name
    default_data_dir_name = args.data_name

args.data_name = resolved_data_name
args.data_dir_name = args.data_dir_name or default_data_dir_name

if args.data_name == 'wikitable' and task == 'tta':
    raise ValueError("wikitable does not support task 'tta'")

if args.data_name == 'dataset-t2d' and task == 'cpa':
    args.num_epochs = 500
    args.batch_size = 8

stage = 'training'
args.λ_node = task in ('cta', 'multitask')
args.λ_edge = task in ('cpa', 'multitask')
args.λ_graph = task in ('tta', 'multitask')
# print("\n λ_node - ", args.λ_node, "\n λ_edge - ", args.λ_edge, "\nλ_graph - ", args.λ_graph)

component = "UNSET"
if task == 'cta':
    component = "node"
elif task == 'cpa':
    component = "edge" 
elif task == 'tta':
    component = "graph"
elif task == 'multitask':
    component = "node"

datasize = "_small" if args.data_name == 'dataset-sota-small' else ""
graph_task = 'cpa' if task == 'tta' else task
graph_labels_path = os.path.join(
    base_dir,
    "data",
    "graph_labels",
    args.data_dir_name,
    graph_task,
    "graph_labels",
    "graph_labels_" + graph_task + "_training" + datasize + ".csv",
)
args.graph_labels = pd.read_csv(graph_labels_path)
# print(graph_labels_path)
args.num_classes_graph = args.graph_labels['label'].unique().shape[0]
# print("\n\nargs.num_classes_graph: ", args.num_classes_graph, "\n\n")


text_column = "embedding"
label_column = "label"


train_embedding = load_data('train', args)
test_embedding = load_data('test', args)
valid_embedding = load_data('validation', args)

if args.data_name == 'dataset-t2d' and task == 'cpa':
    test_embedding, valid_embedding = valid_embedding, test_embedding

# args.classes = len(np.unique(list(train_embedding[label_column]))) + 1
args.classes = len(np.unique(list(train_embedding[label_column])))

if args.data_name == 'wikitable' and task == 'cta':
    args.classes = 255


# print(">>>> Train #. of classes:", len(np.unique(list(train_embedding[label_column]))))
# print(">>>> Valid #. of classes:", len(np.unique(list(valid_embedding[label_column]))))
# print(">>>>  Test #. of classes:", len(np.unique(list(test_embedding[label_column]))))


# print(np.shape(np.unique(list(train_embedding[label_column]))), np.unique(list(train_embedding[label_column])))

train_dataloader = create_dataloaders(train_embedding, text_column, label_column, args.batch_size * args.accumulation_steps, 'train')
test_dataloader = create_dataloaders(test_embedding, text_column, label_column, args.batch_size * args.accumulation_steps, 'test')
valid_dataloader = create_dataloaders(valid_embedding, text_column, label_column, args.batch_size * args.accumulation_steps, 'validation')

L.seed_everything(seed=252)

num_feats = len(train_embedding[text_column][0])
heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
args.num_hidden = args.num_hidden * args.num_layers

# print(heads, num_feats, args.num_hidden, args.classes, args.num_layers)

# Instantiate the neural network model
if args.GNN == 'GAT':
    model = GAT(args.num_layers, num_feats, args.num_hidden, args.classes, args.num_classes_graph, heads, F.gelu, args.in_drop, args.attn_drop, args.alpha, args.residual)
elif args.GNN == 'GCN':
    model = GCN(num_feats, args.num_hidden, args.classes, num_classes_graph=args.num_classes_graph, n_layers=args.num_layers, activation=F.gelu, residual=True, batchnorm=True, dropout=0.2)
elif args.GNN == 'GGNN':
    model = GGNN(num_feats, args.num_hidden, args.classes, args.num_layers, num_classes_graph=args.num_classes_graph, dropout=0, shared_weight=False)
 


# model = LLM(args, access_token, args.gpu)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)


#model = train_model(model, optimizer, lr_scheduler, train_dataloader, test_dataloader, device, args.num_epochs, args.accumulation_steps)
best_model_state = train_model_multitask(model, optimizer, lr_scheduler, train_dataloader, valid_dataloader, test_dataloader, device, args.num_epochs, args.accumulation_steps)

model.load_state_dict(best_model_state)

train_results = calc_accuracy_nodes_edges(model, device, train_dataloader)
test_results = calc_accuracy_nodes_edges(model, device, test_dataloader)

header_res = f'\n\nTask: {task}'
print(header_res)
if args.λ_node:
    node_res = (
            f'NODE\t'
            #f'Train Accuracy: {train_results["node_acc"]:.4f}'
            #f'| Test Accuracy: {test_results["node_acc"]:.4f}'
            f'| Train Micro F1: {train_results["node_micro_f1"]:.4f}'
            f'| Test Micro F1: {test_results["node_micro_f1"]:.4f}'
            )
    print(node_res)
if args.λ_edge:
    edge_res = (
            f'EDGE\t'
            #f'Train Accuracy: {train_results["edge_acc"]:.4f}'
            #f'| Test Accuracy: {test_results["edge_acc"]:.4f}'
            f'| Train Micro F1: {train_results["edge_micro_f1"]:.4f}'
            f'| Test Micro F1: {test_results["edge_micro_f1"]:.4f}'
                    )
    print(edge_res)
if args.λ_graph:
    graph_res = (
            f'GRAPH\t'
            #f'Train Accuracy: {train_results["graph_acc"]:.4f}'
            #f'| Test Accuracy: {test_results["graph_acc"]:.4f}'
            f'| Train Micro F1: {train_results["graph_micro_f1"]:.4f}'
            f'| Test Micro F1: {test_results["graph_micro_f1"]:.4f}'
    )
    print(graph_res)

print(args.model_name, args.data_name)

shutdown_dataloader(train_dataloader)
shutdown_dataloader(valid_dataloader)
shutdown_dataloader(test_dataloader)
