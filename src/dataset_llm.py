import torch
from torch.utils.data import Dataset
import dgl
import networkx as nx
from collections import defaultdict




class GroupedTableDataset(Dataset):
    def __init__(self, args, texts, targets, table_ids, max_columns=10):
        self.args = args
        self.data_name = args.data_name
        self.sample_size = 0
        self.max_columns = max_columns

        # Group samples by table_id
        self.data = defaultdict(list)
        for text, target, table_id in zip(texts, targets, table_ids):
            self.data[table_id].append((text, target))
        self.table_ids = list(self.data.keys())

        #print("Number of Table IDs before grouping - ", np.shape(table_ids))
        #print("Number of Table IDs  after grouping -", np.shape(self.table_ids))

        self.sample_size = len(self.table_ids)
        self.feature_size = len(texts[0])

        print('Sample size', self.sample_size)
        if self.args.λ_graph == 1:
            self.graph_labels = args.graph_labels.set_index('table_name')['label'].to_dict()
            # print(self.graph_labels)
        else:
            self.graph_labels = {}

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        table_id = self.table_ids[idx]
        columns = self.data[table_id]

        # Create feature and label arrays with masking
        features = torch.zeros((self.max_columns, self.feature_size), dtype=torch.float)
        labels = torch.zeros((self.max_columns, 1), dtype=torch.long)
        masks = torch.zeros((self.max_columns,), dtype=torch.bool)

        #####################################################
        # edge_feature_dict = {}
        edge_label_dict = {}
        edge_masks_dict = {}
        #####################################################

        for i, (text, target) in enumerate(columns):
            if i >= self.max_columns:
                break
            features[i] = torch.tensor(text, dtype=torch.float)
            # labels[i] = torch.tensor(target, dtype=torch.long)

            # Wikitable
            # labels[i] = torch.tensor(target, dtype=torch.long) + 1
            labels[i] = torch.tensor(target, dtype=torch.long)

            masks[i] = 1
            ################################################
            # Merge main column '0' data and the current column as edge data
            # edge_feature_dict[(0, i)] = torch.tensor(columns[0][0] + text, dtype=torch.float)
            edge_label_dict[(0, i)] = labels[i]
            edge_masks_dict[(0, i)] = torch.tensor(1, dtype=torch.bool)
            ################################################

        # Create a complete graph for the current table
        num_nodes = masks.sum().item()
        g = nx.complete_graph(num_nodes)
        g = dgl.from_networkx(g)

        # Removing and adding self-loop
        g = dgl.remove_self_loop(g)
        if self.args.self_loop:
            g = dgl.add_self_loop(g)

        # Assign features and labels to graph nodes
        g.ndata['feat'] = features[masks]
        g.ndata['label'] = labels[masks]
        # g.ndata['mask'] = masks.nonzero().squeeze()

        ##################################################
        # Build the edge feature tensor by matching each edge (src[i], dst[i])
        # edge_feat_init = torch.zeros((self.feature_size), dtype=torch.float)
        edge_label_init = torch.zeros((1), dtype=torch.long)
        edge_mask_init = torch.tensor(0, dtype=torch.bool)
        # edge_features = []
        edge_labels = []
        edge_masks = []
        src_nodes, dst_nodes = g.edges()
        edge_pairs = list(zip(src_nodes.tolist(), dst_nodes.tolist()))
        for u, v in edge_pairs: # zip(src.tolist(), dst.tolist()):
            # cur_edge_feat = edge_feature_dict.get((u, v), edge_feat_init)  # default if not found
            cur_edge_label = edge_label_dict.get((u, v), edge_label_init)
            cur_edge_mask = edge_masks_dict.get((u, v), edge_mask_init)
            
            # edge_features.append(cur_edge_feat)
            edge_labels.append(cur_edge_label)
            edge_masks.append(cur_edge_mask)

        #g.edata['e'] = torch.stack(edge_features)
        g.edata['label'] = torch.stack(edge_labels)
        g.edata['mask'] = torch.stack(edge_masks)

        # Wikitable
        #table_id = int(table_id.split("-")[1])
        
        # ALL Others
        if self.data_name != "dataset-t2d":
            if self.data_name == "wikitable":
                table_id = int(table_id.split("-")[1])
            else:
                table_id = table_id.split("_")[0]


        #### print(table_id)
        #### print(table_id, ">>> ", int(table_id.split("-")[1]), self.graph_labels[int(table_id.split("-")[1])])
        #### print(self.graph_labels)
        #### print(self.graph_labels.keys(), type(list(self.graph_labels.keys())[0]), type(table_id.split("_")[1]))
        #### table_name = table_id  # int(table_id.split(" ")[1])
        
        #print(table_id, len(table_id))
        # g.ndata['graph_label'] = torch.tensor(self.graph_labels[table_name])
        if self.args.λ_graph == 1:
            try:
                graph_label = torch.tensor(self.graph_labels[table_id])
            except:
                graph_label = torch.tensor(0)
                print("Table name not found in Graph labels:", table_id)
                import sys
                sys.exit()
        else:
            graph_label = torch.tensor(0)
        #print(graph_label, table_name)
        ################################################## 
        #print("Batch Loading Time:", time.time()-dt)
        return (g, graph_label) 

class CustomDataset(Dataset):
    def __init__(self, texts, targets, table_id, column_id, max_input_size=None):
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.table_id = table_id
        self.column_id = column_id
        self.max = max_input_size

    def __getitem__(self, idx):
        text = self.texts[idx][:self.max] if self.max is not None else self.texts[idx]
        target = self.targets[idx]
        table_id = self.table_id[idx]
        column_id = self.column_id[idx]
        return text, target, table_id, column_id

    def __len__(self):
        return len(self.targets)
