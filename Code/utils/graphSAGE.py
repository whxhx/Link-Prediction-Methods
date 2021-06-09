from torch import tensor
from torch_geometric.nn import SAGEConv
from torch.utils.data import random_split
import torch.nn.functional as F
import torch
import networkx as nx
import argparse
import random
import copy
import time
from sklearn.metrics import auc, roc_curve
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='input dataset')
parser.add_argument('-lr', help='learning rate', type=float)
args = vars(parser.parse_args())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = args['dataset']
lr = args['lr']


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, **kwargs):
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.SAGEConvs = torch.nn.ModuleList()
        self.SAGEConvs.append(SAGEConv(in_channels=in_channels, out_channels=out_channels, **kwargs))
        for i in range(num_layers - 1):
            self.SAGEConvs.append(SAGEConv(in_channels=out_channels, out_channels=out_channels, **kwargs))

    def reset_parameters(self):
        for Conv in self.SAGEConvs:
            Conv.reset_parameters()

    def forward(self, x, edge_index):
        out = x
        for Conv in self.SAGEConvs:
            out = Conv.forward(out, edge_index)
            out = F.softmax(out)
        return out


def sample_negative_edges(G, num_neg_samples):
    neg_edge_list = []
    sample_cnt = 0
    while sample_cnt < num_neg_samples:
        u = random.sample(G.nodes, 1)[0]
        v = random.sample(G.nodes, 1)[0]
        if u == v:
            continue
        if u not in list(G[v]) and v not in list(G[u]):
            neg_edge_list.append((u, v))
            sample_cnt += 1
    return neg_edge_list


def edge_list_to_tensor(edge_list):
    u = [edge[0] for edge in edge_list]
    v = [edge[1] for edge in edge_list]
    edge_index = torch.tensor([u, v], dtype=torch.long, device=device)
    return edge_index


def train(model, x, adj_t, edge_index, label, optimizer, loss_fn):
    model.train()

    optimizer.zero_grad()
    out = model.forward(x, adj_t)
    out = torch.sum(torch.mul(out[edge_index[0]], out[edge_index[1]]), dim=1)
    out = F.softmax(out)
    loss = loss_fn(out, label)

    loss.backward()
    optimizer.step()

    return loss.item()


def accuracy(model, x, adj_t, edge_list, label):
    model.eval()
    out = model.forward(x, adj_t)
    pred = torch.sum(torch.mul(out[edge_list[0]], out[edge_list[1]]), dim=1)
    pred = pred > 0.5
    pred = pred.long()

    acc = pred == label
    return round(torch.mean(acc.float()).item(), 4)

def compute_auc(model, x, adj_t, edge_list, label):
    model.eval()
    out = model.forward(x, adj_t)
    pred = torch.sum(torch.mul(out[edge_list[0]], out[edge_list[1]]), dim=1)

    pred = pred.cpu().detach().numpy()

    fpr, tpr, thresholds = roc_curve(label, pred, pos_label=1)

    return auc(fpr, tpr)

def prepare_dataset(G):
    pos_edge_list = list(G.edges)
    pos_edge_list = [[t, 1] for t in pos_edge_list]
    neg_edge_list = sample_negative_edges(G, len(pos_edge_list))
    neg_edge_list = [[t, 0] for t in neg_edge_list]
    edge_list = pos_edge_list + neg_edge_list

    torch.manual_seed(0)
    valid_test_l = int(len(edge_list) / 5)
    train_l = len(edge_list) - valid_test_l * 2
    train_t, valid_t, test_t = random_split(edge_list, [train_l, valid_test_l, valid_test_l])

    train_edge_list = [t[0] for t in train_t]
    train_edge_label = [t[1] for t in train_t]
    valid_edge_list = [t[0] for t in valid_t]
    valid_edge_label = [t[1] for t in valid_t]
    test_edge_list = [t[0] for t in test_t]
    test_edge_label = [t[1] for t in test_t]

    train_edge_tensor = edge_list_to_tensor(train_edge_list)
    valid_edge_tensor = edge_list_to_tensor(valid_edge_list)
    test_edge_tensor = edge_list_to_tensor(test_edge_list)

    train_edge_label = tensor(train_edge_label, device=device, dtype=torch.float)
    valid_edge_label = tensor(valid_edge_label, device=device, dtype=torch.float)
    test_edge_label = tensor(test_edge_label, device=device, dtype=torch.float)
    return train_edge_tensor, train_edge_label, valid_edge_tensor, valid_edge_label, test_edge_tensor, test_edge_label


G = nx.Graph()

with open(dataset, 'r') as f:
    for line in f.readlines():
        line = line.split(' ')
        u = int(line[0])
        v = int(line[1])
        u -= 1
        v -= 1
        G.add_edge(u, v)

x = torch.rand((G.number_of_nodes(), 128), device=device, dtype=torch.float)

model = GCN(128, 8, 3)
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.BCELoss()

train_edge_tensor, train_edge_label, valid_edge_tensor, \
valid_edge_label, test_edge_tensor, test_edge_label = prepare_dataset(G)

best_model = None
best_valid_acc = 0

adj_t = list(G.edges)
adj_t_v = [(item[1], item[0]) for item in adj_t]
adj_t = adj_t + adj_t_v

adj_t = edge_list_to_tensor(adj_t)


start = time.time()
for epoch in range(5000):
    loss = train(model, x, adj_t, train_edge_tensor, train_edge_label, optimizer, loss_fn)

    train_acc = accuracy(model, x, adj_t, train_edge_tensor, train_edge_label)
    valid_acc = accuracy(model, x, adj_t, valid_edge_tensor, valid_edge_label)
    test_acc = accuracy(model, x, adj_t, test_edge_tensor, test_edge_label)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%')
end = time.time()

acc = accuracy(best_model, x, adj_t, test_edge_tensor, test_edge_label)
auc = compute_auc(best_model, x, adj_t, test_edge_tensor, test_edge_label)

with open('./result.txt', 'a') as f:
    f.write(os.path.basename(dataset))
    f.write(f'\naccuracy:{100 * acc:.2f}%')
    f.write(f'\nauc:{auc:.4f}')
    f.write(f'\nruntime:{(end-start):.0f}s\n\n')
