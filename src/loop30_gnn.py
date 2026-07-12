import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

torch.manual_seed(0)

# small synthetic graph: two communities, GNN should learn to separate them
num_nodes = 20
edge_list = []
# community A: nodes 0-9 densely connected
for i in range(10):
    for j in range(i + 1, 10):
        if (i + j) % 3 == 0:
            edge_list.append((i, j))
            edge_list.append((j, i))
# community B: nodes 10-19 densely connected
for i in range(10, 20):
    for j in range(i + 1, 20):
        if (i + j) % 3 == 0:
            edge_list.append((i, j))
            edge_list.append((j, i))
# a couple bridge edges
edge_list += [(5, 15), (15, 5)]

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
x = torch.eye(num_nodes, dtype=torch.float)  # one-hot node features
y = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[[0, 1, 2, 3, 10, 11, 12, 13]] = True
test_mask = ~train_mask

data = Data(x=x, edge_index=edge_index, y=y)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_nodes, 8)
        self.conv2 = GCNConv(8, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

# naive baseline: majority class
majority_class = y[train_mask].mode().values.item()
baseline_acc = (torch.full_like(y[test_mask], majority_class) == y[test_mask]).float().mean().item()

print(f"final loss: {loss.item():.4f}")
print(f"GNN test accuracy: {test_acc:.3f}")
print(f"majority-class baseline accuracy: {baseline_acc:.3f}")

assert test_acc > baseline_acc, "GNN did not beat naive baseline"
print("OK")
