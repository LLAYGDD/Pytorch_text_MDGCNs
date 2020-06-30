import torch
import torch.nn as nn
import torch.nn.functional as F

from GAT_Layers import  GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nclass=10, dropout=0., nheads=4):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, 4, dropout=dropout, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(4* nheads, nclass, dropout=dropout, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
#
#
# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [SpGraphAttentionLayer(nfeat,
#                                                  nhid,
#                                                  dropout=dropout,
#                                                  alpha=alpha,
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = SpGraphAttentionLayer(nhid * nheads,
#                                              nclass,
#                                              dropout=dropout,
#                                              alpha=alpha,
#                                              concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)

