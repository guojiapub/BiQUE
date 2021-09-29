import torch
from torch import nn
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from qutils import *
import os
import numpy as np


class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks




class BiQUE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(BiQUE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], 8 * rank, sparse=True),
            nn.Embedding(sizes[1], 16 * rank, sparse=True),
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs += rel[:, self.rank * 8:]
        w_a, x_a, y_a, z_a = torch.split(lhs, self.rank * 2, dim=-1)
        w_b, x_b, y_b, z_b = torch.split(rel[:, :self.rank*8], self.rank * 2, dim=-1)

        A = complex_mul(w_a,w_b) - complex_mul(x_a,x_b) - complex_mul(y_a,y_b) - complex_mul(z_a,z_b)  
        B = complex_mul(w_a,x_b) + complex_mul(x_a,w_b) + complex_mul(y_a,z_b) - complex_mul(z_a,y_b)  
        C = complex_mul(w_a,y_b) - complex_mul(x_a,z_b) + complex_mul(y_a,w_b) + complex_mul(z_a,x_b)  
        D = complex_mul(w_a,z_b) + complex_mul(x_a,y_b) - complex_mul(y_a,x_b) + complex_mul(z_a,w_b)  

        res = torch.cat([A, B, C, D], dim=-1)
        return  res @ self.embeddings[0].weight.transpose(0, 1), [(get_norm(lhs, 8), get_norm(rel[:, :self.rank*8], 8), get_norm(rhs, 8))]





