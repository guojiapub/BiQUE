import sys
import torch
from torch import optim

from datasets import Dataset
from models import *
from regularizers import *


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def eval(
        dataset: Dataset, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
        at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
):
    model.eval()
    test = dataset.get_examples(split)
    examples = torch.from_numpy(test.astype('int64')).cuda()
    missing = [missing_eval]
    if missing_eval == 'both':
        missing = ['rhs', 'lhs']

    mean_reciprocal_rank = {}
    hits_at = {}

    flag = False
    for m in missing:
        q = examples.clone()
        if n_queries > 0:
            permutation = torch.randperm(len(examples))[:n_queries]
            q = examples[permutation]
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2
        ranks = model.get_ranking(q, dataset.to_skip[m], batch_size=500)

        if log_result:
            if not flag:
                results = np.concatenate((q.cpu().detach().numpy(),
                                          np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                flag = True
            else:
                results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                          np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

        mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
        hits_at[m] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            at
        ))))

    return mean_reciprocal_rank, hits_at



def load_model(model, save_path):
    state = torch.load(os.path.join(save_path, 'checkpoint'))
    model.load_state_dict(state)

    return model


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': round(m, 3), 'hits@[1,3,10]': list(map(lambda x:round(x*100/100.0, 3), h.numpy()))}


data_path = "../data/"
checkpoint = sys.argv[1]
config = {
    "model": "BiQUE",
    "regularizer": "wN3",
    "optimizer": "Adagrad",
    "rank": 128,
    "batch_size": 5000,
    "init": 0.001,
}


if checkpoint == "WN18RR":
    save_path = '../ckpt/BiQUE_WN18RR_wN3_300_128_0.15_0.1_0'
    config["dataset"] = "WN18RR"
elif checkpoint == "FB237":
    save_path = '../ckpt/BiQUE_FB237_wN3_500_128_0.07_0.1_0'
    config["dataset"] = "FB237"
elif checkpoint == "YAGO3":
    save_path = '../ckpt/BiQUE_YAGO3-10_wN3_1000_128_0.005_0.1_0'
    config["dataset"] = "YAGO3-10"
elif checkpoint == "CN100K":
    save_path = '../ckpt/BiQUE_Concept100k_wN3_5000_128_0.1_0.1_0'
    config["dataset"] = "conceptnet-100k"
elif checkpoint == "ATOMIC":
    save_path = '../ckpt/BiQUE_Atomic_wN3_5000_128_0.005_0.1_0'
    config["dataset"] = "Atomic"


args = dotdict(config)

dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init)')
exec('regularizer = '+args.regularizer+'(args.reg)')
device = 'cuda'
model.to(device)
regularizer.to(device)
model = load_model(model, save_path)
test = avg_both(*eval(dataset, model, 'test', -1))
print(test)