import argparse
import torch
import os
from tqdm import tqdm
from dataset import *
from model import NoName
import logging
from collections import namedtuple
from torch.utils.data import DataLoader
from utils import set_logger
import pickle
import math

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Reasoning Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_root', type=str, default='output')
    parser.add_argument('--model_name', type=str, default='GHT')
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--num_works', type=int, default=8)

    parser.add_argument('--grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.00001)

    parser.add_argument('--d_model', default=100, type=int)
    parser.add_argument('--data', default='icews14', type=str)
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--valid_epoch', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)

    parser.add_argument('--load_model_path', default='output', type=str)

    parser.add_argument('--forecasting_t_win_size', default=1, type=int)

    parser.add_argument('--warm_up', default=0.0, type=float)

    return parser.parse_args(args)

def test(model, testloader, skip_dict, device):
    model.eval()
    ranks = []
    logs = []
    with torch.no_grad():
        for sub, rel, obj, year, month, day,neg in tqdm(testloader):
            sub = sub.to(device, non_blocking=True)
            rel = rel.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            year = year.to(device, non_blocking=True)
            month = month.to(device, non_blocking=True)
            day = day.to(device, non_blocking=True)
            #break
            scores = model.test_forward(sub, rel, obj, year, month, day)
            _, rank_idx = scores.sort(dim=1, descending=True)
            rank = torch.nonzero(rank_idx == obj.view(-1, 1))[:, 1].view(-1)
            ranks.append(rank)
            for i in range(scores.shape[0]):
                src_i = sub[i].item()
                rel_i = rel[i].item()
                dst_i = obj[i].item()
                year_i = year[i].item()
                month_i = month[i].item()
                day_i = day[i].item()

                predict_score = scores[i].tolist()
                answer_prob = predict_score[dst_i]
                for e in skip_dict[(src_i, rel_i)]:
                    if e != dst_i:
                        predict_score[e] = -1e6
                predict_score.sort(reverse=True)
                filter_rank = predict_score.index(answer_prob) + 1

                logs.append({
                        'Filter MR': filter_rank,
                        'Filter MRR': 1.0 / filter_rank,
                        'Filter HITS@1': 1.0 if filter_rank <= 1 else 0.0,
                        'Filter HITS@3': 1.0 if filter_rank <= 3 else 0.0,
                        'Filter HITS@10': 1.0 if filter_rank <= 10 else 0.0,
                    })

    metrics = {}
    ranks = torch.cat(ranks)
    ranks += 1
    mrr = torch.mean(1.0 / ranks.float())
    metrics['Raw MRR'] = mrr
    for hit in [1, 3, 10]:
        avg_count = torch.mean((ranks <= hit).float())
        metrics['Raw Hit@{}'.format(hit)] = avg_count

    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics


def train_epoch(args, model, traindataloader, optimizer, scheduler, device, epoch):
    model.train()
    with tqdm(total=len(traindataloader), unit='ex') as bar:
        bar.set_description('Train')
        total_loss = 0
        total_num = 0
        for sub, rel, obj, year, month, day, neg in traindataloader:
            bs = sub.size(0)
            sub = sub.to(device, non_blocking=True)
            rel = rel.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            year = year.to(device, non_blocking=True)
            month = month.to(device, non_blocking=True)
            day = day.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            #break
            loss = model.train_forward(sub, rel, obj, year, month, day, neg)
            loss.backward()

            total_loss += loss
            total_num += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            bar.update(1)
            bar.set_postfix(loss='%.4f' % loss)

        logging.info('Epoch {} Train Loss: {}'.format(epoch, total_loss/total_num))


class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main(args):
    output_path = os.path.join(args.output_root, '{0}_{1}'.format(args.data, args.model_name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    log_file = os.path.join(output_path, 'log.txt')
    set_logger(log_file)

    logging.info(args)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    data_path = os.path.join(args.data_root, args.data)
    trainpath = os.path.join(data_path, 'train.txt')
    validpath = os.path.join(data_path, 'valid.txt')
    testpath = os.path.join(data_path, 'test.txt')
    statpath = os.path.join(data_path, 'stat.txt')
    dataset = Dataset(args.data)

    trainQuadruples = dataset.get_reverse_quadruples_array(dataset.data['train'],dataset.numRel())
    trainQuadDataset = QuadruplesDataset(trainQuadruples, dataset, 'train')
    trainDataLoader = DataLoader(
        trainQuadDataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_works,
        pin_memory=True
    )


    validQuadruples = dataset.get_reverse_quadruples_array(dataset.data['valid'],dataset.numRel())
    validQuadDataset  = QuadruplesDataset(validQuadruples, dataset, 'valid')
    validDataLoader = DataLoader(
        validQuadDataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_works,
        pin_memory=True
    )

    testQuadruples = dataset.get_reverse_quadruples_array(dataset.data['test'],dataset.numRel())
    testQuadDataset = QuadruplesDataset(testQuadruples, dataset, 'test')
    testDataLoader = DataLoader(
        testQuadDataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_works,
        pin_memory=True
    )

    Config = namedtuple('config', ['n_ent', 'd_model', 'n_rel', 'dropout'])
    config = Config(n_ent=dataset.numEnt() + 1,
                    n_rel=dataset.numRel() * 2,
                    d_model=args.d_model,
                    dropout=args.dropout)
    model = NoName(config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warm_up > 0:
        warmup_scheduler = WarmUpLR(optimizer, len(trainDataLoader) * args.warm_up)
    else:
        warmup_scheduler = None

    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        model.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load pretrain model: {}'.format(args.load_model_path))

    if args.do_train:
        logging.info('Start Training......')

        for i in range(args.max_epochs):
            if i % args.valid_epoch == 0 and i != 0:
                model_save_path = os.path.join(output_path, 'model_{}.pth'.format(i))
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_save_path)

                for delta_t in range(args.forecasting_t_win_size):
                    delta_t = delta_t + 1
                    testDataLoader.dataset.delta_t = delta_t
                    metrics = test(model, testDataLoader, dataset.skip_dict, device)

                    for mode in metrics.keys():
                        logging.info('Delta_t {} Valid {} : {}'.format(delta_t, mode, metrics[mode]))

            train_epoch(args, model, trainDataLoader, optimizer, warmup_scheduler, device, i)

    if args.do_test:
        logging.info('Start Testing......')
        for delta_t in range(args.forecasting_t_win_size):
            delta_t = delta_t + 1
            validDataLoader.dataset.delta_t = delta_t
            metrics = test(model, validDataLoader, dataset.skip_dict, device)
            for mode in metrics.keys():
                logging.info('Delta_t {} Test {} : {}'.format(delta_t, mode, metrics[mode]))


if __name__ == '__main__':
    args = parse_args()
    main(args)