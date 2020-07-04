import argparse
import os
import json
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.optim import SGD, Adam
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel, print_tensor_dict, x_u_split, calculate_accuracy
from torch.backends import cudnn
from model import resnet
from datasets import get_CIFAR10, get_SVHN, Joint, get_AwA2
from flows import Invertible1x1Conv, NormalizingFlowModel
from spline_flows import NSF_CL
from torch.distributions import MultivariateNormal
import itertools
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

cudnn.benchmark = True

parser = argparse.ArgumentParser()
# Model options
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--width', default=2, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--n_workers', default=4, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--eval_batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60, 120, 160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)
parser.add_argument("--no_augment", action="store_false", 
                    dest="augment", help="Augment training data")

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='.', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--download", action="store_true", 
                    help="downloads dataset")

# SSL options
parser.add_argument("--ssl", action="store_true", 
                    help="Do semi-supervised learning")
parser.add_argument("--num_labelled", type=int, default=4000, 
                    help="Number of labelled data points")
parser.add_argument("--min_entropy", action="store_true", 
                    help="Add the minimum entropy loss")
parser.add_argument("--lp", action="store_true", 
                    help="Add the learned prior (LP) loss")
parser.add_argument("--semantic_loss", action="store_true", 
                    help="Add the semantic loss")
parser.add_argument("--unl_weight", type=float, default=0.1, 
                    help="Weight for unlabelled regularizer loss")


def check_dataset(dataset, dataroot, augment, download):
    if dataset == "cifar10":
        dataset = get_CIFAR10(augment, dataroot, download)
    if dataset == "svhn":
        dataset = get_SVHN(augment, dataroot, download)
    if dataset == "awa2":
        dataset = get_AwA2(augment, dataroot)
    return dataset

def check_manual_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parser.parse_args()
    print('parsed options:', vars(args))
    epoch_step = json.loads(args.epoch_step)

    check_manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    ds = check_dataset(args.dataset, args.dataroot, args.augment, args.download)

    if args.dataset == "awa2":
        image_shape, num_classes, train_dataset, test_dataset, all_labels = ds
        all_labels = all_labels.to("cuda:0")
    else:
        image_shape, num_classes, train_dataset, test_dataset = ds
        all_labels = torch.eye(num_classes).to("cuda:0")

    if args.ssl:
        num_labelled = args.num_labelled
        num_unlabelled = len(train_dataset)-num_labelled            
        if args.dataset == "awa2":
            labelled_set, unlabelled_set = data.random_split(train_dataset, [num_labelled, num_unlabelled])
        else:
            td_targets = train_dataset.targets if args.dataset == "cifar10" else train_dataset.labels
            labelled_idxs, unlabelled_idxs = x_u_split(td_targets, num_labelled, num_classes)
            labelled_set, unlabelled_set = [Subset(train_dataset, labelled_idxs), Subset(train_dataset, unlabelled_idxs)] 
        labelled_set = data.ConcatDataset([labelled_set for i in range(num_unlabelled//num_labelled+1)])
        labelled_set, _ = data.random_split(labelled_set, [num_unlabelled, len(labelled_set)-num_unlabelled])
        
        train_dataset = Joint(labelled_set, unlabelled_set)


    def _init_fn(worker_id):
        np.random.seed(args.seed)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        worker_init_fn=_init_fn
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        worker_init_fn=_init_fn
    )

    model, params = resnet(args.depth, args.width, num_classes, image_shape[0])
    
    if args.lp:
        num_flow_classes = num_classes if not num_classes%2 else num_classes+1
        prior_y = MultivariateNormal(torch.zeros(num_flow_classes).to("cuda:0"), 
                                     torch.eye(num_flow_classes).to("cuda:0"))
        num_flows = 3
        flows = [NSF_CL(dim=num_flow_classes, K=8, B=3, hidden_dim=16) for _ in range(num_flows)]
        convs = [Invertible1x1Conv(dim=num_flow_classes) for i in range(num_flows)]
        flows = list(itertools.chain(*zip(convs, flows)))    
        model_y = NormalizingFlowModel(prior_y, flows, num_flow_classes).to("cuda:0")
        optimizer_y = Adam(model_y.parameters(), lr=1e-3, weight_decay=1e-5)

    def create_optimizer(args, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = create_optimizer(args, args.lr)

    epoch = 0

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    if args.dataset == "awa2":
        classacc = tnt.meter.AverageValueMeter()
    else:
        classacc = tnt.meter.ClassErrorMeter(accuracy=True) 
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    global counter
    counter = 0
    
    def compute_loss(sample):
        if not args.ssl:
            inputs = cast(sample[0], args.dtype)
            targets = cast(sample[1], 'long')
            y = data_parallel(model, inputs, params, sample[2], list(range(args.ngpu))).float()
            if args.dataset == "awa2":
                return F.binary_cross_entropy_with_logits(y, targets.float()), y
            else:
                return F.cross_entropy(y, targets), y
        else:
            global counter
            l = sample[0]
            u = sample[1]
            inputs_l = cast(l[0], args.dtype)
            targets_l = cast(l[1], 'long')
            inputs_u = cast(u[0], args.dtype)
            y_l = data_parallel(model, inputs_l, params, sample[2], list(range(args.ngpu))).float()
            y_u = data_parallel(model, inputs_u, params, sample[2], list(range(args.ngpu))).float()
            if args.dataset == "awa2":
                loss = F.binary_cross_entropy_with_logits(y_l, targets_l.float())
            else:
                loss = F.cross_entropy(y_l, targets_l)
            
            if args.min_entropy:
                if args.dataset == "awa2":
                    labels_pred = F.sigmoid(y_u)
                    entropy = -torch.sum(labels_pred * torch.log(labels_pred), dim=1)
                else:
                    labels_pred = F.softmax(y_u, dim=1)
                    entropy = -torch.sum(labels_pred * torch.log(labels_pred), dim=1)
                if counter >= 10:
                    loss_entropy = args.unl_weight * torch.mean(entropy)
                    loss += loss_entropy

            elif args.semantic_loss:
                if args.dataset == "awa2":
                    labels_pred = F.sigmoid(y_u)
                else:
                    labels_pred = F.softmax(y_u, dim=1)
                part1 = torch.stack([labels_pred**all_labels[i] for i in range(all_labels.shape[0])])
                part2 = torch.stack([(1-labels_pred)**(1-all_labels[i]) for i in range(all_labels.shape[0])])
                sem_loss = -torch.log(torch.sum(torch.prod(part1 * part2, dim=2), dim=0))
                if counter >= 10:
                    semantic_loss = args.unl_weight * torch.mean(sem_loss)
                    loss += semantic_loss


            elif args.lp:
                model_y.eval()
                if args.dataset == "awa2":
                    labels_pred = F.sigmoid(y_u)
                else:
                    labels_pred = F.softmax(y_u, dim=1)
                if num_classes%2:
                    labels_pred = torch.cat((labels_pred, torch.zeros((labels_pred.shape[0], 1)).to("cuda:0")), dim=1)
                _, nll_ypred = model_y(labels_pred)
                if counter >= 10:
                    loss_nll_ypred = args.unl_weight * torch.mean(nll_ypred)
                    loss += loss_nll_ypred
                
                model_y.train()
                optimizer_y.zero_grad()
                if args.dataset == "awa2":
                    a = targets_l.float() * 120. + (1-targets_l.float()) * 1.1
                    b = (1-targets_l.float()) * 120. + targets_l.float() * 1.1
                    beta_targets = Beta(a, b).rsample()
                    if num_classes%2:
                        beta_targets = torch.cat((beta_targets, torch.zeros((beta_targets.shape[0], 1)).to("cuda:0")), dim=1)
                    zs, nll_y = model_y(beta_targets)
                else:
                    one_hot_targets = F.one_hot(torch.tensor(targets_l), num_classes).float()
                    one_hot_targets = one_hot_targets * 120 + (1-one_hot_targets) * 1.1
                    dirichlet_targets = torch.stack([Dirichlet(i).sample() for i in one_hot_targets])
                    zs, nll_y = model_y(dirichlet_targets)
                loss_nll_y = torch.mean(nll_y)
                loss_nll_y.backward()
                optimizer_y.step()
            return loss, y_l
    
    def compute_loss_test(sample):
        inputs = cast(sample[0], args.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(model, inputs, params, sample[2], list(range(args.ngpu))).float()
        if args.dataset == "awa2":
            return F.binary_cross_entropy_with_logits(y, targets.float()), y
        else:
            return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params=params, epoch=t['epoch'], optimizer=state['optimizer'].state_dict()),
                   os.path.join(args.save, 'model.pt7'))
        z = {**vars(args), **t}
        with open(os.path.join(args.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        if args.dataset == "awa2":
            if not args.ssl or not state['train']:
                acc = calculate_accuracy(F.sigmoid(state['output'].data), state['sample'][1])
            else:
                acc = calculate_accuracy(F.sigmoid(state['output'].data), state['sample'][0][1])
            classacc.add(acc)
        else:
            if not args.ssl or not state['train']:
                classacc.add(state['output'].data, state['sample'][1])
            else:
                classacc.add(state['output'].data, state['sample'][0][1])
        meter_loss.add(loss)

        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(args, lr * args.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()[0]
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()


        with torch.no_grad():
            engine.test(compute_loss_test, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc,
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
          (args.save, state['epoch'], args.epochs, test_acc))
    
        
        global counter
        counter += 1
        

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(compute_loss, train_loader, args.epochs, optimizer)


if __name__ == '__main__':
    main()
