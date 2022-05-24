from __future__ import division
from __future__ import print_function

import argparse
import os
import winsound

import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from config import Config
from models import GMA_GCN
from utils import *

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, default="citeseer")
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, default=20)
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()
    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

    # load data
    sadj, fadj_1, fadj_2, fadj_3, ppmi = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)
    # idx_train: 120 240 360  idx_test: 1000 # 20, 40, 60 labeled nodes per class
    model = GMA_GCN(nfeat=config.fdim,
                    nhid1=config.nhid1,
                    nhid2=config.nhid2,
                    nclass=config.class_num,
                    n=config.n,
                    dropout=config.dropout)  #
    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj_1 = fadj_1.cuda()
        fadj_2 = fadj_2.cuda()
        fadj_3 = fadj_3.cuda()
        ppmi = ppmi.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # train
    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output, fadj, str, sem, emb = model(features, sadj, fadj_1, fadj_2, fadj_3, ppmi)
        loss_class = F.nll_loss(output[idx_train], labels[idx_train])
        if config.beta == 0 and config.theta == 0:
            loss = loss_class
        else:
            loss_com_1 = common_loss(str, sem)
            loss_com_2 = common_loss(str, fadj)
            loss = loss_class + config.beta * loss_com_1 + config.theta * loss_com_2
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model)

        print('e:{}'.format(epochs),
              'ltr: {:.4f}'.format(loss.item()),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()),
              )
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test

    # test
    def main_test(model):
        model.eval()
        output, fadj, str, sem, emb = model(features, sadj, fadj_1, fadj_2, fadj_3, ppmi)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb


    #
    acc_max = 0
    f1_max = 0
    epoch_max = 0

    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch

    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max))

winsound.PlaySound("SystemExit", winsound.SND_ALIAS)
