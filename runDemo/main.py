import numpy as np

from utils.utils import mkdir

np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import os
import yaml

def runMain(dataset):
    config = yaml.load(open("configGAT.yaml", 'r'))
    dataset_name = dataset

    # input arguments
    parser = argparse.ArgumentParser(description='DMGI')
    parser.add_argument('--embedder', nargs='?', default='DMGI')
    parser.add_argument('--nb_epochs', type=int, default=config[dataset_name]['nb_epochs'])
    parser.add_argument('--sc', type=float, default=config[dataset_name]['sc'], help='GCN self connection')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=config[dataset_name]['isAttn'])
    parser.add_argument('--isMeanOrCat', nargs='?', default=config[dataset_name]['isMeanOrCat'])
    parser.add_argument('--Weight', nargs='?', default=config['Weight'])
    args, unknown = parser.parse_known_args()

    resultsDir = 'baseline/DMGI/{}'.format(dataset_name)
    mkdir(resultsDir)

    args.View_num = config[dataset_name]['View_num']
    args.norm = config[dataset_name]['norm']

    args.rownetworks = None
    # filePath = os.path.join(resultsDir,
    #                         '{}_hid-{}_lr-{}reg_coef-{}_isMeanOrCat-{}_isAttn-{}.txt'.format(dataset_name, hid, lr,
    #                                                                                          reg_coef, args.isMeanOrCat,
    #                                                                                          args.isAttn))
    filePath = os.path.join(resultsDir, '{}_{}.txt'.format(dataset_name,config[dataset_name]['isMeanOrCat']))
    with open(filePath, 'a+') as f:
        for lr in config[dataset_name]['lr']:
            for l2_coef in config[dataset_name]['l2_coef']:
                for reg_coef in config[dataset_name]['reg_coef']:
                    for hid in config[dataset_name]['hid']:
                        args.lr = lr
                        args.hid_units = hid
                        args.l2_coef = l2_coef
                        args.dataset = dataset_name
                        args.reg_coef = reg_coef

                        print(args)

                        from models import DMGI
                        embedder = DMGI(args)

                        nmi,acc,ari,stdacc,stdnmi,stdari,retxt = embedder.training(f)
                        result = "hid:{:d},lr:{},l2_coef:{:.4f},reg_coef:{:.4f},acc:{:.4f},nmi:{:.4f},stdnmi:{},stdacc:{}".format(
                            hid, lr, l2_coef, reg_coef, acc, nmi, stdacc, stdnmi)

                        f.write(retxt)
                        f.write('\n')
                        f.write(result)
                        f.write('\n')
                        f.write('\n')
                        f.flush()

if __name__ == '__main__':
    d=['BBC','BBCSport','small_Reuters','Reuters','MSRCv1','3sources','HW','small_NUS','yale_mtv']
    for data in ['BBCSport']:
        runMain(data)

