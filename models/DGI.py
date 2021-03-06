# Code based on https://github.com/PetarV-/DGI/blob/master/models/dgi.py
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator
import numpy as np
np.random.seed(0)
from evaluate import evaluate

class DGI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self,f):
        features_lst = [feature.to(self.args.device) for feature in self.features]
        adj_lst = [adj_.to(self.args.device) for adj_ in self.adj]

        final_embeds = []
        each_acc=[]
        each_nmi = []
        each_ari = []
        for m_idx, (features, adj) in enumerate(zip(features_lst, adj_lst)):
            metapath = m_idx+1
            print("- Training on {}".format(metapath))
            self.args.m_idx=m_idx
            model = modeler(self.args).to(self.args.device)

            optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
            cnt_wait = 0; best = 1e9
            b_xent = nn.BCEWithLogitsLoss()

            accMax = -1
            for epoch in range(self.args.nb_epochs):
                model.train()
                optimiser.zero_grad()

                idx = np.random.permutation(self.args.nb_nodes)
                shuf_fts = features[:, idx, :].to(self.args.device)

                lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
                lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                lbl = lbl.to(self.args.device)

                logits = model(features, shuf_fts, adj, self.args.sparse, None, None, None)

                loss = b_xent(logits, lbl)


                if loss < best:
                    best = loss
                    cnt_wait = 0

                else:
                    cnt_wait += 1

                if cnt_wait == self.args.patience:
                    break

                loss.backward()
                optimiser.step()
                if(epoch%10==0):
                    with torch.no_grad():
                        embeds, _ = model.embed(features, adj, self.args.sparse)
                        nmi, acc, ari, stdacc, stdnmi, stdari = evaluate(embeds, self.idx_train, self.labels,
                                                                         self.args.device)
                        retxt = "metapath={} loss:{} epoch:{} acc:{} nmi:{} ari:{} curepoch:{}".format(metapath, loss.item(),
                                                                                                 epoch, acc, nmi,ari, epoch)
                        print(retxt)

                        if (accMax < acc):
                            model.eval()
                            accMax = acc

                            torch.save(model.state_dict(),
                                   'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, metapath))

            model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, metapath)))

            # Evaluation
            embeds, _ = model.embed(features, adj, self.args.sparse)
            nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(embeds, self.idx_train, self.labels, self.args.device)
            retxt = "metapath={} loss:{} epoch:{} acc:{} nmi:{} ari:{} curepoch:{}".format(metapath,loss.item(),epoch, acc, nmi,ari,epoch)
            print(retxt)
            f.write(retxt)
            f.write("\n")
            final_embeds.append(embeds)

        embeds = torch.mean(torch.cat(final_embeds), 0).unsqueeze(0)
        print("- Integrated")
        nmi,acc,ari,stdacc,stdnmi,stdari=evaluate(embeds, self.idx_train, self.labels, self.args.device)
        retxt1 = "alll_cat=loss:{} epoch:{} acc:{} nmi:{} ari:{}  curepoch:{}".format(loss.item(), epoch, acc, nmi,ari, epoch)
        print(retxt1)

        f.write(retxt1)
        f.write("\n")

        embeds = torch.cat(final_embeds,2)
        print("- Integrated")
        nmi, acc, ari, stdacc, stdnmi, stdari = evaluate(embeds, self.idx_train, self.labels, self.args.device)
        retxt2 = "alll_cat=loss:{} epoch:{} acc:{} nmi:{} ari:{}  curepoch:{}".format(loss.item(), epoch, acc, nmi,ari, epoch)
        print(retxt2)

        return nmi,acc,ari,stdacc,stdnmi,stdari,retxt2


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = GCN(args.ft_size[args.m_idx], args.hid_units, args.activation, args.drop_prob, args.isBias)

        # one discriminator
        self.disc = Discriminator(args.hid_units)
        self.readout_func = self.args.readout_func

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        c = self.readout_func(h_1)  # equation 9
        c = self.args.readout_act_func(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)

        c = self.readout_func(h_1)  # positive summary vector
        c = self.args.readout_act_func(c)  # equation 9

        return h_1.detach(), c.detach()
