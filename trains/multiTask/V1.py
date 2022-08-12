import os
import time
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop
from torch.autograd import Variable
logger = logging.getLogger('MSA')

class V1():
    def __init__(self, args):
        assert args.datasetName == 'sims3' or args.datasetName == 'sims3l'

        self.args = args
        self.args.tasks = "MTAV"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        
        # initilize results
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = model(text, audio, vision)

                    loss_A_mix = 0.0
                    audio_mix, targets_a, targets_b, targets_a_m, targets_b_m, lam = mixup_data_no_grad(audio, labels['A'], labels['M'])
                    audio_mix, targets_a, targets_b = map(Variable, (audio_mix, targets_a, targets_b))
                    outputs_A_mix = model(text, audio_mix, vision)
                    fusion_mix, targets_a_m, targets_b_m = map(Variable, (outputs_A_mix['Feature_f'], targets_a_m, targets_b_m))
                    loss_A_mix += self.args.A * mixup_criterion(self.criterion, outputs_A_mix['A'], targets_a, targets_b, lam).squeeze()
                    loss_A_mix += self.args.M * mixup_criterion(self.criterion, outputs_A_mix['M'], targets_a_m, targets_b_m, lam).squeeze()
                    for m in 'TV':
                        loss_A_mix += eval('self.args.'+m) * self.criterion(outputs_A_mix[m], labels[m])

                    loss_V_mix = 0.0
                    vision_mix, targets_a2, targets_b2, targets_a2_m, targets_b2_m, lam2 = mixup_data_no_grad(vision, labels['V'], labels['M'])
                    vision_mix, targets_a2, targets_b2 = map(Variable, (vision_mix, targets_a2, targets_b2))
                    outputs_V_mix = model(text, audio, vision_mix)
                    fusion_mix2, targets_a2_m, targets_b2_m = map(Variable, (outputs_V_mix['Feature_f'], targets_a2_m, targets_b2_m))
                    loss_V_mix += self.args.V * mixup_criterion(self.criterion, outputs_V_mix['V'], targets_a2, targets_b2, lam2).squeeze()
                    loss_V_mix += self.args.M * mixup_criterion(self.criterion, outputs_V_mix['M'], targets_a2_m, targets_b2_m, lam2).squeeze()
                    for m in 'TA':
                        loss_V_mix += eval('self.args.'+m) * self.criterion(outputs_V_mix[m], labels[m])

                    # compute loss
                    loss = 0.0
                    loss += loss_A_mix
                    loss += loss_V_mix

                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
            train_loss = train_loss / len(dataloader['train'])

            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs - best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    outputs = model(text, audio, vision)
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.'+m) * self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
        eval_loss = round(eval_loss / len(dataloader), 4)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        eval_results = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            logger.info('%s: >> ' %(m) + dict_to_str(results))
            eval_results[m] = results
        eval_results = eval_results[self.args.tasks[0]]
        eval_results['Loss'] = eval_loss
        return eval_results

def mixup_data_no_grad(x, y, y_m, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
      lam   = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    y_m_a, y_m_b = y_m, y_m[index]
    return mixed_x, y_a, y_b, y_m_a, y_m_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)