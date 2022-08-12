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

class V1_Semi():
    def __init__(self, args):
        assert args.datasetName == 'sims3l'
        self.args = args
        self.args.tasks = "MTAV"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.recloss = nn.MSELoss()
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

            with tqdm(dataloader['train_mix']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    mask = batch_data['mask']
                    labels = batch_data['labels']
                    # clear gradient
                    optimizer.zero_grad()
                    flag = 'train'
                    # forward
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
                    # compute loss
                    loss = 0
                    # 1. Supvised Loss
                    labels_true = {}
                    outputs_true = {}
                    for k in labels.keys():
                        labels[k] = labels[k].to(self.args.device).view(-1, 1)
                        mask_index = torch.where(mask==1)
                        labels_true[k] = labels[k][mask_index]
                        outputs_true[k] = outputs[k][mask_index]
                    
                    for m in self.args.tasks:
                        if mask.sum()>0:
                            loss += eval('self.args.'+m) * self.criterion(outputs_true[m], labels_true[m])
                    
                    # 2. unSupvised Loss
                    flag = 'mix_train'
                    # text_utt = outputs['Feature_t']
                    # pret = outputs['T']
                    audio_utt = outputs['Feature_a']
                    prea = outputs['A']
                    video_utt = outputs['Feature_v']
                    prev = outputs['V']
                    
                    loss_V_mix = 0.0
                    video_utt, video_utt_chaotic, video_utt_mix, y, y2, ymix, lam = mixup_data(video_utt, prev)
                    x_v1 = model.Model.post_video_dropout(video_utt_mix)
                    x_v2 = F.relu(model.Model.post_video_layer_1(x_v1), inplace=True)
                    x_v3 = F.relu(model.Model.post_video_layer_2(x_v2), inplace=True)
                    output_video = model.Model.post_video_layer_3(x_v3)
                    loss_V_mix += self.args.V * self.criterion(output_video, ymix)

                    loss_A_mix = 0.0
                    audio_utt, audio_utt_chaotic, audio_utt_mix, y, y2, ymix, lam = mixup_data(audio_utt, prea)
                    x_a1 = model.Model.post_audio_dropout(audio_utt_mix)
                    x_a2 = F.relu(model.Model.post_audio_layer_1(x_a1), inplace=True)
                    x_a3 = F.relu(model.Model.post_audio_layer_2(x_a2), inplace=True)
                    output_audio = model.Model.post_audio_layer_3(x_a3)
                    loss_A_mix += self.args.A * self.criterion(output_audio, ymix)
                    
                    # backward
                    loss += loss_A_mix
                    loss += loss_V_mix
                    if mask.sum()>0:
                        loss.backward()
                        train_loss += loss.item()
                    # update
                    optimizer.step()
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs_true[m].cpu())
                        y_true[m].append(labels_true['M'].cpu())

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    flag = 'train'
                    # forward
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
                    # compute loss
                    loss = 0.0
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

            train_loss = train_loss / len(dataloader['train_mix'])

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
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    flag = 'train'
                    outputs = outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
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

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
      lam   = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    x2 = x[index, :]
    y2 = y[index]
    xmix = lam * x + (1 - lam) * x2
    ymix = lam * y + (1 - lam) * y2
    y, y2 = y, y[index]
    return x, x2, xmix, y, y2, ymix, lam

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