from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch

import sys, pdb
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens, mode ='en'):
    global CiderD_scorer
    global Bleu_scorer

    if mode == 'en':
        # global CiderD_scorer
        CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens[0])
        # global Bleu_scorer
        Bleu_scorer = Bleu_scorer or Bleu(4)
    elif mode == 'zh':
        # global CiderD_scorer
        CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens[1])
        # global Bleu_scorer
        Bleu_scorer = Bleu_scorer or Bleu(4)
    elif mode == 'pair':
        CiderD_scorer = []
        for i in range(len(cached_tokens)):
            CiderD_scorer_i =  CiderD(df=cached_tokens[i])
            CiderD_scorer.append(CiderD_scorer_i)
        # global CiderD_scorer

        # global Bleu_scorer
        Bleu_scorer = Bleu_scorer or Bleu(4)


def array_to_str(arr, idx, r2l=False, cbt=False):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    if r2l:
        return ' '.join(out.strip().split()[::-1])
    elif cbt:
        if idx%2 == 0:
            return out.strip()
        else:
            return ' '.join(out.strip().split()[::-1])
    else:
        return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data_gts, data_gts_zh,  gen_result, opt):
    # pdb.set_trace()
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)

    
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')
    model.train()


    res = OrderedDict()
    if opt.mode == 'pair':
        res_zh = OrderedDict()
    
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    if opt.mode == 'pair':
        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i,0],i)]
        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_res[i,0],i)]

        for i in range(batch_size):
            res_zh[i] = [array_to_str(gen_result[i,1],i)]
        for i in range(batch_size):
            res_zh[batch_size + i] = [array_to_str(greedy_res[i,1],i)]
    else:
        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i],i)]
        for i in range(batch_size):
            res[batch_size + i] = [array_to_str(greedy_res[i],i)]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j],j) for j in range(len(data_gts[i]))]

    if opt.mode == 'pair':
        gts_zh = OrderedDict()
        for i in range(len(data_gts_zh)):
            gts_zh[i] = [array_to_str(data_gts_zh[i][j],j) for j in range(len(data_gts_zh[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if opt.mode == 'pair':
        res__zh = [{'image_id':i, 'caption': res_zh[i]} for i in range(2 * batch_size)]
        res___zh = {i: res_zh[i] for i in range(2 * batch_size)}
        gts_zh = {i: gts_zh[i % batch_size // seq_per_img] for i in range(2 * batch_size)}


    if opt.cider_reward_weight > 0:
        if opt.mode == 'pair':
            _, cider_scores = CiderD_scorer[0].compute_score(gts, res_)
            print('Cider scores:', _)

            _, cider_scores_zh = CiderD_scorer[1].compute_score(gts_zh, res__zh)
            print('Zh Cider scores:', _)
        else:
            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            print('Cider scores:', _)
    else:
        cider_scores = 0
        if opt.mode == 'pair':
            cider_scores_zh = 0

    if opt.bleu_reward_weight > 0:
        if opt.mode == 'pair':
            _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
            bleu_scores = np.array(bleu_scores[3])
            print('Bleu scores:', _[3])

            _, bleu_scores_zh = Bleu_scorer.compute_score(gts_zh, res___zh)
            bleu_scores_zh = np.array(bleu_scores_zh[3])
            print('Zh Bleu scores:', _[3])
        else:
            _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
            bleu_scores = np.array(bleu_scores[3])
            print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
        if opt.mode == 'pair':
            bleu_scores_zh = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    if opt.mode == 'pair':
        scores_zh = opt.cider_reward_weight * cider_scores_zh + opt.bleu_reward_weight * bleu_scores_zh

    # scores = scores[:batch_size] - scores[batch_size:]
    if opt.nsc:
        scores = scores[:batch_size].reshape(-1,seq_per_img)
        baseline = (scores.sum(1, keepdims=True) - scores) / (scores.shape[1] - 1)
        scores = (scores - baseline).reshape(-1)

        if opt.mode == 'pair':
            scores_zh = scores_zh[:batch_size].reshape(-1,seq_per_img)
            baseline = (scores_zh.sum(1, keepdims=True) - scores_zh) / (scores_zh.shape[1] - 1)
            scores_zh = (scores_zh - baseline).reshape(-1)

    else:
        scores = scores[:batch_size] - scores[batch_size:]
        if opt.mode == 'pair':
            scores_zh = scores_zh[:batch_size] - scores_zh[batch_size:]

    if opt.mode == 'pair':
        rewards_en = np.repeat(scores[:, np.newaxis], gen_result.shape[2], 1)
        rewards_zh = np.repeat(scores_zh[:, np.newaxis], gen_result.shape[2], 1)
        rewards = np.stack((rewards_en, rewards_zh),axis=1)

    else:
        rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
