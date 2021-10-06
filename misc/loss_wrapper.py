import torch, pdb
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gts_zh, gt_indices,
                sc_flag,labels_reverse, labels_zh=None, labels_reverse_zh=None, masks_zh=None):
        out = {}
        # pdb.set_trace()
        if not sc_flag:
            if self.opt.mode == 'pair':
                if self.opt.zh_reverse:
                    output = self.model(fc_feats, att_feats, labels, att_masks, labels_reverse_zh)
                    loss = (self.crit(output[0], labels[:,1:], masks[:,1:]) + self.crit(output[1], labels_reverse_zh[:,1:], masks_zh[:,1:]))/2
                else:
                    output = self.model(fc_feats, att_feats, labels, att_masks, labels_zh)
                    loss = (self.crit(output[0], labels[:,1:], masks[:,1:]) + self.crit(output[1], labels_zh[:,1:], masks_zh[:,1:]))/2
            else:
                loss = self.crit(self.model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            # pdb.set_trace()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            if self.opt.mode == 'pair':
                gts_zh = [gts_zh[_] for _ in gt_indices.tolist()]
        
            reward = get_self_critical_reward(self.model, fc_feats, att_feats, att_masks, gts, gts_zh, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().to(gen_result.device)
            if self.opt.mode == 'pair':
                sample_logprobs = sample_logprobs.view(-1,sample_logprobs.size(-1))
                gen_result = gen_result.view(-1, gen_result.size(-1))
                reward = reward.view(-1, reward.size(-1))

            loss = self.rl_crit(sample_logprobs, gen_result, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
