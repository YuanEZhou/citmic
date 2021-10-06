# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from functools import reduce
import copy
import math, pdb
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, mode):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.mode = mode
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        if self.mode == 'pair':
            w_emb_0 = self.tgt_embed[0](tgt[:,0,:])
            w_emb_1 = self.tgt_embed[1](tgt[:,1,:])
            w_emb = torch.stack((w_emb_0,w_emb_1),dim=1).view(-1,w_emb_0.size(1),w_emb_0.size(2))
            tgt_mask = tgt_mask.view(-1,tgt_mask.size(-2),tgt_mask.size(-1))

            memory = torch.stack((memory,memory),dim=1).view(-1,memory.size(-2),memory.size(-1))
            src_mask = torch.stack((src_mask,src_mask),dim=1).view(-1,src_mask.size(-2),src_mask.size(-1))
        else:
            w_emb = self.tgt_embed(tgt)
        return self.decoder(w_emb, memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, opt):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.opt = opt
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def attention_cb(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # left context
    d_k = query.size(-1)
    scores_left = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores_left = scores_left.masked_fill(mask == 0, -1e9)
    p_attn_left = F.softmax(scores_left, dim = -1)
    if dropout is not None:
        p_attn_left = dropout(p_attn_left)
    left =  torch.matmul(p_attn_left, value)

    # right context 
    key_flip = torch.flip(key, [1])
    value_flip = torch.flip(value, [1])
    scores_right = torch.matmul(query, key_flip.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores_right = scores_right.masked_fill(mask == 0, -1e9)
    p_attn_right = F.softmax(scores_right, dim = -1)
    if dropout is not None:
        p_attn_right = dropout(p_attn_right)
    right =  torch.matmul(p_attn_right, value_flip)

    #combine
    combine = left + 0.1*torch.tanh(right)
    return combine, torch.cat((p_attn_left.unsqueeze(0),p_attn_right.unsqueeze(0)),dim=0)

def attention_ci(query, key, value, mask=None, dropout=None, lang_inter_weight=0, lang_inter_af='relu'):
    "Compute 'Scaled Dot Product Attention'"
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    context_homo = torch.matmul(p_attn, value)


    key_flip = torch.flip(key.view(-1, 2, key.size(-3), key.size(-2), key.size(-1)), [1]).view(-1,key.size(-3), key.size(-2), key.size(-1))
    value_flip = torch.flip(value.view(-1, 2, value.size(-3), value.size(-2), value.size(-1)), [1]).view(-1, value.size(-3), value.size(-2), value.size(-1))
    scores_lang_inter = torch.matmul(query, key_flip.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores_lang_inter = scores_lang_inter.masked_fill(mask == 0, -1e9)
    p_attn_lang_inter = F.softmax(scores_lang_inter, dim = -1)
    if dropout is not None:
        p_attn_lang_inter = dropout(p_attn_lang_inter)
    context_heter =  torch.matmul(p_attn_lang_inter, value_flip)

    #combine
    if lang_inter_af == 'relu':
        combine = context_homo + lang_inter_weight * torch.relu(context_heter)
    elif lang_inter_af == 'tanh':
        combine = context_homo + lang_inter_weight * torch.tanh(context_heter)
    elif lang_inter_af == 'linear':
        combine = context_homo + lang_inter_weight * context_heter
    return combine, torch.cat((p_attn.unsqueeze(0), p_attn_lang_inter.unsqueeze(0)),dim=0)




class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttention_CI(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lang_inter_weight=0, lang_inter_af='relu'):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_CI, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.lang_inter_weight = lang_inter_weight
        self.lang_inter_af = lang_inter_af
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # pdb.set_trace()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention_ci(query, key, value, mask=mask, 
                                 dropout=self.dropout, lang_inter_weight=self.lang_inter_weight, lang_inter_af=self.lang_inter_af)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttention_CB(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_CB, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(2)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, 2, -1, self.h, self.d_k).transpose(2, 3)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention_cb(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(2, 3).contiguous() \
             .view(nbatches, 2, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, opt, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.opt = opt
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CIT(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1, tgt_vocab_zh=0):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        if self.opt.mode == 'pair':
            attn_ci = MultiHeadedAttention_CI(h, d_model, dropout=0.1, lang_inter_weight=self.opt.lang_inter_weight, lang_inter_af=self.opt.lang_inter_af)

        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, self.opt)


        if self.opt.mode == 'pair':
            model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                Decoder(DecoderLayer(d_model, c(attn_ci), c(attn), 
                                    c(ff), dropout, self.opt), N-5),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                nn.ModuleList([nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), nn.Sequential(Embeddings(d_model, tgt_vocab_zh), c(position))]),
                nn.ModuleList([Generator(d_model, tgt_vocab), Generator(d_model, tgt_vocab_zh)]),
                self.opt.mode)
        else:
            model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                    c(ff), dropout, self.opt), N-5),
                lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                Generator(d_model, tgt_vocab),
                self.opt.mode)
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        # pdb.set_trace()
        return model

    def __init__(self, opt):
        super(CIT, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
        tgt_vocab_zh = 0
        if self.opt.mode == 'pair':
            self.vocab_size_zh = self.opt.vocab_size_zh
            tgt_vocab_zh = self.vocab_size_zh + 1

        self.model = self.make_model(0, tgt_vocab,
            N=opt.num_layers,
            d_model=opt.input_encoding_size,
            d_ff=opt.rnn_size,
            tgt_vocab_zh = tgt_vocab_zh)


    def init_hidden(self, bsz):
        return None

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:1], att_feats[...,:1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None, seq_zh = None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:   
            
            # crop the last one
            seq = seq[:,:-1]
            seq_mask = (seq.data > 0)
            # seq_mask[:,0] += 1
            seq_mask[:,0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            if self.opt.mode == 'pair':
                # crop the last one
                seq_zh = seq_zh[:,:-1]
                seq_mask_zh = (seq_zh.data > 0)
                # seq_mask[:,0] += 1
                seq_mask_zh[:,0] += True

                seq_mask_zh = seq_mask_zh.unsqueeze(-2)
                seq_mask_zh = seq_mask_zh & subsequent_mask(seq_zh.size(-1)).to(seq_mask_zh)

                seq = torch.stack((seq,seq_zh),dim=1)
                seq_mask = torch.stack((seq_mask,seq_mask_zh),dim=1)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, seq_zh=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq, seq_zh)

        out = self.model(att_feats, seq, att_masks, seq_mask)
        
        if self.opt.mode == 'pair':
            out = out.view(seq.size(0), seq.size(1),seq.size(2),-1)
            outputs_0 = self.model.generator[0](out[:,0,:])
            outputs_1 = self.model.generator[1](out[:,1,:])
            outputs = [outputs_0,outputs_1]
        else:
            outputs = self.model.generator(out)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
            if self.opt.mode == 'pair':
                ys = torch.stack((ys,ys),dim=1)
        else:
            if self.opt.mode == 'pair':
                ys = torch.cat([state[0][0], it.unsqueeze(2)], dim=2)
            else:
                ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        if self.opt.mode == 'pair':
            out = self.model.decode(memory, mask, 
                                ys, 
                                subsequent_mask(ys.size(2))
                                            .to(memory.device))
        else:
            out = self.model.decode(memory, mask, 
                                ys, 
                                subsequent_mask(ys.size(1))
                                            .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]


    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if self.opt.mode == 'pair':
            outputs = self.logit(output)
            logprobs = list(map(nn.LogSoftmax(dim=-1), outputs))
        else:
            logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    def logit(self, x): # unsafe way
        if self.opt.mode == 'pair':
            x = x.view(-1, 2, x.size(-1))
            outputs_0 = self.model.generator[0](x[:,0,:])
            outputs_1 = self.model.generator[1](x[:,1,:])
            outputs = [outputs_0,outputs_1]
            return outputs
        else:
            return self.model.generator.proj(x)


    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams = [] # will be a list of batch_size dictionaries

        if self.opt.mode == 'pair':
            seq = fc_feats.new_zeros((batch_size, 2, self.seq_length), dtype=torch.long)
            seqLogprobs = fc_feats.new_zeros(batch_size, 2, self.seq_length)
        else:
            seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
            seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # Mess with trigrams
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                if self.opt.mode == 'pair':
                    sampleLogprobs_tmp = []
                    it_tmp = []
                    for logprob in logprobs:
                        sampleLogprob, it = torch.max(logprob, 1)
                        it = it.view(-1).long()
                        sampleLogprobs_tmp.append(sampleLogprob)
                        it_tmp.append(it)

                    sampleLogprobs = torch.stack(sampleLogprobs_tmp, dim=1)
                    it = torch.stack(it_tmp, dim=1)

                else:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
            else:
                if self.opt.mode == 'pair':
                    if temperature == 1.0:
                        prob_prev = [torch.exp(logprob.data) for logprob in  logprobs] # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = [torch.exp(torch.div(logprob.data, temperature)) for logprob in  logprobs]
                    it = [torch.multinomial(prob_prev_each, 1) for prob_prev_each in prob_prev]
                    sampleLogprobs = [logprobs[i].gather(1, it[i]).view(-1) for i in range(len(logprobs))] # gather the logprobs at sampled positions
                    it = [it[i].view(-1).long() for i in range(len(it))] # and flatten indices for downstream processing
                    sampleLogprobs = torch.stack(sampleLogprobs, dim=1)
                    it = torch.stack(it, dim=1)
                else:
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                    it = torch.multinomial(prob_prev, 1)
                    sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                    it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                if self.opt.mode == 'pair':
                    unfinished = torch.any(it > 0, dim = 1, keepdim=True)
                else:
                    unfinished = it > 0
            else:
                if self.opt.mode == 'pair':
                    unfinished = unfinished * torch.any(it > 0, dim = 1, keepdim=True)
                else:
                    unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)

            if self.opt.mode == 'pair':
                seq[:,:,t] = it
                seqLogprobs[:,:,t] = sampleLogprobs
            else:
                seq[:,t] = it
                seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        # pdb.set_trace()
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        # seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        if self.opt.mode == 'pair':
            seq = torch.LongTensor(batch_size, 2, self.seq_length).zero_()
            seqLogprobs = torch.FloatTensor(batch_size, 2, self.seq_length)
        else:
            seq = torch.LongTensor(batch_size, self.seq_length).zero_()
            seqLogprobs = torch.FloatTensor(batch_size, self.seq_length)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            # seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            # seqLogprobs[:, k] = self.done_beams[k][0]['logps']
            if self.opt.mode == 'pair':
                seq[k,0,:] = self.done_beams[k][0][0]['seq']
                seq[k,1,:] = self.done_beams[k][1][0]['seq'] 
                seqLogprobs[k,0,:] = self.done_beams[k][0][0]['logps']
                seqLogprobs[k,1,:] = self.done_beams[k][1][0]['logps'] 
            else:
                seq[k, :] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        # return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
        return seq, seqLogprobs


    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        max_ppl = opt.get('max_ppl', 0)
        length_penalty = utils.penalty_builder(opt.get('length_penalty', ''))
        bdash = beam_size // group_size # beam per group

        if opt['mode'] == 'pair':
            assert group_size == 1, 'following implementation only supports group_size=1'

            # INITIALIZATIONS
            beam_seq_table = [torch.LongTensor(self.seq_length, bdash, 2).zero_() for _ in range(group_size)]
            beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash, 2).zero_() for _ in range(group_size)]
            beam_logprobs_sum_table = [torch.zeros(bdash, 2) for _ in range(group_size)]

            # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
            done_beams_table = [[[],[]] for _ in range(group_size)]
            state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
            # logprobs_table = list(init_logprobs.chunk(group_size, 0))
            logprobs_table = list(init_logprob.chunk(group_size, 0) for init_logprob in init_logprobs)
            # END INIT

            # Chunk elements in the args
            args = list(args)
            args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

            for t in range(self.seq_length + group_size - 1):
                for divm in range(group_size): 
                    if t >= divm and t <= self.seq_length + divm - 1:
                        # add diversity
                        logprobsf = [logprobs_table[i][divm].data.float()  for i in range(len(logprobs_table))]
                        # suppress previous word
                        if decoding_constraint and t-divm > 0:
                            logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                        # suppress UNK tokens in the decoding
                        for j in range(len(logprobsf)):
                            logprobsf[j][:,logprobsf[j].size(1)-1] = logprobsf[j][:, logprobsf[j].size(1)-1] - 1000  
                        # diversity is added here
                        # the function directly modifies the logprobsf values and hence, we need to return
                        # the unaugmented ones for sorting the candidates in the end. # for historical
                        # reasons :-)
                        # unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)
                        unaug_logprobsf = [item.clone() for item in logprobsf]

                        for i in range(len(unaug_logprobsf)):
                            # infer new beams
                            beam_seq_table[divm][:,:,i],\
                            beam_seq_logprobs_table[divm][:,:,i],\
                            beam_logprobs_sum_table[divm][:,i],\
                            [state_table[divm][0][:,:,i,:]],\
                            candidates_divm = beam_step(logprobsf[i],
                                                        unaug_logprobsf[i],
                                                        bdash,
                                                        t-divm,
                                                        beam_seq_table[divm][:,:,i],
                                                        beam_seq_logprobs_table[divm][:,:,i],
                                                        beam_logprobs_sum_table[divm][:,i],
                                                        [state_table[divm][0][:,:,i,:]])

                            # if time's up... or if end token is reached then copy beams
                            for vix in range(bdash):
                                if beam_seq_table[divm][t-divm,vix, i] == 0 or t == self.seq_length + divm - 1:
                                    final_beam = {
                                        'seq': beam_seq_table[divm][:, vix, i].clone(), 
                                        'logps': beam_seq_logprobs_table[divm][:, vix, i].clone(),
                                        'unaug_p': beam_seq_logprobs_table[divm][:, vix, i].sum().item(),
                                        'p': beam_logprobs_sum_table[divm][vix,i].item()
                                    }
                                    final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                    # if max_ppl:
                                    #     final_beam['p'] = final_beam['p'] / (t-divm+1)
                                    done_beams_table[divm][i].append(final_beam)
                                    # don't continue beams from finished sequences
                                    beam_logprobs_sum_table[divm][vix,i] = -1000

                        # move the current group one step forward in time
                        
                        it = beam_seq_table[divm][t-divm]
                        logprobs_table_tmp, state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))
                        for k in range(len(logprobs_table_tmp)):
                            logprobs_table[k] = logprobs_table_tmp[k].chunk(group_size, 0)

            # # all beams are sorted by their log-probabilities
            # done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
            # done_beams = reduce(lambda a,b:a+b, done_beams_table)
            done_beams = []
            for i in range(group_size):
                for j in range(2):
                    done_beams.append(sorted(done_beams_table[i][j], key=lambda x: -x['p'])[:bdash])

            return done_beams
        else:
            # INITIALIZATIONS
            beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
            beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
            beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

            # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
            done_beams_table = [[] for _ in range(group_size)]
            state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
            logprobs_table = list(init_logprobs.chunk(group_size, 0))
            # END INIT

            # Chunk elements in the args
            args = list(args)
            args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
            args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

            for t in range(self.seq_length + group_size - 1):
                for divm in range(group_size): 
                    if t >= divm and t <= self.seq_length + divm - 1:
                        # add diversity
                        logprobsf = logprobs_table[divm].data.float()
                        # suppress previous word
                        if decoding_constraint and t-divm > 0:
                            logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                        # suppress UNK tokens in the decoding
                        logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
                        # diversity is added here
                        # the function directly modifies the logprobsf values and hence, we need to return
                        # the unaugmented ones for sorting the candidates in the end. # for historical
                        # reasons :-)
                        unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                        # infer new beams
                        beam_seq_table[divm],\
                        beam_seq_logprobs_table[divm],\
                        beam_logprobs_sum_table[divm],\
                        state_table[divm],\
                        candidates_divm = beam_step(logprobsf,
                                                    unaug_logprobsf,
                                                    bdash,
                                                    t-divm,
                                                    beam_seq_table[divm],
                                                    beam_seq_logprobs_table[divm],
                                                    beam_logprobs_sum_table[divm],
                                                    state_table[divm])

                        # if time's up... or if end token is reached then copy beams
                        for vix in range(bdash):
                            if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                                final_beam = {
                                    'seq': beam_seq_table[divm][:, vix].clone(), 
                                    'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                    'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                    'p': beam_logprobs_sum_table[divm][vix].item()
                                }
                                final_beam['p'] = length_penalty(t-divm+1, final_beam['p'])
                                # if max_ppl:
                                #     final_beam['p'] = final_beam['p'] / (t-divm+1)
                                done_beams_table[divm].append(final_beam)
                                # don't continue beams from finished sequences
                                beam_logprobs_sum_table[divm][vix] = -1000

                        # move the current group one step forward in time
                        
                        it = beam_seq_table[divm][t-divm]
                        logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))

            # all beams are sorted by their log-probabilities
            done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
            done_beams = reduce(lambda a,b:a+b, done_beams_table)
            return done_beams