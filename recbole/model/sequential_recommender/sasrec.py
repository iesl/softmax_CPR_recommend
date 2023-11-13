# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import sys
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.n_facet_all = config['n_facet_all'] #added for mfs
        self.n_facet = config['n_facet'] #added for mfs
        self.n_facet_window = config['n_facet_window'] #added for mfs
        self.n_facet_hidden = min(config['n_facet_hidden'], config['n_layers']) #added for mfs
        self.n_facet_MLP = config['n_facet_MLP'] #added for mfs
        self.n_facet_context = config['n_facet_context'] #added for dynamic partioning
        self.n_facet_reranker = config['n_facet_reranker'] #added for dynamic partioning
        self.n_facet_emb = config['n_facet_emb'] #added for dynamic partioning
        self.efficient_mode = config['efficient_mode'] #added for mfs
        self.weight_mode = config['weight_mode'] #added for mfs
        self.context_norm = config['context_norm'] #added for mfs
        self.post_remove_context = config['post_remove_context'] #added for mfs
        self.partition_merging_mode = config['partition_merging_mode'] #added for mfs
        self.reranker_merging_mode = config['reranker_merging_mode'] #added for mfs
        self.reranker_CAN_NUM = [int(x) for x in str(config['reranker_CAN_NUM']).split(',')]
        #self.reranker_CAN_NUM = [500, 100, 20]
        #self.reranker_CAN_NUM = [100]
        #self.partition_merging_mode = 'add'
        #self.partition_merging_mode = 'replace'
        self.candidates_from_previous_reranker = True
        if self.weight_mode == 'max_logits':
            self.n_facet_effective = 1
        else:
            self.n_facet_effective = self.n_facet
            
        if self.efficient_mode == 'None':
            assert self.n_facet + self.n_facet_context + self.n_facet_reranker*len(self.reranker_CAN_NUM) + self.n_facet_emb == self.n_facet_all
        else:
            assert self.n_facet + self.n_facet_context + self.n_facet_reranker*len(self.reranker_CAN_NUM) + self.n_facet_emb <= self.n_facet_all
        assert self.n_facet_emb == 0 or self.n_facet_emb == 2
        assert self.n_facet_MLP <= 0 #-1 or 0
        assert self.n_facet_window <= 0
        self.n_facet_window = - self.n_facet_window
        self.n_facet_MLP = - self.n_facet_MLP
        self.softmax_nonlinear='None' #added for mfs
        self.only_compute_loss = True #added for mfs
        self.n_embd = self.hidden_size #added for mfs
        self.use_proj_bias = config['use_proj_bias'] #added for mfs
        #print("use_proj_bias", self.use_proj_bias)
        # for multiple input hidden states
        if self.n_facet_MLP > 0:
            hidden_state_input_ratio = 1 + self.n_facet_MLP #1 + 1
            self.MLP_linear = nn.Linear(self.n_embd * (self.n_facet_hidden * (self.n_facet_window+1) ), self.n_embd * self.n_facet_MLP) # (hid_dim*2) -> (hid_dim)
        else:            
            hidden_state_input_ratio = self.n_facet_hidden * (self.n_facet_window+1) #1 * (0+1)
        total_lin_dim = self.n_embd * hidden_state_input_ratio
        #small_value = 1
        self.project_arr = nn.ModuleList([nn.Linear(total_lin_dim, self.n_embd, bias=self.use_proj_bias) for i in range(self.n_facet_all)])

        self.project_emb = nn.Linear(self.n_embd, self.n_embd, bias=self.use_proj_bias)
        if len(self.weight_mode) > 0:
            self.weight_facet_decoder = nn.Linear(self.hidden_size * hidden_state_input_ratio, self.n_facet_effective)
            #self.weight_facet_decoder = nn.Linear(config.hidden_size * n_facet_hidden * (n_facet_window+1), n_facet)
            self.weight_global = nn.Parameter( torch.ones(self.n_facet_effective) )
        self.output_probs = True
        #self.c = 100
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            #self.loss_fct = nn.CrossEntropyLoss()
            self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=0) #modified for mfs
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        
        small_value = 0.0001
        #print("before init", self.project_arr[0].weight.data)
       # for i in range(self.n_facet_all):
       #     if self.use_proj_bias:
       #         self.project_arr[i].bias.data.zero_()
       #     #linear_weights = torch.zeros_like(self.project_arr[i].weight.data)
       #     linear_weights = small_value * (torch.rand((self.n_embd, total_lin_dim)) - 0.5 )

       #     # if i!= n_facet - 1:
       #     #     linear_weights = linear_weights + small_value * (torch.rand((config.n_embd, total_lin_dim)) - 0.5 )
       #     
       #     #if i < self.n_facet + self.n_facet_context + self.n_facet_reranker*len(self.reranker_CAN_NUM):
       #     #    linear_weights[:,:self.n_embd] = torch.eye(self.n_embd)
       #     
       #     #if self.partition_merging_mode == 'add':
       #     #    if i < self.n_facet:
       #     #        linear_weights[:,:self.n_embd] = torch.eye(self.n_embd)
       #     #else:
       #     #    if i < self.n_facet + self.n_facet_context + self.n_facet_reranker*len(self.reranker_CAN_NUM):
       #     #        linear_weights[:,:self.n_embd] = torch.eye(self.n_embd)
       #     

       #     #if i < n_facet:
       #     #     linear_weights[:,:config.n_embd] = torch.eye(config.n_embd)
       #     # else:
       #     #     linear_weights[:,:config.n_embd] = 1e-10 * torch.eye(config.n_embd)
       #     self.project_arr[i].weight.data = linear_weights
        #print("after init", self.project_arr[0].weight.data)

    def get_facet_emb(self,input_emb, i):
        return self.project_arr[i](input_emb)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
    # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        #output = trm_output[-1]
        #output = self.gather_indexes(output, item_seq_len - 1)
        #return output  # [B H]
        return trm_output


    def calculate_loss_prob(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        all_hidden_states = self.forward(item_seq, item_seq_len)
        seq_output = all_hidden_states[-1]
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            '''mfs code starts'''
            device = all_hidden_states[0].device
            #check seq_len from hidden size

            ## Multi-input hidden states: generate q_ct from hidden states
            #list of hidden state embeddings taken as input
            hidden_emb_arr = []
            # h_facet_hidden -> H, n_face_window -> W, here 1 and 0
            for i in range(self.n_facet_hidden):
                #print('all_hidden_states length is {}. i is {}'.format(len(all_hidden_states), i))
                hidden_states = all_hidden_states[-(i+1)] #i-th hidden-state embedding from the top
                device = hidden_states.device
                hidden_emb_arr.append(hidden_states)
                for j in range(self.n_facet_window):
                    bsz, seq_len, hidden_size = hidden_states.size() #bsz -> , seq_len -> , hidden_size -> 768 in GPT-small?
                    if j+1 < hidden_states.size(1):
                        shifted_hidden = torch.cat( (torch.zeros( (bsz, (j+1), hidden_size), device = device), hidden_states[:,:-(j+1),:]), dim = 1)
                    else:
                        shifted_hidden = torch.zeros( (bsz, hidden_states.size(1), hidden_size), device = device)
                    hidden_emb_arr.append(shifted_hidden)
            #hidden_emb_arr -> (W*H, bsz, seq_len, hidden_size)
            

            #n_facet_MLP -> 1
            if self.n_facet_MLP > 0:
                stacked_hidden_emb_raw_arr = torch.cat(hidden_emb_arr, dim=-1) #(bsz, seq_len, W*H*hidden_size)
                # self.MLP_linear = nn.Linear(config.n_embd * (n_facet_hidden * (n_facet_window+1) ), config.n_embd * n_facet_MLP) -> why +1?
                hidden_emb_MLP = self.MLP_linear(stacked_hidden_emb_raw_arr) #bsz, seq_len, hidden_size
                stacked_hidden_emb_arr_raw = torch.cat([hidden_emb_arr[0], gelu(hidden_emb_MLP)], dim=-1) #bsz, seq_len, 2*hidden_size
            else:
                stacked_hidden_emb_arr_raw = hidden_emb_arr[0]
            
            #Only use the hidden state corresponding to the last word
            stacked_hidden_emb_arr = stacked_hidden_emb_arr_raw[:,-1,:].unsqueeze(dim=1)

            #list of linear projects per facet
            projected_emb_arr = []
            #list of final logits per facet
            facet_lm_logits_arr = []
            facet_lm_logits_real_arr = []

            #logits for orig facets
            if self.efficient_mode == 'even_last_2':
                #bsz, seq_len, hidden_size = all_hidden_states[-1].size()
                assert self.n_facet_context ==0
                assert self.n_facet_reranker == 0
                assert self.n_facet_emb == 0
                bsz, seq_len, hidden_size_cat = stacked_hidden_emb_arr.size()
                logit_all = torch.empty( (bsz, seq_len, self.n_items) , device=all_hidden_states[-1].device )
                n_facet_not_last = self.n_facet_all - (self.n_facet-1) # 6 - (3-1) = 4 -> partitions
                for i in range(n_facet_not_last):
                    #projected_emb = self.project_arr[i](stacked_hidden_emb_arr)
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,i) #bsz, seq_len, n_embd
                    # stacked_hidden_emb_arr -> sz, seq_len, 2*hidden_size
                    # same as project_arr? output_dim -> n_embd, 6 linear models, weights are zero for last one
                    projected_emb_arr.append(projected_emb) #4 partitions
                    #projected_emb_real_arr.append(projected_emb)
                    #self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
                    #if self.item_embedding.bias is None:
                    #    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.item_embedding.weight[i::n_facet_not_last,:], None)
                    #else:
                    #    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.item_embedding.weight[i::n_facet_not_last,:], self.item_embedding.bias[i::n_facet_not_last])
                    logit_all[:,:,i::n_facet_not_last] = F.linear(projected_emb, self.item_embedding.weight[i::n_facet_not_last,:], None)
                facet_lm_logits_arr.append(logit_all)
                #last two softmax, project_arr -> L^f
                for i in range(self.n_facet-1):
                    projected_emb = self.project_arr[-(i+1)](stacked_hidden_emb_arr)
                    #projected_emb = self.project_arr[n_facet_not_last+i](stacked_hidden_emb_arr)
                    #projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,n_facet_not_last+i)
                    projected_emb_arr.append(projected_emb)
                    #projected_emb_real_arr.append(projected_emb)

                    #facet_lm_logits_arr.append( self.item_embedding( projected_emb ) )
                    facet_lm_logits_arr.append(F.linear(projected_emb, self.item_embedding.weight, None))
            else:
                rereanker_candidate_token_ids_arr = []
                for i in range(self.n_facet):
                #     #linear projection
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i) #(bsz, seq_len, hidden_dim)
                    projected_emb_arr.append(projected_emb) 
                    #logits for all tokens in vocab
                    #lm_logits = self.item_embedding(projected_emb) #(bsz, seq_len, vocab_size)
                    lm_logits = F.linear(projected_emb, self.item_embedding.weight, None)
                    facet_lm_logits_arr.append(lm_logits)
                    if i < self.n_facet_reranker and not self.candidates_from_previous_reranker:
                        candidate_token_ids = []
                        for j in range(len(self.reranker_CAN_NUM)):
                            _, candidate_token_ids_ = torch.topk(lm_logits, self.reranker_CAN_NUM[j])
                            candidate_token_ids.append(candidate_token_ids_)
                        rereanker_candidate_token_ids_arr.append(candidate_token_ids)
                #print("before getting facets", self.project_arr[0].weight.data)
                #sys.exit(0)
                #print("before partitions", facet_lm_logits_arr[0].sum())

                for i in range(self.n_facet_reranker):
                    for j in range(len(self.reranker_CAN_NUM)):
                        projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, self.n_facet+i*len(self.reranker_CAN_NUM) + j) #(bsz, seq_len, hidden_dim)
                        projected_emb_arr.append(projected_emb)

                for i in range(self.n_facet_context):
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, self.n_facet+self.n_facet_reranker*len(self.reranker_CAN_NUM)+i) #(bsz, seq_len, hidden_dim)
                    projected_emb_arr.append(projected_emb)

                #to generate context-based embeddings for words in input
                for i in range(self.n_facet_emb):
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr_raw, self.n_facet + self.n_facet_context + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i) #(bsz, seq_len, hidden_dim)
                    projected_emb_arr.append(projected_emb)
                #print("projected_emb_arr", projected_emb_arr)

                for i in range(self.n_facet_reranker):
                    bsz, seq_len, hidden_size = projected_emb_arr[i].size()
                    for j in range(len(self.reranker_CAN_NUM)):
                        if self.candidates_from_previous_reranker:
                            _, candidate_token_ids = torch.topk(facet_lm_logits_arr[i], self.reranker_CAN_NUM[j]) #(bsz, seq_len, topk)
                        else:
                            candidate_token_ids = rereanker_candidate_token_ids_arr[i][j]
                        logit_hidden_reranker_topn = (projected_emb_arr[self.n_facet + i*len(self.reranker_CAN_NUM) + j].unsqueeze(dim=2).expand(bsz, seq_len, self.reranker_CAN_NUM[j], hidden_size) * self.item_embedding.weight[candidate_token_ids, :] ).sum(dim=-1) #(bsz, seq_len, emb_size) x (bsz, seq_len, topk, emb_size) -> (bsz, seq_len, topk)
                        if self.reranker_merging_mode == 'add':
                            #print("inside reranker")
                            facet_lm_logits_arr[i].scatter_add_(2, candidate_token_ids, logit_hidden_reranker_topn) #(bsz, seq_len, vocab_size) <- (bsz, seq_len, topk) x (bsz, seq_len, topk)
                        else:
                            facet_lm_logits_arr[i].scatter_(2, candidate_token_ids, logit_hidden_reranker_topn) #(bsz, seq_len, vocab_size) <- (bsz, seq_len, topk) x (bsz, seq_len, topk)
                        #for k in range(bsz):
                        #    if self.candidates_from_previous_reranker:
                        #        _, candidate_token_ids = torch.topk(facet_lm_logits_arr[i][k], self.reranker_CAN_NUM[j]) #(seq_len, topk)
                        #    else:
                        #        candidate_token_ids = rereanker_candidate_token_ids_arr[i][j][k]
                        #    #logit_hidden_reranker_topn = F.linear(projected_emb_arr[self.n_facet + i*len(self.reranker_CAN_NUM) + j][k],  self.item_embedding.weight[candidate_token_ids, :], None) #(seq_len, emb_size) x (seq_len, topk, emb_size) -> (seq_len, topk)
                        #    logit_hidden_reranker_topn = (projected_emb_arr[self.n_facet + i*len(self.reranker_CAN_NUM) + j][k].unsqueeze(dim=1).expand(seq_len, self.reranker_CAN_NUM[j], hidden_size) * self.item_embedding.weight[candidate_token_ids, :] ).sum(dim=-1) #(seq_len, emb_size) x (seq_len, topk, emb_size) -> (seq_len, topk)
                        #    if self.partition_merging_mode == 'add':
                        #        #print("inside reranker")
                        #        facet_lm_logits_arr[i][k].scatter_add_(1, candidate_token_ids, logit_hidden_reranker_topn) #(seq_len, vocab_size) <- (seq_len, topk) x (seq_len, topk)
                        #    else:
                        #        facet_lm_logits_arr[i][k].scatter_(1, candidate_token_ids, logit_hidden_reranker_topn) #(seq_len, vocab_size) <- (seq_len, topk) x (seq_len, topk)

                for i in range(self.n_facet_context):
                    bsz, seq_len_1, hidden_size = projected_emb_arr[i].size()
                    bsz, seq_len_2 = item_seq.size()
                    logit_hidden_context = (projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i].unsqueeze(dim=2).expand(-1,-1,seq_len_2,-1) * self.item_embedding.weight[item_seq, :].unsqueeze(dim=1).expand(-1,seq_len_1,-1,-1) ).sum(dim=-1)
                    logit_hidden_pointer = 0
                    if self.n_facet_emb == 2:
                        #logit_hidden_context = logit_hidden_context + ( projected_emb_arr[-2][:,-1,:].unsqueeze(dim=1).unsqueeze(dim=1).expand(-1,seq_len_1,seq_len_2,-1) * projected_emb_arr[-1].unsqueeze(dim=1).expand(-1,seq_len_1,-1,-1) ).sum(dim=-1)
                        logit_hidden_pointer = ( projected_emb_arr[-2][:,-1,:].unsqueeze(dim=1).unsqueeze(dim=1).expand(-1,seq_len_1,seq_len_2,-1) * projected_emb_arr[-1].unsqueeze(dim=1).expand(-1,seq_len_1,-1,-1) ).sum(dim=-1)
                    #logit_hidden_context_arr = []
                    #for j in range(bsz):
                    #    logit = F.linear(projected_emb_arr[self.n_facet + self.n_facet_reranker*len(self.reranker_CAN_NUM) + i][j], self.item_embedding.weight[item_seq[j, :], :], None)
                    #    if self.n_facet_emb == 2:
                    #        #print(projected_emb_arr[-2].size())
                    #        #print(projected_emb_arr[-1].size())
                    #        #print(logit.size())
                    #        logit += F.linear(projected_emb_arr[-2][j,-1,:], projected_emb_arr[-1][j], None)
                    #    logit_hidden_context_arr.append(logit)
                    #logit_hidden_context = torch.stack(logit_hidden_context_arr, dim =0) #bsz, seq_len_1, seq_len_2

                    item_seq_expand = item_seq.unsqueeze(dim=1).expand(-1,seq_len_1,-1)
                    only_new_logits = torch.zeros_like(facet_lm_logits_arr[i])
                    if self.context_norm:
                        only_new_logits.scatter_add_(dim=2, index=item_seq_expand, src=logit_hidden_context + logit_hidden_pointer)
                        item_count = torch.zeros_like(only_new_logits) + 1e-15
                        item_count.scatter_add_(dim=2, index=item_seq_expand,src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype))
                        only_new_logits = only_new_logits / item_count
                    else:
                        only_new_logits.scatter_add_(dim=2, index=item_seq_expand, src=logit_hidden_context)
                        item_count = torch.zeros_like(only_new_logits) + 1e-15
                        item_count.scatter_add_(dim=2, index=item_seq_expand,src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype))
                        only_new_logits = only_new_logits / item_count
                        only_new_logits.scatter_add_(dim=2, index=item_seq_expand, src=logit_hidden_pointer)
                        
                    if self.partition_merging_mode == 'replace':
                        facet_lm_logits_arr[i].scatter_(dim=2, index=item_seq_expand, src=torch.zeros_like(item_seq_expand).to(dtype=facet_lm_logits_arr[i].dtype) )
                    facet_lm_logits_arr[i] = facet_lm_logits_arr[i] + only_new_logits
                    #if self.partition_merging_mode == 'add':
                    #    #print("inside context")
                    #    item_seq_expand = item_seq.unsqueeze(dim=1).expand(-1,seq_len,-1)
                    #    only_new_logits = torch.zeros_like(facet_lm_logits_arr[i])
                    #    only_new_logits.scatter(dim=2, index=item_seq_expand, src=logit_hidden_context)
                    #    facet_lm_logits_arr[i] = facet_lm_logits_arr[i] + only_new_logits
                    #else:
                    #    #facet_lm_logits_arr[i].scatter(dim=2, index=item_seq.unsqueeze(dim=1).expand(-1,seq_len,-1), src=logit_hidden_context)
                #print("facet_lm_logits_arr", facet_lm_logits_arr)
                #sys.exit(0)
                #print("after partitions", facet_lm_logits_arr[0].sum())

            #logits for n_facet (==n_facet_effective)
            #for i in range(self.n_facet):       
            #    facet_lm_logits_real_arr.append( facet_lm_logits_arr[i] )

            #weight_mode = ''
            weight = None
            if self.weight_mode == 'dynamic':
                weight = self.weight_facet_decoder(stacked_hidden_emb_arr).softmax(dim=-1) #hidden_dim*hidden_input_state_ration -> n_facet_effective
            elif self.weight_mode == 'static':
                weight = self.weight_global.softmax(dim=-1) #torch.ones(n_facet_effective)
            elif self.weight_mode == 'max_logits':
                stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)
                facet_lm_logits_arr = [stacked_facet_lm_logits.amax(dim=0)]
            #print("after max", facet_lm_logits_arr[0].sum())
            #sys.exit(0)
                
            #print(weight)
            prediction_prob = 0

            for i in range(self.n_facet_effective):
                facet_lm_logits = facet_lm_logits_arr[i]
                if self.softmax_nonlinear == 'sigsoftmax': #'None' here
                    facet_lm_logits_sig = torch.exp(facet_lm_logits - facet_lm_logits.max(dim=-1,keepdim=True)[0]) * (1e-20 + torch.sigmoid(facet_lm_logits))
                    facet_lm_logits_softmax = facet_lm_logits_sig / facet_lm_logits_sig.sum(dim=-1,keepdim=True)
                elif self.softmax_nonlinear == 'None':
                    facet_lm_logits_softmax = facet_lm_logits.softmax(dim=-1) #softmax over final logits
                if self.weight_mode == 'dynamic':
                    prediction_prob += facet_lm_logits_softmax * weight[:,:,i].unsqueeze(-1)
                elif self.weight_mode == 'static':
                    prediction_prob += facet_lm_logits_softmax * weight[i]
                else:
                    prediction_prob += facet_lm_logits_softmax / self.n_facet_effective #softmax over final logits/1
            if item_seq is not None:
                # Shift so that tokens < n predict n            
                # Flatten the tokens

                # shift_logits = lm_logits[..., :-1, :].contiguous()
                # loss_fct = CrossEntropyLoss(ignore_index = -100)
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                inp = torch.log(prediction_prob.view(-1, self.n_items)+1e-8)
                #print("Input length is: {} and target length is: {}".format(len(inp), len(item_seq)))
                #for i in range(len(item_seq)):
                #    print(item_seq[i])
                #loss_raw = self.loss_fct(inp, item_seq.view(-1))
                loss_raw = self.loss_fct(inp, pos_items.view(-1))
                loss = loss_raw.mean()
            else:
                raise Exception("Labels can not be None")
            #logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            #loss = self.loss_fct(logits, pos_items)
            return loss, prediction_prob.squeeze(dim=1)
    
    def calculate_loss(self, interaction):
        loss, prediction_prob = self.calculate_loss_prob(interaction)
        return loss

    def predict(self, interaction):
        print("current mixture of softmax cannot support this function in an efficient way")
        sys.exit(0)
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = torch.FloatTensor([t.cpu().numpy() for t in seq_output]) #added for MFS
        test_item_emb = self.item_embedding(test_item)
        print("seq_output size is {} and test_item_emb size is {}".format(seq_output.shape, test_item_emb.shape))
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        loss, prediction_prob = self.calculate_loss_prob(interaction)
        item_seq = interaction[self.ITEM_SEQ]
        if self.post_remove_context:
            prediction_prob.scatter_(1, item_seq, 0)
        return prediction_prob
    #def full_sort_predict(self, interaction):
    #    item_seq = interaction[self.ITEM_SEQ]
    #    item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #    seq_output = self.forward(item_seq, item_seq_len)
    #    test_items_emb = self.item_embedding.weight
    #    scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
    #    return scores
