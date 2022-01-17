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
        self.n_facet_effective = config['n_facet'] #added for mfs
        self.n_facet = config['n_facet'] #added for mfs
        self.n_facet_window = config['n_facet_window'] #added for mfs
        self.n_facet_hidden = config['n_facet_hidden'] #added for mfs
        self.n_facet_MLP = config['n_facet_MLP'] #added for mfs
        assert self.n_facet_MLP <= 0 #-1 or 0
        assert self.n_facet_window <= 0
        self.n_facet_window = - self.n_facet_window
        self.n_facet_MLP = - self.n_facet_MLP
        self.softmax_nonlinear='None' #added for mfs
        self.efficient_mode = config['efficient_mode'] #added for mfs
        self.only_compute_loss = True #added for mfs
        self.n_embd = self.hidden_size #added for mfs
        self.use_proj_bias = config['use_proj_bias'] #added for mfs
        self.weight_mode = config['weight_mode'] #added for mfs
        # for multiple input hidden states
        if self.n_facet_MLP > 0:
            hidden_state_input_ratio = 1 + self.n_facet_MLP #1 + 1
            self.MLP_linear = nn.Linear(self.n_embd * (self.n_facet_hidden * (self.n_facet_window+1) ), self.n_embd * self.n_facet_MLP) # (hid_dim*2) -> (hid_dim)
        else:            
            hidden_state_input_ratio = self.n_facet_hidden * (self.n_facet_window+1) #1 * (0+1)
        total_lin_dim = self.n_embd * hidden_state_input_ratio
        small_value = 0.0001
        self.project_arr = nn.ModuleList([nn.Linear(total_lin_dim, self.n_embd, bias=self.use_proj_bias) for i in range(self.n_facet_all)])
        for i in range(self.n_facet_all):
            if self.use_proj_bias:
                self.project_arr[i].bias.data.zero_()
            linear_weights = torch.zeros_like(self.project_arr[i].weight.data)

            # if i!= n_facet - 1:
            #     linear_weights = linear_weights + small_value * (torch.rand((config.n_embd, total_lin_dim)) - 0.5 )
            linear_weights[:,:self.n_embd] = torch.eye(self.n_embd)
            #if i < n_facet:
            #     linear_weights[:,:config.n_embd] = torch.eye(config.n_embd)
            # else:
            #     linear_weights[:,:config.n_embd] = 1e-10 * torch.eye(config.n_embd)
            self.project_arr[i].weight.data = linear_weights

        self.project_emb = nn.Linear(self.n_embd, self.n_embd, bias=self.use_proj_bias)
        if len(self.weight_mode) > 0:
            self.weight_facet_decoder = nn.Linear(self.hidden_size * hidden_state_input_ratio, self.n_facet_effective)
            #self.weight_facet_decoder = nn.Linear(config.hidden_size * n_facet_hidden * (n_facet_window+1), n_facet)
            self.weight_global = nn.Parameter( torch.ones(self.n_facet_effective) )
        self.output_probs = True
        self.c = 100
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
            self.loss_fct = nn.NLLLoss(reduction='none') #modified for mfs
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
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

    def calculate_loss(self, interaction):
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
                print('all_hidden_states length is {}. i is {}'.format(len(all_hidden_states), i))
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
                stacked_hidden_emb_arr = torch.cat([hidden_emb_arr[0], gelu(hidden_emb_MLP)], dim=-1) #bsz, seq_len, 2*hidden_size
            else:
                stacked_hidden_emb_arr = hidden_emb_arr[0]

            #list of linear projects per facet
            projected_emb_arr = []
            #list of final logits per facet
            facet_lm_logits_arr = []
            facet_lm_logits_real_arr = []

            #logits for orig facets
            if self.efficient_mode == 'even_last_2':
                bsz, seq_len, hidden_size = all_hidden_states[-1].size()
                logit_all = torch.empty( (bsz, seq_len, self.n_items) , device=all_hidden_states[-1].device )
                n_facet_not_last = self.n_facet_all - (self.n_facet_effective-1) # 6 - (3-1) = 4 -> partitions
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
                for i in range(self.n_facet_effective-1):
                    projected_emb = self.project_arr[-(i+1)](stacked_hidden_emb_arr)
                    #projected_emb = self.project_arr[n_facet_not_last+i](stacked_hidden_emb_arr)
                    #projected_emb = self.get_facet_emb(stacked_hidden_emb_arr,n_facet_not_last+i)
                    projected_emb_arr.append(projected_emb)
                    #projected_emb_real_arr.append(projected_emb)

                    #facet_lm_logits_arr.append( self.item_embedding( projected_emb ) )
                    facet_lm_logits_arr.append(F.linear(projected_emb, self.item_embedding.weight, None))
            else:
                for i in range(self.n_facet):
                #     #linear projection
                    projected_emb = self.get_facet_emb(stacked_hidden_emb_arr, i) #(bsz, seq_len, hidden_dim)
                    projected_emb_arr.append(projected_emb) 
                    #logits for all tokens in vocab
                    #lm_logits = self.item_embedding(projected_emb) #(bsz, seq_len, vocab_size)
                    lm_logits = F.linear(projected_emb, self.item_embedding.weight, None)
                    facet_lm_logits_arr.append(lm_logits)
            

            #logits for n_facet (==n_facet_effective)
            for i in range(self.n_facet):       
                facet_lm_logits_real_arr.append( facet_lm_logits_arr[i] )
            stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)

            #weight_mode = ''
            weight = None
            if self.weight_mode == 'dynamic':
                weight = self.weight_facet_decoder(stacked_hidden_emb_arr).softmax(dim=-1) #hidden_dim*hidden_input_state_ration -> n_facet_effective
            elif self.weight_mode == 'static':
                weight = self.weight_global.softmax(dim=-1) #torch.ones(n_facet_effective)
            #print(weight)
            prediction_prob = 0

            for i in range(self.n_facet_effective):
                facet_lm_logits = facet_lm_logits_real_arr[i]
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
                print("Input length is: {} and target length is: {}".format(len(inp), len(item_seq)))
                loss_raw = self.loss_fct(inp, item_seq)
                loss = loss_raw[item_seq != -100].mean()
            else:
                raise Exception("Labels can not be None")
            #logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            #loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
