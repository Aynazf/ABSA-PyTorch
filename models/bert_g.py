import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention
from layers.point_wise_feed_forward import PositionwiseFeedForward

def self_attention(granules):
    attention_scores = torch.matmul(granules, granules.transpose(1,2))
    attention_weights = nn.functional.softmax(attention_scores, dim=-1)
    return attention_weights

class SelfAttention(nn.Module):
  def __init__(self, input_dim):
    super(SelfAttention, self).__init__()
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.value = nn.Linear(input_dim, input_dim)
    self.softmax = nn.Softmax(dim=2)
   
  def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
    queries = self.query(x)
    keys = self.key(x)
    values = self.value(x)

    score = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
    attention = self.softmax(score)
    weighted = torch.bmm(attention, values)
    return weighted


def weighted_sum(granules, attention_weights):
    weighted_granules = torch.matmul(attention_weights, granules)
    return weighted_granules.sum(dim=1)


class granular_BERT(nn.Module):
    def __init__(self, bert,opt, hidden_dim=128, num_classes=3):
        super(granular_BERT, self).__init__() # shak 2
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        bert_output_dim = self.bert.config.hidden_size  # This is 768 for bert-base-uncased

        self.lstm_sentence = nn.LSTM(bert_output_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm_target = nn.LSTM(bert_output_dim, hidden_dim, bidirectional=True, batch_first=True )
        self.second_lstm_sentense=nn.LSTM(2*hidden_dim,hidden_dim, bidirectional=True, batch_first=True)
        self.second_lstm_target=nn.LSTM(2*hidden_dim,hidden_dim, bidirectional=True, batch_first=True)
        self.W = nn.Parameter(torch.empty(2* hidden_dim, 2* hidden_dim))
        nn.init.xavier_uniform_(self.W)

        self.self_att_context=SelfAttention(2* hidden_dim)
        self.self_att_target=SelfAttention(2* hidden_dim)
        # self.W_a = nn.Parameter(torch.empty( 2* hidden_dim,  2* hidden_dim))
        # nn.init.xavier_uniform_(self.W_a)
        # self.b_a = nn.Parameter(torch.empty(2* hidden_dim))
        # nn.init.zeros_(self.b_a)
        # self.W_b = nn.Parameter(torch.empty(2* hidden_dim, 2* hidden_dim))
        # nn.init.xavier_uniform_(self.W_b)
        # self.b_b = nn.Parameter(torch.empty(2* hidden_dim))
        # nn.init.zeros_(self.b_b)
        
        self.attention_aspect = Attention(2* hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(2* hidden_dim, score_function='bi_linear')        
        self.dense = nn.Linear(hidden_dim * 2, opt.polarities_dim)  # 2 * hidden_dim (sentence) + 2 * hidden_dim (target)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        context = self.squeeze_embedding(context, context_len)
        context = self.bert(context)
        context = self.dropout(context['last_hidden_state'])
        target = self.squeeze_embedding(target, target_len)
        target= self.bert(target)
        target = self.dropout(target['last_hidden_state'])

        sentence_lstm_output, (sentence_hidden, _) = self.lstm_sentence(context)
        target_lstm_output, (target_hidden, _) = self.lstm_target(target)

        G1CT_s=torch.matmul(sentence_lstm_output,self.W)
        G1CT=torch.matmul(G1CT_s, target_lstm_output.transpose(1,2))

        G1C_m=torch.matmul(G1CT,target_lstm_output)
        G1T_m=torch.matmul(G1CT.transpose(1,2),sentence_lstm_output)


        sentence_lstm_output, (sentence_hidden, _) = self.second_lstm_sentense(G1C_m)
        target_lstm_output, (target_hidden, _) = self.second_lstm_target(G1T_m)

        G2CT=Euclidean_distance(sentence_lstm_output, target_lstm_output)
        G2C_m=torch.matmul(G2CT,target_lstm_output)
        G2T_m=torch.matmul(G2CT.transpose(1,2),sentence_lstm_output)

        sentence_lstm_output, (sentence_hidden, _) = self.second_lstm_sentense(G2C_m)
        target_lstm_output, (target_hidden, _) = self.second_lstm_target(G2T_m)
        
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(target_lstm_output, dim=1)
        aspect_pool = torch.div(aspect_pool, target_len.view(target_len.size(0), 1))

        text_raw_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(sentence_lstm_output, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(target_lstm_output, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final,_ = self.attention_context(sentence_lstm_output, aspect_pool)
        context_final = context_final.squeeze(dim=1)
        
        # at=self.self_att_target(target_lstm_output)
        # ac=self.self_att_context(sentence_lstm_output)

        # weighted_t=weighted_sum(target_lstm_output, at)
        # weighted_c=weighted_sum(sentence_lstm_output, ac)
        # shape_t=weighted_t.shape
        # weighted_t=weighted_t.view(shape_t[0],1,shape_t[1])

        # shape_c=weighted_c.shape
        # weighted_c=weighted_c.view(shape_c[0],1,shape_c[1])

        # multi_c_w=torch.matmul(sentence_lstm_output,self.W_a)

        # g_c=torch.tanh(torch.matmul(multi_c_w,at.transpose(1,2))+self.b_a)


        # e_c=torch.exp(g_c)
        # sum_e_c=torch.sum(e_c)
        # alpha_c=e_c/sum_e_c

        # G_final_C=weighted_sum(sentence_lstm_output,alpha_c)


        # multi_t_w=torch.matmul(target_lstm_output,self.W_b)
        # g_t=torch.tanh(torch.matmul(multi_t_w,ac.transpose(1,2))+self.b_b)

        # e_t=torch.exp(g_t)
        # sum_e_t=torch.sum(e_t)
        # alpha_t=e_t/sum_e_t

        # G_final_t=weighted_sum(target_lstm_output,alpha_t)

        # sentence_hidden = torch.cat((sentence_hidden[-2,:,:], sentence_hidden[-1,:,:]), dim=1) #shak 3
        # target_hidden = torch.cat((target_hidden[-2,:,:], target_hidden[-1,:,:]), dim=1) #shak 4
        # print("fff",sentence_hidden.shape)

        # hc_mean = torch.div(torch.sum(hc, dim=1), ac.unsqueeze(1).float())
        # ht_mean = torch.div(torch.sum(ht, dim=1), at.unsqueeze(1).float())
        # s1_mean = torch.div(torch.sum(s1, dim=1), context_len.unsqueeze(1).float())
        combined = torch.cat((aspect_final, context_final), dim=-1)
        # combined = self.dropout(combined)
        output = self.dense(combined)
        out=self.softmax(output)
        print(out.shape)
        return out




def Euclidean_distance(context_hidden, target_hidden):
    batch_size, context_len, _ = context_hidden.size()
    target_len = target_hidden.size(1)

    # Compute Euclidean distance weights
    weights = torch.zeros(batch_size, context_len, target_len).to(context_hidden.device)
    for i in range(context_len):
        for j in range(target_len):
            distance = torch.norm(context_hidden[:, i, :] - target_hidden[:, j, :], dim=1)
            weights[:, i, j] = 1 / (1 + distance)

    return weights #context_representation, target_representation
