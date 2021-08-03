import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from transformers.modeling_bert import BertLayerNorm

class GraphAttentionLayer(nn.Module):
    # def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat, method='self'):
    def __init__(self, args):

        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        # self.dropout = dropout
        # self.output_dim = out_dim
        '''
        self.attentions = [SpGraphAttentionLayer(input_dim,
                                                 output_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 activation=activation,
                                                 method=method) for _ in range(nheads)]
        '''
        self.method = args.method
        if self.method == 'self':
            self.attention_layer = SelfAttention(args)
        else:
            self.attention_layer = CrossAttention(args)
        self.attention_output = GATSelfOutput(args)

    def forward(self, inputs):
        if self.method == 'self':
            x, adj = inputs
        elif self.method == 'cross':
            x = inputs
            adj = None
        h = self.attention_layer(x, adj)
        h = F.relu(h)#v2
        #h = self.attention_output(h) #v1
        # h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)


class SelfAttention(nn.Module):
    '''
    Implementation of scale dot self attention
    '''

    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(SelfAttention, self).__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim, config.n_heads))
        '''
        Note that for adjacancy_approximator and GNN, the attn_drop should be false  
        '''
        self.attn_drop = True
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.num_attention_heads = config.n_heads
        self.attention_head_size = int(config.dim / config.n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.directional = config.directional

        self.query = nn.Linear(config.dim, self.all_head_size)
        self.key = nn.Linear(config.dim, self.all_head_size)
        self.value = nn.Linear(config.dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.do_softmax = True
        self.unmix = False

    def transpose_for_scores(self, x):
        '''
        This kind of operation adds ono-linearity in attention operation with a lower cost.
        In other words, 1024d vector is cut into 16 pieces and made different operations.
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, attention_probs=None):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if not self.unmix:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)
        else:
            query_layer = mixed_query_layer.unsqueeze(1)
            key_layer = mixed_key_layer.unsqueeze(1)
            value_layer = mixed_value_layer.unsqueeze(1)
            self.attention_head_size = self.all_head_size
            # pdb.set_trace()

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.do_softmax:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        else:
            pass

        if attention_mask is not None and self.directional:
            attention_mask[attention_mask < 0.5] = -1e7
            attention_mask[attention_mask > 0.5] = 0
            attention_mask = attention_mask.repeat(16, 1, 1, 1).permute(1, 2, 3, 0)  # 16 for num_head

            attention_scores = (attention_scores.permute(0, 2, 3, 1).contiguous() + attention_mask.type_as(attention_scores)).permute(0, 3, 1,
                                                                                                            2).contiguous()

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Normalize the attention scores to probabilities.
        # Note that the row of each sample each timestrp is scaled to 1.

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.attn_drop:
            attention_probs = self.dropout(attention_probs)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        '''
        if self.output_attentions:
            if self.do_softmax:
                return attention_probs, context_layer
            else:
                return attention_scores, context_layer
        '''
        return context_layer


class CrossAttention(nn.Module):
    '''
    Implementation of scale dot cross attention
    '''

    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(CrossAttention, self).__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.dim, config.n_heads))
        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output
        self.multihead_output = None

        self.num_attention_heads = config.n_heads
        self.attention_head_size = int(config.dim / config.n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.dim, self.all_head_size)
        self.key = nn.Linear(config.dim, self.all_head_size)
        self.value = nn.Linear(config.dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.config = config

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, features, attention_mask=None, head_mask=None, sent_ind=None):
        graph_vectors = features[:, 0, :].unsqueeze(1)
        word_vectors = features[:, 1:, :]
        # word_vectors = features

        mixed_query_layer = self.query(graph_vectors)
        mixed_key_layer = self.key(word_vectors)
        mixed_value_layer = self.value(word_vectors)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if sent_ind is not None and self.config.sep_sent:
            num_sent = attention_scores.shape[2]
            num_batch = attention_scores.shape[0]
            sent_ind = sent_ind[:, :-1]

            for n in range(num_batch):
                for j in range(num_sent):
                    attention_scores[n, :, j, sent_ind[n] != j] = 0

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        if attention_mask is not None:
            # attention_mask = attention_mask.log()
            attention_mask[attention_mask < 0.5] = -1e7
            attention_mask[attention_mask > 0.5] = 0
            attention_mask = attention_mask.repeat(16, 1, 1, 1).permute(1, 2, 3, 0)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.keep_multihead_output:
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        '''
        if self.output_attentions:
            return attention_probs, context_layer
        '''
        return context_layer

class GATSelfOutput(nn.Module):
    def __init__(self, config):
        super(GATSelfOutput, self).__init__()
        self.dense = nn.Linear(config.dim, config.dim)
        self.layer_norm = config.layer_norm
        #self.act_fn = F.softplus
        self.act_fn = F.relu
        if self.layer_norm:
            self.LayerNorm = BertLayerNorm(config.dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        if self.layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states