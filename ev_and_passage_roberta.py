import spacy
import os
import json
import sys
#get current root directory
root_pth = '/home/xxx'

import re
import networkx as nx
import random
import numpy as np
import copy
import pickle
import torch
from nltk.tokenize import sent_tokenize
from torch.utils.data import TensorDataset
from functools import reduce
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
abbreviation = ['maj', 'gen', 'brig', 'u.s', 'col', 'gens']

#   EventWiki数据集路径
input_dir = "{}/HGAM/data/EventWiki".format(root_pth)
event_file = '{}/HGAM/data/EventWiki/event_list.txt'.format(root_pth)  #   存储EventWiki中所有事件名称的文件
#   存储图数据文件夹路径
output_dir = "{}/subeventre/data/graph".format(root_pth)
#hyperparameter
max_L = 20#记录图中最大节点个数
max_seq_length=80#记录三元组/上下文最大分词后长度
max_triples=20#记录节点编码中引入的最多三元组和上下文个数
max_positions=19#记录节点编码中在三元组/上下文中出现位置的最多次数（二元坐标）
max_sentences=5#上下文包含的句子最大数量

class Example(object):
    def __init__(self, id, label, e1, e2, ev_lst, e_triples, events_dict, context_pos):
        '''
        :param label: -1/1/0
        :param e1:
        :param e2:
        :param e_triples: for BERT encoding
        :param ev_lst: pri_lst + common event list
        '''
        self.id = id
        self.label = label
        self.e1 = e1
        self.e2 = e2
        self.ev_lst = ev_lst
        self.e_triples = e_triples
        self.events_dict = events_dict
        self.context_pos = context_pos

class Features(object):
    def __init__(self, ev_pairs, triple_ids, triple_masks, segment_ids, events_pos, graph, label_id, context_pos):
        '''
        :param ev_pais: event pair indexes
        :param triple_ids: 2d list
        :param triple_masks: 2d list
        :param segment_ids: 2d list
        :param events_pos: 2d list
        :param label_id:
        '''
        self.ev_pairs = ev_pairs
        self.triple_ids = triple_ids
        self.triple_masks = triple_masks
        self.segment_ids = segment_ids
        self.events_pos = events_pos
        self.graph = graph
        self.label_id = label_id
        self.context_pos = context_pos

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _truncate_seq_pair(se_tokens, rel_tokens, te_tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(se_tokens) + len(rel_tokens) + len(te_tokens)
        if total_length <= max_length:
            break
        #print('truncate_triples:{},{},{}'.format(se_tokens,rel_tokens,te_tokens))
        tmp = [len(se_tokens), len(rel_tokens), len(te_tokens)]
        if tmp.index(max(tmp)) == 0:
            se_tokens.pop()
        elif tmp.index(max(tmp)) == 1:
            rel_tokens.pop()
        else:
            te_tokens.pop()

def _truncate_triples(e_triples, ev_lst, max_length):
    '''truncate a sample's event_triples'''
    if len(e_triples) <= max_length:
        return e_triples
    length = len(e_triples)
    e_tri_tmp = reduce(lambda x,y:x+y, e_triples)
    res = Counter(e_tri_tmp)
    times = 0
    while True:
        if len(e_triples) <= max_length:
            break
        if times > length:
            print('max_length setting now is not enough!')
            break
        triple = e_triples.pop()
        times += 1
        intersection = set(triple) & set(ev_lst)
        flag = False
        for _ in intersection:
            if res[_] == 1:
                flag = True
                break
        if flag == True:
            e_triples.insert(0, triple)


def process_triples(se_tokens, rel_tokens, te_tokens, tokenizer):

    _truncate_seq_pair(se_tokens, rel_tokens, te_tokens, max_seq_length - 3)

    triple_tokens = ["<s>"] + se_tokens + ["</s>"] + rel_tokens + ["</s>"] + te_tokens
    segment_ids = [0] * (len(se_tokens) + 1) + [1] * (len(rel_tokens) + 1) + [2] * (len(te_tokens) + 1)
    triple_ids = tokenizer.convert_tokens_to_ids(triple_tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    triple_mask = [1] * len(triple_ids)

    # Zero-pad up to the sequence length
    padding_lengths = max_seq_length - len(triple_ids)
    triple_ids = triple_ids + ([0] * padding_lengths)  # pad_token:0
    triple_mask = triple_mask + ([0] * padding_lengths)  # mask_padding_with_zero:True
    segment_ids = segment_ids + ([0] * padding_lengths)  # pad_token_segment_id:0

    assert len(triple_ids) == max_seq_length
    assert len(triple_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return triple_ids, triple_mask, segment_ids

def process_single_node(tokens, tokenizer):

    if len(tokens) > max_seq_length - 3:
        tokens = tokens[:max_seq_length - 3]
    tokens = ["<s>"] + tokens + ["</s>"] + ["</s>"]
    segment_ids = [0] * len(tokens)#暂时没用
    node_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    node_mask = [1] * len(node_ids)

    # Zero-pad up to the sequence length
    padding_lengths = max_seq_length - len(node_ids)
    node_ids = node_ids + ([0] * padding_lengths)  # pad_token:0
    node_mask = node_mask + ([0] * padding_lengths)  # mask_padding_with_zero:True
    segment_ids = segment_ids + ([0] * padding_lengths)  # pad_token_segment_id:0

    assert len(node_ids) == max_seq_length
    assert len(node_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return node_ids, node_mask, segment_ids

def _truncate_seq_pair_context(tokens_list, is_event_flag_list, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = 0
        for tokens in tokens_list:
            total_length += len(tokens)
        if total_length <= max_length:
            break
        tmp_con = {}
        for index, tokens in enumerate(tokens_list):
            if is_event_flag_list[index] == 0:
                tmp_con[len(tokens)] = index
            # else:
            #     tmp_ev[len(tokens)] = index
        max_pcon = tmp_con[max(tmp_con.keys())]
        try:
            start_ev = is_event_flag_list.index(1)
            if max_pcon < start_ev:
                tokens_list[max_pcon].pop(0)
            else:
                try:
                    tokens_list[max_pcon].pop()
                except:
                    print('!')
        except:
            tokens_list[max_pcon].pop()


def process_context(tokens_list, is_event_flag_list, tokenizer):
    # tokens: [CLS] xxx [UNK] e1 [UNK] xxx [UNK] e2 [UNK] xxxxxxx [SEP]
    #   sep: "</s>" 只有一个
    #   cls: "<s>"
    #   unk: 区分事件'<unk>'
    context_tokens = ["<s>"]
    tokens_length = 0
    event_num = 0
    segment_length_list = []

    for index in range(len(tokens_list)):
        if is_event_flag_list[index]:
            event_num += 1
    new_length = max_seq_length - 2 - 2 * event_num
    _truncate_seq_pair_context(tokens_list, is_event_flag_list, new_length)
    for index in range(len(tokens_list)):
        tokens_length += len(tokens_list[index])
        if is_event_flag_list[index]:
            tokens_tmp = ["</s>"] + tokens_list[index] + ["</s>"]
            context_tokens = context_tokens + tokens_tmp
        else:
            context_tokens = context_tokens + tokens_list[index]
        if index == (len(tokens_list) - 1):
            context_tokens = context_tokens + ["</s>"]
    index_ss = []#slash s, i.e., </s>
    for index_token, token in enumerate(context_tokens):
        if token == "</s>":
            index_ss.append(index_token)
    index_ss.pop()#最少包含一个元素
    if 1 in is_event_flag_list:
        segment_length_list.append(index_ss[0])#记录初始index
        for i in range(2 * event_num - 1):#0-2*event_num-2
            segment_length_list.append(index_ss[i+1] - index_ss[i])#记录每段长度
        segment_length_list.append((len(context_tokens)-index_ss[-1]))#记录剩余长度
        segment_ids = []
        for id in range(len(segment_length_list)):
            segment_ids += [id] * segment_length_list[id]#记录</s>个数
    else:
        segment_ids = [0] * len(context_tokens)
    context_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    context_mask = [1] * len(context_ids)
    padding_lengths = max_seq_length - len(context_ids)
    context_ids = context_ids + ([0] * padding_lengths)  # pad_token:0
    context_mask = context_mask + ([0] * padding_lengths)  # mask_padding_with_zero:True
    segment_ids = segment_ids + ([0] * padding_lengths)  # pad_token_segment_id:0

    assert len(context_ids) == max_seq_length
    assert len(context_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return context_ids, context_mask, segment_ids

#   生成triple_ids, triple_mask, segment_ids三类feature
def get_triple_features(ev_lst, e_triples_all, e1, e2, events_dict, tokenizer, context_pos):
    ev_index = {ev: index for index, ev in enumerate(ev_lst)}
    #TODO:格式index:(x,y),index表示事件唯一序号，(x,y)表示该事件出现在第x个triples中的第y个位置，最多有max_triples
    #TODO:先放三元组，再放seperated_nodes，最后放上下文(和seperated_nodes相比更容易引入噪声)
    events_pos = {}
    # locate event's position with 2d coordinates
    ev_times = {ev: 0 for ev in ev_lst}
    #   分割三元组和上下文，分别以不同的方式进行拼接处理
    e_triples = e_triples_all[0:context_pos]
    context_list = e_triples_all[context_pos:]
    context_list_length = len(context_list)
    for ev in ev_lst:
        for tri in e_triples_all:
            if ev in tri:
                ev_times[ev] += 1
    seperated_nodes = [ev for ev, time in ev_times.items() if time == 0]
    max_triples_new = max_triples - context_list_length - len(seperated_nodes)
    # truncate redundant triples
    if len(e_triples) > 0:
        _truncate_triples(e_triples, ev_lst, max_triples_new)
    context_pos_new = len(e_triples)+len(seperated_nodes)
    '''nodes may have been truncated while still in e_triples, so traverse ev_lst not e_triples to construct events_pos'''
    # total_set需要array_like
    total_set = e_triples + [[node] for node in seperated_nodes] + context_list
    if len(total_set) > 0:
        max_len = max([len(e) for e in total_set])
        tmp_all = []
        for e in total_set:
            e_cp = copy.deepcopy(e)
            if len(e_cp) != max_len:
                e_cp.extend(['0'] * (max_len - len(e)))
            tmp_all.append(e_cp)
        for index, ev in enumerate(ev_lst):
            if events_pos.get(ev) == None:
                events_pos[ev_index[ev]] = []
            for pos in np.argwhere(np.array(tmp_all) == ev):
                events_pos[ev_index[ev]].append(tuple(pos))
    triple_ids_lst = []
    triple_mask_lst = []
    segment_ids_lst = []
    #   三元组编码
    #   对于段落中事件，不使用原来的三元组编码形式，使用事件的上下文编码（roberta+bilstm），对于同一事件出现在多句话中使用表示平均作为事件表示
    #   对于外部KG中事件及事件关系，仍采用三元组拼接方式进行编码，将bert替换为roberta即可
    for index, e_triple in enumerate(e_triples):
        source_e, rel, target_e = e_triple[0], e_triple[1], e_triple[2]
        rel_, _ = rel.split('__')
        rel = ' '.join(rel_.split('/'))
        se_tokens = tokenizer.tokenize(source_e)
        rel_tokens = tokenizer.tokenize(rel)
        te_tokens = tokenizer.tokenize(target_e)

        triple_ids, triple_mask, segment_ids = process_triples(se_tokens, rel_tokens, te_tokens, tokenizer)
        triple_ids_lst.append(triple_ids)
        triple_mask_lst.append(triple_mask)
        segment_ids_lst.append(segment_ids)

    # TODO:放中间
    for index, node in enumerate(seperated_nodes):
        tokens = tokenizer.tokenize(node)
        node_ids, node_mask, segment_ids = process_single_node(tokens, tokenizer)
        triple_ids_lst.append(node_ids)
        triple_mask_lst.append(node_mask)
        segment_ids_lst.append(segment_ids)

    #TODO: 由于e_triples_all中先triple后context，所以该for循环一定要在最后
    for index, context in enumerate(context_list):
        is_event_flag_list = [] #   记录每个分隔是上下文（0）还是事件（1）
        tokens_list = []
        for piece in context:
            if piece in ev_lst:
                is_event_flag_list.append(1)
            else:
                is_event_flag_list.append(0)
            tokens = tokenizer.tokenize(piece)
            tokens_list.append(tokens)
        # tokens: [CLS] xxx [SEP] e1 [SEP] xxx [SEP] e2 [SEP] xxxxxxx [SEP]
        #   sep: "</s>" 奇数个
        #   cls: "<s>"
        #   </s>: 区分事件，忽略一个在句尾
        if [] in tokens_list:
            print('11')
        if 1 not in is_event_flag_list:
            print('22')
        context_ids, context_mask, segment_ids = process_context(tokens_list, is_event_flag_list, tokenizer)
        triple_ids_lst.append(context_ids)
        triple_mask_lst.append(context_mask)
        segment_ids_lst.append(segment_ids)

    # events list pad up to the max_L
    padding_events = max_L - len(events_pos)
    # coordinates-pad up to the max_positions with (-1,-1)
    events_pos_lst = []
    for i in range(len(events_pos)):
        events_pos_tmp = []
        try:
            padding_pos = max_positions - len(events_pos[i])#二维坐标，最多出现max_positions次

        except Exception:
            print(e1, e2)
            padding_events += 1
            continue
        events_pos_tmp.extend(events_pos[i])
        events_pos_tmp.extend(([(-1, -1)]) * padding_pos)
        events_pos_lst.append(events_pos_tmp)
    for i in range(padding_events):
        events_pos_lst.append(([(-1, -1)]) * max_positions)

    # Zero-pad up to the max triples' length
    padding_lengths = max_triples - len(triple_ids_lst)
    triple_ids_lst = triple_ids_lst + ([([0] * max_seq_length)]) * padding_lengths  # pad_token:0
    triple_mask_lst = triple_mask_lst + ([([0] * max_seq_length)]) * padding_lengths  # mask_padding_with_zero:True
    segment_ids_lst = segment_ids_lst + ([([0] * max_seq_length)]) * padding_lengths  # pad_token_segment_id:0
    try:
        assert len(triple_ids_lst) == max_triples
    except:
        print(padding_lengths)
    assert len(triple_mask_lst) == max_triples
    assert len(segment_ids_lst) == max_triples
    try:
        assert len(events_pos_lst) == max_L
    except:
        print('max_L')

    adjacency_matrix = adjacency_matrix_matching(ev_lst, events_dict)

    return ev_index, triple_ids_lst, triple_mask_lst, segment_ids_lst, events_pos_lst, adjacency_matrix, context_pos_new



def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span

def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index



def convert_examples_to_features(examples, tokenizer):
    '''

    :param examples:
    :param tokenizer:
    :param max_L: The maximum number of nodes in the graph, nodes have been truncated in the generation of examples, here just need padding
    :param max_seq_length: the min dijstra passage+ kg is 19, need padding or truncating
    :param max_triples: the max occurrence time is 67, need padding or truncating
    :param max_positions: get the num 19 from statistics from the train/eval/test dataset, just need padding without truncating
    :return:
    '''
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {'-1':0, '0':1, '1':2}
    features = []
    for example_index, example in enumerate(examples):
        '''
        ev_lst:[EA, EB, EC, ED, EE, RF, RG, RH, RI, RJ]
        e_triples:[[EA,RF,EB],[EE,RJ,EA],[EA,RG,EE],[EE,RH,EC],[EC,RI,EE]]
        nodes do not appear in the triples: ED
        '''
        ev_lst = example.ev_lst#图上节点集合
        e_triples = example.e_triples#triples+context
        events_dict = example.events_dict#刻画有向边+无向边
        context_pos = example.context_pos
        e1 = example.e1
        e2 = example.e2
        ev_index, triple_ids_lst, triple_mask_lst, segment_ids_lst, events_pos_lst, adjacency_matrix, context_pos_new = get_triple_features(
            ev_lst,
            e_triples,
            e1, e2,
            events_dict,
            tokenizer,
            context_pos)


        features.append(
            Features(ev_pairs=[ev_index[example.e1], ev_index[example.e2]],
                     triple_ids=triple_ids_lst,
                     triple_masks=triple_mask_lst,
                     segment_ids=segment_ids_lst,
                     events_pos=events_pos_lst,
                     graph=adjacency_matrix,
                     label_id=label_map[example.label],
                     context_pos=context_pos_new
                     )
            )

    return features


def load_and_cache_examples(args, status, tokenizer):
    '''

    :param status:train/eval/test
    :param tokenizer:
    :return:
    '''
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    examples_files = '{}/subeventre/data/graph/{}_examples_roberta{}_maxL{}_maxtriples{}_maxseqlen{}_maxpos{}_maxsens{}.pkl'.\
            format(root_pth, status, others, max_L, max_triples, max_seq_length, max_positions, max_sentences)
    features_files = '{}/subeventre/data/graph/{}_features_roberta{}_maxL{}_maxtriples{}_maxseqlen{}_maxpos{}_maxsens{}'.\
            format(root_pth, status, others, max_L, max_triples, max_seq_length, max_positions, max_sentences)

    if os.path.exists(features_files):
        print("Loading features form file {}".format(features_files))
        features = torch.load(features_files)
    else:
        print("Loading {} examples from file {}".format(status, examples_files))
        with open(examples_files, 'rb') as f:
            examples = pickle.load(f)
            features = convert_examples_to_features(examples, tokenizer)
            if args.local_rank in [-1, 0]:
                print("Saving {} features into file {}".format(status, features_files))
                torch.save(features, features_files)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_ev_pairs = torch.tensor([f.ev_pairs for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.triple_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.triple_masks for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_evs_pos = torch.tensor([f.events_pos for f in features], dtype=torch.long)
    all_graphs = torch.tensor([f.graph for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_context_pos = torch.tensor([f.context_pos for f in features], dtype=torch.long)

    dataset = TensorDataset(all_ev_pairs, all_input_ids, all_input_mask, all_segment_ids, all_evs_pos, all_graphs,
                            all_label_ids, all_context_pos)
    return dataset



class EWProcessor(object):
    '''
    processor for EventWiki data set
    '''
    def get_examples(self, input_dir, Graph, Graph_T, test_lines):
        return self._create_examples(
            self.read_infobox_dat(os.path.join(input_dir, 'infobox_event_relation_new.dat')), self.read_text_dat(os.path.join(input_dir, 'text_event_relation.dat')), Graph, Graph_T, test_lines)

    def read_infobox_dat(self, input_file):
        '''
        样本中包含重复样本
        :param input_file:
        :return:
        '''
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    lines.append(line.lower())
        li = list(set(lines))
        li.sort()
        return li
    def _create_examples(self, infobox_lines, text_lines, Graph, Graph_T, test_lines):
        #examples = []   #   存储事件对信息（e1，e2，relation）
        #   两事件同时存在多种关系id列表
        #multi_rel = []
        #   将infobox文档读取为三元组并构图
        filter_num = 0#记录子事件关系被过滤个数
        index_rel = 0
        for infobox_line in infobox_lines:
            infobox_line = re.sub('\t', '', infobox_line)
            e1, e2, rel = infobox_line.strip().split('|||') #   按照格式读取e1，e2，relation信息
            #   去掉rel末尾的无意义数字
            rel = re.sub(r'[\|\d]+', '', rel).strip()#e.g.filter the case like "2 || style", except one case "24 June 1995 || rowspan"
            # #   判断是否存在知识泄露
            # if e1 in test_event_list:
            #     continue
            # if e2 in test_event_list:
            #     continue
            #   构建图
            if e1 not in Graph:
                Graph.add_node(e1)
                Graph_T.add_node(e1)
            if e2 not in Graph:
                Graph.add_node(e2)
                Graph_T.add_node(e2)
            #   如果存在多种关系，则拼接并更新边的属性
            #TODO:表示子事件的关系有Subevent,partof,part of
            if rel in ['Subevent', 'partof', 'part of'] and (
                    (e1, e2) in test_lines.keys() or (e2, e1) in test_lines.keys()):
                filter_num+=1
                if (e1, e2) in test_lines.keys():
                    del test_lines[(e1, e2)]
                if (e2, e1) in test_lines.keys():
                    del test_lines[(e2, e1)]
                #continue
            try:
                r_raw = Graph.edges[e1,e2]['relation']
                r, index_rel_tmp = r_raw.split('__')
                rlst = r.split('/')
                if rel not in rlst:
                    rel = r + '/' + rel + '__' + index_rel_tmp
                    Graph.edges[e1,e2]['relation'] = rel
            except Exception:
                rel_suf = rel + '__' + str(index_rel)
                Graph.add_edge(e1, e2, relation=rel_suf)
                Graph_T.add_edge(e2, e1)
                index_rel += 1

        #   将text文档读取为三元组并构图
        title = ''  #   存储最近的title事件
        for text_line in text_lines:
            #   如果该行内容为title
            if '[' not in text_line:
                title = text_line.replace('\n', '')
                # if title in test_event_list:
                #     continue
            #   如果该行内容为子事件信息
            else:
                events = [] #   该行所有相关事件列表
                rel, event_list = text_line.strip().split('\t', maxsplit=1) #   分隔关系与事件
                r = '[[\\]]+'   #   去除[]
                event_list = re.sub(r, '', event_list)
                #首先判断并筛选出""内的内容
                doubles = re.findall('"(.*?)"',event_list)
                #   如果存在双引号，存储事件内容后，将文本中双引号内容去除
                if doubles:
                    for double in doubles:
                        events.append(double)
                        double = '"' + double + '"' + ','
                        event_list = re.sub(double, '', event_list)
                #   双引号内容已去除/本来就没有双引号的情况，只提取单引号内的事件内容
                singles = re.findall('\'(.*?)\'', event_list)
                #   假如存在单引号
                if singles:
                    for single in singles:
                        events.append(single)
                #   获得该行的事件列表后构建图
                for event in events:
                    # if event in test_event_list:
                    #     continue
                    # TODO:表示子事件的关系有Subevent,partof,part of
                    if rel in ['Subevent', 'partof', 'part of'] and (
                            (title, event) in test_lines.keys() or (event, title) in test_lines.keys()):
                        filter_num += 1
                        if (title, event) in test_lines.keys():
                            del test_lines[(title, event)]
                        if (event, title) in test_lines.keys():
                            del test_lines[(event, title)]
                        #continue
                    #   如果以存在事件间关系，则修改关系名称并统计多重关系的事件对数量
                    try:
                        rel_before_raw = Graph.edges[title,event]['relation']
                        rel_before, index_rel_tmp = rel_before_raw.split('__')
                        rel_lst = rel_before.split('/')
                        if rel not in rel_lst:
                            rel_after = rel_before + '/' + rel + '__' + index_rel_tmp
                            Graph.edges[title,event]['relation'] = rel_after
                    except Exception:
                        rel_suf = rel + '__' + str(index_rel)
                        Graph.add_edge(title, event, relation=rel_suf)
                        Graph_T.add_edge(event, title)
                        index_rel += 1
        print('filter nums:{}'.format(filter_num))
        return Graph, Graph_T, test_lines

    def read_text_dat(self, input_file):
        '''
        样本中包含重复样本
        :param input_file:
        :return:
        '''
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    lines.append(line.lower())
        return lines



def bfs(event, events_dict, depth=3):
    '''
    Counter({2: 1737, 3: 377, 4: 111, 5: 38, 7: 10, 6: 9, 9: 2, 1: 2})
    Counter({2: 2223, 3: 492, 4: 137, 5: 46, 6: 12, 7: 12, 1: 3, 9: 2, 8: 1})
    Counter({2: 2415, 3: 541, 4: 153, 5: 50, 6: 14, 7: 12, 1: 3, 9: 2, 8: 1})
    '''
    neighborhood = set()#string:tuple
    #kset = set() #all keys
    if depth == 1:
        try:
            nbs = list(events_dict[event].keys())
            for nb in nbs:
                neighborhood.add((event, nb, json.dumps(events_dict[event][nb])))
            return neighborhood
        except:
            return set()

    else:
        try:
            #neighborhood = list(events_dict[event].keys())
            nbs = list(events_dict[event].keys())
            for nb in nbs:
                neighborhood.add((event, nb, json.dumps(events_dict[event][nb])))
        except:
            return set()

        if len(neighborhood) > 0:
            unvisited_neighbors = copy.deepcopy(neighborhood)
            while len(unvisited_neighbors) > 0:
                neighbor = unvisited_neighbors.pop()
                # print(neighbor)
                tmp = bfs(neighbor[1], events_dict, depth=depth - 1)
                neighborhood = neighborhood.union(tmp)
                #neighborhood += tmp
        else:
            neighborhood = set()

    return neighborhood

def adjacency_matrix_matching(evidence_ls, events_dict):

    adjacency_matrix = np.zeros((max_L, max_L))

    for ith, event_1 in enumerate(evidence_ls):
        for jth, event_2 in enumerate(evidence_ls):
            try:
                if event_2 in events_dict[event_1]:
                    adjacency_matrix[ith, jth] = 1
            except:
                pass
        adjacency_matrix[ith, ith] = 1#self-loop

    return adjacency_matrix

def truncate(ls, triples):
    '''
    if the length of evidence_id is shorter than max_L, it will be padded in the data pretraining process before feed into NN.
    '''
    triples_ = []
    L = len(ls)
    diff = max_L - L
    if diff < 0:
        ls = ls[:max_L]# in order of importance
        for triple in triples:
            if len(set(triple)-set(ls))==3:#remove the triples whose 3 elements are all out of event list
                continue
            else:
                triples_.append(triple)
        return ls, triples_
    else:
        return ls, triples

def truncate_nodes(ls, triples, sentences):
    '''
    截断节点时，同时更新三元组和上下文结构
    '''
    triples_ = []
    sentences_ = []
    L = len(ls)
    diff = max_L - L
    if diff < 0:
        ls = ls[:max_L]# in order of importance
        for triple in triples:
            if len(set(triple)-set(ls))==3:#remove the triples whose 3 elements are all out of event list
                continue
            else:
                triples_.append(triple)
        for sentence in sentences:
            if len(set(sentence)-set(ls)) == len(set(sentence)):
                continue
            else:
                #sentence中包含的所有事件可能只有部分保留
                sentence_ = ['']
                last_type = False#表示sentence_中当前最后一个状态，两种，上下文(False)/事件(True)
                for piece in sentence:
                    if piece in ls:
                        sentence_.append(piece)
                        last_type = True
                    else:
                        if not last_type:
                            sentence_[-1] +=piece
                        else:
                            sentence_.append(piece)
                sentences_.append([piece for piece in sentence_ if piece.strip()!=''])
        return ls, triples_, sentences_
    else:
        return ls, triples, sentences

def path_info(path, G):
    print(path)
    index = 0
    length = len(path)
    path_info = []
    while index < (length - 1):
        #   获取相邻节点信息
        source = path[index]
        target = path[index + 1]
        #   获取边信息
        edge = G[source][target]['relation']
        #   规范化格式 (e1|||e2|||rel),存储进该事件对路径列表中
        path_unit = '(' + source + '|||' + target + '|||' + edge + ')'
        path_info.append(path_unit)
        index = index + 1
    #   生成每行文本
    result_path = ','.join(path_info)
    return result_path

def get_nbs(e1_e2_pth, e2_e1_pth, e1, e2, events_dict, events_dict_T):

    nb_last1 = set()
    nb_last2 = set()
    if e1_e2_pth:
        n1 = bfs(e1, events_dict)
        n2 = bfs(e2, events_dict_T)

        neighbors = copy.deepcopy(n1)
        n1_ = set([(e1,e2) for (e1,e2,r) in n1])
        for (e1,e2,r) in n2:
            if (e1, e2) not in n1_ and (e2, e1) not in n1_:
                neighbors.add((e1,e2,r))

        #neighbors = n1.intersection(n2)
        for n in neighbors:
            if n in n1:
                if n not in nb_last1:
                    nb_last1.add(n)
            if n in n2:
                if (n[1], n[0], json.dumps(events_dict[n[1]][n[0]])) not in nb_last1:
                    nb_last1.add((n[1], n[0], json.dumps(events_dict[n[1]][n[0]])))

    if e2_e1_pth:
        n1 = bfs(e2, events_dict)
        n2 = bfs(e1, events_dict_T)

        neighbors = copy.deepcopy(n1)
        n1_ = set([(e1, e2) for (e1, e2, r) in n1])
        for (e1, e2, r) in n2:
            if (e1, e2) not in n1_ and (e2, e1) not in n1_:
                neighbors.add((e1, e2, r))

        #neighbors = n1.intersection(n2)
        for n in neighbors:
            if n in n1:
                if n not in nb_last2:
                    nb_last2.add(n)
            if n in n2:
                if (n[1], n[0], json.dumps(events_dict[n[1]][n[0]])) not in nb_last2:
                    nb_last2.add((n[1], n[0], json.dumps(events_dict[n[1]][n[0]])))

    nb_last = nb_last1.union(nb_last2)

    return nb_last

def get_nbs_in_KG(e1_e2_pth, e2_e1_pth, e1, e2, events_dict, events_dict_T):
    #nb_last = {}  # to supplement the current event_dict
    # 用于补充KG中的邻居，目标：同时出现在e1,e2邻域内的事件，相当于取交集，但因为events_dict_T中无关系类型，需要特殊处理一下，不能直接intersection
    nb_last1 = set()
    nb_last2 = set()
    if e1_e2_pth:
        n1 = bfs(e1, events_dict)
        n2 = bfs(e2, events_dict_T)

        neighbors = set()
        n2_ = set([(e1, e2) for (e1, e2, r) in n2])
        for (e1, e2, r) in n1:
            if (e1, e2) in n2_ or (e2, e1) in n2_:
                neighbors.add((e1, e2, r))

        for n in neighbors:
            if n in n1:
                if n not in nb_last1:
                    nb_last1.add(n)
            if n in n2:
                if (n[1], n[0], json.dumps(events_dict[n[1]][n[0]])) not in nb_last1:
                    nb_last1.add((n[1], n[0], json.dumps(events_dict[n[1]][n[0]])))

    if e2_e1_pth:
        n1 = bfs(e2, events_dict)
        n2 = bfs(e1, events_dict_T)

        neighbors = set()
        n2_ = set([(e1, e2) for (e1, e2, r) in n2])
        for (e1, e2, r) in n1:
            if (e1, e2) in n2_ or (e2, e1) in n2_:
                neighbors.add((e1, e2, r))

        for n in neighbors:
            if n in n1:
                if n not in nb_last2:
                    nb_last2.add(n)
            if n in n2:
                if (n[1], n[0], json.dumps(events_dict[n[1]][n[0]])) not in nb_last2:
                    nb_last2.add((n[1], n[0], json.dumps(events_dict[n[1]][n[0]])))

    nb_last = nb_last1.union(nb_last2)
    return nb_last

#   不仅过滤目标事件对，还要过滤文本中的事件
def get_test_event_list(pair_list):
    event_dic = {}
    with open(pair_list) as f:
        for line in f:
            if line.strip():
                rel, cont = line.lower().strip().split('\t', maxsplit=1)
                e1, e2, passage, *section_names = cont.strip().split('|||')
                e1, e2, passage = e1.strip(), e2.strip(), passage.strip()
                event_dic[(e1,e2)] = line

    return event_dic

#   读取每行的e1，e2
def read_seri_eventpairs(line):
    rel, cont = line.lower().strip().split('\t', maxsplit=1)
    e1, e2, passage, *section_names = cont.strip().split('|||')
    e1, e2, passage = e1.strip(), e2.strip(), passage.strip()
    return e1, e2



#   读取每行的e1，e2，关系rel, 段落标题列表section_name_list, 上下文passage, 文本内普通事件列表event_list_normal, index_sentence
def read_seri_line(line, eventwiki_set, tokenizer):
    rel, cont = line.lower().strip().split('\t', maxsplit=1)
    e1, e2, passage, *section_names = cont.strip().split('|||')
    e1, e2, passage = e1.strip(), e2.strip(), passage.strip()
    section_names = [name.strip() for name in section_names]
    #   新增的上下文二维列表，每个item为一句含有事件的句子list（按照事件分段）,list长度最多为5
    sentences_list = []
    #   存储段落标题section_name（可能有多个），标题顺序从前往后为又大至小（后属于前，前属于title）
    section_name_list = []
    for r in section_names:
        if r != '':
            if r != e1 and r in eventwiki_set:
                section_name_list.append([r])#保证和sentences_list维度相同
    #   获取句内普通事件节点
    #   2.  对以上信息进行预处理，title等使用规则提取事件名，passage按照'''event'''[[event]]进行过滤
    sentences = []
    passage = passage.replace('maj.', 'maj')
    passage = passage.replace('gen.', 'gen')
    passage = passage.replace('brig.', 'brig')
    passage = passage.replace('gens.', 'gens')
    passage = passage.replace('col.', 'col')
    #   使用共指消解替换事件
    #passage = coref(passage)
    sentences_raw = sent_tokenize(passage)
    num_sentence = len(sentences_raw)
    index_split = 0
    for sentence in sentences_raw:
        if len(sentence) < 25:
            if (index_split == 0):
                if (num_sentence > 1):
                    sentence = sentence + sentences_raw[index_split + 1]
                sentences.append(sentence)
            else:
                if (index_split < (num_sentence - 1)):
                    sentence = sentences[-1] + sentence + sentences_raw[index_split + 1]
                    sentences.pop()
                    sentences.append(sentence)
                else:
                    sentence = sentences[-1] + sentence
                    sentences.pop()
                    sentences.append(sentence)
        else:
            sentences.append(sentence)
        index_split += 1
    event_list_normal = [[] for _ in range(len(sentences))]  # 存储每句话中的每个普通事件
    #   对seri的passage进行分句，仅保留含有事件的句子（含段落标题）
    # TODO:对序列最大长度max_seq_length的约束在example生成前处理,即如果仅包含的事件在分词后长度就大于max_seq_length,就去掉该句话
    event_tokens_len = 0#记录一句中事件分词后的总长度
    event_nums = 0#记录一句中的事件个数
    for index_sentence, sentence in enumerate(sentences):
        has_event_in_sen = False
        sentence_split_list = []  #   句子按照事件分段
        #   使用正则passage中[[]]和'''xxx'''内信息然后使用EventWiki过滤出事件
        event_link = []  # 存储该句话从[[]]/''''''中抽取的事件
        pattern = r"'''[\d\s\w\-\'\"\{\}\(\)\[\]|\\*&.?!,…:;]*?'''"
        quote_links = re.findall(pattern, sentence)
        quote_links = [link[3:-3] for link in quote_links]
        context = re.split(pattern, sentence)
        sentence_split_list.append(context[0])
        for i in range(len(quote_links)):
            has_event = False
            tmp = quote_links[i].split('[[')
            if len(tmp) == 1:
                tmp = tmp[0]
                if tmp not in event_link:
                    if tmp in eventwiki_set:
                        event_link.append(tmp)
                        has_event = True
                        event_nums += 1
                        event_tokens_len += len(tokenizer.tokenize(tmp))
                        sentence_split_list.append(tmp)
                        sentence_split_list.append(context[i + 1])
                    else:
                        sentence_split_list[-1] += tmp
                        sentence_split_list[-1] += context[i + 1]
                else:
                    has_event = True
                    event_nums += 1
                    event_tokens_len += len(tokenizer.tokenize(tmp))
                    sentence_split_list.append(tmp)
                    sentence_split_list.append(context[i + 1])
            else:
                tmp_ = ''.join(''.join(tmp).split(']]'))
                if tmp_ not in event_link:
                    if tmp_ in eventwiki_set:
                        event_link.append(tmp_)
                        has_event = True
                        event_nums += 1
                        event_tokens_len += len(tokenizer.tokenize(tmp_))
                        sentence_split_list.append(tmp_)
                        sentence_split_list.append(context[i + 1])
                    else:
                        tmp[-1] = '[[' + tmp[-1]
                        sentence_split_list[-1] += ''.join(tmp)
                        sentence_split_list[-1] += context[i + 1]
                else:
                    has_event = True
                    event_nums += 1
                    event_tokens_len += len(tokenizer.tokenize(tmp))
                    sentence_split_list.append(tmp)
                    sentence_split_list.append(context[i + 1])
            if not has_event:
                continue
            else:
                has_event_in_sen = True

        #   在已被''' '''类型事件分割的句子中继续按照[[]]分割
        pattern = r'\[\[[\d\s\w\-\'\"\{\}\(\)|\\*&.?!,…:;]*\]\]'

        for index_piece, piece in enumerate(sentence_split_list):
            has_event = False
            if piece in event_link:
                continue
            if len(re.findall(pattern, piece)) == 0:
                continue
            piece_split_list = []
            ref_links = re.findall(pattern, piece)
            ref_links = [link[2:-2] for link in ref_links]
            contexts_piece = re.split(pattern, piece)
            piece_split_list.append(contexts_piece[0])
            for i in range(len(ref_links)):
                tmp = ref_links[i].split('|')
                if len(tmp) == 1:
                    t = tmp[0]
                    if t not in event_link:
                        if t in eventwiki_set:
                            event_link.append(t)
                            has_event = True
                            event_nums += 1
                            event_tokens_len += len(tokenizer.tokenize(t))
                            piece_split_list.append(t)
                            piece_split_list.append(contexts_piece[i + 1])
                        else:
                            piece_split_list[-1] += t
                            piece_split_list[-1] += contexts_piece[i + 1]
                    else:
                        has_event = True
                        event_nums += 1
                        event_tokens_len += len(tokenizer.tokenize(t))
                        piece_split_list.append(t)
                        piece_split_list.append(contexts_piece[i + 1])
                else:
                    #[event|event mention]，有个别列子event描述和e1,e2不同，event mention反而和e1,e2匹配
                    #存在两个都是知识库里的event的情况，此时判断有没有e1、e2在其中
                    if tmp[0] == e1 or tmp[0] == e2:
                        if tmp[0] not in event_link:
                            event_link.append(tmp[0])
                        has_event = True
                        event_nums += 1
                        event_tokens_len += len(tokenizer.tokenize(tmp[0]))
                        piece_split_list.append(tmp[0])
                        piece_split_list.append(contexts_piece[i + 1])
                    elif tmp[1] == e1 or tmp[1] == e2:
                        if tmp[1] not in event_link:
                            event_link.append(tmp[1])
                        has_event = True
                        event_nums += 1
                        event_tokens_len += len(tokenizer.tokenize(tmp[1]))
                        piece_split_list.append(tmp[1])
                        piece_split_list.append(contexts_piece[i + 1])
                    else:
                        if tmp[0] not in event_link:
                            if tmp[0] in eventwiki_set:
                                event_link.append(tmp[0])
                                has_event = True
                                event_nums += 1
                                event_tokens_len += len(tokenizer.tokenize(tmp[0]))
                                piece_split_list.append(tmp[0])
                                piece_split_list.append(contexts_piece[i + 1])
                                continue
                            else:
                                #如果tmp[0]不是，在判断下tmp[1]是不是，如果都不是该例子跳过
                                if tmp[1] not in event_link:
                                    if tmp[1] in eventwiki_set:
                                        event_link.append(tmp[1])
                                        has_event = True
                                        event_nums += 1
                                        event_tokens_len += len(tokenizer.tokenize(tmp[1]))
                                        piece_split_list.append(tmp[1])
                                        piece_split_list.append(contexts_piece[i + 1])
                                        continue
                                    else:
                                        piece_split_list[-1] += tmp[1]
                                        piece_split_list[-1] += contexts_piece[i + 1]
                                        continue
                                else:
                                    has_event = True
                                    event_nums += 1
                                    event_tokens_len += len(tokenizer.tokenize(tmp[1]))
                                    piece_split_list.append(tmp[1])
                                    piece_split_list.append(contexts_piece[i + 1])
                        else:
                            has_event = True
                            event_nums += 1
                            event_tokens_len += len(tokenizer.tokenize(tmp[0]))
                            piece_split_list.append(tmp[0])
                            piece_split_list.append(contexts_piece[i + 1])
            #   如果该分段中没有事件，不做分割处理
            if not has_event:
                continue
            else:
                has_event_in_sen = True
                del sentence_split_list[index_piece]
                #sentence_split_list中元素可能为string,可能为list
                sentence_split_list.insert(index_piece, piece_split_list)

        #   生成event_list_normal
        for event_link_item in event_link:
            if event_link_item not in event_list_normal[index_sentence]:
                event_list_normal[index_sentence].append(event_link_item)
        #TODO:保证句子在长度截断时无异常
        if event_tokens_len >= max_seq_length- 2 - 2 * event_nums:#刨除首尾<s></s>和每个事件首尾的</s>
            continue
        #sentence_split_list统一，元素均为string
        sentence_split_tmp = []
        for item in sentence_split_list:
            if type(item) == list:
                sentence_split_tmp.extend(item)
            else:
                sentence_split_tmp.append(item)

        if has_event_in_sen:
            sentence_split_tmp = [token for token in sentence_split_tmp if token.strip() != '']#过滤空字符
            if len(sentence_split_tmp) != 0:
                sentences_list.append(sentence_split_tmp)

    #sentences_list = sentences_list + section_name_list#封掉，因为标题事件和seperated_nodes没有区别
    #TODO:对句子最大的长度max_sentences在example生成前处理
    if len(sentences_list) > max_sentences:
        sentences_list = sentences_list[0:max_sentences]
    return e1, e2, rel, section_name_list, event_list_normal, sentences_list

#   获取数据集中所有存在对称事件对的事件对 [(e1, e2), (e2, e1), ... ]
def get_symmetry_pair_list(file):
    symmetry_pair_list = []
    all_seri_pair_list = get_all_seri_pair_list(file)#已无重复
    for (e1,e2) in all_seri_pair_list:
        if (e2,e1) in all_seri_pair_list and e2!=e1:
            symmetry_pair_list.append((e1, e2))#包含双向
    return symmetry_pair_list

#   获取数据集中所有事件对 [[e1, e2], ... ]
def get_all_seri_pair_list(file):
    all_seri_pair_list = []
    for line in file:
        if line.strip():
            e1, e2= read_seri_eventpairs(line)
            all_seri_pair_list.append((e1, e2))
    all_seri_pair_list = list(set(all_seri_pair_list))
    return all_seri_pair_list

#   构建passage小图
def graph_passage(e1, section_name_list, event_list_normal, G):
    events_dict = {}  # save corresponding events about the sample (e1,e2)

    #   passage小图定义
    G_passage = nx.DiGraph()
    G_passage_T = nx.DiGraph()
    #   标题节点
    events_dict[e1] = set()
    G_passage.add_node(e1)
    G_passage_T.add_node(e1)
    #   段落标题节点
    # section_name_list:[[],[],...]->[xx,xx,xx,..]
    section_name_list = [section_name[0] for section_name in section_name_list]
    for section_name_item in section_name_list:
        events_dict[section_name_item] = set()
        G_passage.add_node(section_name_item)
        G_passage_T.add_node(section_name_item)
    #   句内普通节点
    for i in range(len(event_list_normal)):
        for sentence_event in event_list_normal[i]:
            events_dict[sentence_event] = set()
            if sentence_event not in G_passage:
                G_passage.add_node(sentence_event)
                G_passage_T.add_node(sentence_event)
    #   Step1：无向图构建--句内共现原则全连接
    for i in range(len(event_list_normal)):
        for sentence_event_1 in event_list_normal[i]:
            for sentence_event_2 in event_list_normal[i]:
                if sentence_event_1 != sentence_event_2:
                    events_dict[sentence_event_1].add(sentence_event_2)
                    G_passage.add_edge(sentence_event_1, sentence_event_2)
                    G_passage.add_edge(sentence_event_2, sentence_event_1)
                    G_passage_T.add_edge(sentence_event_2, sentence_event_1)
                    G_passage_T.add_edge(sentence_event_1, sentence_event_2)
    #   Step2：无向图构建--段落标题节点与事件e1的连接
    #   如果存在段落标题
    if len(section_name_list) > 0:
        #   段落标题与根结点连接
        index_section = 0  # 当前段落标题编号
        section_name_last = ''  # 记录最后一个段落标题
        for section_name_item in section_name_list:
            #   将列表的首个段落标题作为第一层节点
            if index_section == 0:
                events_dict[e1].add(section_name_item)
                events_dict[section_name_item].add(e1)
                G_passage.add_edge(e1, section_name_item)
                G_passage.add_edge(section_name_item, e1)
                G_passage_T.add_edge(e1, section_name_item)
                G_passage_T.add_edge(section_name_item, e1)
            else:
                #   剩下的段落标题依次向下连接
                events_dict[section_name_item].add(section_name_list[index_section - 1])
                events_dict[section_name_list[index_section - 1]].add(section_name_item)
                G_passage.add_edge(section_name_item, section_name_list[index_section - 1])
                G_passage.add_edge(section_name_list[index_section - 1], section_name_item)
                G_passage_T.add_edge(section_name_item, section_name_list[index_section - 1])
                G_passage_T.add_edge(section_name_list[index_section - 1], section_name_item)
            index_section += 1
            section_name_last = section_name_item
        #   句子节点在内部全连接后与段落标题相连
        for i in range(len(event_list_normal)):
            # index_in_sentence = 0
            for sentence_event_1 in event_list_normal[i]:
                #   每个节点与段落标题相连(【排除与段落标题相同的情况)
                if section_name_last != sentence_event_1:
                    events_dict[section_name_last].add(sentence_event_1)
                    events_dict[sentence_event_1].add(section_name_last)
                    G_passage.add_edge(section_name_last, sentence_event_1)
                    G_passage.add_edge(sentence_event_1, section_name_last)
                    G_passage_T.add_edge(section_name_last, sentence_event_1)
                    G_passage_T.add_edge(sentence_event_1, section_name_last)
    #else:
    #TODO:无论是否存在段落标题，句内节点在内部全连接后直接与根结点连接
    for i in range(len(event_list_normal)):
        for sentence_event_1 in event_list_normal[i]:
            #   每个节点与标题根结点相连(【排除与标题相同的情况)
            if e1 != sentence_event_1:
                events_dict[e1].add(sentence_event_1)
                events_dict[sentence_event_1].add(e1)
                G_passage.add_edge(e1, sentence_event_1)
                G_passage.add_edge(sentence_event_1, e1)
                G_passage_T.add_edge(e1, sentence_event_1)
                G_passage_T.add_edge(sentence_event_1, e1)
    #   Step3:引入有向边（节点形式），利用远程监督思想确定边的方向性和边上的类型信息
    event_dict_tmp = copy.deepcopy(events_dict)
    del_set = []
    for ea in event_dict_tmp.keys():
        for eb in event_dict_tmp[ea]:
            # delete the same event pair only onces, e.g. (ea, eb) equals (eb, ea)
            try:
                r = G.edges[ea, eb]['relation']

                # delete undirected edge between the event pair if it was not deleted
                if (ea, eb) not in del_set:
                    events_dict[ea].discard(eb)
                    events_dict[eb].discard(ea)
                    G_passage.remove_edge(ea, eb)
                    G_passage.remove_edge(eb, ea)
                    G_passage_T.remove_edge(ea, eb)
                    G_passage_T.remove_edge(eb, ea)
                    # update del_set
                    del_set.append((ea, eb))
                    del_set.append((eb, ea))
                #   新增有向边
                events_dict[ea].update((r, eb))
                events_dict[r] = set()
                events_dict[r].add(eb)
                G_passage.add_edge(ea, eb, relation=r)
                G_passage_T.add_edge(eb, ea)
            except Exception:
                continue
    return G_passage, G_passage_T, events_dict

#   对G_passage，G_passage_T进行bfs，dijstra得到邻居节点，事件列表priority_set, priority_set_passage
def get_nb_last(e1, e2, G_passage, G_passage_T, events_dict, G, events_dict_EW, events_dict_EWT):
    e1_e2_pth = False
    e2_e1_pth = False
    e1_e2_pth_passage = False
    e2_e1_pth_passage = False
    priority_set = set()  # save the neighbors in the dijstra_path, if event list longer than the max length, prevent from deleting
    priority_set_passage = set()
    # nb_last = set()
    #   先在passage小图中搜索路径
    #   对G_passage，G_passage_T进行bfs，dijstra得到邻居节点，事件列表
    events_dict_passage = {i: dict(j) for i, j in dict(G_passage.adj).items()}
    events_dict_passageT = {i: dict(j) for i, j in dict(G_passage_T.adj).items()}
    try:
        pth1_passage = nx.dijkstra_path(G_passage, e1, e2)
        e1_e2_pth_passage = True
        # get priority nodes set if it exists
        for i in range(len(pth1_passage) - 1):#自身到自身不考虑
            source = pth1_passage[i]
            target = pth1_passage[i + 1]
            try:  # directed/undirected edge
                edge = G_passage[source][target]['relation']
                priority_set_passage.update((source, edge, target))
            except Exception:
                priority_set_passage.update((source, target))
    except Exception:
        pass
    try:
        pth2_passage = nx.dijkstra_path(G_passage, e2, e1)
        e2_e1_pth_passage = True
        # get priority nodes set if it exists
        for i in range(len(pth2_passage) - 1):
            source = pth2_passage[i]
            target = pth2_passage[i + 1]
            try:  # directed/undirected edge
                edge = G_passage[source][target]['relation']
                priority_set_passage.update((source, edge, target))
            except Exception:
                priority_set_passage.update((source, target))
    except Exception:
        pass
    nb_last_passage = get_nbs(e1_e2_pth_passage, e2_e1_pth_passage, e1, e2, events_dict_passage, events_dict_passageT)

    #   在大图中搜索路径
    '''
    try to get all possible neighbors occurred in the path between e1 and e2
    using function 'dijkstra_path()' to judge if a path exists
    ----------------------------------------------------------------------------
    if a path exists between e1 to e2, but not exists between e2 to e1
        n1 = bfs(e1, events_dict_EW), n2 = bfs(e2, events_dict_EWT), neighbors = intersection(n1,n2)
    if a path exists between e2 to e1, but not exists between e1 to e2
        n1 = bfs(e2, events_dict_EW), n2 = bfs(e1, events_dict_EWT), neighbors = intersection(n1,n2)
    if a path exists between e2 to e1, and exists between e1 to e2
        n1 = bfs(e1, events_dict_EW), n2 = bfs(e2, events_dict_EWT), n3 = intersection(n1,n2)
        n4 = bfs(e2, events_dict_EW), n5 = bfs(e1, events_dict_EWT), n6 = intersection(n4,n5)
        neighbors = union(n3,n6)
    if no path between e1 and e2
        do nothing to prevent the introduction of noise
    '''
    try:
        pth1 = nx.dijkstra_path(G, e1, e2)
        e1_e2_pth = True
        # get priority nodes set if it exists
        for i in range(len(pth1) - 1):
            source = pth1[i]
            target = pth1[i + 1]
            # create nodes
            edge = G[source][target]['relation']
            if events_dict.get(source) == None:
                events_dict[source] = set()
            if events_dict.get(target) == None:
                events_dict[target] = set()
            if events_dict.get(edge) == None:
                events_dict[edge] = set()
            # create edges
            events_dict[source].update((edge, target))
            events_dict[edge].add(target)
            priority_set.update((source, edge, target))

    except Exception:
        pass
    try:
        pth2 = nx.dijkstra_path(G, e2, e1)
        e2_e1_pth = True
        # get priority nodes set if it exists
        for i in range(len(pth2) - 1):
            source = pth2[i]
            target = pth2[i + 1]
            # create nodes
            edge = G[source][target]['relation']
            if events_dict.get(source) == None:
                events_dict[source] = set()
            if events_dict.get(target) == None:
                events_dict[target] = set()
            if events_dict.get(edge) == None:
                events_dict[edge] = set()
            # create edges
            events_dict[source].update((edge, target))
            events_dict[edge].add(target)
            priority_set.update((source, edge, target))

    except Exception:
        pass

    nb_last = get_nbs_in_KG(e1_e2_pth, e2_e1_pth, e1, e2, events_dict_EW, events_dict_EWT)
    return priority_set, priority_set_passage, nb_last, nb_last_passage

#update events_dict via nb_last
def update_evdict_via_nb(events_dict, nb_last):
    #for nb in nb_last.keys():
        # use set() to filter same elements, nb can be source/target event
        #for triple in nb_last[nb]:
    for triple in nb_last:
        source_e = triple[0]  # format:(source_e, nb, edge)
        edge = json.loads(triple[2])['relation']
        target_e = triple[1]
        if events_dict.get(source_e) == None:
            events_dict[source_e] = set()
        if events_dict.get(edge) == None:
            events_dict[edge] = set()
        if events_dict.get(target_e) == None:
            events_dict[target_e] = set()

            # create edges
            events_dict[source_e].update((edge, target_e))
            events_dict[edge].add(target_e)
    return events_dict

def final_process(e1, e2, priority_set, priority_set_passage,nb_last, nb_last_passage, sentences_list):
    # keep e1,e2 a higher priority
    priority_lst = list(priority_set_passage)
    if e1 in priority_lst:
        priority_lst.remove(e1)
    if e2 in priority_lst:
        priority_lst.remove(e2)
    priority_lst.insert(0, e1)
    if e2 != e1:  # e1 can be same with e2,e.g. battle of surabaya
        priority_lst.insert(1, e2)

    priority_set_passage = set(priority_lst)
    triple_nodes = []
    passage_nodes = []
    triples = []
    for triple in nb_last:
        tmp = list(triple)
        tmp[2] = json.loads(tmp[2])['relation']
        for ev in tmp:
            if ev != '{}':
                triple_nodes.append(ev)
        # triple format:(e1, r, e2)
        tmp[1], tmp[2] = tmp[2], tmp[1]
        if tmp not in triples:
            triples.append(tmp)

    for triple in nb_last_passage:
        tmp = list(triple)
        try:
            tmp[2] = json.loads(tmp[2])['relation']  # undirected edge
            for ev in tmp:
                passage_nodes.append(ev)
            tmp[1], tmp[2] = tmp[2], tmp[1]
            if tmp not in triples:
                triples.append(tmp)
        except Exception:
            for event in tmp:#表示没有triples，只有共现
                if event != '{}':
                    passage_nodes.append(event)

    passage_nodes = set(passage_nodes)
    passa = list(passage_nodes - priority_set_passage)

    triple_nodes = set(triple_nodes)
    remain = list(priority_set - priority_set_passage - passage_nodes)
    general_nodes = list(triple_nodes - priority_set - priority_set_passage - passage_nodes)

    random.shuffle(general_nodes)
    ev_lst = priority_lst + passa + remain + general_nodes
    # truncate event_list and filter redundant triples
    neighbors_lst, triples_fil, sentences_fil = truncate_nodes(ev_lst, triples, sentences_list)
    if len(neighbors_lst) < (len(remain) + len(priority_lst) + len(passa)):
        print(e1, e2, 'prior_len', (len(remain) + len(priority_lst) + len(passa)), 'nbs_len_truncated', len(neighbors_lst))
    return neighbors_lst, triples_fil, sentences_fil

#   判断是否是对称事件
def is_symmetry(e1, e2, symmetry_pair_list):
    if (e1,e2) in symmetry_pair_list:
        return True
    else:
        return False

def load_save_all_examples(tokenizer):
    #   获取EventWiki中所有事件列表
    eventwiki_list = []
    with open(event_file, 'r') as f_ev:
        for line in f_ev.readlines():
            if line.strip():
                eventwiki_list.append(re.sub('\n', '', line.lower()))
    eventwiki_set = set(eventwiki_list)
    test_file = '{}/HGAM/data/SeRI_mod/test_pair.dat'.format(root_pth)
    test_lines = get_test_event_list(test_file)
    #   有向有环图定义
    G = nx.DiGraph()
    G_T = nx.DiGraph()

    processor = EWProcessor()
    print('test_lines length:{}'.format(len(test_lines)))#1671
    G, G_T, test_lines = processor.get_examples(input_dir, G, G_T, test_lines)
    print('test_lines length:{}'.format(len(test_lines)))#1055
    with open('{}/HGAM/data/SeRI_mod/{}_pair.dat'.format(root_pth, 'test1'), 'w') as ftest:
        for line in test_lines.values():
            ftest.write(line)

    events_dict_EW = {i:dict(j) for i,j in dict(G.adj).items()}
    events_dict_EWT = {i:dict(j) for i,j in dict(G_T.adj).items()}

    for status in ['eval', 'test1', 'train']:
        wo_times = 0
        pair_list = '{}/HGAM/data/SeRI_mod/{}_pair.dat'.format(root_pth, status)
        examples = []
        with open(pair_list) as f:
            sample_id = 0
            for line in f:
                if line.strip():
                    e1, e2, rel, section_name_list, event_list_normal, sentences_list = read_seri_line(line, eventwiki_set, tokenizer)
                    G_passage, G_passage_T, events_dict = graph_passage(e1, section_name_list, event_list_normal, G)
                    priority_set, priority_set_passage, nb_last, nb_last_passage = get_nb_last(e1, e2, G_passage, G_passage_T, events_dict, G, events_dict_EW, events_dict_EWT)

                    events_dict = update_evdict_via_nb(events_dict, nb_last)
                    neighbors_lst, triples_fil, sentences_fil = final_process(e1, e2, priority_set, priority_set_passage, nb_last, nb_last_passage, sentences_list)

                    context_pos = len(triples_fil)  #第context_pos个元组为上下文,当前为边长，转化为feature的时候进一步更新
                    triples_context_fil = triples_fil + sentences_fil
                    #TODO:当无外部信息利用时，直接单节点编码
                    if len(triples_context_fil) == 0:#事件对既没有在三元组中也没有在上下文中，过滤掉，当做噪声
                        wo_times += 1
                        print('wo external info for encoding nodes for {} times'.format(wo_times))
                    examples.append(Example(
                        id='{}-{}'.format(status, sample_id),
                        label=rel,
                        e1=e1,
                        e2=e2,
                        context_pos=context_pos,
                        ev_lst=neighbors_lst,
                        e_triples=triples_context_fil,
                        events_dict=events_dict
                    ))
                    sample_id += 1
        examples_files = '{}/HGAM/data/graph/{}_examples_roberta_maxL{}_maxtriples{}_maxseqlen{}_maxpos{}_maxsens{}.pkl'.\
            format(root_pth, status, max_L, max_triples, max_seq_length, max_positions, max_sentences)
        print("Saving {} examples into file {}".format(status, examples_files))
        with open(examples_files, 'wb') as f:
            pickle.dump(examples, f)
    return
if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('{}/HGAM/model/roberta_base/'.format(root_pth),
                                                do_lower_case=True, unk_token='<unk>')
    load_save_all_examples(tokenizer)