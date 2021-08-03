import argparse
import os
import sys

root_pth = '/home/xxx'
o_path = '{}/HGAM'.format(root_pth)
sys.path.insert(0, o_path)#load local transformer not the package one
print(sys.path)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import precision_score,recall_score, f1_score, classification_report

'''https://stackoverflow.com/questions/61211685/how-can-i-load-a-partial-pretrained-pytorch-model'''
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel, RobertaTokenizer

from transformers import AdamW, get_linear_schedule_with_warmup
import logging
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from ev_and_passage_roberta_ck3 import *
from graph_model import *

def simple_accuracy(preds, labels):
    return (preds == labels).mean()#.float().mean()

def pre_rec_f1(preds, labels):

    precision = precision_score(labels, preds, labels=[0,2], average = 'micro')#micro
    recall = recall_score(labels, preds, labels=[0,2], average = 'micro')
    f1 = f1_score(labels, preds, labels=[0,2], average = 'micro')
    print(classification_report(labels, preds))

    return {
        "pre": precision,
        "rec": recall,
        "f1": f1
    }

def has_improved(m1, m2):
    return m1["f1"] < m2["f1"]

def init_metric_dict():
    return {'pre': -1, 'rec': -1, 'f1': -1}


class RobertaHeteGat(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.dim*2, config.num_labels)

        self.init_weights()
        self.emb_dim = config.hidden_size#768
        self.num_layers = args.num_layers#2
        args.hidden_dropout_prob = config.hidden_dropout_prob
        args.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        args.method = 'self'
        gat_layer = []
        for i in range(self.num_layers):
            gat_layer.append(
                GraphAttentionLayer(args)
            )
        self.layers = nn.Sequential(*gat_layer)
        self.weight = args.weight
        self.weights = torch.Tensor([2.966,0.536,1.254])
        args.method = 'cross'
        self.outlayer = GraphAttentionLayer(args)
        self.gamma = 4
        self.alpha = [2.966,0.536,1.254]

    def compute_metrics(self, se_features, te_features, output, label_ids):
        # caculate the second term of the loss function
        label_map = {'-1':-2, '0': -1, '1': 0, '2': 1, '3': 2}
        se_norm = torch.norm(se_features, dim=1)
        te_norm = torch.norm(te_features, dim=1)
        cos_sim = F.cosine_similarity(se_features, te_features, dim=-1)
        loss_func = nn.CrossEntropyLoss(weight=self.weights.type_as(output))
        label_ids_lst = label_ids.cpu().numpy().tolist()
        label_ids_lst_new = [label_map[str(label)] for label in label_ids_lst]

        label_ids_new = torch.tensor(label_ids_lst_new).to(device=cos_sim.device, dtype=cos_sim.dtype)
        loss_h_total = torch.tensor((), requires_grad=True, device=cos_sim.device)

        for m in range(se_features.shape[0]):#依次遍历batch中元素
            if label_ids_new[m] == 1 or label_ids_new[m] == -1:#A1
                loss_h = torch.mul((1 - cos_sim[m]), torch.exp(torch.mul(label_ids_new[m], (se_norm[m] - te_norm[m]))/ torch.max(
                    torch.tensor((se_norm[m], te_norm[m])).type_as(cos_sim[m]))))
                loss_h_total = torch.hstack((loss_h_total, loss_h))
            elif label_ids_new[m] == 2:#A2
                loss_h = torch.mul((1 - cos_sim[m]), torch.square(se_norm[m] - te_norm[m])/ torch.max(
                    torch.tensor((se_norm[m], te_norm[m])).type_as(cos_sim[m])))
                loss_h_total = torch.hstack((loss_h_total, loss_h))

        loss = loss_func(output, label_ids)
        loss_h = torch.mean(loss_h_total) if len(loss_h_total)>0 else torch.tensor(0., requires_grad=True)
        print('loss:{},loss_h:{}'.format(loss, loss_h))
        loss_total = loss + self.weight*loss_h
        return loss_total

    def get_output(self, input_ids, evs_pos, input_mask, segment_ids, context_pos):
        batch_size = input_ids.shape[0]
        num_triples = input_ids.shape[1]#max_triples,12
        num_nodes = evs_pos.shape[1]#max_L,15
        max_triple_len = input_ids.shape[2]#max_seq_length,80

        node_embeddings = torch.zeros([batch_size, num_nodes, self.emb_dim], dtype=torch.float, device=input_ids.device)
        for m in range(batch_size):
            input_ids_ = input_ids[m, :, :]
            input_mask_ = input_mask[m, :, :]
            segment_ids_ = segment_ids[m, :, :]
            evs_pos_ = evs_pos[m, :, :].cpu().numpy().tolist()#(num_nodes, max_postions,2)
            evs_pos_dict = {}#{node_index, coordianate}
            for i in range(num_nodes):
                for [r, c] in evs_pos_[i]:
                    if [r,c] != [-1,-1]:
                        if evs_pos_dict.get(i) == None:
                            evs_pos_dict[i] = []
                        evs_pos_dict[i].append((r,c))
                    else:
                        break
            average_m = torch.ones([num_nodes, self.emb_dim], dtype=torch.float, device=input_ids.device)
            for i, tus in evs_pos_dict.items():
                weighted = 1./len(tus)
                average_m[i] *= weighted

            evs_pos_r_dict = {}#{coordinate:node_index}
            for i, tus in evs_pos_dict.items():
                for tu in tus:
                    evs_pos_r_dict[tu] = i

            for n in range(num_triples):
                if n < context_pos[m]:
                    input_ids_tmp = input_ids_[n, :].unsqueeze(dim=0)
                    #judge whether should stop, if input_ids_tmp == [0,0,...,0], it means no triples in this sample
                    if torch.equal(input_ids_tmp, torch.zeros([1, max_triple_len]).type_as(input_ids_tmp)):
                        break
                    input_mask_tmp = input_mask_[n,:].unsqueeze(dim=0)


                    output = self.roberta(input_ids=input_ids_tmp,
                                       attention_mask=input_mask_tmp)#, token_type_ids = segment_ids_tmp

                    tmp = input_ids_tmp.squeeze().cpu().numpy().tolist()
                    #   这块是否需要根据roberta修改
                    se_index = 0    #[CLS] 定位了[CLS]token,用于表示E1
                    rel_index = tmp.index(2)  #[SEP] 定位了第一个[SEP],用于表示R12
                    te_index = tmp.index(2, rel_index+1)  #   定位了第二个[SEP]，用于表示E2
                    if rel_index + 1 != te_index:
                        # that means that's a triple, has relation and target event, or rel_index+1== te_index means a single node
                        '''due to truncated event list, nodes in triples may not appear in event_pos，i.e. cannot find in evs_pos_r_dic, just ignore this case'''
                        try:
                            rel = evs_pos_r_dict[(n, 1)]
                            node_embeddings[m][rel] += output[0][:, rel_index, :].squeeze()  # (n,1)
                            #node_embeddings[m][rel] += output[0][:, rel_index+1:te_index, :].squeeze().mean(dim=0)
                        except:
                            pass
                        try:
                            te = evs_pos_r_dict[(n, 2)]
                            node_embeddings[m][te] += output[0][:, te_index, :].squeeze()  # (n,2)
                        except Exception:
                            pass
                    try:
                        se = evs_pos_r_dict[(n, 0)]
                        node_embeddings[m][se] += output[0][:, se_index, :].squeeze()  # (n,0)
                    except Exception:
                        pass
                else:
                    input_ids_tmp = input_ids_[n, :].unsqueeze(dim=0)
                    # judge whether should stop, if input_ids_tmp == [0,0,...,0], it means no triples in this sample
                    if torch.equal(input_ids_tmp, torch.zeros([1, max_triple_len]).type_as(input_ids_tmp)):
                        break
                    input_mask_tmp = input_mask_[n, :].unsqueeze(dim=0)
                    segment_ids_tmp = segment_ids_[n, :].unsqueeze(dim=0)
                    output = self.roberta(input_ids=input_ids_tmp,
                                          attention_mask=input_mask_tmp)

                    tmp = input_ids_tmp.squeeze().cpu().numpy().tolist()
                    #   获取分隔符</s>(2)的位置,共有2*事件数个</s> 第2n+1（n从0开始）个</s>的表示代表该事件的表示
                    split_index_list = [i for (i,v) in enumerate(tmp) if v == 2]
                    split_index_list.pop()
                    piece_num = int(segment_ids_tmp.max()) + 1
                    cur_ev = 0#表示当前是第几个事件
                    for i in range(piece_num):
                        try:
                            ev_index = evs_pos_r_dict[(n, i)]
                            #TODO: 表示当前有事件
                            #  	node_embeddings[m][rel]表示batch中第m个样本（一个图）的第rel个节点的向量表示
                            assert cur_ev != len(split_index_list)/2
                            node_embeddings[m][ev_index] += output[0][:, split_index_list[2*cur_ev], :].squeeze() # (n,1)
                            cur_ev += 1
                        except Exception:
                            pass
            node_embeddings[m] = torch.mul(node_embeddings[m], average_m)

        return batch_size, node_embeddings


    def forward(self, ev_pairs, input_ids, input_mask, segment_ids, evs_pos, graphs, label_ids, context_pos):
        '''caculate nodes representations by averaging'''
        batch_size, node_embeddings = self.get_output(input_ids, evs_pos, input_mask, segment_ids, context_pos)
        #------------------------------------------------------------------------------------------
        features, _ = self.layers.forward([node_embeddings, graphs])
        se_features = features[range(batch_size), ev_pairs[:, 0], :]
        te_features = features[range(batch_size), ev_pairs[:, 1], :]

        features_out = torch.cat([se_features, te_features], dim=-1)
        output = self.classifier(features_out)

        return se_features, te_features, output, label_ids


def train(args, train_dataset, model, tokenizer):
    mod_type = args.model_type
    tb_writer = SummaryWriter(os.path.join(args.log_dir, mod_type))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight','LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        #`model = model.to(torch.device('cuda'))
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    best_val_metrics = init_metric_dict()
    counter = -1
    exit_flag = False
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'ev_pairs': batch[0],
                      'input_ids': batch[1],
                      'input_mask': batch[2],
                      'segment_ids': batch[3],
                      'evs_pos': batch[4],
                      'graphs': batch[5],
                      'label_ids': batch[6],
                      'context_pos': batch[7]}
            #   se_features, te_features, output, label_ids, se_features_r, te_features_r, output_r, label_ids_r
            se_features, te_features, output, label_ids = model(**inputs)
            model = model.module if hasattr(model, 'module') else model
            loss = model.compute_metrics(se_features, te_features, output, label_ids)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    print('training p, r, f1')
                    preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                    pre_rec_f1(preds, label_ids.detach().cpu().numpy())
                    print('*********')
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        best_val_metrics, exit_flag, counter, results = \
                            evaluate(args=args,
                                     model=model,
                                     tokenizer=tokenizer,
                                     best_val_metrics=best_val_metrics,
                                     counter=counter,
                                     epoch=epoch,
                                     global_step=global_step)
                        #TODO:TEST
                        evaluate(args, model, tokenizer, type='test1')

                        if counter == 0 and args.save:
                            output_dir = os.path.join(args.output_dir, args.model_type)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))

                            tokenizer.save_pretrained(output_dir)
                            logging.info("Saving best model to %s", output_dir)

                        if exit_flag:
                            break
                        else:
                            for key, value in results.items():
                                tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if exit_flag:
            # jump from the outer loop
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args,
             model,
             tokenizer,
             best_val_metrics=None,
             counter=-1,
             epoch=-1,
             global_step=-1,
             type='eval'):

    eval_output_dir = os.path.join(args.output_dir, args.model_type)
    eval_dataset = load_and_cache_examples(args, type, tokenizer)
    results = {}

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logging.info("***** Running {} evaluation *****".format(type))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'ev_pairs': batch[0],
                      'input_ids': batch[1],
                      'input_mask': batch[2],
                      'segment_ids': batch[3],
                      'evs_pos': batch[4],
                      'graphs': batch[5],
                      'label_ids': batch[6],
                      'context_pos': batch[7]}

            se_features, te_features, cls_scores, label_ids = model(**inputs)
            loss = model.compute_metrics(se_features, te_features, cls_scores, label_ids)
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = cls_scores.detach().cpu().numpy()
            out_label_ids = inputs['label_ids'].detach().cpu().numpy()
        else:
            preds = np.append(preds, cls_scores.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['label_ids'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = pre_rec_f1(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        logging.info("***** {} results *****".format(type))
        writer.write("global_step: %s\n" % (global_step))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n")

    if type == 'eval':
        if has_improved(best_val_metrics, result):
            best_val_metrics = result
            counter = 0
        else:
            counter += 1
            if counter >= args.patience and epoch >= args.min_epochs:
                logging.warning("Early stopping, current epoch:{}".format(epoch))
                return best_val_metrics, True, counter, results
        return best_val_metrics, False, counter, results
    else:
        output_pred_file = os.path.join(eval_output_dir, "predict.txt")
        with open(output_pred_file, "a+") as pre_file:

            label_dic = ['-1', '0', '1']
            for pred_label in preds:
                pred_label = int(pred_label)
                pred_label_new = label_dic[pred_label]
                pre_file.write(pred_label_new)
                pre_file.write("\n")
        return results

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-log_dir", default='{}/HGAM/log/graph/'.format(root_pth), type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("-data_dir", default='{}/HGAM/data/'.format(root_pth), type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("-model_type", default='wo_all_subevent_8', type=str, help="Model type")
    parser.add_argument("-model_name_or_path", default='{}/HGAM/model/roberta_base/'.format(root_pth), type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("-output_dir", default='{}/HGAM/model/gen'.format(root_pth), type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("-config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("-tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("-cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    # gat newly added
    parser.add_argument("--num_layers", default=2, type=int, help='number of hidden layers in encoder')
    parser.add_argument("--n_heads", default=16, type=int, help='number of attention heads for graph attention networks, must be a divisor dim')
    parser.add_argument("--dim", default=768, type=int, help='dimension of graph reasoner')#512
    parser.add_argument("--directional", default=True, action='store_true', help="Use directed attention or not.")#False
    parser.add_argument("--dropout", default=0.12, type=float, help="The probability of drop out.")
    parser.add_argument("--layer_norm", default=False, action='store_true', help="If conducting layer normalization for the auxiliary graph branch.")
    parser.add_argument("--weight", default=0.1, type=float, help="The weight of the loss function's second term.")
    parser.add_argument("-do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("-do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("-evaluate_during_training", action='store_true', default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("-do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("-per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")  # 8
    parser.add_argument("-per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")  # 8
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("-learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("-weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")  # 0.0
    parser.add_argument("-adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("-max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("-num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")  # 5.0
    parser.add_argument("-max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("-warmup_steps", default=200, type=int,
                        help="Linear warmup over warmup_steps.")  # 0,200

    parser.add_argument('-logging_steps', type=int, default=50,
                        help="Log every X updates steps.")  # 50
    parser.add_argument("-no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('-overwrite_output_dir', action='store_true', default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('-overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('-seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('-fp16', action='store_true', default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('-fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("-local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('-server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('-server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('-save', type=bool, default=True, help='to save models with the best performance in evaluation')
    parser.add_argument('-min_epochs', type=int, default=5, help='do not early stop before min-epochs')#5
    parser.add_argument('-patience', type=int, default=3, help='patience for early stopping')#3

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaHeteGat, RobertaTokenizer
    '''type_vocab_size is needed to avoid multiple token_type_ids types beyond 2'''
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=3, finetuning_task=None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, unk_token='<unk>')
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                config=config, args=args)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logging.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, 'train', tokenizer)
        # print(model)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        save_final_path = os.path.join(args.output_dir, args.model_type)

        if not os.path.exists(save_final_path) and args.local_rank in [-1, 0]:
            os.makedirs(save_final_path)

        model = model_class.from_pretrained(save_final_path, config=config, args=args)
        tokenizer = tokenizer_class.from_pretrained(save_final_path)
        model.to(args.device)

    if args.do_eval and args.local_rank in [-1, 0]:

        model = model.module if hasattr(model, 'module') else model
        result = evaluate(args, model, tokenizer, type='test1')
        print(result)
        return result

main()