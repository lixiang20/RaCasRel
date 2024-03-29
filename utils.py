#! -*- coding:utf-8 -*-
import keras.backend as K
from keras_bert import Tokenizer
import numpy as np
import codecs
from tqdm import tqdm
import json
import unicodedata

# from data_loader import seq_padding

BERT_MAX_LEN = 512

class HBTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)        # 单词按形式分词
            tokens.append('[unused1]')
        return tokens

def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)

def seq_padding(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])  # np.concatenate：默认最外层

def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)

def extract_items(subject_model, object_model, tokenizer, text_in, id2rel,id2rel_mine, h_bar=0.5, t_bar=0.5):
    tokens = tokenizer.tokenize(text_in)
    token_ids, segment_ids = tokenizer.encode(first=text_in) 
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
    
    ###################################### 
    rel_tokens = []
    segment_tokens = []
    for i in range(len(id2rel_mine)):
        # rel_text = ' '.join(id2rel_mine[i].split('_')).lower()
        rel_text = ' '.join(id2rel_mine[i].split('_'))
        rel_token = ' '.join(tokenizer.tokenize(rel_text)[1:-1:2])

        token_id, segment_id = tokenizer.encode(first=rel_token)
        rel_tokens.append(token_id)
        segment_tokens.append(segment_id)

    rel_tokens = seq_padding(rel_tokens)
    segment_tokens = seq_padding(segment_tokens)

    rel_tokens,segment_tokens=np.array([rel_tokens]),np.array([segment_tokens])

    ###################################### 


    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:,:BERT_MAX_LEN]    
        segment_ids = segment_ids[:,:BERT_MAX_LEN] 
    sub_heads_logits, sub_tails_logits = subject_model.predict([token_ids, segment_ids])
    sub_heads, sub_tails = np.where(sub_heads_logits[0] > h_bar)[0], np.where(sub_tails_logits[0] > t_bar)[0]
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]
            subject = tokens[sub_head: sub_tail]
            subjects.append((subject, sub_head, sub_tail)) 
    if subjects:
        triple_list = []
        token_ids = np.repeat(token_ids, len(subjects), 0) 
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        rel_tokens=np.repeat(rel_tokens,len(subjects),0)  
        segment_tokens=np.repeat(segment_tokens,len(subjects),0)  
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
        obj_heads_logits, obj_tails_logits = object_model.predict([rel_tokens,segment_tokens,token_ids, segment_ids, sub_heads, sub_tails])
        for i, subject in enumerate(subjects):
            sub = subject[0]
            sub = ''.join([i.lstrip("##") for i in sub])
            sub = ' '.join(sub.split('[unused1]'))
            obj_heads, obj_tails = np.where(obj_heads_logits[i] > h_bar), np.where(obj_tails_logits[i] > t_bar)
            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2rel[rel_head]
                        obj = tokens[obj_head: obj_tail]
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]'))
                        triple_list.append((sub, rel, obj))
                        break
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []

def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold

def metric(subject_model, object_model, eval_data, id2rel, id2rel_mine, tokenizer, exact_match=False, output_path=None, elements_match=0):
    if output_path:
        F = open(output_path, 'w',encoding='utf-8')
    orders = ['subject', 'relation', 'object'] 
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    for line in tqdm(iter(eval_data)):
        Pred_triples = set(extract_items(subject_model, object_model, tokenizer, line['text'], id2rel,id2rel_mine))
        Gold_triples = set(line['triple_list'])

        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (Pred_triples, Gold_triples)        

        if elements_match==1:
            Pred_triples_eval = {(i[0], i[1]) for i in Pred_triples_eval}
            Gold_triples_eval = {(i[0], i[1]) for i in Gold_triples_eval}
        elif elements_match==2:
            Pred_triples_eval = {(i[0], i[2]) for i in Pred_triples_eval}
            Gold_triples_eval = {(i[0], i[2]) for i in Gold_triples_eval}
        elif elements_match==3:
            Pred_triples_eval = {(i[1], i[2]) for i in Pred_triples_eval}
            Gold_triples_eval = {(i[1], i[2]) for i in Gold_triples_eval}

        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print('correct_num:{}\npredict_num:{}\ngold_num:{}'.format(correct_num,predict_num,gold_num))
    return precision, recall, f1_score
