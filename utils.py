import re
import os
import json
import copy
import random
import torch
import pickle
import zipfile
import numpy as np
from math import ceil
from collections import Counter
from collections import defaultdict
import operator
from tqdm import tqdm
import torch.nn as nn
from parse_args import args

EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def index2word(word2id):
    return {i: w for w, i in word2id.items()}

def get_dataset_path(args):
    if args.dataset.startswith("MetaQA"):
        dataset_name = "MetaQA"
    
        if args.dataset.endswith("1H"):
            args.max_hop = 1
        
        elif args.dataset.endswith("2H"):
            args.max_hop = 2
        
        elif args.dataset.endswith("3H"):
            args.max_hop = 3
        
        entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/entity2id.txt".format(dataset_name)
        kb_relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/kb_relation2id.txt".format(dataset_name)
        wiki_relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/wiki_relation2id.txt".format(dataset_name)
        entity_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/entity_embeddings_ConvE.npy".format(dataset_name)
        relation_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/relation_embeddings_ConvE.npy".format(dataset_name)
        kb_triples_file = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/kb_triples_id.txt".format(dataset_name)
        wiki_triples_file = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/wiki_triples_id.txt".format(dataset_name)

        word2id_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/{}/{}/word2id.pkl").format(dataset_name, args.dataset)
        word_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/{}/{}/word_embeddings.npy").format(dataset_name, args.dataset)
        output_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/outputs/{}/").format(args.dataset)

        train_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/{}/qa_train_full_split.json".format(dataset_name, args.dataset)
        valid_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/{}/qa_dev_full_split.json".format(dataset_name, args.dataset)
        test_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/{}/qa_test_full_split.json".format(dataset_name, args.dataset)

    elif args.dataset.startswith("WebQSP"):
        dataset_name = "WebQSP"
        args.max_hop = 2

        entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/Freebase/entity2id_step2.txt".format(dataset_name)
        kb_relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/Freebase/kb_relation2id_step2.txt".format(dataset_name)
        wiki_relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/data/wiki_relation2id.txt".format(dataset_name)
        entity_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/data/entity_embeddings_ConvE.npy".format(dataset_name)
        relation_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/data/relation_embeddings_ConvE.npy".format(dataset_name)
        word2id_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/{}/data/word2id.pkl").format(dataset_name)
        word_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/{}/data/word_embeddings.npy").format(dataset_name)
        kb_triples_file = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/Freebase/kb_triples_id.txt".format(dataset_name)
        wiki_triples_file = os.path.abspath(os.path.join(os.getcwd(), ".."))+ "/datasets/{}/Freebase/wiki_triples_id.txt".format(dataset_name)

        output_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/outputs/{}/").format(args.dataset)

        train_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/data/WebQSP.train_step3_split.json".format(dataset_name)
        valid_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/data/WebQSP.test_step3_split.json".format(dataset_name)
        test_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/datasets/{}/data/WebQSP.test_step3_split.json".format(dataset_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return train_path, valid_path, test_path, entity2id_path, kb_relation2id_path, wiki_relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, kb_triples_file, wiki_triples_file, output_path

def read_vocab(vocab_file):
    d_item2id = {}

    with open(vocab_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            d_item2id[items[0]] = int(items[1])

    return d_item2id

def get_id_vocab(entity2id_path, kb_relation2id_path, wiki_relation2id_path):
    d_entity2id = read_vocab(entity2id_path)
    d_relation2id_kb = read_vocab(kb_relation2id_path)
    d_relation2id_wiki = read_vocab(wiki_relation2id_path)

    d_relation2id_wiki["DUMMY_RELATION"] = 0
    d_relation2id_wiki["NO_OP_RELATION"] = 2

    return d_entity2id, d_relation2id_kb, d_relation2id_wiki


def build_qa_vocab(train_file, valid_file, wiki_relation2id_path, word2id_output_file, min_freq):
    flag_words = ['<pad>', '<unk>']
    count = Counter()

    with open(train_file) as f:
        for i, line in enumerate(f):
            if i > 0 and i % 5000 == 0:
                print(i)
            qa_data = json.loads(line.strip())
            question_pattern = qa_data["question_pattern"]

            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)
    
    with open(valid_file) as f:
        for i, line in enumerate(f):
            if i > 0 and i % 5000 == 0:
                print(i)
            qa_data = json.loads(line.strip())
            question_pattern = qa_data["question_pattern"]

            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)
    
    with open(wiki_relation2id_path) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            wiki_relation = items[0].strip()
            words_relation = [word for word in wiki_relation.split(" ")]
            count.update(words_relation)

    count = {k: v for k, v in count.items()}
    count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    vocab = [w[0] for w in count if w[1] >= min_freq]
    vocab = flag_words + vocab
    word2id = {k: v for k, v in zip(vocab, range(0, len(vocab)))}
    print("word len: ", len(word2id))
    assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"

    with open(word2id_output_file, 'wb') as fw:
        pickle.dump(word2id, fw)

    return word2id

def initialize_word_embedding(word2id, glove_path, word_embedding_file):
    word_embeddings = np.random.uniform(-0.1, 0.1, (len(word2id), 300))
    seen_words = []

    gloves = zipfile.ZipFile(glove_path)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0].decode('utf-8')
                    embedding = splitline[1:]
                    if word in word2id and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        word_embeddings[word2id[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen_words.append(word)

    word_embeddings = word_embeddings.astype(np.float32)
    word_embeddings[0, :] = 0.
    print("pretrained vocab %s among %s" %(len(seen_words), len(word_embeddings)))
    unseen_words = set([k for k in word2id]) - set(seen_words)
    print("unseen words = ", len(unseen_words), unseen_words)
    np.save(word_embedding_file, word_embeddings)
    return word_embeddings

def token_to_id(token, token2id, flag_words = "<unk>"):
    return token2id[token] if token in token2id else token2id[flag_words]

def load_all_triples_from_txt(data_path, keep_ratio = 1.0):
    triples = []
    
    with open(data_path) as f:
        for line in f.readlines():
            s, r, o = line.strip().split("\t")
            s, r, o = int(s), int(r), int(o)
            triples.append((s, r, o))
    
    random.shuffle(triples)
    triples_len = len(triples)
    keep_num = ceil(keep_ratio * triples_len)

    triples = triples[:keep_num]

    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def get_adjacent(triples):
    triple_dict = defaultdict(defaultdict)

    for triple in triples:
        s_id, r_id, o_id = triple
        
        if r_id not in triple_dict[s_id]:
            triple_dict[s_id][r_id] = set()
        triple_dict[s_id][r_id].add(o_id)

    return triple_dict

def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l
    
def process_qa_file(qa_file, d_word2id, keep_ratio):
    l_data = []
    with open(qa_file, "r") as f:
        for i, line in enumerate(f):
            qa_data = json.loads(line.strip())

            qid = qa_data["id"]
            question_pattern = qa_data["question_pattern"]
            topic_entity_id = qa_data["topic_entity_id"]
            answer_entities_id = qa_data["answer_entities_id"]

            question_pattern_id = [token_to_id(word, d_word2id) for word in question_pattern.strip().split(" ")]
            l_data.append([qid, question_pattern_id, topic_entity_id, answer_entities_id])
    
    random.shuffle(l_data)
    data_len = len(l_data)
    keep_num = ceil(keep_ratio * data_len)

    l_data = l_data[:keep_num]
            
    return l_data

def getEntityActions(subject, triple_dict, NO_OP_RELATION = 2):
    action_space = []

    if subject in triple_dict:
        for relation in triple_dict[subject]:
            objects = triple_dict[subject][relation]
            for obj in objects: 
                action_space.append((relation, obj))
        
    action_space.insert(0, (NO_OP_RELATION, subject))

    return action_space

def vectorize_action_space(action_space_list, action_space_size, DUMMY_ENTITY = 0, DUMMY_RELATION = 0):
    bucket_size = len(action_space_list)
    r_space = torch.zeros(bucket_size, action_space_size) + DUMMY_ENTITY 
    e_space = torch.zeros(bucket_size, action_space_size) + DUMMY_RELATION
    r_space = r_space.long()
    e_space = e_space.long()
    action_mask = torch.zeros(bucket_size, action_space_size)
    for i, action_space in enumerate(action_space_list):
        for j, (r, e) in enumerate(action_space):
            r_space[i, j] = r
            e_space[i, j] = e
            action_mask[i, j] = 1
    action_mask = action_mask.long()
    return (r_space, e_space), action_mask
    
def initialize_action_space(num_entities, triple_dict, bucket_interval):
    d_action_space_buckets = {}
    d_action_space_buckets_discrete = defaultdict(list)
    d_entity2bucketid = torch.zeros(num_entities, 2).long()
    num_facts_saved_in_action_table = 0

    for e1 in range(num_entities):
        action_space = getEntityActions(e1, triple_dict)
        key = int(len(action_space) / bucket_interval) + 1 
        d_entity2bucketid[e1, 0] = key
        d_entity2bucketid[e1, 1] = len(d_action_space_buckets_discrete[key])
        d_action_space_buckets_discrete[key].append(action_space)
        num_facts_saved_in_action_table += len(action_space)
    
    for key in d_action_space_buckets_discrete:
        d_action_space_buckets[key] = vectorize_action_space(
            d_action_space_buckets_discrete[key], key * bucket_interval)
    
    return d_entity2bucketid, d_action_space_buckets

def kb_relation_global2local(r_tensor):
    l_r_tensor = r_tensor.cpu().numpy().tolist()
    
    l_all_relations_id = []
    for l_r in l_r_tensor:
        l_all_relations_id += l_r
    l_all_relations_id = list(set(l_all_relations_id))
    
    d_batch_global2local_id = {}
    
    for i, global_id in enumerate(l_all_relations_id): 
        if global_id not in d_batch_global2local_id:
            d_batch_global2local_id[global_id] = i
        
    r_tensor_localId = list(map(
            lambda x: list(map(lambda y: d_batch_global2local_id[y], x)),
            l_r_tensor
        ))

    r_tensor_localId = torch.LongTensor(r_tensor_localId).cuda()
    r_tensor_globalId = torch.LongTensor(l_all_relations_id).cuda()

    return r_tensor_globalId, r_tensor_localId

def wiki_relationid_to_name(r_tensor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id):
    r_tensor = r_tensor.cpu().numpy().tolist()
    l_r_text = list(map(
            lambda x: list(map(lambda y: d_relationid2text_wiki[y], x)),
            r_tensor
        ))
    
    l_all_relations = []
    for l_r in l_r_text:
        l_all_relations += l_r
    l_all_relations = list(set(l_all_relations))

    d_batch_local_relation2id = {} 
    d_batch_local_id2relation = {}

    l_sorted_relations = sorted(l_all_relations, key=lambda sample: len(sample.split()), reverse=True)
    l_sorted_seq_lengths = [len(i.split()) for i in l_sorted_relations]
    longest_sample = l_sorted_seq_lengths[0]

    for i, r_name in enumerate(l_sorted_relations):
        if r_name not in d_batch_local_relation2id:
            d_batch_local_relation2id[r_name] = i

        r_local_id = d_batch_local_relation2id[r_name]
        r_global_id = d_relation2id_wiki[r_name]

        if r_local_id not in d_batch_local_id2relation:
            d_batch_local_id2relation[r_local_id] = r_global_id
    
    r_tensor_localId = list(map(
            lambda x: list(map(lambda y: d_batch_local_relation2id[y], x)),
            l_r_text
        ))

    r_tensor_localId = torch.LongTensor(r_tensor_localId).cuda()

    l_sorted_relations_id = []
    for i, r_name in enumerate(l_sorted_relations):
        relation_id = [token_to_id(word, d_word2id) for word in r_name.strip().split()]
        l_sorted_relations_id.append(relation_id)
    
    batch_relations_tokenId = torch.zeros(len(l_sorted_relations), longest_sample, dtype=torch.long)

    for x in range(len(l_sorted_relations)):
        relation_id = torch.LongTensor(l_sorted_relations_id[x])
        seq_len = len(relation_id)
        batch_relations_tokenId[x].narrow(0,0,seq_len).copy_(relation_id)

    return l_sorted_relations, torch.LongTensor(l_sorted_seq_lengths).cuda(), batch_relations_tokenId.cuda(), r_tensor_localId
