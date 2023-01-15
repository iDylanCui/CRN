import torch
from parse_args import args
from torch.utils.data import Dataset, DataLoader
from utils import process_qa_file

class Dataset_Model(Dataset):
    def __init__(self, data_path, entity_num, d_word2id):
        self.data_path = data_path
        self.entity_num = entity_num
        self.d_word2id = d_word2id

        self.data = process_qa_file(self.data_path, self.d_word2id, args.qa_data_percentage)
    
    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        one_hot = torch.LongTensor(self.entity_num)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        q_id = line[0]
        question_pattern_id = line[1]
        topic_entity_id = line[2]
        answer_entities_id = line[3]
        answer_entities_onehot = self.toOneHot(answer_entities_id)

        return [q_id, torch.LongTensor(question_pattern_id), topic_entity_id, answer_entities_onehot]


def _collate_fn(batch):
    sorted_seq = sorted(batch, key=lambda sample: len(sample[1]), reverse=True)
    sorted_seq_lengths = [len(i[1]) for i in sorted_seq]
    
    if args.use_ecm_tokens_internal_memory:
        longest_sample = args.max_question_len_global
    else:
        longest_sample = sorted_seq_lengths[0]

    minibatch_size = len(batch)
    qids = []
    input_lengths = []
    p_head_global = []
    p_tail_global = []
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)

    for x in range(minibatch_size):
        qid = sorted_seq[x][0]
        qids.append(qid)

        question_pattern_id = sorted_seq[x][1]
        seq_len = len(question_pattern_id)

        input_lengths.append(seq_len)
        inputs[x].narrow(0,0,seq_len).copy_(question_pattern_id)

        topic_entity_id = sorted_seq[x][2]
        tail_onehot = sorted_seq[x][3]

        p_head_global.append(topic_entity_id)
        p_tail_global.append(tail_onehot)

    return inputs, torch.LongTensor(input_lengths), torch.LongTensor(p_head_global), torch.stack(p_tail_global), qids

class DataLoader_Model(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_Model, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn