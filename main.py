import os
import torch
import pickle
import itertools
import numpy as np
from parse_args import args
from utils import set_seed, get_dataset_path, get_id_vocab, build_qa_vocab, initialize_word_embedding, index2word, flatten
from training_strategy import pretrain_reasoner, pretrain_extractor, collaborate_train, inference

def get_hyperparameter_range():
    hp_KEQA = ["seed", "extractor_topK", "keep_wiki_path_num"]
    
    hp_KEQA_range = {
        "seed": [1002], 
        "extractor_topK": [10],
        "keep_wiki_path_num": [3]
    }

    return hp_KEQA, hp_KEQA_range

def grid_search(train_path, valid_path, kb_triples_file, wiki_triples_file, output_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda):
    hp_model, hp_model_range = get_hyperparameter_range()
    grid = hp_model_range[hp_model[0]]
    for hp in hp_model[1:]:
        grid = itertools.product(grid, hp_model_range[hp])
    
    grid_results = {}
    grid = list(grid)

    out_log_path = os.path.join(output_path, "log.txt")
    if not os.path.exists(out_log_path):
        with open(out_log_path, "w")  as ft:
            ft.write("** Grid Search **\n")
            ft.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
            ft.write('Signature\tbest_epoch_valid\tHits@1_valid\n')
    
    for i, grid_entry in enumerate(grid):
        if type(grid_entry) is not list:
            grid_entry = [grid_entry]
        
        grid_entry = flatten(grid_entry)
        print('* Hyperparameter Set {} = {}'.format(i, grid_entry)) 

        signature = ''
        for j in range(len(grid_entry)):
            hp = hp_model[j]
            value = grid_entry[j]
            if hp == "loss_with_beam_reasoner":
                setattr(args, hp, bool(value))
            else:
                setattr(args, hp, int(value))
            
            signature += '{}_{} '.format(hp, value)
        
        signature = signature.strip()

        set_seed(args.seed)

        grid_entry_path = os.path.join(output_path, signature)

        grid_pretrain_ckpt_path = grid_entry_path

        if not os.path.exists(grid_entry_path):
            os.makedirs(grid_entry_path)
        
        best_epoch, best_dev_metrics = run_training(train_path, valid_path, kb_triples_file, wiki_triples_file, grid_entry_path, grid_pretrain_ckpt_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda)

        with open(out_log_path, "a") as f:
            f.write(signature + "\t" + str(best_epoch) + "\t" + str(round(best_dev_metrics, 4)) + "\n")
        
        grid_results[signature] = best_dev_metrics
    

def run_training(train_path, valid_path, kb_triples_file, wiki_triples_file, output_path, pretrain_ckpt_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda):

    pretrain_reasoner(train_path, valid_path, kb_triples_file, wiki_triples_file, pretrain_ckpt_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda)

    pretrain_extractor(train_path, valid_path, kb_triples_file, wiki_triples_file, pretrain_ckpt_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda)

    best_epoch, best_dev_metrics = collaborate_train(train_path, valid_path, kb_triples_file, wiki_triples_file, output_path, pretrain_ckpt_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda)

    return best_epoch, best_dev_metrics

def run_inference(test_path, kb_triples_file, wiki_triples_file, output_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda):
    args.pretrain_reasoner = False
    args.pretrain_extractor = False
    args.collaborate_training = True
    
    result3 = inference(test_path, kb_triples_file, wiki_triples_file, output_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    qa_data_percentages = [1.0]
    kb_triples_percentages = [1.0]

    train_path, valid_path, test_path, entity2id_path, kb_relation2id_path, wiki_relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, kb_triples_file, wiki_triples_file, output_path = get_dataset_path(args)

    d_entity2id, d_relation2id_kb, d_relation2id_wiki = get_id_vocab(entity2id_path, kb_relation2id_path, wiki_relation2id_path)

    d_relationid2text_wiki = index2word(d_relation2id_wiki)

    if not os.path.isfile(word2id_path):
        d_word2id = build_qa_vocab(train_path, valid_path, wiki_relation2id_path, word2id_path, args.min_freq)
    else:
        d_word2id = pickle.load(open(word2id_path, 'rb'))

    if not os.path.isfile(word_embedding_path):
        glove_path = os.path.abspath(os.path.join(os.getcwd(), "..") + "/datasets/glove.840B.300d.zip")
        word_embeddings = initialize_word_embedding(d_word2id, glove_path, word_embedding_path)
    else:
        word_embeddings = np.load(word_embedding_path)
    word_embeddings = torch.from_numpy(word_embeddings)

    if os.path.isfile(entity_embedding_path):
        entity_embeddings = np.load(entity_embedding_path)
        entity_embeddings = torch.from_numpy(entity_embeddings)
    
    if os.path.isfile(relation_embedding_path):
        relation_embeddings = np.load(relation_embedding_path)
        relation_embeddings = torch.from_numpy(relation_embeddings)

    for qa_data_percentage in qa_data_percentages:
        args.qa_data_percentage = qa_data_percentage

        for kb_triples_percentage in kb_triples_percentages:
            args.kb_triples_percentage = kb_triples_percentage

            print("data percentage = {}, triple percentage = {}.".format(args.qa_data_percentage, args.kb_triples_percentage))

            out_path = os.path.join(output_path, "data_{}_kb_{}".format(args.qa_data_percentage, args.kb_triples_percentage))

            if not os.path.exists(out_path):
                os.makedirs(out_path)

            if args.grid_search:
                grid_search(train_path, valid_path, kb_triples_file, wiki_triples_file, out_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda)