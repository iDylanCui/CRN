import os
import time
import torch
import linecache
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict
from rollout import rollout_beam
from parse_args import args
from torch.nn.utils import clip_grad_norm_
from environment import Environment
from Reasoner import Reasoner_Network
from Extractor import Extractor_Network
from dataloader import Dataset_Model, DataLoader_Model
from utils import load_all_triples_from_txt, get_adjacent, initialize_action_space, index2word


def loading_data(train_path, valid_path, kb_triples_file, wiki_triples_file, d_entity2id, d_word2id):
    print("Loading triples and adjacent ...")
    if args.pretrain_reasoner or args.collaborate_training:
        kb_triples = load_all_triples_from_txt(kb_triples_file, args.kb_triples_percentage)
    else:
        kb_triples = []
    
    if args.pretrain_extractor or args.collaborate_training:
        wiki_triples = load_all_triples_from_txt(wiki_triples_file, args.wiki_triples_percentage)
    else:
        wiki_triples = []

    kb_triple_dict = get_adjacent(kb_triples) 
    wiki_triple_dict = get_adjacent(wiki_triples)

    del kb_triples
    del wiki_triples

    entity2bucketid_kb, d_action_space_buckets_kb = initialize_action_space(len(d_entity2id), kb_triple_dict, args.bucket_interval)
    entity2bucketid_wiki, d_action_space_buckets_wiki = initialize_action_space(len(d_entity2id), wiki_triple_dict, args.bucket_interval)

    del kb_triple_dict
    del wiki_triple_dict

    if args.grid_search:
        print("Loading train_dataset and train_dataloader ...")
        train_dataset = Dataset_Model(train_path, len(d_entity2id), d_word2id)
        train_dataloader = DataLoader_Model(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        
        print("Loading valid_dataset and valid_dataloader ...")
        valid_dataset = Dataset_Model(valid_path, len(d_entity2id), d_word2id)
        valid_dataloader = DataLoader_Model(valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

        return train_dataloader, valid_dataloader, entity2bucketid_kb, d_action_space_buckets_kb, entity2bucketid_wiki, d_action_space_buckets_wiki
    
    elif args.eval:
        print("Loading test_dataset and test_dataloader ...")
        test_dataset = Dataset_Model(valid_path, len(d_entity2id), d_word2id)
        test_dataloader = DataLoader_Model(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

        return test_dataloader, entity2bucketid_kb, d_action_space_buckets_kb, entity2bucketid_wiki, d_action_space_buckets_wiki

def pretrain_reasoner(train_path, valid_path, kb_triples_file, wiki_triples_file, output_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda):
    reasoner_pre_ckpt_path = os.path.join(output_path, "reasoner_pretrain.ckpt")

    if os.path.exists(reasoner_pre_ckpt_path):
        return

    print("pretrain_reasoner:")
    args.pretrain_reasoner = True
    args.pretrain_extractor = False
    args.collaborate_training = False

    train_dataloader, valid_dataloader, entity2bucketid_kb, d_action_space_buckets_kb, entity2bucketid_wiki, d_action_space_buckets_wiki = loading_data(train_path, valid_path, kb_triples_file, wiki_triples_file, d_entity2id, d_word2id)

    env_train = Environment(args, len(d_relation2id_kb))
    env_valid = Environment(args, len(d_relation2id_kb))

    reasoner = Reasoner_Network(args, len(d_relation2id_kb), word_embeddings, entity_embeddings, relation_embeddings, entity2bucketid_kb, d_action_space_buckets_kb).cuda()
    extractor = Extractor_Network(args, word_embeddings, entity2bucketid_wiki, d_action_space_buckets_wiki).cuda()

    for name, param in reasoner.named_parameters():
        if name.split(".")[0].endswith("wiki"):
            param.requires_grad = False

    optimizer_reasoner = optim.Adam(filter(lambda p: p.requires_grad, reasoner.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)

    best_dev_metrics = -float("inf")
    iters_not_improved = 0
    best_reasoner_model = reasoner.state_dict()

    l_epochs_valid = []
    l_hits1_valid =[]

    start = time.time()
    for epoch_id in range(0, args.total_epoch):
        reasoner.train()
        extractor.eval()
        total_reasoner_loss = 0

        train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")

        for i_batch, batch_data in enumerate(train_loader):
            batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid = batch_data

            if use_cuda:
                batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_global.cuda()
            
            batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid)
            
            env_train.reset(batch_data, reasoner)

            rollout_beam(env_train, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "train")

            reasoner_loss, _ = env_train.calculate_beam_loss()
            optimizer_reasoner.zero_grad()
            reasoner_loss.backward()
            if args.grad_norm > 0:
                clip_grad_norm_(reasoner.parameters(), args.grad_norm)
            
            optimizer_reasoner.step()

            total_reasoner_loss += reasoner_loss.item()

            if use_cuda:
                torch.cuda.empty_cache()
            
            linecache.clearcache()
        
        print("epoch = {}, reasoner_loss = {}.".format(epoch_id, total_reasoner_loss))

        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            reasoner.eval()
            extractor.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches")
            total_hits1 = 0.0
            total_num = 0.0

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid = batch_data

                    if use_cuda:
                        batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_global.cuda()
                    
                    batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid)

                    env_valid.reset(batch_data, reasoner)

                    rollout_beam(env_valid, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "inference")

                    hits1_item = env_valid.inference_hits1()
                    total_hits1 += hits1_item
                    total_num += env_valid.batch_size


                answer_hits_1 = 1.0 * total_hits1 / total_num

                l_epochs_valid.append(epoch_id + 1)

                if answer_hits_1 > best_dev_metrics:
                    best_dev_metrics = answer_hits_1
                    best_reasoner_model = reasoner.state_dict()
                    torch.save(best_reasoner_model, reasoner_pre_ckpt_path)
                    iters_not_improved = 0
                    print('Epoch {}: best vaild Hits@1 = {}.'.format(epoch_id, best_dev_metrics))
                
                elif answer_hits_1 < best_dev_metrics and iters_not_improved * args.num_wait_epochs < args.early_stop_patience:
                    iters_not_improved += 1
                    print("Vaild Hits@1 decreases to %f from %f, %d more epoch to check"%(answer_hits_1, best_dev_metrics, args.early_stop_patience - iters_not_improved * args.num_wait_epochs))
                
                elif iters_not_improved * args.num_wait_epochs == args.early_stop_patience:
                    end = time.time()
                    print("Model has exceed patience. Saving best model and exiting. Using {} seconds.".format(round(end - start, 2)))
                    break
        

def pretrain_extractor(train_path, valid_path, kb_triples_file, wiki_triples_file, output_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda):
    reasoner_pre_ckpt_path = os.path.join(output_path, "reasoner_pretrain.ckpt")
    extractor_pre_ckpt_path = os.path.join(output_path, "extractor_pretrain.ckpt")
    reasoner_pre_ckpt_path_2 = os.path.join(output_path, "reasoner_pretrain_ext.ckpt")

    if os.path.exists(extractor_pre_ckpt_path):
        return

    print("pretrain_extractor:")
    args.pretrain_reasoner = False
    args.pretrain_extractor = True
    args.collaborate_training = False

    train_dataloader, valid_dataloader, entity2bucketid_kb, d_action_space_buckets_kb, entity2bucketid_wiki, d_action_space_buckets_wiki = loading_data(train_path, valid_path, kb_triples_file, wiki_triples_file, d_entity2id, d_word2id)

    env_train = Environment(args, len(d_relation2id_kb))
    env_valid = Environment(args, len(d_relation2id_kb))

    reasoner = Reasoner_Network(args, len(d_relation2id_kb), word_embeddings, entity_embeddings, relation_embeddings, entity2bucketid_kb, d_action_space_buckets_kb).cuda()
    extractor = Extractor_Network(args, word_embeddings, entity2bucketid_wiki, d_action_space_buckets_wiki).cuda()

    reasoner.load(reasoner_pre_ckpt_path)

    for name, param in reasoner.named_parameters():
        if not name.split(".")[0].endswith("wiki"):
            param.requires_grad = False

    optimizer_reasoner = optim.Adam(filter(lambda p: p.requires_grad, reasoner.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)
    optimizer_extractor = optim.Adam(filter(lambda p: p.requires_grad, extractor.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)

    best_dev_metrics = -float("inf")
    iters_not_improved = 0
    best_reasoner_model = reasoner.state_dict()
    best_extractor_model = extractor.state_dict()

    l_epochs_valid = []
    l_hits1_valid =[]
    
    
    start = time.time()
    for epoch_id in range(0, args.total_epoch):
        extractor.train()
        total_reasoner_loss = 0
        total_extractor_loss = 0

        train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")

        for i_batch, batch_data in enumerate(train_loader):
            batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid = batch_data

            if use_cuda:
                batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_global.cuda()
            
            batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid)
            
            env_train.reset(batch_data, reasoner)
            rollout_beam(env_train, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "train")

            reasoner_loss, extractor_loss = env_train.calculate_beam_loss()

            optimizer_reasoner.zero_grad()
            optimizer_extractor.zero_grad()
            reasoner_loss.backward()
            extractor_loss.backward()
            if args.grad_norm > 0:
                clip_grad_norm_(reasoner.parameters(), args.grad_norm)
                clip_grad_norm_(extractor.parameters(), args.grad_norm)
            
            optimizer_reasoner.step()
            optimizer_extractor.step()

            total_reasoner_loss += reasoner_loss.item()
            total_extractor_loss += extractor_loss.item()

            if use_cuda:
                torch.cuda.empty_cache()
            
            linecache.clearcache()
        
        print("epoch = {}, reasoner_loss = {}, extractor_loss = {}.".format(epoch_id, total_reasoner_loss, total_extractor_loss))
    
        # Check dev set performance
        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            reasoner.eval()
            extractor.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches")
            total_hits1 = 0.0
            total_num = 0.0

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid = batch_data

                    if use_cuda:
                        batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_global.cuda()
                    
                    batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid)

                    env_valid.reset(batch_data, reasoner)

                    rollout_beam(env_valid, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "inference")

                    hits1_item = env_valid.inference_hits1()
                    total_hits1 += hits1_item
                    total_num += env_valid.batch_size


                answer_hits_1 = 1.0 * total_hits1 / total_num

                l_epochs_valid.append(epoch_id + 1)
                l_hits1_valid.append(round(answer_hits_1, 4))

                if answer_hits_1 > best_dev_metrics:
                    best_dev_metrics = answer_hits_1
                    best_reasoner_model = reasoner.state_dict()
                    best_extractor_model = extractor.state_dict()
                    torch.save(best_reasoner_model, reasoner_pre_ckpt_path_2)
                    torch.save(best_extractor_model, extractor_pre_ckpt_path)
                    iters_not_improved = 0
                    print('Epoch {}: best vaild Hits@1 = {}.'.format(epoch_id, best_dev_metrics))
                
                elif answer_hits_1 < best_dev_metrics and iters_not_improved * args.num_wait_epochs < args.early_stop_patience:
                    iters_not_improved += 1
                    print("Vaild Hits@1 decreases to %f from %f, %d more epoch to check"%(answer_hits_1, best_dev_metrics, args.early_stop_patience - iters_not_improved * args.num_wait_epochs))
                
                elif iters_not_improved * args.num_wait_epochs == args.early_stop_patience:
                    end = time.time()
                    print("Model has exceed patience. Saving best model and exiting. Using {} seconds.".format(round(end - start, 2)))
                    break


def collaborate_train(train_path, valid_path, kb_triples_file, wiki_triples_file, output_path, pretrain_ckpt_path, d_entity2id, d_relation2id_kb, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, word_embeddings, entity_embeddings, relation_embeddings, use_cuda):
    extractor_pre_ckpt_path = os.path.join(pretrain_ckpt_path, "extractor_pretrain.ckpt")
    reasoner_pre_ckpt_path = os.path.join(pretrain_ckpt_path, "reasoner_pretrain_ext.ckpt")
    reasoner_co_ckpt_path = os.path.join(output_path, "reasoner_collaborate.ckpt")
    extractor_co_ckpt_path = os.path.join(output_path, "extractor_collaborate.ckpt")

    if os.path.exists(reasoner_co_ckpt_path) and os.path.exists(extractor_co_ckpt_path):
        return 0, -1

    print("collaborate_train:")
    args.pretrain_reasoner = False
    args.pretrain_extractor = False
    args.collaborate_training = True

    train_dataloader, valid_dataloader, entity2bucketid_kb, d_action_space_buckets_kb, entity2bucketid_wiki, d_action_space_buckets_wiki = loading_data(train_path, valid_path, kb_triples_file, wiki_triples_file, d_entity2id, d_word2id)

    env_train = Environment(args, len(d_relation2id_kb))
    env_valid = Environment(args, len(d_relation2id_kb))

    reasoner = Reasoner_Network(args, len(d_relation2id_kb), word_embeddings, entity_embeddings, relation_embeddings, entity2bucketid_kb, d_action_space_buckets_kb).cuda()
    extractor = Extractor_Network(args, word_embeddings, entity2bucketid_wiki, d_action_space_buckets_wiki).cuda()

    reasoner.load(reasoner_pre_ckpt_path)
    extractor.load(extractor_pre_ckpt_path)

    optimizer_reasoner = optim.Adam(filter(lambda p: p.requires_grad, reasoner.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)
    optimizer_extractor = optim.Adam(filter(lambda p: p.requires_grad, extractor.parameters()), lr = args.learning_rate, weight_decay = args.weight_decay)

    best_dev_metrics = -float("inf")
    iters_not_improved = 0
    best_epoch = -1
    best_reasoner_model = reasoner.state_dict()
    best_extractor_model = extractor.state_dict()

    l_epochs_valid = []
    l_hits1_valid =[]

    start = time.time()
    for epoch_id in range(0, args.total_epoch):
        reasoner.train()
        extractor.train()
        total_reasoner_loss = 0
        total_extractor_loss = 0

        train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit = "batches")

        for i_batch, batch_data in enumerate(train_loader):
            batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid = batch_data

            if use_cuda:
                batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_global.cuda()
            
            batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid)
            
            env_train.reset(batch_data, reasoner)

            rollout_beam(env_train, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "train")

            reasoner_loss, extractor_loss = env_train.calculate_beam_loss()

            optimizer_reasoner.zero_grad()
            optimizer_extractor.zero_grad()
            reasoner_loss.backward()
            extractor_loss.backward()
            if args.grad_norm > 0:
                clip_grad_norm_(reasoner.parameters(), args.grad_norm)
                clip_grad_norm_(extractor.parameters(), args.grad_norm)
            
            optimizer_reasoner.step()
            optimizer_extractor.step()

            total_reasoner_loss += reasoner_loss.item()
            total_extractor_loss += extractor_loss.item()

            if use_cuda:
                torch.cuda.empty_cache()
            
            linecache.clearcache()
        
        print("epoch = {}, reasoner_loss = {}, extractor_loss = {}.".format(epoch_id, total_reasoner_loss, total_extractor_loss))
    
        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            reasoner.eval()
            extractor.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches")
            total_hits1 = 0.0
            total_num = 0.0

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid = batch_data

                    if use_cuda:
                        batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global = batch_question.cuda(), batch_question_seq_lengths.cuda(), batch_topic_ent_global.cuda(), batch_answers_global.cuda()
                    
                    batch_data = (batch_question, batch_question_seq_lengths, batch_topic_ent_global, batch_answers_global, batch_qid)

                    env_valid.reset(batch_data, reasoner)

                    rollout_beam(env_valid, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "inference")

                    hits1_item = env_valid.inference_hits1()
                    total_hits1 += hits1_item
                    total_num += env_valid.batch_size


                answer_hits_1 = 1.0 * total_hits1 / total_num

                l_epochs_valid.append(epoch_id + 1)
                l_hits1_valid.append(round(answer_hits_1, 4))

                if answer_hits_1 > best_dev_metrics:
                    best_dev_metrics = answer_hits_1
                    best_epoch = epoch_id
                    best_reasoner_model = reasoner.state_dict()
                    best_extractor_model = extractor.state_dict()
                    torch.save(best_reasoner_model, reasoner_co_ckpt_path)
                    torch.save(best_extractor_model, extractor_co_ckpt_path)
                    iters_not_improved = 0
                    print('Epoch {}: best vaild Hits@1 = {}.'.format(epoch_id, best_dev_metrics))
                
                elif answer_hits_1 < best_dev_metrics and iters_not_improved * args.num_wait_epochs < args.early_stop_patience:
                    iters_not_improved += 1
                    print("Vaild Hits@1 decreases to %f from %f, %d more epoch to check"%(answer_hits_1, best_dev_metrics, args.early_stop_patience - iters_not_improved * args.num_wait_epochs))
                
                elif iters_not_improved * args.num_wait_epochs == args.early_stop_patience:
                    end = time.time()
                    print("Model has exceed patience. Saving best model and exiting. Using {} seconds.".format(round(end - start, 2)))
                    break

    return best_epoch, best_dev_metrics