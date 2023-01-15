import torch
import torch.nn as nn
from utils import safe_log
from parse_args import args
    
def pad_and_cat1d(a, padding_value, inv_offset, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()[inv_offset]

def pad_and_cat2d(a, padding_value, inv_offset, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad2d((0, 0, 0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()[inv_offset]

def adaptive_sampling_actions(topK, keep_wiki_path_num, log_reasoner_action_dist_sum, batch_buckets_is_wiki_relation_in_path):
    if keep_wiki_path_num > topK:
        keep_wiki_path_num = topK
    
    batch_size, action_num = log_reasoner_action_dist_sum.shape
    l_log_action_prob = []
    l_action_ind = []

    for i in range(batch_size):
        log_reasoner_action_dist_i = log_reasoner_action_dist_sum[i]
        buckets_is_wiki_relation_in_path_i = batch_buckets_is_wiki_relation_in_path[i]

        wiki_path_num_i = torch.sum(buckets_is_wiki_relation_in_path_i, dim =0, keepdim=False).item()

        real_keep_wiki_path_num = keep_wiki_path_num if wiki_path_num_i > keep_wiki_path_num else wiki_path_num_i

        if real_keep_wiki_path_num > 0:
            log_reasoner_action_dist_i_mask_1 = log_reasoner_action_dist_i.masked_fill((1 - buckets_is_wiki_relation_in_path_i.int()).bool(), float('-inf'))
            log_reasoner_prob_wiki, action_ind_wiki = torch.topk(log_reasoner_action_dist_i_mask_1, real_keep_wiki_path_num)

            kb_select_num = topK - real_keep_wiki_path_num
            is_selected_wiki_index = torch.zeros(action_num).cuda().bool()
            is_selected_wiki_index[action_ind_wiki] = True

            log_reasoner_action_dist_i_mask_2 = log_reasoner_action_dist_i.masked_fill(is_selected_wiki_index, float('-inf'))

            log_reasoner_prob_kb, action_ind_kb = torch.topk(log_reasoner_action_dist_i_mask_2, kb_select_num)

            log_reasoner_prob_merge = torch.cat([log_reasoner_prob_wiki, log_reasoner_prob_kb], dim = 0) 
            action_ind_merge = torch.cat([action_ind_wiki, action_ind_kb], dim = 0)

            log_reasoner_prob_merge_descend, descend_index = log_reasoner_prob_merge.sort(0, descending=True)
            action_ind_merge_descend = action_ind_merge[descend_index]

            l_log_action_prob.append(log_reasoner_prob_merge_descend)
            l_action_ind.append(action_ind_merge_descend)
        
        else:
            log_action_prob, action_ind = torch.topk(log_reasoner_action_dist_i, topK)
            l_log_action_prob.append(log_action_prob)
            l_action_ind.append(action_ind)
    
    batch_log_action_prob_topK = torch.stack(l_log_action_prob, dim=0)
    batch_action_index_topK = torch.stack(l_action_ind, dim = 0)

    return batch_log_action_prob_topK, batch_action_index_topK


def adaptive_top_k_actions(batch_size, beam_size, keep_wiki_path_num, log_reasoner_action_dist_sum, batch_buckets_r_global, batch_buckets_e_global, batch_buckets_is_wiki_relation_in_path, batch_buckets_extractor_action_dist_log, batch_buckets_reasoner_action_dist_log, batch_buckets_token_states, func_type = "train"):
    full_size, action_space_size, max_question_len_global = batch_buckets_token_states.shape
    last_k = int(full_size / batch_size)

    log_reasoner_action_dist_sum = log_reasoner_action_dist_sum.view(batch_size, -1)
    batch_buckets_is_wiki_relation_in_path = batch_buckets_is_wiki_relation_in_path.view(batch_size, -1)
    batch_buckets_r_global = batch_buckets_r_global.view(batch_size, -1)
    batch_buckets_e_global = batch_buckets_e_global.view(batch_size, -1)

    if args.pretrain_extractor or args.collaborate_training:
        batch_buckets_extractor_action_dist_log = batch_buckets_extractor_action_dist_log.view(batch_size, -1)

    batch_buckets_reasoner_action_dist_log = batch_buckets_reasoner_action_dist_log.view(batch_size, -1)
    batch_buckets_token_states = batch_buckets_token_states.view(batch_size, -1, max_question_len_global)

    beam_action_space_size = log_reasoner_action_dist_sum.size()[1]
    topK = min(beam_size, beam_action_space_size)

    if func_type == "train":
        batch_log_action_prob_sum_topK, batch_action_index_topK = adaptive_sampling_actions(topK, keep_wiki_path_num, log_reasoner_action_dist_sum, batch_buckets_is_wiki_relation_in_path)
    
    elif func_type == "inference":
        batch_log_action_prob_sum_topK, batch_action_index_topK = torch.topk(log_reasoner_action_dist_sum, topK)

    batch_buckets_r_global_topK = torch.gather(batch_buckets_r_global, 1, batch_action_index_topK).view(-1)
    batch_buckets_e_global_topK = torch.gather(batch_buckets_e_global, 1, batch_action_index_topK).view(-1)

    if args.pretrain_extractor or args.collaborate_training:
        batch_buckets_extractor_action_dist_log_topK = torch.gather(batch_buckets_extractor_action_dist_log, 1, batch_action_index_topK).view(-1)
    else:
        batch_buckets_extractor_action_dist_log_topK = None

    batch_buckets_reasoner_action_dist_log_topK = torch.gather(batch_buckets_reasoner_action_dist_log, 1, batch_action_index_topK).view(-1)
    batch_log_action_prob_sum_topK = batch_log_action_prob_sum_topK.view(-1)

    new_token_states_list = []
    for b_index in range(batch_size):
        action_indices = batch_action_index_topK[b_index]
        state_vector = batch_buckets_token_states[b_index][action_indices]
        new_token_states_list.append(state_vector)
    
    token_states_matrix = torch.stack(new_token_states_list, dim=0).view(-1, max_question_len_global)
    
    action_beam_offset = batch_action_index_topK // action_space_size
    action_batch_offset = (torch.arange(batch_size).cuda() * last_k).unsqueeze(1)
    
    action_offset = (action_batch_offset + action_beam_offset).view(-1)

    return action_offset, batch_log_action_prob_sum_topK, batch_buckets_r_global_topK, batch_buckets_e_global_topK, batch_buckets_extractor_action_dist_log_topK, batch_buckets_reasoner_action_dist_log_topK, token_states_matrix

def rollout_beam(env, reasoner, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, func_type = "train"):

    batch_question, batch_question_seq_lengths = env.return_batch_data()

    batch_size, _ = batch_question.shape
    
    log_reasoner_action_prob_history = torch.zeros(batch_size).cuda()

    for t in range(0, env.max_hop):
        path_trace_global, l_path_hidden, l_token_memory_state = env.observe()

        _, e_t_global = path_trace_global[-1]
        batch_path_hidden = l_path_hidden[-1][0][-1, :, :]
        batch_token_memory_state = l_token_memory_state[-1]

        batch_is_wiki_in_history = env.judge_wiki_relation_in_history()

        k = int(e_t_global.size()[0] / batch_size)

        beam_question = batch_question.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1)
        beam_question_len = batch_question_seq_lengths.unsqueeze(1).repeat(1, k).view(batch_size * k)

        if args.pretrain_extractor or args.collaborate_training:
            batch_wiki_action_space, batch_wiki_action_dist_log = extractor.wiki_transit(t, reasoner, e_t_global, beam_question, beam_question_len, batch_path_hidden, d_relation2id_wiki, d_relationid2text_wiki, d_word2id)
        else:
            batch_wiki_action_space = None
            batch_wiki_action_dist_log = None
        
        inv_offset, values_list, l_2D, l_internal_token_states = reasoner.transit(t, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, e_t_global, beam_question, beam_question_len, batch_path_hidden, batch_token_memory_state, batch_is_wiki_in_history, batch_wiki_action_space, batch_wiki_action_dist_log)

        values_t = torch.cat(values_list, dim=0)[inv_offset]

        l_buckets_r_global = [r for r, _, _, _, _, _ in l_2D]
        l_buckets_e_global = [e_g for _, e_g, _, _, _, _ in l_2D]
        l_buckets_is_wiki_relation_in_path = [w_r for _, _, _, w_r, _, _ in l_2D]
        l_buckets_extractor_action_dist_log = [e_a for _, _, _, _, e_a, _ in l_2D]
        l_buckets_reasoner_action_dist = [r_a for _, _, _, _, _, r_a in l_2D]

        batch_buckets_r_global = pad_and_cat1d(l_buckets_r_global, args.DUMMY_RELATION_idx, inv_offset, padding_dim=1)
        batch_buckets_e_global = pad_and_cat1d(l_buckets_e_global, args.DUMMY_ENTITY_idx, inv_offset, padding_dim=1)

        batch_buckets_is_wiki_relation_in_path = pad_and_cat1d(l_buckets_is_wiki_relation_in_path, 0, inv_offset, padding_dim=1)

        if args.pretrain_extractor or args.collaborate_training:
            batch_buckets_extractor_action_dist_log = pad_and_cat1d(l_buckets_extractor_action_dist_log, float("-inf"), inv_offset, padding_dim=1)
        else:
            batch_buckets_extractor_action_dist_log = None
        
        batch_buckets_reasoner_action_dist = pad_and_cat1d(l_buckets_reasoner_action_dist, 0, inv_offset, padding_dim=1)
        batch_buckets_reasoner_action_dist_log = safe_log(batch_buckets_reasoner_action_dist)

        batch_buckets_token_states = pad_and_cat2d(l_internal_token_states, 0, inv_offset, padding_dim=1)

        log_reasoner_action_dist_sum = log_reasoner_action_prob_history.view(-1, 1) + batch_buckets_reasoner_action_dist_log

        if func_type == "train":
            beam_size = args.beam_size_train
        elif func_type == "inference":
            beam_size = args.beam_size_inference

        action_offset, log_reasoner_action_prob_history, batch_buckets_r_global_topK, batch_buckets_e_global_topK, batch_buckets_extractor_action_dist_log_topK, batch_buckets_reasoner_action_dist_log_topK, token_states_matrix = adaptive_top_k_actions(batch_size, beam_size, args.keep_wiki_path_num, log_reasoner_action_dist_sum, batch_buckets_r_global, batch_buckets_e_global, batch_buckets_is_wiki_relation_in_path, batch_buckets_extractor_action_dist_log, batch_buckets_reasoner_action_dist_log, batch_buckets_token_states, func_type)

        path_list, (h_t, c_t) = reasoner.update_path(extractor, (batch_buckets_r_global_topK, batch_buckets_e_global_topK), l_path_hidden, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, offset = action_offset)
        new_hidden = (h_t, c_t)

        env.step(path_list, new_hidden, batch_buckets_r_global_topK, batch_buckets_e_global_topK, token_states_matrix, values_t, batch_buckets_extractor_action_dist_log_topK, batch_buckets_reasoner_action_dist_log_topK, offset = action_offset)
    
    return log_reasoner_action_prob_history
