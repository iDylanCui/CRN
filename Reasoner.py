import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from Transformer import TransformerModel
from utils import wiki_relationid_to_name, kb_relation_global2local, index2word
from parse_args import args
from transformers import RobertaTokenizerFast, RobertaModel

class Reasoner_Network(nn.Module):
    def __init__(self, args, kb_relation_num, word_embeddings, entity_embeddings, relation_embeddings, entity2bucketid_kb, d_action_space_buckets_kb):
        super(Reasoner_Network, self).__init__()

        self.max_hop = args.max_hop
        self.kb_relation_num = kb_relation_num
        self.word_dim = args.word_dim
        self.word_padding_idx = args.word_padding_idx
        self.DUMMY_RELATION_idx = args.DUMMY_RELATION_idx
        self.DUMMY_ENTITY_idx = args.DUMMY_ENTITY_idx
        self.word_dropout_rate = args.word_dropout_rate
        self.is_train_emb = args.is_train_emb
        self.max_question_len_global = args.max_question_len_global
        self.use_ecm_tokens_internal_memory = args.use_ecm_tokens_internal_memory

        self.entity2bucketid_kb = entity2bucketid_kb
        self.d_action_space_buckets_kb = d_action_space_buckets_kb

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate

        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.head_num = args.head_num
        self.hidden_dim = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.transform_output_dim = self.word_dim

        self.relation_only = args.relation_only
        self.history_dim = args.history_dim
        self.history_layers = args.history_layers
        self.rl_dropout_rate = args.rl_dropout_rate

        self.word_embeddings = Embedding(word_embeddings, self.word_dropout_rate, self.is_train_emb, self.word_padding_idx)

        self.entity_embeddings = Embedding(entity_embeddings, self.emb_dropout_rate, self.is_train_emb, self.DUMMY_ENTITY_idx)
        self.relation_embeddings = Embedding(relation_embeddings, self.emb_dropout_rate, self.is_train_emb, self.DUMMY_RELATION_idx)

        self.Transformer = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate)

        if self.use_ecm_tokens_internal_memory:
            self.l_internal_state_W_read_KG = nn.ModuleList([nn.Linear(self.transform_output_dim + self.history_dim + self.relation_dim, self.max_question_len_global) for i in range(self.max_hop)])

            self.l_internal_state_W_write_KG = nn.ModuleList([nn.Linear(self.transform_output_dim + self.relation_dim, self.max_question_len_global) for i in range(self.max_hop)])
        
        else:
            self.l_question_linears = nn.ModuleList([nn.Linear(self.word_dim, self.relation_dim) for i in range(self.max_hop)])
            self.relation_linear = nn.Linear(self.relation_dim, self.relation_dim)
            self.W_att = nn.Linear(self.relation_dim, 1)

        self.input_dim = self.history_dim + self.word_dim
        
        if self.relation_only:
            self.action_dim = self.relation_dim
        else:
            self.action_dim = self.relation_dim + self.entity_dim
        
        self.lstm_input_dim = self.action_dim
        
        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(self.rl_dropout_rate)
        self.W2Dropout = nn.Dropout(self.rl_dropout_rate)

        self.use_entity_embedding_in_vn = args.use_entity_embedding_vn
        if self.use_entity_embedding_in_vn:
            self.W_value = nn.Linear(self.history_dim + self.entity_dim, 1)
        else:
            self.W_value = nn.Linear(self.history_dim, 1) 

        self.path_encoder = nn.LSTM(input_size=self.lstm_input_dim,
                                    hidden_size=self.history_dim,
                                    num_layers=self.history_layers,
                                    batch_first=True)
        
        self.wiki_encoding_model_name = args.wiki_encoding_model_name
        self.hugging_face_path = args.hugging_face_path
        if self.wiki_encoding_model_name == "Transformer":
            self.wiki_relation_dim = self.word_dim
        else:
            self.wiki_relation_dim = 768
        
        if self.use_ecm_tokens_internal_memory:
            self.l_internal_state_W_read_wiki = nn.ModuleList([nn.Linear(self.transform_output_dim + self.history_dim + self.wiki_relation_dim, self.max_question_len_global) for i in range(self.max_hop)])

            self.l_internal_state_W_write_wiki = nn.ModuleList([nn.Linear(self.transform_output_dim + self.wiki_relation_dim, self.max_question_len_global) for i in range(self.max_hop)])

        if self.relation_only:
            self.action_wiki_dim = self.wiki_relation_dim
        else:
            self.action_wiki_dim = self.wiki_relation_dim + self.entity_dim

        self.W1_wiki = nn.Linear(self.input_dim, self.action_wiki_dim)
        self.W2_wiki = nn.Linear(self.action_wiki_dim, self.action_wiki_dim)
        self.rel_linear_wiki = nn.Linear(self.wiki_relation_dim, self.relation_dim)
        
        self.initialize_modules()
    
    def get_plm_question_representation(self, l_batch_question):
        encoded_input = self.tokenizer(l_batch_question, 
                padding = True,
                truncation = True,
                return_tensors = "pt")
        
        d_encoded_input = {}
        for k, v in encoded_input.items():
            d_encoded_input[k] = v.cuda()

        output = self.Transformer(**d_encoded_input)
        last_hidden_states = output.last_hidden_state
        
        cls_hidden_states = last_hidden_states[:, 0, :]

        return encoded_input["attention_mask"], last_hidden_states, cls_hidden_states, 

    def get_question_representation(self, batch_question, batch_sent_len):
        batch_question_embedding = self.word_embeddings(batch_question) 

        mask = self.batch_sentence_mask(batch_sent_len) 
        transformer_output = self.Transformer(batch_question_embedding.permute(1, 0 ,2), mask)
        
        transformer_output = transformer_output.permute(1, 0 ,2)

        return transformer_output, mask
    
    def question_max_pooling(self, transformer_out, question_mask):
        _, _, output_dim = transformer_out.shape
        question_mask = question_mask.unsqueeze(-1).repeat(1,1, output_dim)
        transformer_out_masked = transformer_out.masked_fill(question_mask, float('-inf'))
        question_transformer_masked = transformer_out_masked.transpose(1, 2)
        question_mp = F.max_pool1d(question_transformer_masked, question_transformer_masked.size(2)).squeeze(2)

        return question_mp
    
    def get_relation_aware_question_vector_attention(self, t, b_question_vectors, b_question_mask, b_r_embeddings, b_r_localId):
        relation_num, _ = b_r_embeddings.shape
        b_size, seq_len = b_question_mask.shape

        b_question_vectors = b_question_vectors.unsqueeze(1).repeat(1, relation_num, 1, 1).view(b_size * relation_num, seq_len, -1)
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, relation_num, 1).view(b_size * relation_num, seq_len)

        b_question_project = self.l_question_linears[t](b_question_vectors)
        
        b_relation_vector = b_r_embeddings.unsqueeze(1).unsqueeze(0).repeat(b_size, 1, seq_len, 1).view(b_size * relation_num, seq_len, -1)
        b_relation_project = self.relation_linear(b_relation_vector)

        b_att_features = b_question_project + b_relation_project

        b_att_features_tanh = torch.tanh(b_att_features)
        b_linear_result = self.W_att(b_att_features_tanh).squeeze(-1)
        b_linear_result_masked = b_linear_result.masked_fill(b_question_mask, float('-inf'))
        b_matrix_alpha = F.softmax(b_linear_result_masked, 1).unsqueeze(1)

        b_relation_aware_question_vector = torch.matmul(b_matrix_alpha, b_question_vectors).squeeze(1).view(b_size, relation_num, -1)
        b_matrix_alpha = b_matrix_alpha.squeeze(1).view(b_size, relation_num, -1)

        l_relation_aware_question_vector = []
        l_matrix_alpha = []

        for batch_i in range(b_size):
            output_i = b_relation_aware_question_vector[batch_i]
            matrix_i = b_matrix_alpha[batch_i]
            relation_i = b_r_localId[batch_i]
            new_output_i = output_i[relation_i]
            new_matrix_i = matrix_i[relation_i]

            l_relation_aware_question_vector.append(new_output_i)
            l_matrix_alpha.append(new_matrix_i)
        
        b_relation_aware_question_vector = torch.stack(l_relation_aware_question_vector, 0)
        b_matrix_alpha = torch.stack(l_matrix_alpha, 0)

        return b_relation_aware_question_vector, b_matrix_alpha

    def get_relation_aware_question_vector_ecm(self, t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_embeddings, b_token_memory_state):
        b_size, action_num, hidden_dim = b_r_embeddings.shape

        b_question_mp = b_question_mp.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)
        b_question_vectors = b_question_vectors.unsqueeze(1).repeat(1, action_num, 1, 1).view(b_size * action_num, self.max_question_len_global, -1)

        b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)
        b_relation_vector = b_r_embeddings.view(b_size * action_num, -1)
        b_token_memory_state = b_token_memory_state.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, -1)

        b_internal_state_input = torch.cat([b_question_mp, b_path_hidden, b_relation_vector], dim = -1)

        if hidden_dim == self.relation_dim:
            b_internal_state_read = torch.sigmoid(self.l_internal_state_W_read_KG[t](b_internal_state_input))
        else:
            b_internal_state_read = torch.sigmoid(self.l_internal_state_W_read_wiki[t](b_internal_state_input))

        b_internal_memory = b_internal_state_read * b_token_memory_state

        b_internal_memory_masked = b_internal_memory.masked_fill(b_question_mask, float('-inf'))
        b_internal_memory_softmax = F.softmax(b_internal_memory_masked, 1).unsqueeze(1)

        b_question_vectors_att = torch.matmul(b_internal_memory_softmax, b_question_vectors).squeeze(1)
        
        b_internal_state_output = torch.cat([b_relation_vector, b_question_vectors_att], dim = -1)

        if hidden_dim == self.relation_dim:
            b_internal_state_write = torch.sigmoid(self.l_internal_state_W_write_KG[t](b_internal_state_output)) 
        else:
            b_internal_state_write = torch.sigmoid(self.l_internal_state_W_write_wiki[t](b_internal_state_output))

        b_token_memory_state_new = b_internal_state_write * b_token_memory_state

        return b_question_vectors_att.view(b_size, action_num, -1), b_token_memory_state_new.view(b_size, action_num, -1)
    
    def policy_linear(self, b_input_vector): 
        X = self.W1(b_input_vector)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)
        return X2

    def policy_linear_wiki(self, b_input_vector):
        X = self.W1_wiki(b_input_vector)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2_wiki(X)
        X2 = self.W2Dropout(X)
        return X2
    
    def get_action_embedding_kg(self, action):
        r, e = action
        relation_embedding = self.relation_embeddings(r)
        
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = self.entity_embeddings(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding
    
    def get_action_embedding_merge(self, batch_wiki_r_embeddings, batch_wiki_e_embeddings, batch_kb_r_embeddings, batch_kb_e_embeddings, l_references):
        batch_r_merge = torch.cat([batch_wiki_r_embeddings, batch_kb_r_embeddings], dim = 0)
        batch_e_merge = torch.cat([batch_wiki_e_embeddings, batch_kb_e_embeddings], dim = 0)

        if self.relation_only:
            batch_action_embeddings = batch_r_merge
        else:
            batch_action_embeddings = torch.cat([batch_r_merge, batch_e_merge], dim=-1)

        inv_offset = [i for i, _ in sorted(enumerate(l_references), key=lambda x: x[1])]
        batch_action_embeddings = batch_action_embeddings[inv_offset]
        
        return batch_action_embeddings

    def calculate_kg_policy(self, t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_space, b_e_space_global, b_token_memory_state):
        b_size, action_num = b_r_space.shape
        if self.use_ecm_tokens_internal_memory:
            b_r_embeddings = self.relation_embeddings(b_r_space)
            b_question_vectors_att, b_token_memory_state_new = self.get_relation_aware_question_vector_ecm(t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_embeddings, b_token_memory_state)
        else:
            r_tensor_globalId, b_r_localId = kb_relation_global2local(b_r_space)
            b_r_embeddings = self.relation_embeddings(r_tensor_globalId)

            b_question_vectors_att, _ = self.get_relation_aware_question_vector_attention(t, b_question_vectors, b_question_mask, b_r_embeddings, b_r_localId)

            b_token_memory_state_new = torch.zeros(b_size, action_num, self.max_question_len_global).cuda()

        b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num, 1)
        b_policy_network_input = torch.cat([b_path_hidden, b_question_vectors_att], -1)

        b_policy_network_output = self.policy_linear(b_policy_network_input)
        b_action_embedding = self.get_action_embedding_kg((b_r_space, b_e_space_global))

        b_action_embedding = b_action_embedding.view(-1, self.action_dim).unsqueeze(1)
        b_output_vector = b_policy_network_output.view(-1, self.action_dim).unsqueeze(-1)
        b_action_logit = torch.matmul(b_action_embedding, b_output_vector).squeeze(-1).view(-1, action_num)

        return b_token_memory_state_new, b_action_logit
    
    def calculate_wiki_policy(self, t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden,b_token_memory_state, b_wiki_r_hiddens, b_wiki_e_space_global):
        _, action_num_wiki = b_wiki_e_space_global.shape
        b_question_vectors_att, b_token_memory_state_new = self.get_relation_aware_question_vector_ecm(t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_wiki_r_hiddens, b_token_memory_state)

        b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num_wiki, 1)
        b_policy_network_input = torch.cat([b_path_hidden, b_question_vectors_att], -1)

        b_policy_network_output = self.policy_linear_wiki(b_policy_network_input)

        if self.relation_only:
            b_action_embedding = b_wiki_r_hiddens
        else:
            b_entity_embedding = self.entity_embeddings(b_wiki_e_space_global)
            b_action_embedding = torch.cat([b_wiki_r_hiddens, b_entity_embedding], dim=-1)
        
        b_action_embedding = b_action_embedding.view(-1, self.action_wiki_dim).unsqueeze(1)
        b_output_vector = b_policy_network_output.view(-1, self.action_wiki_dim).unsqueeze(-1)
        b_action_logit = torch.matmul(b_action_embedding, b_output_vector).squeeze(-1).view(-1, action_num_wiki)

        return b_token_memory_state_new, b_action_logit
    
    def get_wiki_relation_hiddens(self, extractor, batch_wiki_r_space_global, d_relation2id_wiki, d_relationid2text_wiki, d_word2id):
        l_batch_wiki_relations, batch_wiki_relation_lengths, batch_wiki_relations_tokenIds, batch_wiki_r_space_local = wiki_relationid_to_name(batch_wiki_r_space_global, d_relation2id_wiki, d_relationid2text_wiki, d_word2id)

        if self.wiki_encoding_model_name == "Transformer":
            with torch.no_grad():
                batch_wiki_r_hiddens = extractor.get_transformer_rel_hidden(batch_wiki_relation_lengths, batch_wiki_relations_tokenIds, batch_wiki_r_space_local)

        elif self.wiki_encoding_model_name == "DistilBert" or self.wiki_encoding_model_name == "RoBERTa":
            with torch.no_grad():
                batch_wiki_r_hiddens = extractor.get_plm_rel_hidden(l_batch_wiki_relations, batch_wiki_r_space_local)
        
        return batch_wiki_r_hiddens
    
    def get_action_space_in_buckets(self, batch_e_t, d_entity2bucketid, d_action_space_buckets):
        db_action_spaces, db_references = [], []

        entity2bucketid = d_entity2bucketid[batch_e_t.tolist()]
        key1 = entity2bucketid[:, 0]
        key2 = entity2bucketid[:, 1]
        batch_ref = {}

        for i in range(len(batch_e_t)):
            key = int(key1[i])
            if not key in batch_ref:
                batch_ref[key] = []
            batch_ref[key].append(i)
        for key in batch_ref:
            action_space = d_action_space_buckets[key]

            l_batch_refs = batch_ref[key]
            g_bucket_ids = key2[l_batch_refs].tolist()
            r_space_b = action_space[0][0][g_bucket_ids]
            e_space_b = action_space[0][1][g_bucket_ids]
            action_mask_b = action_space[1][g_bucket_ids]

            r_space_b = r_space_b.cuda()
            e_space_b = e_space_b.cuda()
            action_mask_b = action_mask_b.cuda()

            action_space_b = ((r_space_b, e_space_b), action_mask_b)
            db_action_spaces.append(action_space_b)
            db_references.append(l_batch_refs)

        return db_action_spaces, db_references

    def transit(self, t, extractor, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, e_t_global, batch_question, batch_question_seq_lengths, batch_path_hidden, batch_token_memory_state, batch_is_wiki_in_history, batch_wiki_action_space, batch_extractor_action_dist_log):
        db_action_spaces_kg, db_references_kg = self.get_action_space_in_buckets(e_t_global, self.entity2bucketid_kb, self.d_action_space_buckets_kb)

        batch_question_vectors, batch_question_mask = self.get_question_representation(batch_question, batch_question_seq_lengths)

        batch_question_mp = self.question_max_pooling(batch_question_vectors, batch_question_mask)

        if args.pretrain_extractor or args.collaborate_training:
            (batch_wiki_r_space_global, batch_wiki_e_space_global), batch_wiki_action_mask = batch_wiki_action_space

            batch_wiki_r_hiddens = self.get_wiki_relation_hiddens(extractor, batch_wiki_r_space_global, d_relation2id_wiki, d_relationid2text_wiki, d_word2id)

        references = []
        values_list = []
        l_2D = []
        l_internal_token_states = []

        for b_action_space_kg, b_reference_kg in zip(db_action_spaces_kg, db_references_kg):
            b_e_t_global = e_t_global[b_reference_kg]
            b_question_vectors = batch_question_vectors[b_reference_kg]
            b_question_mask = batch_question_mask[b_reference_kg]
            b_question_mp = batch_question_mp[b_reference_kg]
            b_path_hidden = batch_path_hidden[b_reference_kg]
            b_token_memory_state = batch_token_memory_state[b_reference_kg]
            b_is_wiki_in_history = batch_is_wiki_in_history[b_reference_kg].unsqueeze(-1)

            (b_r_space_global_kg, b_e_space_global_kg), b_action_mask_kg = b_action_space_kg
            b_size, action_num_kg = b_r_space_global_kg.shape

            if self.use_entity_embedding_in_vn:
                b_e_t_global_embeddings = self.entity_embeddings(b_e_t_global)
                value_input = torch.cat([b_e_t_global_embeddings, b_path_hidden], dim = -1)
                b_value = self.W_value(value_input).view(-1)
            else:
                b_value = self.W_value(b_path_hidden).view(-1)

            b_value = torch.sigmoid(b_value)
            values_list.append(b_value)

            b_token_memory_state_new_kg, b_action_logit_kg = self.calculate_kg_policy(t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_r_space_global_kg, b_e_space_global_kg, b_token_memory_state)

            if args.pretrain_extractor or args.collaborate_training:
                b_e_space_global_wiki = batch_wiki_e_space_global[b_reference_kg]
                b_r_space_global_wiki = batch_wiki_r_space_global[b_reference_kg]
                b_action_mask_wiki = batch_wiki_action_mask[b_reference_kg]
                b_r_hiddens_wiki = batch_wiki_r_hiddens[b_reference_kg]
                b_extractor_action_dist_log = batch_extractor_action_dist_log[b_reference_kg]
                padding = nn.ConstantPad1d((action_num_kg, 0), float("-inf"))
                b_extractor_action_dist_log_pad = padding(b_extractor_action_dist_log)

                b_token_memory_state_new_wiki, b_action_logit_wiki = self.calculate_wiki_policy(t, b_question_vectors, b_question_mask, b_question_mp, b_path_hidden, b_token_memory_state, b_r_hiddens_wiki, b_e_space_global_wiki)

                b_r_global_merge = torch.cat([b_r_space_global_kg, b_r_space_global_wiki], dim = -1)
                b_e_global_merge = torch.cat([b_e_space_global_kg, b_e_space_global_wiki], dim = -1)
                b_action_mask_merge = torch.cat([b_action_mask_kg, b_action_mask_wiki], dim = -1)
                b_action_logit_merge = torch.cat([b_action_logit_kg, b_action_logit_wiki], dim = -1)
                b_token_memory_state_new_merge = torch.cat([b_token_memory_state_new_kg, b_token_memory_state_new_wiki], dim = 1)
            
            else:
                b_r_global_merge = b_r_space_global_kg
                b_e_global_merge = b_e_space_global_kg
                b_action_mask_merge = b_action_mask_kg
                b_action_logit_merge = b_action_logit_kg
                b_token_memory_state_new_merge = b_token_memory_state_new_kg
                b_extractor_action_dist_log_pad = None

            b_is_wiki_relation_in_path_merge = self.judge_wiki_relation_in_path(b_is_wiki_in_history, b_r_global_merge)

            b_action_logit_merge_masked = b_action_logit_merge.masked_fill((1 - b_action_mask_merge).bool(), float('-inf'))
            b_reasoner_action_dist_merge = F.softmax(b_action_logit_merge_masked, 1)

            b_2D_merge = (b_r_global_merge, b_e_global_merge, b_action_mask_merge, b_is_wiki_relation_in_path_merge, b_extractor_action_dist_log_pad, b_reasoner_action_dist_merge)

            references.extend(b_reference_kg)
            l_2D.append(b_2D_merge)
            l_internal_token_states.append(b_token_memory_state_new_merge)
        
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        
        return inv_offset, values_list, l_2D, l_internal_token_states

    def initialize_path(self, init_action):
        init_action_embedding = self.get_action_embedding_kg(init_action)
        init_action_embedding.unsqueeze_(1)
        init_h = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim])
        init_c = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim])

        init_h = init_h.cuda()
        init_c = init_c.cuda()

        h_n, c_n = self.path_encoder(init_action_embedding, (init_h, init_c))[1]
        return (h_n, c_n)

    def update_path(self, extractor, action, path_list, d_relation2id_wiki, d_relationid2text_wiki, d_word2id, offset=None):
        
        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        batch_r_global, batch_e_global = action

        batch_r_is_wiki = batch_r_global >= self.kb_relation_num
        batch_r_is_kb = batch_r_global < self.kb_relation_num
        batch_r_is_wiki_index = torch.nonzero(batch_r_is_wiki).view(-1)
        batch_r_is_kb_index = torch.nonzero(batch_r_is_kb).view(-1)
        l_r_is_wiki_index = batch_r_is_wiki_index.cpu().numpy().tolist()
        l_r_is_kb_index = batch_r_is_kb_index.cpu().numpy().tolist()

        if len(l_r_is_wiki_index) > 0:
            l_references = l_r_is_wiki_index + l_r_is_kb_index

            batch_r_global_wiki = batch_r_global[l_r_is_wiki_index].unsqueeze(-1)
            batch_e_global_wiki = batch_e_global[l_r_is_wiki_index]

            batch_wiki_r_hiddens = self.get_wiki_relation_hiddens(extractor, batch_r_global_wiki, d_relation2id_wiki, d_relationid2text_wiki, d_word2id).squeeze(1)
            batch_wiki_r_embeddings = self.rel_linear_wiki(batch_wiki_r_hiddens)
            batch_wiki_r_embeddings = self.W1Dropout(batch_wiki_r_embeddings)
            batch_wiki_e_embeddings = self.entity_embeddings(batch_e_global_wiki)

            batch_r_global_kb = batch_r_global[l_r_is_kb_index]
            batch_e_global_kb = batch_e_global[l_r_is_kb_index]

            batch_kb_r_embeddings = self.relation_embeddings(batch_r_global_kb)
            batch_kb_e_embeddings = self.entity_embeddings(batch_e_global_kb)

            batch_action_embeddings = self.get_action_embedding_merge(batch_wiki_r_embeddings, batch_wiki_e_embeddings, batch_kb_r_embeddings, batch_kb_e_embeddings, l_references).unsqueeze(1)
        
        else:
            batch_action_embeddings = self.get_action_embedding_kg(action)
            batch_action_embeddings.unsqueeze_(1)

        if offset is not None:
            offset_path_history(path_list, offset)

        h_n, c_n = self.path_encoder(batch_action_embeddings, path_list[-1])[1]
        return path_list, (h_n, c_n)

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.0)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.constant_(self.W_value.bias, 0.0)
        nn.init.xavier_uniform_(self.W1_wiki.weight)
        nn.init.constant_(self.W1_wiki.bias, 0.0)
        nn.init.xavier_uniform_(self.W2_wiki.weight)
        nn.init.constant_(self.W2_wiki.bias, 0.0)
        nn.init.xavier_uniform_(self.rel_linear_wiki.weight)
        nn.init.constant_(self.rel_linear_wiki.bias, 0.0)

        if not self.use_ecm_tokens_internal_memory:
            nn.init.xavier_uniform_(self.relation_linear.weight)
            nn.init.constant_(self.relation_linear.bias, 0.0)
            nn.init.xavier_uniform_(self.W_att.weight)
            nn.init.constant_(self.W_att.bias, 0.0)


        for i in range(self.max_hop):
            if self.use_ecm_tokens_internal_memory:
                nn.init.xavier_uniform_(self.l_internal_state_W_read_KG[i].weight)
                nn.init.constant_(self.l_internal_state_W_read_KG[i].bias, 0.0)

                nn.init.xavier_uniform_(self.l_internal_state_W_write_KG[i].weight)
                nn.init.constant_(self.l_internal_state_W_write_KG[i].bias, 0.0)

                nn.init.xavier_uniform_(self.l_internal_state_W_read_wiki[i].weight)
                nn.init.constant_(self.l_internal_state_W_read_wiki[i].bias, 0.0)

                nn.init.xavier_uniform_(self.l_internal_state_W_write_wiki[i].weight)
                nn.init.constant_(self.l_internal_state_W_write_wiki[i].bias, 0.0)
            else:
                nn.init.xavier_uniform_(self.l_question_linears[i].weight)
                nn.init.constant_(self.l_question_linears[i].bias, 0.0)

        for name, param in self.path_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def batch_sentence_mask(self, batch_sent_len):
        batch_size = len(batch_sent_len)
        if self.use_ecm_tokens_internal_memory:
            max_sent_len = self.max_question_len_global
        else:
            max_sent_len = batch_sent_len[0]
        
        mask = torch.zeros(batch_size, max_sent_len, dtype=torch.long)

        for i in range(batch_size):
            sent_len = batch_sent_len[i]
            mask[i][sent_len:] = 1
        
        mask = (mask == 1)
        mask = mask.cuda()
        return mask
    
    def judge_wiki_relation_in_path(self, b_is_wiki_in_history, b_r_global_merge):
        b_r_is_wiki = b_r_global_merge >= self.kb_relation_num
        b_result = b_is_wiki_in_history | b_r_is_wiki
        return b_result
    
    def return_entity_kg_embeddings(self, b_e_global):
        return self.entity_embeddings(b_e_global)
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)