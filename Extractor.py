import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from Transformer import TransformerModel
from transformers import RobertaTokenizerFast, DistilBertTokenizerFast, DistilBertModel, RobertaModel
from utils import wiki_relationid_to_name, safe_log

class Extractor_Network(nn.Module):
    def __init__(self, args, word_embeddings, entity2bucketid_wiki, d_action_space_buckets_wiki):
        super(Extractor_Network, self).__init__()
        self.max_hop = args.max_hop
        self.wiki_encoding_model_name = args.wiki_encoding_model_name
        self.hugging_face_path = args.hugging_face_path
        self.extractor_topK = args.extractor_topK
        self.use_ecm_tokens_internal_memory = args.use_ecm_tokens_internal_memory
        self.entity2bucketid_wiki = entity2bucketid_wiki
        self.d_action_space_buckets_wiki = d_action_space_buckets_wiki

        self.word_dim = args.word_dim
        self.word_padding_idx = args.word_padding_idx
        self.word_dropout_rate = args.word_dropout_rate
        self.is_train_emb = args.is_train_emb
        self.max_question_len_global = args.max_question_len_global

        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.head_num = args.head_num
        self.hidden_dim = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.transform_output_dim = self.word_dim

        self.relation_only = args.relation_only
        self.history_dim = args.history_dim
        self.rl_dropout_rate = args.rl_dropout_rate

        self.word_embeddings = Embedding(word_embeddings, self.word_dropout_rate, self.is_train_emb, self.word_padding_idx)

        self.entity_dim = args.entity_dim

        if self.wiki_encoding_model_name == "Transformer":
            self.wiki_encoding_model = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate)
            self.relation_dim = self.word_dim
        
        elif self.wiki_encoding_model_name == "DistilBert":
            l_additional_special_tokens = ["SUBSUBJECT", "OBJOBJECT", "DUMMY_RELATION",	"START_RELATION", "NO_OP_RELATION"]
            distilbert_base_uncased_file = os.path.join(self.hugging_face_path, "distilbert-base-uncased")

            self.tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_base_uncased_file, additional_special_tokens = l_additional_special_tokens)
            self.wiki_encoding_model = DistilBertModel.from_pretrained(distilbert_base_uncased_file)
            self.wiki_encoding_model.resize_token_embeddings(len(self.tokenizer))
            self.relation_dim = 768

        elif self.wiki_encoding_model_name == "RoBERTa":
            l_additional_special_tokens = ["SUBSUBJECT", "OBJOBJECT", "DUMMY_RELATION",	"START_RELATION", "NO_OP_RELATION"]
            roberta_base_file = os.path.join(self.hugging_face_path, "roberta-base")

            self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_base_file, additional_special_tokens = l_additional_special_tokens)
            self.wiki_encoding_model = RobertaModel.from_pretrained(roberta_base_file)
            self.wiki_encoding_model.resize_token_embeddings(len(self.tokenizer))
            self.relation_dim = 768

        self.Transformer_q = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate)

        self.l_stepwise_q_W = nn.ModuleList([nn.Linear(self.transform_output_dim, self.transform_output_dim) for i in range(self.max_hop)])
        self.W_q_att = nn.Linear(self.transform_output_dim, 1)

        self.input_dim = self.history_dim + self.word_dim
        
        if self.relation_only:
            self.action_dim = self.relation_dim
        else:
            self.action_dim = self.relation_dim + self.entity_dim

        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(self.rl_dropout_rate)
        self.W2Dropout = nn.Dropout(self.rl_dropout_rate)

        self.initialize_modules()

    def get_question_representation(self, batch_question, batch_sent_len):
        batch_question_embedding = self.word_embeddings(batch_question)

        mask = self.batch_sentence_mask(batch_sent_len, func_type= "question")
        transformer_output = self.Transformer_q(batch_question_embedding.permute(1, 0 ,2), mask)
        
        transformer_output = transformer_output.permute(1, 0 ,2)

        return transformer_output, mask
    
    def question_max_pooling(self, transformer_out, question_mask):
        _, _, output_dim = transformer_out.shape
        question_mask = question_mask.unsqueeze(-1).repeat(1,1, output_dim)
        transformer_out_masked = transformer_out.masked_fill(question_mask, float('-inf'))
        question_transformer_masked = transformer_out_masked.transpose(1, 2)
        question_mp = F.max_pool1d(question_transformer_masked, question_transformer_masked.size(2)).squeeze(2)

        return question_mp
    
    def get_timestep_question_vector(self, t, batch_question_vectors, batch_question_mask, batch_question_mp):
        batch_question_t = self.l_stepwise_q_W[t](batch_question_mp).unsqueeze(1)

        batch_tokens_logit = self.W_q_att(batch_question_t * batch_question_vectors).squeeze(-1)
        
        batch_tokens_logit_masked = batch_tokens_logit.masked_fill(batch_question_mask, float('-inf'))
        batch_tokens_logit_softmax = F.softmax(batch_tokens_logit_masked, 1).unsqueeze(1)
        
        batch_question_selfatt = torch.matmul(batch_tokens_logit_softmax, batch_question_vectors).squeeze(1)
        
        return batch_question_selfatt

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
    
    def policy_linear(self, b_input_vector):
        X = self.W1(b_input_vector)
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X)
        return X2
    
    def get_transformer_rel_hidden(self, b_wiki_relation_lengths, b_wiki_relations_tokenIds, b_r_local):
        b_size, r_num = b_r_local.shape
        batch_wiki_rel_embeddings = self.word_embeddings(b_wiki_relations_tokenIds)
        batch_wiki_rel_mask = self.batch_sentence_mask(b_wiki_relation_lengths, func_type = "relation")

        transformer_output = self.wiki_encoding_model(batch_wiki_rel_embeddings.permute(1, 0 ,2), batch_wiki_rel_mask)
        transformer_output = transformer_output.permute(1, 0 ,2)
        wiki_rel_hidden_mp = self.question_max_pooling(transformer_output, batch_wiki_rel_mask)

        b_r_hidden_mp = torch.index_select(input = wiki_rel_hidden_mp, dim = 0, index = b_r_local.view(-1))

        b_r_hidden_mp = b_r_hidden_mp.view(b_size, r_num, -1)
        return self.W1Dropout(b_r_hidden_mp)
    
    def get_plm_rel_hidden(self, l_b_wiki_relations, b_r_local):
        b_size, r_num = b_r_local.shape
        encoded_input = self.tokenizer(l_b_wiki_relations, 
                padding = True,
                truncation = True,
                return_tensors = "pt")
            
        d_encoded_input = {}
        for k, v in encoded_input.items():
            d_encoded_input[k] = v.cuda()

        output = self.wiki_encoding_model(**d_encoded_input)
        last_hidden_states = output.last_hidden_state
        cls_hidden_states = last_hidden_states[:, 0, :]

        b_r_hidden_cls = torch.index_select(input = cls_hidden_states, dim = 0, index = b_r_local.view(-1))

        b_r_hidden_cls = b_r_hidden_cls.view(b_size, r_num, -1)
        return self.W1Dropout(b_r_hidden_cls)

    def get_wiki_relation_embedding(self, l_b_wiki_relations, b_wiki_relation_lengths, b_wiki_relations_tokenIds, b_r_local):
        if self.wiki_encoding_model_name == "Transformer":
            b_r_hidden_mp = self.get_transformer_rel_hidden( b_wiki_relation_lengths, b_wiki_relations_tokenIds, b_r_local)
            
            return b_r_hidden_mp
        
        else:
            b_r_hidden_cls = self.get_plm_rel_hidden(l_b_wiki_relations, b_r_local)

            return b_r_hidden_cls
            
    def get_action_embedding(self, action, reasoner, d_relation2id_wiki, d_relationid2text_wiki, d_word2id):
        b_r_global, b_e_global = action
        l_b_wiki_relations, b_wiki_relation_lengths, b_wiki_relations_tokenIds, b_r_local = wiki_relationid_to_name(b_r_global, d_relation2id_wiki, d_relationid2text_wiki, d_word2id)

        b_r_relation_hidden = self.get_wiki_relation_embedding(l_b_wiki_relations, b_wiki_relation_lengths, b_wiki_relations_tokenIds, b_r_local)

        with torch.no_grad():
            b_e_embeddings = reasoner.return_entity_kg_embeddings(b_e_global) 
        
        if self.relation_only:
            action_embedding = b_r_relation_hidden
        else:
            action_embedding = torch.cat([b_r_relation_hidden, b_e_embeddings], dim=-1)
            
        return action_embedding
    
    def wiki_transit(self, t, reasoner, e_t_global, batch_question, batch_question_seq_lengths, batch_path_hidden, d_relation2id_wiki, d_relationid2text_wiki, d_word2id):
        batch_path_hidden_ng = batch_path_hidden.detach()

        db_action_spaces, db_references = self.get_action_space_in_buckets(e_t_global, self.entity2bucketid_wiki, self.d_action_space_buckets_wiki)

        batch_question_vectors, batch_question_mask = self.get_question_representation(batch_question, batch_question_seq_lengths)

        batch_question_mp = self.question_max_pooling(batch_question_vectors, batch_question_mask)

        batch_question_selfatt = self.get_timestep_question_vector(t, batch_question_vectors, batch_question_mask, batch_question_mp)
        
        l_references = []
        l_batch_top_r_space = []
        l_batch_top_e_space_global = []
        l_batch_top_action_mask = []
        l_batch_top_action_dist_log = []

        for b_action_space, b_reference in zip(db_action_spaces, db_references):
            (b_r_space, b_e_space_global), b_action_mask = b_action_space

            b_question_selfatt = batch_question_selfatt[b_reference]
            b_path_hidden = batch_path_hidden_ng[b_reference]

            b_policy_network_input = torch.cat([b_path_hidden, b_question_selfatt], -1)

            b_policy_network_output = self.policy_linear(b_policy_network_input).unsqueeze(-1)

            b_action_embedding = self.get_action_embedding((b_r_space, b_e_space_global), reasoner, d_relation2id_wiki, d_relationid2text_wiki, d_word2id)

            b_action_logit = torch.matmul(b_action_embedding, b_policy_network_output).squeeze(-1)
            b_action_mask[:, 0] = 0

            b_action_logit_masked = b_action_logit.masked_fill((1- b_action_mask).bool(), float('-inf'))
            b_action_dist = F.softmax(b_action_logit_masked, 1)

            b_action_dist = b_action_dist.masked_fill(torch.isnan(b_action_dist), 0)
            b_top_action_dist, b_top_action_index = torch.topk(b_action_dist, self.extractor_topK)

            b_top_r_space = torch.gather(b_r_space, 1, b_top_action_index)
            b_top_e_space_global = torch.gather(b_e_space_global, 1, b_top_action_index)
            b_top_action_mask = torch.gather(b_action_mask, 1, b_top_action_index)

            b_top_action_dist_log = safe_log(b_top_action_dist)
            
            l_references.extend(b_reference)
            l_batch_top_r_space.append(b_top_r_space)
            l_batch_top_e_space_global.append(b_top_e_space_global)
            l_batch_top_action_mask.append(b_top_action_mask)
            l_batch_top_action_dist_log.append(b_top_action_dist_log)
        
        inv_offset = [i for i, _ in sorted(enumerate(l_references), key=lambda x: x[1])]

        batch_top_r_space = torch.cat(l_batch_top_r_space, dim=0)[inv_offset]
        batch_top_e_space_global = torch.cat(l_batch_top_e_space_global, dim=0)[inv_offset]
        batch_top_action_mask = torch.cat(l_batch_top_action_mask, dim=0)[inv_offset]
        batch_top_action_dist_log = torch.cat(l_batch_top_action_dist_log, dim=0)[inv_offset]

        return ((batch_top_r_space, batch_top_e_space_global), batch_top_action_mask), batch_top_action_dist_log

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.0)
        nn.init.xavier_uniform_(self.W_q_att.weight)
        nn.init.constant_(self.W_q_att.bias, 0.0)

        for i in range(self.max_hop):
            nn.init.xavier_uniform_(self.l_stepwise_q_W[i].weight)
            nn.init.constant_(self.l_stepwise_q_W[i].bias, 0.0)
    
    def batch_sentence_mask(self, batch_sent_len, func_type = "question"):
        batch_size = len(batch_sent_len)

        if func_type == "question":
            if self.use_ecm_tokens_internal_memory:
                max_sent_len = self.max_question_len_global
            else:
                max_sent_len = batch_sent_len[0]
                
        elif func_type == "relation":
            max_sent_len = batch_sent_len[0]

        mask = torch.zeros(batch_size, max_sent_len, dtype=torch.long)

        for i in range(batch_size):
            sent_len = batch_sent_len[i]
            mask[i][sent_len:] = 1
        
        mask = (mask == 1)
        mask = mask.cuda()
        return mask
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)