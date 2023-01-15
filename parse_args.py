import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                        type=str,
                        default = "WebQSP",
                        help="dataset for training")

argparser.add_argument('--gpu', type=int, default=0,
                    help='gpu device')

argparser.add_argument('--num_workers', type=int, default=4, help="Dataloader workers")

argparser.add_argument('--grid_search', action='store_true',
                    help='Conduct grid search of hyperparameters')

argparser.add_argument('--train', action='store_true',
                    help='train model')

argparser.add_argument('--eval', action='store_true',
                    help='evaluate the results on the test set')   

argparser.add_argument('--pretrain_reasoner', action='store_true',
                    help='pretrain reasoner model')

argparser.add_argument('--pretrain_extractor', action='store_true',
                    help='pretrain extractor model')

argparser.add_argument('--collaborate_training', action='store_true',
                    help='collaborate training reasoner and extractor')

argparser.add_argument('--total_epoch', type=int, default=20,
                    help='adversarial learning epoch number.')

argparser.add_argument("--min_freq", type=int, default=0, help="Minimum frequency for words")

argparser.add_argument("--max_question_len_global", type=int, default=20, help="Maximum question pattern words length")

argparser.add_argument("--hugging_face_path",
                        type=str,
                        default = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/HuggingFace",
                        help="hugging_face path")

argparser.add_argument("--max_hop",
                        type=int,
                        default=3,
                        help="max reasoning hop")

argparser.add_argument("--num_wait_epochs",
                        type=int,
                        default=1,
                        help="valid wait epochs")

argparser.add_argument('--entity_dim', type=int, default=200,
                    help='entity embedding dimension')

argparser.add_argument('--relation_dim', type=int, default=200,
                    help='relation embedding dimension')

argparser.add_argument('--word_dim', type=int, default=300,
                    help='word embedding dimension')

argparser.add_argument('--word_dropout_rate', type=float, default=0.3,
                    help='word embedding dropout rate')

argparser.add_argument('--word_padding_idx', type=int, default=0,
                    help='word padding index')

argparser.add_argument('--DUMMY_RELATION_idx', type=int, default=0,
                    help='DUMMY_RELATION index')

argparser.add_argument('--DUMMY_ENTITY_idx', type=int, default=0,
                    help='DUMMY_ENTITY index')  

argparser.add_argument('--is_train_emb', type=bool, default=True,
                    help='train word/entity/relation embedding or not')

argparser.add_argument('--grad_norm', type=float, default=50,
                    help='norm threshold for gradient clipping')

argparser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate')

argparser.add_argument('--head_num', type=int, default=4,
                    help='Transformer head number')

argparser.add_argument('--hidden_dim', type=int, default=100,
                    help='Transformer hidden dimension')

argparser.add_argument('--encoder_layers', type=int, default=2,
                    help='Transformer encoder layers number')

argparser.add_argument('--encoder_dropout_rate', type=float, default=0.3,
                    help='Transformer encoder dropout rate')

argparser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay rate')

argparser.add_argument('--history_dim', type=int, default=200,
                    help='path encoder LSTM hidden dimension')

argparser.add_argument('--relation_only', type=bool, default=False,
                    help='search with relation information only, ignoring entity representation')

argparser.add_argument('--rl_dropout_rate', type=float, default=0.3,
                    help='reinforce learning dropout rate')

argparser.add_argument('--history_layers', type=int, default=2,
                    help='path encoder LSTM layers')

argparser.add_argument('--gamma', type=float, default=0.95,
                    help='moving average weight') 

argparser.add_argument('--tau', type=float, default=1.00,
                    help='GAE tau')

argparser.add_argument("--early_stop_patience", type=int, default=10,
                        help="early stop epoch")

argparser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')

argparser.add_argument('--bucket_interval', type=int, default=20,
                    help='adjacency list bucket size')

argparser.add_argument('--batch_size', type=int, default=256,
                    help='mini-batch size')

argparser.add_argument('--beam_size_train', type=int, default=5, help='size of beam used in train')

argparser.add_argument('--beam_size_inference', type=int, default=5, help='size of beam used in inference')

argparser.add_argument('--keep_wiki_path_num', type=int, default=2, help='adaptive sampling how many wiki path')

argparser.add_argument('--use_entity_embedding_vn', type=bool, default=True, help='use entity embedding in value netwok or not')

argparser.add_argument('--use_actor_critic', type=bool, default=True, help='use actor critic optimization.')

argparser.add_argument('--use_gae', type=bool, default=True, help='use gae in actor critic optimization.')

argparser.add_argument('--use_ecm_tokens_internal_memory', type=bool, default= True, help='use emotional chatting machine tokens internal memory.')

argparser.add_argument('--use_tokens_memory_normalization', type=bool, default= True, help='use tokens memory normalization.')

argparser.add_argument('--loss_with_beam_reasoner', type=bool, default = True, help='calculate full beam size reasoner loss not top-1.')

argparser.add_argument('--loss_with_beam_extractor', type=bool, default = True, help='calculate full beam size extractor loss not top-1.')

argparser.add_argument('--seed', type=int, default=2022, help='random seed')

argparser.add_argument('--value_loss_coef', type=float, default=0.1,
                    help = "value loss coefficient")

argparser.add_argument("--wiki_encoding_model_name",
                        type=str,
                        default = 'Transformer',
                        help="wiki relation encoding model")

argparser.add_argument('--extractor_topK', type=int, default=5, help='Top K actions from extractor')

argparser.add_argument('--kb_triples_percentage', type=float, default=1.0,
                    help = "kg triples percentage used for training")

argparser.add_argument('--wiki_triples_percentage', type=float, default=1.0,
                    help = "wiki triples percentage used for training")

argparser.add_argument('--qa_data_percentage', type=float, default=1.0,
                    help = "qa data percentage used for training")

args = argparser.parse_args()