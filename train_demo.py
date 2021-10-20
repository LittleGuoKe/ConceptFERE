# import fastseq
from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTConceptSentenceEncoder, \
    BERTPAIRSentenceEncoder, \
    RobertaSentenceEncoder, RobertaPAIRSentenceEncoder, BERTPAIRConceptSentenceEncoder
# RobertaSentenceEncoder, RobertaPAIRSentenceEncoder

import models
from models.proto import Proto
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.d import Discriminator
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import os
from datetime import datetime
from fewshot_re_kit.conceptgraph_utils import loadingInstance2concept, loadingConceptGraphEntity2ID, load
from fewshot_re_kit.data_kg_loader import get_concept_loader_pair, get_concept_loader
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train',
                        help='train file')
    parser.add_argument('--val', default='val',
                        help='val file')
    parser.add_argument('--test', default='test_wiki',
                        help='test file')
    parser.add_argument('--adv', default=None,
                        help='adv file')
    parser.add_argument('--trainN', default=10, type=int,
                        help='N in train')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=5, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=5, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='proto',
                        help='model name')
    parser.add_argument('--encoder', default='cnn',
                        help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--lr', default=1e-1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
                        help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default='checkpoint/10way1shot.ConceptFere.pth.tar',
                        help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
                        help='only test')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
                        help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true',
                        help='use dot instead of L2 distance for proto')

    # experiment
    parser.add_argument('--mask_entity', action='store_true',
                        help='mask entity names')
    # concept
    parser.add_argument('--ins2cpt', default='conceptgraph/instance2concept',
                        help='instance2concept in conceptgraph file')

    # BeyondWordEmbedding
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--model_format', dest='model_format', default='bin', nargs='?', type=str)
    parser.add_argument('--model_file', dest='model_file', default='/home/LAB/zhaoqh/yangshan/kg/pretrainingConceptGraph/models/cme.bin', nargs='?',
                        type=str)
    parser.add_argument('--id_from', default='',
                        help='BeyondWordEmbedding Or keEmbedding Or MultiHeadAttentionAndBeyondWordEmbedding')
    parser.add_argument('--concept', action='store_true', help='use concept in kg(ConceptGraph)')
    parser.add_argument('--entity2id', default='conceptgraphEmbedding/TransE_l2_concetgraph_2/entities2id',
                        help='entity2id in conceptgraph file path')
    parser.add_argument('--word2id', default='BeyondWordEmbedding/word2id', help='word2id file path')
    parser.add_argument('--title2id', default='BeyondWordEmbedding/all_titles2id', help='title2id file path')

    # kg embedding
    parser.add_argument('--id2embeddingID', default='BeyondWordEmbedding/id2embeddingID', help='file path')
    parser.add_argument('--BeyondWordEmbedding', default='BeyondWordEmbedding/partOfBeyondWordEmbedding',
                        help='file path')
    parser.add_argument('--conceptEmbedding',
                        default='conceptgraphEmbedding/TransE_l2_concetgraph_2/concetgraph_TransE_l2_entity',
                        help='file path')

    parser.add_argument('--sentenceORword', default='sentence', help='select bert output')

    opt = parser.parse_args()

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    starting_time = datetime.now()
    print('starting time', starting_time)
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
            glove_mat,
            glove_word2id,
            max_length)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if opt.pair:

            # sentence_encoder = BERTPAIRSentenceEncoder(
            #     pretrain_ckpt,
            #     max_length)
            # titles, redirects, vector_size, W, id2word, word2id, all_titles = load(model_path=opt.model_file,
            #                                                                        format=opt.model_format,
            #                                                                        load_concepts=True,
            #                                                                        normalize=opt.normalize,
            #                                                                        log_every=1000000)

            if (opt.id_from == 'BeyondWordEmbedding') | (opt.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding'):
                with open('./data/BeyondWordEmbedding/id2embeddingID.json', mode='r', encoding='utf-8') as fr:
                    id2embeddingID = json.load(fr)
                BeyondWordEmbedding = np.loadtxt('./data/BeyondWordEmbedding/partOfBeyondWordEmbedding.npy')
                BeyondWordEmbedding = torch.from_numpy(BeyondWordEmbedding)

                # print('loading conceptEmbedding')
                # path = './data/conceptgraphEmbedding/TransE_l2_concetgraph_2/concetgraph_TransE_l2_entity.npy'
                # conceptEmbedding = np.load(path)
                # conceptEmbedding = torch.from_numpy(conceptEmbedding)
                conceptEmbedding = {}
                sentence_encoder = BERTPAIRConceptSentenceEncoder(
                    pretrain_ckpt,
                    max_length, conceptEmbedding, BeyondWordEmbedding, id2embeddingID,
                    id_from=opt.id_from)
            else:
                print('init BERTPAIRSentenceEncoder')
                sentence_encoder = BERTPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            if opt.concept | (opt.id_from == 'BeyondWordEmbedding') | (
                    opt.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding'):
                # print('concept: use')
                sentence_encoder = BERTConceptSentenceEncoder(pretrain_ckpt,
                                                              max_length, sentenceORword=opt.sentenceORword,
                                                              cat_entity_rep=opt.cat_entity_rep,
                                                              mask_entity=opt.mask_entity)
            else:
                sentence_encoder = BERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep,
                    mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'
        if opt.pair:
            sentence_encoder = RobertaPAIRSentenceEncoder(
                pretrain_ckpt,
                max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep)
    else:
        raise NotImplementedError

    if opt.pair:

        if (opt.id_from == 'keEmbedding') | (
                opt.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding') | (opt.id_from == 'BeyondWordEmbedding'):
            ins2cpt = loadingInstance2concept(path='./data/conceptgraph/instance2concept.pickle')
            entity2id = loadingConceptGraphEntity2ID(root='./data/')
            path1 = './data/BeyondWordEmbedding/word2id.json'
            with open(path1, mode='r', encoding='utf-8') as f1:
                word2id = json.load(f1)

            path2 = './data/BeyondWordEmbedding/all_titles2id.json'
            with open(path2, mode='r', encoding='utf-8') as f2:
                title2id = json.load(f2)

            train_data_loader = get_concept_loader_pair(opt.train, ins2cpt, entity2id, title2id, word2id,
                                                        sentence_encoder,
                                                        nWay=trainN, K=K, Q=Q, batch_size=batch_size, num_workers=0,
                                                        na_rate=opt.na_rate, encoder_name=encoder_name,
                                                        id_from=opt.id_from)
            val_data_loader = get_concept_loader_pair(opt.val, ins2cpt, entity2id, title2id, word2id, sentence_encoder,
                                                      nWay=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                                      num_workers=0,
                                                      encoder_name=encoder_name, id_from=opt.id_from)
            test_data_loader = get_concept_loader_pair(opt.test, ins2cpt, entity2id, title2id, word2id,
                                                       sentence_encoder,
                                                       num_workers=0,
                                                       nWay=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                                       encoder_name=encoder_name, id_from=opt.id_from)
        else:
            ins2cpt = loadingInstance2concept(path='./data/conceptgraph/instance2concept.pickle')
            train_data_loader = get_loader_pair(opt.train, ins2cpt,
                                                sentence_encoder,
                                                nWay=trainN, K=K, Q=Q, batch_size=batch_size, num_workers=8,
                                                na_rate=opt.na_rate, encoder_name=encoder_name)
            val_data_loader = get_loader_pair(opt.val, ins2cpt, sentence_encoder,
                                              nWay=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                              num_workers=8,
                                              encoder_name=encoder_name)
            test_data_loader = get_loader_pair(opt.test, ins2cpt,
                                               sentence_encoder,
                                               num_workers=8,
                                               nWay=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                               encoder_name=encoder_name)
    else:
        if opt.concept | (opt.id_from == 'keEmbedding') | (
                opt.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding') | (opt.id_from == 'BeyondWordEmbedding'):
            train_data_loader = get_concept_loader(opt.train, sentence_encoder,
                                                   nWay=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                                   ins2cpt=opt.ins2cpt, concept=opt.concept, id_from=opt.id_from,
                                                   entity2id=opt.entity2id, title2id=opt.title2id, word2id=opt.word2id)
            val_data_loader = get_concept_loader(opt.val, sentence_encoder,
                                                 nWay=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                                 ins2cpt=opt.ins2cpt, concept=opt.concept, id_from=opt.id_from,
                                                 entity2id=opt.entity2id, title2id=opt.title2id, word2id=opt.word2id)
            test_data_loader = get_concept_loader(opt.test, sentence_encoder,
                                                  nWay=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,
                                                  ins2cpt=opt.ins2cpt, concept=opt.concept, id_from=opt.id_from,
                                                  entity2id=opt.entity2id, title2id=opt.title2id, word2id=opt.word2id)
        else:
            train_data_loader = get_loader(opt.train, sentence_encoder,
                                           N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
            val_data_loader = get_loader(opt.val, sentence_encoder,
                                         N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
            test_data_loader = get_loader(opt.test, sentence_encoder,
                                          N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        if opt.adv:
            adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
                                                      N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv:
        d = Discriminator(opt.hidden_size)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader,
                                       adv=opt.adv, d=d)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)

    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'

    if model_name == 'proto':

        model = Proto(sentence_encoder, dot=opt.dot)
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, N, K, hidden_size=opt.hidden_size)
    elif model_name == 'metanet':
        model = MetaNet(N, K, sentence_encoder.embedding, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)

    # elif model_name=='concept':
    #     model=Pair(sentence_encoder,hidden_size=opt.hidden_sieze)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        print('-------------------------------------model.cuda()-------------------------', model.cuda())
        model.cuda()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        framework.train(device, model, prefix, batch_size, trainN, N, K, Q,
                        pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair,
                        train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(device, model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt,
                         pair=opt.pair)
    print("RESULT: %.2f" % (acc * 100))
    ending_time = datetime.now()
    print('ending time', ending_time)
    print('training takes time', ending_time - starting_time)


if __name__ == "__main__":
    main()
