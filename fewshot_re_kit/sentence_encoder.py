import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, \
    RobertaTokenizer, RobertaForSequenceClassification
from fewshot_re_kit.conceptgraph_utils import instance2conept, instance2coneptPlus


class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length,
                                                     word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim,
                                               pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

    def forward(self, inputs):
        if not self.cat_entity_rep:
            # _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
            x = self.bert(inputs['word'], attention_mask=inputs['mask'])[1]

            return x
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            # print('*' * 100)
            # print('inputs[word].size()', inputs['word'].size())
            # print('inputs[word].size()[0]', inputs['word'].size()[0])
            # print('tensor_range', tensor_range)
            # print('inputs["pos1"]', inputs["pos1"])
            # print('*' * 100)

            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (
                    pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask


class BERTConceptSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, sentenceORword, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.sentenceORword = sentenceORword

    def forward(self, inputs):
        if not self.cat_entity_rep:
            if self.sentenceORword == 'sentence':
                _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
                return x
            elif self.sentenceORword == 'word':

                wordEmbedding = self.bert(inputs['word'], attention_mask=inputs['mask'])[0]
                sentenceEmbedding = self.bert(inputs['word'], attention_mask=inputs['mask'])[1]

                return wordEmbedding, sentenceEmbedding
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (
                    pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

    def tokenize_concept(self, raw_tokens, pos_head, pos_tail, h, t, ins2cpt):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (
                    pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        '''添加实体的概念到tokens中'''
        h = h.lower()
        h2concept = instance2conept(ins2cpt, h)
        h2concept1 = h2concept[0].lower()
        h2concept2 = h2concept[1].lower()
        t = t.lower()
        t2concept = instance2conept(ins2cpt, t)
        t2concept1 = t2concept[0].lower()
        t2concept2 = t2concept[1].lower()

        tokens.append('[unused4]')
        if (h2concept1 == 'unknowconcept1') or (h2concept1 == 'unknowconcept2'):
            tokens.append(h2concept1)
        else:
            tokens += self.tokenizer.tokenize(h2concept1)

        tokens.append('[unused5]')
        if (h2concept2 == 'unknowconcept1') or (h2concept2 == 'unknowconcept2'):
            tokens.append(h2concept2)
        else:
            tokens += self.tokenizer.tokenize(h2concept2)

        tokens.append('[unused6]')
        if (t2concept1 == 'unknowconcept1') or (t2concept1 == 'unknowconcept2'):
            tokens.append(t2concept1)
        else:
            tokens += self.tokenizer.tokenize(t2concept1)

        tokens.append('[unused7]')
        if (t2concept2 == 'unknowconcept1') or (t2concept2 == 'unknowconcept2'):
            tokens.append(t2concept2)
        else:
            tokens += self.tokenizer.tokenize(t2concept2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask


class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrain_path,
            num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):

        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens

    def tokenize_concept(self, raw_tokens, pos_head, pos_tail, h, t, ins2cpt):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        '''添加实体的概念到tokens中'''
        h = h.lower()
        h2concept = instance2conept(ins2cpt, h)
        h2concept1 = h2concept[0].lower()
        h2concept2 = h2concept[1].lower()
        t = t.lower()
        t2concept = instance2conept(ins2cpt, t)
        t2concept1 = t2concept[0].lower()
        t2concept2 = t2concept[1].lower()

        tokens.append('[unused4]')
        if (h2concept1 == 'unknowconcept1') or (h2concept1 == 'unknowconcept2'):
            # print('-----------I am running-----------------------')
            tokens.append(h2concept1)
        else:
            tokens += self.tokenizer.tokenize(h2concept1)

        tokens.append('[unused5]')
        if (h2concept2 == 'unknowconcept1') or (h2concept2 == 'unknowconcept2'):
            tokens.append(h2concept2)
        else:
            tokens += self.tokenizer.tokenize(h2concept2)

        tokens.append('[unused6]')
        if (t2concept1 == 'unknowconcept1') or (t2concept1 == 'unknowconcept2'):
            tokens.append(t2concept1)
        else:
            tokens += self.tokenizer.tokenize(t2concept1)

        tokens.append('[unused7]')
        if (t2concept2 == 'unknowconcept1') or (t2concept2 == 'unknowconcept2'):
            tokens.append(t2concept2)
        else:
            tokens += self.tokenizer.tokenize(t2concept2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return indexed_tokens

    def tokenize_concept_plus(self, raw_tokens, pos_head, pos_tail, h, t, ins2cpt):
        # token -> index
        # tokens = ['[CLS]']

        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        '''添加实体的概念到tokens中'''
        h = h.lower()
        h2concept = instance2coneptPlus(ins2cpt, h)
        t = t.lower()
        t2concept = instance2coneptPlus(ins2cpt, t)

        tokens.append('[unused4]')
        for cpt in h2concept:
            if cpt == 'unknowConcept':
                tokens.append(cpt)
            else:
                tokens += self.tokenizer.tokenize(cpt)

        tokens.append('[unused5]')
        for cpt in t2concept:
            if cpt == 'unknowConcept':
                tokens.append(cpt)
            else:
                tokens += self.tokenizer.tokenize(cpt)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return indexed_tokens


class BERTPAIRConceptSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, conceptEmbedding, BeyondWordEmbedding, id2embeddingID,
                 id_from='kgEmbeddingOrBeyondWordEmbedding'):
        nn.Module.__init__(self)
        print('----------------------BERTPAIRConceptSentenceEncoder initializing----------------------------------')
        self.bert = BertModel.from_pretrained(
            pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.conceptEmbedding = conceptEmbedding
        self.beyondWordEmbedding = BeyondWordEmbedding  # 词向量500维
        self.id2embeddingID = id2embeddingID
        self.id_from = id_from

        if self.id_from == 'keEmbedding':
            self.projector1 = torch.nn.Linear(256, 768)
            self.projector2 = torch.nn.Linear(768, 768)
            self.fusionLayer = torch.nn.MultiheadAttention(embed_dim=768, num_heads=12, dropout=0.1)
            self.classifier = torch.nn.Linear(136 * 768, 2)

        elif self.id_from == 'BeyondWordEmbedding':
            self.projector1 = torch.nn.Linear(500, 128)
            self.projector2 = torch.nn.Linear(768, 128)

            self.classifier = torch.nn.Linear(1792, 2)
        elif self.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding':
            word_dim = 768
            self.concept_project = True
            # word_dim = 500
            self.projector1 = torch.nn.Linear(500, word_dim)  # 对概念向量投影
            self.projector2 = torch.nn.Linear(768, word_dim)  # 对句子向量投影
            # self.projector3 = torch.nn.Linear(768, 120)

            self.fusionLayer = torch.nn.MultiheadAttention(embed_dim=word_dim, num_heads=12, dropout=0.1)
            self.classifier = torch.nn.Linear((self.max_length + 8) * word_dim, 2)
        else:
            assert ('please input right id source')

    def forward(self, inputs):

        '''计算query句子与其实体对应的概念相似度'''
        query_sen = self.bert(inputs['query_sen'], attention_mask=inputs['query_mask'])[1]
        queryConceptID = inputs['queryConceptID']
        quer_sen_hava_cpt = self.sentence_have_concept(query_sen, queryConceptID)
        '''计算support句子与其实体对应的概念相似度'''
        support_sen = self.bert(inputs['support_sen'], attention_mask=inputs['support_mask'])[1]
        supportConceptID = inputs['supportConceptID']
        support_sen_have_cpt = self.sentence_have_concept(support_sen, supportConceptID)

        '''句子和概念embedding拼接'''
        if self.id_from == 'keEmbedding':
            x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[1]
            sen_cpt_vec = self.sen_pair_cat_cpt(x, quer_sen_hava_cpt,
                                                support_sen_have_cpt)  # sen_cpt_vec shape:(-1,768+128*8=1792)
            sen_cpt_vec = self.fusionLayer(sen_cpt_vec)
            x = self.classifier(sen_cpt_vec)
        elif self.id_from == 'BeyondWordEmbedding':
            x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[1]
            sen_cpt_vec = self.sen_pair_cat_cpt(x, quer_sen_hava_cpt,
                                                support_sen_have_cpt)  # sen_cpt_vec shape:(-1,768+500*8=4768)
            # sen_cpt_vec = self.fusionLayer(sen_cpt_vec)
            x = self.classifier(sen_cpt_vec)
        elif self.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding':
            x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[
                0]  # 取出词向量矩阵[-1,单词个数（句子长度），词向量的维度]
            # word_num = x.shape[1]
            x = self.pair_projecter(x)
            sen_cpt_vec = self.sen_pair_cat_cpt(x, quer_sen_hava_cpt,
                                                support_sen_have_cpt)

            sen_cpt_vec, _ = self.fusionLayer(sen_cpt_vec, sen_cpt_vec, sen_cpt_vec)

            # print('----------------------------sen_cpt_vec.shape----------------------')
            # print(sen_cpt_vec.shape)
            # sen_cpt_vec = self.projector3(sen_cpt_vec)
            # sen_cpt_vec = sen_cpt_vec.reshape(-1, (word_num + 8) * 384)
            sen_cpt_vec = sen_cpt_vec.reshape(-1, sen_cpt_vec.shape[1] * sen_cpt_vec.shape[2])

            # print('---------------------sen_cpt_vec.shape-----------------',sen_cpt_vec.shape)
            x = self.classifier(sen_cpt_vec)

        return x

    def pair_projecter(self, wordVec):

        sample_num = wordVec.shape[0]
        word_num = wordVec.shape[1]
        sen_len = self.projector2.bias.shape[0]

        projected_wordVec = torch.zeros([sample_num, word_num, sen_len]).cuda()
        for i in range(sample_num):
            wv = wordVec[i, :, :]
            wv = self.projector2(wv)
            projected_wordVec[i, :, :] = wv
        # print('projected_wordVec.shape', projected_wordVec.shape)
        return projected_wordVec

    def sentence_have_concept(self, sen, conceptID):
        '''
        计算句子与concept的相似度，值为0或1,返回value为1的concept embedding
        '''

        sen_have_cpt = []  # 用于存储根据句意选择的实体
        sen_num = sen.shape[0]
        sen_len = sen.shape[1]  # 768

        for i in range(sen_num):
            if self.id_from == 'kgEmbedding':
                '''获取概念向量'''
                cptID = conceptID[i]
                h_cpt1ID = cptID[0]  # TOdo 转换成int
                h_cpt2ID = cptID[1]
                t_cpt1ID = cptID[2]
                t_cpt2ID = cptID[3]

                h_cpt1_vec = self.id2embedding(h_cpt1ID, self.conceptEmbedding).cuda()
                h_cpt2_vec = self.id2embedding(h_cpt2ID, self.conceptEmbedding).cuda()
                t_cpt1_vec = self.id2embedding(t_cpt1ID, self.conceptEmbedding).cuda()
                t_cpt2_vec = self.id2embedding(t_cpt2ID, self.conceptEmbedding).cuda()
                '''获取句子向量'''
                sen_vec = sen[i, :]
                '''概念向量投影'''
                h_cpt1_vec = self.projector1(h_cpt1_vec)  # size 由(1,256)变成(1，128)
                h_cpt2_vec = self.projector1(h_cpt2_vec)
                t_cpt1_vec = self.projector1(t_cpt1_vec)
                t_cpt2_vec = self.projector1(t_cpt2_vec)
                '''句子向量投影'''
                sen_vec = self.projector2(sen_vec)
            elif (self.id_from == 'BeyondWordEmbedding') | (self.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding'):
                '''获取概念向量'''
                cptID = conceptID[i]
                all_cpt_vec = self.id2BeyondWordEmbedding(cptID)
                h_cpt1_vec = all_cpt_vec[0]
                h_cpt2_vec = all_cpt_vec[1]
                t_cpt1_vec = all_cpt_vec[2]
                t_cpt2_vec = all_cpt_vec[3]
                '''获取句子向量'''
                sen_vec = sen[i, :]

                # '''句子向量投影'''
                # sen_vec = self.projector2(sen_vec)
                '''概念向量投影'''
                if self.concept_project:
                    h_cpt1_vec = self.projector1(h_cpt1_vec)  # size 由(1,500)变成(1，768)
                    h_cpt2_vec = self.projector1(h_cpt2_vec)
                    t_cpt1_vec = self.projector1(t_cpt1_vec)
                    t_cpt2_vec = self.projector1(t_cpt2_vec)
                else:
                    '''句子向量投影'''
                    sen_vec = self.projector2(sen_vec)

            '''计算句子和概念的相似度'''
            h_cpt1_sen_sim = torch.matmul(sen_vec, h_cpt1_vec.t()).float()  # size(1),计算结果为标量
            h_cpt2_sen_sim = torch.matmul(sen_vec, h_cpt2_vec.t()).float()
            t_cpt1_sen_sim = torch.matmul(sen_vec, t_cpt1_vec.t()).float()
            t_cpt2_sen_sim = torch.matmul(sen_vec, t_cpt2_vec.t()).float()
            '''相似度01化,相似度的值只取0或1'''
            # softmax
            sim = torch.tensor([h_cpt1_sen_sim, h_cpt2_sen_sim, t_cpt1_sen_sim, t_cpt2_sen_sim])
            sim = torch.softmax(sim, dim=0)
            [h_cpt1_sen_sim, h_cpt2_sen_sim, t_cpt1_sen_sim, t_cpt2_sen_sim] = sim
            '''O1-GATE threshold, alpha: 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1'''
            threshold_alpha = 0.3
            h_cpt1_sen_sim = 1 if h_cpt1_sen_sim >= threshold_alpha else 0
            h_cpt2_sen_sim = 1 if h_cpt2_sen_sim >= threshold_alpha else 0
            t_cpt1_sen_sim = 1 if t_cpt1_sen_sim >= threshold_alpha else 0
            t_cpt2_sen_sim = 1 if t_cpt2_sen_sim >= threshold_alpha else 0

            h_cpt1_vec = h_cpt1_vec * h_cpt1_sen_sim
            h_cpt2_vec = h_cpt2_vec * h_cpt2_sen_sim
            t_cpt1_vec = t_cpt1_vec * t_cpt1_sen_sim
            t_cpt2_vec = t_cpt2_vec * t_cpt2_sen_sim
            sen_have_cpt.append([h_cpt1_vec, h_cpt2_vec, t_cpt1_vec, t_cpt2_vec])

        return sen_have_cpt

    def id2BeyondWordEmbedding(self, cptID):
        '''
        cptID = torch.tensor([[9.6963e+06, -2.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00],
                              [-1.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00],
                              [-1.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00],
                              [-1.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00, -2.0000e+00]])
        '''
        cpt_num = cptID.shape[0]
        all_cpt_id = []
        all_cpt_vec = []
        for i in range(cpt_num):
            wordsID = cptID[i]
            cpt_id = []
            for id in wordsID:
                if id == -2:  # -2为无效ID，填充向量用的
                    continue
                else:
                    cpt_id.append(id)
            all_cpt_id.append(cpt_id)
        for j in range(cpt_num):  # 概念只有一个词组成
            j_cpt_id = all_cpt_id[j]
            if len(j_cpt_id) == 1:
                j_id = j_cpt_id[0]
                if j_id == -1:
                    cpt_vec = torch.zeros((1, 500)).float().cuda()
                    all_cpt_vec.append(cpt_vec)
                else:
                    # j_id = j_id.long()

                    j_id = j_id.cpu().numpy()
                    j_id = str(j_id)
                    j_id = j_id.split('.')[0]
                    j_id = self.id2embeddingID[j_id]

                    cpt_vec = self.beyondWordEmbedding[j_id, :].float().cuda()
                    cpt_vec = cpt_vec.view([1, 500])
                    all_cpt_vec.append(cpt_vec)
            else:
                for id in j_cpt_id:  # 概念由多个词组成，把词的向量叠加，作为概念的向量
                    word_vec = torch.zeros((1, 500)).float().cuda()
                    if id == -1:
                        cpt_vec = torch.zeros((1, 500)).float().cuda()
                        word_vec = word_vec + cpt_vec
                    else:
                        # id = id.long()

                        id = id.cpu().numpy()
                        id = str(id)
                        id = id.split('.')[0]
                        id = self.id2embeddingID[id]

                        cpt_vec = self.beyondWordEmbedding[id, :].float().cuda()
                        cpt_vec = cpt_vec.view([1, 500])
                        word_vec = word_vec + cpt_vec
                    # all_cpt_vec.append(word_vec)
                all_cpt_vec.append(word_vec)
        return all_cpt_vec

    def id2embedding(self, id, conceptEmbedding):
        if id == -1:
            cpt_vec = torch.zeros((1, 256))
        else:
            cpt_vec = conceptEmbedding[id, :]
            cpt_vec = cpt_vec.view([1, 256])

        return cpt_vec

    def sen_pair_cat_cpt(self, sen_pair, query_sen_hava_cpt, support_sen_have_cpt):
        if self.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding':

            sen_num = sen_pair.shape[0]
            word_num = sen_pair.shape[1]
            sen_len = sen_pair.shape[2]
            cpt_num = 8
            sen_cpt_vec = torch.zeros([sen_num, word_num + cpt_num, sen_len]).cuda()

            # sen_shape = [sen_pair.shape[1], sen_pair.shape[2]]
            for i in range(sen_num):
                sen_vec = sen_pair[i, :, :]
                # sen_vec = sen_vec.view(sen_shape)
                query_cpt_vec = query_sen_hava_cpt[i]
                query_h_cpt1_vec = query_cpt_vec[0]
                query_h_cpt2_vec = query_cpt_vec[1]
                query_t_cpt1_vec = query_cpt_vec[2]
                query_t_cpt2_vec = query_cpt_vec[3]
                support_cpt_vec = support_sen_have_cpt[i]
                support_h_cpt1_vec = support_cpt_vec[0]
                support_h_cpt2_vec = support_cpt_vec[1]
                support_t_cpt1_vec = support_cpt_vec[2]
                support_t_cpt2_vec = support_cpt_vec[3]
                i_sen_cpt_vec = torch.cat(
                    (sen_vec, query_h_cpt1_vec, query_h_cpt2_vec, query_t_cpt1_vec, query_t_cpt2_vec,
                     support_h_cpt1_vec, support_h_cpt2_vec, support_t_cpt1_vec, support_t_cpt2_vec),
                    0)

                sen_cpt_vec[i, :, :] = i_sen_cpt_vec.cuda()
        else:
            sen_cpt_vec = []
            # print(sen_cpt_vec)
            sen_num = sen_pair.shape[0]
            sen_len = sen_pair.shape[1]
            for i in range(sen_num):
                sen_vec = sen_pair[i, :]
                sen_vec = sen_vec.view([1, sen_len])
                query_cpt_vec = query_sen_hava_cpt[i]
                query_h_cpt1_vec = query_cpt_vec[0]
                query_h_cpt2_vec = query_cpt_vec[1]
                query_t_cpt1_vec = query_cpt_vec[2]
                query_t_cpt2_vec = query_cpt_vec[3]
                support_cpt_vec = support_sen_have_cpt[i]
                support_h_cpt1_vec = support_cpt_vec[0]
                support_h_cpt2_vec = support_cpt_vec[1]
                support_t_cpt1_vec = support_cpt_vec[2]
                support_t_cpt2_vec = support_cpt_vec[3]
                i_sen_cpt_vec = torch.cat(
                    (sen_vec, query_h_cpt1_vec, query_h_cpt2_vec, query_t_cpt1_vec, query_t_cpt2_vec,
                     support_h_cpt1_vec, support_h_cpt2_vec, support_t_cpt1_vec, support_t_cpt2_vec),
                    1)
                sen_cpt_vec.append(i_sen_cpt_vec)
            sen_cpt_vec = torch.cat(sen_cpt_vec, 0)

        return sen_cpt_vec

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens

    def tokenize_concept(self, raw_tokens, pos_head, pos_tail, h, t, ins2cpt):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        '''添加实体的概念到tokens中'''
        h = h.lower()
        h2concept = instance2conept(ins2cpt, h)
        h2concept1 = h2concept[0].lower()
        h2concept2 = h2concept[1].lower()
        t = t.lower()
        t2concept = instance2conept(ins2cpt, t)
        t2concept1 = t2concept[0].lower()
        t2concept2 = t2concept[1].lower()

        tokens.append('[unused4]')
        if (h2concept1 == 'unknowconcept1') or (h2concept1 == 'unknowconcept2'):
            tokens.append(h2concept1)
        else:
            tokens += self.tokenizer.tokenize(h2concept1)

        tokens.append('[unused5]')
        if (h2concept2 == 'unknowconcept1') or (h2concept2 == 'unknowconcept2'):
            tokens.append(h2concept2)
        else:
            tokens += self.tokenizer.tokenize(h2concept2)

        tokens.append('[unused6]')
        if (t2concept1 == 'unknowconcept1') or (t2concept1 == 'unknowconcept2'):
            tokens.append(t2concept1)
        else:
            tokens += self.tokenizer.tokenize(t2concept1)

        tokens.append('[unused7]')
        if (t2concept2 == 'unknowconcept1') or (t2concept2 == 'unknowconcept2'):
            tokens.append(t2concept2)
        else:
            tokens += self.tokenizer.tokenize(t2concept2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return indexed_tokens


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False):
        nn.Module.__init__(self)
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.cat_entity_rep = cat_entity_rep

    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        sst = ['<s>'] + sst
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(1)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask


class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            pretrain_path,
            num_labels=2)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens
