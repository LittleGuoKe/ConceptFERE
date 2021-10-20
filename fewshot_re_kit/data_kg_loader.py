import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import argparse
from fewshot_re_kit.sentence_encoder import BERTPAIRSentenceEncoder, BERTConceptSentenceEncoder
from fewshot_re_kit.conceptgraph_utils import loadingInstance2concept, loadingConceptGraphEntity2ID, instance2conept

import pickle
from tqdm import tqdm


class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name, ins2cpt, entity2id, title2id, word2id,
                 id_from='kgEmbeddingOrBeyondWordEmbedding'):
        self.root = root
        path = os.path.join(root, name + ".json")
        # print('file path', path)

        if not os.path.exists(path):
            print('file path', path)
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length
        self.ins2cpt = ins2cpt
        self.entity2id = entity2id
        self.title2id = title2id
        self.word2id = word2id
        self.id_from = id_from
        if self.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding':
            self.id_from = 'BeyondWordEmbedding'

    def __getraw__(self, item, ins2cpt):
        # word = self.encoder.tokenize(item['tokens'],
        #                              item['h'][2][0],
        #                              item['t'][2][0])

        word = self.encoder.tokenize_concept(item['tokens'],
                                             item['h'][2][0],
                                             item['t'][2][0],
                                             item['h'][0],
                                             item['t'][0],
                                             ins2cpt)

        return word

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        support_entitis = []  # [(h1,t1),(h2,t2)...]
        query_entities = []
        # support_sentence = []
        # query_sentence = []
        entities = []
        # fusion_set = {'word': [], 'mask': [], 'seg': [], 'entities': []}
        # fusion_set = {'word': [], 'mask': [], 'seg': []}
        fusion_set = {'word': [], 'mask': [], 'seg': [], 'query_sen': [], 'support_sen': [],
                      'query_mask': [], 'support_mask': [], 'queryConceptID': [], 'supportConceptID': []}

        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word = self.__getraw__(
                    self.json_data[class_name][j], self.ins2cpt)
                if count < self.K:
                    support.append(word)
                    '''获取头实体尾实体'''
                    item = self.json_data[class_name][j]
                    h = item['h'][0]
                    t = item['t'][0]
                    support_entitis.append((h, t))
                    # support_sentence.append(self.json_data[class_name][j])
                else:
                    query.append(word)
                    '''获取头实体尾实体'''
                    item = self.json_data[class_name][j]
                    h = item['h'][0]
                    t = item['t'][0]
                    query_entities.append((h, t))
                    # query_sentence.append(self.json_data[class_name][j])
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word = self.__getraw__(
                self.json_data[cur_class][index], self.ins2cpt)
            query.append(word)
        query_label += [self.N] * Q_na

        '''验证句子是否和实体一一对应'''
        # for m,s1 in enumerate(query_sentence):
        #     for n,s2 in enumerate(support_sentence):
        #         print('-------------------分割线------------------------')
        #         query_h_t = query_entities[m]
        #         support_h_t = support_entitis[n]
        #         print((query_h_t, support_h_t))
        #         print(s1)
        #         print(s2)
        #         print('-------------------分割线------------------------')

        for m, word_query in enumerate(query):
            for n, word_support in enumerate(support):
                # print(m, n)

                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                    word_query_tensor = torch.zeros((self.max_length)).long()
                    word_support_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

                q_w = CLS + word_query + SEP
                s_w = CLS + word_support + SEP

                for i in range(min(self.max_length, len(q_w))):
                    word_query_tensor[i] = q_w[i]

                for i in range(min(self.max_length, len(s_w))):
                    word_support_tensor[i] = s_w[i]
                '''将查询集中的句子和支撑集中的句子分别存储'''
                query_mask_tensor = torch.zeros((self.max_length)).long()
                support_mask_tensor = torch.zeros((self.max_length)).long()

                query_mask_tensor[:min(self.max_length, len(q_w))] = 1
                support_mask_tensor[:min(self.max_length, len(s_w))] = 1

                fusion_set['query_sen'].append(word_query_tensor)
                fusion_set['support_sen'].append(word_support_tensor)

                fusion_set['query_mask'].append(query_mask_tensor)
                fusion_set['support_mask'].append(support_mask_tensor)

                query_h_t = query_entities[m]
                support_h_t = support_entitis[n]

                '''返回实体对中的实体在concpetgraph中的概念的id'''
                queryConceptID = self.entityPair2concept2id(query_h_t, id_from=self.id_from)
                supportConceptID = self.entityPair2concept2id(support_h_t, id_from=self.id_from)
                queryConceptIDtensor, supportConceptIDtensor = self.conceptID2tensor(queryConceptID, supportConceptID,
                                                                                     id_from=self.id_from)
                fusion_set['queryConceptID'].append(queryConceptIDtensor)
                fusion_set['supportConceptID'].append(supportConceptIDtensor)
        return fusion_set, query_label

    def __len__(self):
        return 1000000000

    def conceptID2tensor(self, queryConceptID, supportConceptID, id_from='kgEmbeddingOrBeyondWordEmbedding'):
        '''
        给定查询集和支撑集句子中的概念ID或ID list返回其对应的tensor
        #Para:id_from: id源于预训练的kg embedding中词的id 或者源于论文[1]中预训练词向量word2id中词的id
         References:
            [1]Beyond Word Embeddings: Learning Entity and Concept Representations from Large Scale Knowledge Bases
        '''
        if id_from == 'kgEmbedding':
            queryConceptIDtensor = torch.zeros(4).long()
            supportConceptIDtensor = torch.zeros(4).long()

            for p in range(len(queryConceptID)):
                queryConceptIDtensor[p] = queryConceptID[p]

            for q in range(len(supportConceptID)):
                supportConceptIDtensor[q] = supportConceptID[q]
        elif id_from == 'BeyondWordEmbedding':
            # 概念中的词最多12个，平均值为2.026551292913871
            queryConceptIDtensor = torch.ones([4, 12]) * -2  # 初始化值为-2的tensor,查询集每个句子头尾实体对应四个概念，每个概念最多对应五个词
            supportConceptIDtensor = torch.ones([4, 12]) * -2
            for i in range(len(queryConceptID)):
                for j in range(len(queryConceptID[i])):
                    try:
                        queryConceptIDtensor[i, j] = queryConceptID[i][j]
                    except Exception as e:
                        print('-------------queryConceptID-----------')
                        print(queryConceptID)
                        print(i, j)

            for i in range(len(supportConceptID)):
                for j in range(len(supportConceptID[i])):
                    try:
                        supportConceptIDtensor[i, j] = supportConceptID[i][j]
                    except Exception as e:
                        print('-------------------supportConceptID-----------------')
                        print('supportConceptID', supportConceptID)
                        print(i, j)
        else:
            print('please input right id source')
        return queryConceptIDtensor, supportConceptIDtensor

    def entityPair2concept2id(self, entityPair, id_from='kgEmbeddingOrBeyondWordEmbedding'):
        '''
        返回实体对中的实体在concpetgraph中的概念的id，[h1cpt1toID,h2cpt2toID,t1cpt2toID,t2cpt2toID]
        h1cpt1toID：头实体对应的概念1对应的ID 或 ID list
        #Para:id_from: id源于预训练的kg embedding中词的id 或者源于论文[1]中预训练词向量word2id中词的id
        References:
            [1]Beyond Word Embeddings: Learning Entity and Concept Representations from Large Scale Knowledge Bases
        '''
        id = []

        for i in range(len(entityPair)):
            entity = entityPair[i]
            h2concept1toid, h2concept2toid = self.entity2concept2id(entity, id_from)
            id.append(h2concept1toid)
            id.append(h2concept2toid)
        return id

    def entity2concept2id(self, entity, id_from):
        entity = entity.lower()
        e2concept = instance2conept(self.ins2cpt, entity)
        e2concept1 = e2concept[0].lower()
        e2concept2 = e2concept[1].lower()

        if (e2concept1 == 'unknowconcept1') or (e2concept1 == 'unknowconcept2'):

            if id_from == 'kgEmbedding':
                e2concept1toid = -1
            elif id_from == 'BeyondWordEmbedding':
                e2concept1toid = [-1]
            else:
                print('please input right id source')
        else:
            if id_from == 'kgEmbedding':
                e2concept1toid = self.entity2id[e2concept1]
            elif id_from == 'BeyondWordEmbedding':
                e2concept1toid = self.id_from_BeyondWordEmbedding(e2concept1)
            else:
                print('please input right id source')

        if (e2concept2 == 'unknowconcept1') or (e2concept2 == 'unknowconcept2'):
            if id_from == 'kgEmbedding':
                e2concept2toid = -1
            elif id_from == 'BeyondWordEmbedding':
                e2concept2toid = [-1]
            else:
                print('please input right id source')
        else:
            if id_from == 'kgEmbedding':
                e2concept2toid = self.entity2id[e2concept2]
            elif id_from == 'BeyondWordEmbedding':
                e2concept2toid = self.id_from_BeyondWordEmbedding(e2concept2)
            else:
                print('please input right id source')

        return e2concept1toid, e2concept2toid

    def id_from_BeyondWordEmbedding(self, concept: str) -> list:
        '''
        输入概念，优先获得其在title2id中的id，如果title2id的key中没有concept,把concept进行分词，获取每个词在word2id中的id,返回ID list
        '''
        id = []
        concept = concept.lower()
        cptID = self.title2id.get(concept)
        if cptID == None:
            cpt_words = concept.split(' ')
            for word in cpt_words:
                w2id = self.word2id.get(word)
                if w2id == None:
                    w2id = -1
                    id.append(w2id)
                else:
                    id.append(w2id)
        else:
            cptID = cptID[0]
            cptID = self.word2id.get(cptID)
            if cptID == None:
                cptID = -1
                id.append(cptID)
            else:
                id.append(cptID)
        return id


def collate_fn_pair(data):
    batch_set = {'word': [], 'mask': [], 'seg': [], 'query_sen': [], 'support_sen': [],
                 'query_mask': [], 'support_mask': [], 'queryConceptID': [], 'supportConceptID': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)

    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_set, batch_label


def get_concept_loader_pair(name, ins2cpt, entity2id, title2id, word2id, encoder, nWay, K, Q, batch_size,
                            num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='./data',
                            encoder_name='bert', id_from='kgEmbeddingOrBeyondWordEmbedding'):
    dataset = FewRelDatasetPair(name, encoder, nWay, K, Q, na_rate, root, encoder_name, ins2cpt, entity2id, title2id,
                                word2id, id_from)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset+conceptgraph
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, ins2cpt, concept, id_from, entity2id, title2id, word2id):
        self.root = root
        path = os.path.join(root, name + ".json")

        concept_path = os.path.join(root, ins2cpt + '.pickle')
        entity2id_path = os.path.join(root, entity2id + '.pickle')
        title2id_path = os.path.join(root, title2id + '.json')
        word2id_path = os.path.join(root, word2id + '.json')
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!", path)
            assert (0)
        if not os.path.exists(concept_path):
            print("[ERROR] Data file does not exist!", concept_path)
            assert (0)
        if not os.path.exists(entity2id_path):
            print("[ERROR] Data file does not exist!", entity2id_path)
            assert (0)
        if not os.path.exists(title2id_path):
            print("[ERROR] Data file does not exist!", title2id_path)
            assert (0)
        if not os.path.exists(word2id_path):
            print("[ERROR] Data file does not exist!", word2id_path)
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.ins2cpt = loadingInstance2concept(concept_path)
        self.concept = concept
        self.id_from = id_from
        self.entity2id = load_pickle(entity2id_path)
        self.title2id = load_json(title2id_path)
        self.word2id = load_json(word2id_path)
        if self.id_from == 'MultiHeadAttentionAndBeyondWordEmbedding':
            self.id_from = 'BeyondWordEmbedding'

    def __getraw__(self, item):
        # word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
        #                                                item['h'][2][0],
        #                                                item['t'][2][0])
        if self.concept:
            word, pos1, pos2, mask = self.encoder.tokenize_concept(item['tokens'],
                                                                   item['h'][2][0],
                                                                   item['t'][2][0],
                                                                   item['h'][0],
                                                                   item['t'][0],
                                                                   self.ins2cpt)
        else:
            word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                           item['h'][2][0],
                                                           item['t'][2][0])
        return word, pos1, pos2, mask

    #
    def __additem__(self, d, word, pos1, pos2, mask, conceptIDtensor):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['conceptID'].append(conceptIDtensor)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'conceptID': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'conceptID': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()

                item = self.json_data[class_name][j]
                h = item['h'][0]
                t = item['t'][0]
                conceptID = self.entityPair2concept2id((h, t), id_from=self.id_from)
                conceptIDtensor = self.conceptID2tensor(conceptID, id_from=self.id_from)
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, conceptIDtensor)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, conceptIDtensor)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000

    def conceptID2tensor(self, ConceptID, id_from='kgEmbeddingOrBeyondWordEmbedding'):
        '''
        给定查询集和支撑集句子中的概念ID或ID list返回其对应的tensor
        #Para:id_from: id源于预训练的kg embedding中词的id 或者源于论文[1]中预训练词向量word2id中词的id
        References:
            [1]Beyond Word Embeddings: Learning Entity and Concept Representations from Large Scale Knowledge Bases
        '''
        if id_from == 'kgEmbedding':
            conceptIDtensor = torch.zeros(4).long()

            for p in range(len(ConceptID)):
                conceptIDtensor[p] = ConceptID[p]

        elif id_from == 'BeyondWordEmbedding':
            # 概念中的词最多12个，平均值为2.026551292913871
            conceptIDtensor = torch.ones([4, 12]) * -2  # 初始化值为-2的tensor,查询集每个句子头尾实体对应四个概念，每个概念最多对应五个词
            for i in range(len(ConceptID)):
                for j in range(len(ConceptID[i])):
                    try:
                        conceptIDtensor[i, j] = ConceptID[i][j]
                    except Exception as e:
                        print('-------------queryConceptID-----------')
                        print(ConceptID)
                        print(i, j)
        else:
            print('please input right id source')
        return conceptIDtensor

    def entityPair2concept2id(self, entityPair, id_from='kgEmbeddingOrBeyondWordEmbedding'):
        '''
        返回实体对中的实体在concpetgraph中的概念的id，[h1cpt1toID,h2cpt2toID,t1cpt2toID,t2cpt2toID]
        h1cpt1toID：头实体对应的概念1对应的ID 或 ID list
        #Para:id_from: id源于预训练的kg embedding中词的id 或者源于论文[1]中预训练词向量word2id中词的id
        References:
            [1]Beyond Word Embeddings: Learning Entity and Concept Representations from Large Scale Knowledge Bases
        '''
        id = []

        for i in range(len(entityPair)):
            entity = entityPair[i]
            h2concept1toid, h2concept2toid = self.entity2concept2id(entity, id_from)
            id.append(h2concept1toid)
            id.append(h2concept2toid)
        return id

    def entity2concept2id(self, entity, id_from):
        entity = entity.lower()
        e2concept = instance2conept(self.ins2cpt, entity)
        e2concept1 = e2concept[0].lower()
        e2concept2 = e2concept[1].lower()

        if (e2concept1 == 'unknowconcept1') or (e2concept1 == 'unknowconcept2'):

            if id_from == 'kgEmbedding':
                e2concept1toid = -1
            elif id_from == 'BeyondWordEmbedding':
                e2concept1toid = [-1]
            else:
                print('please input right id source')
        else:
            if id_from == 'kgEmbedding':
                e2concept1toid = self.entity2id[e2concept1]
            elif id_from == 'BeyondWordEmbedding':
                e2concept1toid = self.id_from_BeyondWordEmbedding(e2concept1)
            else:
                print('please input right id source')

        if (e2concept2 == 'unknowconcept1') or (e2concept2 == 'unknowconcept2'):
            if id_from == 'kgEmbedding':
                e2concept2toid = -1
            elif id_from == 'BeyondWordEmbedding':
                e2concept2toid = [-1]
            else:
                print('please input right id source')
        else:
            if id_from == 'kgEmbedding':
                e2concept2toid = self.entity2id[e2concept2]
            elif id_from == 'BeyondWordEmbedding':
                e2concept2toid = self.id_from_BeyondWordEmbedding(e2concept2)
            else:
                print('please input right id source')

        return e2concept1toid, e2concept2toid

    def id_from_BeyondWordEmbedding(self, concept: str) -> list:
        '''
        输入概念，优先获得其在title2id中的id，如果title2id的key中没有concept,把concept进行分词，获取每个词在word2id中的id,返回ID list
        '''
        id = []
        concept = concept.lower()
        cptID = self.title2id.get(concept)
        if cptID == None:
            cpt_words = concept.split(' ')
            for word in cpt_words:
                w2id = self.word2id.get(word)
                if w2id == None:
                    w2id = -1
                    id.append(w2id)
                else:
                    id.append(w2id)
        else:
            cptID = cptID[0]
            cptID = self.word2id.get(cptID)
            if cptID == None:
                cptID = -1
                id.append(cptID)
            else:
                id.append(cptID)
        return id


def load_json(path):
    with tqdm(total=1, desc=f'loading' + path) as pbar:
        with open(path, mode='r', encoding='utf-8') as fr:
            data = json.load(fr)
        pbar.update(1)
    return data


def load_pickle(path):
    with tqdm(total=1, desc=f'loading' + path) as pbar:
        with open(path, mode='rb') as fr:
            data = pickle.load(fr)
        pbar.update(1)
    return data


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'conceptID': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'conceptID': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_concept_loader(name, encoder, nWay, K, Q, batch_size, ins2cpt, concept, id_from, entity2id, title2id, word2id,
                       num_workers=32, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, nWay, K, Q, na_rate, root, ins2cpt, concept, id_from, entity2id, title2id,
                            word2id)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train',
                        help='train file')
    parser.add_argument('--val', default='val',
                        help='val file')
    parser.add_argument('--test', default='test_wiki',
                        help='test file')
    parser.add_argument('--trainN', default=5, type=int,
                        help='N in train')
    parser.add_argument('--K', default=5, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=5, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: cnn or bert or roberta')
    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
                        help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')
    parser.add_argument('--na_rate', default=0, type=int,
                        help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--max_length', default=1, type=int,
                        help='max length')
    parser.add_argument('--id_from', default='MultiHeadAttentionAndBeyondWordEmbedding',
                        help='BeyondWordEmbeddingOrkeEmbeddingOrMultiHeadAttentionAndBeyondWordEmbedding')
    parser.add_argument('--concept', action='store_true', help='use concept in kg(conceptgraph)')
    parser.add_argument('--entity2id', default='conceptgraphEmbedding/TransE_l2_concetgraph_2/entities2id',
                        help='entity2id in conceptgraph file path')
    parser.add_argument('--word2id', default='BeyondWordEmbedding/word2id', help='word2id file path')
    parser.add_argument('--title2id', default='BeyondWordEmbedding/all_titles2id', help='title2id file path')
    opt = parser.parse_args()
    trainN = opt.trainN
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size

    encoder_name = opt.encoder
    max_length = opt.max_length
    pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
    entity2id = loadingConceptGraphEntity2ID(root='../data/')
