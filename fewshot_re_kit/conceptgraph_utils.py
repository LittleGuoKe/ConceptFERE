import os
from datetime import datetime
from tqdm import tqdm
import pickle
import json
import numpy as np
import torch


def conceptgraph2id(
        root='../data/conceptgraphEmbedding/TransE_l2_concetgraph_2/',
        name1='entities', name2='relations', format=".tsv"):
    '''
    加载 conceptgraph中实体和关系及其对应的id
    '''

    print('starting:', datetime.now())
    entity2id = {}
    relation2id = {}
    path1 = os.path.join(root, name1 + format)
    path2 = os.path.join(root, name2 + format)
    if not os.path.exists(path1):
        print(path1)
        print("[ERROR] {} file does not exist!".format(name1))
        assert (0)
    # if not os.path.exists(path2):
    #     print("[ERROR] {} file does not exist!".format(name2))
    #     assert (0)
    with open(path1, mode='r') as f1:
        for line in tqdm(f1):
            id1, entity = line.split('\t')
            entity = entity.strip('\n')
            entity2id[entity] = int(id1)
    # with open(path2, mode='r') as f2:
    #     for line in tqdm(f2):
    #         id2, relaiton = line.split('\t')
    #         relaiton = relaiton.strip('\n')
    #         relation2id[relaiton] = int(id2)
    print('store entity2id')
    with open(root + 'entities2id.pickle', mode='wb') as f3:
        pickle.dump(entity2id, f3)
    # print('store relaiont2id')
    # with open(root + 'relations2id.pickle', mode='wb') as f4:
    #     pickle.dump(relation2id, f4)
    # with open(root + 'entities2id.json', mode='w') as f3:
    #     json.dump(entity2id, f3)
    # print('store relaiont2id')
    # with open(root + 'relations2id.json', mode='w') as f4:
    #     json.dump(relation2id, f4)

    loadingtime = datetime.now()
    # with open(root + 'entities2id.pickle', mode='rb') as f5:
    #     entity2id = pickle.load(f5)
    # with open(root + 'relations2id.pickle', mode='rb') as f6:
    #     relation2id = pickle.load(f6)

    # with open(root + 'entities2id.json', mode='r') as f5:
    #     entity2id = json.load(f5)
    # with open(root + 'relations2id.json', mode='r') as f6:
    #     relation2id = json.load(f6)
    #
    # donetime = datetime.now()
    # print('loading time', donetime - loadingtime)
    # print(len(entity2id))
    # print(len(relation2id))
    # print('ending:', datetime.now())


def entity2vec(entity: str, entity2id, entityEmbedding):
    entityID = entity2id[entity]
    entityVec = entityEmbedding[entityID, :]

    return entityVec


def relation2vec(relation: str, relation2id, relationEmbedding):
    relaiontID = relation2id[relation]
    relationVec = relationEmbedding[relaiontID, :]
    return relationVec


def conceptgraphInitial(root='../data/conceptgraphEmbedding/TransE_l2_concetgraph_2/'):
    '''加载entity2id， relaion2id，entityEmbedding，relationEmbedding 文件'''
    loadingtime = datetime.now()

    with tqdm(total=4, desc=f'loading entity2id，relaion2id，entityEmbedding，relationEmbedding file') as pbar:
        with open(root + 'entities2id.pickle', mode='rb') as f5:
            entity2id = pickle.load(f5)
        pbar.update(1)

        with open(root + 'relations2id.pickle', mode='rb') as f6:
            relation2id = pickle.load(f6)
        pbar.update(1)

        entityEmbedding = np.load(root + 'concetgraph_TransE_l2_entity.npy')
        pbar.update(1)

        relaitonEmbedding = np.load(root + 'concetgraph_TransE_l2_relation.npy')
        pbar.update(1)
    donetime = datetime.now()
    print('initializing time', donetime - loadingtime)
    return entity2id, relation2id, entityEmbedding, relaitonEmbedding


def loadingConceptGraphEntity(root='../data/conceptgraphEmbedding/TransE_l2_concetgraph_2/'):
    '''加载conceptgraph中实体以及embedding'''

    with tqdm(total=2, desc=f'loading entity2id, entityEmbeddingfile') as pbar:
        with open(root + 'entities2id.pickle', mode='rb') as f5:
            entity2id = pickle.load(f5)
        pbar.update(1)

        entityEmbedding = np.load(root + 'concetgraph_TransE_l2_entity.npy')
        pbar.update(1)
    return entity2id, entityEmbedding


def loadingConceptGraphEntity2ID(root, path='conceptgraphEmbedding/TransE_l2_concetgraph_2/'):
    file = root + path + 'entities2id.pickle'
    with tqdm(total=1, desc=f'loading entity2id in conceptgraph') as pbar:
        with open(file, mode='rb') as f5:
            entity2id = pickle.load(f5)
        pbar.update(1)
    return entity2id


def loadingInstance2concept(path='../data/conceptgraph/instance2concept.pickle'):
    with tqdm(total=1, desc=f'loading Instance2concept file') as pbar:
        with open(path, mode='rb') as f:
            instance2concept = pickle.load(f)
        pbar.update(1)
    return instance2concept


def instance2conept(ins2cpt: dict, instance: str, top=2) -> list:
    '''给定实例，返回其对应的概念，最多两个'''
    concept = ins2cpt.get(instance)
    if concept == None:
        concept = ['unknowConcept1', 'unknowConcept2']
    elif len(concept) == 1:
        concept.append('unknowConcept1')
    else:
        concept = concept[:top]
    return concept


def instance2coneptPlus(ins2cpt: dict, instance: str, top=2) -> list:
    '''给定实例，返回其对应的概念，最多top个'''
    concept = ins2cpt.get(instance)
    if concept == None:
        '''查找实体中的词在知识库中的概念'''
        cpt_list = word2concept(ins2cpt, instance, top=top)
        if len(cpt_list) == 0:
            concept = ['unknowConcept']
        else:
            # print('zhao dao la')
            # print(instance,cpt_list)
            concept = cpt_list
    else:
        concept = concept[:top]
    return concept


def entity2id(root='../data/conceptgraphEmbedding/TransE_l2_concetgraph_2', name='entities'):
    path = os.path.join(root, name + ".tsv")
    e2id = {}
    if not os.path.exists(path):
        print('file path', path)
        print("[ERROR] Data file does not exist!")
        assert (0)

    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            entityID, entity = line.split('\t')
            entityID = int(entityID)
            entity = entity.strip('\n')
            e2id[entity] = entityID

    return e2id


def generateZeroVec(shape=(256,), dtype="float32"):
    zeroVec = np.zeros(shape, dtype)
    return zeroVec


def concept2vec(cpt: list, entity2id, entityEmbedding):
    cpt2vec = {}
    for cpt in cpt:
        if (cpt == 'unknowConcept1') or (cpt == 'unknowConcept2'):
            cpt2vec[cpt] = generateZeroVec()
        else:
            cpt2vec[cpt] = entity2vec(cpt, entity2id, entityEmbedding)
    return cpt2vec


def getConceptVec(entities: list, ins2cpt: dict, entity2id: dict, entityEmbedding):
    '''
    传入实体，查找实体对应的概念，然后返回概念对应的concept embedding
    entities:((h1,r1),(h2,r2))
    entity2id和entityEmbedding为训练conceptgraph pre-training kg embedding产生的文件
    '''
    h1 = entities[0][0]
    r1 = entities[0][1]
    h2 = entities[1][0]
    r2 = entities[1][1]

    h1_cpt = instance2conept(ins2cpt, h1, top=2)
    r1_cpt = instance2conept(ins2cpt, r1, top=2)
    h2_cpt = instance2conept(ins2cpt, h2, top=2)
    r2_cpt = instance2conept(ins2cpt, r2, top=2)

    h1_cpt2vec = concept2vec(h1_cpt, entity2id, entityEmbedding)
    r1_cpt2vec = concept2vec(r1_cpt, entity2id, entityEmbedding)
    h2_cpt2vec = concept2vec(h2_cpt, entity2id, entityEmbedding)
    r2_cpt2vec = concept2vec(r2_cpt, entity2id, entityEmbedding)

    '''取出向量concept embedding，去除concept name'''
    h1_cpt2vec = [vec for vec in h1_cpt2vec.values()]
    r1_cpt2vec = [vec for vec in r1_cpt2vec.values()]
    h2_cpt2vec = [vec for vec in h2_cpt2vec.values()]
    r2_cpt2vec = [vec for vec in r2_cpt2vec.values()]

    h1_cpt2vec = np.array(h1_cpt2vec)
    r1_cpt2vec = np.array(r1_cpt2vec)
    h2_cpt2vec = np.array(h2_cpt2vec)
    r2_cpt2vec = np.array(r2_cpt2vec)

    return (h1_cpt2vec, r1_cpt2vec, h2_cpt2vec, r2_cpt2vec)


def getBatchConceptVec(batchEntities: list, ins2cpt: dict, e2id: dict, entityEmbedding):
    '''
    获取一个Batch里句子的头尾实体的对应的concept的kg embedding
    '''
    batch_h_r2vec = []
    for entities in batchEntities[0]:
        (h1_cpt2vec, r1_cpt2vec, h2_cpt2vec, r2_cpt2vec) = getConceptVec(entities, ins2cpt, e2id, entityEmbedding)
        batch_h_r2vec.append((h1_cpt2vec, r1_cpt2vec, h2_cpt2vec, r2_cpt2vec))
    batch_h_r2vec = np.array(batch_h_r2vec)
    return batch_h_r2vec


def word2concept(instance2concept, word, top=2):
    '''给定一个词查找其在conceptgrap中的concept'''
    word = word.split(' ')
    concept = []
    for w in word:
        cpt = instance2concept.get(w)
        if cpt == None:
            continue
        else:
            for c in cpt[:top]:
                concept.append(c)
    return concept


def load(model_path=None, first=0, normalize=False, log_every=0, load_concepts=True, format='bin',
         concepts_pattern='id[0-9]+di'):
    """
    load word2vec vocabulary vectors from binary/text file
    """
    if format == 'txt':
        return load_text(model_path, first, load_concepts, normalize, log_every, concepts_pattern)
    else:
        return load_binary(model_path, first, load_concepts, normalize, log_every, concepts_pattern)

    if log_every > 0:
        print('done loading!')

    return titles, redirects, vector_size, W, id2word, word2id, get_all_titles(W, titles, redirects, word2id)


def load_text():
    pass


def load_binary(model_path=None, first=0, load_concepts=True, normalize=False, log_every=0,
                concepts_pattern='id[0-9]+di'):
    """
    load word2vec vocabulary vectors from binary file
    这部分代码源于论文Beyond Word Embeddings: Learning Entity and Concept Representations from Large Scale Knowledge Bases的开源代码
    """
    import pickle
    import numpy as np
    import re
    import os

    if load_concepts == False:
        concepts_re = re.compile(concepts_pattern)

    with open(model_path, 'rb') as inp:
        if log_every > 0:
            print('start loading!')

        # read titles meta
        titles = pickle.load(inp)
        if log_every > 0:
            print('loaded ({0}) titles'.format(len(titles)))
            # read redirects meta
        redirects = pickle.load(inp)
        if log_every > 0:
            print('loaded ({0}) redirects'.format(len(redirects)))
        # read vectors
        vectors_pairs = []
        while inp.tell() < os.fstat(inp.fileno()).st_size:
            vectors_pairs.extend(pickle.load(inp))
        num = len(vectors_pairs)
        if num > 0:
            vector_size = len(vectors_pairs[0][1])
        else:
            vector_size = 0
        if log_every > 0:
            print('loading ({0}) vectors of size ({1})'.format(len(vectors_pairs), vector_size))
        W = np.zeros((num, vector_size))
        id2word = []
        word2id = {}
        total = 0
        for i in range(num):
            term = vectors_pairs[i][0]
            if load_concepts == False:
                if concepts_re.match(term) != None:
                    continue
            vec = vectors_pairs[i][1]
            W[total] = vec
            id2word.append(term)
            word2id[term] = total
            total += 1
            if first > 0 and total >= first:
                break
            if log_every > 0 and total > 0 and total % log_every == 0:
                print('loaded ({0}) vectors'.format(total))
        if load_concepts == False:
            W = W[:total, ]  # take only loaded vectors

        if normalize == True:
            W = (W.T / (np.linalg.norm(W, axis=1))).T

        if log_every > 0:
            print('done loading ({0}) vectors!'.format(total))
        return titles, redirects, vector_size, W, id2word, word2id, get_all_titles(W, titles, redirects, word2id)


def get_all_titles(model, titles, redirects, word2id, orig_titles=True, lower=True, prefix='', postfix=''):
    """
    return a map of all wikipedia titles and redirects existing in the model
    as keys and article id as values
    """
    all_pairs = []
    all_titles = {}
    for i, j in sorted(titles.items()):
        all_pairs.append((i, prefix + j + postfix, i))
    for i, j in sorted(redirects.items()):
        all_pairs.append((i, prefix + titles[j] + postfix, j))
    for i, id, j in all_pairs:
        if model is None or id in word2id:
            if lower == True:
                newi = i.lower()
            if orig_titles == True:
                oldval = all_titles.setdefault(newi, (id, j))
                if oldval != (id, j):  # this is a duplicate
                    if i.isupper() == False:  # keep the lower version Iowa vs. IOWA and America vs. AMERICA
                        all_titles[newi] = (id, j)
                #    print('unexpected duplicate title ({0}) for orginal title ({1}) where old title ({2})'.format(i,j,oldval[1]))
            else:
                oldval = all_titles.setdefault(i, (id,))
                # if oldval!= (id,):
                #    print('unexpected duplicate title ({0}) for orginal title ({1})'.format(i,j))
    return all_titles


def loadJson(root, name):
    path = os.path.join(root, name + ".json")
    if not os.path.exists(path):
        print("[ERROR] Data file does not exist!", path)
        assert (0)
    with tqdm(total=1, desc=f'loading' + path) as pbar:
        with open(path, mode='r', encoding='utf-8') as fr:
            data = json.load(fr)
        pbar.update(1)
    return data


def load_numpy_file_to_tensor(root,name):
    path = os.path.join(root, name + ".npy")
    if not os.path.exists(path):
        print("[ERROR] Data file does not exist!", path)
        assert (0)
    with tqdm(total=1, desc=f'loading' + path) as pbar:
        matrix = np.load(path,allow_pickle=True)
        matrix = torch.from_numpy(matrix)
        pbar.update(1)

    return matrix


if __name__ == '__main__':
    # print('starting loading')
    # entity2id, relation2id, entityEmbedding, relaitonEmbedding = conceptgraphInitial()
    e2id, entityEmbedding = loadingConceptGraphEntity()
    # entity = 'age'
    # entityVec = entity2vec(entity, entity2id, entityEmbedding)
    # print(entityVec)
    # path = '/home/yangshan/pycharm2server/KG/FewRel/data/conceptgraphEmbedding/TransE_l2_concetgraph_2/entities.tsv'
    # ins2cpt = loadingInstance2concept()
    # count = 0
    # with open(path, mode='r', encoding='utf-8') as f:
    #     for line in tqdm(f):
    #         entityID, entity = line.split('\t')
    #         entity = entity.strip('\n')
    #         concept = instance2conept(ins2cpt, entity, top=2)
    #         # if count < 100:
    #         #     print(concept)
    #     count = count + 1
    # generateZeroVec(shape=(256,), dtype="float32")
    # entity2id()
    # batch_h_r = [[[['east midlands airport', 'nottingham'], ['tjq', 'tanjung pandan']],
    #               [['east midlands airport', 'nottingham'], ['east midlands airport', 'nottingham']],
    #               [['tjq', 'tanjung pandan'], ['tjq', 'tanjung pandan']],
    #               [['tjq', 'tanjung pandan'], ['east midlands airport', 'nottingham']]]]
    #
    # entities = [['east midlands airport', 'nottingham'], ['tjq', 'tanjung pandan']]
    # (h1_cpt2vec, r1_cpt2vec, h2_cpt2vec, r2_cpt2vec) = getConceptVec(entities, ins2cpt, e2id, entityEmbedding)
    # batch_h_r2vec = getBatchConceptVec(batch_h_r, ins2cpt, e2id, entityEmbedding)
    # print(batch_h_r2vec)
    # conceptgraph2id()
