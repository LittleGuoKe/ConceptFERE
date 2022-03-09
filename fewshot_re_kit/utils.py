import json
from datetime import datetime
import os
from tqdm import tqdm


def loadJson(path: str):
    with open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def dictSaveToJson(d: dict, Path: str):
    with open(Path, mode='w', encoding='utf-8') as f:
        json.dump(d, f)


def re_split_dataset(
        root='/home/yangshan/pycharm2server/KG/FewRel/data/',
        name='train_wiki', format=".json"):
    '''
    将原始的训练数据集划分成训练集和测试集，然后原始的验证集当作测试集
    '''

    print('starting:', datetime.now())
    path1 = os.path.join(root, name + format)

    if not os.path.exists(path1):
        print("[ERROR] {} file does not exist!".format(name))
        assert (0)
    traindata = loadJson(path1)
    # rel_num = len(traindata)
    new_train = {}
    new_val = {}
    count = 0
    for k, v in traindata.items():
        if count < 50:
            new_train[k] = v
            count = count + 1
        else:
            new_val[k] = v
            count = count + 1
    with tqdm(total=2, desc=f'store new_train and new_val file') as pbar:
        dictSaveToJson(new_train, root + 'train.json')
        pbar.update(1)
        dictSaveToJson(new_val, root + 'val.json')
        pbar.update(1)
    print('new_train size', len(new_train))
    print('new_val size', len(new_val))
    print('done:', datetime.now())


if __name__ == '__main__':
    # re_split_dataset()
    root = '/home/yangshan/pycharm2server/KG/FewRel/data/'
    val=loadJson(root + 'test_wiki.json')
    print(len(val))
