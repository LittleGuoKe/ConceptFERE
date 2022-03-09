# Entity Concept-enhanced Few-shot Relation Extraction


This code is the source code of our paper "Entity Concept-enhanced Few-shot Relation Extraction" in the ACL2021.

Our code is based on [Bert-Pair](https://github.com/thunlp/fewrel).

### Requirements

```
 conda env create -f environment.yml
```

# Checkpoint, Data files used in the code

Since the files are very large, they are placed on the [Beihang cloud disk](https://bhpan.buaa.edu.cn:443/link/BF14053D268CED261B525666BF1638A2).

# Training data
For the Details of training data, you  can refer to FewRel: https://thunlp.github.io/2/fewrel2_da.html.
NOTE：We divide the original training dataset into a new training dataset and a new validation dataset, the corresponding code is in the re_split_dataset module in fewshot_re_kit/utils.py, and the validation set in the original dataset is used as the new test dataset
# How the code is executed

Example:

```
python train_demo.py --trainN 5 --N 5 --K 1 --Q 1 --model pair --encoder bert --pair --hidden_size 768 --val_step 1000  --save_ckpt checkpoint/5way1shot.ConceptFere.pth.tar --batch_size 1 --grad_iter 4  --optim adam --fp16 --id_from MultiHeadAttentionAndBeyondWordEmbedding > 5way1shot.ConceptFere.log 2>&1
```

--trainN  --N  --K  --Q: N-way-K-shot.

--model: specify the name of the model, such as proto, pair, etc.

--id_from: specify the source of the pre-trained concept embedding.

--grad_iter: in the case of insufficient GPU memory, set a small batchsize accumulate gradient every x iterations.

--fp16: use nvidia apex fp16.

## Citing

If you used our code, please kindly cite our paper:

```
@inproceedings{yang-etal-2021-entity,
    title = "Entity Concept-enhanced Few-shot Relation Extraction",
    author = "Yang, Shan  and
      Zhang, Yongfei  and
      Niu, Guanglin  and
      Zhao, Qinghua  and
      Pu, Shiliang",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.124",
    doi = "10.18653/v1/2021.acl-short.124",
    pages = "987--991"
}
```
