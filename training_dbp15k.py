# -*- coding: utf-8 -*-
"""
Created on Fri May 14 09:34:25 2021

@author: 
"""


from util import get_neg, get_hits, loadfile, rpl_module, compute_r, rfunc, neighbor_rels_1hop
import numpy as np

seed = 2
np.random.seed(seed)

language = 'zh_en' # zh_en | ja_en | fr_en
prior_align_percentage = 0 # 0% of seeds
#%%
e1 = 'data/' + language + '/ent_ids_1'
e2 = 'data/' + language + '/ent_ids_2'
ill = 'data/' + language + '/ref_ent_ids'
kg1 = 'data/' + language + '/triples_1'
kg2 = 'data/' + language + '/triples_2'


ill = loadfile(ill, 2)
illL = len(ill)
np.random.shuffle(ill)
train = np.array(ill[:int(illL // 10 * prior_align_percentage)])
valid = ill[int(illL // 10 * prior_align_percentage) : int(illL // 10 * prior_align_percentage) + 2000]
test = ill[int(illL // 10 * prior_align_percentage) : ]



kg1 = loadfile(kg1, 3)
kg2 = loadfile(kg2, 3)

e1 = loadfile(e1, 1)
e2 = loadfile(e2, 1)
e = len(set(e1) | set(e2))
kg = kg1 + kg2
ents = e1 + e2

head_r, tail_r, rel_type, rel_num = rfunc(kg, e)


ent_r = neighbor_rels_1hop(kg, e, rel_num)

# load embedding_lists
import torch
import json
lang = language[0:2]
with open(file='data/' + lang + '_en/' + lang + '_bert300.json', mode='r', encoding='utf-8') as f:
    embedding_list = json.load(f)

word_embeddings = torch.FloatTensor(embedding_list)

r_kg_1 = set()
for tri in kg1:
    r_kg_1.add(tri[1])
l1 = len(r_kg_1)
#%%
from CPL_OT_class import SSL
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def sp_mat(xxxx_r):
    xxxx_r = xxxx_r/np.expand_dims(np.sum(xxxx_r, -1), -1) # row normalization
    return torch.sparse_coo_tensor(np.stack(xxxx_r.nonzero()), xxxx_r[xxxx_r.nonzero()], xxxx_r.shape, dtype = torch.float32, device = device)

head_r_temp = head_r
tail_r_temp = tail_r
head_r_l_sp = sp_mat(head_r_temp.T)
tail_r_l_sp = sp_mat(tail_r_temp.T)
inputs = [head_r_l_sp, tail_r_l_sp]


model = SSL(
    input_dim = 300,
    primal_X_0 = word_embeddings.to(device=device),
    gamma = 1,
    e = e,
    KG = kg,
    inputs = inputs,
    ent_r = sp_mat(ent_r),
    device = device,
    ).to(device = device)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# head_r, tail_r, rel_type = rfunc_rnm(kg, e)
k = 125
batch_size = 256 # one of the hyper-parameters
epochs = 80
training_losses = []
valid_hit1s = []
output_layers = []

theta=4
init_align = train

for i in range(epochs+1):

    if i % 10 == 0:
        # if i == 0:

        out = model.build(inputs)

        outvec_r = compute_r(out.detach().cpu(), head_r_temp, tail_r_temp, 300)
        init_align, r_score, head_r_temp, tail_r_temp = rpl_module(out.detach(), outvec_r.to(device=device), l1, kg, train, rel_type, 
                                                         e1, e2, theta, head_r, tail_r, iters=1)
        
        head_r_l_sp = sp_mat(head_r_temp.T)
        tail_r_l_sp = sp_mat(tail_r_temp.T)
        inputs = [head_r_l_sp, tail_r_l_sp, torch.FloatTensor(np.sum(head_r_temp.T, -1)).to(device=device), torch.FloatTensor(np.sum(tail_r_temp.T, -1)).to(device=device)]

        neg_right = get_neg(init_align[:, 0], out, k)
        neg2_left = get_neg(init_align[:, 1], out, k)


    examples_train_index = np.arange(len(init_align))
    np.random.shuffle(examples_train_index)
    batches = init_align.shape[0] // (batch_size)
    for j in range(batches):
        index_range = examples_train_index[j*batch_size:(j+1)*batch_size]
        examples_train = init_align[index_range]
        neg_right_sample = neg_right[index_range]
        neg2_left_sample = neg2_left[index_range]
        r_score_sample = r_score[index_range]
        
        Inputs = [examples_train[:, 0],
                  examples_train[:, 1],
                  torch.LongTensor(neg_right_sample).to(device=device),
                  torch.LongTensor(neg2_left_sample).to(device=device),
                  inputs,
                  torch.Tensor(r_score_sample).to(device=device),
                  ]

        loss = model(Inputs) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out = model.build(inputs)

    if i % 10 == 0:
        _ = get_hits(out, test)
        print('')

    print('Epoch: {}/{}  Loss: {:.4f}'.format(i+1, epochs, loss.item()))

