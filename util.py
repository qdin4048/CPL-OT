# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:42:38 2020

@author: 
"""

import numpy as np
import pandas as pd
import torch

def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def get_neg(ILL, output_layer, k):
    KG_vec = output_layer.detach()
    ILL_vec = KG_vec[ILL]
    neg = []
    t = len(ILL)
    
    with torch.no_grad():
        sim = torch.cdist(ILL_vec, output_layer, p=1)
        for i in range(t):
            rank = sim[i, :].argsort()
            neg.append(rank[0:k].cpu().numpy())

    del sim
    del rank
    del ILL_vec
    del KG_vec
    torch.cuda.empty_cache()

    return np.array(neg)


def get_hits(vec, test_pair, top_k=(1, 10), display = True):
    vec = vec.detach()
    test_pair = np.array(test_pair)
    Lvec = vec[test_pair[:, 0]]
    Rvec = vec[test_pair[:, 1]]

    mrr_l = []
    mrr_r = []

    with torch.no_grad():
        sim_temp = torch.cdist(Lvec, Rvec, p=1)

    sim = sim_temp.cpu().numpy().copy()
    del sim_temp
    torch.cuda.empty_cache()

    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_l.append(1.0 / (rank_index+1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_r.append(1.0 / (rank_index+1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    if display == True:
        print('source to target (left):')
        for i in range(len(top_lr)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
        print('MRR: %.4f' % (np.mean(mrr_l)))

        print('target to source (right):')
        for i in range(len(top_rl)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
        print('MRR: %.4f' % (np.mean(mrr_r)))
        return None
    else:
        print('non_English to English (left):')
        for i in range(len(top_lr)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
        print('MRR: %.4f' % (np.mean(mrr_l)))

        print('English to non_English (right):')
        for i in range(len(top_rl)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
        print('MRR: %.4f' % (np.mean(mrr_r)))
        mean_hit1 = (top_lr[0] / len(test_pair) + top_rl[0] / len(test_pair)) / 2

        return mean_hit1


def cpl_ot(vec, vec_r, l1, M0, ref_data, rel_type, sim_e, sim_r, refine_dists_mat, e1, e2, theta, head_r, tail_r, final=False):
    ref = set()
    if len(ref_data) != 0:
        for pair in ref_data:
            ref.add((pair[0], pair[1]))
    
    r_num = len(vec_r)//2
    
    kg = {}
    rel_ent = {}
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[0] not in kg:
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()
        
        kg[tri[0]].add((tri[1], tri[2]))
        kg[tri[2]].add((tri[1]+r_num, tri[0]))
        rel_ent[tri[1]].add((tri[0], tri[2]))
    
    if len(ref_data) != 0:
        L = np.array(list(set(np.array(e1).reshape(-1,))-set(ref_data[:,0])))
        R = np.array(list(set(np.array(e2).reshape(-1,))-set(ref_data[:,1])))
    else:
        L = np.array(list(set(np.array(e1).reshape(-1,))))
        R = np.array(list(set(np.array(e2).reshape(-1,))))
        
    Lvec = vec[L]
    Rvec = vec[R]

    if sim_e is None:
        with torch.no_grad():
            sim = torch.cdist(Lvec, Rvec, p=1)
        sim_e = sim.cpu().numpy().copy()
        
        del sim
        torch.cuda.empty_cache()

    ref_raw = ref.copy()

    
    L_ent2index = {}
    R_set = {}
    for i in range(len(L)):
        L_ent2index[L[i]] = i
        j = sim_e[i, :].argsort()[0]
        if sim_e[i,j] >= theta:
            continue
        if j in R_set and sim_e[i, j] < R_set[j][1]:
            ref.remove((L[R_set[j][0]], R[j]))
            ref.add((L[i], R[j]))
            R_set[j] = (i, sim_e[i, j])
        if j not in R_set:
            ref.add((L[i], R[j]))
            R_set[j] = (i, sim_e[i, j])
    
    R_ent2index = {}
    for i in range(len(R)):
        R_ent2index[R[i]] = i

    L_1 = np.array(list(set(np.array(e1).reshape(-1,))-set(np.array(list(ref))[:,0])))
    R_1 = np.array(list(set(np.array(e2).reshape(-1,))-set(np.array(list(ref))[:,1])))

    sim_e_1 = sim_e[[L_ent2index[ent] for ent in L_1], :][:, [R_ent2index[ent] for ent in R_1]]

    len_L1 = len(L)
    len_R1 = len(R)
    L_1 = np.array(list(set(np.array(e1).reshape(-1,))-set(np.array(list(ref))[:,0])))
    R_1 = np.array(list(set(np.array(e2).reshape(-1,))-set(np.array(list(ref))[:,1])))
    while len_L1 - len(L_1) > 0 or len_R1 - len(R_1) > 0:
    
        sim_e_1 = sim_e[[L_ent2index[ent] for ent in L_1], :][:, [R_ent2index[ent] for ent in R_1]]
        
        R_set = {}
        for i in range(len(L_1)):
            j = sim_e_1[i, :].argsort()[0]
            if sim_e_1[i,j] >= theta:
                continue
            if j in R_set and sim_e_1[i, j] < R_set[j][1]:
                ref.remove((L_1[R_set[j][0]], R_1[j]))
                ref.add((L_1[i], R_1[j]))
                R_set[j] = (i, sim_e_1[i, j])
            if j not in R_set:
                ref.add((L_1[i], R_1[j]))
                R_set[j] = (i, sim_e_1[i, j])
        
        len_L1 = len(L_1)
        len_R1 = len(R_1) 
        L_1 = np.array(list(set(np.array(e1).reshape(-1,))-set(np.array(list(ref))[:,0])))
        R_1 = np.array(list(set(np.array(e2).reshape(-1,))-set(np.array(list(ref))[:,1])))
            
    
    if final == False:
        if sim_r is None:
            with torch.no_grad():
                sim = torch.cdist(vec_r[:l1], vec_r[l1:r_num], p=1)
            sim_r = sim.cpu().numpy().copy()
            
            del sim
            torch.cuda.empty_cache()
        
        ref_r = set()
        for i in range(l1):
            j = sim_r[i, :].argsort()[0]
            if sim_r[i,j] < 3:
                ref_r.add((i, j+l1))
                ref_r.add((i+r_num,j+l1+r_num))
        
        for i in range(len(L)):
            rank = sim_e[i, :].argsort()[:100]
            for j in rank:
                if R[j] in kg and L[i] in kg:
                    match_num = 0
                    for n_1 in kg[L[i]]:
                        for n_2 in kg[R[j]]:
                            if (n_1[1], n_2[1]) in ref and (n_1[0], n_2[0]) in ref_r:
                                w = rel_type[str(n_1[0]) + ' ' + str(n_1[1])] * rel_type[str(n_2[0]) + ' ' + str(n_2[1])]
                                match_num += w
                    sim_e[i,j] -= 10 * match_num / (len(kg[L[i]]) + len(kg[R[j]]))


        for i in range(l1):
            rank = sim_r[i, :].argsort()[:20]
            for j in rank:
                if i in rel_ent and j+l1 in rel_ent:
                    match_num = 0
                    for n_1 in rel_ent[i]:
                        for n_2 in rel_ent[j+l1]:
                            if (n_1[0],n_2[0]) in ref and (n_1[1],n_2[1]) in ref: 
                                match_num += 1
                    sim_r[i,j] -= 200 * match_num / (len(rel_ent[i])+len(rel_ent[j+l1]))

    ref_new = np.array(list(ref-ref_raw))
    new_dists = sim_e[[L_ent2index[ent] for ent in ref_new[:,0]], 
                      [R_ent2index[ent] for ent in ref_new[:,1]]]
    r_new_score = sigmoid(0.25*theta-new_dists).reshape(-1,1)

    if len(ref_data) != 0:
        ref_raw = np.array(list(ref_raw))
        r_raw_score = np.ones((ref_raw.shape[0],)).reshape(-1,1)
        ref_com = np.vstack((ref_raw, ref_new))
        r_score = np.vstack((r_raw_score, r_new_score))
    else:
        ref_com = ref_new
        r_score = r_new_score
        
    return sim_e, sim_r, ref_com, r_score, head_r, tail_r


def rpl_module(vec, vec_r, l1, M0, ref_data, rel_type, e1, e2, theta, head_r, tail_r, iters=1):
    sim_e, sim_r, refine_dists_mat = None, None, None
    for n in range(iters):
        sim_e, sim_r, _, _, head_r, tail_r = cpl_ot(vec, vec_r, l1, M0, ref_data, rel_type, sim_e, sim_r, refine_dists_mat, e1, e2, theta, head_r, tail_r)
    _, _, init_align, r_score, head_r, tail_r = cpl_ot(vec, vec_r, l1, M0, ref_data, rel_type, sim_e, sim_r, refine_dists_mat, e1, e2, theta, head_r, tail_r, final = True)

    return init_align, r_score, head_r, tail_r

def compute_r(inlayer, head_r, tail_r, dimension):
    head_l=torch.transpose(torch.FloatTensor(head_r), 0, 1)
    tail_l=torch.transpose(torch.FloatTensor(tail_r), 0, 1)
    L=torch.matmul(head_l,inlayer)/torch.unsqueeze(torch.sum(head_l, -1),-1)
    R=torch.matmul(tail_l,inlayer)/torch.unsqueeze(torch.sum(tail_l, -1),-1)
    
    r_forward=torch.cat((L,R),axis=-1)
    r_reverse=torch.cat((-L,-R),axis=-1)
    r_embeddings = torch.cat((r_forward, r_reverse), axis=0)
    
    return r_embeddings


def rfunc(KG, e):
    head = {}
    cnt = {}
    rel_type = {}
    cnt_r = {}
    for tri in KG:
        r_e = str(tri[1]) + ' ' + str(tri[2])
        if r_e not in cnt:
            cnt[r_e] = 1
            head[r_e] = set([tri[0]])
        else:
            cnt[r_e] += 1
            head[r_e].add(tri[0])
        
        if tri[1] not in cnt_r:
            cnt_r[tri[1]] = 1

    r_num = len(cnt_r)
    
    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value
    
    del cnt
    del head
    del cnt_r
    cnt = {}
    head = {}
    
    for tri in KG:
        r_e_new = str(tri[1]+r_num) + ' ' + str(tri[0])
        if r_e_new not in cnt:
            cnt[r_e_new] = 1
            head[r_e_new] = set([tri[2]])
        else:
            cnt[r_e_new] += 1
            head[r_e_new].add(tri[2])
    
    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value
    
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
    
    return head_r, tail_r, rel_type, r_num

def neighbor_rels_1hop(kg, e, r_num, reverse_r = True):
    kg_dataframe = pd.DataFrame(kg).astype(int)
    kg_sorted_h = kg_dataframe.sort_values(by = [0,1])
    kg_sorted_t = kg_dataframe.sort_values(by = [2,1])

    ent_r = np.zeros((e, 2*r_num))
    for i in range(e):
        r_hrt = kg_sorted_h[kg_sorted_h[0]==i][1]
        r_trh = kg_sorted_t[kg_sorted_t[2]==i][1]
        r_hrt_counts = r_hrt.value_counts()
        r_trh_counts = r_trh.value_counts()
        r_hrt_indeces = r_hrt_counts.index.values
        r_trh_indeces = r_trh_counts.index.values
        for count, j in zip(r_hrt_counts, r_hrt_indeces):
            ent_r[i][j] += count

        for count, j in zip(r_trh_counts, r_trh_indeces):
            ent_r[i][r_num+j] += count

    return ent_r
