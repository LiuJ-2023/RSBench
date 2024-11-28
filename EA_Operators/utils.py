import random
import time
import openai
import re
import json
import numpy as np

def extract_edit_prompt(response):
    pattern = r'<START>\s*(.*?)\s*<END>'
    result_list = re.findall(pattern, response, re.DOTALL)
    if len(result_list) == 0:
        pattern = r'<START>(.*?)<END>'
        result_list = re.findall(pattern, response, re.DOTALL)
    return result_list 

def PBI(f_input,w,z,z_max):
    if f_input.ndim == 1:
        f_input = np.array([f])
    if w.ndim == 1:
        w = np.array([w])
    z = z
    theta = 15
    
    f = f_input
    w_norm = np.sqrt(np.sum(w**2,axis=1))
    d1 = np.abs(np.sum((w*(f-z)),axis=1))/w_norm
    d2 = np.sqrt(np.sum((f - (z+d1.reshape(-1,1)*w/w_norm.reshape(-1,1)))**2,axis=1))

    # f = (f_input - z.reshape(1,-1))/(z_max.reshape(1,-1) - z.reshape(1,-1))
    # d1 = np.abs(np.sum((w*f),axis=1))/w_norm
    # d2 = np.sqrt(np.sum((f - (d1.reshape(-1,1)*w/w_norm.reshape(-1,1)))**2,axis=1))
    return d1 + theta*d2

def Tchebycheff(f_input,w):
    if f_input.ndim == 1:
        f_input = np.array([f])
    if w.ndim == 1:
        w = np.array([w])
    f_agg = np.max(f_input*w,axis=1) + 0.05*np.sum(f_input*w,axis=1)
    return f_agg

def choice_matrix(p,n):
    c = p.cumsum(axis=1)
    c = c.reshape(c.shape[0],c.shape[1],1)
    c = c.repeat(n,axis=2)
    u = np.random.rand(c.shape[0],1,n)
    choices = (u<c).argmax(axis=1)
    return choices.T

def nondominated_sort(pop_obj, n_sort):
    """
    :rtype:
    :param n_sort:
    :param pop_obj: objective vectors
    :return: [FrontNo, MaxFNo]
    """
    n, m_obj = np.shape(pop_obj)
    a, loc = np.unique(pop_obj[:, 0], return_inverse=True)
    index = pop_obj[:, 0].argsort()
    new_obj = pop_obj[index, :]
    front_no = np.inf * np.ones(n)
    max_front = 0
    while np.sum(front_no < np.inf) < min(n_sort, len(loc)):
        max_front += 1
        for i in range(n):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i, 0, -1):
                    if front_no[j - 1] == max_front:
                        m = 2
                        while (m <= m_obj) and (new_obj[i, m - 1] >= new_obj[j - 1, m - 1]):
                            m += 1
                        dominated = m > m_obj
                        if dominated or (m_obj == 2):
                            break
                if not dominated:
                    front_no[i] = max_front
    return front_no[loc], max_front

def crowding_distance(pop_obj, front_no):
    """
    The crowding distance of each Pareto front
    :param pop_obj: objective vectors
    :param front_no: front numbers
    :return: crowding distance
    """
    n, M = np.shape(pop_obj)
    crowd_dis = np.zeros(n)
    front = np.unique(front_no)
    Fronts = front[front != np.inf]
    for f in range(len(Fronts)):
        Front = np.array([k for k in range(len(front_no)) if front_no[k] == Fronts[f]])
        Fmax = pop_obj[Front, :].max(0)
        Fmin = pop_obj[Front, :].min(0)
        for i in range(M):
            rank = np.argsort(pop_obj[Front, i])
            crowd_dis[Front[rank[0]]] = np.inf
            crowd_dis[Front[rank[-1]]] = np.inf
            for j in range(1, len(Front) - 1):
                crowd_dis[Front[rank[j]]] = crowd_dis[Front[rank[j]]] + (pop_obj[(Front[rank[j + 1]], i)] - pop_obj[
                    (Front[rank[j - 1]], i)]) / (Fmax[i] - Fmin[i])
    return crowd_dis

def environment_selection(population, N):
    '''
    environmental selection in NSGA-II
    :param population: current population
    :param N: number of selected individuals
    :return: next generation population
    '''
    front_no, max_front = nondominated_sort(population[1], N)
    next_label = [False for i in range(front_no.size)]
    for i in range(front_no.size):
        if front_no[i] < max_front:
            next_label[i] = True
    crowd_dis = crowding_distance(population[1], front_no)
    last = [i for i in range(len(front_no)) if front_no[i]==max_front]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[: (N - int(np.sum(next_label)))]
    rest = [last[i] for i in delta_n]
    for i in rest:
        next_label[i] = True
    index = np.array([i for i in range(len(next_label)) if next_label[i]])
    next_pop = [[population[0][i] for i in index], population[1][index]]

    while len(next_pop[0]) != N:
        if len(next_pop[0]) > N:
            idx_ = np.random.randint(0,len(next_pop[0]))
            next_pop[0].pop(idx_)
            next_pop[1] = np.delete(next_pop[1], idx_,axis = 0)
        elif len(next_pop[0]) < N:
            idx_ = np.random.randint(0,len(population[0]))
            next_pop[0].append(population[0][idx_])
            next_pop[1] = np.concatenate((next_pop[1],population[1][idx_:idx_+1]),axis=0)
    return next_pop, front_no[index], crowd_dis[index],index

def CalFitIBEA(F,kappa):
    N = F.shape[0]
    F_max = np.max(F, axis=0, keepdims=True)
    F_min = np.min(F, axis=0, keepdims=True)
    F = (F - F_min)/(F_max - F_min)
    I = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            I[i,j] = np.max(F[i,:] - F[j,:])
    C = np.max(np.abs(I),axis=0,keepdims=True)
    Fit = np.sum( -np.exp( -I/C/kappa), axis=0 ) + 1
    return Fit, I, C

def IBEA_Selection(Population, F, N, kappa):
    Fit, I, C = CalFitIBEA(F,kappa)
    while Fit.shape[0] > N:
        idx_min = np.argmin(Fit)
        Fit = Fit + np.exp( -I[idx_min:idx_min + 1,:]/C[:,idx_min:idx_min + 1]/kappa )
        Fit = np.delete(Fit, idx_min)
        Population.pop(idx_min)
        # Population = np.delete(Population, idx_min, axis=0)
        F = np.delete(F, idx_min, axis=0)
        I = np.delete(I, idx_min, axis=0)
        I = np.delete(I, idx_min, axis=1)
        C = np.delete(C, idx_min, axis=1) 
    return Population, F