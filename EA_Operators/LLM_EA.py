from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .utils import extract_edit_prompt, environment_selection, Tchebycheff, choice_matrix, IBEA_Selection
import json
import numpy as np
import random
import os
import time

class LLM_EA():
    def __init__(self, pop_size, initialize_prompt, crossover_prompt, llm_model, api_key):
        # Initial the parameters of the EA
        self.pop_size = pop_size

        # Initial the LLM for initialization
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            llm_initialize = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_initialize = ChatOpenAI(api_key=api_key)
        prompt_initialize = ChatPromptTemplate.from_messages([
            ("system", "You are an initializer to provide a set of initial prompts according to user's requirement"),
            ("user", initialize_prompt)]
            )
        output_parser_initialize = StrOutputParser()
        self.chain_initialize = prompt_initialize | llm_initialize | output_parser_initialize

        # Initial the LLM for crossover
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            llm_operator = ChatZhipuAI(model="glm-3-turbo")
        elif llm_model == 'gpt':
            llm_operator = ChatOpenAI(api_key=api_key)
        prompt_operator = ChatPromptTemplate.from_messages([
            ("system", "You are an evolutionary operator for prompt optimization."),
            ("user", crossover_prompt)]
            )
        output_parser_operator = StrOutputParser()
        self.chain_operator = prompt_operator | llm_operator | output_parser_operator

    def initialize(self,example):
        pop = []
        for i in range(self.pop_size):
            while True:
                try:
                    output = self.chain_initialize.invoke(example)
                    break
                except:
                    print('Lost the connect! Wait 20 sec and Query again!')
                    time.sleep(20)
            individual = extract_edit_prompt(output)
            pop.extend(individual)
        return pop

    def crossover(self,pop):
        offsprings = []
        for i in range(self.pop_size):
            idx = np.random.choice(len(pop),2,replace=False)
            while True:
                try:
                    output = self.chain_operator.invoke({"prompt1": pop[idx[0]],"prompt2": pop[idx[1]]})
                    break
                except:
                    print('Lost the connect! Wait 20 sec and Query again!')
                    time.sleep(20)
            offspring = extract_edit_prompt(output)
            offsprings.extend(offspring)
        return offsprings

    def naive(self,pop):
        offsprings = []
        for i in range(self.pop_size):
            while True:
                try:
                    output = self.chain_operator.invoke({"pop": pop})
                    break
                except:
                    print('Lost the connect! Wait 20 sec and Query again!')
                    time.sleep(20)
            offspring = extract_edit_prompt(output)
            offsprings.extend(offspring)
        return offsprings
    
    def enviromnent_selection(self,pop,y_pop,offspring,y_offspring):
        pop.extend(offspring)
        y_pop = np.concatenate((y_pop,y_offspring))
        pop_next,_,_,_ = environment_selection([pop,y_pop],self.pop_size)
        print(len(pop_next[0]))
        return pop_next[0],pop_next[1]

    def IBEA_selection(self,pop,y_pop,offspring,y_offspring):
        pop.extend(offspring)
        y_pop = np.concatenate((y_pop,y_offspring),axis=0)
        pop, y_pop = IBEA_Selection(pop, y_pop, self.pop_size, 0.05)
        return pop, y_pop

class LLM_MOEAD():
    def __init__(self, pop_size, obj_num, initialize_prompt, crossover_prompt, weight, num_sub_set, llm_model, api_key):
        # Initial the parameters of the EA
        self.pop_size = pop_size
        self.weight = weight
        self.num_sub_set = num_sub_set
        self.obj_num = obj_num

        # Set corresponding weights
        w_repeat1 = weight.reshape(1,self.pop_size,self.obj_num).repeat(self.pop_size,axis=0)
        w_repeat2 = weight.reshape(self.pop_size,1,self.obj_num).repeat(self.pop_size,axis=1)
        dist = np.sqrt(np.sum((w_repeat1 - w_repeat2)**2,axis=2))
        B = np.argsort(dist,axis=1)
        self.B = B[:,0:self.num_sub_set]
        self.p_sel = np.ones([pop_size,self.num_sub_set])/self.num_sub_set

        # Initial the LLM for initialization
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            llm_initialize = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_initialize = ChatOpenAI(api_key=api_key)
        prompt_initialize = ChatPromptTemplate.from_messages([
            ("system", "You are an initializer to provide a set of initial prompts according to user's requirement"),
            ("user", initialize_prompt)]
            )
        output_parser_initialize = StrOutputParser()
        self.chain_initialize = prompt_initialize | llm_initialize | output_parser_initialize

        # Initial the LLM for crossover
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            llm_operator = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_operator = ChatOpenAI(api_key=api_key)
        prompt_operator = ChatPromptTemplate.from_messages([
            ("system", "You are an evolutionary operator for prompt optimization."),
            ("user", crossover_prompt)]
            )
        output_parser_operator = StrOutputParser()
        self.chain_operator = prompt_operator | llm_operator | output_parser_operator

    def initialize(self,example):
        pop = []
        for i in range(self.pop_size):
            output = self.chain_initialize.invoke(example)
            individual = extract_edit_prompt(output)
            pop.extend(individual)
        return pop

    def crossover_(self,parent1,parent2):
        output = self.chain_operator.invoke({"prompt1": parent1,"prompt2": parent2})
        offspring = extract_edit_prompt(output)
        if len(offspring) == 0:
            output = self.chain_operator.invoke({"prompt1": parent1,"prompt2": parent2})
            offspring = extract_edit_prompt(output)
        return [offspring[0]]
    
    def evolution(self,pop,y_pop,obj_func):
        # Randomly select two solution from each subset
        idx_choice = np.random.choice(self.pop_size, self.pop_size, replace=False)
        idx_sel = choice_matrix(self.p_sel,2)
        w_rand1 = self.B[idx_choice,idx_sel[0,idx_choice]]
        w_rand2 = self.B[idx_choice,idx_sel[1,idx_choice]]
        # Evolution
        for i in range(self.pop_size):
            # Generate offspring
            parent1 = pop[w_rand1[i]]
            parent2 = pop[w_rand2[i]]
            offspring = self.crossover_(parent1,parent2)
            # Evaluate offspring
            y_offspring = obj_func(offspring)
            # Update Population 
            z_min = np.min( np.vstack((y_pop,y_offspring)),axis=0)
            z_max = np.max(np.vstack((y_pop,y_offspring)),axis=0)
            y_pop_tch = Tchebycheff(y_pop[self.B[idx_choice[i]]],self.weight[self.B[idx_choice[i]]]) 
            y_offspring_tch = Tchebycheff(y_offspring,self.weight[self.B[idx_choice[i]]])
            idx_update = self.B[idx_choice[i],y_offspring_tch<y_pop_tch]
            for idx_update_ in idx_update:
                pop[idx_update_] = offspring[0]
                y_pop[idx_update] = y_offspring
        return pop, y_pop