import numpy as np
import random
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
import os
import pickle
import time
from Utils import nondomination
import Utils.hypervolume as hypervolume
import copy
import os
import time

# Basic Functions
#####################################################################
def extract_item_list(response, target):
    try:
        response = response.replace(" ", " ")
        target = target.replace(" ", " ").replace("&amp;", "&").replace("&reg;","®")
        index = response.rfind(target)
        if index != -1:
            preceding_text = response[:index].strip()
            numbers = re.findall(r'\d+', preceding_text)
            if numbers:
                result_list = numbers
            else:
                result_list = []
        else:
            result_list = []
    except:
        result_list = []
    return result_list

def detect_error(response, target, mode='improve'):
    try:
        idx = response.index(target)
        return True, idx
    except:
        return False, None

def diversity_calculate(list_recommond,train_data):
    record_category = []
    for product in list_recommond:
        try:
            index = train_data["candidate_set"].index(product)
            category = train_data["category_list"][index]
            record_category.extend(category)
        except:
            pass
    unique_category = list(set(record_category))
    diversity = (len(unique_category))/(len(record_category) + 1e-10)
    return diversity

def APT(list_recommond,original_data):
    record_set_label = []
    for product in list_recommond:
        try:
            idx = original_data["candidate_set"].index(product)
            record_set_label.append(original_data["popular_list"][idx])
        except:
            pass
    vec_set_label = np.array(record_set_label)
    fariness = (np.sum(vec_set_label) + 1e-5)/(len(list_recommond) + 1e-5)
    return fariness

# The Accuracy vs. Diversity Problem
#####################################################################
class Acc_Div():
    def __init__(self,
                train_data,
                batch_num,
                api_key,
                llm_model = 'gpt'):
        # Parameter setting
        self.train_data = train_data
        self.batch_num = batch_num
        self.api_key = api_key
        self.llm_model = llm_model
        self.obj_num = 2

        # Initial the LLM for initialization
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            self.llm_recommond = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            self.llm_recommond = ChatOpenAI(api_key=self.api_key)
        self.output_parser_recommond = StrOutputParser()

        # Initial the LLM for translate
        # Build a LLM that can translate the output to a list that can be read by python
        prompt_translate = ChatPromptTemplate.from_messages([
            ("user", 'Please transfer a set of product names into a list that can be read by the python. \n \
              Please note that: \
              1. All of the elements in your generated list should be enclosed by " ". \n \
              2. The name of the output list should be named as "output" \n \
              3. The set of product names is given as follows: {input}')]
            )
        # 1. The list should contains only the product names. \n \
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            llm_translate = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_translate = ChatOpenAI(api_key=self.api_key)
        output_parser_translate = StrOutputParser()
        self.chain_translate = prompt_translate | llm_translate | output_parser_translate

        # Initial the LLM for error correction
        # Build a LLM that can correct the error
        prompt_correct = ChatPromptTemplate.from_messages([
            ("user", "The python list generated by you cannot be executed by python directly. \n \
             I will give you the python list you just generate. Please check the error and regenerate it. \n \
             A possible reason may be the wrong using of quotation marks. \n \
              The python list you just generated is given as follows: {input}")]
            )
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            llm_correct = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_correct = ChatOpenAI(api_key=self.api_key)
        output_parser_correct = StrOutputParser()
        self.chain_correct = prompt_correct | llm_correct | output_parser_correct

    # Sample a sub set of samples from the training data
    def Sample_Test_Data(self):
        self.sample_data = random.sample(self.train_data, self.batch_num)

    # Evaluate a single prompt
    def Evaluate_(self, prompt):
        # Setting the LLM recommonder
        prompt_recommond = ChatPromptTemplate.from_messages([
            ("system", "You are a recommonder for the shopping"),
            ("user", prompt + "\n Note that, you should make the recommond for only the candidate set \n The samples are listed as follows: \n {samples}")
            ])
        chain_recommond = prompt_recommond | self.llm_recommond | self.output_parser_recommond
        
        # Estimate the score for the samples
        reward_record = []
        diversity_record = []
        for data in self.sample_data:
            # Make the recommond by using LLM
            while True:
                try:
                    response = chain_recommond.invoke({"samples": data["input"]})
                    break
                except:
                    print('Lost the connect! Wait 20 sec and Query again!')
                    time.sleep(20)
            
            # Estimate the recommond quality for each sample
            try:
                # Translate the output of LLM to a list
                response = self.Translate(response)
                # Check whether the recommond is right
                flag_error, target_index = detect_error(response, data['target'], mode='select')
                
                # Record the reward (i.e., the index of the target)
                if flag_error:
                    reward_record.append(target_index/(len(response) + 1e-10))
                else:
                    reward_record.append(1)

                # Record the diversity
                diversity = diversity_calculate(response[:10],data)
                diversity_record.append(1 - diversity)
            except:
                reward_record.append(1)
                diversity_record.append(1)
        
        # Estimate the final objective function value
        reward_mean = np.mean(np.array(reward_record))
        diversity_mean = np.mean(np.array(diversity_record))
        return reward_mean, diversity_mean

    # Evaluate a set of prompts
    def Evaluate(self,pop):
        f = []
        for prompt in pop:
            f1, f2 = self.Evaluate_(prompt)
            f.append([f1,f2])
        f = np.array(f)
        return f
    
    # Since the output of the LLM is a string, and what we want is a list that contains all of the produce name
    # This function aims to translate the output of the LLM to a list, and the process is achieving by another LLM
    def Translate(self,input):
        while True:
            try:
                response = self.chain_translate.invoke({"input":input})
                break
            except:
                print('Lost the connect! Wait 20 sec and Query again!')
                time.sleep(20)    
        if self.llm_model == 'glm':
            regex_pattern = r"'```python(.*?)```'"
            response = re.findall(r"```python\s(.*?)```",response,re.DOTALL)[0]
        record = {}
        exec(response,globals(),record)
        output = record["output"]
        return output
    
    def Correction(self,input):
        while True:
            try:
                response = self.chain_correct.invoke({"input":input})
                break
            except:
                print('Lost the connect! Wait 20 sec and Query again!')
                time.sleep(20)
        record = {}
        exec(response,globals(),record)
        output = record["output"]
        return output

# The Accuracy vs. Fairness Problem
#####################################################################
class Acc_Fair():
    def __init__(self,
                train_data,
                batch_num,
                api_key,
                llm_model = 'gpt'):
        # Parameter setting
        self.train_data = train_data
        self.batch_num = batch_num
        self.api_key = api_key
        self.llm_model = llm_model
        self.obj_num = 2

        # Initial the LLM for initialization
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            self.llm_recommond = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            self.llm_recommond  = ChatOpenAI(api_key=self.api_key)
        self.output_parser_recommond = StrOutputParser()

        # Initial the LLM for translate
        # Build a LLM that can translate the output to a list that can be read by python
        prompt_translate = ChatPromptTemplate.from_messages([
            ("user", 'Please transfer a set of product names into a list that can be read by the python. \n \
              Please note that: \
              1. All of the elements in your generated list should be enclosed by " ". \n \
              2. The name of the output list should be named as "output" \n \
              3. The set of product names is given as follows: {input}')]
            )
        # 1. The list should contains only the product names. \n \
        # llm_translate = ChatOpenAI(api_key=self.api_key)
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            llm_translate = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_translate = ChatOpenAI(api_key=self.api_key)
        output_parser_translate = StrOutputParser()
        self.chain_translate = prompt_translate | llm_translate | output_parser_translate

        # Initial the LLM for error correction
        # Build a LLM that can correct the error
        prompt_correct = ChatPromptTemplate.from_messages([
            ("user", "The python list generated by you cannot be executed by python directly. \n \
             I will give you the python list you just generate. Please check the error and regenerate it. \n \
             A possible reason may be the wrong using of quotation marks. \n \
              The python list you just generated is given as follows: {input}")]
            )
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            llm_correct = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_correct = ChatOpenAI(api_key=self.api_key)
        output_parser_correct = StrOutputParser()
        self.chain_correct = prompt_correct | llm_correct | output_parser_correct

    def Sample_Test_Data(self):
        self.sample_data = random.sample(self.train_data, self.batch_num)

    def Evaluate_(self, prompt):
        # Setting the LLM recommonder
        # prompt_recommond = ChatPromptTemplate.from_messages([
        #     ("system", "You are a recommonder for the shopping"),
        #     ("user", prompt + "\n The samples are listed as follows: \n {samples} \
        #      I will also give the click through rate (high or low) of the products in the candidate set. \
        #      Please provide fair recommendations for products with both high or low clicks, and avoid the recommend of products with only high click through rates. \
        #      The click through rate are given as follows: \n {ctr}")]
        #     )
        prompt_recommond = ChatPromptTemplate.from_messages([
            ("system", "You are a recommonder for the shopping"),
            ("user", prompt + "\n Note that, you should make the recommond for only the candidate set \n The samples are listed as follows: \n {samples}.")
            ])
        chain_recommond = prompt_recommond | self.llm_recommond | self.output_parser_recommond
        
        # Estimate the score for the samples
        reward_record = []
        fairness_record = []
        for data in self.sample_data:
            # Make the recommond by using LLM
            while True:
                try:
                    # response = chain_recommond.invoke({"samples": data["input"],"ctr": data["Click Through Rate"]})
                    response = chain_recommond.invoke({"samples": data["input"]})
                    break
                except:
                    print('Lost the connect! Wait 20 sec and Query again!')
                    time.sleep(20)

            try:
                # Translate the output of LLM to a list
                response = self.Translate(response)
                # Check whether the recommond is right
                flag_error, target_index = detect_error(response, data['target'], mode='select')
                # Record the reward (i.e., the index of the target)
                if flag_error:
                    reward_record.append(target_index/(len(response) + 1e-10))
                else:
                    reward_record.append(1)

                # Record the intra-list binary unfairness
                apt = APT(response[:10],data)
                fairness_record.append(1 - apt)
            except:
                reward_record.append(1)
                fairness_record.append(1)
        reward_mean = np.mean(np.array(reward_record))
        fairness_mean = np.mean(np.array(fairness_record))
        return reward_mean, fairness_mean

    # Evaluate a set of prompts
    def Evaluate(self,pop):
        f = []
        for prompt in pop:
            f1, f2 = self.Evaluate_(prompt)
            f.append([f1,f2])
        f = np.array(f)
        return f
    
    # Since the output of the LLM is a string, and what we want is a list that contains all of the produce name
    # This function aims to translate the output of the LLM to a list, and the process is achieving by another LLM
    def Translate(self,input):
        while True:
            try:
                response = self.chain_translate.invoke({"input":input})
                break
            except:
                print('Lost the connect! Wait 20 sec and Query again!')
                time.sleep(20)    
        if self.llm_model == 'glm':
            regex_pattern = r"'```python(.*?)```'"
            response = re.findall(r"```python\s(.*?)```",response,re.DOTALL)[0]
        record = {}
        exec(response,globals(),record)
        output = record["output"]
        return output
    
    def Correction(self,input):
        response = self.chain_correct.invoke({"input":input})
        record = {}
        exec(response,globals(),record)
        output = record["output"]
        return output

# The Accuracy vs. Diversity vs. Fairness Problem 
#####################################################################
class Acc_Div_Fair():
    def __init__(self,
                train_data,
                batch_num,
                api_key,
                llm_model = 'gpt'):
        # Parameter setting
        self.train_data = train_data
        self.batch_num = batch_num
        self.api_key = api_key
        self.llm_model = llm_model
        self.obj_num = 3

        # Initial the LLM for initialization
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            self.llm_recommond = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            self.llm_recommond  = ChatOpenAI(api_key=self.api_key)
        self.output_parser_recommond = StrOutputParser()

        # Initial the LLM for translate
        # Build a LLM that can translate the output to a list that can be read by python
        prompt_translate = ChatPromptTemplate.from_messages([
            ("user", 'Please transfer a set of product names into a list that can be read by the python. \n \
              Please note that: \
              1. All of the elements in your generated list should be enclosed by " ". \n \
              2. The name of the output list should be named as "output" \n \
              3. The set of product names is given as follows: {input}')]
            )
        # 1. The list should contains only the product names. \n \
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            llm_translate = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_translate = ChatOpenAI(api_key=self.api_key)
        output_parser_translate = StrOutputParser()
        self.chain_translate = prompt_translate | llm_translate | output_parser_translate

        # Initial the LLM for error correction
        # Build a LLM that can correct the error
        prompt_correct = ChatPromptTemplate.from_messages([
            ("user", "The python list generated by you cannot be executed by python directly. \n \
             I will give you the python list you just generate. Please check the error and regenerate it. \n \
             A possible reason may be the wrong using of quotation marks. \n \
              The python list you just generated is given as follows: {input}")]
            )
        if llm_model == 'glm':
            os.environ["ZHIPUAI_API_KEY"] = api_key
            # llm_correct = ChatZhipuAI(model="glm-3-turbo")
            llm_correct = ChatZhipuAI(model="glm-4")
        elif llm_model == 'gpt':
            llm_correct = ChatOpenAI(api_key=self.api_key)
        output_parser_correct = StrOutputParser()
        self.chain_correct = prompt_correct | llm_correct | output_parser_correct

    def Sample_Test_Data(self):
        self.sample_data = random.sample(self.train_data, self.batch_num)

    def Evaluate_(self, prompt):
        # Setting the LLM recommonder
        prompt_recommond = ChatPromptTemplate.from_messages([
            ("system", "You are a recommonder for the shopping"),
            ("user", prompt + "\n Note that, you should make the recommond for only the candidate set \n The samples are listed as follows: \n {samples}")
            ])
        chain_recommond = prompt_recommond | self.llm_recommond | self.output_parser_recommond

        # Estimate the score for the samples
        reward_record = []
        diversity_record = []
        fairness_record = []
        for data in self.sample_data:
            # Make the recommond
            while True:
                try:
                    response = chain_recommond.invoke({"samples": data["input"]})
                    break
                except:
                    print('Lost the connect! Wait 20 sec and Query again!')
                    time.sleep(20)
                    
            try:
                # Translate the output of LLM to a list
                response = self.Translate(response)
                # Check whether the recommond is right
                flag_error, target_index = detect_error(response, data['target'], mode='select')
                # Record the reward (i.e., the index of the target)
                if flag_error:
                    reward_record.append(target_index/(len(response) + 1e-10))
                else:
                    reward_record.append(1)

                # Record the diversity
                diversity = diversity_calculate(response[:10],data)
                diversity_record.append(1 - diversity)

                # Record the intra-list binary unfairness
                apt = APT(response[:10],data)
                fairness_record.append(1 - apt)
            except:
                reward_record.append(1)
                diversity_record.append(1)
                fairness_record.append(1)
        reward_mean = np.mean(np.array(reward_record))
        diversity_mean = np.mean(np.array(diversity_record))
        fairness_mean = np.mean(np.array(fairness_record))
        return reward_mean, diversity_mean, fairness_mean

    def Evaluate(self,pop):
        f = []
        for prompt in pop:
            f1, f2, f3 = self.Evaluate_(prompt)
            f.append([f1,f2,f3])
        f = np.array(f)
        return f
    
    # Since the output of the LLM is a string, and what we want is a list that contains all of the produce name
    # This function aims to translate the output of the LLM to a list, and the process is achieving by another LLM
    def Translate(self,input):
        while True:
            try:
                response = self.chain_translate.invoke({"input":input})
                break
            except:
                print('Lost the connect! Wait 20 sec and Query again!')
                time.sleep(20)    
        if self.llm_model == 'glm':
            regex_pattern = r"'```python(.*?)```'"
            response = re.findall(r"```python\s(.*?)```",response,re.DOTALL)[0]
        record = {}
        exec(response,globals(),record)
        output = record["output"]
        return output
    
    def Correction(self,input):
        response = self.chain_correct.invoke({"input":input})
        record = {}
        exec(response,globals(),record)
        output = record["output"]
        return output