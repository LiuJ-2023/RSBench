from EA_Opeators.LLM_EA import LLM_MOEAD
from pymoo.util.ref_dirs import get_reference_directions
import json
import pickle
import time
import copy


def MOEAD_LLM(problem, max_iter, pop_size, num_sub_set, api_key, llm_model, save_path):
    # Parameter settings
    #####################################################################
    # Set the prompts
    initial_prompt = "Now, I have a prompt for may task. I want to modify this prompt to better achieve my task. \n \
                        I will give an example of my current prompt. Please randomly generate a prompt based on my example. \n \
                        My example is as follows: \n \
                        {example} \n \
                        Note that the final prompt should be bracketed with <START> and <END>."

    example = "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n" \
                        "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n" \
                        "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n" \
                        "3. Select the intent from the inferred ones that best represents the user's current preferences.\n" \
                        "4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.\n" \
                        "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.\n"

    crossover_prompt = "Please follow the instruction step-by-step to generate a better prompt. \n \
                        1. Cross over the following prompts and generate a new prompt: \n \
                        Prompt 1: {prompt1} \n \
                        Prompt 2: {prompt2}. \n \
                        2. Mutate the prompt generated in Step 1 and generate \
                        a final prompt bracketed with <START> and <END>."

    # Evolutionary Optimization
    ############################################
    # Initialization
    print('The Algorithm is Starting!')
    print('Initializing the Population...')
    weights = get_reference_directions("energy", problem.obj_num, pop_size)
    llm_ea = LLM_MOEAD(pop_size,problem.obj_num,initial_prompt,crossover_prompt, weights, num_sub_set, llm_model, api_key)
    pop = llm_ea.initialize(example)
    # Evaluate the initial population
    problem.Sample_Test_Data()
    start_time = time.time()
    y_pop =  problem.Evaluate(pop)
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码执行时间为：", execution_time/60, "分钟")
    print('Initialization has been accomplished!')
    
    # Evolution
    print('Evolution is starting!')
    Record_All = {'Iteration 0': {'Population':copy.deepcopy(pop),'Reward':copy.deepcopy(y_pop)}}
    print('Saving the Data')
    pickle.dump(Record_All, open(save_path, "wb"))
    for iter in range(max_iter):
        print('Generation' + str(iter))
        # MOEA/D Evolution
        problem.Sample_Test_Data()
        pop,y_pop = llm_ea.evolution(pop,y_pop,problem.Evaluate)

        # Pring and save the data
        print('Accomplish iteration ' + str(iter))
        Record_ = {'Iteration ' + str(iter + 1): {'Population':copy.deepcopy(pop),'Reward':copy.deepcopy(y_pop)}}
        Record_All.update(Record_)
        print('Saving the Data')
        pickle.dump(Record_All, open(save_path, "wb"))
        print('*************************************')
    print('Evolution has been finished!')
    return pop, y_pop
