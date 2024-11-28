from EA_Opeators.LLM_EA import LLM_EA
import pickle
import time
import copy

def IBEA_LLM_wo_Init(problem, max_iter, pop_size, api_key, llm_model, save_path):
    # Parameter settings
    #####################################################################
    # Set the prompts
    initial_prompt = "Now, I have a prompt for may task. I want to modify this prompt to better achieve my task. \n \
                        I will give an example of my current prompt. Please randomly generate a prompt based on my example. \n \
                        My example is as follows: \n \
                        {example} \n \
                        Note that the final prompt should be bracketed with <START> and <END>."

    example = "Please recommend: \n"

    crossover_prompt = "Please follow the instruction step-by-step to generate a better prompt. \n \
                        1. Cross over the following prompts and generate two new prompts: \n \
                        Prompt 1: {prompt1} \n \
                        Prompt 2: {prompt2}. \n \
                        2. Mutate the prompt generated in Step 1 and generate \
                        a final prompt bracketed with <START> and <END>."

    # Evolutionary Optimization
    ###########################################################
    # Initialization
    print('The Algorithm is Starting!')
    print('Initializing the Population...')
    llm_ea = LLM_EA(pop_size, initial_prompt, crossover_prompt, llm_model, api_key)
    pop = llm_ea.initialize(example)
    # Evaluate the initial population
    problem.Sample_Test_Data()
    start_time = time.time()
    y_pop =  problem.Evaluate(pop)
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码执行时间为：", execution_time/60, "分钟")
    print('Initialization has been accomplished!')
    print('*************************************')

    # Evolution
    print('Evolution is starting!')
    Record_All = {'Iteration 0': {'Population':copy.deepcopy(pop),'Reward':copy.deepcopy(y_pop)}}
    print('Saving the Data')
    pickle.dump(Record_All, open(save_path, "wb"))
    for iter in range(max_iter):
        print('Generation' + str(iter))
        # Generate offspring
        offspring = llm_ea.crossover(pop)

        # Evaluate the offspring
        problem.Sample_Test_Data()
        y_offspring =  problem.Evaluate(offspring)
        
        # Environment Selection
        pop,y_pop = llm_ea.IBEA_selection(pop,y_pop,offspring,y_offspring)
        
        # Print and Save the data
        print('Accomplish iteration ' + str(iter))
        Record_ = {'Iteration ' + str(iter + 1): {'Population':copy.deepcopy(pop),'Reward':copy.deepcopy(y_pop)}}
        Record_All.update(Record_)
        print('Saving the Data')
        pickle.dump(Record_All, open(save_path, "wb"))
        print('*************************************')
    print('Evolution has been finished!')
    return pop, y_pop
