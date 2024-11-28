from Problems import RCBench
from Algorithms.MOEAD_LLM import MOEAD_LLM
import time
import pickle
import json

# Parameter Settings
#########################################################################
# Set the api_key for the LLM
# GPT setting
openai_key = " "
llm_model = 'gpt'
# GLM setting
# openai_key = " "
# llm_model = 'glm'

# Set the Dataset and Objective Function
Data_Obj = [['Movie','Acc_Div'],
            ['Game','Acc_Div'],
            ['Bundle','Acc_Div'],
            ['Movie','Acc_Fair'],
            ['Game','Acc_Fair'],
            ['Bundle','Acc_Fair'],
            ['Movie','Acc_Div_Fair'],
            ['Game','Acc_Div_Fair'],
            ['Bundle','Acc_Div_Fair']]
# Data_Obj = [['Game','Acc_Div_Fair'],
#             ['Bundle','Acc_Div_Fair']]
seed_ = 625

# Run Experiments
#########################################################################
time_record = {}
for setting in Data_Obj:
    # Setting Optimization Tasks
    func = eval('RCBench.' + setting[1])
    with open(f"Dataset/" + setting[0] + "/train_seed_" + str(seed_) + ".json", 'r') as json_file:
        train_data = json.load(json_file)
    bench = func(train_data,10,openai_key,llm_model='gpt')

    # Evolutionary Optimization
    start_time = time.time()
    Pop, Obj = MOEAD_LLM(bench, 
                        20, # Max number of iterations
                        10, # Population size
                        5,  # Number of weights in each subset
                        openai_key, 
                        'gpt',
                        'Results/' + setting[0] + '/MOEAD-LLM_' + setting[1] + '_Seed_' + str(seed_)
                        )
    end_time = time.time()
    print('Consumed Time: ' + str(end_time - start_time))
    time_record.update({setting[0] + ' & ' + setting[1]: end_time - start_time})
pickle.dump(time_record, open('Results/TimeConsumption_MOEAD-LLM_Seed_' + str(seed_), "wb"))