import main
import json
import time
import numpy as np

if __name__ == "__main__":

    start_time = time.time()

    train = False #Train if true or Test if false
    override = False #If want custom data / days

    if train:
        file = open('2008till2022-10-15.json')
        days = 4000
    elif not train:
        file = open('2021till2022nov.json')
        days = 674
    if override:# Custom stuff here
        file = open('virtualdata.json')
        days = 50000

    data = json.load(file)

    save = False
    load = True
    save_path = "Q_EvalTest_FromOld.pth"
    load_path = "Q_EvalTest_FromOld.pth"

    step = 60*60

    #return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)
    #it retuns profit
    combined = []
    for i in range(100):
        run = main.deep_q_agent(days, step, False, data, save=save, save_path=save_path, load=load, load_path=load_path)
        combined.append(run)

    result_json = json.dumps(combined)
    with open('mutliple_runs_result_100runs.json', 'w') as outfile:
        outfile.write(result_json)

    #print(result)
    #with open('mutliple_runs_result_100runs.json', 'r') as outfile:
    #    result_json = outfile.read()
    
    result = json.loads(result_json)

    mean = np.mean(combined)
    max = np.max(combined)
    min = np.min(combined)
    print("\n-----------------------------------------------------------------------------------------------\n")
    print(f"Mean: {mean}, Max: {max}, Min: {min}")
    print("\n-----------------------------------------------------------------------------------------------\n")
    print("Finished after: " + (str(np.round((time.time() - start_time),3))) + " seconds")
