import main
import json
import time
import numpy as np
import threading

def run_agent(return_result_arr):

    #start_time = time.time()

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
    times = 10
    for i in range(times):
        run = main.deep_q_agent(days, step, False, data, save=save, save_path=save_path, load=load, load_path=load_path)
        return_result_arr.append(run)
    
if __name__ == "__main__":

    start_time = time.time()

    #return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)
    #it retuns profit
    
    
    amount = 50
    results = []
    run_test = False
    if run_test:
        threads = [threading.Thread(target=run_agent, args=[results])for i in range(amount)]
        for thread in threads:
            print(type(thread))
            thread.start()

        #Wait for each thread to finish
        for thread in threads:
            thread.join()

        print(results)

        result_json = json.dumps(results)
        with open('mutliple_runs_result_50runs.json', 'w') as outfile:
            outfile.write(result_json)

    else:#Not running, meaning loading tests
        with open('mutliple_runs_result_100runs.json', 'r') as outfile:
            result_json = outfile.read()
        results = json.loads(result_json)


    print(results)

    mean = np.mean(results)
    max = np.max(results)
    min = np.min(results)
    print("\n-----------------------------------------------------------------------------------------------\n")
    print(f"Mean: {mean}, Max: {max}, Min: {min}")
    print("\n-----------------------------------------------------------------------------------------------\n")
    print("Finished after: " + (str(np.round((time.time() - start_time),3))) + " seconds")
