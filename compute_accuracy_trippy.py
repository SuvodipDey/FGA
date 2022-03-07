#-----------------------------------
# Run Command
# python compute_accuracy_trippy.py
#-----------------------------------

import math
import json
import os

def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data
        
def getErrorIndexes(val):
    err = set()
    for i in range(len(val)):
        if val[i]==0:
            err.add(i)
    return err

def load_dataset_config(dataset_config):
    with open(dataset_config, "r", encoding='utf-8') as f:
        raw_config = json.load(f)
    return raw_config['class_types'], raw_config['slots'], raw_config['label_maps']

def getNotNoneIndexes(gt, slot_dict):
    id_list = []
    for dom in gt:
        for sl in gt[dom]:
            sl2 = sl.replace("book ", "")
            sl_key = dom+"-"+sl2
            id_list.append(slot_dict[sl_key])
    return id_list

def getAvgGoalAccuracy(id_list, val):  
    acc = -1
    if(len(id_list)>0):
        c = 0
        for i in id_list:
            c+=val[i]
        acc = c/float(len(id_list))
    return acc


filename = os.path.join("trippy", "trippy_result.json")
data = loadJson(filename)

total = 0
cor = 0
fga_cor = [0, 0, 0, 0]
turn_cor = 0
slot_acc = 0
lst_lambda = [0.25, 0.5, 0.75, 1.0]

for k in data:
    #Ignoring PMUL1455.json since it is not part of the official MultiWOZ test data
    if k=='PMUL1455.json':
        continue
        
    fga_prev = None
    err_set = []
    err_turn = 0
    
    for turn in data[k]:
        val = data[k][turn]
        c = 1
        for v in val:
            c = c*v
        total+=1
        cor+=c
        
        for l in range(len(lst_lambda)):
            fga = 1
            if(c==0):
                if(int(turn)==0):
                    #Type 1 error
                    #First turn is wrong
                    fga = 0
                elif(fga_prev==1):
                    #Type 1 error
                    #Last turn was correct i.e the error in current turn
                    fga = 0
                else:
                    err = getErrorIndexes(val)
                    diff = err_set.symmetric_difference(err)
                    if(len(diff)>0):
                        #Type 1 error
                        #There exists some undetected/false positive intent in the current prediction
                        fga = 0
                    else:
                        #Type 2 error
                        #Current turn is correct but source of the error is some previous turn
                        turn_diff = int(turn)-err_turn
                        fga = (1-math.exp(-lst_lambda[l]*turn_diff))
            fga_cor[l]+=fga
        if(fga==0):
            err_set = getErrorIndexes(val)
            err_turn = int(turn)
        else:
            turn_cor+=1
        
        sa = sum(val)/30.0
        slot_acc+=sa
        fga_prev = fga
    
filename = os.path.join("som-dst", "som-dst_result.json")
data_som = loadJson(filename)
dataset_config = os.path.join("trippy", "multiwoz21.json")
class_types, slots, label_maps = load_dataset_config(dataset_config)

slot_dict = {}
i = 0
for slot in slots:
    arr = slot.split("-")
    dom = arr[0]
    sl = arr[1].lower()
    sl = sl.replace("book_", "")
    sl_key = dom+"-"+sl
    slot_dict[sl_key] = i
    i+=1

avgGoalAcc = []
for k in data_som:
    for turn in data_som[k]:
        gt = data_som[k][turn]['gt']
        id_list = getNotNoneIndexes(gt, slot_dict)
        val = data[k][turn]
        
        aga = getAvgGoalAccuracy(id_list, val)
        if(aga>=0):
            avgGoalAcc.append(aga)
avg_goal_acc = round(sum(avgGoalAcc)*100.0/len(avgGoalAcc),2)

print("-"*40)
print(f"Total: {total}, Exact Match: {cor}, Turn Match: {turn_cor}")
print(f"Joint Acc = {round(cor*100.0/total,2)}, Slot Acc = {round(slot_acc*100.0/total,2)}, Avg. Goal Acc = {avg_goal_acc}")

for l in range(len(lst_lambda)):
    fga_acc = round(fga_cor[l]*100.0/total,2)
    print(f"FGA L={lst_lambda[l]} : {fga_acc}")
print("-"*40)