#-----------------------------------
# Run Command
# python compute_accuracy_trade_somdst.py
#-----------------------------------

import os
import json
import pandas as pd
import math

def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data
    
def getBeliefSet(ds):
    bs = set()
    for dom in ds:
        for slot in ds[dom]:
            t = dom+"-"+slot+"-"+ds[dom][slot]
            bs.add(t)
    return bs

# Slot Accuracy
def getSlotAccuracy(gt, pr):
    d1 = gt.difference(pr)
    d2 = pr.difference(gt)
    
    s1 = set([d.rsplit("-", 1)[0] for d in d1])
    s2 = set([d.rsplit("-", 1)[0] for d in d2])    
    
    set_i = s1.intersection(s2)
    acc = (30 - len(d1) - len(d2) + len(set_i))/30.0
    return acc

# Slot Accuracy Computation taken from TRADE model
def compute_acc(gold, pred):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = 30
    ACC = 30 - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

# Average Goal Accuracy
def getAvgGoalAccuracy(gt, pr):    
    set_i = gt.intersection(pr)
    acc = -1
    if(len(gt)>0):
        acc = len(set_i)/float(len(gt))
    return acc

# Flexible Goal Accuracy
def getFGA(gt_list, pr_list, turn_diff, L):
    gt = gt_list[-1]
    pr = pr_list[-1]
    diff1 = gt.symmetric_difference(pr)
    if len(diff1)==0: #Exact match
        return 1
    else:
        if len(gt_list)==1: 
            #Type 1 error
            #First turn is wrong
            return 0
        else:
            diff2 = gt_list[-2].symmetric_difference(pr_list[-2])
            if len(diff2)==0: 
                #Type 1 error
                #Last turn was correct i.e the error in current turn
                return 0
            else:
                tgt = gt.difference(gt_list[-2])
                tpr = pr.difference(pr_list[-2])
                if(not tgt.issubset(pr) or not tpr.issubset(gt)): 
                    #Type 1 error
                    #There exists some undetected/false positive intent in the current prediction
                    return 0
                else:
                    #Type 2 error
                    #Current turn is correct but source of the error is some previous turn
                    return (1-math.exp(-L*turn_diff))
    
def getModifiedBS(bs):
    bs_new = {}
    for k in bs:
        bs_new[k] = {}
        for slot in bs[k]:
            sl = slot
            v = bs[k][slot]
            if "book" in slot:
                sl = slot.split(' ')[1]
            bs_new[k][sl] = v
    return bs_new

def getModelAccuracy(model_name, dialog_data):
    dst_res_path = os.path.join(model_name, model_name+"_result.json")
    dst_res = loadJson(dst_res_path)
    
    joint_acc = 0
    slot_acc = 0
    avgGoalAcc = []
    fga = [0, 0, 0, 0]
    turn_acc = 0
    total = 0
    lst_lambda = [0.25, 0.5, 0.75, 1.0]
    
    for idx in dst_res:
        res = dst_res[idx]
        log = dialog_data[idx]['log']
        sys = " "
        
        gt_list = []
        pr_list = []
        error_turn = -1
        for turn in res:
            total+=1
            i = 2*int(turn)
            usr = log[i]['text'].strip()
            if(i>0):
                sys = log[i-1]['text'].strip()

            gt = getBeliefSet(res[turn]['gt'])
            pr = getBeliefSet(res[turn]['pr'])
            gt_list.append(gt)
            pr_list.append(pr)

            #print(f"Sys_{turn}: {sys}")
            #print(f"Usr_{turn}: {usr}")
            #print(f"GT_{turn}: {getModifiedBS(res[turn]['gt'])}")
            #print(f"PR_{turn}: {getModifiedBS(res[turn]['pr'])}")
            #print("-"*40)

            diff = gt.symmetric_difference(pr)
            m = 1 if len(diff)==0 else 0
            joint_acc+=m

            #sa = getSlotAccuracy(gt, pr)
            sa = compute_acc(gt, pr)
            slot_acc+=sa

            aga = getAvgGoalAccuracy(gt, pr)
            if(aga>=0):
                avgGoalAcc.append(aga)
            
            m = 0
            for l in range(len(lst_lambda)):
                m = getFGA(gt_list, pr_list, int(turn)-error_turn, lst_lambda[l])
                fga[l]+=m
            if(m==0):
                error_turn = int(turn)
            else:
                turn_acc+=1

    print(f"Total: {total}, Exact Match: {joint_acc}, Turn Match: {turn_acc}")
    joint_acc = round(joint_acc*100.0/total,2)
    slot_acc = round(slot_acc*100.0/total,2)
    avg_goal_acc = round(sum(avgGoalAcc)*100.0/len(avgGoalAcc),2)
    print(f"Joint Acc = {joint_acc}, Slot Acc = {slot_acc}, Avg. Goal Acc = {avg_goal_acc}")
    for l in range(len(lst_lambda)):
        fga_acc = round(fga[l]*100.0/total,2)
        print(f"FGA with L={lst_lambda[l]} : {fga_acc}")

#-----------------------------------

#Load raw data
dialog_data_file = os.path.join('data.json')
dialog_data = loadJson(dialog_data_file)

print("-"*40)
print("Trade :-")
getModelAccuracy("trade",  dialog_data)
print("-"*40)
print("SOM-DST :-")
getModelAccuracy("som-dst", dialog_data)
print("-"*40)
#-----------------------------------