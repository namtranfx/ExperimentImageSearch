

def Similarity(true_label, predicted_label):
    n_same = 0
    for sub_pred_label in predicted_label:
        for sub_true_label in true_label:
            if sub_pred_label == sub_true_label: n_same = n_same + 1
    if n_same >= 1: return True # Threshold
    return False

def precisionAtk(ground_true, retrieved, k)->float:
    if k == 0: 
        print("Cannot calculate precision@0")
        return 0
    relevant = 0
    for item in retrieved[:k]:
        if Similarity(ground_true, item): relevant = relevant + 1
        # print("compare two label: ", ground_true, "[", ground_true == item, "]", item)
    return float(relevant)/k
def AP(ground_true, retrieved, k_top)->float:
    # print("-----------one query----------------")
    # print("ground true = ", ground_true, "|retrieved = ", retrieved)
    sum_precision = 0
    n_relevant = 0
    for item in retrieved:
        if Similarity(ground_true, item): n_relevant = n_relevant + 1
    # print("Number of relevant item = ", n_relevant)
    # print("num of relevant item: ", n_relevant)
    if n_relevant == 0: return 0
    for i in range(1, k_top + 1):
        sum_precision = sum_precision + precisionAtk(ground_true, retrieved, i) * (1 if Similarity(ground_true, retrieved[i - 1]) else 0 ) 
    # print("AP============", float(sum_precision)/k_top)
    return float(sum_precision)/k_top
