import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

def bce_np_evaluate(gt_output, output) :
    
    return accuracy_score(gt_output, output)
    
def metric_with_missing_rate(gt_text, predicted_text, dataset):
    output_data = []
    gt_data = []
    missing_count = 0
    ecg_dict = {'normal' : 0, 'abnormal' : 1, 'Normal' : 0, 'Abnormal' : 1}
    
    for i in range(len(gt_text)):
        predicted_line = predicted_text[i]
        gt_line = gt_text[i]
        out = predicted_line.split(" ")[-1][:].strip()
        gt_out = gt_line.split(" ")[-1][:-1].strip()
        
        tmp = [0,0]
        gt_tmp = [0,0]
        a = 0
        b = 0

        if out in ecg_dict :
            tmp[ecg_dict[out]] = 1
            a = 1

        if gt_out in ecg_dict :
            gt_tmp[ecg_dict[gt_out]] = 1
            b = 1
        if a == 1 and b == 1 : 
            output_data.append(tmp)
            gt_data.append(gt_tmp)
            
        else :
            missing_count += 1
    
    accuracy = bce_np_evaluate(np.array(output_data), np.array(gt_data))
    missing_rate = missing_count / len(gt_text)
    print(accuracy)
    return accuracy, 0, missing_rate















