import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, roc_auc_score


def np_evaluate(gt_output, pred_output):
    rmse = mean_squared_error(gt_output, pred_output, squared=False)
    mae = mean_absolute_error(gt_output, pred_output)

    return rmse, mae

def bce_np_evaluate(gt_output, output) :
    print(gt_output, output)
    
    logloss = roc_auc_score(gt_output, output)
    
    return logloss
    

def metric_with_missing_rate(gt_text, predicted_text, dataset):
    output_data = []
    gt_data = []
    missing_count = 0
    ecg_dict = {'normal' : 0, 'abnormal' : 1}

    for i in range(len(gt_text)):
        predicted_line = predicted_text[i]
        gt_line = gt_text[i]
        out = predicted_line.split(" ")[-1].strip()
        gt_out = gt_line.split(" ")[-1].strip()
        print(out)
        print(gt_out)
        
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
            
        print(tmp, gt_tmp)
        if a == 1 and b == 1 : 
            output_data.append(tmp)
            gt_data.append(gt_tmp)
            
        else :
            missing_count += 1

        # except Exception:
        #     missing_count += 1

    # output = np.reshape(output_data, [len(output_data), 1])
    # gt_output = np.reshape(gt_data, [len(gt_data), 1])

    
    log_loss = bce_np_evaluate(output_data, gt_data)
    missing_rate = missing_count / len(gt_text)

    return log_loss, 0, missing_rate

# def metric_with_missing_rate(gt_text, predicted_text, dataset):
#     output_data = []
#     gt_data = []
#     missing_count = 0

#     for i in range(len(gt_text)):
#         predicted_line = predicted_text[i]
#         gt_line = gt_text[i]
#         try:
#             if dataset == 'SG':
#                 out = int(predicted_line.split(" ")[3])
#                 gt_out = int(gt_line.split(" ")[3])
#             else:
#                 out = int(predicted_line.split(" ")[4])
#                 gt_out = int(gt_line.split(" ")[4])
#             output_data.append(out)
#             gt_data.append(gt_out)
#         except Exception:
#             missing_count += 1

#     output = np.reshape(output_data, [len(output_data), 1])
#     gt_output = np.reshape(gt_data, [len(gt_data), 1])

#     rmse, mae = np_evaluate(gt_output, output)
#     missing_rate = missing_count / len(gt_text)

#     return rmse, mae, missing_rate








