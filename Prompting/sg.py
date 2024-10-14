import numpy as np
import os
import datetime


def add_days(start_day, delta_days):
    date_1 = datetime.datetime.strptime(start_day, '%B %d, %Y')
    end_date = date_1 + datetime.timedelta(days=delta_days)
    out = end_date.strftime('%B %d, %Y')

    return out


def generate_numerical(raw_folder, save_path, mode="test", obs_length=15):
    raw_data = np.load(os.path.join(raw_folder, "sg_raw_" + mode + ".npy"))
    data_x = []
    data_y = []
    for i in range(len(raw_data)):
        data = raw_data[i]
        number_of_instance = len(data) - obs_length
        for j in range(number_of_instance):
            y = data[obs_length + j]
            x = data[j: obs_length + j]
            data_x.append(x)
            data_y.append(y)

    data_x = np.reshape(data_x, [-1, obs_length])
    np.save(os.path.join(save_path, mode + "_" + str(obs_length) + "_x.npy"), data_x)
    data_y = np.reshape(data_y, [-1, 1])
    np.save(os.path.join(save_path, mode + "_" + str(obs_length) + "_y.npy"), data_y)


def output_sentence(target_usage):
    out = f"The patient is classified as having {target_usage} condition."
    return out


def input_sentence(usage, poi_id, start_date, obs_length):
    out = f"The following ECG signal of length 10 seconds, sampled at 8.3 Hz, was recorded. The signal consists of multiple leads representing different heart activities. The possible diagnoses based on the ECG signal are: NORM (Normal), MI (Myocardial Infarction), STTC (ST-T Change), CD (Conduction Disturbance), and HYP (Hypertrophy). **ECG Data: "
    

    return out


def generate_prompt(raw_folder, save_path, mode="train", obs_length=15, first_day="January 1, 2012"):
    raw_data = np.load(os.path.join(raw_folder, "sg_raw_" + mode + ".npy"))
    data_x_prompt = []
    data_y_prompt = []
    for i in range(len(raw_data)):
        data = raw_data[i]
        number_of_instance = len(data) - obs_length
        for j in range(number_of_instance):
            start_day = add_days(first_day, j)
            y = data[obs_length + j]
            x = data[j: obs_length + j]
            data_y_prompt.append(output_sentence(y))
            data_x_prompt.append(input_sentence(x, i+1, start_day, obs_length))

    with open(os.path.join(save_path, mode + "_15_x_prompt.txt"), "w") as f:
        for i in data_x_prompt:
            f.write(i + "\n")
        f.close()

    with open(os.path.join(save_path, mode + "_y_prompt.txt"), "w") as f:
        for i in data_y_prompt:
            f.write(i + "\n")
        f.close()
