import pandas as pd
import os
import glob
import numpy as np

input_dir = './slog/'
output_file = 'time.csv'
file_prefix = 'time_book_'

#for filename in os.listdir(input_dir+file_prefix+'*'):
print(input_dir+file_prefix+'*')
for filename in glob.glob(input_dir+file_prefix+'*'):
    if not os.path.isfile(filename):
        continue
    #assert '_bias_fixed' in filename
    #print(filename)
    f_name_list = filename.split('_')
    model_name = f_name_list[2]
    method_name = '_'.join(f_name_list[3:-3])
    val_time_arr = []
    train_time_arr = []
    with open(filename) as f_in:
        for line in f_in:
            fields = line.split()
            if 'time:' in line:
                #print(line.split())
                time = float(fields[8][:-2])
                if fields[6] == 'training':
                    train_time_arr.append(time)
                elif fields[6] == 'evaluating':
                    val_time_arr.append(time)
                else:
                    print(fields[6])
                    assert False
            elif 'total parameters:' in line:
                #print(fields)
                num_para = int(fields[-1])
    print(model_name, method_name, num_para, np.mean(train_time_arr), np.mean(val_time_arr) )

    #time_book_SAS_MoSe_bias_fixed_8738347
