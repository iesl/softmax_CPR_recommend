import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#grid_run = 'old'
grid_run = 'new'

#input_dir = './hyper_results'
if grid_run == 'new':
    input_dir = './hyper_results_hsz_128'
    output_file = 'small_results_para_128.csv'
elif grid_run == 'old':
    input_dir = './hyper_results_hsz'
    #output_file = 'small_results_para_max.csv'
    output_file = 'small_results_para.csv'
major_metric = 'ndcg@10'

must_contain_file_name = '_bias_fixed'
cannot_contain_file_name = 'lR_bias_fixed'
only_include_file_names = {'softmax_CPR_Mi_R100R_bias_fixed': 'Softmax + CPR:100 + Mi', 'softmax_C_bias_fixed': 'Softmax + C', 'softmax_Mi_bias_fixed': 'Softmax + Mi', 'softmax_bias_fixed': 'Softmax' }
if grid_run == 'new':
    name_d2_model_name = {'SAS': 'SASRec', 'GRU': 'GRU4Rec', 'GRUORG': 'GRU4RecORG', 'RepeatNet': 'RepeatNet'}
elif grid_run == 'old':
    name_d2_model_name = {'SAS': 'SASRec', 'GRU': 'GRU4Rec'}

dataset_d2_new = {'amazon-beauty': 'beauty', 'yelp2018': 'yelp', 'ml10m': 'ml-10m'}
including_dataset = set(['ml1m', 'beauty', 'game', 'steam'])
excluding_dataset = set([])
#excluding_dataset = set(['ml-10m'])
#excluding_dataset = set(['ml-10m','steam'])

#results_set = 'val,test'
results_set = 'test'

def parse_line(line, dataset_prefix):
    results = line.split('    ')
    out_dict = {}
    for result in results:
        #print(result)
        if ' : ' not in result:
            continue
        metric, score = result.split(' : ')
        out_dict[dataset_prefix + metric] = float(score)
    return out_dict

def get_best_score(f_in, dataset_name):
    result_line_num = 6
    lines = f_in.readlines()
    assert len(lines) % result_line_num == 0
    num_results = int(len(lines) / result_line_num)
    all_val_score, all_test_score, all_parameters, all_para_str = [], [], [], []
    for i in range(num_results):
        parameters = lines[i*6]
        val_score = parse_line(lines[i*6+2], dataset_name+'_val_')
        test_score = parse_line(lines[i*6+4], dataset_name+'_test_')
        #para_d2_val = {}
        #for para_val in parameters.split(', '):
        #    para, val = para_val.split(':')
        #    para_d2_val['para_'+dataset_name+ '_' +para] = val
        all_val_score.append(val_score)
        all_test_score.append(test_score)
        #all_parameters.append(para_d2_val)
        if grid_run == 'new':
            parameters = ', '.join(parameters.split(', ')[1::2])
        all_para_str.append(parameters)

    #return all_val_score, all_test_score, all_parameters, all_para_str
    return all_val_score, all_test_score, all_para_str

output_dict = {}
for filename in os.listdir(input_dir):
    f = os.path.join(input_dir, filename)
    if not os.path.isfile(f):
        continue
    f_name_list = filename.replace('GRU_d01','GRU-d01').split('_')
    model_name = f_name_list[1]
    dataset_name = f_name_list[2]
    if dataset_name in dataset_d2_new:
        dataset_name = dataset_d2_new[dataset_name]
    if dataset_name not in including_dataset:
        continue
    if len(f_name_list)>3:
        method_name = '_'.join(f_name_list[3:])
        if method_name not in only_include_file_names:
            continue
        if cannot_contain_file_name in method_name:
            continue
        if must_contain_file_name not in method_name:
            continue
        method_name = only_include_file_names[method_name]
        model_name = name_d2_model_name[model_name]
    else:
        method_name = 'Softmax'
    with open(f) as f_in:
        all_val_score, all_test_score, all_para_str = get_best_score(f_in, dataset_name)
        for i in range(len(all_para_str)):
            para_str = all_para_str[i]
            #parameters = all_parameters[i]
            test_score = all_test_score[i]
            val_score = all_val_score[i]
            index = (model_name, method_name, para_str)
            if index not in output_dict:
                output_dict[index] = {}
            #output_dict[index].update(parameters)
            if 'val' in results_set:
                output_dict[index].update(val_score)
            if 'test' in results_set:
                output_dict[index].update(test_score)


df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.sort_index()
#metric_list = ['ndcg', 'hit', 'recall', 'mrr', 'precision']
#metric_list = ['ndcg', 'hit', 'mrr']
metric_list = ['ndcg', 'hit']
#uniq_datasets = ['steam', 'beauty', 'game', 'ml1m']
#prob_datasets = ['tmall', 'twitch', 'ml-10m', 'book', 'yoochoose']
#prob_datasets = ['twitch', 'ml-10m', 'book', 'yoochoose']
#for metric in metric_list:
#    if 'val' in results_set:
#        val_col_names = ['val_'+ x +'_'+metric+'@10' for x in uniq_datasets if x not in excluding_dataset]
#        df['uniq_val_prod_'+metric] = df[val_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(val_col_names))
#    if 'test' in results_set:
#        test_col_names = ['test_'+ x +'_'+metric+'@10' for x in uniq_datasets if x not in excluding_dataset]
#        df['uniq_test_prod_'+metric] = df[test_col_names].prod(min_count=len(test_col_names),axis=1).pow(1.0/len(test_col_names))

df = df.reset_index()
#print(df)
#df = df.reset_index(names=['index_model','index_method','index_para'])
if grid_run == 'new':
    all_para = ['para_Hidden size', 'para_Batch size']
    plot_para = ['para_Hidden size']
else:
    all_para = ['para_Dropout', 'para_Learning rate', 'para_Batch size']
    plot_para = ['para_Dropout', 'para_Learning rate', 'para_Batch size']
new = df['level_2'].str.split(', ', n = len(all_para)-1, expand=True)

#print(new)

for i in range(len(all_para)):
    para = all_para[i]
    df[para] = new[i].str.split(':', n = 1, expand=True)[1]
#df['para_dropout'] = new[0].str.split(':', n = 1, expand=True)[1]
#df['para_learning_rate'] = new[1].str.split(':', n = 1, expand=True)[1]
#df['para_batch_size'] = new[2].str.split(':', n = 1, expand=True)[1]

#print(df)

for metric in metric_list:
    if 'val' in results_set:
        val_col_names = [x for x in df.columns if 'val_' in x  and '_'+metric in x]
        df['val_prod_'+metric] = df[val_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(val_col_names))
    if 'test' in results_set:
        test_col_names = [x for x in df.columns if 'test_' in x  and '_'+metric in x]
        df['test_prod_'+metric] = df[test_col_names].prod(min_count=len(test_col_names),axis=1).pow(1.0/len(test_col_names))
    #print(val_col_names)
    #print(df[val_col_names].prod(min_count=len(val_col_names),axis=1))
#print(df)

rest_columns = []
for x in df.columns:
    for metric in metric_list+['level', 'para']:
        if metric in x:
            rest_columns.append(x)
            break

df = df[rest_columns]

df['method_name'] = df[ ['level_0','level_1'] ].agg(', '.join, axis=1)

marker_arr = ['v', 'o', 's', '+', 'x',  '>', '<', '^']
linestyle_arr = ['-', '--', ':', '-.']

def plot_fig(method_d2_x_y, method_arr, method_suffix):
    plt.figure()
    plot_count = 0
    for method in method_arr:
    #for method in method_d2_x_y:
        #valid_method = False
        #for valid_name in method_arr:
        #    if valid_name in method:
        #        valid_method = True
        #        break
        #if not valid_method:
        #    continue
        x_arr = method_d2_x_y[method]['x']
        y_arr = method_d2_x_y[method]['y']
        x_arr, y_arr = zip( *sorted( zip(x_arr, y_arr), key=lambda x:x[0] ) )
        marker = marker_arr[plot_count % len(marker_arr)]
        linestyle = linestyle_arr[plot_count % len(linestyle_arr)]
        if method == 'RepeatNet, Softmax':
            method = 'RepeatNet'

        plt.plot(x_arr, y_arr, label=method, marker=marker, linestyle=linestyle)
        plot_count += 1
    plt.legend(fontsize=11)
    #plt.ylim(bottom=0)
    plt.xlabel(para.replace('para_', ''), fontsize=16)
    plt.ylabel('NDCG@10', fontsize=16)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.savefig(output_file.replace('.csv',para+'_'+method_suffix+'.png'))


for para in plot_para:
    #df.groupby(['level_0', 'level_1', para] ).mean().to_csv(output_file.replace('.csv',para+'.csv'))
    df_para = df.groupby(['method_name', para] ).mean()
    #df_para = df.groupby(['method_name', para] ).max()
    df_para = df_para.reset_index()
    
    method_d2_x_y = {}
    for index, row in df_para.iterrows():
        method_name = row['method_name']
        if method_name not in method_d2_x_y:
            method_d2_x_y[method_name] = {'x': [], 'y': []}
        x = float(row[para])
        y = float(row['test_prod_ndcg'])
        method_d2_x_y[method_name]['x'].append(x)
        method_d2_x_y[method_name]['y'].append(y)

    #GRU_method_order = ['RepeatNet, Softmax', 'GRU4Rec, Softmax + Mi', 'GRU4Rec, Softmax + C', 'GRU4Rec, Softmax + CPR:100 + Mi', 'GRU4RecORG, Softmax']
    #GRU_method_order = ['RepeatNet, Softmax', 'GRU4Rec, Softmax + Mi', 'GRU4RecORG, Softmax']
    GRU_method_order = ['RepeatNet, Softmax', 'GRU4Rec, Softmax + Mi', 'GRU4Rec, Softmax + C', 'GRU4Rec, Softmax + CPR:100 + Mi']
    SAS_method_order = ['SASRec, Softmax + Mi', 'SASRec, Softmax + C', 'SASRec, Softmax + CPR:100 + Mi']
    #SAS_method_order = ['SAS4Rec, Softmax + Mi']
    plot_fig(method_d2_x_y, GRU_method_order, 'GRU')
    plot_fig(method_d2_x_y, SAS_method_order, 'SAS')
    df_para.to_csv(output_file.replace('.csv',para+'.csv'))


df[rest_columns].to_csv(output_file)
