import pandas as pd
import os

input_dir = '../hyper_results'
output_file = 'all_results_2.csv'
major_metric = 'ndcg@10'

must_contain_file_name = '_bias_fixed'
cannot_contain_file_name = 'lR_bias_fixed'

dataset_d2_new = {'amazon-beauty': 'beauty', 'yelp2018': 'yelp', 'ml10m': 'ml-10m'}
excluding_dataset = set([])
#excluding_dataset = set(['ml-10m'])
#excluding_dataset = set(['ml-10m','steam'])

#results_set = 'val,test'
results_set = 'test'

def parse_line(line):
    results = line.split('    ')
    out_dict = {}
    for result in results:
        #print(result)
        if ' : ' not in result:
            continue
        metric, score = result.split(' : ')
        out_dict[metric] = float(score)
    return out_dict

def get_best_score(f_in):
    result_line_num = 6
    lines = f_in.readlines()
    assert len(lines) % result_line_num == 0
    num_results = int(len(lines) / result_line_num)
    max_metric = 0
    best_val_score, best_test_score, best_parameters = [], [], []
    for i in range(num_results):
        parameters = lines[i*6]
        val_score = parse_line(lines[i*6+2])
        test_score = parse_line(lines[i*6+4])
        if val_score[major_metric] > max_metric:
            max_metric = val_score[major_metric]
            best_val_score = val_score
            best_test_score = test_score
            best_parameters = parameters
    return best_val_score, best_test_score, best_parameters

output_dict = {}
for filename in os.listdir(input_dir):
    f = os.path.join(input_dir, filename)
    if not os.path.isfile(f):
        continue
    #if 'bias_fixed' not in filename and 'Repeat' not in filename:
    #    continue
    f_name_list = filename.replace('GRU_d01','GRU-d01').split('_')
    model_name = f_name_list[1]
    dataset_name = f_name_list[2]
    if dataset_name in dataset_d2_new:
        dataset_name = dataset_d2_new[dataset_name]
    if dataset_name in excluding_dataset:
        continue
    if len(f_name_list)>3:
        method_name = '_'.join(f_name_list[3:])
        if cannot_contain_file_name in method_name:
            continue
        if must_contain_file_name not in method_name:
            continue
    else:
        method_name = 'Softmax'
    index = (model_name, method_name)
    if index not in output_dict:
        output_dict[index] = {}
    para_col = 'best_parameter_'+dataset_name
    with open(f) as f_in:
        best_val_score, best_test_score, best_parameters = get_best_score(f_in)
        #output_dict['model_name'].append(model_name)
        #output_dict['method_name'].append(method_name)
        #if para_col not in output_dict:
        #    output_dict[para_col] = []
        #output_dict[para_col].append(best_parameters)
        #output_dict['best_parameter'].append(best_parameters)
        output_dict[index][para_col] = best_parameters
        for metric in best_val_score:
            if 'val' in results_set:
                val_col = 'val_'+dataset_name+'_'+metric
                output_dict[index][val_col] = best_val_score[metric]
            if 'test' in results_set:
                test_col = 'test_'+dataset_name+'_'+metric
                output_dict[index][test_col] = best_test_score[metric]
            #if val_col not in output_dict:
            #    output_dict[val_col] = []
            #    output_dict[test_col] = []
            #output_dict[val_col].append(best_val_score[metric])
            #output_dict[test_col].append(best_test_score[metric])


df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.sort_index()
#metric_list = ['ndcg', 'hit', 'recall', 'mrr', 'precision']
#metric_list = ['ndcg', 'hit', 'mrr']
metric_list = ['ndcg', 'hit']
duplicated_datasets = ['yoochoose', 'algebra', 'gowalla', 'steam', 'tmall']
uniq_datasets = ['twitch', 'book', 'ml-10m', 'beauty', 'game', 'ml1m', 'yelp']
#prob_datasets = ['tmall', 'twitch', 'ml-10m', 'book', 'yoochoose']
#prob_datasets = ['twitch', 'ml-10m', 'book', 'yoochoose']
for metric in metric_list:
    if 'val' in results_set:
        val_col_names = ['val_'+ x +'_'+metric+'@10' for x in duplicated_datasets if x not in excluding_dataset]
        df['dup_val_prod_'+metric] = df[val_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(val_col_names))

    if 'test' in results_set:
        test_col_names = ['test_'+ x +'_'+metric+'@10' for x in duplicated_datasets if x not in excluding_dataset]
        df['dup_test_prod_'+metric] = df[test_col_names].prod(min_count=len(test_col_names),axis=1).pow(1.0/len(test_col_names))
    
for metric in metric_list:
    if 'val' in results_set:
        val_col_names = ['val_'+ x +'_'+metric+'@10' for x in uniq_datasets if x not in excluding_dataset]
        df['uniq_val_prod_'+metric] = df[val_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(val_col_names))
    if 'test' in results_set:
        test_col_names = ['test_'+ x +'_'+metric+'@10' for x in uniq_datasets if x not in excluding_dataset]
        df['uniq_test_prod_'+metric] = df[test_col_names].prod(min_count=len(test_col_names),axis=1).pow(1.0/len(test_col_names))

#for metric in metric_list:
#    val_col_names = ['val_'+ x +'_'+metric+'@10' for x in prob_datasets if x not in excluding_dataset]
#    test_col_names = ['test_'+ x +'_'+metric+'@10' for x in prob_datasets if x not in excluding_dataset]
#    df['prob_val_prod_'+metric] = df[val_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(val_col_names))
#    df['prob_test_prod_'+metric] = df[test_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(test_col_names))

for metric in metric_list:
    if 'val' in results_set:
        val_col_names = [x for x in df.columns if 'val_' in x  and '_'+metric in x]
        df['val_prod_'+metric] = df[val_col_names].prod(min_count=len(val_col_names),axis=1).pow(1.0/len(val_col_names))
    if 'test' in results_set:
        test_col_names = [x for x in df.columns if 'test_' in x  and '_'+metric in x]
        df['test_prod_'+metric] = df[test_col_names].prod(min_count=len(test_col_names),axis=1).pow(1.0/len(test_col_names))
    #print(val_col_names)
    #print(df[val_col_names].prod(min_count=len(val_col_names),axis=1))

rest_columns = []
for x in df.columns:
    for metric in metric_list+['best']:
        if metric in x:
            rest_columns.append(x)
            break

df[rest_columns].to_csv(output_file)
