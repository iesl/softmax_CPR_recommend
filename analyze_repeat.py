import pandas as pd
import matplotlib.pyplot as plt

#max_seq_len = 100
max_seq_len = 50
#max_user_num = 1000
max_user_num = 10000000000

out_image_folder = 'out_fig/'

#dataset_name = 'Retailrocket'
#input_path = "dataset/retailrocket/retailrocket.inter"
#item_col_name = 'item_id:token'
#user_col_name = 'visitor_id:token'
#time_col_name = "timestamp:float"

#dataset_name = 'Steam'
#input_path = "dataset/steam/steam.inter"
#item_col_name = 'product_id:token'
#user_col_name = 'user_id:token'
#time_col_name = "timestamp:float"

#dataset_name = 'Bridge to Algebra (2008-2009)'
#input_path = "dataset/algebra2008_2009/algebra2008_2009.inter"
#item_col_name = 'problem_step_name:token'
#user_col_name = 'student_id:token'
#time_col_name = "step_start_time:float"

#dataset_name = 'Gowalla'
#input_path = "dataset/gowalla/gowalla.inter"
#item_col_name = 'item_id:token'
#user_col_name = 'user_id:token'
#time_col_name = "timestamp:float"

dataset_name = 'Yoochoose-clicks'
input_path = "dataset/yoochoose-clicks/yoochoose-clicks.inter"
item_col_name = 'item_id:token'
user_col_name = 'session_id:token'
time_col_name = "timestamp:float"

#dataset_name = 'Tmall-buy'
#input_path = "dataset/tmall-buy/tmall-buy.inter"
#item_col_name = 'item_id:token'
#user_col_name = 'user_id:token'
#time_col_name = "timestamp:float"

#dataset_name = 'Twitch-100k'
#input_path = "dataset/Twitch-100k/Twitch-100k.inter"
#item_col_name = 'item_id:token'
#user_col_name = 'user_id:token'
#time_col_name = "time_start:float"

df = pd.read_table(input_path)
#user_d2_inter = df.set_index(user_col_name).to_dict('index')
#user_d2_inter = dict(zip(df[user_col_name],df[item_col_name]) )
#user_d2_inter = {k: list(zip(g[item_col_name].tolist(),g[time_col_name].tolist())) for k,g in df.groupby(user_col_name)}
user_d2_inter = {}
for x in range(len(df)):
    user = df.loc[x,user_col_name]
    item = df.loc[x,item_col_name]
    time = df.loc[x,time_col_name]
    user_d2_inter.setdefault(user, [])
    user_d2_inter[user].append([item, time])

seq_len_d2_stats = {}
for i in range(2,max_seq_len):
    seq_len_d2_stats[i] = {'uniq_uniq': 0, 'uniq_repeat': 0, 'repeat_uniq': 0, 'repeat_repeat': 0}

count = 0
for user in user_d2_inter:
    inter = user_d2_inter[user]
    #print(inter)
    inter_seq = sorted(inter, key=lambda x: x[1])
    start_to_repeat = False
    prev_set = set([inter_seq[0][0]])
    for i in range(1, min( len(inter_seq),max_seq_len-1 ) ):
        current_item = inter_seq[i][0]
        prev_repeat = start_to_repeat
        if current_item in prev_set:
            start_to_repeat = True
            if prev_repeat:
                seq_len_d2_stats[i+1]['repeat_repeat'] += 1
            else:
                seq_len_d2_stats[i+1]['uniq_repeat'] += 1
        else:
            if prev_repeat:
                seq_len_d2_stats[i+1]['repeat_uniq'] += 1
            else:
                seq_len_d2_stats[i+1]['uniq_uniq'] += 1

        prev_set.add(current_item)
    #print(inter_seq)
    count += 1
    if count > max_user_num:
        break
#print(seq_len_d2_stats)
g_stats = {'uniq_uniq': 0, 'uniq_repeat': 0, 'repeat_uniq': 0, 'repeat_repeat': 0}
pre_repeat_curve = []
pre_uniq_curve = []
x_value = []
for i in range(2,max_seq_len):
    rr_count = seq_len_d2_stats[i]['repeat_repeat']
    ur_count = seq_len_d2_stats[i]['uniq_repeat']
    ru_count = seq_len_d2_stats[i]['repeat_uniq']
    uu_count = seq_len_d2_stats[i]['uniq_uniq']
    repeat_repeat_ratio = 0
    uniq_repeat_ratio = 0
    if rr_count+ru_count > 0:
        repeat_repeat_ratio = rr_count / float(rr_count+ru_count)
    if ur_count+uu_count > 0:
        uniq_repeat_ratio = ur_count / float(ur_count+uu_count)
    #if min(rr_count+ru_count, ur_count+uu_count) < 100:
    #    break
    if i > 2:
        g_stats['repeat_repeat'] += rr_count
        g_stats['uniq_repeat'] += ur_count
        g_stats['repeat_uniq'] += ru_count
        g_stats['uniq_uniq'] += uu_count
        x_value.append(i)
        pre_repeat_curve.append(repeat_repeat_ratio)
        pre_uniq_curve.append(uniq_repeat_ratio)
    print(i)
    print(repeat_repeat_ratio, rr_count, ru_count)
    print(uniq_repeat_ratio, ur_count, uu_count)

print(g_stats)
print(g_stats['repeat_repeat'] / float(g_stats['repeat_repeat'] + g_stats['repeat_uniq']) )
print(g_stats['uniq_repeat'] / float(g_stats['uniq_repeat'] + g_stats['uniq_uniq']) )

plt.plot(x_value,pre_repeat_curve,label='Previous Repeat (Count '+str(g_stats['repeat_repeat'] + g_stats['repeat_uniq'])+')')
plt.plot(x_value,pre_uniq_curve,label='Previous Unique (Count '+str(g_stats['uniq_repeat'] + g_stats['uniq_uniq'])+')', linestyle='dashed')
plt.legend(fontsize=14)
plt.title(dataset_name, fontsize=20)
plt.xlabel('Sequence Length', fontsize=16 )
plt.ylabel('Last Repeat Ratio', fontsize=16 )
plt.savefig(out_image_folder+dataset_name+'_fig.png')
