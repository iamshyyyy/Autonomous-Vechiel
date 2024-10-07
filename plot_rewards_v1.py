import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
sns.set()
import pickle
import pandas as pd
from plot_data import ppo_data

# 加载SAC算法的奖励记录
with open('check_point_low_traffic/sac/train_1/saved_training_dict.json', 'r') as file:
    sac_training_data_1 = json.load(file)
    sac_training_data_1 = sac_training_data_1['reward'][:1995]
with open('check_point_low_traffic/sac/train_2/saved_training_dict.json', 'r') as file:
    sac_training_data_2 = json.load(file)
    sac_training_data_2 = sac_training_data_2['reward'][:1995]
#加载自己的算法
with open('check_point_low_traffic/Sq_v1/train1/saved_training_dict.json', 'r') as file:
    sq_training_data_1 = json.load(file)
    sq_training_data_1 = sq_training_data_1['reward'][:1995]
with open('check_point_low_traffic/Sq_v1/train2/saved_training_dict.json', 'r') as file:
    sq_training_data_2 = json.load(file)
    sq_training_data_2 = sq_training_data_2['reward'][:1995]
with open('check_point_low_traffic/Sq_v1/train3/saved_training_dict.json', 'r') as file:
    sq_training_data_3 = json.load(file)
    sq_training_data_3 = sq_training_data_3['reward'][:1995]
# 加载PPO算法的检查点数据
with open('check_point_low_traffic/PPO/train_1/checkpoint_ppo_.pickle', 'rb') as file:
    PPO_data = pickle.load(file)
    PPO_data = PPO_data['rewards'][:1995]


def get_data():
    sac_data = np.array([sac_training_data_1,
                         sac_training_data_2,
                         sac_training_data_1])

    sq_data = np.array([sq_training_data_1,
                      sq_training_data_2,
                      sq_training_data_3])

    ppo_data = np.array([PPO_data,
                      PPO_data,
                      PPO_data])
    return sac_data, sq_data, ppo_data

data = get_data()
label = ['SAC', 'Ours', 'PPO']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='loss'))
    df[i]['algo']= label[i]
df=pd.concat(df) # 合并
print(df)
sns.lineplot(x="episode", y="loss", hue="algo", style="algo",data=df)
plt.title("some loss")
plt.show()