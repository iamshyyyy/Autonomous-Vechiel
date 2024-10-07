import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def load_data(file_path):
    """
    加载数据
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
            data = data['reward'][:1995]
    elif file_path.endswith('.pickle'):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            data = data['rewards'][:1995]

    data = pd.DataFrame(data, columns=['Reward'])
    data['EWMA_Reward'] = data['Reward'].ewm(span=20, adjust=False).mean()
    return data['EWMA_Reward']


def get_data():
    """
    获取并处理数据
    """
    # 加载数据
    sac_training_data_1 = load_data('check_point_low_traffic/sac/train_1/saved_training_dict.json')
    sac_training_data_2 = load_data('check_point_low_traffic/sac/train_2/saved_training_dict.json')

    sq_training_data_1 = load_data('check_point_low_traffic/Sq_v1/train1/saved_training_dict.json')
    sq_training_data_2 = load_data('check_point_low_traffic/Sq_v1/train2/saved_training_dict.json')
    sq_training_data_3 = load_data('check_point_low_traffic/Sq_v1/train3/saved_training_dict.json')

    PPO_data = load_data('check_point_low_traffic/PPO/train_1/checkpoint_ppo_.pickle')
    attention_training = load_data('check_point_low_traffic/attention/saved_training_dict (1).json')

    #转换数据格式
    sac_data = np.array([sac_training_data_1, sac_training_data_2, sac_training_data_1])
    sq_data = np.array([sq_training_data_1, sq_training_data_2, sq_training_data_3])
    ppo_data = np.array([PPO_data, PPO_data,PPO_data])
    attention_data = np.array([attention_training, attention_training,attention_training])

    return sq_data, sac_data,ppo_data,attention_data
    # return sq_data,sac_data

data = get_data()
# 创建数据帧
df = []
labels = ['OURS', 'SAC', 'PPO', 'Attention']
# labels = ['OURS','SAC']
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode', value_name='reward'))
    df[i]['algo'] = labels[i]
df = pd.concat(df)  # 合并
print(df)
sns.lineplot(x="episode", y="reward", hue="algo", style="algo", data=df)
plt.title("Episode Reward")
plt.show()
