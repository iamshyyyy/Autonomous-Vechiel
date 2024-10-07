import json
import pickle
import yaml
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from env import Sq_Carla_Env_no_actor_reward_Sq
from Agent.sac import SAC
from plot_data import *
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
path = "check_point_low_traffic/sac/no_actor_camera/sq/"
cfg = yaml.safe_load(open("env/config.yaml", "r"))
env = Sq_Carla_Env_no_actor_reward_Sq.CarlaEnv(cfg=cfg, host="localhost")
obs = env.reset() # 初始化环境
action_dim = env.action_space.shape[0]
#获取obs维度，以此来确定state_dim

state_dim = 5


#初始化智能体
sac = SAC(
    n_channels=12,
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim_a=512,
    hidden_dim_c=512,
    alpha = 0.2,
    reward_scale=4,
    gamma = 0.99, # 0.998
    tau = 0.001, # 0.001
    q_lr = 3e-4,
    policy_lr = 3e-4,
    batch_size = 256,
)
# load checkpoint
try:
    with open(path + 'saved_training_dict.json', 'r') as jf:
        load_dict = json.load(jf)
        is_load_checkpoint = True
except:
    is_load_checkpoint = False


train_reward_record = deque(maxlen=100)
best_reward = float('-inf')
reward = []
acc_reward = 0
step = 0
Total_episodes = 2000
# 加载保存节点
if is_load_checkpoint:
    sac.log_alpha = torch.tensor(load_dict["model_log_alpha"], dtype=torch.float).cuda()
    sac.log_alpha.requires_grad = True
    sac.alpha = sac.log_alpha.exp()
    # sac.cnn.load_state_dict(torch.load(path + "/cnn.saved_params"))
    sac.critic.load_state_dict(torch.load(path + "critic.saved_params"))
    sac.critic_target.load_state_dict(torch.load(path + "critic.saved_params"))
    sac.policy.load_state_dict(torch.load(path + "policy.saved_params"))
    sac.policy_optim.load_state_dict(torch.load(path + "policy_optim.saved_params"))
    sac.critic_optim.load_state_dict(torch.load(path + "critic_optim.saved_params"))
    sac.alpha_optim.load_state_dict(torch.load(path + "alpha_optim.saved_params"))
    #with open('./check_point_low_traffic/buffer.pkl','rb') as bf:
    #     sac.memory.buffer = pickle.load(bf)
    #     sac.memory.pointer = load_dict["pointer"]
    #     print(sac.memory.pointer)
    #     print("加载完成")
    with open(path + 'train_reward_record.pkl', 'rb') as tf:
        train_reward_record = pickle.load(tf)
    best_reward = load_dict["best_reward"]
    reward = load_dict["reward"]


for i in tqdm(range(Total_episodes)):
    while True:
        print(f"---------Episode{i}------------step{step}------------")
        # action = np.random.normal(0, 1, size=action_dim).clip(-1, 1)
        action = sac.select_action(obs, False)
        # action = [0.5, 0]
        obs_, r, terminate, info= env.step(action)
        done = terminate
        acc_reward += r
        step += 1
        sac.memory.push((obs, action, r, obs_, done))
        obs = obs_
        if sac.memory.pointer >= 500:  # 防止冷启动，buffer大于一定数量才开始训练
            qf1_loss, qf2_loss, qf1_mean, qf2_mean, policy_loss, alpha_loss, alpha_tlogs = sac.update_parameters()# 更新sac参数
            print(f"q_loss: {qf1_loss}, q: {qf1_mean}, policy_loss: {policy_loss}, alpha_loss: {alpha_loss}, alpha: {alpha_tlogs}")
            if sac.memory.pointer % 500 == 0:
                # torch.save(sac.cnn.state_dict(), path + 'cnn.saved_params')
                torch.save(sac.critic.state_dict(), path + '/critic.saved_params')
                torch.save(sac.policy.state_dict(), path + 'policy.saved_params')
                torch.save(sac.critic_optim.state_dict(), path + 'critic_optim.saved_params')
                torch.save(sac.policy_optim.state_dict(), path + 'policy_optim.saved_params')
                torch.save(sac.alpha_optim.state_dict(), path + 'alpha_optim.saved_params')
                # sac.memory.save('./check_point_low_traffic/buffer.pkl')
                with open(path + '/train_reward_record.pkl', 'wb') as bf:
                    pickle.dump(train_reward_record, bf)
                saved_dict = {

                    "model_log_alpha": sac.log_alpha.item(),
                    "pointer": sac.memory.pointer,
                    'best_reward': best_reward,
                    'reward': reward,
                }
                with open(path + 'saved_training_dict.json', 'w') as jf:
                    json.dump(saved_dict, jf)
        if done:
            train_reward_record.append(acc_reward)
            reward.append(acc_reward)
            acc_reward = 0
            step = 0
            obs = env.reset()
            plt.plot(np.array(reward))
            plt.savefig('plot/train/sac/no_actor/train_sac_intersection'+ current_time+'.png')
            if np.mean(train_reward_record) > best_reward:
                best_reward = np.mean(train_reward_record)
                try:
                    sac.save('saved_params/sac/no_actor/sac_agent_params.pkl')
                    print(f'saved params with avg return {np.mean(train_reward_record)}')
                except:
                    pass
            break

print("训练结束")