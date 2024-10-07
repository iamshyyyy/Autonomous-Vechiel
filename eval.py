import pandas as pd
import yaml
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from env import Sq_Carla_Env
from env import Sq_Carla_Env_Graph
from env import Sq_Carla_Env_attention
from Agent.sac import SAC
from train.train_SAC import action

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
sac_path = "../check_point_low_traffic/sac/train_1/"
state_dim = 0
cfg = yaml.safe_load(open("../env/config.yaml", "r"))
Normal_env = Sq_Carla_Env.CarlaEnv(cfg=cfg, host="localhost")

#初始化智能体
sac = SAC(
    n_channels=12,
    state_dim=5+12,
    action_dim=Normal_env.action_space.shape[0],
    hidden_dim_a=512,
    hidden_dim_c=512,
    alpha = 0.2,
    reward_scale=10,
    gamma = 0.99, # 0.998
    tau = 0.001, # 0.001
    q_lr = 3e-4,
    policy_lr = 3e-4,
    batch_size = 512,
)
sac.load(sac_path)

def eval(agent,env,agent_name):
    #初始化指标
    results = []
    collisions = 0
    success = 0
    Each_steps = []
    total_episodes = 100

    test_rewards = []
    for episode in range(100):
        state = env.reset()
        episode_steps = 0
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            episode_steps += 1
            if 'collision' in info and info['collision']:
                collisions += 1
            if 'success' in info and info['success']:
                success += 1
                Each_steps.append(episode_steps)
        test_rewards.append(total_reward)

    #计算最终指标
    collision_rate = collisions / total_episodes
    success_rate = success / total_episodes
    mean_done_steps = sum(Each_steps) / len(Each_steps)
    print(f"{agent_name}---collision_rate:{collision_rate}, success_rate:{success_rate}. mean_done_steps:{mean_done_steps}")

    results.append({
        'algorithm': agent_name,
        'collision_rate': collision_rate,
        'success_rate': success_rate,
        'mean_done_steps': mean_done_steps
    })
    results_df = pd.DataFrame(results)
    results_df.to_csv('sac_test_results.csv', index=False)



eval(sac, Normal_env, agent_name="SAC")
