import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 data 是包含奖励数据的 pandas 数据帧
data = pd.DataFrame({
    'Episode': np.arange(1, 2001),
    'Reward': np.random.randn(2000).cumsum()  # 模拟奖励数据
})

# 计算移动平均
data['MA_Reward'] = data['Reward'].rolling(window=40, min_periods=1).mean()

# 绘制原始数据
plt.figure(figsize=(10, 6))
sns.lineplot(x='Episode', y='Reward', data=data, label='Original')

# 绘制移动平均
sns.lineplot(x='Episode', y='MA_Reward', data=data, label='Moving Average', color='red')

plt.title('Rewards Over Episodes with Moving Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()