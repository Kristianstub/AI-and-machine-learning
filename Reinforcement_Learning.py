import gymnasium as gym
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def not_slippy_four_grid():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=10000)

    env = model.get_env()
    obs = env.reset()
    accumulated, runs = 0, 0
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        accumulated += reward
        
        if done:
            
            obs = env.reset()

            runs += 1
            
    #env.render()
    mean_rewards, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    env.close()
    
    #print(f"Average reward: {mean_rewards} for non slippery 4x4 map")
    return model, env

def slippy_four_grid():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    policy_kwargs = dict(net_arch=[128, 128])
    model = stable_baselines3.DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.2,   
        policy_kwargs=policy_kwargs,
        buffer_size=100_000,      
        learning_rate=1e-3,       
        batch_size=64,           
        learning_starts=1000,    
        target_update_interval=500)

    model.learn(total_timesteps=100000)

    env = model.get_env()
    obs = env.reset()
    accumulated, runs = 0, 0
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        accumulated += reward
        
        if done:
            
            obs = env.reset()

            runs += 1
            print(f"Total reward: {accumulated}, average reward: {accumulated/(accumulated/runs)}")
    #env.render()
    mean_rewards, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    env.close()
    
    #print(f"Average reward: {mean_rewards} for slippery 4x4 map")
    return model, env
def eightxeight_grid():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

    policy_kwargs = dict(net_arch=[128, 128])
    model = stable_baselines3.DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.2,   
        policy_kwargs=policy_kwargs,
        buffer_size=100_000,      
        learning_rate=1e-3,       
        batch_size=64,           
        learning_starts=1000,    
        target_update_interval=500)

    model.learn(total_timesteps=100000)

    env = model.get_env()
    obs = env.reset()
    accumulated, runs = 0, 0
    for i in range(50000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        accumulated += reward
        
        if done:
            
            obs = env.reset()

            runs += 1
            
    mean_rewards, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    env.close()
    
    print(f"Average reward: {mean_rewards} for slippery 8x8 map")
    return model, env

def print_policy_grid(model, env):
    action_symbols = ['←', '↓', '→', '↑']
    policy_grid = []

    for state in range(env.observation_space.n):
        obs = np.array([state])
        action = model.predict(obs, deterministic=True)[0]
        action = int(action)
        policy_grid.append(action_symbols[action])

    size = 4
    for i in range(size):
        row = policy_grid[i*size:(i+1)*size]
        print(' '.join(row))
# model, env = not_slippy_four_grid()
# model1, env1 = slippy_four_grid()
model2, env2 = eightxeight_grid()
# mean_rewards1, _1 = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
# mean_rewards2, _2 = evaluate_policy(model1, model1.get_env(), n_eval_episodes=100)
mean_rewards3, _3 = evaluate_policy(model2, model2.get_env(), n_eval_episodes=100)


print("Policy grid for non slippery 4x4 map")
# print_policy_grid(model, env)
# print(f"Average reward: {mean_rewards1} for non slippery 4x4 map")
# print(f"Average reward: {mean_rewards2} for slippery 4x4 map")
print(f"Average reward: {mean_rewards3} for slippery 8x8 map")


# print("Policy grid for slippery 4x4 map")
# print_policy_grid(model1, env1)
