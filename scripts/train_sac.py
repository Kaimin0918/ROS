#!/usr/bin/env python3
#coding:utf-8

import rospy
import os
import time
import numpy as np
import random
import copy
import rospkg
import pickle 
import glob
import sys
import math
from collections import OrderedDict

# ROS & Torch 相關匯入
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

# 匯入我們最終版的自訂環境！
from turtlebot3_sac_env import TurtleBot3Env

# --- SAC 演算法核心 (與之前版本相同) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, 1)
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
    def forward(self, state, action):
        x_state_action = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = self.linear3_q1(x1)
        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = F.relu(self.linear2_q2(x2))
        x2 = self.linear3_q2(x2)
        return x1, x2

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std
    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std

class SAC(object):
    def __init__(self, state_dim, action_dim, gamma=0.99, tau=1e-2, alpha=0.2, hidden_dim=256, lr=0.00005):
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.q_loss, self.policy_loss, self.alpha_loss = 0.0, 0.0, 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- 使用的設備: {self.device} ---")
        
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)
        
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)
        return action.detach().cpu().numpy()[0]
        
    def update_parameters(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.q_loss = qf_loss.item()
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        pi, log_pi, _, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_loss = policy_loss.item()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_loss = alpha_loss.item()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        soft_update(self.critic_target, self.critic, self.tau)

    def save_checkpoint(self, episode_count, replay_buffer, models_path):
        if not os.path.exists(models_path): os.makedirs(models_path)
        checkpoint_path = os.path.join(models_path, f"checkpoint_ep{episode_count}.pth")
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optim.state_dict(),
        }, checkpoint_path)
        if len(replay_buffer) > 0:
            with open(os.path.join(models_path, f"replay_buffer_ep{episode_count}.pkl"), 'wb') as f:
                pickle.dump(replay_buffer, f)
            print(f"\n{'='*30}\n完整的訓練檢查點已儲存於 Ep {episode_count}\n{'='*30}")
        else:
            print(f"\n{'='*30}\n模型權重已儲存於 Ep {episode_count} (Replay Buffer 為空)\n{'='*30}")

    def load_checkpoint(self, episode, models_path, load_replay_buffer=True):
        checkpoint_path = os.path.join(models_path, f"checkpoint_ep{episode}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"錯誤：找不到 Ep {episode} 的檢查點檔案。")
            return False, None
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        hard_update(self.critic_target, self.critic)
        replay_buffer = None
        if load_replay_buffer:
            replay_buffer_path = os.path.join(models_path, f"replay_buffer_ep{episode}.pkl")
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, 'rb') as f:
                    replay_buffer = pickle.load(f)
                print(f"*** 成功載入 Ep {episode} 的完整檢查點 (包含 Replay Buffer) ***")
            else:
                print(f"警告：找到了模型權重但找不到 Replay Buffer 檔案。")
        else:
            print(f"*** 成功載入 Ep {episode} 的模型權重 (跳過 Replay Buffer) ***")
        return True, replay_buffer

    def load_checkpoint_for_transfer(self, episode, models_path):
        checkpoint_path = os.path.join(models_path, f"checkpoint_ep{episode}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"錯誤：找不到用於遷移的 Ep {episode} 檢查點檔案。")
            return False
        print(f"--- 開始從 Ep {episode} 進行遷移學習 ---")
        source_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        target_policy_dict = self.policy.state_dict()
        source_policy_dict = source_checkpoint['policy_state_dict']
        new_policy_dict = OrderedDict()
        for target_key, target_tensor in target_policy_dict.items():
            if target_key in source_policy_dict and source_policy_dict[target_key].shape == target_tensor.shape:
                new_policy_dict[target_key] = source_policy_dict[target_key]
                print(f"  [Policy] 成功遷移層: {target_key}")
            else:
                new_policy_dict[target_key] = target_tensor
                print(f"  [Policy] 跳過不匹配層: {target_key} (目標尺寸: {target_tensor.shape}, 來源尺寸: {source_policy_dict.get(target_key, 'N/A').shape if hasattr(source_policy_dict.get(target_key), 'shape') else 'N/A'})")
        self.policy.load_state_dict(new_policy_dict)
        target_critic_dict = self.critic.state_dict()
        source_critic_dict = source_checkpoint['critic_state_dict']
        new_critic_dict = OrderedDict()
        for target_key, target_tensor in target_critic_dict.items():
            if target_key in source_critic_dict and source_critic_dict[target_key].shape == target_tensor.shape:
                new_critic_dict[target_key] = source_critic_dict[target_key]
                print(f"  [Critic] 成功遷移層: {target_key}")
            else:
                new_critic_dict[target_key] = target_tensor
                print(f"  [Critic] 跳過不匹配層: {target_key} (目標尺寸: {target_tensor.shape}, 來源尺寸: {source_critic_dict.get(target_key, 'N/A').shape if hasattr(source_critic_dict.get(target_key), 'shape') else 'N/A'})")
        self.critic.load_state_dict(new_critic_dict)
        hard_update(self.critic_target, self.critic)
        print(f"*** 成功將 Ep {episode} 的駕駛技巧遷移至新模型！***")
        return True


if __name__ == '__main__':
    rospy.init_node('sac_train_main')
    
    env = TurtleBot3Env()
    
    # --- 使用者設定 ---
    TRANSFER_FROM_EPISODE = None
    FORCE_LOAD_EPISODE = None
    MAX_CHECKPOINTS_TO_KEEP = 99999
    CONSTANT_RANDOM_NOISE_STD = 0.05
    
    # ... (其他參數不變)
    max_episodes  = 30001
    max_steps   = 500
    replay_buffer_size = 50000
    RANDOM_STEPS_WARMUP = 2000
    BATCH_SIZE = 1024
    ACTION_V_MAX, ACTION_W_MAX = 0.22, 2.0
    initial_noise = 0.2 
    exploration_noise = initial_noise 
    noise_decay = 0.999 
    
    agent = SAC(env.state_dim, env.action_dim)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('blender_map_project')
    
    models_path = os.path.join(pkg_path, "models_checkpoints")
    run_log_dir = os.path.join(pkg_path, "tensorboard_logs", "sac_logs", 'run_' + time.strftime("%Ym%d-%H%M%S"))
    
    if not os.path.exists(models_path): os.makedirs(models_path)
    if not os.path.exists(run_log_dir): os.makedirs(run_log_dir)
    writer = SummaryWriter(run_log_dir)
    
    start_episode = 0
    skip_warmup = False

    # ... (載入邏輯與之前版本相同)
    print("--- 檢查訓練進度 ---")
    latest_episode_found = None
    if os.path.exists(models_path):
        checkpoints = glob.glob(os.path.join(models_path, "checkpoint_ep*.pth"))
        if checkpoints:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).split('ep')[1].split('.pth')[0]))
            latest_checkpoint = checkpoints[-1] 
            latest_episode_found = int(os.path.basename(latest_checkpoint).split('ep')[1].split('.pth')[0])
    if FORCE_LOAD_EPISODE is not None:
        print(f"--- 0. 啟動時光機！將強制從 Ep {FORCE_LOAD_EPISODE} 恢復訓練 ---")
        load_success, loaded_buffer = agent.load_checkpoint(FORCE_LOAD_EPISODE, models_path)
        if load_success:
            start_episode = FORCE_LOAD_EPISODE + 1
            if loaded_buffer: replay_buffer = loaded_buffer
            exploration_noise = 0.10
            agent.policy_optim.param_groups[0]['lr'] = 0.00001
            agent.critic_optim.param_groups[0]['lr'] = 0.00001
            print(f"--- 學習率已降低，探索雜訊已重置為 {exploration_noise:.4f} ---")
            skip_warmup = True
        else:
            rospy.logerr(f"時光機載入 Ep {FORCE_LOAD_EPISODE} 失敗！請檢查檔案是否存在。")
            exit()
    elif latest_episode_found is not None and TRANSFER_FROM_EPISODE is not None and latest_episode_found > TRANSFER_FROM_EPISODE:
        print(f"--- 1. 偵測到第二階段的最新進度為 Ep {latest_episode_found}，將從此進度繼續 ---")
        load_success, loaded_buffer = agent.load_checkpoint(latest_episode_found, models_path)
        if load_success:
            start_episode = latest_episode_found + 1
            if loaded_buffer: replay_buffer = loaded_buffer
            exploration_noise = initial_noise * (noise_decay ** start_episode)
            print(f"--- 探索雜訊已根據回合數 {start_episode} 更新為 {exploration_noise:.4f} ---")
            skip_warmup = True
        else:
            rospy.logerr(f"載入 Ep {latest_episode_found} 失敗！程式將終止。")
            exit()
    elif TRANSFER_FROM_EPISODE is not None:
        print(f"--- 2. 未找到第二階段的續練進度，將從 Ep {TRANSFER_FROM_EPISODE} 進行首次遷移學習 ---")
        load_success = agent.load_checkpoint_for_transfer(TRANSFER_FROM_EPISODE, models_path)
        if load_success:
            start_episode = TRANSFER_FROM_EPISODE
            skip_warmup = True
            exploration_noise = 0.15
            print(f"--- 探索雜訊已為遷移學習設定為 {exploration_noise:.4f} ---")
        else:
            rospy.logerr("遷移學習載入失敗！請檢查模型檔案是否存在。程式將終止。")
            exit()
    elif latest_episode_found is not None:
        print(f"--- 3. 偵測到第一階段的最新進度為 Ep {latest_episode_found}，將從此進度繼續 ---")
        load_success, loaded_buffer = agent.load_checkpoint(latest_episode_found, models_path)
        if load_success:
            start_episode = latest_episode_found + 1
            if loaded_buffer: replay_buffer = loaded_buffer
            exploration_noise = initial_noise * (noise_decay ** start_episode)
            print(f"--- 探索雜訊已根據回合數 {start_episode} 更新為 {exploration_noise:.4f} ---")
            skip_warmup = True
        else:
            rospy.logerr(f"載入 Ep {latest_episode_found} 失敗！程式將終止。")
            exit()
    else:
        print("--- 4. 未找到任何進度，將從頭開始訓練 ---")
        start_episode = 0
        skip_warmup = False
    
    print(f'State Dim: {env.state_dim}, Action Dim: {env.action_dim}')
    print(f'--- 開始訓練 (從 Episode {start_episode} 開始) ---')
    
    total_steps_counter = 0 # ★★★ 核心修正：在這裡初始化總步數計數器 ★★★
    is_collision_at_last_ep = False
    
    try:
        for ep in range(start_episode, max_episodes):
            state = env.reset(is_collision=is_collision_at_last_ep)
            if state is None or not isinstance(state, np.ndarray):
                print(f"Episode {ep} 因環境重置失敗而被跳過。")
                continue
            rewards_current_episode = 0.
            ep_orientation_errors = []
            if exploration_noise > 0.05:
                exploration_noise *= noise_decay
            for step in range(max_steps):
                total_steps_counter += 1
                if not skip_warmup and total_steps_counter < RANDOM_STEPS_WARMUP:
                    action = np.random.uniform(low=-1.0, high=1.0, size=env.action_dim)
                    if total_steps_counter == 1: print("--- 開始隨機探索熱身階段 ---")
                else:
                    if not skip_warmup and total_steps_counter == RANDOM_STEPS_WARMUP:
                        print("--- 熱身階段結束，開始使用神經網路策略 ---")
                    
                    action = agent.select_action(state)
                    
                    exploration_noise_component = np.random.normal(0, exploration_noise, size=env.action_dim)
                    random_noise_component = np.random.normal(0, CONSTANT_RANDOM_NOISE_STD, size=env.action_dim)
                    
                    action = (action + exploration_noise_component + random_noise_component).clip(-1.0, 1.0) 

                unnorm_action = np.array([
                    action_unnormalized(action[0], ACTION_V_MAX, 0.0), 
                    action_unnormalized(action[1], ACTION_W_MAX, -ACTION_W_MAX)
                ])
                next_state, reward, done, info = env.step(unnorm_action, action)
                ep_orientation_errors.append(info.get('orientation_error', 0))
                replay_buffer.push(state, action, reward, next_state, done)
                
                should_update_network = total_steps_counter > RANDOM_STEPS_WARMUP and len(replay_buffer) > BATCH_SIZE
                if should_update_network:
                    agent.update_parameters(replay_buffer, BATCH_SIZE)
                
                state = next_state
                rewards_current_episode += reward
                if done:
                    is_collision_at_last_ep = info.get('is_collision', False)
                    break

            if ep_orientation_errors:
                avg_orientation_error = np.mean([e for e in ep_orientation_errors if e is not None])
                avg_orientation_error_deg = math.degrees(avg_orientation_error)
                writer.add_scalar('metrics/avg_orientation_error_deg', avg_orientation_error_deg, ep)

            writer.add_scalar('reward/score_per_episode', rewards_current_episode, ep)
            if 'should_update_network' in locals() and should_update_network:
                writer.add_scalar('loss/q_loss', agent.q_loss, ep)
                writer.add_scalar('loss/policy_loss', agent.policy_loss, ep)
                writer.add_scalar('loss/alpha_loss', agent.alpha_loss, ep)
            
            print(f'Episode: {ep}, Reward: {rewards_current_episode:.2f}, Steps: {step+1}')
            print(f'    └─ Exp Noise: {exploration_noise:.4f}, Const Noise: {CONSTANT_RANDOM_NOISE_STD:.2f}, LR: {agent.policy_optim.param_groups[0]["lr"]:.6f}')
            
            # --- 檢查點儲存 ---
            is_restart_episode = (ep % 180 == 0 and ep > 0)
            is_save_episode = (ep % 60 == 0 and ep > 0)

            if is_restart_episode or is_save_episode:
                if is_restart_episode:
                    print(f"\n{'='*60}\n達到第 {ep} 回合的計畫性重啟點，儲存最後的檢查點...\n{'='*60}")
                
                agent.save_checkpoint(ep, replay_buffer, models_path)
            
            if is_restart_episode:
                print("檢查點儲存完畢。程式將正常退出，由外部腳本自動重啟。")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n使用者手動中斷訓練...")
    finally:
        print("執行最終儲存...")
        # 確保即使在迴圈開始前中斷，ep 變數也存在
        final_episode = ep if 'ep' in locals() else start_episode
        agent.save_checkpoint(final_episode, replay_buffer, models_path)
        writer.close()
        print("最終進度儲存完畢。程式結束。")

