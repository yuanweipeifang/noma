"""
NOMA Secure Transmission with Deep Reinforcement Learning
补充实验完整实现：窃听者强度变化 + 用户数量扩展

基于《实验设计建模》和《补充实验》文档要求
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import copy
import os
import json
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== 1. NOMA 安全传输环境 ====================

class NOMASecureEnv:
    """
    支持可变用户数量的NOMA安全传输环境
    """
    def __init__(self, 
                 num_users=2, 
                 P_max=10.0, 
                 noise_power=1.0,
                 R_min=1.0,
                 eavesdropper_gain_scale=1.0,
                 L1=1.0, L2=2.0, 
                 L_J1=0.5, L_J2=0.5,
                 L_e=1.5, L_Je=3.0,
                 use_jammer=True,
                 seed=None):
        """
        Args:
            num_users: 合法用户数量 (M)
            P_max: 总功率约束
            noise_power: 噪声功率 sigma^2
            R_min: 最低速率要求 (QoS)
            eavesdropper_gain_scale: 窃听者强度缩放因子
            L1, L2: 用户路径损耗 (当num_users>2时会扩展)
            L_J1, L_J2: Jammer到用户路径损耗
            L_e: 窃听者路径损耗
            L_Je: Jammer到窃听者路径损耗
            use_jammer: 是否使用友好干扰
        """
        self.num_users = num_users
        self.P_max = P_max
        self.noise_power = noise_power
        self.R_min = R_min
        self.eavesdropper_gain_scale = eavesdropper_gain_scale
        self.use_jammer = use_jammer
        
        # 路径损耗设置
        # 用户路径损耗：确保U2强于U1，后续用户依次设置
        self.L_users = [L1 + i * 0.5 for i in range(num_users)]  # U1, U2, ... 递增
        if num_users >= 2:
            self.L_users[1] = L2  # 确保第二个用户是强用户
        
        # Jammer到各用户的路径损耗
        self.L_J_users = [L_J1 + i * 0.1 for i in range(num_users)]
        
        # 窃听者相关路径损耗
        self.L_e = L_e
        self.L_Je = L_Je
        
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random
        
        # 状态维度：用户信道(2*M) + 窃听者信道(2) + Jammer链路(2*M + 2) + 约束信息(3)
        # 具体: |h_i|^2 (M), |g|^2 (1), |h_Ji|^2 (M), |g_J|^2 (1), R_min, P_max, noise
        self.state_dim = num_users + 1 + num_users + 1 + 3
        
        # 动作维度：M个用户功率 + 1个Jammer功率
        self.action_dim = num_users + 1 if use_jammer else num_users
        
        # 当前信道状态
        self.current_channels = None
        
    def reset(self):
        """生成新的信道状态"""
        # 用户信道：路径损耗 + 瑞利衰落
        self.h_users = []
        for i in range(self.num_users):
            h_tilde = (self.rng.randn() + 1j * self.rng.randn()) / np.sqrt(2)
            h = np.sqrt(self.L_users[i]) * h_tilde
            self.h_users.append(h)
        
        # 窃听者信道（应用强度缩放）
        g_tilde = (self.rng.randn() + 1j * self.rng.randn()) / np.sqrt(2)
        self.g = np.sqrt(self.L_e) * g_tilde * np.sqrt(self.eavesdropper_gain_scale)
        
        # Jammer到用户信道
        self.h_J_users = []
        for i in range(self.num_users):
            h_J_tilde = (self.rng.randn() + 1j * self.rng.randn()) / np.sqrt(2)
            h_J = np.sqrt(self.L_J_users[i]) * h_J_tilde
            self.h_J_users.append(h_J)
        
        # Jammer到窃听者信道
        g_J_tilde = (self.rng.randn() + 1j * self.rng.randn()) / np.sqrt(2)
        self.g_J = np.sqrt(self.L_Je) * g_J_tilde
        
        # 构建状态向量
        state = []
        # 用户信道增益 |h_i|^2
        for h in self.h_users:
            state.append(np.abs(h)**2)
        # 窃听者信道增益 |g|^2
        state.append(np.abs(self.g)**2)
        # Jammer到用户信道增益 |h_Ji|^2
        for h_J in self.h_J_users:
            state.append(np.abs(h_J)**2)
        # Jammer到窃听者信道增益 |g_J|^2
        state.append(np.abs(self.g_J)**2)
        # 约束信息
        state.append(self.R_min)
        state.append(self.P_max)
        state.append(self.noise_power)
        
        self.current_channels = {
            'h_users': self.h_users,
            'g': self.g,
            'h_J_users': self.h_J_users,
            'g_J': self.g_J,
            'state': np.array(state, dtype=np.float32)
        }
        
        return self.current_channels['state']
    
    def step(self, action):
        """
        执行动作，计算奖励和指标
        action: [a_1, a_2, ..., a_M] 用户功率分配
                如果use_jammer=True，最后一个是Jammer功率 [a_1, ..., a_M, a_J]
                Jammer有独立的功率预算 P_jammer_max = P_max
        返回: next_state, reward, done, info
        """
        action = np.clip(action, 0, 1)

        if self.use_jammer:
            # 用户功率和Jammer功率分别归一化
            # 用户功率部分
            user_action = action[:-1]
            user_sum = np.sum(user_action)
            if user_sum > 1.0:
                user_action = user_action / user_sum
            P_users = user_action * self.P_max

            # Jammer功率独立
            P_J = action[-1] * self.P_max  # Jammer也有P_max的独立预算
        else:
            user_sum = np.sum(action)
            if user_sum > 1.0:
                action = action / user_sum
            P_users = action * self.P_max
            P_J = 0.0

        # 计算各用户SINR和速率（合法用户可以进行SIC）
        gamma_users = []
        R_users = []
        SIC_success = []

        for i in range(self.num_users):
            h_i = self.h_users[i]
            h_Ji = self.h_J_users[i]
            h_gain = np.abs(h_i)**2
            h_J_gain = np.abs(h_Ji)**2

            if i == 0:
                # 弱用户U1：直接解码自己信号，承受其他所有用户干扰
                # 注意：Jammer只干扰窃听者，不干扰合法用户（协作Jamming）
                interference = sum([h_gain * P_users[j] for j in range(1, self.num_users)])
                interference += self.noise_power  # 不加Jammer干扰
                gamma_1 = (h_gain * P_users[0]) / interference if interference > 0 else 0
                gamma_users.append(gamma_1)
                R_users.append(np.log2(1 + gamma_1))
                SIC_success.append(True)
            else:
                # 强用户Ui：先SIC解码弱用户U1,...,U_{i-1}
                # 检查SIC可行性：解码U1的SINR
                # 注意：Jammer只干扰窃听者，不干扰合法用户
                interference_for_U1 = sum([h_gain * P_users[j] for j in range(1, self.num_users)])
                interference_for_U1 += self.noise_power  # 不加Jammer干扰
                gamma_sic = (h_gain * P_users[0]) / interference_for_U1 if interference_for_U1 > 0 else 0

                # SIC成功后，解码自己信号
                interference_self = sum([h_gain * P_users[j] for j in range(i+1, self.num_users)])
                interference_self += self.noise_power  # 不加Jammer干扰
                gamma_i = (h_gain * P_users[i]) / interference_self if interference_self > 0 else 0

                gamma_users.append(gamma_i)
                R_users.append(np.log2(1 + gamma_i))

                R_sic = np.log2(1 + gamma_sic)
                SIC_success.append(R_sic >= self.R_min)

        # 窃听者SINR（窃听者不能SIC，只能把其他信号当作干扰）
        g_gain = np.abs(self.g)**2
        g_J_gain = np.abs(self.g_J)**2

        gamma_eaves = []
        R_eaves = []

        for i in range(self.num_users):
            # 窃听者不能SIC：所有其他用户信号都是干扰
            interference_e = sum([g_gain * P_users[j] for j in range(self.num_users) if j != i])
            # 加上Jammer干扰
            interference_e += g_J_gain * P_J + self.noise_power
            gamma_e = (g_gain * P_users[i]) / interference_e if interference_e > 0 else 0
            gamma_eaves.append(gamma_e)
            R_eaves.append(np.log2(1 + gamma_e))

        # 计算保密容量
        secrecy_rates = []
        for i in range(self.num_users):
            rs = max(0, R_users[i] - R_eaves[i])
            secrecy_rates.append(rs)

        R_sum_secrecy = sum(secrecy_rates)
        R_sum_legit = sum(R_users)
        R_sum_eaves = sum(R_eaves)

        # 检查约束
        qos_satisfied = all([R >= self.R_min for R in R_users])
        sic_feasible = all(SIC_success[1:]) if self.num_users > 1 else True
        power_valid = np.sum(P_users) <= self.P_max + 1e-6

        # 计算奖励
        lambda1 = 5.0
        lambda2 = 5.0
        lambda3 = 10.0

        qos_penalty = sum([max(0, self.R_min - R) for R in R_users])
        sic_penalty = 0.0 if sic_feasible else 1.0
        power_penalty = max(0, np.sum(P_users) - self.P_max)

        reward = R_sum_secrecy - lambda1 * qos_penalty - lambda2 * sic_penalty - lambda3 * power_penalty

        # 构建info
        info = {
            'secrecy_sum': R_sum_secrecy,
            'legit_sum': R_sum_legit,
            'eaves_sum': R_sum_eaves,
            'secrecy_rates': secrecy_rates,
            'legit_rates': R_users,
            'eaves_rates': R_eaves,
            'qos_satisfied': qos_satisfied,
            'sic_feasible': sic_feasible,
            'power_valid': power_valid,
            'jammer_power': P_J,
            'total_power': np.sum(P_users) + P_J,
            'powers': np.concatenate([P_users, [P_J]]) if self.use_jammer else P_users
        }

        # 生成下一个状态
        next_state = self.reset()
        done = False

        return next_state, reward, done, info

    def evaluate_action(self, action):
        """评估动作，返回各项指标（不更新状态）"""
        if self.current_channels is None:
            self.reset()
        state = self.current_channels['state'].copy()
        _, reward, _, info = self.step(action)
        # 恢复状态（因为step会reset）
        self.current_channels['state'] = state
        return reward, info


# ==================== 2. DRL 智能体 ====================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 输出(0,1)
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    """
    DDPG智能体，支持连续动作空间
    """
    def __init__(self, state_dim, action_dim, 
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005,
                 hidden_dim=256,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 128
        
        # 探索噪声
        self.noise_std = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.01
        
    def select_action(self, state, evaluate=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if not evaluate:
            # 添加探索噪声
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, 0, 1)
            # 用户功率和Jammer功率分别归一化
            if self.action_dim > 1:
                # 用户部分归一化
                user_sum = np.sum(action[:-1])
                if user_sum > 1.0:
                    action[:-1] = action[:-1] / user_sum
                # Jammer部分保持独立（已经在[0,1]范围内）

        return action
    
    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions = torch.clamp(next_actions, 0, 1)
            # 用户功率归一化，Jammer功率独立（避免in-place操作）
            if next_actions.shape[1] > 1:
                user_sum = next_actions[:, :-1].sum(dim=1, keepdim=True)
                user_actions = torch.where(user_sum > 1, next_actions[:, :-1] / user_sum, next_actions[:, :-1])
                next_actions = torch.cat([user_actions, next_actions[:, -1:]], dim=1)

            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        pred_actions = self.actor(states)
        pred_actions = torch.clamp(pred_actions, 0, 1)
        # 用户功率归一化，Jammer功率独立（避免in-place操作）
        if pred_actions.shape[1] > 1:
            user_sum = pred_actions[:, :-1].sum(dim=1, keepdim=True)
            user_actions = torch.where(user_sum > 1, pred_actions[:, :-1] / user_sum, pred_actions[:, :-1])
            pred_actions = torch.cat([user_actions, pred_actions[:, -1:]], dim=1)

        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        # 衰减噪声
        self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


class TD3Agent(DDPGAgent):
    """
    TD3: DDPG改进版，双Critic + 延迟更新 + 目标噪声
    """
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        
        # 第二个Critic
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=kwargs.get('critic_lr', 1e-3))
        
        self.policy_delay = 2
        self.update_count = 0
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 目标动作平滑
        with torch.no_grad():
            noise = (torch.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
            next_actions = self.actor_target(next_states)
            next_actions = (next_actions + noise).clamp(0, 1)
            # 用户功率归一化，Jammer功率独立（避免in-place操作）
            if next_actions.shape[1] > 1:
                user_sum = next_actions[:, :-1].sum(dim=1, keepdim=True)
                user_actions = torch.where(user_sum > 1, next_actions[:, :-1] / user_sum, next_actions[:, :-1])
                next_actions = torch.cat([user_actions, next_actions[:, -1:]], dim=1)

            target_q1 = self.critic_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 更新两个Critic
        current_q1 = self.critic(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss = 0
        if self.update_count % self.policy_delay == 0:
            pred_actions = self.actor(states)
            pred_actions = torch.clamp(pred_actions, 0, 1)
            # 用户功率归一化，Jammer功率独立（避免in-place操作）
            if pred_actions.shape[1] > 1:
                user_sum = pred_actions[:, :-1].sum(dim=1, keepdim=True)
                user_actions = torch.where(user_sum > 1, pred_actions[:, :-1] / user_sum, pred_actions[:, :-1])
                pred_actions = torch.cat([user_actions, pred_actions[:, -1:]], dim=1)

            actor_loss = -self.critic(states, pred_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.critic2, self.critic2_target)
        
        self.update_count += 1
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss
        }


class SACAgent:
    """
    SAC: Soft Actor-Critic，最大熵强化学习
    """
    def __init__(self, state_dim, action_dim,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 hidden_dim=256,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor (高斯策略)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean和log_std
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic (双Q)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # 自动调整温度参数alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.batch_size = 128
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean_logstd = self.actor(state)
            mean, log_std = mean_logstd.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            
            if evaluate:
                action = torch.sigmoid(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.sigmoid(x_t)
        
        action = action.cpu().numpy()[0]
        # 用户功率归一化，Jammer功率独立
        if len(action) > 1:
            user_sum = np.sum(action[:-1])
            if user_sum > 1.0:
                action[:-1] = action[:-1] / user_sum
        return action
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        alpha = self.log_alpha.exp()

        # 更新Critic
        with torch.no_grad():
            mean_logstd = self.actor(next_states)
            mean, log_std = mean_logstd.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            normal = torch.distributions.Normal(mean, std)
            x_t_next = normal.sample()
            next_actions = torch.sigmoid(x_t_next)
            # 用户功率归一化，Jammer功率独立（避免in-place操作）
            if next_actions.shape[1] > 1:
                user_sum = next_actions[:, :-1].sum(dim=1, keepdim=True)
                user_actions = torch.where(user_sum > 1, next_actions[:, :-1] / user_sum, next_actions[:, :-1])
                next_actions = torch.cat([user_actions, next_actions[:, -1:]], dim=1)

            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # 计算熵项：注意sigmoid变换后的Jacobian修正
            # log_prob(action) = log_prob(x_t) - sum(log(action*(1-action)))
            log_prob = normal.log_prob(x_t_next).sum(dim=1, keepdim=True)
            # Jacobian修正: d/dx sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
            # log|det(J)| = sum(log(action*(1-action)))
            log_prob = log_prob - torch.log(next_actions * (1 - next_actions) + 1e-6).sum(dim=1, keepdim=True)
            next_q = next_q - alpha * log_prob

            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新Actor
        mean_logstd = self.actor(states)
        mean, log_std = mean_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        actions_pred = torch.sigmoid(x_t)
        # 用户功率归一化，Jammer功率独立（避免in-place操作）
        if actions_pred.shape[1] > 1:
            user_sum = actions_pred[:, :-1].sum(dim=1, keepdim=True)
            user_actions = torch.where(user_sum > 1, actions_pred[:, :-1] / user_sum, actions_pred[:, :-1])
            actions_pred = torch.cat([user_actions, actions_pred[:, -1:]], dim=1)

        # 计算正确的log_prob（带Jacobian修正）
        log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)
        log_prob = log_prob - torch.log(actions_pred * (1 - actions_pred) + 1e-6).sum(dim=1, keepdim=True)

        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        q = torch.min(q1, q2)

        actor_loss = (alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item()
        }
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ==================== 3. 传统对比算法 ====================

class BaselineAlgorithms:
    """
    传统功率分配算法
    """
    
    @staticmethod
    def _normalize_action(env, action):
        """归一化动作：用户功率和Jammer功率分别处理"""
        action = np.clip(action, 0, 1)
        if env.use_jammer and len(action) > 1:
            user_sum = np.sum(action[:-1])
            if user_sum > 1:
                action[:-1] = action[:-1] / user_sum
        else:
            action_sum = np.sum(action)
            if action_sum > 1:
                action = action / action_sum
        return action

    @staticmethod
    def random_allocation(env, num_samples=100):
        """随机功率分配"""
        best_reward = -np.inf
        best_info = None
        best_action = None

        for _ in range(num_samples):
            action = np.random.rand(env.action_dim)
            action = BaselineAlgorithms._normalize_action(env, action)
            reward, info = env.evaluate_action(action)
            if reward > best_reward:
                best_reward = reward
                best_info = info
                best_action = action

        return best_action, best_reward, best_info

    @staticmethod
    def equal_allocation(env):
        """等功率分配"""
        if env.use_jammer:
            # 用户等功率，Jammer独立
            user_part = np.ones(env.num_users) / env.num_users
            action = np.concatenate([user_part, [1.0]])  # Jammer也满功率
        else:
            action = np.ones(env.action_dim) / env.action_dim
        reward, info = env.evaluate_action(action)
        return action, reward, info

    @staticmethod
    def heuristic_allocation(env, ratio=None):
        """启发式固定比例分配"""
        if ratio is None:
            # 默认比例：弱用户多分配，Jammer中等
            if env.num_users == 2 and env.use_jammer:
                ratio = [0.5, 0.3, 0.5]  # [U1, U2, Jammer]
            elif env.num_users == 2 and not env.use_jammer:
                ratio = [0.6, 0.4]
            else:
                # 多用户默认
                ratio = [0.5] + [0.3/(env.num_users-1)]*(env.num_users-1) + [0.5]
                if not env.use_jammer:
                    ratio = ratio[:-1]
                    s = sum(ratio)
                    ratio = [r/s for r in ratio]

        action = np.array(ratio[:env.action_dim])
        action = BaselineAlgorithms._normalize_action(env, action)
        reward, info = env.evaluate_action(action)
        return action, reward, info

    @staticmethod
    def pso_allocation(env, particles=30, iterations=50):
        """
        粒子群优化 (PSO)
        """
        def objective(x):
            # x in [0,1]^action_dim
            x = BaselineAlgorithms._normalize_action(env, x)
            reward, _ = env.evaluate_action(x)
            return -reward  # 最小化负奖励
        
        # 初始化粒子
        dim = env.action_dim
        pos = np.random.rand(particles, dim)
        vel = np.random.randn(particles, dim) * 0.1
        pbest_pos = pos.copy()
        pbest_val = np.array([objective(p) for p in pos])
        
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]
        
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for _ in range(iterations):
            for i in range(particles):
                r1, r2 = np.random.rand(2)
                vel[i] = (w * vel[i] + 
                         c1 * r1 * (pbest_pos[i] - pos[i]) + 
                         c2 * r2 * (gbest_pos - pos[i]))
                pos[i] = pos[i] + vel[i]
                pos[i] = np.clip(pos[i], 0, 1)
                
                val = objective(pos[i])
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = pos[i].copy()
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = pos[i].copy()
        
        best_action = BaselineAlgorithms._normalize_action(env, gbest_pos)
        reward, info = env.evaluate_action(best_action)
        return best_action, reward, info

    @staticmethod
    def grid_search_allocation(env, grid_points=10):
        """
        网格搜索（仅适用于低维）
        """
        dim = env.action_dim

        if dim > 3:
            # 高维时使用粗网格或随机采样
            return BaselineAlgorithms.random_allocation(env, num_samples=500)

        # 生成网格
        grids = [np.linspace(0, 1, grid_points) for _ in range(dim)]

        best_reward = -np.inf
        best_info = None
        best_action = None

        if dim == 2:
            for a1 in grids[0]:
                for a2 in grids[1]:
                    action = np.array([a1, a2])
                    action = BaselineAlgorithms._normalize_action(env, action)
                    reward, info = env.evaluate_action(action)
                    if reward > best_reward:
                        best_reward = reward
                        best_info = info
                        best_action = action
        elif dim == 3:
            for a1 in grids[0]:
                for a2 in grids[1]:
                    for a3 in grids[2]:
                        action = np.array([a1, a2, a3])
                        action = BaselineAlgorithms._normalize_action(env, action)
                        reward, info = env.evaluate_action(action)
                        if reward > best_reward:
                            best_reward = reward
                            best_info = info
                            best_action = action

        if best_action is None:
            return BaselineAlgorithms.random_allocation(env)

        return best_action, best_reward, best_info


# ==================== 4. 训练与评估框架 ====================

class Trainer:
    """
    训练框架
    """
    def __init__(self, env, agent, agent_name='DDPG'):
        self.env = env
        self.agent = agent
        self.agent_name = agent_name
        self.episode_rewards = []
        self.eval_rewards = []
    
    def train(self, episodes=3000, steps_per_episode=100, eval_interval=100):
        """训练智能体"""
        print(f"开始训练 {self.agent_name}...")

        for ep in range(episodes):
            state = self.env.reset()
            ep_reward = 0

            for step in range(steps_per_episode):
                action = self.agent.select_action(state, evaluate=False)
                next_state, reward, done, info = self.env.step(action)

                self.agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    self.agent.update()

                state = next_state
                ep_reward += reward

            self.episode_rewards.append(ep_reward)

            if (ep + 1) % eval_interval == 0:
                eval_results = self.evaluate(episodes=10)
                avg_reward = eval_results['avg_reward']
                self.eval_rewards.append(avg_reward)
                print(f"Episode {ep+1}/{episodes}, Eval Reward: {avg_reward:.4f}, "
                      f"Secrecy: {eval_results['avg_secrecy_sum']:.4f}")

        print(f"{self.agent_name} 训练完成!")
        return self.episode_rewards, self.eval_rewards
    
    def evaluate(self, episodes=200, verbose=False):
        """评估智能体"""
        rewards = []
        secrecy_sums = []
        legit_sums = []
        eaves_sums = []
        qos_satisfactions = []
        secrecy_outages = []
        decision_times = []

        for ep in range(episodes):
            state = self.env.reset()

            start_time = time.time()
            action = self.agent.select_action(state, evaluate=True)
            decision_time = (time.time() - start_time) * 1000  # ms

            _, reward, _, info = self.env.step(action)

            rewards.append(reward)
            secrecy_sums.append(info['secrecy_sum'])
            legit_sums.append(info['legit_sum'])
            eaves_sums.append(info['eaves_sum'])
            qos_satisfactions.append(1.0 if info['qos_satisfied'] else 0.0)
            secrecy_outages.append(1.0 if info['secrecy_sum'] < 0.1 else 0.0)
            decision_times.append(decision_time)

        results = {
            'avg_reward': np.mean(rewards),
            'avg_secrecy_sum': np.mean(secrecy_sums),
            'avg_legit_sum': np.mean(legit_sums),
            'avg_eaves_sum': np.mean(eaves_sums),
            'qos_satisfaction_rate': np.mean(qos_satisfactions),
            'secrecy_outage_prob': np.mean(secrecy_outages),
            'avg_decision_time_ms': np.mean(decision_times),
            'std_secrecy_sum': np.std(secrecy_sums)
        }
        
        if verbose:
            print(f"评估结果:")
            print(f"  平均奖励: {results['avg_reward']:.4f}")
            print(f"  平均保密容量: {results['avg_secrecy_sum']:.4f}")
            print(f"  QoS满足率: {results['qos_satisfaction_rate']:.2%}")
            print(f"  保密中断概率: {results['secrecy_outage_prob']:.2%}")
            print(f"  平均决策时间: {results['avg_decision_time_ms']:.4f} ms")
        
        return results


class Evaluator:
    """
    评估器：评估所有算法
    """
    def __init__(self, env):
        self.env = env
        self.baseline = BaselineAlgorithms()
    
    def evaluate_all(self, agent_dict=None, episodes=200, verbose=True):
        """
        评估所有算法
        agent_dict: {'name': agent_instance}
        """
        results = {}
        
        # 评估DRL智能体
        if agent_dict:
            for name, agent in agent_dict.items():
                print(f"\n评估 {name}...")
                rewards = []
                secrecy_sums = []
                legit_sums = []
                eaves_sums = []
                qos_sats = []
                sec_outages = []
                decision_times = []
                jammer_powers = []
                sic_feasibles = []
                
                for ep in range(episodes):
                    state = self.env.reset()
                    start = time.time()
                    action = agent.select_action(state, evaluate=True)
                    dt = (time.time() - start) * 1000
                    
                    _, reward, _, info = self.env.step(action)
                    
                    rewards.append(reward)
                    secrecy_sums.append(info['secrecy_sum'])
                    legit_sums.append(info['legit_sum'])
                    eaves_sums.append(info['eaves_sum'])
                    qos_sats.append(1.0 if info['qos_satisfied'] else 0.0)
                    sec_outages.append(1.0 if info['secrecy_sum'] < 0.1 else 0.0)
                    decision_times.append(dt)
                    jammer_powers.append(info['jammer_power'])
                    sic_feasibles.append(1.0 if info['sic_feasible'] else 0.0)
                
                results[name] = {
                    'avg_reward': np.mean(rewards),
                    'avg_secrecy_sum': np.mean(secrecy_sums),
                    'avg_legit_sum': np.mean(legit_sums),
                    'avg_eaves_sum': np.mean(eaves_sums),
                    'qos_satisfaction_rate': np.mean(qos_sats),
                    'secrecy_outage_prob': np.mean(sec_outages),
                    'avg_decision_time_ms': np.mean(decision_times),
                    'avg_jammer_power': np.mean(jammer_powers),
                    'sic_feasible_rate': np.mean(sic_feasibles),
                    'std_secrecy_sum': np.std(secrecy_sums)
                }
        
        # 评估传统算法
        traditional_algorithms = {
            'Random': lambda: self.baseline.random_allocation(self.env),
            'Equal': lambda: self.baseline.equal_allocation(self.env),
            'Heuristic': lambda: self.baseline.heuristic_allocation(self.env),
            'PSO': lambda: self.baseline.pso_allocation(self.env),
        }
        
        # Grid Search只在低维时运行
        if self.env.action_dim <= 3:
            traditional_algorithms['GridSearch'] = lambda: self.baseline.grid_search_allocation(self.env)
        
        for name, algo in traditional_algorithms.items():
            print(f"\n评估 {name}...")
            
            # 传统算法每次重新优化，需要多次运行取平均
            rewards = []
            secrecy_sums = []
            legit_sums = []
            eaves_sums = []
            qos_sats = []
            sec_outages = []
            decision_times = []
            jammer_powers = []
            sic_feasibles = []
            
            for ep in range(episodes):
                state = self.env.reset()
                
                start = time.time()
                action, reward, info = algo()
                dt = (time.time() - start) * 1000
                
                rewards.append(reward)
                secrecy_sums.append(info['secrecy_sum'])
                legit_sums.append(info['legit_sum'])
                eaves_sums.append(info['eaves_sum'])
                qos_sats.append(1.0 if info['qos_satisfied'] else 0.0)
                sec_outages.append(1.0 if info['secrecy_sum'] < 0.1 else 0.0)
                decision_times.append(dt)
                jammer_powers.append(info['jammer_power'])
                sic_feasibles.append(1.0 if info['sic_feasible'] else 0.0)
            
            results[name] = {
                'avg_reward': np.mean(rewards),
                'avg_secrecy_sum': np.mean(secrecy_sums),
                'avg_legit_sum': np.mean(legit_sums),
                'avg_eaves_sum': np.mean(eaves_sums),
                'qos_satisfaction_rate': np.mean(qos_sats),
                'secrecy_outage_prob': np.mean(sec_outages),
                'avg_decision_time_ms': np.mean(decision_times),
                'avg_jammer_power': np.mean(jammer_powers),
                'sic_feasible_rate': np.mean(sic_feasibles),
                'std_secrecy_sum': np.std(secrecy_sums)
            }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        print("\n" + "="*80)
        print("评估结果汇总")
        print("="*80)
        print(f"{'算法':<<15} {'保密容量':<<12} {'QoS满足率':<<12} {'中断概率':<<12} {'决策时间(ms)':<<15} {'Jammer功率':<<12}")
        print("-"*80)
        for name, res in results.items():
            print(f"{name:<15} {res['avg_secrecy_sum']:<12.4f} {res['qos_satisfaction_rate']:<12.2%} "
                  f"{res['secrecy_outage_prob']:<12.2%} {res['avg_decision_time_ms']:<15.4f} "
                  f"{res['avg_jammer_power']:<12.4f}")


# ==================== 5. 补充实验一：窃听者强度变化 ====================

class Experiment1_EavesdropperStrength:
    """
    实验一：窃听者强度变化实验
    测试不同eavesdropper_gain_scale下的算法性能
    """
    
    def __init__(self, base_config=None):
        self.base_config = base_config or {}
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        self.results = {scale: {} for scale in self.scales}
    
    def run(self, algorithms=['Random', 'Equal', 'Heuristic', 'PSO', 'DDPG', 'DDPG_wo_Jammer'],
            num_users=2, 
            episodes_per_eval=100,
            train_episodes=1000,
            seeds=[42, 123, 456]):
        """
        运行实验
        """
        print("="*80)
        print("实验一：窃听者强度变化实验")
        print("="*80)
        
        for scale in self.scales:
            print(f"\n{'='*40}")
            print(f"窃听者强度系数: {scale}")
            print(f"{'='*40}")
            
            for seed in seeds:
                print(f"\n随机种子: {seed}")
                set_seed(seed)
                
                # 创建环境
                env = NOMASecureEnv(
                    num_users=num_users,
                    eavesdropper_gain_scale=scale,
                    use_jammer=True,
                    seed=seed,
                    **self.base_config
                )
                
                # 训练DDPG（with Jammer）
                if 'DDPG' in algorithms:
                    print("训练 DDPG with Jammer...")
                    agent_ddpg = DDPGAgent(env.state_dim, env.action_dim)
                    trainer = Trainer(env, agent_ddpg, 'DDPG')
                    trainer.train(episodes=train_episodes, steps_per_episode=100, eval_interval=200)
                    
                    # 评估
                    evaluator = Evaluator(env)
                    res = evaluator.evaluate_all({'DDPG': agent_ddpg}, episodes=episodes_per_eval, verbose=False)
                    
                    if 'DDPG' not in self.results[scale]:
                        self.results[scale]['DDPG'] = []
                    self.results[scale]['DDPG'].append(res['DDPG'])
                
                # 训练DDPG without Jammer
                if 'DDPG_wo_Jammer' in algorithms:
                    env_wo = NOMASecureEnv(
                        num_users=num_users,
                        eavesdropper_gain_scale=scale,
                        use_jammer=False,
                        seed=seed,
                        **self.base_config
                    )
                    print("训练 DDPG without Jammer...")
                    agent_ddpg_wo = DDPGAgent(env_wo.state_dim, env_wo.action_dim)
                    trainer_wo = Trainer(env_wo, agent_ddpg_wo, 'DDPG_wo_Jammer')
                    trainer_wo.train(episodes=train_episodes, steps_per_episode=100, eval_interval=200)
                    
                    evaluator_wo = Evaluator(env_wo)
                    res_wo = evaluator_wo.evaluate_all({'DDPG_wo_Jammer': agent_ddpg_wo}, 
                                                     episodes=episodes_per_eval, verbose=False)
                    
                    if 'DDPG_wo_Jammer' not in self.results[scale]:
                        self.results[scale]['DDPG_wo_Jammer'] = []
                    self.results[scale]['DDPG_wo_Jammer'].append(res_wo['DDPG_wo_Jammer'])
                
                # 传统算法（不需要训练）
                env_eval = NOMASecureEnv(
                    num_users=num_users,
                    eavesdropper_gain_scale=scale,
                    use_jammer=True,
                    seed=seed,
                    **self.base_config
                )
                evaluator = Evaluator(env_eval)
                baseline_res = evaluator.evaluate_all(episodes=episodes_per_eval, verbose=False)
                
                for algo in ['Random', 'Equal', 'Heuristic', 'PSO', 'GridSearch']:
                    if algo in baseline_res:
                        if algo not in self.results[scale]:
                            self.results[scale][algo] = []
                        self.results[scale][algo].append(baseline_res[algo])
        
        # 汇总结果（取平均）
        self.summarized_results = self._summarize_results()
        return self.summarized_results
    
    def _summarize_results(self):
        """汇总多随机种子结果"""
        summary = {}
        for scale in self.scales:
            summary[scale] = {}
            for algo, runs in self.results[scale].items():
                if not runs:
                    continue
                summary[scale][algo] = {}
                for metric in runs[0].keys():
                    values = [run[metric] for run in runs]
                    summary[scale][algo][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        return summary
    
    def plot_results(self, save_path='experiment1_results.png'):
        """绘制实验结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        scales = self.scales
        algorithms = list(self.summarized_results[scales[0]].keys())
        
        # 颜色映射
        colors = {
            'DDPG': 'red',
            'DDPG_wo_Jammer': 'orange',
            'TD3': 'darkred',
            'SAC': 'purple',
            'Random': 'gray',
            'Equal': 'blue',
            'Heuristic': 'green',
            'PSO': 'cyan',
            'GridSearch': 'black'
        }
        
        # 图1: 窃听者强度 vs 平均保密容量
        ax1 = axes[0, 0]
        for algo in algorithms:
            means = [self.summarized_results[s][algo]['avg_secrecy_sum']['mean'] 
                    for s in scales if algo in self.summarized_results[s]]
            stds = [self.summarized_results[s][algo]['avg_secrecy_sum']['std'] 
                   for s in scales if algo in self.summarized_results[s]]
            valid_scales = [s for s in scales if algo in self.summarized_results[s]]
            
            if means:
                ax1.plot(valid_scales, means, marker='o', label=algo, color=colors.get(algo, None))
                ax1.fill_between(valid_scales, 
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.2)
        
        ax1.set_xlabel('Eavesdropper Gain Scale')
        ax1.set_ylabel('Average Secrecy Sum Rate (bit/s/Hz)')
        ax1.set_title('Secrecy Capacity vs Eavesdropper Strength')
        ax1.legend()
        ax1.grid(True)
        
        # 图2: 窃听者强度 vs 保密中断概率
        ax2 = axes[0, 1]
        for algo in algorithms:
            means = [self.summarized_results[s][algo]['secrecy_outage_prob']['mean'] 
                    for s in scales if algo in self.summarized_results[s]]
            valid_scales = [s for s in scales if algo in self.summarized_results[s]]
            
            if means:
                ax2.plot(valid_scales, means, marker='s', label=algo, color=colors.get(algo, None))
        
        ax2.set_xlabel('Eavesdropper Gain Scale')
        ax2.set_ylabel('Secrecy Outage Probability')
        ax2.set_title('Outage Probability vs Eavesdropper Strength')
        ax2.legend()
        ax2.grid(True)
        
        # 图3: 窃听者强度 vs QoS满足率
        ax3 = axes[1, 0]
        for algo in algorithms:
            means = [self.summarized_results[s][algo]['qos_satisfaction_rate']['mean'] 
                    for s in scales if algo in self.summarized_results[s]]
            valid_scales = [s for s in scales if algo in self.summarized_results[s]]
            
            if means:
                ax3.plot(valid_scales, means, marker='^', label=algo, color=colors.get(algo, None))
        
        ax3.set_xlabel('Eavesdropper Gain Scale')
        ax3.set_ylabel('QoS Satisfaction Rate')
        ax3.set_title('QoS Satisfaction vs Eavesdropper Strength')
        ax3.legend()
        ax3.grid(True)
        
        # 图4: 窃听者强度 vs 平均Jammer功率
        ax4 = axes[1, 1]
        for algo in algorithms:
            if algo in ['DDPG', 'DDPG_wo_Jammer', 'TD3', 'SAC', 'Heuristic']:
                means = [self.summarized_results[s][algo]['avg_jammer_power']['mean'] 
                        for s in scales if algo in self.summarized_results[s]]
                valid_scales = [s for s in scales if algo in self.summarized_results[s]]
                
                if means:
                    ax4.plot(valid_scales, means, marker='d', label=algo, color=colors.get(algo, None))
        
        ax4.set_xlabel('Eavesdropper Gain Scale')
        ax4.set_ylabel('Average Jammer Power')
        ax4.set_title('Jammer Power vs Eavesdropper Strength')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"实验一结果图已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def save_results(self, filename='experiment1_results.json'):
        """保存结果到JSON"""
        # 转换numpy类型为Python原生类型
        save_dict = {}
        for scale, algos in self.summarized_results.items():
            save_dict[str(scale)] = {}
            for algo, metrics in algos.items():
                save_dict[str(scale)][algo] = {}
                for metric, values in metrics.items():
                    save_dict[str(scale)][algo][metric] = {
                        'mean': float(values['mean']),
                        'std': float(values['std'])
                    }
        
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"实验一结果已保存至: {filename}")


# ==================== 6. 补充实验二：用户数量扩展 ====================

class Experiment2_UserScalability:
    """
    实验二：用户数量扩展实验
    测试不同用户数量下的算法性能和决策时间
    """
    
    def __init__(self, base_config=None):
        self.base_config = base_config or {}
        self.user_counts = [2, 3, 4, 5]
        self.results = {m: {} for m in self.user_counts}
    
    def run(self, algorithms=['Random', 'Equal', 'Heuristic', 'PSO', 'DDPG'],
            episodes_per_eval=100,
            train_episodes=1000,
            seeds=[42, 123, 456]):
        """
        运行实验
        """
        print("="*80)
        print("实验二：用户数量扩展实验")
        print("="*80)
        
        for num_users in self.user_counts:
            print(f"\n{'='*40}")
            print(f"用户数量: {num_users}")
            print(f"{'='*40}")
            
            for seed in seeds:
                print(f"\n随机种子: {seed}")
                set_seed(seed)
                
                # 创建环境
                env = NOMASecureEnv(
                    num_users=num_users,
                    use_jammer=True,
                    seed=seed,
                    **self.base_config
                )
                
                print(f"状态维度: {env.state_dim}, 动作维度: {env.action_dim}")
                
                # 训练DDPG
                if 'DDPG' in algorithms:
                    print(f"训练 DDPG for {num_users} users...")
                    agent_ddpg = DDPGAgent(env.state_dim, env.action_dim)
                    trainer = Trainer(env, agent_ddpg, f'DDPG_M{num_users}')
                    trainer.train(episodes=train_episodes, steps_per_episode=100, eval_interval=200)
                    
                    # 评估
                    evaluator = Evaluator(env)
                    res = evaluator.evaluate_all({'DDPG': agent_ddpg}, episodes=episodes_per_eval, verbose=False)
                    
                    if 'DDPG' not in self.results[num_users]:
                        self.results[num_users]['DDPG'] = []
                    self.results[num_users]['DDPG'].append(res['DDPG'])
                
                # 传统算法
                evaluator = Evaluator(env)
                baseline_res = evaluator.evaluate_all(episodes=episodes_per_eval, verbose=False)
                
                for algo in ['Random', 'Equal', 'Heuristic', 'PSO', 'GridSearch']:
                    if algo in baseline_res:
                        if algo not in self.results[num_users]:
                            self.results[num_users][algo] = []
                        self.results[num_users][algo].append(baseline_res[algo])
        
        # 汇总
        self.summarized_results = self._summarize_results()
        return self.summarized_results
    
    def _summarize_results(self):
        """汇总结果"""
        summary = {}
        for num_users in self.user_counts:
            summary[num_users] = {}
            for algo, runs in self.results[num_users].items():
                if not runs:
                    continue
                summary[num_users][algo] = {}
                for metric in runs[0].keys():
                    values = [run[metric] for run in runs]
                    summary[num_users][algo][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        return summary
    
    def plot_results(self, save_path='experiment2_results.png'):
        """绘制实验结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        user_counts = self.user_counts
        algorithms = []
        for m in user_counts:
            if m in self.summarized_results and self.summarized_results[m]:
                algorithms.extend(list(self.summarized_results[m].keys()))
        algorithms = list(set(algorithms))
        
        colors = {
            'DDPG': 'red',
            'TD3': 'darkred',
            'SAC': 'purple',
            'Random': 'gray',
            'Equal': 'blue',
            'Heuristic': 'green',
            'PSO': 'cyan',
            'GridSearch': 'black'
        }
        
        # 图1: 用户数量 vs 平均决策时间（最重要）
        ax1 = axes[0, 0]
        for algo in algorithms:
            means = [self.summarized_results[m][algo]['avg_decision_time_ms']['mean'] 
                    for m in user_counts if algo in self.summarized_results[m]]
            stds = [self.summarized_results[m][algo]['avg_decision_time_ms']['std'] 
                   for m in user_counts if algo in self.summarized_results[m]]
            valid_m = [m for m in user_counts if algo in self.summarized_results[m]]
            
            if means:
                ax1.plot(valid_m, means, marker='o', label=algo, color=colors.get(algo, None))
                ax1.fill_between(valid_m,
                               np.array(means) - np.array(stds),
                               np.array(means) + np.array(stds),
                               alpha=0.2)
        
        ax1.set_xlabel('Number of Users (M)')
        ax1.set_ylabel('Average Decision Time (ms)')
        ax1.set_title('Decision Time vs Number of Users')
        ax1.set_yscale('log')  # 对数尺度更好显示差异
        ax1.legend()
        ax1.grid(True)
        
        # 图2: 用户数量 vs 总保密容量
        ax2 = axes[0, 1]
        for algo in algorithms:
            means = [self.summarized_results[m][algo]['avg_secrecy_sum']['mean'] 
                    for m in user_counts if algo in self.summarized_results[m]]
            valid_m = [m for m in user_counts if algo in self.summarized_results[m]]
            
            if means:
                ax2.plot(valid_m, means, marker='s', label=algo, color=colors.get(algo, None))
        
        ax2.set_xlabel('Number of Users (M)')
        ax2.set_ylabel('Total Secrecy Sum Rate (bit/s/Hz)')
        ax2.set_title('Total Secrecy Capacity vs Number of Users')
        ax2.legend()
        ax2.grid(True)
        
        # 图3: 用户数量 vs 单用户平均保密容量
        ax3 = axes[1, 0]
        for algo in algorithms:
            means = []
            for m in user_counts:
                if algo in self.summarized_results[m]:
                    avg_sum = self.summarized_results[m][algo]['avg_secrecy_sum']['mean']
                    means.append(avg_sum / m)
            valid_m = [m for m in user_counts if algo in self.summarized_results[m]]
            
            if means:
                ax3.plot(valid_m, means, marker='^', label=algo, color=colors.get(algo, None))
        
        ax3.set_xlabel('Number of Users (M)')
        ax3.set_ylabel('Average Secrecy Rate per User (bit/s/Hz)')
        ax3.set_title('Per-User Secrecy Capacity vs Number of Users')
        ax3.legend()
        ax3.grid(True)
        
        # 图4: 用户数量 vs QoS满足率
        ax4 = axes[1, 1]
        for algo in algorithms:
            means = [self.summarized_results[m][algo]['qos_satisfaction_rate']['mean'] 
                    for m in user_counts if algo in self.summarized_results[m]]
            valid_m = [m for m in user_counts if algo in self.summarized_results[m]]
            
            if means:
                ax4.plot(valid_m, means, marker='d', label=algo, color=colors.get(algo, None))
        
        ax4.set_xlabel('Number of Users (M)')
        ax4.set_ylabel('QoS Satisfaction Rate')
        ax4.set_title('QoS Satisfaction vs Number of Users')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"实验二结果图已保存至: {save_path}")
        plt.show()
        
        return fig
    
    def save_results(self, filename='experiment2_results.json'):
        """保存结果"""
        save_dict = {}
        for num_users, algos in self.summarized_results.items():
            save_dict[str(num_users)] = {}
            for algo, metrics in algos.items():
                save_dict[str(num_users)][algo] = {}
                for metric, values in metrics.items():
                    save_dict[str(num_users)][algo][metric] = {
                        'mean': float(values['mean']),
                        'std': float(values['std'])
                    }
        
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=2)
        print(f"实验二结果已保存至: {filename}")


# ==================== 7. 主程序入口 ====================

def run_experiment1_quick():
    """
    快速运行实验一（演示版本，减少训练轮数）
    """
    print("运行实验一：窃听者强度变化实验（快速演示版）")
    
    config = {
        'P_max': 10.0,
        'noise_power': 1.0,
        'R_min': 1.0,
        'L1': 1.0,
        'L2': 2.0,
        'L_J1': 0.5,
        'L_J2': 0.5,
        'L_e': 1.5,
        'L_Je': 3.0
    }
    
    exp1 = Experiment1_EavesdropperStrength(base_config=config)
    
    # 使用较少轮数进行快速演示
    results = exp1.run(
        algorithms=['Random', 'Equal', 'Heuristic', 'PSO', 'DDPG', 'DDPG_wo_Jammer'],
        num_users=2,
        episodes_per_eval=50,  # 演示用，实际建议100-200
        train_episodes=500,     # 演示用，实际建议2000-3000
        seeds=[42]              # 演示用，实际建议3-5个种子
    )
    
    exp1.plot_results('experiment1_quick.png')
    exp1.save_results('experiment1_quick.json')
    
    return exp1


def run_experiment2_quick():
    """
    快速运行实验二（演示版本）
    """
    print("运行实验二：用户数量扩展实验（快速演示版）")
    
    config = {
        'P_max': 10.0,
        'noise_power': 1.0,
        'R_min': 1.0,
        'L1': 1.0,
        'L2': 2.0,
        'L_J1': 0.5,
        'L_J2': 0.5,
        'L_e': 1.5,
        'L_Je': 3.0
    }
    
    exp2 = Experiment2_UserScalability(base_config=config)
    
    results = exp2.run(
        algorithms=['Random', 'Equal', 'Heuristic', 'PSO', 'DDPG'],
        episodes_per_eval=50,
        train_episodes=500,
        seeds=[42]
    )
    
    exp2.plot_results('experiment2_quick.png')
    exp2.save_results('experiment2_quick.json')
    
    return exp2


def run_full_experiments():
    """
    完整运行两个实验（按照文档要求）
    """
    print("="*80)
    print("完整运行补充实验")
    print("="*80)
    
    config = {
        'P_max': 10.0,
        'noise_power': 1.0,
        'R_min': 1.0,
        'L1': 1.0,
        'L2': 2.0,
        'L_J1': 0.5,
        'L_J2': 0.5,
        'L_e': 1.5,
        'L_Je': 3.0
    }
    
    # 实验一
    print("\n" + "="*80)
    print("开始实验一：窃听者强度变化实验")
    print("="*80)
    
    exp1 = Experiment1_EavesdropperStrength(base_config=config)
    results1 = exp1.run(
        algorithms=['Random', 'Equal', 'Heuristic', 'PSO', 'DDPG', 'DDPG_wo_Jammer'],
        num_users=2,
        episodes_per_eval=200,
        train_episodes=3000,
        seeds=[42, 123, 456, 789, 1011]
    )
    exp1.plot_results('experiment1_full.png')
    exp1.save_results('experiment1_full.json')
    
    # 实验二
    print("\n" + "="*80)
    print("开始实验二：用户数量扩展实验")
    print("="*80)
    
    exp2 = Experiment2_UserScalability(base_config=config)
    results2 = exp2.run(
        algorithms=['Random', 'Equal', 'Heuristic', 'PSO', 'DDPG'],
        episodes_per_eval=200,
        train_episodes=3000,
        seeds=[42, 123, 456, 789, 1011]
    )
    exp2.plot_results('experiment2_full.png')
    exp2.save_results('experiment2_full.json')
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)
    
    return exp1, exp2


def demo_single_run_enhanced():
    """
    增强版单次演示：调整信道设置使Jammer效果显现
    """
    print("="*80)
    print("增强版演示：Jammer效果显著的NOMA安全传输场景")
    print("="*80)
    print("场景设定：窃听者能力强 + Jammer能有效干扰窃听者")
    print("-"*80)

    # 场景A：原始设置（Jammer效果弱）
    print("\n【场景A】原始设置：Jammer远离窃听者")
    env_weak = NOMASecureEnv(
        num_users=2,
        P_max=10.0,
        noise_power=1.0,
        R_min=1.0,
        eavesdropper_gain_scale=1.0,  # 窃听者强度一般
        L1=1.0, L2=2.0,
        L_J1=0.5, L_J2=0.5,  # Jammer离用户很近
        L_e=1.5,
        L_Je=3.0,  # Jammer离窃听者很远 ← 问题在这里！
        use_jammer=True,
        seed=42
    )

    print(f"信道参数: L_e={env_weak.L_e}, L_Je={env_weak.L_Je}, "
          f"L_J1={env_weak.L_J_users[0]}, L_J2={env_weak.L_J_users[1]}")
    print(f"Jammer→窃听者链路损耗 / 基站→窃听者链路损耗 = {env_weak.L_Je/env_weak.L_e:.2f}")
    print("(比值>1表示Jammer对窃听者效果差)")

    # 快速对比：有/无Jammer
    state = env_weak.reset()
    action_with_jammer = np.array([0.4, 0.4, 0.2])
    action_no_jammer = np.array([0.5, 0.5, 0.0])

    _, reward_w, _, info_w = env_weak.step(action_with_jammer)
    env_weak.current_channels['state'] = state  # 恢复状态
    _, reward_n, _, info_n = env_weak.step(action_no_jammer)

    print(f"\n固定动作对比:")
    print(f"  有Jammer (0.4,0.4,0.2): 保密容量={info_w['secrecy_sum']:.4f}, 奖励={reward_w:.4f}")
    print(f"  无Jammer (0.5,0.5,0.0): 保密容量={info_n['secrecy_sum']:.4f}, 奖励={reward_n:.4f}")
    print(f"  Jammer收益: {info_w['secrecy_sum'] - info_n['secrecy_sum']:+.4f}")

    # 场景B：方案D - 综合调整使Jammer效果显现
    print("\n" + "="*80)
    print("【场景B】方案D：综合调整使Jammer效果显著")
    print("-"*80)
    print("调整策略：")
    print("  1. 降低QoS要求: R_min=0.5 (原1.0)")
    print("  2. 增加功率预算: P_max=15.0 (原10.0)")
    print("  3. Jammer靠近窃听者: L_Je=0.5 (原3.0)")
    print("  4. Jammer远离合法用户: L_J1=2.0, L_J2=2.0 (原0.5)")
    print("  5. 窃听者离基站远: L_e=2.0 (原1.5), gain_scale=1.0 (原2.0)")
    env_strong = NOMASecureEnv(
        num_users=2,
        P_max=15.0,                   # 功率预算增加
        noise_power=1.0,
        R_min=0.5,                    # QoS要求降低
        eavesdropper_gain_scale=1.0,  # 窃听者增益恢复正常
        L1=1.0, L2=2.0,
        L_J1=2.0, L_J2=2.0,           # Jammer离用户更远
        L_e=2.0,                      # 窃听者离基站更远
        L_Je=0.5,                     # Jammer离窃听者更近！
        use_jammer=True,
        seed=42
    )

    print(f"信道参数: L_e={env_strong.L_e}, L_Je={env_strong.L_Je}, "
          f"L_J1={env_strong.L_J_users[0]}, L_J2={env_strong.L_J_users[1]}")
    print(f"Jammer→窃听者链路损耗 / 基站→窃听者链路损耗 = {env_strong.L_Je/env_strong.L_e:.2f}")
    print("(比值<1表示Jammer对窃听者效果好)")

    state = env_strong.reset()
    _, reward_w2, _, info_w2 = env_strong.step(action_with_jammer)
    env_strong.current_channels['state'] = state
    _, reward_n2, _, info_n2 = env_strong.step(action_no_jammer)

    print(f"\n固定动作对比:")
    print(f"  有Jammer (0.4,0.4,0.2): 保密容量={info_w2['secrecy_sum']:.4f}, 奖励={reward_w2:.4f}")
    print(f"  无Jammer (0.5,0.5,0.0): 保密容量={info_n2['secrecy_sum']:.4f}, 奖励={reward_n2:.4f}")
    print(f"  Jammer收益: {info_w2['secrecy_sum'] - info_n2['secrecy_sum']:+.4f}")

    # 在场景B上训练多个DRL算法
    print("\n" + "="*80)
    print("在场景B上训练DRL算法 (2000 episodes)...")
    print("="*80)

    # 训练DDPG
    print("\n[1/3] 训练 DDPG...")
    agent_ddpg = DDPGAgent(env_strong.state_dim, env_strong.action_dim)
    trainer_ddpg = Trainer(env_strong, agent_ddpg, 'DDPG')
    trainer_ddpg.train(episodes=2000, steps_per_episode=100, eval_interval=200)

    # 训练TD3
    print("\n[2/3] 训练 TD3...")
    agent_td3 = TD3Agent(env_strong.state_dim, env_strong.action_dim)
    trainer_td3 = Trainer(env_strong, agent_td3, 'TD3')
    trainer_td3.train(episodes=2000, steps_per_episode=100, eval_interval=200)

    # 训练SAC
    print("\n[3/3] 训练 SAC...")
    agent_sac = SACAgent(env_strong.state_dim, env_strong.action_dim)
    trainer_sac = Trainer(env_strong, agent_sac, 'SAC')
    trainer_sac.train(episodes=2000, steps_per_episode=100, eval_interval=200)

    # 评估所有DRL算法
    print("\n" + "="*80)
    print("评估所有DRL算法（场景B）...")
    print("="*80)
    evaluator = Evaluator(env_strong)
    results = evaluator.evaluate_all(
        {'DDPG': agent_ddpg, 'TD3': agent_td3, 'SAC': agent_sac},
        episodes=100, verbose=True
    )

    # 对比：with Jammer vs without Jammer
    print("\n" + "="*80)
    print("DRL with Jammer vs DRL without Jammer 对比")
    print("="*80)

    env_no_jammer = NOMASecureEnv(
        num_users=2,
        P_max=15.0,
        noise_power=1.0,
        R_min=0.5,
        eavesdropper_gain_scale=1.0,
        L1=1.0, L2=2.0,
        L_J1=2.0, L_J2=2.0,
        L_e=2.0,
        L_Je=0.5,
        use_jammer=False,
        seed=42
    )

    # 训练无Jammer版本
    print("\n训练无Jammer版本...")
    agent_ddpg_nj = DDPGAgent(env_no_jammer.state_dim, env_no_jammer.action_dim)
    trainer_ddpg_nj = Trainer(env_no_jammer, agent_ddpg_nj, 'DDPG_noJammer')
    trainer_ddpg_nj.train(episodes=2000, steps_per_episode=100, eval_interval=200)

    agent_td3_nj = TD3Agent(env_no_jammer.state_dim, env_no_jammer.action_dim)
    trainer_td3_nj = Trainer(env_no_jammer, agent_td3_nj, 'TD3_noJammer')
    trainer_td3_nj.train(episodes=2000, steps_per_episode=100, eval_interval=200)

    agent_sac_nj = SACAgent(env_no_jammer.state_dim, env_no_jammer.action_dim)
    trainer_sac_nj = Trainer(env_no_jammer, agent_sac_nj, 'SAC_noJammer')
    trainer_sac_nj.train(episodes=2000, steps_per_episode=100, eval_interval=200)

    evaluator2 = Evaluator(env_no_jammer)
    results2 = evaluator2.evaluate_all(
        {'DDPG_noJammer': agent_ddpg_nj, 'TD3_noJammer': agent_td3_nj, 'SAC_noJammer': agent_sac_nj},
        episodes=100, verbose=False
    )

    print("\n" + "-"*80)
    print(f"{'算法':<20} {'有Jammer保密容量':<18} {'无Jammer保密容量':<18} {'Jammer提升':<12} {'Jammer功率':<12}")
    print("-"*80)
    for algo in ['DDPG', 'TD3', 'SAC']:
        with_jammer = results[algo]['avg_secrecy_sum']
        without_jammer = results2[f"{algo}_noJammer"]['avg_secrecy_sum']
        improvement = with_jammer - without_jammer
        jammer_power = results[algo]['avg_jammer_power']
        print(f"{algo:<20} {with_jammer:<18.4f} {without_jammer:<18.4f} {improvement:+<12.4f} {jammer_power:<12.4f}")

    # 测试不同功率分配策略
    print("\n" + "="*80)
    print("测试不同功率分配策略（场景B）:")
    print("="*80)
    test_actions = {
        '全给U1': np.array([1.0, 0.0, 0.0]),
        '全给U2': np.array([0.0, 1.0, 0.0]),
        '全给Jammer': np.array([0.0, 0.0, 1.0]),
        '等功率+满Jammer': np.array([0.5, 0.5, 1.0]),
        'Jammer主导': np.array([0.2, 0.2, 1.0]),
    }

    for name, action in test_actions.items():
        reward, info = env_strong.evaluate_action(action)
        print(f"{name:16s}: 奖励={reward:8.4f}, 保密容量={info['secrecy_sum']:.4f}, "
              f"QoS={'满足' if info['qos_satisfied'] else '不满足'}, "
              f"Jammer功率={info['jammer_power']:.2f}")

    # 额外测试：观察DRL学到的策略
    print("\n" + "="*80)
    print("观察DRL学到的策略（随机3个信道状态）:")
    print("="*80)
    for i in range(3):
        state = env_strong.reset()
        action_ddpg = agent_ddpg.select_action(state, evaluate=True)
        action_td3 = agent_td3.select_action(state, evaluate=True)
        action_sac = agent_sac.select_action(state, evaluate=True)

        _, _, _, info_ddpg = env_strong.step(action_ddpg)
        env_strong.current_channels['state'] = state
        _, _, _, info_td3 = env_strong.step(action_td3)
        env_strong.current_channels['state'] = state
        _, _, _, info_sac = env_strong.step(action_sac)

        print(f"\n信道状态 {i+1}:")
        print(f"  DDPG: 动作=[{action_ddpg[0]:.3f}, {action_ddpg[1]:.3f}, {action_ddpg[2]:.3f}], "
              f"保密容量={info_ddpg['secrecy_sum']:.4f}, Jammer={info_ddpg['jammer_power']:.2f}")
        print(f"  TD3:  动作=[{action_td3[0]:.3f}, {action_td3[1]:.3f}, {action_td3[2]:.3f}], "
              f"保密容量={info_td3['secrecy_sum']:.4f}, Jammer={info_td3['jammer_power']:.2f}")
        print(f"  SAC:  动作=[{action_sac[0]:.3f}, {action_sac[1]:.3f}, {action_sac[2]:.3f}], "
              f"保密容量={info_sac['secrecy_sum']:.4f}, Jammer={info_sac['jammer_power']:.2f}")

    return env_strong, {'DDPG': agent_ddpg, 'TD3': agent_td3, 'SAC': agent_sac}, results


if __name__ == "__main__":
    import sys

    # 选择运行模式
    print("NOMA安全传输 - 补充实验实现")
    print("="*80)
    print("请选择运行模式：")
    print("1. 单次演示（快速验证环境）")
    print("2. 快速实验（减少轮数，快速查看趋势）")
    print("3. 完整实验（按照文档要求，耗时较长）")
    print("="*80)

    # 支持命令行参数或交互式输入
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        try:
            choice = input("请输入选项 (1/2/3，默认1): ").strip() or "1"
        except EOFError:
            choice = "1"
            print("未检测到输入，默认运行模式1（单次演示）...")

    if choice == "1":
        demo_single_run_enhanced()
    elif choice == "2":
        print("\n运行快速实验...")
        exp1 = run_experiment1_quick()
        exp2 = run_experiment2_quick()
    elif choice == "3":
        print("\n运行完整实验（这可能需要数小时）...")
        run_full_experiments()
    else:
        print("无效选项，运行单次演示...")
        demo_single_run_enhanced()
