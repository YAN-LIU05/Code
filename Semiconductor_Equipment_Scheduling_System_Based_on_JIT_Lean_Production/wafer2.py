import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import json
import seaborn as sns
from typing import Dict, List, Tuple

# ==================== 环境参数 ====================
NUM_MACHINES_GROUP1 = 8  # 组1机器0-7
NUM_MACHINES_GROUP2 = 8  # 组2机器8-15
VALVE_PAIRS = [(3,8), (4,15)]  # 阀门连接对
PROC_TIME = {
    2:72, 5:72,          # 组1长加工(可互换)
    6:202, 9:202, 11:202, # 组2长加工
    4:[2,72,72],          # 组1的4号(阀门2)分阶段时间
    15:[2,72,72]          # 组2的15号(阀门2)同步时间
}
BASE_PROC_SEQ = [3,11,15,9,15,6,15]  # 基础加工序列
ENTRY_EXIT_POS = [0, 7]  # 可选的进口/出口位置
STEP2_OPTIONS = [2, 5]   # 第一步加工可选位置
PICK_PLACE_TIME = 4  
ROTATE_TIME_PER_45 = 0.5  
NUM_WAFERS = 10  # 默认晶圆数量

# ==================== 强化学习参数 ====================
ACTION_DIM = (8 + 8) * (3 ** 2)  # 两组位置×槽位操作组合
HIDDEN_DIM = 512  
LEARNING_RATE = 0.0005
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 5000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 200
MAX_STEPS = 8000  

# ==================== 设备状态常量 ====================
IDLE = 0
PROCESSING = 1
WAITING = 2

# ==================== 晶圆类 ====================
class Wafer:
    def __init__(self, wafer_id: int, entry_time: float = 0.0):
        self.id = wafer_id
        self.step = 0
        self.group = 0  # 0=组1, 1=组2
        self.position = 0  
        self.slot = -1  # -1:不在槽位, 0-3对应slot1-4
        self.completed = False
        self.entry_time = entry_time
        self.last_process_time = entry_time
        self.valve_pass_count = 0
        self.entry_pos = random.choice(ENTRY_EXIT_POS)
        self.exit_pos = random.choice(ENTRY_EXIT_POS)
        self.step2_pos = random.choice(STEP2_OPTIONS)
        self.proc_seq = self._generate_processing_sequence()
        self.history = []  # 记录完整加工路径

    def _generate_processing_sequence(self) -> List[int]:
        """生成单次加工序列：入口 → 第一步加工 → 基础序列 → 出口"""
        return [self.entry_pos, self.step2_pos] + BASE_PROC_SEQ + [self.exit_pos]

    def record_step(self, machine_pos: int):
        """记录加工步骤和时间戳"""
        self.history.append((self.step, machine_pos, time.time()))

# ==================== 增强环境类 ====================
class EnhancedEnvironment:
    def __init__(self):
        self.state_dim = None
        self.performance_metrics = {
            'wafer_counts': [],
            'completion_times': [],
            'module_utilization': {},
            'action_types': {0: 0, 1: 0, 2: 0},
            'path_lengths': [],
            'valve_transfers': 0
        }
        self.episode = -1
        self.time = 0.0
        self.reset()
        
    def reset(self):
        """重置环境并保存上一轮的统计数据"""
        if hasattr(self, 'time') and hasattr(self, 'machines') and self.time > 0:
            total_time = self.time
            self.performance_metrics['module_utilization'][self.episode] = {
                m['id']: m.get('busy_time', 0)/total_time if total_time > 0 else 0 
                for m in self.machines
            }
            self.performance_metrics['completion_times'].append(total_time)
            self.performance_metrics['wafer_counts'].append(self.completed_wafers)
            
            if hasattr(self, 'current_wafer') and self.current_wafer:
                self.performance_metrics['path_lengths'].append(len(self.current_wafer.history))
            
            for k, v in getattr(self, 'action_counts', {0:0, 1:0, 2:0}).items():
                self.performance_metrics['action_types'][k] += v

        self.episode += 1
        self.time = 0.0
        self.current_wafer = None
        self.arm1_pos = 0
        self.arm1_slots = [None, None]
        self.arm1_busy_until = 0.0
        self.arm2_pos = 8 
        self.arm2_slots = [None, None]
        self.arm2_busy_until = 0.0
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        self.machines = []
        for i in range(NUM_MACHINES_GROUP1 + NUM_MACHINES_GROUP2):
            self.machines.append({
                'id': i,
                'status': IDLE,
                'wafer': None,
                'end_time': -1.0,
                'group': 0 if i < NUM_MACHINES_GROUP1 else 1,
                'busy_time': 0.0
            })
        
        self.wafers = [Wafer(i) for i in range(NUM_WAFERS)]
        self.current_wafer = self.wafers[0]
        entry_pos = self.current_wafer.entry_pos
        self.machines[entry_pos]['status'] = WAITING
        self.machines[entry_pos]['wafer'] = self.current_wafer
        self.current_wafer.position = entry_pos
        self.current_wafer.record_step(entry_pos)
        
        self.completed_wafers = 0
        self.next_wafer_idx = 1
        self.last_wafer_entry_time = 0.0
        self.step_count = 0
        
        state = self._get_state()
        if self.state_dim is None:
            self.state_dim = len(state)
        return state

    def _get_valve_pair(self, pos: int) -> int:
        """获取阀门连接的另一端位置"""
        for p1, p2 in VALVE_PAIRS:
            if pos == p1: return p2
            if pos == p2: return p1
        return None

    def _get_state(self) -> np.ndarray:
        """获取当前环境状态向量"""
        state = []
        
        for arm_pos, slots, busy in [(self.arm1_pos, self.arm1_slots, self.arm1_busy_until),
                                    (self.arm2_pos, self.arm2_slots, self.arm2_busy_until)]:
            state.append(arm_pos / 15.0)
            state.append(max(0, busy - self.time) / 100.0)
            for slot in slots:
                state.append(1.0 if slot else 0.0)
                state.append(slot.id if slot else -1.0)
                state.append(slot.step/25.0 if slot else 0.0)
                if slot:
                    state.append(slot.entry_pos/7.0)
                    state.append(slot.exit_pos/7.0)
                    state.append(slot.step2_pos/7.0)
                else:
                    state.extend([0.0, 0.0, 0.0])
        
        for machine in self.machines:
            state.append(machine['status'] / 2.0)
            state.append(machine['wafer'].id if machine['wafer'] else -1.0)
            state.append(machine['wafer'].step/25.0 if machine['wafer'] else 0.0)
            state.append(max(0, machine['end_time'] - self.time)/300.0 if machine['status'] == PROCESSING else 0.0)
            if machine['wafer']:
                state.append(machine['wafer'].entry_pos/7.0)
                state.append(machine['wafer'].exit_pos/7.0)
                state.append(machine['wafer'].step2_pos/7.0)
            else:
                state.extend([0.0, 0.0, 0.0])
        
        for wafer in self.wafers:
            state.append(wafer.group)
            state.append(wafer.position/15.0)
            state.append(wafer.step/25.0)
            state.append(wafer.slot if wafer.slot >=0 else -1.0)
            state.append((self.time - wafer.last_process_time)/100.0)
            state.append(wafer.entry_pos/7.0)
            state.append(wafer.exit_pos/7.0)
            state.append(wafer.step2_pos/7.0)
        
        state.append(self.time / 2000.0)
        state.append(self.completed_wafers / NUM_WAFERS)
        
        return np.array(state, dtype=np.float32)

    def _rotate_arm(self, arm_idx: int, target_pos: int) -> float:
        """旋转机械臂并返回所需时间"""
        if arm_idx == 0:
            current_pos = self.arm1_pos
            diff = abs(target_pos - current_pos)
            rotate_time = min(diff, NUM_MACHINES_GROUP1 - diff) * ROTATE_TIME_PER_45
            self.arm1_pos = target_pos
            self.arm1_busy_until = self.time + rotate_time
        else:
            current_pos = self.arm2_pos
            diff = abs(target_pos - current_pos - NUM_MACHINES_GROUP1)
            rotate_time = min(diff, NUM_MACHINES_GROUP2 - diff) * ROTATE_TIME_PER_45
            self.arm2_pos = target_pos
            self.arm2_busy_until = self.time + rotate_time
        return rotate_time

    def _get_processing_time(self, machine_pos: int, wafer: Wafer) -> float:
        """获取当前机器位置的加工时间"""
        if machine_pos in [4,15] and wafer.valve_pass_count < 3:
            return PROC_TIME[machine_pos][wafer.valve_pass_count]
        return PROC_TIME.get(machine_pos, 2)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一个时间步"""
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            return self._get_state(), -100, True, {"reason": "max_steps"}
        
        arm1_pos = action // (8 * 9)
        arm2_pos = (action % (8 * 9)) // 9
        slot_actions = action % 9
        
        rotate_time1 = self._rotate_arm(0, arm1_pos)
        rotate_time2 = self._rotate_arm(1, arm2_pos + NUM_MACHINES_GROUP1)
        reward = -(rotate_time1 + rotate_time2) * 0.05
        
        valid_actions = 0
        for arm_idx in [0, 1]:
            current_arm_pos = self.arm1_pos if arm_idx == 0 else self.arm2_pos
            slots = self.arm1_slots if arm_idx == 0 else self.arm2_slots
            
            for slot_idx in range(2):
                action_type = (slot_actions // (3**slot_idx)) % 3
                self.action_counts[action_type] += 1
                
                if action_type == 1:
                    if slots[slot_idx] is None and self.machines[current_arm_pos]['status'] == WAITING:
                        wafer = self.machines[current_arm_pos]['wafer']
                        slots[slot_idx] = wafer
                        wafer.slot = slot_idx + (0 if arm_idx == 0 else 2)
                        wafer.position = -1
                        self.machines[current_arm_pos]['wafer'] = None
                        self.machines[current_arm_pos]['status'] = IDLE
                        valid_actions += 1
                        reward += 2.0
                
                elif action_type == 2:
                    if slots[slot_idx] is not None:
                        wafer = slots[slot_idx]
                        target_pos = current_arm_pos + (4 if arm_idx == 0 else -4) % 8
                        target_pos += (0 if arm_idx == 0 else NUM_MACHINES_GROUP1)
                        
                        valve_pair = self._get_valve_pair(target_pos)
                        if valve_pair is not None and wafer.group != arm_idx:
                            target_pos = valve_pair
                            wafer.group = 1 - wafer.group
                            if target_pos in [4,15]:
                                wafer.valve_pass_count += 1
                            self.performance_metrics['valve_transfers'] += 1
                        
                        if self.machines[target_pos]['status'] == IDLE:
                            slots[slot_idx] = None
                            wafer.slot = -1
                            wafer.position = target_pos
                            self.machines[target_pos]['wafer'] = wafer
                            
                            proc_time = self._get_processing_time(target_pos, wafer)
                            self.machines[target_pos]['end_time'] = self.time + proc_time
                            self.machines[target_pos]['status'] = PROCESSING
                            self.machines[target_pos]['busy_time'] += proc_time
                            wafer.record_step(target_pos)
                            valid_actions += 1
                            reward += 5.0
        
        next_event_time = self.time + 1.0
        next_event_time = min(next_event_time, 
                            max(self.arm1_busy_until, self.arm2_busy_until))
        
        processing_ends = [m['end_time'] for m in self.machines 
                         if m['status'] == PROCESSING and m['end_time'] > self.time]
        if processing_ends:
            next_event_time = min(next_event_time, min(processing_ends))
        
        self.time = next_event_time
        
        for machine in self.machines:
            if machine['status'] == PROCESSING and self.time >= machine['end_time']:
                wafer = machine['wafer']
                if wafer:
                    wafer.step += 1
                    wafer.last_process_time = self.time
                    
                    if wafer.step >= len(wafer.proc_seq):
                        wafer.completed = True
                        self.completed_wafers += 1
                        machine['wafer'] = None
                        machine['status'] = IDLE
                        reward += 20.0
                    else:
                        next_pos = wafer.proc_seq[wafer.step]
                        if machine['group'] == 0 and next_pos >= NUM_MACHINES_GROUP1:
                            next_pos = self._get_valve_pair(next_pos) or next_pos
                        machine['status'] = WAITING
        
        if (self.next_wafer_idx < NUM_WAFERS and 
            self.time - self.last_wafer_entry_time >= 15.0):
            for entry_pos in ENTRY_EXIT_POS:
                if self.machines[entry_pos]['status'] == IDLE:
                    self.current_wafer = self.wafers[self.next_wafer_idx]
                    self.machines[entry_pos]['wafer'] = self.current_wafer
                    self.machines[entry_pos]['status'] = WAITING
                    self.current_wafer.position = entry_pos
                    self.current_wafer.entry_time = self.time
                    self.last_wafer_entry_time = self.time
                    self.next_wafer_idx += 1
                    self.current_wafer.record_step(entry_pos)
                    break
        
        progress_reward = sum(w.step for w in self.wafers) * 0.5
        time_penalty = -self.time * 0.01
        reward += progress_reward + time_penalty
        
        done = self.completed_wafers >= NUM_WAFERS
        if done:
            reward += 500.0
        
        # 确保reward是数值类型
        reward = float(reward)
        
        return self._get_state(), reward, done, {
            "completed": self.completed_wafers,
            "time": self.time,
            "valid_actions": valid_actions
        }

# ==================== DQN网络 ====================
class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# ==================== 智能体类 ====================
class Agent:
    def __init__(self, env: EnhancedEnvironment):
        self.env = env
        self.action_dim = ACTION_DIM
        self.policy_net = DQN(env.state_dim, self.action_dim)
        self.target_net = DQN(env.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        print(f"Using device: {self.device}")

    def act(self, state: np.ndarray) -> int:
        """ε-greedy策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def store_transition(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """存储转移样本到经验回放缓冲区"""
        if not isinstance(reward, (int, float)):
            reward = float(reward)
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self) -> float:
        """执行一次训练"""
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return 0.0

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        try:
            # 确保rewards是数值类型
            rewards = np.array(rewards, dtype=np.float32)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
        except Exception as e:
            print(f"张量转换错误: {e}")
            print(f"rewards内容: {rewards}")
            return 0.0

        current_q = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """保存模型权重"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """加载模型权重"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==================== 训练和评估函数 ====================
def train_and_evaluate(wafer_counts: List[int] = [5, 10, 15, 20], 
                      scenarios: int = 4) -> Tuple[pd.DataFrame, dict]:
    """训练和评估不同配置下的性能"""
    results = []
    global NUM_WAFERS, BASE_PROC_SEQ, VALVE_PAIRS
    
    for count in wafer_counts:
        scenario_results = []
        NUM_WAFERS = count
        
        for scenario in range(scenarios):
            if scenario == 0:
                BASE_PROC_SEQ = [3,11,15,9,15,6,15]
                VALVE_PAIRS = [(3,8), (4,15)]
            elif scenario == 1:
                BASE_PROC_SEQ = [3,11,15,6,15,9,15]
                VALVE_PAIRS = [(3,8), (4,15)]
            elif scenario == 2:
                BASE_PROC_SEQ = [3,11,15,9,15,6,15]
                VALVE_PAIRS = [(3,8), (4,15)]
            else:
                BASE_PROC_SEQ = [3,11,15,9,15,6,15]
                VALVE_PAIRS = [(3,8)]
            
            env = EnhancedEnvironment()
            agent = Agent(env)
            
            for _ in tqdm(range(100), desc=f"Training Wafers={count}, Scenario={scenario+1}"):
                state = env.reset()
                done = False
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.store_transition(state, action, reward, next_state, done)
                    agent.train()
                agent.update_epsilon()
            
            test_times = []
            for _ in range(5):
                state = env.reset()
                done = False
                while not done:
                    action = agent.act(state)
                    state, _, done, _ = env.step(action)
                test_times.append(env.time)
            
            scenario_results.append(np.mean(test_times))
            print(f"Scenario {scenario+1} - Wafers {count}: Avg Time = {np.mean(test_times):.2f}s")
        
        results.append({
            'Wafer Count': count,
            'Scenario 1': scenario_results[0],
            'Scenario 2': scenario_results[1],
            'Scenario 3': scenario_results[2],
            'Scenario 4': scenario_results[3]
        })
    
    return pd.DataFrame(results), env.performance_metrics

# ==================== 主执行流程 ====================
if __name__ == "__main__":
    print("="*50)
    print("晶圆调度系统强化学习训练与评估")
    print("="*50)
    
    result_df, metrics = train_and_evaluate(
        wafer_counts=[5, 10, 15, 20],
        scenarios=4
    )
    
    print("\n不同晶圆数量性能表:")
    print(result_df.to_string(index=False))
    
    print("\n关键性能指标:")
    print(f"- 平均完成时间: {np.mean(metrics['completion_times']):.2f}秒")
    print(f"- 平均路径长度: {np.mean(metrics['path_lengths']):.1f}步")
    print(f"- 阀门传输次数: {metrics['valve_transfers']}次")
    print(f"- 动作分布: 无操作={metrics['action_types'][0]}, 拾取={metrics['action_types'][1]}, 放置={metrics['action_types'][2]}")
    
    result_df.to_csv('performance_results.csv', index=False)
    with open('performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n所有分析结果已保存:")
    print("- 性能表格: performance_results.csv")
    print("- 详细指标: performance_metrics.json")