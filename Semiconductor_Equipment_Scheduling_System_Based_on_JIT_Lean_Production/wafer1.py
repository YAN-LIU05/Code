import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 环境参数
NUM_MACHINES_GROUP1 = 8  # 组1机器0-7
NUM_MACHINES_GROUP2 = 8  # 组2机器8-15
VALVE_PAIRS = [(3,8), (4,15)]  # 阀门连接对
PROC_TIME = {
    2:72, 5:72,          # 组1长加工(可互换)
    6:202, 9:202, 11:202, # 组2长加工
    4:[2,72,72],          # 组1的4号(阀门2)分阶段时间
    15:[2,72,72]          # 组2的15号(阀门2)同步时间
}
BASE_PROC_SEQ = [3,11,15,9,15,6,15]  # 基础加工序列（不含第一步和最后一步）
ENTRY_EXIT_POS = [0, 7]  # 可选的进口/出口位置
STEP2_OPTIONS = [2, 5]   # 第一步加工可选位置
PICK_PLACE_TIME = 4  
ROTATE_TIME_PER_45 = 0.5  
NUM_WAFERS = 10  # 晶圆数量

# 强化学习参数
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

# 设备状态常量
IDLE = 0
PROCESSING = 1
WAITING = 2

class Wafer:
    def __init__(self, wafer_id, entry_time=0.0):
        self.id = wafer_id
        self.step = 0
        self.group = 0  # 0=组1, 1=组2
        self.position = 0  
        self.slot = -1  # -1:不在槽位, 0-3对应slot1-4
        self.progress = 0
        self.completed = False
        self.entry_time = entry_time
        self.last_process_time = entry_time
        self.valve_pass_count = 0
        self.entry_pos = random.choice(ENTRY_EXIT_POS)  # 随机选择进口位置
        self.exit_pos = random.choice(ENTRY_EXIT_POS)   # 随机选择出口位置
        self.step2_pos = random.choice(STEP2_OPTIONS)   # 随机选择步骤2的加工位置
        self.proc_seq = [self.entry_pos, self.step2_pos] + BASE_PROC_SEQ + [self.exit_pos]  # 单次加工序列

class Environment:
    def __init__(self):
        self.state_dim = None
        self.reset()
        
    def reset(self):
        self.time = 0.0
        self.arm1_pos = 0
        self.arm1_slots = [None, None]
        self.arm1_busy_until = 0.0
        self.arm2_pos = 8 
        self.arm2_slots = [None, None]
        self.arm2_busy_until = 0.0
        
        self.machines = []
        for i in range(NUM_MACHINES_GROUP1 + NUM_MACHINES_GROUP2):
            self.machines.append({
                'status': IDLE,
                'wafer': None,
                'end_time': -1.0,
                'group': 0 if i < NUM_MACHINES_GROUP1 else 1
            })
        
        self.wafers = [Wafer(i) for i in range(NUM_WAFERS)]
        first_wafer = self.wafers[0]
        entry_pos = first_wafer.entry_pos
        self.machines[entry_pos]['status'] = WAITING
        self.machines[entry_pos]['wafer'] = first_wafer
        first_wafer.position = entry_pos
        
        self.completed_wafers = 0
        self.next_wafer_idx = 1
        self.last_wafer_entry_time = 0.0
        self.step_count = 0
        
        state = self._get_state()
        if self.state_dim is None:
            self.state_dim = len(state)
        return state

    def _get_valve_pair(self, pos):
        for p1, p2 in VALVE_PAIRS:
            if pos == p1: return p2
            if pos == p2: return p1
        return None

    def _get_state(self):
        state = []
        
        # 机械臂状态
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
        
        # 机器状态
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
        
        # 晶圆状态
        for wafer in self.wafers:
            state.append(wafer.group)
            state.append(wafer.position/15.0)
            state.append(wafer.step/25.0)
            state.append(wafer.slot if wafer.slot >=0 else -1.0)
            state.append((self.time - wafer.last_process_time)/100.0)
            state.append(wafer.entry_pos/7.0)
            state.append(wafer.exit_pos/7.0)
            state.append(wafer.step2_pos/7.0)
        
        # 全局状态
        state.append(self.time / 2000.0)
        state.append(self.completed_wafers / NUM_WAFERS)
        
        return np.array(state, dtype=np.float32)

    def _rotate_arm(self, arm_idx, target_pos):
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

    def _get_processing_time(self, machine_pos, wafer):
        if machine_pos in [4,15] and wafer.valve_pass_count < 3:
            return PROC_TIME[machine_pos][wafer.valve_pass_count]
        return PROC_TIME.get(machine_pos, 2)

    def step(self, action):
        self.step_count += 1
        if self.step_count >= MAX_STEPS:
            return self._get_state(), -100, True, {"reason": "max_steps"}
        
        # 1. 解码动作
        arm1_pos = action // (8 * 9)
        arm2_pos = (action % (8 * 9)) // 9
        slot_actions = action % 9
        
        # 2. 旋转机械臂
        rotate_time1 = self._rotate_arm(0, arm1_pos)
        rotate_time2 = self._rotate_arm(1, arm2_pos + NUM_MACHINES_GROUP1)
        reward = -(rotate_time1 + rotate_time2) * 0.05
        
        # 3. 执行槽位操作
        valid_actions = 0
        for arm_idx in [0, 1]:
            current_arm_pos = self.arm1_pos if arm_idx == 0 else self.arm2_pos
            slots = self.arm1_slots if arm_idx == 0 else self.arm2_slots
            
            for slot_idx in range(2):
                action_type = (slot_actions // (3**slot_idx)) % 3
                if action_type == 0: continue
                
                if action_type == 1:  # 拾取
                    if slots[slot_idx] is None and self.machines[current_arm_pos]['status'] == WAITING:
                        wafer = self.machines[current_arm_pos]['wafer']
                        slots[slot_idx] = wafer
                        wafer.slot = slot_idx + (0 if arm_idx == 0 else 2)
                        wafer.position = -1
                        self.machines[current_arm_pos]['wafer'] = None
                        self.machines[current_arm_pos]['status'] = IDLE
                        valid_actions += 1
                        reward += 2.0
                
                elif action_type == 2:  # 放置
                    if slots[slot_idx] is not None:
                        wafer = slots[slot_idx]
                        target_pos = current_arm_pos + (4 if arm_idx == 0 else -4) % 8
                        target_pos += (0 if arm_idx == 0 else NUM_MACHINES_GROUP1)
                        
                        valve_pair = self._get_valve_pair(target_pos)
                        if valve_pair is not None:
                            if wafer.group != arm_idx:
                                target_pos = valve_pair
                                wafer.group = 1 - wafer.group
                                if target_pos in [4,15]:
                                    wafer.valve_pass_count += 1
                        
                        if self.machines[target_pos]['status'] == IDLE:
                            slots[slot_idx] = None
                            wafer.slot = -1
                            wafer.position = target_pos
                            self.machines[target_pos]['wafer'] = wafer
                            
                            proc_time = self._get_processing_time(target_pos, wafer)
                            self.machines[target_pos]['end_time'] = self.time + proc_time
                            self.machines[target_pos]['status'] = PROCESSING
                            valid_actions += 1
                            reward += 5.0
        
        # 4. 更新时间
        next_event_time = self.time + 1.0  # 最小时间单位
        
        # 考虑机械臂忙碌时间
        next_event_time = min(next_event_time, 
                            max(self.arm1_busy_until, self.arm2_busy_until))
        
        # 考虑机器处理时间
        processing_ends = [m['end_time'] for m in self.machines 
                         if m['status'] == PROCESSING and m['end_time'] > self.time]
        if processing_ends:
            next_event_time = min(next_event_time, min(processing_ends))
        
        self.time = next_event_time
        
        # 5. 更新机器状态
        for i, machine in enumerate(self.machines):
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
        
        # 6. 晶圆投放
        if (self.next_wafer_idx < NUM_WAFERS and 
            self.time - self.last_wafer_entry_time >= 15.0):
            
            # 检查所有可能的入口位置是否有空闲
            for entry_pos in ENTRY_EXIT_POS:
                if self.machines[entry_pos]['status'] == IDLE:
                    wafer = self.wafers[self.next_wafer_idx]
                    self.machines[entry_pos]['wafer'] = wafer
                    self.machines[entry_pos]['status'] = WAITING
                    wafer.position = entry_pos
                    wafer.entry_time = self.time
                    self.last_wafer_entry_time = self.time
                    self.next_wafer_idx += 1
                    break
        
        # 7. 计算奖励
        progress_reward = sum(w.step for w in self.wafers) * 0.5
        time_penalty = -self.time * 0.01
        reward += progress_reward + time_penalty
        
        # 8. 终止条件
        done = self.completed_wafers >= NUM_WAFERS
        if done:
            reward += 500.0
        
        return self._get_state(), reward, done, {
            "completed": self.completed_wafers,
            "time": self.time,
            "valid_actions": valid_actions
        }

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class Agent:
    def __init__(self, env):
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

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return 0.0

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

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

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn(episodes=250):  # 训练轮次减半
    env = Environment()
    agent = Agent(env)
    
    total_rewards = []
    episode_times = []
    moving_avg_rewards = []
    start_time = time.time()

    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < MAX_STEPS:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train()
            
            state = next_state
            total_reward += reward
            step_count += 1

        agent.update_epsilon()
        total_rewards.append(total_reward)
        episode_times.append(env.time)
        
        if episode >= 10:
            moving_avg = np.mean(total_rewards[-10:])
            moving_avg_rewards.append(moving_avg)

        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:]) if len(total_rewards)>=10 else np.mean(total_rewards)
            print(f"\nEpisode {episode}: Reward={total_reward:.1f}, "
                  f"Avg10={avg_reward:.1f}, Epsilon={agent.epsilon:.3f}, "
                  f"Time={env.time:.2f}s, Loss={loss:.4f}")

    torch.save(agent.policy_net.state_dict(), "dqn_wafer_scheduler.pth")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_rewards, alpha=0.5, label='每轮奖励')
    plt.plot(moving_avg_rewards, 'r-', label='10轮平均')
    plt.xlabel('训练轮次')
    plt.ylabel('奖励')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(episode_times, 'g-')
    plt.xlabel('训练轮次')
    plt.ylabel('耗时(秒)')
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

    print(f"训练完成! 总耗时: {time.time()-start_time:.1f}秒")
    return total_rewards, episode_times

def test_dqn(episodes=5):
    env = Environment()
    agent = Agent(env)
    agent.load("dqn_wafer_scheduler.pth")
    agent.epsilon = 0.01

    total_times = []
    detailed_logs = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            steps.append({
                'time': env.time,
                'arm1_pos': env.arm1_pos,
                'arm2_pos': env.arm2_pos,
                'slot1': env.arm1_slots[0].id if env.arm1_slots[0] else None,
                'slot2': env.arm1_slots[1].id if env.arm1_slots[1] else None,
                'slot3': env.arm2_slots[0].id if env.arm2_slots[0] else None,
                'slot4': env.arm2_slots[1].id if env.arm2_slots[1] else None,
                'action': action,
                'reward': reward,
                'step2_pos': env.wafers[0].step2_pos
            })
            state = next_state

        total_time = env.time
        total_times.append(total_time)
        detailed_logs.append(steps)
        print(f"Test Episode {episode}: Total Time: {total_time:.2f} seconds, "
              f"Step2 Pos: {env.wafers[0].step2_pos}")

    avg_time = np.mean(total_times)
    print(f"Average Completion Time: {avg_time:.2f} seconds")

    with open('simulation_log.txt', 'w') as f:
        for i, log in enumerate(detailed_logs):
            f.write(f"\n=== Episode {i} === (Total Time: {total_times[i]:.2f}s, Step2 Pos: {log[0]['step2_pos']})\n")
            for step in log:
                f.write(f"Time: {step['time']:.2f}s | "
                       f"Arm1: {step['arm1_pos']} | Arm2: {step['arm2_pos']} | "
                       f"Slots: {step['slot1']}/{step['slot2']}/{step['slot3']}/{step['slot4']} | "
                       f"Action: {step['action']} | Reward: {step['reward']:.2f}\n")

    return total_times, detailed_logs

if __name__ == "__main__":
    # 训练轮次减半为250
    train_rewards, train_times = train_dqn(episodes=250)
    
    # 测试保持不变
    test_times, test_logs = test_dqn(episodes=5)
    
    # 打印最终结果
    print(f"\n训练平均耗时: {np.mean(train_times):.2f}秒")
    print(f"测试平均耗时: {np.mean(test_times):.2f}秒")
    print(f"测试中使用的step2位置分布: {[log[0]['step2_pos'] for log in test_logs]}")