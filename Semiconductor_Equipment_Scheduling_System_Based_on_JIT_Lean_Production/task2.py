from collections import deque
from heapq import heappush, heappop
import json
import random
import math
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import copy
import logging

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Module definitions (unchanged)
step_durations = {
    'LP1': 5, 'TM1': 6, 'AL': 8, 'LLA': 15, 'LLB': 15, 'LLC': 15, 'LLD': 70,
    'PM7': 70, 'PM8': 70, 'PM3': 200, 'PM1': 200, 'PM10': 200, 'TM2': 13, 'TM3': 11.5
}
module_action_durations = {'LLA': 2, 'LLB': 2, 'LLC': 2, 'LLD': 5}
steps = [
    'LP1', 'TM1', 'AL', 'LLA/LLB', 'TM2', 'PM7/PM8', 'TM2', 'LLC',
    'TM3', 'PM3', 'TM3', 'LLD', 'TM3', 'PM1', 'TM3', 'LLD', 'TM2',
    'PM10', 'TM2', 'LLD', 'TM2', 'LLA/LLB', 'TM1', 'LP1'
]
units = ['LP1', 'TM1', 'AL', 'LLA', 'LLB', 'LLC', 'LLD', 'PM7', 'PM8', 'PM3', 'PM1', 'PM10', 'TM2', 'TM3']
unit_slots = {
    unit: {1: {'available_time': 0, 'wafer_id': None}, 2: {'available_time': 0, 'wafer_id': None}}
    if unit in ['LLA', 'LLB', 'LLC', 'LLD', 'TM1', 'TM2', 'TM3'] else {1: {'available_time': 0, 'wafer_id': None}}
    for unit in units
}
unit_slot_queue_length = {
    unit: {1: 0, 2: 0} if unit in ['LLA', 'LLB', 'LLC', 'LLD', 'TM1', 'TM2', 'TM3'] else {1: 0}
    for unit in units
}
max_slot_queue_length = {
    unit: {1: 0, 2: 0} if unit in ['LLA', 'LLB', 'LLC', 'LLD', 'TM1', 'TM2', 'TM3'] else {1: 0}
    for unit in units
}
step_units = {
    'LP1': ['LP1'], 'TM1': ['TM1'], 'AL': ['AL'], 'LLA/LLB': ['LLA', 'LLB'],
    'TM2': ['TM2'], 'PM7/PM8': ['PM7', 'PM8'], 'LLC': ['LLC'], 'TM3': ['TM3'],
    'PM3': ['PM3'], 'PM1': ['PM1'], 'LLD': ['LLD'], 'PM10': ['PM10']
}
NUM_WAFERS = 75
default_params = {
    'w1': 0.5, 'w2': 0.3, 'w3': 0.2, 'time_window': 20,
    'module_preference': {'LLA': 0.5, 'LLB': 0.5, 'PM7': 0.5, 'PM8': 0.5},
    'slot_preference': {unit: {1: 0.5, 2: 0.5} for unit in ['LLA', 'LLB', 'LLC', 'LLD', 'TM1', 'TM2', 'TM3']},
    'conflict_penalty': 2.0, 'process_priority': 0.2
}

def get_move_types(step, module, slot_id, start_time, end_time, wafer_id, move_id_counter):
    moves = []
    mat_id = f"{wafer_id + 1}.{step}"
    move_type = {
        'LP1': 1, 'TM1': 3, 'TM2': 3, 'TM3': 3, 'AL': 10,
        'LLA': 2, 'LLB': 2, 'LLC': 2, 'LLD': 2, 'PM7': 8, 'PM8': 8,
        'PM3': 8, 'PM1': 8, 'PM10': 8
    }.get(module, 3)
    moves.append({
        "StartTime": start_time,
        "EndTime": end_time,
        "MoveID": move_id_counter,
        "MoveType": move_type,
        "ModuleName": module,
        "MatID": mat_id,
        "SlotID": slot_id
    })
    return moves, move_id_counter + 1

def select_module_and_slot(step, params, adaptive=True, module_score_cache=None):
    if module_score_cache is None:
        module_score_cache = {}
    
    candidates = step_units[step]
    if not adaptive:
        unit = random.choice(candidates)
        slots = [1, 2] if unit in ['LLA', 'LLB', 'LLC', 'LLD', 'TM1', 'TM2', 'TM3'] else [1]
        return unit, random.choice(slots)
    
    cache_key = (step, tuple(sorted(params['module_preference'].items())))
    if cache_key in module_score_cache:
        return module_score_cache[cache_key]
    
    best_score = -float('inf')
    best_unit, best_slot = None, None
    for unit in candidates:
        for slot in unit_slots[unit]:
            load_factor = sum(max_slot_queue_length[unit].values()) / 10
            slot_load = unit_slots[unit][slot]['available_time'] / 10000
            score = (params['module_preference'].get(unit, 0.5) * (1 - load_factor) +
                     params['slot_preference'].get(unit, {1: 0.5, 2: 0.5})[slot] * (1 - slot_load))
            if score > best_score:
                best_score = score
                best_unit = unit
                best_slot = slot
    module_score_cache[cache_key] = (best_unit, best_slot)
    return best_unit, best_slot

def handle_disruption(unit, slot, current_time, disruption_type, unit_slots, unit_slot_queue_length):
    if disruption_type == 'none' or random.random() > 0.05:
        return False
    if disruption_type in ['fault', 'mixed']:
        unit_slots[unit][slot]['available_time'] = current_time + 100
        unit_slots[unit][slot]['wafer_id'] = None
        unit_slot_queue_length[unit][slot] = max(0, unit_slot_queue_length[unit][slot] - 1)
        return True
    return False

def run_scheduling(params, wafer_tasks, step_units, step_durations, disruption_type='none', adaptive=True):
    try:
        # Initialize local state
        local_unit_slots = copy.deepcopy(unit_slots)
        local_unit_slot_queue_length = copy.deepcopy(unit_slot_queue_length)
        local_max_slot_queue_length = copy.deepcopy(max_slot_queue_length)
        move_list = []
        conflict_log = []
        unit_usage = {unit: [] for unit in units}
        move_id_counter = 0
        event_queue = []
        wafer_paths = [[] for _ in range(NUM_WAFERS)]
        wafer_tasks = [deque(steps) for _ in range(NUM_WAFERS)]
        max_completion_time = 0
        last_tm_action = {unit: 0 for unit in ['TM1', 'TM2', 'TM3']}
        module_score_cache = {}
        
        # Initial task allocation
        for i in range(NUM_WAFERS):
            step = wafer_tasks[i].popleft()
            unit, slot = select_module_and_slot(step, params, adaptive, module_score_cache)
            local_unit_slot_queue_length[unit][slot] += 1
            local_max_slot_queue_length[unit][slot] = max(
                local_max_slot_queue_length[unit][slot], local_unit_slot_queue_length[unit][slot]
            )
            start_time = local_unit_slots[unit][slot]['available_time']
            if unit in ['TM1', 'TM2', 'TM3']:
                start_time = max(start_time, last_tm_action[unit] - 4)
                last_tm_action[unit] = start_time + step_durations[unit]
            end_time = start_time + step_durations[unit]
            local_unit_slots[unit][slot]['available_time'] = end_time
            local_unit_slots[unit][slot]['wafer_id'] = i + 1
            wafer_paths[i].append((step, start_time, end_time, unit, slot))
            unit_usage[unit].append((start_time, end_time, i + 1, step, slot))
            moves, move_id_counter = get_move_types(step, unit, slot, start_time, end_time, i, move_id_counter)
            move_list.extend(moves)
            heappush(event_queue, (end_time, i, step, unit, slot, 0))
        
        # Event loop
        step_indices = {wafer_id: 0 for wafer_id in range(NUM_WAFERS)}
        while event_queue:
            current_time, wafer_id, completed_step, completed_unit, completed_slot, _ = heappop(event_queue)
            max_completion_time = max(max_completion_time, current_time)
            local_unit_slot_queue_length[completed_unit][completed_slot] -= 1
            local_unit_slots[completed_unit][completed_slot]['wafer_id'] = None
            step_indices[wafer_id] += 1
            
            if handle_disruption(completed_unit, completed_slot, current_time, disruption_type,
                               local_unit_slots, local_unit_slot_queue_length):
                continue
            
            if not wafer_tasks[wafer_id]:
                continue
            
            next_step = wafer_tasks[wafer_id].popleft()
            unit, slot = select_module_and_slot(next_step, params, adaptive, module_score_cache)
            local_unit_slot_queue_length[unit][slot] += 1
            local_max_slot_queue_length[unit][slot] = max(
                local_max_slot_queue_length[unit][slot], local_unit_slot_queue_length[unit][slot]
            )
            
            start_time = max(current_time, local_unit_slots[unit][slot]['available_time'])
            if unit in ['TM1', 'TM2', 'TM3']:
                start_time = max(start_time, last_tm_action[unit] - 4)
                last_tm_action[unit] = start_time + step_durations[unit]
            
            end_time = start_time + step_durations[unit]
            local_unit_slots[unit][slot]['available_time'] = end_time
            local_unit_slots[unit][slot]['wafer_id'] = wafer_id + 1
            wafer_paths[wafer_id].append((next_step, start_time, end_time, unit, slot))
            unit_usage[unit].append((start_time, end_time, wafer_id + 1, next_step, slot))
            moves, move_id_counter = get_move_types(next_step, unit, slot, start_time, end_time, wafer_id, move_id_counter)
            move_list.extend(moves)
            
            heappush(event_queue, (end_time, wafer_id, next_step, unit, slot, 0))
        
        # Write JSON
        if adaptive==True:
            with open(f'task_2_wafer_trajectory_{disruption_type}_adaptive.json', 'w') as f:
                json.dump({"MoveList": move_list}, f, indent=4)
        else:
            with open(f'task_2_wafer_trajectory_{disruption_type}.json', 'w') as f:
                json.dump({"MoveList": move_list}, f, indent=4)
        
        return {
            'makespan': max_completion_time,
            'num_conflicts': len(conflict_log),
            'load_balance': sum(sum(v for v in slots.values()) for slots in local_max_slot_queue_length.values())
        }
    except Exception as e:
        logging.error(f"Error in run_scheduling (disruption={disruption_type}, adaptive={adaptive}): {str(e)}")
        return {'makespan': float('inf'), 'num_conflicts': 0, 'load_balance': 0}

def optimize_parameters():
    try:
        best_params = default_params.copy()
        best_cost = float('inf')
        T = 1000
        alpha = 0.95
        
        for _ in range(5):
            params = default_params.copy()
            params['w1'] = random.uniform(0.4, 0.6)
            params['w2'] = random.uniform(0.2, 0.4)
            params['w3'] = 1 - params['w1'] - params['w2']
            
            for _ in range(100):
                new_params = params.copy()
                new_params['w1'] += random.uniform(-0.05, 0.05)
                new_params['w2'] += random.uniform(-0.05, 0.05)
                new_params['w3'] = max(0, 1 - new_params['w1'] - new_params['w2'])
                if new_params['w3'] < 0:
                    continue
                cost = run_scheduling(params, [], step_units, step_durations, adaptive=False)['makespan']
                new_cost = run_scheduling(new_params, [], step_units, step_durations, adaptive=False)['makespan']
                if new_cost < cost or random.random() < math.exp(-(new_cost - cost) / T):
                    params = new_params
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_params = new_params
                T *= alpha
        
        return best_params
    except Exception as e:
        logging.error(f"Error in optimize_parameters: {str(e)}")
        return default_params

def run_single_experiment(args):
    scenario, params, adaptive = args
    try:
        result = run_scheduling(params, [], step_units, step_durations, scenario, adaptive)
        return scenario, result, adaptive, params
    except Exception as e:
        logging.error(f"Error in run_single_experiment (scenario={scenario}, adaptive={adaptive}): {str(e)}")
        return scenario, {'makespan': float('inf'), 'num_conflicts': 0, 'load_balance': 0}, adaptive, params

def run_validation_experiments():
    scenarios = ['none', 'fault', 'time_variation', 'mixed']
    scenario_labels = {'none': '无干扰', 'fault': '故障', 'time_variation': '时间变化', 'mixed': '混合干扰'}
    modes = ['baseline', 'static', 'adaptive']
    mode_labels = {'baseline': '基线', 'static': '静态', 'adaptive': '自适应'}
    results = {
        mode: {s: {'makespan': [], 'num_conflicts': [], 'load_balance': []} for s in scenarios}
        for mode in modes
    }
    static_params = optimize_parameters()
    
    tasks = [(s, default_params, False) for s in scenarios] * 50 + \
            [(s, static_params, False) for s in scenarios] * 50 + \
            [(s, static_params, True) for s in scenarios] * 50
    
    with Pool(4) as pool:
        for scenario, result, adaptive, params in tqdm(pool.imap_unordered(run_single_experiment, tasks),
                                                    total=len(tasks), desc="运行实验"):
            mode = 'baseline' if params == default_params and not adaptive else \
                   'static' if params == static_params and not adaptive else 'adaptive'
            results[mode][scenario]['makespan'].append(result['makespan'])
            results[mode][scenario]['num_conflicts'].append(result['num_conflicts'])
            results[mode][scenario]['load_balance'].append(result['load_balance'])
    
    # 计算统计数据
    summary = {}
    for s in scenarios:
        summary[s] = {}
        for m in modes:
            makespan_data = [x for x in results[m][s]['makespan'] if x != float('inf')]
            summary[s][m] = {
                'makespan_mean': statistics.mean(makespan_data) if makespan_data else 0,
                'makespan_std': statistics.stdev(makespan_data) if len(makespan_data) > 1 else 0,
                'conflicts_mean': statistics.mean(results[m][s]['num_conflicts']) if results[m][s]['num_conflicts'] else 0,
                'load_balance_mean': statistics.mean(results[m][s]['load_balance']) if results[m][s]['load_balance'] else 0
            }
    
    # 绘制四个参数的图表
    parameters = [
        ('makespan_mean', '总完成时间均值', '总完成时间均值 (秒)', 'task_2_makespan_mean.png'),
        ('makespan_std', '总完成时间标准差', '总完成时间标准差 (秒)', 'task_2_makespan_std.png'),
        ('conflicts_mean', '冲突次数均值', '冲突次数均值', 'task_2_conflicts_mean.png'),
        ('load_balance_mean', '负载均衡均值', '负载均衡均值', 'task_2_load_balance_mean.png')
    ]
    
    for param_key, param_title, y_label, filename in parameters:
        plt.figure(figsize=(10, 6))
        x = range(len(scenarios))
        width = 0.25
        
        for i, mode in enumerate(modes):
            values = []
            for scenario in scenarios:
                value = summary[scenario][mode][param_key]
                values.append(value if value else 0)
            plt.bar([xi + width * i for xi in x], values, width, label=mode_labels[mode])
        
        plt.xlabel('场景')
        plt.ylabel(y_label)
        plt.title(param_title)
        plt.xticks([xi + width for xi in x], [scenario_labels[s] for s in scenarios])
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    return summary

if __name__ == '__main__':
    try:
        summary = run_validation_experiments()
        print(json.dumps(summary, indent=4))
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")