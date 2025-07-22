from collections import deque
from heapq import heappush, heappop
import json

# 步骤持续时间（从修订表格）
step_durations = {
    0: 15, 1: 4, 2: 1, 3: 1, 4: 1, 5: 4, 6: 8, 7: 4, 8: 1, 9: 1, 10: 4, 11: 1, 12: 15,
    13: 1, 14: 4, 15: 1, 16: 1, 17: 1, 18: 4, 19: 1, 20: 70, 21: 1, 22: 4, 23: 1, 24: 1,
    25: 1, 26: 4, 27: 1, 28: 70, 29: 1, 30: 4, 31: 1, 32: 1, 33: 1, 34: 4, 35: 1, 36: 1,
    37: 4, 38: 1, 39: 0.5, 40: 1, 41: 4, 42: 1, 43: 70, 44: 1, 45: 4, 46: 1, 47: 1.5,
    48: 1, 49: 4, 50: 1, 51: 1, 52: 4, 53: 1, 54: 1.5, 55: 1, 56: 4, 57: 1, 58: 20,
    59: 1, 60: 4, 61: 1, 62: 1, 63: 1, 64: 4
}

# 步骤列表（包含循环：20–57 × 4，20–50 × 1）
steps = (
    list(range(0, 20)) +
    [i for i in range(20, 58)] * 4 +
    list(range(20, 51)) +
    list(range(58, 65))
)

# 模块资源
units = ['LLB', 'LP1', 'AL', 'LLA', 'LLC', 'LLD', 'PM7', 'PM8', 'TM1', 'TM2', 'TM3']
unit_available = {unit: 0 for unit in units}
unit_last_used = {unit: 0 for unit in units}  # 跟踪模块最后使用时间
pm_wafer_count = {'PM7': 0, 'PM8': 0}  # 跟踪每个PM处理的晶圆数

# 步骤到执行单元的映射（基于表格“执行单元”）
step_units = {
    0: 'LLB', 1: 'TM1', 2: 'TM1', 3: 'TM1', 4: 'TM1', 5: 'TM1', 6: 'AL', 7: 'TM1',
    8: 'TM1', 9: 'TM1', 10: 'TM1', 11: 'LLA', 12: 'LLA', 13: 'LLA', 14: 'TM2',
    15: 'TM2', 16: 'TM2', 17: 'TM2', 18: 'TM2', 19: 'PM7', 20: 'PM7', 21: 'TM2',
    22: 'TM2', 23: 'TM2', 24: 'TM2', 25: 'TM2', 26: 'TM2', 27: 'PM8', 28: 'PM8',
    29: 'TM2', 30: 'TM2', 31: 'TM2', 32: 'TM2', 33: 'TM2', 34: 'TM2', 35: 'LLC',
    36: 'LLC', 37: 'TM3', 38: 'TM3', 39: 'TM3', 40: 'TM3', 41: 'TM3', 42: 'TM3',
    43: 'LLD', 44: 'LLD', 45: 'TM2', 46: 'TM2', 47: 'TM2', 48: 'TM2', 49: 'TM2',
    50: 'LLB', 51: 'LLB', 52: 'TM2', 53: 'TM2', 54: 'TM2', 55: 'TM2', 56: 'TM2',
    57: 'PM7', 58: 'LLB', 59: 'TM1', 60: 'TM1', 61: 'TM1', 62: 'TM1', 63: 'TM1', 64: 'TM1'
}

# 初始化 75 片晶圆任务
NUM_WAFERS = 75
wafer_tasks = [deque(steps) for _ in range(NUM_WAFERS)]
wafer_paths = [[] for _ in range(NUM_WAFERS)]
move_list = []
event_queue = []
conflict_log = []
cleaning_log = []
unit_usage = {unit: [] for unit in units}
move_id_counter = 0

# 清洗参数
IDLE_THRESHOLD = 70  # 空闲时间阈值（秒）
IDLE_CLEAN_DURATION = 30  # 空闲清洗持续时间（秒）
WAFER_COUNT_THRESHOLD = 13  # 每处理13个晶圆
WAFER_COUNT_CLEAN_DURATION = 300  # 晶圆计数清洗持续时间（秒）

# 动作持续时间假设
AUXILIARY_MOVE_DURATION = 1  # 辅助动作（如Pickmove, Placemove等）持续时间（秒）

# MoveType 映射函数
def get_move_types(step, module, start_time, end_time, wafer_id):
    global move_id_counter
    mat_id = f"{wafer_id + 1}.{step}"
    moves = []
    duration = step_durations[step]

    # 对于持续时间小于3秒的步骤，仅使用主要动作
    if duration < 3:
        move_type = {
            0: 6, 1: 1, 2: 5, 3: 3, 4: 4, 5: 2, 6: 10, 7: 1, 8: 3, 9: 4, 10: 2, 11: 5,
            12: 6, 13: 4, 14: 1, 15: 5, 16: 3, 17: 4, 18: 2, 19: 5, 20: 8, 21: 4, 22: 1,
            23: 5, 24: 3, 25: 4, 26: 2, 27: 5, 28: 8, 29: 4, 30: 1, 31: 5, 32: 3, 33: 4,
            34: 2, 35: 5, 36: 4, 37: 1, 38: 5, 39: 3, 40: 4, 41: 2, 42: 5, 43: 8, 44: 4,
            45: 1, 46: 5, 47: 3, 48: 4, 49: 2, 50: 5, 51: 4, 52: 1, 53: 5, 54: 3, 55: 4,
            56: 2, 57: 5, 58: 7, 59: 4, 60: 1, 61: 5, 62: 3, 63: 4, 64: 2
        }[step]
        moves.append({
            "StartTime": start_time,
            "EndTime": end_time,
            "MoveID": move_id_counter,
            "MoveType": move_type,
            "ModuleName": module,
            "MatID": mat_id,
            "SlotID": 1
        })
        move_id_counter += 1
        return moves

    # 特殊步骤：抽气、充气、校准
    if step in [0, 12]:  # LL抽气 (PumpMove)
        moves.extend([
            {"StartTime": start_time, "EndTime": start_time + AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter, "MoveType": 4, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + AUXILIARY_MOVE_DURATION, "EndTime": end_time - AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 1, "MoveType": 6, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": end_time - AUXILIARY_MOVE_DURATION, "EndTime": end_time, "MoveID": move_id_counter + 2, "MoveType": 5, "ModuleName": module, "MatID": mat_id, "SlotID": 1}
        ])
        move_id_counter += 3
    elif step == 58:  # LL充气 (VentMove)
        moves.extend([
            {"StartTime": start_time, "EndTime": start_time + AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter, "MoveType": 4, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + AUXILIARY_MOVE_DURATION, "EndTime": end_time - AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 1, "MoveType": 7, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": end_time - AUXILIARY_MOVE_DURATION, "EndTime": end_time, "MoveID": move_id_counter + 2, "MoveType": 5, "ModuleName": module, "MatID": mat_id, "SlotID": 1}
        ])
        move_id_counter += 3
    elif step == 6:  # 校准 (AlignMove)
        moves.extend([
            {"StartTime": start_time, "EndTime": start_time + AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter, "MoveType": 4, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + AUXILIARY_MOVE_DURATION, "EndTime": end_time - AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 1, "MoveType": 10, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": end_time - AUXILIARY_MOVE_DURATION, "EndTime": end_time, "MoveID": move_id_counter + 2, "MoveType": 5, "ModuleName": module, "MatID": mat_id, "SlotID": 1}
        ])
        move_id_counter += 3
    elif step in [20, 28, 43]:  # 加工 (ProcessMove)
        moves.extend([
            {"StartTime": start_time, "EndTime": start_time + AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter, "MoveType": 1, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + AUXILIARY_MOVE_DURATION, "EndTime": start_time + 2 * AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 1, "MoveType": 3, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + 2 * AUXILIARY_MOVE_DURATION, "EndTime": start_time + 3 * AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 2, "MoveType": 2, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + 3 * AUXILIARY_MOVE_DURATION, "EndTime": start_time + 4 * AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 3, "MoveType": 4, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + 4 * AUXILIARY_MOVE_DURATION, "EndTime": end_time - AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 4, "MoveType": 8, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": end_time - AUXILIARY_MOVE_DURATION, "EndTime": end_time, "MoveID": move_id_counter + 5, "MoveType": 5, "ModuleName": module, "MatID": mat_id, "SlotID": 1}
        ])
        move_id_counter += 6
    else:  # 其他步骤（传输相关）
        moves.extend([
            {"StartTime": start_time, "EndTime": start_time + AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter, "MoveType": 1, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + AUXILIARY_MOVE_DURATION, "EndTime": start_time + 2 * AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 1, "MoveType": 3, "ModuleName": module, "MatID": mat_id, "SlotID": 1},
            {"StartTime": start_time + 2 * AUXILIARY_MOVE_DURATION, "EndTime": end_time, "MoveID": move_id_counter + 2, "MoveType": 2, "ModuleName": module, "MatID": mat_id, "SlotID": 1}
        ])
        move_id_counter += 3

    return moves

# 清洗移动记录函数
def get_cleaning_move(unit, start_time, end_time):
    global move_id_counter
    mat_id = f"CLEAN.{unit}"
    moves = [
        {"StartTime": start_time, "EndTime": start_time + AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter, "MoveType": 4, "ModuleName": unit, "MatID": mat_id, "SlotID": 1},
        {"StartTime": start_time + AUXILIARY_MOVE_DURATION, "EndTime": end_time - AUXILIARY_MOVE_DURATION, "MoveID": move_id_counter + 1, "MoveType": 9, "ModuleName": unit, "MatID": mat_id, "SlotID": 1},
        {"StartTime": end_time - AUXILIARY_MOVE_DURATION, "EndTime": end_time, "MoveID": move_id_counter + 2, "MoveType": 5, "ModuleName": unit, "MatID": mat_id, "SlotID": 1}
    ]
    move_id_counter += 3
    return moves

# 初始任务分配
for i in range(NUM_WAFERS):
    step = wafer_tasks[i].popleft()
    unit = step_units[step]
    start_time = unit_available[unit]
    end_time = start_time + step_durations[step]
    unit_available[unit] = end_time
    unit_last_used[unit] = end_time
    if unit in pm_wafer_count:
        pm_wafer_count[unit] += 1
    wafer_paths[i].append((step, start_time, end_time))
    unit_usage[unit].append((start_time, end_time, i + 1, step))
    move_list.extend(get_move_types(step, unit, start_time, end_time, i))
    heappush(event_queue, (end_time, i, step, 'wafer'))

# 模拟执行
max_completion_time = 0
while event_queue:
    current_time, wafer_id, completed_step, event_type = heappop(event_queue)
    max_completion_time = max(max_completion_time, current_time)

    if event_type == 'wafer':
        if not wafer_tasks[wafer_id]:
            continue
        next_step = wafer_tasks[wafer_id].popleft()
        unit = step_units[next_step]
        start_time = max(current_time, unit_available[unit])

        # 检查PM模块的空闲时间
        if unit in ['PM7', 'PM8']:
            idle_time = current_time - unit_last_used[unit]
            if idle_time >= IDLE_THRESHOLD and start_time >= current_time:
                clean_start = current_time
                clean_end = clean_start + IDLE_CLEAN_DURATION
                unit_available[unit] = clean_end
                unit_last_used[unit] = clean_end
                unit_usage[unit].append((clean_start, clean_end, 0, 'clean'))
                move_list.extend(get_cleaning_move(unit, clean_start, clean_end))
                cleaning_log.append({
                    'unit': unit,
                    'reason': 'idle',
                    'start_time': clean_start,
                    'end_time': clean_end
                })
                print(f"模块 {unit} 在 {clean_start:.1f} 秒因空闲 {idle_time:.1f} 秒开始清洗，持续 {IDLE_CLEAN_DURATION} 秒")
                start_time = max(start_time, clean_end)

        # 检查PM模块的晶圆计数
        if unit in pm_wafer_count:
            if pm_wafer_count[unit] >= WAFER_COUNT_THRESHOLD:
                clean_start = max(current_time, unit_available[unit])
                clean_end = clean_start + WAFER_COUNT_CLEAN_DURATION
                unit_available[unit] = clean_end
                unit_last_used[unit] = clean_end
                unit_usage[unit].append((clean_start, clean_end, 0, 'clean'))
                move_list.extend(get_cleaning_move(unit, clean_start, clean_end))
                cleaning_log.append({
                    'unit': unit,
                    'reason': 'wafer_count',
                    'start_time': clean_start,
                    'end_time': clean_end,
                    'wafer_count': pm_wafer_count[unit]
                })
                print(f"模块 {unit} 在 {clean_start:.1f} 秒因处理 {pm_wafer_count[unit]} 个晶圆开始清洗，持续 {WAFER_COUNT_CLEAN_DURATION} 秒")
                pm_wafer_count[unit] = 0
                start_time = max(start_time, clean_end)

        # 分配下一步
        if start_time > current_time:
            delay = start_time - current_time
            conflict_log.append({
                'wafer_id': wafer_id + 1,
                'step': next_step,
                'unit': unit,
                'conflict_time': current_time,
                'delay': delay
            })
            print(f"晶圆 {wafer_id + 1} 在步骤 {next_step} 延迟 {delay:.1f} 秒")

        end_time = start_time + step_durations[next_step]
        unit_available[unit] = end_time
        unit_last_used[unit] = end_time
        if unit in pm_wafer_count:
            pm_wafer_count[unit] += 1
        wafer_paths[wafer_id].append((next_step, start_time, end_time))
        unit_usage[unit].append((start_time, end_time, wafer_id + 1, next_step))
        move_list.extend(get_move_types(next_step, unit, start_time, end_time, wafer_id))
        heappush(event_queue, (end_time, wafer_id, next_step, 'wafer'))

# 写入 JSON 文件
with open('task_1_wafer_trajectory.json', 'w') as f:
    json.dump({"MoveList": move_list}, f, indent=4)

# 检查模块占用重叠
def check_overlap(unit_usage):
    overlap_issues = []
    for unit, intervals in unit_usage.items():
        intervals.sort()
        for i in range(1, len(intervals)):
            prev_start, prev_end, prev_wafer, prev_step = intervals[i - 1]
            curr_start, curr_end, curr_wafer, curr_step = intervals[i]
            if curr_start < prev_end:
                overlap_issues.append({
                    'unit': unit,
                    'wafer1': prev_wafer,
                    'step1': prev_step,
                    'time1_start': prev_start,
                    'time1_end': prev_end,
                    'wafer2': curr_wafer,
                    'step2': curr_step,
                    'time2_start': curr_start,
                    'time2_end': curr_end
                })
    return overlap_issues

# 输出晶圆轨迹
for i in range(NUM_WAFERS):
    print(f"\n晶圆 {i + 1} 的执行路径:")
    for step, st, et in wafer_paths[i]:
        print(f"  步骤 {step:<2} 开始: {st:<8.1f} 结束: {et:<8.1f}")

# 输出总完成时间
print(f"\n总完成时间: {max_completion_time:.1f} 秒")

# 输出冲突日志
print("\n模块冲突日志:")
if conflict_log:
    for conflict in conflict_log:
        print(f"晶圆 {conflict['wafer_id']:<3} 步骤 {conflict['step']:<2} "
              f"模块 {conflict['unit']:<4} 冲突时间: {conflict['conflict_time']:<8.1f} "
              f"延迟: {conflict['delay']:.1f} 秒")
else:
    print("无模块冲突。")

# 输出清洗日志
print("\n清洗日志:")
if cleaning_log:
    for clean in cleaning_log:
        print(f"模块 {clean['unit']:<4} 因 {clean['reason']:<12} "
              f"开始: {clean['start_time']:<8.1f} 结束: {clean['end_time']:<8.1f}")
else:
    print("无清洗事件。")

# 输出重叠检查
overlap_issues = check_overlap(unit_usage)
print("\n模块占用重叠检查:")
if overlap_issues:
    print("发现模块重叠：")
    for issue in overlap_issues:
        print(f"模块 {issue['unit']:<4} 冲突：")
        print(f"  晶圆 {issue['wafer1']:<3} 步骤 {issue['step1']:<2} "
              f"开始: {issue['time1_start']:<8.1f} 结束: {issue['time1_end']:<8.1f}")
        print(f"  晶圆 {issue['wafer2']:<3} 步骤 {issue['step2']:<2} "
              f"开始: {issue['time2_start']:<8.1f} 结束: {issue['time2_end']:<8.1f}")
else:
    print("无模块重叠，模拟可行。")