- 该代码为第九届集创赛“北方华创杯”基于JIT精益生产的半导体设备调度系统赛题的解决代码示例

## 环境与文件夹结构

### 环境

- 使用Python 3.10

### 文件夹结构

```bash
当前文件夹/
│
├── task_1_wafer_trajectory.json              # 晶圆轨迹数据（Task 1）
│
├── 可视化图像（Task 2评估图）/
│   ├── task_2_conflicts_mean.png            # 平均冲突次数
│   ├── task_2_load_balance_mean.png         # 平均负载均衡
│   ├── task_2_makespan_mean.png             # 平均完工时间
│   └── task_2_makespan_std.png              # 完工时间标准差
│
├── 晶圆轨迹数据（Task 2各实验条件）/
│   ├── task_2_wafer_trajectory_none.json
│   ├── task_2_wafer_trajectory_none_adaptive.json
│   ├── task_2_wafer_trajectory_fault.json
│   ├── task_2_wafer_trajectory_fault_adaptive.json
│   ├── task_2_wafer_trajectory_time_variation.json
│   ├── task_2_wafer_trajectory_time_variation_adaptive.json
│   ├── task_2_wafer_trajectory_mixed.json
│   └── task_2_wafer_trajectory_mixed_adaptive.json
│
├── 脚本文件/
│   ├── task1.py                             # Task 1 处理脚本
│   └── task2.py                             # Task 2 处理脚本
│
├── requirements.txt                        # Python库配置文档
└── README.txt                              # 说明文档
```

### task1.py和task2.py

- `task1.py` 和 `task2.py` 分别是任务一和任务二的代码，其余是生成的 `json` 和 `png` 文件。

### task1.py 生成的 JSON 文件

- **文件名**: task_1_wafer_trajectory.json
  - **场景**: 无干扰（none）
  - **模式**: 基线模式（按晶圆编号顺序分配任务，未使用优化参数或自适应策略）
  - **内容**:
    - 记录75片晶圆的完整调度轨迹，包含每个晶圆在各步骤的动作记录。
    - 字段包括开始时间（StartTime）、结束时间（EndTime）、动作ID（MoveID）、动作类型（MoveType）、模块名称（ModuleName）、晶圆ID与步骤组合（MatID）和插槽ID（SlotID）。
    - ```json
       {
          "MoveList": [
              {
                  "StartTime": 0,
                  "EndTime": 1,
                  "MoveID": 0,
                  "MoveType": 4,
                  "ModuleName": "LLB",
                  "MatID": "1.0",
                  "SlotID": 1
              }
          ]
      }

      ```
  - **用途**:
    - 验证调度方案是否满足赛题要求（如晶圆编号顺序、阀门互斥、JIT时间约束）。
    - 分析晶圆加工路径、模块占用情况和时间分布。
    - 提供数据支持，用于后续可视化（如Gantt图）或算法优化。

### task2.py 生成的 JSON 文件

- **文件名**: task_2_wafer_trajectory_{disruption_type}_{mode}.json
  - **disruption_type**: 表示实验场景（none：无干扰，fault：故障，time_variation：时间变化，mixed：混合干扰）
  - **mode**: 表示调度模式（空或adaptive，分别对应静态模式和自适应模式；基线模式未单独生成JSON，但其结果包含在性能评估中）
  -

#### 每个 JSON 文件的详细说明

- **task_2_wafer_trajectory_none.json**

  - 场景: 无干扰（none）
  - 模式: 静态模式（使用优化参数，随机选择模块和插槽）
  - 内容: 记录75片晶圆在无干扰环境下的完整调度轨迹，包含每个晶圆的加工步骤、模块分配、插槽选择及时间信息。
  - 用途: 用于分析理想条件下调度算法的性能，验证JIT时间约束（节点驻留≤15秒，转移≤30秒）。
  - ```json
    {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
        ]
    }
    ```
- **task_2_wafer_trajectory_none_adaptive.json**

  - 场景: 无干扰（none）
  - 模式: 自适应模式（使用优化参数，基于负载和偏好动态选择模块和插槽）
  - 内容: 同上，但采用自适应调度策略，优先选择负载较低的模块和插槽。
  - 用途: 对比静态模式，评估自适应策略在无干扰场景下的效果。
    - ```json
      {
          "MoveList": [
              {
                  "StartTime": 0,
                  "EndTime": 5,
                  "MoveID": 0,
                  "MoveType": 1,
                  "ModuleName": "LP1",
                  "MatID": "1.LP1",
                  "SlotID": 1
              }
          ]
      }
      ```
- **task_2_wafer_trajectory_fault.json**

  - 场景: 故障（fault）
  - 模式: 静态模式
  - 内容: 模拟设备故障（模块不可用时间增加100秒，概率5%），记录晶圆调度轨迹。
  - 用途: 分析故障场景下调度算法的鲁棒性，检查是否能有效重新分配任务。
  - ```json
    {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
       ]
    }
    ```
- **task_2_wafer_trajectory_fault_adaptive.json**

  - 场景: 故障（fault）
  - 模式: 自适应模式
  - 内容: 同上，但采用自适应策略动态调整模块选择。
  - 用途: 评估自适应模式在故障场景下的性能优势。
  - ```json
    {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
        ]
    }
    ```
- **task_2_wafer_trajectory_time_variation.json**

  - 场景: 时间变化（time_variation）
  - 模式: 静态模式
  - 内容: 模拟加工时间变化（当前代码中时间变化逻辑未生效，实际与无干扰场景类似），记录调度轨迹。
  - 用途: 用于验证时间变化场景的调度效果（需进一步完善时间变化逻辑）。
  - ```json
    {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
        ]
    }
    ```
- **task_2_wafer_trajectory_time_variation_adaptive.json**

  - 场景: 时间变化（time_variation）
  - 模式: 自适应模式
  - 内容: 同上，自适应模式下的调度轨迹。
  - 用途: 对比静态模式，分析自适应策略在时间变化场景中的表现。
  - ```json
    {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
       ]
    }
    ```
- **task_2_wafer_trajectory_mixed.json**

  - 场景: 混合干扰（mixed）
  - 模式: 静态模式
  - 内容: 模拟故障和时间变化的组合干扰（当前以故障为主），记录调度轨迹。
  - 用途: 评估调度算法在复杂干扰环境下的稳定性。
  - ```json
    {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
        ]
    }
    ```
- **task_2_wafer_trajectory_mixed_adaptive.json**

  - 场景: 混合干扰（mixed）
  - 模式: 自适应模式
  - 内容: 同上，自适应模式下的调度轨迹。
  - 用途: 验证自适应模式在混合干扰场景中的鲁棒性与性能。
  - ```json
      {
        "MoveList": [
            {
                "StartTime": 0,
                "EndTime": 5,
                "MoveID": 0,
                "MoveType": 1,
                "ModuleName": "LP1",
                "MatID": "1.LP1",
                "SlotID": 1
            }
        ]
    }
    ```

### task2.py 生成的 JSON 文件结构

- 每个 JSON 文件包含一个 MoveList 数组，数组元素为动作记录。
- 字段包括:
  - StartTime: 动作开始时间（秒）
  - EndTime: 动作结束时间（秒）
  - MoveID: 动作唯一标识
  - MoveType: 动作类型（如1表示Pick，2表示Place，8表示加工等）
  - ModuleName: 模块名称（如LP1、TM1、PM7等）
  - MatID: 晶圆ID与步骤的组合（如“1.LP1”表示晶圆1在LP1步骤）
  - SlotID: 插槽编号（1或2）

### 生成的 PNG 图片文件

- **文件名**: task_2_makespan_mean.png

  - 内容: 各场景下三种调度模式的总完成时间均值（秒）。
  - 含义: 反映调度算法的效率。较低的 Makespan 表示更快的加工完成时间。
  - 观察: 静态模式在无干扰和时间变化场景中略优（约16208秒），自适应模式在故障和混合场景中表现较好（约9417-9550秒），但在无干扰场景中次优（16631秒）。
  - 用途: 比较不同模式在各场景下的调度效率，识别性能瓶颈。
    -![平均完工时间](README/task_2_makespan_mean.png)
- **文件名**: task_2_makespan_std.png

  - 内容: 各场景下三种调度模式的总完成时间标准差（秒）。
  - 含义: 反映调度结果的稳定性。标准差为0（如自适应模式在无干扰场景）可能表明逻辑问题（如固定选择）。
  - 观察: 自适应模式在无干扰和时间变化场景标准差为0，需检查模块选择缓存问题；基线和静态模式标准差较小，稳定性较高。
  - 用途: 评估调度算法的鲁棒性，识别潜在的逻辑错误。
  - ![完工时间标准差](README/task_2_makespan_std.png)
- **文件名**: task_2_conflicts_mean.png

  - 内容: 各场景下三种调度模式的冲突次数均值。
  - 含义: 反映资源竞争或互斥约束违反情况。当前实验中冲突次数为0，表明冲突检测逻辑未实现。
  - 观察: 所有场景和模式下冲突次数均为0，需完善冲突检测机制。
  - 用途: 用于未来优化冲突检测功能，验证调度方案的安全性。
  - ![平均冲突次数](README/task_2_conflicts_mean.png)
- **文件名**: task_2_load_balance_mean.png

  - 内容: 各场景下三种调度模式的负载均衡均值。
  - 含义: 反映模块和插槽的利用均衡性。较低的值表示更均匀的资源分配。
  - 观察: 自适应模式负载均衡最佳（约236-279），尤其在故障和混合场景中；静态模式较差（约300+）。
  - 用途: 评估资源分配的均匀性，优化模块选择策略。
  - ![平均负载均衡](README/task_2_load_balance_mean.png)
