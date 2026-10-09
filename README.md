# Language-Conditioned Manipulation in MuJoCo

T1000 4GB桌面操作仿真项目：RGB、9维状态与语言输入，BC或动作分块Transformer输出动作。包含Push与Pick-place。

环境采用运动学Cartesian夹爪、确定性抓取约束和平面推动近似。ACT实现没有原论文中的CVAE组件；闭环每步重新预测并执行动作分块的第一步。

## 环境

在VS Code终端进入项目目录并激活VLA：

```text
conda activate VLA
python -m pip install -e .
```

## 当前运行流程

v2数据已补齐释放动作，ACT训练及Test评估已完成。当前Pick-place成功判定要求物体到达目标、降低高度并释放。旧实验输出已移到 `archive/results_20261009/`，数据与权重保留以供复现。

以下命令均为单行，依次运行：

```text
python scripts/generate_data.py --output datasets/mujoco_language_v2 --episodes-per-task 100 --max-steps 100 --seed 42
python scripts/create_split.py --data datasets/mujoco_language_v2 --seed 42
python scripts/train.py --data datasets/mujoco_language_v2 --output runs/act_v2_seed42 --model act --epochs 40 --batch-size 4 --chunk-size 16 --device cuda --seed 42
python scripts/evaluate_split.py --checkpoint runs/act_v2_seed42/best.pt --data datasets/mujoco_language_v2 --split test --output outputs/act_v2_seed42_test --videos-per-task 3 --device cuda
```

200条轨迹按任务分层划分为160条Train、20条Validation、20条Test。Test不参与训练或选模。训练前检查专家成功率和最后的释放动作。

BC使用同一划分：

```text
python scripts/train.py --data datasets/mujoco_language_v2 --output runs/bc_v2_seed42 --model bc --epochs 40 --batch-size 8 --chunk-size 1 --device cuda --seed 42
python scripts/evaluate_split.py --checkpoint runs/bc_v2_seed42/best.pt --data datasets/mujoco_language_v2 --split test --output outputs/bc_v2_seed42_test --videos-per-task 3 --device cuda
```

结果包含evaluation.json、每个任务的成功视频及所有失败视频。当前距离阈值0.075米，Pick-place还要求物体高度低于0.075米且attached=False。

## 纠正演示与分类版BC

分类版BC对XYZ使用回归损失，对夹爪使用二分类损失；类别权重只由Train计算。训练日志记录XYZ损失、分类损失、闭合召回率与错误闭合率。

纠正数据已生成到 `datasets/mujoco_language_v3_corrections`：在v2基础上增加200条随机夹爪起点的专家演示，仅加入Train。Train共360条，Validation/Test各20条，与v2保持相同编号及文件内容。不要对该目录重新运行create_split.py。

```text
python scripts/train.py --data datasets/mujoco_language_v3_corrections --output runs/bc_gripper_cls_corrections_seed42 --model bc_gripper_cls --epochs 80 --batch-size 8 --chunk-size 1 --gripper-loss-weight 1.0 --device cuda --seed 42
python scripts/evaluate_split.py --checkpoint runs/bc_gripper_cls_corrections_seed42/best.pt --data datasets/mujoco_language_v3_corrections --split val --output outputs/bc_gripper_cls_corrections_seed42_val --videos-per-task 3 --device cuda
python scripts/evaluate_split.py --checkpoint runs/bc_gripper_cls_corrections_seed42/best.pt --data datasets/mujoco_language_v3_corrections --split test --output outputs/bc_gripper_cls_corrections_seed42_test --videos-per-task 3 --device cuda
```

截至本次更新，已确认的结果如下：

| 模型 | Train轨迹 | 评估集合 | Push | Pick-place | 总成功率 |
|---|---:|---|---:|---:|---:|
| ACT v2，40轮 | 160 | Test，20条 | 10/10 | 10/10 | 100% |
| 分类BC + 纠正演示，80轮 | 360 | Validation，20条 | 10/10 | 10/10 | 100% |
| 分类BC + 纠正演示，80轮 | 360 | Test，20条 | 10/10 | 8/10 | 90% |

纠正版BC的Test结果保存在 `outputs/bc_gripper_cls_corrections_seed42_test/evaluation.json`，平均推理约3.61ms；8条成功Pick-place均释放物体，2条失败为episode113与165。ACT与纠正版BC采用不同训练数据量，不能据此作同等数据条件下的模型优劣判断。增加纠正数据也增加了每轮更新次数，尚未区分数量和状态覆盖的贡献。结果仅适用于当前简化仿真和这些保留场景。Test结果用于最终报告，后续调参应使用Validation或独立开发场景。

evaluate_split.py同时保存traces目录：CSV逐步状态/动作、JSON抓取释放事件与启发式失败标签、SVG距离/夹爪/抓取状态曲线。标签描述观察到的现象，不证明因果。

## 实时查看

```text
python scripts/view_env.py --mode expert --task pick_place
python scripts/view_env.py --mode policy --task pick_place --checkpoint runs/act_v2_seed42/best.pt --device cuda
python scripts/view_env.py --mode policy --task pick_place --checkpoint runs/bc_gripper_cls_corrections_seed42/best.pt --device cuda
```

模型命令需在新训练完成后运行。Space暂停/继续，R重置，T切换任务；鼠标旋转、平移、缩放。

## 代码与数据

vla_sim包含env（环境）、expert（专家）、data（数据与划分）、models（模型）、losses（损失与指标）、diagnostics（轨迹诊断）、checkpoint（保存加载）、inference（共用推理）。scripts包含generate_data、generate_corrections、create_split、train、evaluate_split、view_env六个入口。generate_corrections.py要求输出目录不存在，以避免覆盖已有实验。

每个episode_xxxxx.npz包含images、states、actions、tokens、task、instruction、success，是自定义NPZ格式，尚未导出标准LeRobot格式。状态为夹爪XYZ、物体XYZ、目标XY、attached；动作为delta_x、delta_y、delta_z、夹爪开合。

## 验证

## 当前视觉控制结果

4维机器人状态ACT不接收物体/目标坐标。global与spatial使用同样160条Train、40轮、seed42；spatial多了空间投影参数。Validation：global80%，spatial95%；20条Test：global80%，spatial100%。

以下位置泛化结果每组100条轨迹，采用同一批场景：

| 场景 | global Push | global Pick-place | global总计 | spatial Push | spatial Pick-place | spatial总计 |
|---|---:|---:|---:|---:|---:|---:|
| ID | 68% | 62% | 65% | 94% | 100% | 97% |
| Boundary | 42% | 44% | 43% | 94% | 86% | 90% |
| OOD | 0% | 0% | 0% | 0% | 2% | 1% |

结果目录 `outputs/visual_act_position_comparison_seed90001` 的act标签代表global ACT，bc标签代表spatial ACT，实际两者均为ACT。空间特征改善ID与边缘表现，但范围外视觉定位尚未解决。不能把早期完整9维坐标模型的OOD成绩当作视觉泛化结果。

空间版视觉消融：正常图像95%，全黑0%，无关静态图像15%（Validation各20条）。这支持模型依赖正确图像，不表示它能适应任意图像扰动。单种子结果需要后续重复验证。

旧产物已移入archive/cleanup_20261009与archive/results_20261009，可恢复。当前数据主线是mujoco_language_v2和mujoco_language_v3_corrections。后续优先检查OOD场景中的可见性，再使用独立开发场景研究覆盖和视觉增强；已有OOD评估不用于反复选模。

空间视觉编码器对照：spatial将CNN特征保留为4×4网格再投影，global为原来的全局平均池化。spatial增加了投影层参数，性能差异可能同时受到空间信息和模型容量影响。checkpoint记录配置，旧权重默认global。

```text
python scripts/train.py --data datasets/mujoco_language_v2 --output runs/act_robot_spatial_seed42 --model act --state-mode robot --visual-mode spatial --epochs 40 --batch-size 4 --chunk-size 16 --device cuda --seed 42
python scripts/evaluate_split.py --checkpoint runs/act_robot_spatial_seed42/best.pt --data datasets/mujoco_language_v2 --split val --output outputs/act_robot_spatial_seed42_val --videos-per-task 3 --device cuda
```

先比较Validation与原robot/global模型的80%成功率，不根据Test调参。

机器人状态消融：`--state-mode robot`只提供夹爪XYZ及attached（4维），不提供物体与目标坐标。环境仍保留完整状态用于仿真和诊断，数据加载和推理接口负责屏蔽特权坐标。checkpoint记录模式，旧checkpoint默认full（9维）。

```text
python scripts/train.py --data datasets/mujoco_language_v2 --output runs/act_robot_state_seed42 --model act --state-mode robot --epochs 40 --batch-size 4 --chunk-size 16 --device cuda --seed 42
python scripts/evaluate_split.py --checkpoint runs/act_robot_state_seed42/best.pt --data datasets/mujoco_language_v2 --split val --output outputs/act_robot_state_seed42_val --videos-per-task 3 --device cuda
```

与ACT v2保持相同数据、划分和40轮设置，单独改变状态可见性。先分析Validation闭环表现，再开展图像消融及最终Test。

视觉消融（无需训练，默认使用20条Validation场景）：

```text
python scripts/evaluate_visual_ablation.py --device cuda
```

normal为原图，black为全黑输入，shuffle为从Train随机帧库抽取并在每条episode内固定的无关图像。两模型使用相同场景、替换图像和种子。替换仅影响策略输入，视频仍显示真实场景。输出到outputs/visual_ablation_val_seed17001，包括comparison.json、shuffle_bank.npz及bank_sources.json，记录图像来源。该实验检验冻结模型对图像的敏感性；黑图是分布外输入，不能单凭掉点证明视觉定位能力。不重新训练也不能证明robot-state-only设置的效果。

位置泛化评估（冻结模型；两模型共享场景，每组50条Push和50条Pick-place）：

```text
python scripts/evaluate_generalization.py --episodes-per-task 50 --seed 90001 --device cuda
```

ID采用训练位置范围内的新随机场景；Boundary位于训练范围内边缘；OOD的物体和目标Y坐标超出训练范围，但仍在夹爪工作空间内。语言与视觉外观保持原设置。scenarios.json记录每个场景，comparison.json汇总三组结果。scenarios目录是评估初始状态文件，不是专家演示，不可用于训练。两模型训练数据量不同，比较需注明。每次使用新输出目录，避免覆盖。该测试不表示真机或完整接触动力学性能。

```text
python -m pytest -q
```

run_smoke.ps1为可选PowerShell流程，生成20条轨迹、划分、训练1轮并评估，用于验证接口。

后续实验：语言消融、删除物体与目标特权坐标、位置与视觉泛化、多随机种子及完整Panda机械臂。
