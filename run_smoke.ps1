$ErrorActionPreference = "Stop"
$Python = "G:\Anaconda\envs\VLA\python.exe"
& $Python -m pip install -e . --no-deps
& $Python -m pytest -q
& $Python scripts/generate_data.py --output datasets/mujoco_language_smoke --episodes-per-task 10 --max-steps 100 --seed 7
& $Python scripts/create_split.py --data datasets/mujoco_language_smoke --seed 7
& $Python scripts/train.py --data datasets/mujoco_language_smoke --output runs/smoke_act --model act --epochs 1 --batch-size 4 --chunk-size 8 --device cpu
& $Python scripts/evaluate_split.py --checkpoint runs/smoke_act/best.pt --data datasets/mujoco_language_smoke --output outputs/mujoco_language_smoke --videos-per-task 1 --device cpu
