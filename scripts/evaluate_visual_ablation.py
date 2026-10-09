"""Compare frozen policies with normal, black and unrelated fixed images."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='datasets/mujoco_language_v2')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--act-checkpoint', default='runs/act_v2_seed42/best.pt')
    parser.add_argument('--bc-checkpoint', default='runs/bc_gripper_cls_corrections_seed42/best.pt')
    parser.add_argument('--output', default='outputs/visual_ablation_val_seed17001')
    parser.add_argument('--seed', type=int, default=17001)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--videos-per-task', type=int, default=1)
    args = parser.parse_args()
    root, output = Path(args.data), Path(args.output)
    if output.exists():
        parser.error('Choose a fresh output directory')
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    for checkpoint in [args.act_checkpoint, args.bc_checkpoint]:
        if not Path(checkpoint).is_file():
            parser.error(f'Missing checkpoint: {checkpoint}')
    # Bank sampled only from training frames. Random time produces visual
    # contradictions, not just a black-image distribution shift.
    rng = np.random.default_rng(args.seed)
    images, sources = [], []
    for episode_id in sorted(split['train']):
        with np.load(root / f'episode_{episode_id:05d}.npz', allow_pickle=False) as ep:
            frame = int(rng.integers(len(ep['images'])))
            images.append(ep['images'][frame].copy())
            sources.append({'episode_id': episode_id, 'frame': frame})
    if not images:
        parser.error('Training image bank is empty')
    output.mkdir(parents=True)
    bank = output / 'shuffle_bank.npz'
    np.savez_compressed(bank, images=np.stack(images))
    (output / 'bank_sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    results = {}
    for mode in ['normal', 'black', 'shuffle']:
        results[mode] = {}
        for label, checkpoint in [('act', args.act_checkpoint), ('bc', args.bc_checkpoint)]:
            destination = output / mode / label
            subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_split.py')),
                '--checkpoint', checkpoint, '--data', str(root), '--split', args.split,
                '--output', str(destination), '--image-mode', mode, '--image-seed', str(args.seed),
                '--shuffle-images', str(bank), '--device', args.device,
                '--videos-per-task', str(args.videos_per_task)], check=True)
            report = json.loads((destination / 'evaluation.json').read_text(encoding='utf-8'))
            results[mode][label] = report['summary']
        (output / 'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('\nMode       Model    Success')
    for mode, models in results.items():
        for label, summary in models.items():
            print(f"{mode:10s} {label:7s} {summary['success_rate']:.1%}")
    print(f'Results: {output.resolve()}')


if __name__ == '__main__':
    main()
