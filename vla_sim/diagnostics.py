"""Observable trajectory diagnostics; labels are heuristic, not causal proofs."""

import csv
import json
from pathlib import Path


def diagnose(task, rows, success):
    grasps = [r['step'] for r in rows if not r['attached_before'] and r['attached_after']]
    releases = [r for r in rows if r['attached_before'] and not r['attached_after']]
    near = [r for r in rows if r['grasp_xy_distance'] < 0.065 and r['grasp_z_distance'] < 0.07]
    labels = []
    if not success:
        if task == 'pick_place':
            if not grasps:
                labels.append('near_object_without_grasp' if near else 'never_reached_grasp_region')
            if any(r['distance_after'] >= 0.075 for r in releases):
                labels.append('released_outside_target')
            if rows[-1]['attached_after']:
                labels.append('still_holding_at_timeout')
        else:
            labels.append('push_target_not_reached')
        if len(rows) >= 10 and abs(rows[-1]['distance_after'] - rows[-10]['distance_after']) < 0.005:
            labels.append('distance_stalled_last_10_steps')
    return {'success': bool(success), 'observed_failure_labels': labels,
            'grasp_steps': grasps, 'release_steps': [r['step'] for r in releases],
            'grasp_region_steps': len(near),
            'open_commands_in_grasp_region': sum(r['action_gripper'] <= 0 for r in near),
            'minimum_target_distance': min(r['distance_after'] for r in rows),
            'note': 'Heuristic observations under the current simulator thresholds; not a causal diagnosis.'}


def save_trace(output: Path, stem: str, rows: list, diagnosis: dict):
    directory = output / 'traces'
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f'{stem}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / f'{stem}.json').write_text(json.dumps(diagnosis, indent=2), encoding='utf-8')
    # Portable standalone SVG plots, requiring no additional plotting package.
    panels = [('Target distance (m)', 'distance_after', 0.8),
              ('Gripper command (-1 open, +1 close)', 'action_gripper', 1.0),
              ('Attached (0/1)', 'attached_after', 1.0)]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600" viewBox="0 0 900 600">',
           '<rect width="900" height="600" fill="white"/>']
    for index, (title, key, scale) in enumerate(panels):
        top = 35 + index * 185
        svg.append(f'<text x="60" y="{top}" font-family="sans-serif" font-size="16">{title}</text>')
        svg.append(f'<path d="M60 {top+15} V{top+155} H850" fill="none" stroke="black"/>')
        points = []
        for i, row in enumerate(rows):
            value = float(row[key])
            normalized = (value + 1) / 2 if key == 'action_gripper' else value / scale
            points.append(f'{60+790*i/max(1,len(rows)-1):.2f},{top+155-140*normalized:.2f}')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#2563eb" stroke-width="2"/>')
        svg.append(f'<text x="60" y="{top+174}" font-family="sans-serif">step 1 to {len(rows)}</text>')
    svg.append('</svg>')
    (directory / f'{stem}.svg').write_text('\n'.join(svg), encoding='utf-8')
    return str(csv_path.resolve())
