"""Show paired safety/tracking outcomes without hiding seed-level tradeoffs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit', type=Path)
    a = p.parse_args()
    data = json.loads(a.audit.read_text())
    rows, full_success = [], []
    names = ['static25', 'directional25_5', 'static5']
    for seed in data['seeds']:
        m = {r['case']: r['metrics'] for r in data['per_seed'] if r['seed'] == seed}
        success = (m['static25']['position_violation_steps'] > 0
                   and m['directional25_5']['position_violation_steps'] == 0
                   and m['static5']['position_violation_steps'] == 0
                   and m['directional25_5']['tracking_rmse_mm'] < m['static5']['tracking_rmse_mm'])
        if success:
            full_success.append(seed)
        rows.append([seed, *[f'{m[c]["position_violation_max_rad"]:.4f}' for c in names],
                     *[f'{m[c]["tracking_rmse_mm"]:.2f}' for c in names], 'Yes' if success else 'No'])
    fig, ax = plt.subplots(figsize=(15, 2.2 + len(rows)*.4)); ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=['Seed', 'Max viol.\nStatic 2.5% [rad]',
        'Max viol.\nDynamic [rad]', 'Max viol.\nStatic 5% [rad]',
        'RMSE\nStatic 2.5% [mm]', 'RMSE\nDynamic [mm]', 'RMSE\nStatic 5% [mm]',
        'Full relation\nsame seed?'], loc='center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.3)
    ax.set_title('Paired-seed check (not selected examples)\n'
                 'Full relation: low static violates; dynamic/high static safe; dynamic RMSE < high static', pad=15)
    fig.savefig(a.audit.parent / 'per_seed_table.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    result = {'source': str(a.audit), 'seed_count': len(data['seeds']),
              'full_relation_satisfied_seeds': full_success,
              'definition': 'Low static violates AND dynamic/high static safe AND dynamic RMSE < high static in SAME seed',
              'all_seeds_satisfy_full_relation': len(full_success) == len(data['seeds'])}
    (a.audit.parent / 'paired_relation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
