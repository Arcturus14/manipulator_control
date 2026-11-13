"""Exploratory search summary figure (not a research report)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    rows=json.loads((a.directory/'search_results.json').read_text())
    keys=list(dict.fromkeys((r['candidate'],r['amplitude']) for r in rows))
    table=[]
    for candidate,amplitude in keys:
        group=[r for r in rows if r['candidate']==candidate and r['amplitude']==amplitude]
        dyn=next(r for r in group if r['case'].startswith('directional'))
        high='static75' if dyn['case']=='directional25_75' else 'static5'
        names=['static25',dyn['case'],high];by={r['case']:r['metrics'] for r in group}
        if any(n not in by for n in names):continue
        manifest=json.loads((Path(group[0]['directory'])/'manifest.json').read_text())
        joint=manifest['arguments']['joint']+1
        table.append([f'{candidate}/J{joint}',group[0]['shape'],f'{amplitude:g}',
            ' / '.join(f'{by[n]["tracking_rmse_mm"]:.2f}' for n in names),
            ' / '.join(f'{by[n]["position_violation_max_rad"]:.4f}' for n in names),
            ' / '.join(f'{by[n]["test_joint_saturation_steps"]*.02:.2f}' for n in names),
            ' / '.join('hit' if by[n]['first_collision_index'] is not None else 'clear' for n in names),
            '7.5%' if high=='static75' else '5%'])
    fig,ax=plt.subplots(figsize=(17,max(3,1.5+.39*len(table))));ax.axis('off')
    t=ax.table(cellText=table,colLabels=['ID/joint','Shape','|d| [Nm]',
        'EE RMSE [mm]\nlow / dynamic / high','Max q violation [rad]\nlow / dynamic / high',
        'Test joint saturation [s]\nlow / dynamic / high','Mesh @20ms\nlow / dynamic / high','Upper cap'],
        cellLoc='center',loc='center',colWidths=[.055,.075,.07,.20,.20,.16,.17,.07])
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.8)
    ax.set_title('Exploratory pilot seed 807: noisy Post-DOB; lower margin 2.5%; no headroom\n'
        'Same reference and sustained disturbance within each row. Colliding episodes are not validated contact physics.',pad=18)
    fig.tight_layout();fig.savefig(a.directory/'search_summary.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
