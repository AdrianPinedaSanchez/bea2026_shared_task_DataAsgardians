"""Regenerates Figure 12 (test RMSE per L1, all systems) with the paper color
scheme: three highlighted models in shades of blue, the rest in grey.

Note on reproducibility: the RMSE values below are the recorded test results
of the completed GPU training runs (Google Colab, T4 and A100). This script
redraws the figure from those recorded values; it does not retrain any model.
Runs on CPU only. Output is written to figures/paper/.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines

OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "paper"

FULL_ORDER = [
    'xlm-roberta-base (baseline)',
    'xlm-roberta-large — Solo',
    'xlm-roberta-large — Hybrid',
    'XGBoost',
    'Ridge',
    'mDeBERTa-v3-base — Solo',
    'mDeBERTa-v3-base — Hybrid',
    'mDeBERTa-v3-base — Ensemble',
]

# ── Paper color scheme ──────────────────────────────────────────────────
# Only three models highlighted in shades of blue, the rest in grey.
GREY = '#b8b8b8'
BLUE_LIGHT = '#9ecae1'   # baseline
BLUE_MED   = '#3182bd'   # xlm-roberta-large Hybrid
BLUE_DARK  = '#08519c'   # mDeBERTa Ensemble

TEST_PALETTE_FULL = {
    'xlm-roberta-base (baseline)':     BLUE_LIGHT,
    'xlm-roberta-large — Solo':        GREY,
    'xlm-roberta-large — Hybrid':      BLUE_MED,
    'XGBoost':                         GREY,
    'Ridge':                           GREY,
    'mDeBERTa-v3-base — Solo':         GREY,
    'mDeBERTa-v3-base — Hybrid':       GREY,
    'mDeBERTa-v3-base — Ensemble':     BLUE_DARK,
}

FULL_ORDER_LABELS = {
    'xlm-roberta-base (baseline)':     'xlm-roberta-base\n(baseline)',
    'xlm-roberta-large — Solo':        'xlm-roberta-large\n— Solo',
    'xlm-roberta-large — Hybrid':      'xlm-roberta-large\n— Hybrid',
    'XGBoost':                         'XGBoost',
    'Ridge':                           'Ridge',
    'mDeBERTa-v3-base — Solo':         'mDeBERTa-v3-base\n— Solo',
    'mDeBERTa-v3-base — Hybrid':       'mDeBERTa-v3-base\n— Hybrid',
    'mDeBERTa-v3-base — Ensemble':     'mDeBERTa-v3-base\n— Ensemble',
}

L1_NAMES = {'es': 'Spanish (ES)', 'de': 'German (DE)', 'cn': 'Mandarin (CN)'}

# Recorded test RMSE values (Variant B) from the completed training runs
RMSE = {
    'ES': {
        'xlm-roberta-base (baseline)': 1.257,
        'xlm-roberta-large — Solo': 1.182,
        'xlm-roberta-large — Hybrid': 1.186,
        'XGBoost': 1.461,
        'Ridge': 1.505,
        'mDeBERTa-v3-base — Solo': 1.152,
        'mDeBERTa-v3-base — Hybrid': 1.180,
        'mDeBERTa-v3-base — Ensemble': 1.037,
    },
    'DE': {
        'xlm-roberta-base (baseline)': 1.258,
        'xlm-roberta-large — Solo': 1.177,
        'xlm-roberta-large — Hybrid': 1.117,
        'XGBoost': 1.351,
        'Ridge': 1.407,
        'mDeBERTa-v3-base — Solo': 1.141,
        'mDeBERTa-v3-base — Hybrid': 1.157,
        'mDeBERTa-v3-base — Ensemble': 0.997,
    },
    'CN': {
        'xlm-roberta-base (baseline)': 1.140,
        'xlm-roberta-large — Solo': 1.008,
        'xlm-roberta-large — Hybrid': 1.006,
        'XGBoost': 1.279,
        'Ridge': 1.293,
        'mDeBERTa-v3-base — Solo': 1.037,
        'mDeBERTa-v3-base — Hybrid': 1.112,
        'mDeBERTa-v3-base — Ensemble': 0.913,
    },
}
# Ref line = xlm-roberta-large Hybrid
XLMR_REF = {l1: RMSE[l1]['xlm-roberta-large — Hybrid'] for l1 in RMSE}

fig, axes = plt.subplots(1, 3, figsize=(42, 19))
plt.subplots_adjust(wspace=0.62, bottom=0.25)

for ax, l1 in zip(axes, ['ES', 'DE', 'CN']):
    sub = pd.DataFrame({'Model': FULL_ORDER,
                        'RMSE': [RMSE[l1][m] for m in FULL_ORDER]})
    colors = [TEST_PALETTE_FULL.get(m, '#333') for m in sub['Model']]
    bars = ax.barh(range(len(sub)), sub['RMSE'].values,
                   color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, sub['RMSE'].values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left',
                fontsize=30, fontweight='bold')

    ax.axvline(XLMR_REF[l1], color='#2e7d32', lw=3, ls='--', alpha=0.85)

    ax.set_title(L1_NAMES[l1.lower()], fontsize=34, fontweight='bold', pad=16)
    ax.set_xlabel('TEST RMSE (lower is better)', fontsize=28, fontweight='bold')
    ax.set_xlim(left=sub['RMSE'].min() * 0.80, right=sub['RMSE'].max() * 1.16)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([FULL_ORDER_LABELS[m] for m in sub['Model']],
                       fontsize=30, fontweight='bold', linespacing=0.85)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    ax.tick_params(axis='x', labelsize=20)

# ── Legend: the three highlighted models plus the grey group ────────────
legend_elements = [
    Patch(facecolor=BLUE_LIGHT, edgecolor='#333', linewidth=2,
          label='xlm-roberta-base (baseline)'),
    Patch(facecolor=BLUE_MED, edgecolor='#333', linewidth=2,
          label='xlm-roberta-large — Hybrid'),
    Patch(facecolor=BLUE_DARK, edgecolor='#333', linewidth=2,
          label='mDeBERTa-v3-base — Ensemble'),
    Patch(facecolor=GREY, edgecolor='#333', linewidth=2,
          label='Other systems'),
    mlines.Line2D([], [], color='#2e7d32', lw=4, ls='--',
                  label='Best Exp. 1 (xlm-roberta-large Hybrid)'),
]
leg = fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                 fontsize=33, frameon=True, bbox_to_anchor=(0.5, -0.01),
                 prop={'weight': 'bold', 'size': 33},
                 handlelength=3, handleheight=2.5,
                 borderpad=0.8, labelspacing=0.8, columnspacing=2.0)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor('#666')

plt.suptitle('TEST RMSE — all systems compared\n'
             'BEA 2026 Shared Task, Closed track',
             fontsize=34, y=0.98, fontweight='bold')

fig.savefig(OUT_DIR / 'fig12_test_rmse_per_l1_all_models.png',
            dpi=200, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig12_test_rmse_per_l1_all_models.pdf',
            bbox_inches='tight')
print('saved to', OUT_DIR)
