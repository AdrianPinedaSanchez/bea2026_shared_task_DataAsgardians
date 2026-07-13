# Quick summary of results already computed
import matplotlib.pyplot as plt
import pandas as pd

# This is a helper to generate the comparison cell content
summary = """
# ==============================================================================
# INTERIM RESULTS SUMMARY (Phase 2)
# ==============================================================================
# Compile results from all strategies executed so far

import pandas as pd
import matplotlib.pyplot as plt

# Collect all results
all_results = []

# Strategy A: Transformer with text-serialized features
try:
    all_results.append(strat_a_results_df.assign(strategy="Strategy A (Transformer+Features)"))
    print("✓ Strategy A results loaded")
except NameError:
    print("⚠ Strategy A results not available")

# Strategy C Part 1: Global Gradient Boosting
try:
    gb_renamed = gb_results_df.copy()
    gb_renamed['strategy'] = gb_renamed['model'].apply(lambda x: f"Strategy C ({x})")
    all_results.append(gb_renamed[['strategy', 'L1', 'rmse', 'pearson']])
    print("✓ Strategy C Part 1 results loaded")
except NameError:
    print("⚠ Strategy C Part 1 results not available")

# Strategy C Part 2: Per-L1 XGBoost
try:
    per_l1_renamed = per_l1_results_df.copy()
    per_l1_renamed['strategy'] = per_l1_renamed['model'].apply(lambda x: f"Strategy C ({x})")
    all_results.append(per_l1_renamed[['strategy', 'L1', 'rmse', 'pearson']])
    print("✓ Strategy C Part 2 results loaded")
except NameError:
    print("⚠ Strategy C Part 2 results not available")

# Combine
if all_results:
    comparison_df = pd.concat(all_results, ignore_index=True)
    comparison_df = comparison_df.rename(columns={'strategy': 'Model'})
    
    print("\\n" + "="*80)
    print("  INTERIM COMPARISON — All Strategies So Far")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 2: Results Summary (RMSE & Pearson Correlation)", 
                 fontsize=14, fontweight="bold")
    
    # Plot 1: RMSE by Model and Language
    rmse_pivot = comparison_df.pivot_table(index='L1', columns='Model', values='rmse')
    rmse_pivot.plot(kind='bar', ax=axes[0], rot=0, width=0.8)
    axes[0].set_title("RMSE by Language (Lower is Better ↓)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xlabel("Language (L1)")
    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=1.206, color='red', linestyle='--', linewidth=2, label='Baseline ES', alpha=0.5)
    axes[0].axhline(y=1.149, color='orange', linestyle='--', linewidth=2, label='Baseline DE', alpha=0.5)
    axes[0].axhline(y=1.021, color='green', linestyle='--', linewidth=2, label='Baseline CN', alpha=0.5)
    
    # Plot 2: Pearson by Model and Language
    pearson_pivot = comparison_df.pivot_table(index='L1', columns='Model', values='pearson')
    pearson_pivot.plot(kind='bar', ax=axes[1], rot=0, width=0.8)
    axes[1].set_title("Pearson Correlation by Language (Higher is Better ↑)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Pearson Correlation")
    axes[1].set_xlabel("Language (L1)")
    axes[1].legend(fontsize=9, loc='lower right')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0.787, color='red', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].axhline(y=0.800, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    axes[1].axhline(y=0.804, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # --- Best model per metric ---
    print("\\n" + "="*80)
    print("  BEST MODELS (Per-Language Metrics)")
    print("="*80)
    
    for lang in ['es', 'de', 'cn']:
        lang_data = comparison_df[comparison_df['L1'] == lang]
        best_rmse_idx = lang_data['rmse'].idxmin()
        best_pearson_idx = lang_data['pearson'].idxmax()
        
        print(f"\\n  L1={lang.upper()}:")
        print(f"    Best RMSE:    {lang_data.loc[best_rmse_idx, 'Model']}")
        print(f"                  RMSE={lang_data.loc[best_rmse_idx, 'rmse']:.4f}")
        print(f"    Best Pearson: {lang_data.loc[best_pearson_idx, 'Model']}")
        print(f"                  Pearson={lang_data.loc[best_pearson_idx, 'pearson']:.4f}")
    
    # --- Save ---
    comparison_df.to_csv('results_interim_summary.csv', index=False)
    print(f"\\n✓ Summary saved to results_interim_summary.csv")
else:
    print("⚠ No results available yet")
"""

print(summary)
