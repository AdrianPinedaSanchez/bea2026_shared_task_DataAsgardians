"""Regenerates eda_target_distribution and eda_feature_correlations with the
blue/grey palette used in the paper. Reproduces the full set of 33 linguistic
features (add_all_features) from notebooks/01_eda_complete.ipynb.

Requires the official shared task data: clone
https://github.com/britishcouncil/bea2026st into the repository root so that
bea2026st/data/train/{es,de,cn}/ is available. Runs on CPU only.
Output figures are written to figures/paper/.
"""
import math, re, warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import nltk
from nltk.corpus import cmudict, wordnet as wn

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk", palette="Set2")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100

for pkg in ["wordnet", "omw-1.4", "cmudict"]:
    nltk.download(pkg, quiet=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = REPO_ROOT / "bea2026st"
DATA_DIR = BASE / "data"
OUT_DIR = REPO_ROOT / "figures" / "paper"

L1S = ["es", "de", "cn"]
L1_LABELS = {"es": "Spanish", "de": "German", "cn": "Mandarin"}
TARGET = "GLMM_score"

# ── Paper palette ───────────────────────────────────────────────────────
# Distribution: shades of blue (light to dark)
L1_COLORS = {"es": "#9ecae1", "de": "#4292c6", "cn": "#08519c"}
# Correlation: Global in grey (neutral aggregate), L1s in a blue gradient
CORR_STYLE = [
    ("pearson_global", "#737373", "Global"),
    ("pearson_es",     "#6baed6", "ES"),
    ("pearson_de",     "#3182bd", "DE"),
    ("pearson_cn",     "#08519c", "CN"),
]

FEATURE_COLS = [
    "clue_ratio", "polysemy_pos", "is_homonym", "n_consonant_clusters",
    "r_count", "has_r", "silent_letters", "difficult_phonemes",
    "word_frequency", "polysemy_all", "homonym_pos_count",
    "max_cluster_length", "spelling_phoneme_ratio", "has_unfamiliar",
    "cosine_dist_l1_en", "morphological_complexity", "final_cluster_size",
    "primary_stress_pos", "has_diphthong", "min_sense_depth",
    "word_family_size", "context_char_length", "source_word_length",
    "source_word_count", "shared_prefix_len",
    "pos_adjective", "pos_adverb", "pos_determiner", "pos_misc",
    "pos_noun", "pos_number", "pos_preposition", "pos_verb",
]

# ── Raw data loading ────────────────────────────────────────────────────
def load_split(split):
    frames = []
    for l1 in L1S:
        path = DATA_DIR / split / l1 / f"kvl_shared_task_{l1}_{split}.csv"
        if path.exists():
            df = pd.read_csv(path); df["split"] = split; frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

train = load_split("train")

# ========================================================================
# FIGURE 1: eda_target_distribution
# ========================================================================
fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

for i, l1 in enumerate(L1S):
    ax = fig.add_subplot(gs[0, i])
    subset = train[train["L1"] == l1]
    ax.hist(subset[TARGET], bins=50, color=L1_COLORS[l1], edgecolor="white", alpha=0.85)
    ax.axvline(subset[TARGET].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={subset[TARGET].mean():.2f}")
    ax.axvline(subset[TARGET].median(), color="red", linestyle=":", linewidth=1.5,
               label=f"median={subset[TARGET].median():.2f}")
    ax.set_title(f"{L1_LABELS[l1]}", fontsize=15, fontweight="bold")
    ax.set_xlabel(TARGET, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)

ax_kde = fig.add_subplot(gs[1, 0:2])
for l1 in L1S:
    subset = train[train["L1"] == l1]
    subset[TARGET].plot.kde(ax=ax_kde, label=L1_LABELS[l1], color=L1_COLORS[l1], linewidth=2.5)
ax_kde.set_title("GLMM_score density by L1", fontsize=15, fontweight="bold")
ax_kde.set_xlabel(TARGET, fontsize=13)
ax_kde.set_ylabel("Density", fontsize=13)
ax_kde.tick_params(labelsize=11)
ax_kde.legend(fontsize=11)

ax_box = fig.add_subplot(gs[1, 2])
colors = [L1_COLORS[l1] for l1 in L1S]
bp = ax_box.boxplot([train[train["L1"] == l1][TARGET].values for l1 in L1S],
                    tick_labels=[L1_LABELS[l1] for l1 in L1S], patch_artist=True, widths=0.6)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax_box.set_title("Box plot by L1", fontsize=15, fontweight="bold")
ax_box.set_ylabel(TARGET, fontsize=13)
ax_box.tick_params(labelsize=11)

plt.suptitle("Target Distribution (GLMM_score)", fontsize=17, fontweight="bold", y=1.01)
plt.savefig(OUT_DIR / "eda_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved eda_target_distribution.png")

# ========================================================================
# FIGURE 2: eda_feature_correlations, 33 features (add_all_features)
# ========================================================================
CMU_DICT = cmudict.dict()
POS_MAP = {"noun": wn.NOUN, "verb": wn.VERB, "adjective": wn.ADJ, "adverb": wn.ADV}
L1_ABSENT_PHONEMES = {
    "es": {"TH", "DH", "SH", "ZH", "V", "Z"},
    "de": {"TH", "DH"},
    "cn": {"TH", "DH", "SH", "ZH", "R", "V", "Z", "NG"},
}
L1_UNFAMILIAR_LETTERS = {"es": set("wk"), "de": set(), "cn": set()}
PREFIXES = ["un","re","in","im","ir","il","dis","en","em","non","pre","mis","over","under","out","sub","super"]
SUFFIXES = ["ing","tion","sion","ment","ness","able","ible","ful","less","ous","ive","al","ly","er","est","ed","es","s"]
DIPHTHONGS = {"AY", "OY", "AW", "EY", "OW", "EH", "IY", "UW"}
POS_CATEGORIES = ["adjective", "adverb", "determiner", "misc", "noun", "number", "preposition", "verb"]

@lru_cache(maxsize=None)
def get_pronunciation(word): return CMU_DICT.get(str(word).lower(), [[]])[0]
@lru_cache(maxsize=None)
def get_synsets(word): return wn.synsets(str(word).lower(), lang="eng")

def polysemy_pos(word, pos):
    wn_pos = POS_MAP.get(str(pos).lower())
    return len(wn.synsets(str(word).lower(), pos=wn_pos, lang="eng")) if wn_pos else len(get_synsets(word))
def is_homonym(word): return int(len({s.pos() for s in get_synsets(word)}) > 1)
def consonant_cluster_count(word): return len(re.findall(r"[bcdfghjklmnpqrstvwxyz]{2,}", str(word).lower()))
def r_count(word):
    pron = get_pronunciation(word)
    return sum(1 for p in pron if p.startswith("R")) if pron else str(word).lower().count("r")
def has_r(word): return int("r" in str(word).lower())
def phoneme_count(word):
    pron = get_pronunciation(word)
    if pron: return len(pron)
    return max(len(re.findall(r"[aeiouy]+", str(word).lower())) * 2, 1)
def silent_letter_count(word): return max(len(str(word)) - phoneme_count(word), 0)
def difficult_phoneme_count(word, l1):
    pron = get_pronunciation(word); absent = L1_ABSENT_PHONEMES.get(l1, set())
    if not pron or not absent: return 0
    return sum(1 for p in [re.sub(r"\d", "", ph) for ph in pron] if p in absent)
def wordnet_frequency(word):
    total = 0
    for syn in get_synsets(word):
        for lem in syn.lemmas():
            if lem.name().lower() == str(word).lower(): total += lem.count()
    return math.log1p(total)
def polysemy_all(word): return len(get_synsets(word))
def homonym_pos_count(word): return len({s.pos() for s in get_synsets(word)})
def max_cluster_length(word):
    clusters = re.findall(r"[bcdfghjklmnpqrstvwxyz]{2,}", str(word).lower())
    return max((len(c) for c in clusters), default=0)
def spelling_phoneme_ratio(word): return len(str(word)) / max(phoneme_count(word), 1)
def unfamiliar_letter_count(word, l1):
    return sum(1 for ch in str(word).lower() if ch in L1_UNFAMILIAR_LETTERS.get(l1, set()))
def has_unfamiliar(word, l1): return int(unfamiliar_letter_count(word, l1) > 0)
def char_ngram_distance(word1, word2, n=2):
    def ngrams(w):
        w = str(w).lower(); return set(w[i:i+n] for i in range(len(w)-n+1)) if len(w) >= n else {w}
    s1, s2 = ngrams(word1), ngrams(word2); union = len(s1 | s2)
    return 1.0 - (len(s1 & s2) / union) if union > 0 else 1.0
def morphological_complexity(word):
    w = str(word).lower(); count = 0
    for p in PREFIXES:
        if w.startswith(p) and len(w) > len(p) + 2: count += 1; break
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if w.endswith(s) and len(w) > len(s) + 2: count += 1; break
    return count
def final_cluster_size(word):
    m = re.search(r"[bcdfghjklmnpqrstvwxyz]+$", str(word).lower())
    return len(m.group()) if m else 0
def compute_primary_stress_pos(word):
    pron = get_pronunciation(word)
    if not pron: return -1
    for i, ph in enumerate(pron):
        if ph.endswith("1"): return i
    return -1
def compute_has_diphthong(word):
    pron = get_pronunciation(word)
    if not pron: return 0
    return int(any(re.sub(r"\d", "", ph) in DIPHTHONGS for ph in pron))
def compute_min_sense_depth(word):
    syns = get_synsets(word)
    if not syns: return 0
    return min(s.min_depth() for s in syns)
def compute_word_family_size(word):
    syns = get_synsets(word); related = set()
    for syn in syns:
        for lem in syn.lemmas():
            if lem.name().lower() == str(word).lower():
                for dr in lem.derivationally_related_forms():
                    related.add(dr.name().lower())
    return len(related)
def shared_prefix_length(w1, w2):
    w1, w2 = str(w1).lower(), str(w2).lower(); n = min(len(w1), len(w2))
    for i in range(n):
        if w1[i] != w2[i]: return i
    return n

def add_all_features(df):
    d = df.copy(); w = "en_target_word"
    d["n_consonant_clusters"]  = d[w].apply(consonant_cluster_count)
    d["max_cluster_length"]    = d[w].apply(max_cluster_length)
    d["final_cluster_size"]    = d[w].apply(final_cluster_size)
    d["r_count"]               = d[w].apply(r_count)
    d["has_r"]                 = d[w].apply(has_r)
    d["silent_letters"]        = d[w].apply(silent_letter_count)
    d["spelling_phoneme_ratio"]= d[w].apply(spelling_phoneme_ratio)
    d["difficult_phonemes"]    = d.apply(lambda r: difficult_phoneme_count(r[w], r["L1"]), axis=1)
    d["has_unfamiliar"]        = d.apply(lambda r: has_unfamiliar(r[w], r["L1"]), axis=1)
    d["has_diphthong"]         = d[w].apply(compute_has_diphthong)
    d["primary_stress_pos"]    = d[w].apply(compute_primary_stress_pos)
    d["polysemy_pos"]          = d.apply(lambda r: polysemy_pos(r[w], r["en_target_pos"]), axis=1)
    d["polysemy_all"]          = d[w].apply(polysemy_all)
    d["is_homonym"]            = d[w].apply(is_homonym)
    d["homonym_pos_count"]     = d[w].apply(homonym_pos_count)
    d["min_sense_depth"]       = d[w].apply(compute_min_sense_depth)
    d["morphological_complexity"] = d[w].apply(morphological_complexity)
    d["word_family_size"]      = d[w].apply(compute_word_family_size)
    d["word_frequency"]        = d[w].apply(wordnet_frequency)
    d["cosine_dist_l1_en"]     = d.apply(lambda r: char_ngram_distance(r["L1_source_word"], r[w]), axis=1)
    d["clue_ratio"]            = (d["en_target_clue"].astype(str).str.count("_") /
                                  d[w].astype(str).str.len().clip(lower=1))
    d["source_word_length"]    = d["L1_source_word"].astype(str).str.len()
    d["source_word_count"]     = d["L1_source_word"].astype(str).str.split().str.len()
    d["context_char_length"]   = d["L1_context"].astype(str).str.len()
    d["shared_prefix_len"]     = d.apply(lambda r: shared_prefix_length(r["L1_source_word"], r[w]), axis=1)
    for pos_cat in POS_CATEGORIES:
        d[f"pos_{pos_cat}"] = (d["en_target_pos"].str.lower() == pos_cat).astype(int)
    return d

print("computing 33 features...")
train_feat = add_all_features(train)

def full_correlation_table(df, features, target):
    rows = []
    for feat in features:
        row = {"feature": feat}
        valid = df[[feat, target]].dropna()
        if len(valid) > 2 and valid[feat].std() > 0:
            row["pearson_global"] = pearsonr(valid[feat], valid[target])[0]
            row["spearman_global"] = spearmanr(valid[feat], valid[target])[0]
        for l1 in L1S:
            sub = df[df["L1"] == l1][[feat, target]].dropna()
            if len(sub) > 2 and sub[feat].std() > 0:
                row[f"pearson_{l1}"] = pearsonr(sub[feat], sub[target])[0]
                row[f"spearman_{l1}"] = spearmanr(sub[feat], sub[target])[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")

corr_df = full_correlation_table(train_feat, FEATURE_COLS, TARGET)
corr_df["abs_pearson"] = corr_df["pearson_global"].abs()
corr_df = corr_df.sort_values("abs_pearson", ascending=False)

plot_corr = corr_df[["pearson_global","pearson_es","pearson_de","pearson_cn"]].sort_values("pearson_global")

fig, ax = plt.subplots(figsize=(22, max(len(FEATURE_COLS) * 0.85, 14)))
x = np.arange(len(plot_corr))
bar_h = 0.2
for i, (col, color, label) in enumerate(CORR_STYLE):
    vals = plot_corr[col].fillna(0)
    ax.barh(x + i * bar_h, vals, height=bar_h, label=label, color=color, alpha=0.9)
ax.set_yticks(x + bar_h * 1.5)
ax.set_yticklabels(plot_corr.index, fontsize=24)
ax.set_xlabel("Pearson r with GLMM_score", fontsize=26)
ax.set_title("Feature correlation with lexical difficulty (by L1)", fontsize=30, fontweight="bold")
ax.legend(loc="lower right", fontsize=22)
ax.tick_params(axis="x", labelsize=22)
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_DIR / "eda_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved eda_feature_correlations.png")
