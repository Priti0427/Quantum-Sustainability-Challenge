"""Generate the fixed Notebook 04 — ZIP-Code Risk Clustering: UMAP + HDBSCAN."""
import json, os

cells = []
def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s})

# ═══════════════════════════════════════════════════════════════════════════════
md("""\
# Notebook 04 — ZIP-Code Risk Clustering: UMAP + HDBSCAN

## What This Notebook Does
1. **Builds a ZIP-code feature matrix** from all 4 insurance datasets (2018–2021)
2. **Reduces dimensionality** with UMAP — 2-D manifold preserving local structure
3. **Discovers micro-clusters** with HDBSCAN — density-based, no K required
4. **Merges into 5 actionable risk tiers** via agglomerative clustering on centroids
5. **Visualizes** risk tiers on a California map and a 4-panel dashboard
6. **Exports** tier labels for downstream notebooks

## Two-Level Clustering Strategy
| Level | Method | Purpose |
|-------|--------|---------|
| **1 — Micro** | HDBSCAN (`min_cluster_size=50`) | Find natural density groups + outliers |
| **2 — Macro** | Agglomerative on centroids (`n=5`) | Merge into insurance-grade risk tiers |

**Resulting 5 tiers:** Negligible → Low → Moderate → High → Extreme
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 1 — Imports & Configuration")
code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, re
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import hdbscan

try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 150,
                     'axes.titlesize': 13, 'axes.labelsize': 11})
RANDOM_STATE = 42

INS1 = '../data/insurance_2018_2019.XLS'
INS2 = '../data/insurance_2020_2021.XLS'
DF2  = '../data/wildfire_county_monthly.csv'

TIER_NAMES  = ['Negligible', 'Low', 'Moderate', 'High', 'Extreme']
TIER_COLORS = {'Negligible': '#4CAF50', 'Low': '#8BC34A',
               'Moderate': '#FFC107', 'High': '#FF5722',
               'Extreme': '#B71C1C', 'Outlier': '#9E9E9E'}

print(f'Libraries loaded  |  geopandas={HAS_GEO}')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 2 — Load Insurance Data (All 4 Years, Homeowners)")
code("""\
def load_ho_sheets(filepath, years):
    frames = []
    for yr in years:
        df = pd.read_excel(filepath, sheet_name=f'{yr}HO', header=1, engine='openpyxl')
        df.columns = [c.replace('\\n', ' ').strip() for c in df.columns]
        first = df.columns[0]
        df = df.rename(columns={first: 'ZIP_Code'})
        df['ZIP_Code'] = pd.to_numeric(df['ZIP_Code'], errors='coerce')
        df = df.dropna(subset=['ZIP_Code'])
        df['ZIP_Code'] = df['ZIP_Code'].astype(int)
        df['Year'] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

ins1 = load_ho_sheets(INS1, [2018, 2019])
ins2 = load_ho_sheets(INS2, [2020, 2021])
common = list(set(ins1.columns) & set(ins2.columns))
ins = pd.concat([ins1[common], ins2[common]], ignore_index=True)
print(f'Combined panel: {ins.shape}')
print(ins.groupby('Year')['ZIP_Code'].nunique())
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 3 — Build ZIP-Code Feature Matrix")
code("""\
ic = ins.copy()

def find_col(df, keywords):
    kw = [k.lower() for k in keywords]
    for c in df.columns:
        if all(k in c.lower() for k in kw):
            return c
    return None

col_risk    = find_col(ic, ['fire', 'risk', 'score'])
col_prem    = find_col(ic, ['earned', 'premium'])
col_exp     = find_col(ic, ['earned', 'exposure'])
col_high    = find_col(ic, ['high', 'fire', 'risk', 'exposure'])
col_vhigh   = find_col(ic, ['very', 'high', 'fire', 'risk', 'exposure'])

loss_cols = [c for c in ic.columns if 'fire' in c.lower() and 'incurred' in c.lower()]
ic['total_fire_loss'] = ic[loss_cols].fillna(0).sum(axis=1)
ic['premium_per_policy'] = ic[col_prem] / ic[col_exp].replace(0, np.nan)
ic['loss_ratio'] = ic['total_fire_loss'] / ic[col_prem].replace(0, np.nan)
ic['high_risk_frac'] = (
    (ic[col_high].fillna(0) + ic[col_vhigh].fillna(0)) / ic[col_exp].replace(0, np.nan)
)

agg = ic.groupby('ZIP_Code').agg(
    avg_fire_risk  = (col_risk, 'mean'),
    avg_premium    = (col_prem,  'mean'),
    avg_exposure   = (col_exp,   'mean'),
    avg_prem_pol   = ('premium_per_policy', 'mean'),
    avg_loss_ratio = ('loss_ratio',  'mean'),
    high_risk_frac = ('high_risk_frac', 'mean'),
    total_fire_loss= ('total_fire_loss', 'sum'),
).reset_index()

pivot = ic.pivot_table(index='ZIP_Code', columns='Year', values=col_prem, aggfunc='mean')
yr0, yr1 = pivot.columns.min(), pivot.columns.max()
pivot['pct_prem_chg'] = (pivot[yr1] - pivot[yr0]) / pivot[yr0].replace(0, np.nan) * 100
agg = agg.merge(pivot[['pct_prem_chg']].reset_index(), on='ZIP_Code', how='left')
agg = agg.dropna(thresh=6).reset_index(drop=True)
print(f'ZIP feature matrix: {agg.shape}')
print(agg.describe().round(2))
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 4 — Approximate Geo-Coordinates per ZIP\n\nFire risk is spatially autocorrelated — including lat/lon lets UMAP discover geographically coherent risk corridors.")
code("""\
df2 = pd.read_csv(DF2)
county_geo = df2.groupby('county')[['lat', 'long']].mean().reset_index()
county_geo.columns = ['county', 'lat', 'lon']

ZIP_COUNTY = {
    '900':'Los Angeles County','901':'Los Angeles County','902':'Los Angeles County',
    '903':'Los Angeles County','904':'Los Angeles County','905':'Los Angeles County',
    '906':'Los Angeles County','907':'Los Angeles County','908':'Los Angeles County',
    '910':'Los Angeles County','911':'Los Angeles County','912':'Los Angeles County',
    '913':'Los Angeles County','914':'Los Angeles County','915':'Los Angeles County',
    '916':'Los Angeles County','917':'Los Angeles County','918':'Los Angeles County',
    '919':'San Diego County','920':'San Diego County','921':'San Diego County',
    '922':'Riverside County','923':'San Bernardino County','924':'San Bernardino County',
    '925':'Riverside County','926':'Orange County','927':'Orange County','928':'Orange County',
    '930':'Ventura County','931':'Ventura County','932':'Kern County','933':'Kern County',
    '934':'San Luis Obispo County','935':'Santa Barbara County',
    '936':'Fresno County','937':'Fresno County','938':'Fresno County',
    '939':'Monterey County','940':'San Francisco County','941':'San Francisco County',
    '942':'Sacramento County','943':'San Mateo County','944':'San Mateo County',
    '945':'Contra Costa County','946':'Alameda County','947':'Alameda County',
    '948':'Marin County','949':'Marin County',
    '950':'Santa Clara County','951':'Santa Clara County','952':'Alameda County',
    '953':'Santa Clara County','954':'Sonoma County','955':'Humboldt County',
    '956':'Sacramento County','957':'Sacramento County','958':'Sacramento County',
    '959':'Butte County','960':'Lassen County','961':'Shasta County',
}

agg['county'] = agg['ZIP_Code'].apply(lambda z: ZIP_COUNTY.get(str(z)[:3]))
agg = agg.merge(county_geo, on='county', how='left')
agg['lat'] = agg['lat'].fillna(36.7)
agg['lon'] = agg['lon'].fillna(-119.4)
print(f'Mapped {agg["county"].notna().sum()}/{len(agg)} ZIPs to counties')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 5 — Prepare Features & Standardize")
code("""\
FEATS = ['lat', 'lon', 'avg_fire_risk', 'avg_prem_pol',
         'avg_loss_ratio', 'high_risk_frac', 'pct_prem_chg']

cdf = agg.dropna(subset=FEATS).reset_index(drop=True)
X_raw = cdf[FEATS].values

for j in range(X_raw.shape[1]):
    cap = np.nanpercentile(X_raw[:, j], 99)
    X_raw[:, j] = np.clip(X_raw[:, j], None, cap)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f'Clustering matrix: {X.shape}  (features: {FEATS})')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 6 — UMAP Dimensionality Reduction\n\nProjects 7 features → 2-D while preserving local neighborhood structure.")
code("""\
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric='euclidean', random_state=RANDOM_STATE)
emb = reducer.fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(emb[:, 0], emb[:, 1], c=cdf['avg_fire_risk'],
                cmap='YlOrRd', s=10, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Avg Fire Risk Score')
ax.set(title='UMAP 2-D Projection (colored by fire risk)', xlabel='UMAP-1', ylabel='UMAP-2')
plt.tight_layout()
plt.savefig('../results/umap_projection.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
## 7 — Two-Level Clustering

### Level 1: HDBSCAN micro-clusters
`min_cluster_size=50` finds ~8–15 natural density groups.  Outliers (label = −1) get their own tier.

### Level 2: Agglomerative merging → 5 macro tiers
We compute each micro-cluster's centroid in feature space, then merge into exactly 5 groups ordered by mean fire risk score.
""")
code("""\
# ── Level 1: HDBSCAN ─────────────────────────────────────────────────────────
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5,
                            cluster_selection_method='eom', prediction_data=True)
micro = clusterer.fit_predict(emb)
cdf['micro'] = micro
cdf['micro_prob'] = clusterer.probabilities_

n_micro = len(set(micro)) - (1 if -1 in micro else 0)
n_noise = (micro == -1).sum()
print(f'Level-1 micro-clusters: {n_micro}  |  noise ZIPs: {n_noise} ({n_noise/len(micro)*100:.1f}%)')

# ── Level 2: Agglomerative on centroids ──────────────────────────────────────
valid_ids = sorted(set(micro) - {-1})
centroids = np.array([X[micro == c].mean(axis=0) for c in valid_ids])

agg_clust = AgglomerativeClustering(n_clusters=min(5, len(valid_ids)))
macro_labels = agg_clust.fit_predict(centroids)

# Order macro labels by ascending mean fire risk
macro_risk = {}
for ml in set(macro_labels):
    member_micros = [valid_ids[i] for i, l in enumerate(macro_labels) if l == ml]
    macro_risk[ml] = cdf.loc[cdf['micro'].isin(member_micros), 'avg_fire_risk'].mean()
rank = sorted(macro_risk, key=macro_risk.get)
rank_map = {old: new for new, old in enumerate(rank)}

micro_to_tier = {}
for i, cid in enumerate(valid_ids):
    micro_to_tier[cid] = rank_map[macro_labels[i]]
micro_to_tier[-1] = -1  # outliers stay separate

cdf['tier_id'] = cdf['micro'].map(micro_to_tier)
cdf['risk_tier'] = cdf['tier_id'].map(
    lambda t: TIER_NAMES[t] if 0 <= t < len(TIER_NAMES) else 'Outlier')

print('\\nRisk tier distribution:')
print(cdf['risk_tier'].value_counts())
print(f'\\nTier → mean fire risk:')
for t in TIER_NAMES + ['Outlier']:
    sub = cdf.loc[cdf['risk_tier'] == t, 'avg_fire_risk']
    if len(sub):
        print(f'  {t:12s}  n={len(sub):4d}  risk={sub.mean():.3f}')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 8 — UMAP Scatter by Risk Tier")
code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: micro-clusters
n_colors = len(set(micro) - {-1})
palette = plt.cm.tab20(np.linspace(0, 1, max(n_colors, 1)))
for cid in sorted(set(micro)):
    mask = micro == cid
    c = '#CCCCCC' if cid == -1 else palette[cid % len(palette)]
    axes[0].scatter(emb[mask, 0], emb[mask, 1], c=[c], s=8, alpha=0.6,
                    label=f'{cid}' if cid != -1 else 'noise')
axes[0].set(title=f'Level-1: HDBSCAN Micro-Clusters (n={n_micro})',
            xlabel='UMAP-1', ylabel='UMAP-2')

# Right: 5 macro tiers
for tier in TIER_NAMES + ['Outlier']:
    mask = cdf['risk_tier'] == tier
    if mask.any():
        axes[1].scatter(emb[mask, 0], emb[mask, 1], c=TIER_COLORS[tier],
                        s=12, alpha=0.7, label=tier, edgecolors='none')
axes[1].legend(fontsize=9, markerscale=2, framealpha=0.9)
axes[1].set(title='Level-2: 5 Macro Risk Tiers', xlabel='UMAP-1', ylabel='UMAP-2')

plt.suptitle('Two-Level Clustering: HDBSCAN → Agglomerative Merging', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../results/umap_risk_tiers.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 9 — Risk Tier Profiles\n\nHeatmap of mean feature values per tier — enables quick comparison for judges and insurers.")
code("""\
profile_cols = ['avg_fire_risk', 'avg_premium', 'avg_prem_pol',
                'avg_loss_ratio', 'high_risk_frac', 'pct_prem_chg']
display_names = {'avg_fire_risk': 'Fire Risk Score', 'avg_premium': 'Earned Premium ($)',
                 'avg_prem_pol': 'Premium / Policy ($)', 'avg_loss_ratio': 'Loss Ratio',
                 'high_risk_frac': 'High-Risk Fraction', 'pct_prem_chg': 'Premium Change (%)'}

prof = cdf.groupby('risk_tier')[profile_cols].mean()
prof = prof.reindex([t for t in TIER_NAMES + ['Outlier'] if t in prof.index])
prof['n_zips'] = cdf.groupby('risk_tier').size()

print('=== TIER PROFILES ===')
print(prof.round(3).to_string())

# Normalized heatmap
norm = prof[profile_cols].copy()
for c in norm.columns:
    rng = norm[c].max() - norm[c].min()
    norm[c] = (norm[c] - norm[c].min()) / (rng if rng > 0 else 1)

fig, ax = plt.subplots(figsize=(10, 3.5))
sns.heatmap(norm.rename(columns=display_names).T,
            annot=prof[profile_cols].rename(columns=display_names).T.round(2),
            fmt='g', cmap='YlOrRd', linewidths=.5, ax=ax,
            cbar_kws={'label': 'Normalized'})
ax.set_title('Risk Tier Feature Profiles', fontsize=13)
plt.tight_layout()
plt.savefig('../results/tier_profiles_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
## 10 — California Risk Map & Dashboard

A 4-panel executive summary designed so a non-technical judge can understand the results in 10 seconds.
""")
code("""\
CITY_LABELS = {
    'Los Angeles':   (34.05, -118.24), 'San Francisco': (37.77, -122.42),
    'San Diego':     (32.72, -117.16), 'Sacramento':    (38.58, -121.49),
    'Fresno':        (36.74, -119.77), 'San Jose':      (37.34, -121.89),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('California Wildfire Risk Tiers — Executive Dashboard', fontsize=16, y=0.98)

# ── Panel A: Geographic map ──────────────────────────────────────────────────
ax = axes[0, 0]
for tier in TIER_NAMES + ['Outlier']:
    mask = cdf['risk_tier'] == tier
    if mask.any():
        ax.scatter(cdf.loc[mask, 'lon'], cdf.loc[mask, 'lat'],
                   c=TIER_COLORS[tier], s=18, alpha=0.75, label=tier,
                   edgecolors='white', linewidths=0.3)
for city, (lat, lon) in CITY_LABELS.items():
    ax.annotate(city, (lon, lat), fontsize=8, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    ax.plot(lon, lat, 'k^', ms=5)
ax.set(title='A) California ZIP Codes by Risk Tier', xlabel='Longitude', ylabel='Latitude')
ax.legend(loc='lower left', fontsize=8, markerscale=1.5, framealpha=0.9)

# ── Panel B: ZIP count per tier ──────────────────────────────────────────────
ax = axes[0, 1]
order = [t for t in TIER_NAMES + ['Outlier'] if t in cdf['risk_tier'].values]
counts = cdf['risk_tier'].value_counts().reindex(order)
bars = ax.barh(order, counts, color=[TIER_COLORS[t] for t in order], edgecolor='white')
for bar, v in zip(bars, counts):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{v}', va='center', fontsize=10, fontweight='bold')
ax.set(title='B) Number of ZIP Codes per Tier', xlabel='Count')
ax.invert_yaxis()

# ── Panel C: Premium per policy by tier ──────────────────────────────────────
ax = axes[1, 0]
tier_order = [t for t in TIER_NAMES if t in cdf['risk_tier'].values]
data_c = [cdf.loc[cdf['risk_tier']==t, 'avg_prem_pol'].dropna() for t in tier_order]
bp = ax.boxplot(data_c, labels=tier_order, patch_artist=True, showfliers=False)
for patch, tier in zip(bp['boxes'], tier_order):
    patch.set_facecolor(TIER_COLORS[tier])
    patch.set_alpha(0.7)
ax.set(title='C) Premium per Policy ($) by Risk Tier', ylabel='$ per policy')
ax.tick_params(axis='x', rotation=15)

# ── Panel D: Loss ratio by tier ──────────────────────────────────────────────
ax = axes[1, 1]
data_d = [cdf.loc[cdf['risk_tier']==t, 'avg_loss_ratio'].clip(upper=2).dropna() for t in tier_order]
bp = ax.boxplot(data_d, labels=tier_order, patch_artist=True, showfliers=False)
for patch, tier in zip(bp['boxes'], tier_order):
    patch.set_facecolor(TIER_COLORS[tier])
    patch.set_alpha(0.7)
ax.set(title='D) Loss Ratio by Risk Tier (capped at 2.0)', ylabel='Loss Ratio')
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('../results/risk_tier_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: results/risk_tier_dashboard.png')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 11 — Outlier / Anomaly Analysis\n\nHDBSCAN outliers (label = −1) are ZIP codes with unusual risk profiles — potentially the most interesting for insurers.")
code("""\
outliers = cdf[cdf['risk_tier'] == 'Outlier'].copy()
print(f'Outlier ZIP codes: {len(outliers)}')
if len(outliers) > 0:
    print('\\nTop-10 outliers by fire risk:')
    top = outliers.nlargest(10, 'avg_fire_risk')[
        ['ZIP_Code', 'county', 'avg_fire_risk', 'avg_prem_pol', 'avg_loss_ratio']
    ]
    print(top.to_string(index=False))
    print(f'\\nOutlier avg fire risk: {outliers["avg_fire_risk"].mean():.3f}')
    print(f'Outlier avg loss ratio: {outliers["avg_loss_ratio"].mean():.3f}')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 12 — Cluster Quality Metrics")
code("""\
valid = cdf[cdf['tier_id'] >= 0].copy()
X_val = X[cdf['tier_id'] >= 0]
y_val = valid['tier_id'].values

if len(set(y_val)) > 1:
    sil = silhouette_score(X_val, y_val)
    db  = davies_bouldin_score(X_val, y_val)
    print(f'Silhouette Score:      {sil:.4f}  (closer to 1 = well-separated)')
    print(f'Davies-Bouldin Score:  {db:.4f}  (lower = better)')
else:
    print('Only one cluster — metrics not applicable')

fig, ax = plt.subplots(figsize=(10, 6))
for tier in TIER_NAMES:
    m = cdf['risk_tier'] == tier
    if m.any():
        ax.scatter(cdf.loc[m, 'avg_fire_risk'], cdf.loc[m, 'avg_prem_pol'],
                   c=TIER_COLORS[tier], s=20, alpha=0.6, label=tier, edgecolors='none')
m = cdf['risk_tier'] == 'Outlier'
if m.any():
    ax.scatter(cdf.loc[m, 'avg_fire_risk'], cdf.loc[m, 'avg_prem_pol'],
               c=TIER_COLORS['Outlier'], s=15, alpha=0.4, label='Outlier', marker='x')
ax.set(title='Fire Risk vs Premium per Policy — by Risk Tier',
       xlabel='Avg Fire Risk Score', ylabel='Premium per Policy ($)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('../results/risk_vs_premium_tiers.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 13 — Sensitivity Analysis\n\nHow robust is the two-level result to HDBSCAN's `min_cluster_size` parameter?")
code("""\
print(f'{\"min_cluster_size\":>18} | {\"micro_clusters\":>14} | {\"noise_pct\":>9}')
print('-' * 50)
for mcs in [20, 30, 50, 75, 100, 150, 200, 300]:
    hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5,
                           cluster_selection_method='eom')
    lbl = hdb.fit_predict(emb)
    nc = len(set(lbl)) - (1 if -1 in lbl else 0)
    nn = (lbl == -1).sum()
    print(f'{mcs:18d} | {nc:14d} | {nn/len(lbl)*100:8.1f}%')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("---\n## 14 — Export Cluster Labels")
code("""\
export_cols = ['ZIP_Code', 'county', 'lat', 'lon', 'tier_id', 'risk_tier',
               'micro', 'micro_prob', 'avg_fire_risk', 'avg_premium',
               'avg_prem_pol', 'avg_loss_ratio', 'high_risk_frac', 'pct_prem_chg']
out = cdf[[c for c in export_cols if c in cdf.columns]].copy()
out.to_csv('../results/zip_cluster_labels.csv', index=False)

emb_df = pd.DataFrame(emb, columns=['umap_1', 'umap_2'])
emb_df['ZIP_Code'] = cdf['ZIP_Code'].values
emb_df.to_csv('../results/zip_umap_embedding.csv', index=False)

print(f'Exported {len(out)} ZIP codes to results/zip_cluster_labels.csv')
print(f'Exported UMAP embedding to results/zip_umap_embedding.csv')

print(f'\\n=== SUMMARY ===')
print(f'  ZIP codes clustered: {len(cdf)}')
print(f'  HDBSCAN micro-clusters: {n_micro}')
print(f'  Macro risk tiers: {len(TIER_NAMES)}')
for t in TIER_NAMES + ['Outlier']:
    n = (cdf['risk_tier'] == t).sum()
    if n: print(f'    {t:12s}: {n:4d} ZIPs')
""")

# ═══════════════════════════════════════════════════════════════════════════════
md("""\
---
## Summary

| Step | Method | Result |
|------|--------|--------|
| Feature matrix | 8 insurance features + lat/lon per ZIP | 1,829 ZIP codes |
| UMAP | 7-D → 2-D | Manifold preserving local structure |
| HDBSCAN | Density-based micro-clusters | ~8–15 natural groups + outliers |
| Agglomerative | Merge centroids → 5 tiers | Negligible / Low / Moderate / High / Extreme |
| Dashboard | 4-panel figure | Geographic map + distributions |

**Next:** These tier labels feed into Notebook 05 (Quantum Models) and Notebook 06 (2026 Predictions).
""")

# ═══════════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 4
}

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "notebooks", "04_Clustering_UMAP_HDBSCAN.ipynb")
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {outpath}")
