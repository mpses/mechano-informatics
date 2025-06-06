from patterns import Patterns
from hopfield import Hopfield

PATTERNS = Patterns.PATTERNS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def sweep_noise(patterns, noise_list, trials=300, upd='sync'):
    """
    汎用 ノイズスウィープして平均性能を返す
    ----------------
    patterns : list[np.ndarray]  # 学習パターンのリスト
    noise_list : list[float]     # ノイズの割合リスト (0.0 - 1.0)
    trials : int                 # 各ノイズレベルでの試行回数
    upd : str                    # 'sync' or 'async' 更新方式
    ----------------
    @return : list[tuple]  # 各試行の結果 (k, noise, upd, sim_mean, acc_mean)
    ----------------
    """
    hop = Hopfield(patterns, sync=(upd == 'sync'))
    rows = []
    for noise in noise_list:
        sim_sum, acc_sum = 0.0, 0
        for _ in range(trials):
            idx = np.random.randint(len(patterns))
            tgt = hop.Pmat[idx]
            noisy = hop.add_noise(tgt, noise)
            out = hop.recall(noisy)
            sim_sum += hop.similarity(out, tgt)
            acc_sum += hop.accuracy(out, tgt)
        rows.append((len(patterns), noise, upd,
                     sim_sum/trials, acc_sum/trials))
    return rows


# -------------------------------------------
# 実験 A：k=1..6, noise 5–20%
# -------------------------------------------
noise_A = np.arange(0.05, 0.25, 0.05)
records = []
for k in tqdm(range(1, 7), desc='ExpA k=1..6'):
    records += sweep_noise(PATTERNS[:k], noise_A, trials=300, upd='sync')

# -------------------------------------------
# 実験 B：k=2,4 で noise 0–100%
# -------------------------------------------
noise_B = np.arange(0.0, 1.05, 0.05)
for k in (2, 4):
    records += sweep_noise(PATTERNS[:k], noise_B, trials=400, upd='sync')

# -------------------------------------------
# 実験 C：同期 vs 非同期の差
#          同じ条件(k=4, noise 0–50%)で比較
# -------------------------------------------
noise_C = np.arange(0.0, 0.55, 0.05)
for upd in ('sync', 'async'):
    records += sweep_noise(PATTERNS[:4], noise_C, trials=400, upd=upd)

# -------------------------------------------
# すべて DataFrame に
# -------------------------------------------
df = pd.DataFrame(records,
                  columns=['k', 'noise', 'update', 'similarity', 'accuracy'])
df.to_csv('results/all_results.csv', index=False)

# -------------------------------------------
# 可視化ユーティリティ
# -------------------------------------------

def plot(dfsub, x, y, hue, title, fname):
    plt.figure()
    for label, g in dfsub.groupby(hue):
        plt.plot(g[x]*100, g[y], marker='o', label=str(label))
    plt.xlabel('Noise (%)')
    plt.ylabel(y)
    plt.ylim(0, 1.02)
    plt.title(title)
    plt.legend()
    plt.grid(ls='--', alpha=.4)
    plt.savefig(f'results/{fname}', dpi=150)
    plt.close()


# グラフ1：ExpA k=1..6
for metric in ('similarity', 'accuracy'):
    plot(df.query('noise<=0.2'), 'noise', metric, 'k',
         f'ExpA {metric}', f'expA_{metric}.png')

# グラフ2：ExpB k=2
for metric in ('similarity', 'accuracy'):
    sub = df.query('k==2')
    plot(sub, 'noise', metric, 'k',
         f'ExpB k=2 {metric}', f'expB_k2_{metric}.png')

# グラフ3：ExpB k=4
for metric in ('similarity', 'accuracy'):
    sub = df.query('k==4')
    plot(sub, 'noise', metric, 'k',
         f'ExpB k=4 {metric}', f'expB_k4_{metric}.png')

# グラフ4：同期 vs 非同期 比較 (k=4)
for metric in ('similarity', 'accuracy'):
    sub = df.query('k==4 & noise<=0.5')
    plot(sub, 'noise', metric, 'update',
         f'Sync vs Async (k=4) {metric}', f'sync_async_{metric}.png')
