from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from patterns import Patterns
from hopfield import Hopfield

PATTERNS = Patterns.PATTERNS


def sweep_noise(patterns, noise_list, trials=300, upd='sync'):
    """
    汎用 ノイズスウィープして平均性能を返す
    ----------------
    patterns : list[np.ndarray]  # 学習パターンのリスト
    noise_list : list[float]     # ノイズの割合リスト (0.0 - 1.0)
    trials : int                 # 各ノイズレベルでの試行回数
    upd : str                    # 'sync' or 'async' 更新方式
    ----------------
    @return : list[tuple]  # 各試行の結果 (phase, k, noise, upd, sim_mean, acc_mean)
    ----------------
    """
    hop = Hopfield(patterns, sync=(upd == 'sync'))
    rows = []
    for noise in noise_list:
        sim_sum, acc_sum = 0.0, 0
        for _ in range(trials):
            idx = np.random.randint(len(patterns))
            tgt = hop.P[idx]
            noisy = hop.add_noise(tgt, noise)
            out = hop.recall(noisy)
            sim_sum += hop.similarity(out, tgt)
            acc_sum += hop.accuracy(out, tgt)
        # フェーズは呼び出し側で付与するので、ここでは phase="" のまま返す
        rows.append((None, len(patterns), noise, upd,
                     sim_sum/trials, acc_sum/trials))
    return rows


# -------------------------------------------
# 実験 A：k=1..6, noise 5–20%
# -------------------------------------------
fixed_noise = 0.10  # ノイズ 10% に固定
records = []
for k in tqdm(range(1, 7), desc='ExpA k=1..6 (noise=10%)'):
    # パターン数 k で一度だけ sweep_noise を呼び出し
    base = sweep_noise(PATTERNS[:k], [fixed_noise], trials=300, upd='async')[0]
    records += [('ExpA',) + base[1:]]

# -------------------------------------------
# 実験 B：k=2,4 で noise 0–100%
# -------------------------------------------
noise_B = np.arange(0.0, 1.05, 0.05)
for k in (2, 4):
    base = sweep_noise(PATTERNS[:k], noise_B, trials=400, upd='async')
    records += [('ExpB',) + row[1:] for row in base]


# -------------------------------------------
# 実験 C：同期 vs 非同期の差
#          同じ条件(k=4, noise 0–50%)で比較
# -------------------------------------------
noise_C = np.arange(0.0, 0.55, 0.05)
for upd in ('sync', 'async'):
    base = sweep_noise(PATTERNS[:4], noise_C, trials=400, upd=upd)
    records += [('ExpC',) + row[1:] for row in base]


# -------------------------------------------
# すべて DataFrame に
# -------------------------------------------
df = pd.DataFrame(records,
                  columns=['phase', 'k', 'noise', 'update', 'similarity', 'accuracy'])
df.to_csv('results/all_results.csv', index=False)


# -------------------------------------------
# 可視化ユーティリティ
# -------------------------------------------

def plot(dfsub, x, y, hue, title, fname):
    plt.figure()
    for label, g in dfsub.groupby(hue):
        g = g.sort_values(x)  # x 軸（noise）で必ずソート
        plt.plot(g[x]*100, g[y], marker='o', label=str(label))
    plt.xlabel('Noise (%)')
    plt.ylabel(y)
    plt.ylim(0, 1.02)
    plt.title(title)
    plt.legend()
    plt.grid(ls='--', alpha=.4)
    plt.savefig(f'results/{fname}', dpi=150)
    plt.close()


# -------------------------------------------
# グラフ1：ExpA k=1..6
# -------------------------------------------
dfA = df[df['phase']=='ExpA']
for metric in ('similarity', 'accuracy'):
    # x 軸を noise→k に変更
    plt.figure()
    g = dfA.sort_values('k')
    plt.plot(g['k'], g[metric], marker='o')
    plt.xlabel('Number of patterns (k)')
    plt.ylabel(metric)
    plt.title(f'ExpA ({metric}), noise={int(fixed_noise*100)}%')
    plt.xticks(range(1,7))
    plt.grid(ls='--', alpha=.4)
    plt.savefig(f'results/expA_k_vs_{metric}.png', dpi=150)
    plt.close()


# -------------------------------------------
# グラフ2：ExpB k=2
# -------------------------------------------
dfB = df[df['phase'] == 'ExpB']
for metric in ('similarity', 'accuracy'):
    plot(dfB.query('k==2'),
         'noise', metric, 'k',
         f'ExpB k=2 {metric}', f'expb_k2_{metric}.png')


# -------------------------------------------
# グラフ3：ExpB k=4
# -------------------------------------------
for metric in ('similarity', 'accuracy'):
    plot(dfB.query('k==4'),
         'noise', metric, 'k',
         f'ExpB k=4 {metric}', f'expb_k4_{metric}.png')


# -------------------------------------------
# グラフ4：ExpC (sync vs async, k=4)
# -------------------------------------------
dfC = df[df['phase'] == 'ExpC']
for metric in ('similarity', 'accuracy'):
    plot(dfC.query('k==4 & noise<=0.5'),
         'noise', metric, 'update',
         f'Sync vs Async (k=4) {metric}', f'sync_async_{metric}.png')
