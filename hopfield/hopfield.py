import numpy as np

class Hopfield:
    """
    patterns : list[np.ndarray]  # 各要素は +1/-1 の 2D 正方形 (sqrt(N) * sqrt(N)) 配列
    sync     : True=同期更新 / False=非同期更新
    max_iter : 収束判定の上限ステップ
    """
    def __init__(self, patterns, sync=False, max_iter=50):
        self.patterns = patterns
        self.P = np.stack([p.flatten() for p in patterns])
        self.N = self.P.shape[1] # n * n
        self.sync = sync
        self.max_iter = max_iter
        self.W = self.train()

    def train(self):
        """
        重み行列 W を学習する
        """
        W = (self.P.T @ self.P) / self.N 
        np.fill_diagonal(W, 0)
        return W
    
    def add_noise(self, vector, ratio):
        """
        vector に反転させる割合 ratio のノイズを加える
        ----------------
        vector : 1D (+1/-1) 状態  # N 次元ベクトル
        ratio  : 反転させる割合 (0 - 1)
        """
        v = vector.copy()
        n_flip = int(self.N * ratio)
        idx = np.random.choice(self.N, n_flip, replace=False)
        v[idx] *= -1
        return v
    
    def recall(self, state):
        """
        学習済みの重み行列 W を用いて状態 state を再帰する
        ----------------
        state : 1D (+1/-1) 状態  # N 次元ベクトル
        """
        state = state.copy()
        for _ in range(self.max_iter):
            if self.sync:
                new_state = np.sign(self.W @ state)
            else:
                idx = np.random.permutation(self.N)
                new_state = state.copy()
                for i in idx:
                    new_state[i] = np.sign(self.W[i] @ state)

            if np.array_equal(new_state, state):
                break
            state = new_state
        
        return state
    

    @staticmethod
    def similarity(v1, v2):
        return (v1 == v2).mean()

    @staticmethod
    def accuracy(v1, v2):
        return 1 if np.array_equal(v1, v2) else 0