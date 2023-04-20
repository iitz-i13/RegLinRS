"""StableLinRS"""
#import math
from operator import index
from typing import Union
import numpy as np
from collections import deque
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
# import pyflann
# from pyflann import FLANN, set_distance_type

from policy.base_policy import BaseContextualPolicy


#@は行列用の掛け算
class RegLinRS(BaseContextualPolicy):
    """Linear Risk-sensitive Satisficing Value Function

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数
        aleph (float): 満足化基準値
        k (int): k-近傍法
        episodic_memory (int): メモリー
        memory_capacity (int): メモリー容量
        zeta (float): 非常に小さいユークリッド距離を丸める閾値
        epsilon (float): 0除算を防ぐための微小な定数
        stable_flag (bool): Stableを加えるか否か
        w (float): Stableを使うときの重み



    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1, aleph: float=1.0, k: int=10, memory_capacity: int=30000, zeta: float=0.008, epsilon: float=0.0001, stable_flag: bool=False, w: float=0.1 ,data_batch_size: int=128) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)
        self.aleph = aleph
        self.k = k
        self.memory_capacity = memory_capacity
        self.zeta = zeta
        self.epsilon = epsilon
        self.stable_flag = stable_flag
        self.w = w
        # self.episodic_memory = np.zeros((self.memory_capacity,(self.n_features + self.n_arms)))
        self.episodic_memory = []
        
        #self.name = 'Regional LinRS(stable) ℵ={}'.format(self.aleph)
        self.name = 'RegLinRS'
        
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)
        self.rs = np.zeros(self.n_arms)
        self.n = np.zeros(self.n_arms)

        self.data_batch_size = data_batch_size
        
        # self.feature = np.zeros((self.n_arms, self.data_batch_size, self.n_features))
        # self.reward_list = np.zeros((self.data_batch_size,(self.n_features + self.n_arms)), dtype=int)
        self.feature = [[] for _ in range(self.n_arms)]
        self.reward_list = [[] for _ in range(self.n_arms)]
        
        self.num = 0 #step数
        self.chosen_arm_num = np.zeros(self.n_arms, dtype=int) #各腕が選ばれた数

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        #a:行動数,f:特徴量の次元数
        self.A = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        # self.episodic_memory = np.zeros((self.memory_capacity,(self.n_features + self.n_arms)))
        self.episodic_memory = []
        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)
        
        self.rs = np.zeros(self.n_arms)
        self.n = np.zeros(self.n_arms)

        # self.feature = np.zeros((self.n_arms, self.data_batch_size, self.n_features))
        # self.reward_list = np.zeros((self.data_batch_size,(self.n_features + self.n_arms)), dtype=int)
        self.feature = [[] for _ in range(self.n_arms)]
        self.reward_list = [[] for _ in range(self.n_arms)]
        
        self.num = 0 #step数
        self.chosen_arm_num = np.zeros(self.n_arms, dtype=int) #各腕が選ばれた数
        
    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す.
        Args:
            x(int, float):特徴量
            step(int):現在のstep数
        Retuens:
            result(int):選んだ行動
        """
        # print(self.counts)
        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
            # print('episodic_memory: ') 
            # print(self.episodic_memory[:self.num])
            
        else:
            """報酬期待値の不偏推定量を計算"""
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f,[2,117]
            self.theta_hat_x = self.theta_hat @ x

            """疑似試行割合の計算"""
            # 入力に対するk近傍法(ユークリッド距離)を計算しリストN_kに格納 (episodicメモリーに対してk-近傍法を行う)

            #episodic memory から中身を抽出
            # context = np.array([r for r in self.episodic_memory[:self.num]])
            context = np.array([r for r in self.episodic_memory])
            #print(context.dtype)
            #print(len(x))

            # 中身を特徴とaction vectorに分離
            context_x = context[:,:len(x)]
            action_vec = context[:,len(x):]
            
            # 次元を増やしてepisodic memory の特徴群に合わせる
            x = np.expand_dims(x, axis = 0)

            #episodic_memoryの特徴群に対してk近傍法のモデルを適用
            if len(context_x) <= self.k:
                k_num = len(context_x)
            elif len(context_x) > self.k:
                k_num = self.k
            
            #エラー出るよ！！
            # pyflann.set_distance_type('euclidean') # defaultはeuclidean
            # flann = pyflann.FLANN()
            # flann.build_index(context_x)
            # indices, distance = flann.nn_index(x_t, num_neighbors=k_num)
            nbrs = NearestNeighbors(n_neighbors = k_num, algorithm='brute', metric='euclidean').fit(context_x)
            distance, indices = nbrs.kneighbors(x) #現状態xと特徴群の距離を算出

            # 次元が多いので削除して計算しやすいようにする
            distance = np.squeeze(distance)
            #action_vecをk個に絞ってからさらにサイズを小さくしている
            action_vec = action_vec[indices]
            action_vec = np.squeeze(action_vec)
            
            # カーネルの計算の準備
            ## 平方ユークリッド距離(ユークリッドの2乗)を計算しd_kに格納
            d_k = np.asarray(distance) ** 2
            ## d_kを使ってユークリッド距離の移動平均d^2_mを計算
            d_m_ave = np.average(d_k)
            ## カーネル値の分母の分数を計算(d_kの正則化)
            #d_n = d_k / d_m_ave #ここでときどきinvalid value encountered in true_divideが起きる→0除算によりNanが生まれるため発生、代わりに0を置き換えるようにする
            d_n = np.divide(d_k, d_m_ave, out=np.zeros_like(d_k), where = d_m_ave!=0)
            ## d_nをクラスタリング(具体的にはあまりに小さい場合0に更新)
            d_n -= self.zeta
            d_n = [i if i > 0 else 0 for i in d_n]
            ## 入力と近傍値のカーネル値(類似度)K_vを計算
            K_v = [self.epsilon / (i + self.epsilon) for i in d_n]
            # 類似度K_vから総和が1となる重み生成。疑似試行回数 n の総和を1にしたいため
            sum_K = np.sum(K_v)
            weight = [K_i/sum_K for K_i in K_v]
            #類似度から算出した重みと action vector で加重平均を行い疑似試行割合を計算
            # print("indices.T: ",indices.T)
            self.n = np.average(action_vec, weights=weight, axis=0)
            
            if self.stable_flag:
                base = self.counts / self.steps
                self.n = base * (1-self.w) + self.n * self.w

            self.rs = self.n * (self.theta_hat_x - self.aleph)  # a*1,[2]
            result = np.argmax(self.rs) #random な argmax に変更

        return result

    def update_full(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        super().update(x, chosen_arm, reward)
        
        # エピソードメモリに現特徴量を格納 (FIFO形式)
        self.T = np.zeros((self.n_arms))
        self.T[chosen_arm] = 1 # 選択した腕に対してのみ1を加える
        x = np.ravel(x)
        memory = np.append(x,self.T,axis=0)
        # python配列
        self.episodic_memory.append(memory)
        # numpy配列
        # if self.num < self.memory_capacity:
        #     self.episodic_memory[self.num] = memory
        # else:
        #     self.episodic_memory[:-1] = self.episodic_memory[1:]
        #     self.episodic_memory[-1] = memory
        # self.num += 1

        """パラメータの更新"""
        #従来版
        x = np.expand_dims(x, axis=1)
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        self.b[chosen_arm] += np.ravel(x) * reward
        
        # エピソードメモリの容量を超えてるかチェック
        if len(self.episodic_memory) > self.memory_capacity:
            self.episodic_memory.pop(0)
        
        # 更新
        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)
    
    def update_limit(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        super().update(x, chosen_arm, reward)
        # エピソードメモリに現特徴量を格納 (FIFO形式)
        self.T = np.zeros((self.n_arms))
        self.T[chosen_arm] = 1 # 選択した腕に対してのみ1を加える
        x = np.ravel(x)
        memory = np.append(x,self.T,axis=0)
        # python配列
        self.episodic_memory.append(memory)
        # if self.num < self.memory_capacity:
        #     self.episodic_memory[self.num] = memory
        # else:
        #     self.episodic_memory[:-1] = self.episodic_memory[1:]
        #     self.episodic_memory[-1] = memory
        # self.num += 1
        
        """パラメータの更新"""
        # 更新版
        #行動価値の更新のため
        self.feature[chosen_arm].append(x)
        self.reward_list[chosen_arm].append(reward)
        # if self.chosen_arm_num[chosen_arm] < self.data_batch_size:
        #     self.feature[chosen_arm][self.chosen_arm_num[chosen_arm]] = x
        #     self.reward_list[chosen_arm][self.chosen_arm_num[chosen_arm]] = reward
        # else:
        #     self.feature[chosen_arm][:-1] = self.feature[chosen_arm][1:]
        #     self.feature[chosen_arm][-1] = x
        #     self.reward_list[chosen_arm][:-1] = self.reward_list[chosen_arm][1:]
        #     self.reward_list[chosen_arm][-1] = reward
                # 用いる data_batch_size の特徴ベクトルの数と reward のリスト
        if len(self.feature[chosen_arm]) < self.data_batch_size:
            x_list = self.feature[chosen_arm]
            reward_list = self.reward_list[chosen_arm]
        else:
            x_list = self.feature[chosen_arm][-self.data_batch_size:]
            reward_list = self.reward_list[chosen_arm][-self.data_batch_size:]
        
        # print('self.feature[chosen_arm]: ')
        # print(self.feature[chosen_arm])
        
        A_inv = np.identity(self.n_features)
        b = np.zeros(self.n_features)
        # if self.chosen_arm_num[chosen_arm] < self.data_batch_size:
        #     x = np.expand_dims(x, axis=1)
        #     self.A_inv[chosen_arm] -= \
        #         self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        #     self.b[chosen_arm] += np.ravel(x) * reward
        # else:
        #     for i in range(self.data_batch_size):
        #         x_ = np.expand_dims(self.feature[chosen_arm][i], axis=1)
        #         A_inv -= A_inv @ x_ @ x_.T @ A_inv / (1 + x_.T @ A_inv @ x_)
        #         b += np.ravel(self.feature[chosen_arm][i]) * self.reward_list[chosen_arm][i]
        #     self.A_inv[chosen_arm] = A_inv
        #     self.b[chosen_arm] = b
        
        if len(self.feature[chosen_arm]) < self.data_batch_size:
            x = np.expand_dims(x, axis=1)
            self.A_inv[chosen_arm] -= \
                self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
            self.b[chosen_arm] += np.ravel(x) * reward
        else:
            for i in range(self.data_batch_size):
                x_ = np.expand_dims(self.feature[chosen_arm][i], axis=1)
                A_inv -= A_inv @ x_ @ x_.T @ A_inv / (1 + x_.T @ A_inv @ x_)
                b += np.ravel(self.feature[chosen_arm][i]) * self.reward_list[chosen_arm][i]
            self.A_inv[chosen_arm] = A_inv
            self.b[chosen_arm] = b
        
        # for i in range(min(len(self.feature[chosen_arm]),self.data_batch_size)):
        #     x_ = np.expand_dims(x_list[i], axis=1)
        #     A_inv -= A_inv @ x_ @ x_.T @ A_inv / (1 + x_.T @ A_inv @ x_)
        #     b += np.ravel(x_list[i]) * reward_list[i]
        # self.A_inv[chosen_arm] = A_inv
        # self.b[chosen_arm] = b

        # self.chosen_arm_num[chosen_arm] += 1

        # エピソードメモリの容量を超えてるかチェック
        if len(self.episodic_memory) > self.memory_capacity:
            self.episodic_memory.pop(0)
            
        # 更新
        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)

    def get_theta(self) -> np.ndarray:
        """推定量を返す"""
        return self.theta_hat

    def get_theta_x(self) -> np.ndarray:
        """推定量_特徴量ありを返す"""    
        return self.theta_hat_x
    
    def get_entropy_arm(self) -> np.ndarray:
        if np.sum(self.n)==0:
            return 1
        return entropy(self.n, base=self.n_arms)

    """def get_phi_x(self) -> np.ndarray:
        #疑似試行回数_特徴量ありを返す
        return np.array(self.n[-1])    
    
    def get_target(self) -> np.ndarray:
        return np.array(self.m[-1])"""
