"""greedy policy"""
import math
from typing import Union
import numpy as np

from policy.base_policy import BaseContextualPolicy


#@は行列用の掛け算
class Greedy(BaseContextualPolicy):
    """Greedy policy

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): 特徴量の次元数
        warmup (int): 各腕を引く回数の最低値
        batch_size (int): パラメータの更新を行う間隔となるstep数
        counts (int): 各腕が選択された回数

    """
    def __init__(self, n_arms: int, n_features: int, warmup: int=1, batch_size: int=1, n_steps:int=1, rwd_gamma: float=1.0, data_batch_size: int=1) -> None:
        """クラスの初期化"""
        super().__init__(n_arms, n_features, warmup, batch_size)

        self.name = 'Greedy'

        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)
        self.n_steps = n_steps

        self.feature = [[] for _ in range(self.n_arms)]
        self.reward_list = [[] for _ in range(self.n_arms)]
        self.data_batch_size = data_batch_size
        

    def initialize(self) -> None:
        """パラメータの初期化"""
        super().initialize()
        #a:行動数,f:特徴量の次元数
        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f

        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)
        
        self.feature = [[] for _ in range(self.n_arms)]
        self.reward_list = [[] for _ in range(self.n_arms)]

    def choose_arm(self, x: np.ndarray) -> np.ndarray:
        """腕の中から1つ選択肢し、インデックスを返す.

        Args:
            x(int, float):特徴量
            step(int):現在のstep数
        Retuens:
            result(int):選んだ行動
        """

        if True in (self.counts < self.warmup):
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f,[2,117]
            self.theta_hat_x = self.theta_hat @ x

            result = np.argmax(self.theta_hat_x)

        return result

    def update_full(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        super().update(x, chosen_arm, reward)
        """パラメータの更新"""
        x = np.expand_dims(x, axis=1)
        #A->ウッドベリーの公式を使って逆行列の更新を効率的にするため(カブトムシ本7章,付録A 参照)
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        self.b[chosen_arm] += np.ravel(x) * reward

        #更新
        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)
            
    def update_limit(self, x: np.matrix, chosen_arm: int, reward: Union[int, float]) -> None:
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        super().update(x, chosen_arm, reward)
        """パラメータの更新"""
        x = np.ravel(x)
        self.feature[chosen_arm].append(x)
        self.reward_list[chosen_arm].append(reward)
        
        if len(self.feature[chosen_arm])<self.data_batch_size:
            x_list = self.feature[chosen_arm]
            reward_list = self.reward_list[chosen_arm]
        else:
            x_list = self.feature[chosen_arm][-self.data_batch_size:]
            reward_list = self.reward_list[chosen_arm][-self.data_batch_size:]

        A_inv = np.identity(self.n_features)
        b = np.zeros(self.n_features)
        
        for i in range(min(len(self.feature[chosen_arm]),self.data_batch_size)):
            x_ = np.expand_dims(x_list[i], axis=1)
            A_inv -= A_inv @ x_ @ x_.T @ A_inv / (1 + x_.T @ A_inv @ x_)
            b += np.ravel(x_list[i]) * reward_list[i]
        self.A_inv[chosen_arm] = A_inv
        self.b[chosen_arm] = b

        #更新
        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)

    def get_theta(self) -> np.ndarray:
        """推定量を返す"""
        return self.theta_hat

    def get_theta_x(self) -> np.ndarray:
        """推定量_特徴量ありを返す"""
        return self.theta_hat_x