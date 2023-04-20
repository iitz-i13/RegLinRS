# -*- coding: utf-8 -*-
"""実行モジュール"""

from simulator import ContextualBanditSimulator
from bandit.contextual_bandit import ContextualBandit
from policy.linucb import LinUCB
from policy.lints import LinTS
from policy.linear_full_posterior_sampling import LinearTS
from policy.linear_full_posterior_sampling_fixed import LinearTSfixed
from policy.linrs import LinRS
from policy.linrs_stable import StableLinRS
from policy.reglinrs import RegLinRS
from policy.uniform import Uniform
from policy.greedy import Greedy
from realworld.setup_context import ContextData

def main():
    """main

    Args:
        N_SIMS(int) : sim数
        N_STEPS(int) : step数
        n_contexts(int) : データの個数(step数)
        N_ARMS(int) : 行動の選択肢
        N_FEATURES(int) : 特徴量の次元数

        aleph(float) : 満足化基準値

        policy_list(list[str]) : 検証する方策クラスをまとめたリスト
        bandit : バンディット環境
        bs : バンディットシュミレーター
        data_type(str) : データの種類[mushroom, financial, jester, 
                                        artificial_0.5, artificial_0.7, artificial_0.9, 
                                        mixed_artificial_0.5, mixed_artificial_0.7, mixed_artificial_0.9]
    """
    
    #data_type = 'mushroom'
    #data_type = 'financial'
    #data_type = 'jester'
    # data_type = 'artificial_0.7'
    # data_type = 'mixed_artificial_0.8'
    data_type = 'mixed_artificial_0.75'

    num_actions, context_dim = ContextData.get_data_info(data_type)

    N_SIMS = 1
    N_STEPS = 20000

    N_ARMS = num_actions
    N_FEATURES = context_dim

    #定常環境：True, 非定常環境：False
    steady_flag = True
    change_step = 4000 #何stepで環境変化させるか
    
    if steady_flag == True:
        change_step = N_STEPS
    
    #従来版：True、制限版：False
    Q_flag = False
    data_batch_size_ = 100 #直近何個の特徴ベクトルから行動価値を算出するか
    
    #ReglinRS
    aleph_ = 0.6 #0.6
    memory = 100 #信頼度のメモリ数

    #Mushroom
    # policy_list = [LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, alpha=0.1)
    #             , LinearTSfixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, lambda_prior=0.25, a0=6, b0=6)
    #             , Stablefixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20,n_steps=N_STEPS, aleph=0.4,eta=0.1)]

    # policy_list = [RegionalLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=0.6, k = 50, memory_capacity = 10000, zeta = 0.008, epsilon = 0.0001)]

    #Jester
    """policy_list = [LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, alpha=0.1)
                , LinearTSfixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, lambda_prior=0.25, a0=6, b0=6)
                , Stable(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20,n_steps=N_STEPS, aleph=0.2,eta=0.01)]"""

    #artificial
    policy_list = [RegLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=aleph_, k = 50, memory_capacity=memory, zeta=0.008, epsilon=0.0001, data_batch_size=data_batch_size_),
                   LinearTSfixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, lambda_prior=0.25, a0=6, b0=6, data_batch_size=data_batch_size_),
                   LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, alpha=0.1, data_batch_size=data_batch_size_),
                   Greedy(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20,n_steps=N_STEPS, data_batch_size=data_batch_size_)]
    
    # policy_list = [RegLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=aleph_, k = 50, memory_capacity=memory, zeta=0.008, epsilon=0.0001, data_batch_size=data_batch_size_)]
  
    bandit = ContextualBandit(n_arms=N_ARMS, n_features=N_FEATURES, n_contexts=N_STEPS, data_type=data_type)
    bs = ContextualBanditSimulator(policy_list=policy_list, bandit=bandit, n_sims=N_SIMS,
                         n_steps=N_STEPS, n_arms=N_ARMS, aleph=aleph_, n_features=N_FEATURES, data_type=data_type,change_step=change_step, memory_capacity=memory,data_batch_size=data_batch_size_, steady_flag=steady_flag, Q_flag=Q_flag)

    bs.run()

main()
