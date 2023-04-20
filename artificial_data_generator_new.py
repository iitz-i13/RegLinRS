import numpy as np
import random
import csv
import pandas as pd


class Main():
    def __init__(self, flag):
        self.num_arm = 8 #腕の数
        self.N = 16
        self.feature_dim = self.num_arm * self.N #特徴量の数
        self.aleph_opt = 0.75 #最適な腕の報酬期待値
        self.data_size = 100000
        self.mixed = flag

    def sigmoid(self, x):
        """sigmoid"""
        return 1.0 / (1.0 + np.exp(-x))

    def param_generation(self, p_seed):
        """任意の報酬確率をsigmoidを通す前の重みに変換"""
        w_tmp = np.reciprocal(p_seed)#引数の逆数を要素ごと返す
        w = - np.log(w_tmp-1) #重み計算
        #print(sigmoid(w))#確認用
        return w

    def one_hot_data_sampling(self):
        """one-hot 特徴量の data set 生成"""
        #data_size分の特徴ベクトル(one-hot)作成
        feature = np.zeros((self.data_size, self.feature_dim))
        one_index = [random.randrange(self.feature_dim) for i in range(self.data_size)]#one-hot の1の箇所を決定
        [np.put(feature[i,:], one_index[i], 1) for i in range(self.data_size)]
        # print("one-hot feature:",feature)

        #重み生成
        ##定数表(list)作成
        table = np.array([range(1,self.num_arm+1)]*self.feature_dim)
        roll_num = np.repeat([range(self.num_arm)], self.N)
        table = [np.roll(table[i], roll_num[i]).tolist() for i in range(self.feature_dim)]
        np.random.shuffle(table)
        # print("Rank Table: ")
        # print(table)

        ##ランクに応じた報酬確率定義
        # max_p = self.aleph_opt + 0.05 #最大報酬確率は希求水準+0.05にしとく
        max_p = self.aleph_opt #最大報酬確率は希求水準+0.05にしとく
        diff_p = max_p / self.num_arm #腕の差分計算(等間隔)
        p_seed = np.arange(0+diff_p, self.aleph_opt+diff_p, diff_p)#希求水準にあった報酬確率生成  
        #各腕の報酬確率
        # print("p_seed:")
        # print(p_seed)

        ##報酬確率から重みを決定
        param_seed = self.param_generation(p_seed)
        param_seed = param_seed.tolist()
        sub = {i+1: param_seed[i] for i in range(self.num_arm)}#ランクと重みを結びつけ
        param_data = [[sub.get(x, x) for x in i] for i in table]#置き換え
        # print("param_data: ")
        # print(param_data)

        new_param_data = np.random.permutation(param_data)
        # print("new_param_data: ")
        # print(new_param_data)
        
        #特徴ベクトル * 重み = 報酬確率 を特徴ベクトルの横に結合
        p = [np.dot(feature[i], param_data).tolist() for i in range(self.data_size)]
        p = self.sigmoid(np.array(p))#報酬確率の計算
        # print("final_p : ")
        # print(p)

        new_p = [np.dot(feature[i], new_param_data).tolist() for i in range(self.data_size)]
        new_p = self.sigmoid(np.array(new_p))#報酬確率の計算
        # print("new_p : ")
        # print(new_p)

        feature_data = np.append(feature, p, axis=1).tolist()#結合
        new_feature_data = np.append(feature, new_p, axis=1).tolist()#結合
        # print("feature_data: ")
        # print(feature_data)
        # print("new_feature_data")
        # print(new_feature_data)

        return feature_data, param_data, new_feature_data, new_param_data

    def mixed_data_sampling(self):
        """one-hotのデータセットを元に混合特徴量と生成"""
        mixed_feature_data = []
        mixed_feature_data2 = []
        final_p = []
        new_final_p = []
        #one-hot のデータセット読み込み
        df_feature = pd.read_csv('./datasets/artificial_feature_data_' + str(self.aleph_opt) + '_'+ str(1) + '.csv')
        df_param = pd.read_csv('./datasets/artificial_param_' + str(self.aleph_opt) + '_'+ str(1) +'.csv')
        df_new_feature = pd.read_csv('./datasets/artificial_feature_data_' + str(self.aleph_opt) + '_'+ str(2) + '.csv')
        df_new_param = pd.read_csv('./datasets/artificial_param_' + str(self.aleph_opt) + '_'+ str(2) +'.csv')
        df_p_maxidx = df_param.idxmax(axis = 1)#トップのindex格納
        df_p_maxidx2 = df_new_param.idxmax(axis = 1)#トップのindex格納
        a_param = df_param.values
        new_a_param = df_new_param.values
        # print("a_param : ")
        # print(a_param)
        # print("new_a_param : ")
        # print(new_a_param)


        df_p = df_feature.iloc[:, self.feature_dim:] #報酬確率のみを抽出
        df_p2 = df_new_feature.iloc[:, self.feature_dim:] #報酬確率のみを抽出
        # print("old_p")
        # print(df_p)
        df_maxidx = df_p.idxmax(axis = 1)
        df_maxidx2 = df_p2.idxmax(axis = 1)
        vc = df_maxidx.value_counts()
        vc = [vc["p"+str(i)] for i in range(1, len(vc)+1)]
        vc2 = df_maxidx2.value_counts()
        vc2 = [vc2["p"+str(i)] for i in range(1, len(vc2)+1)]
        #print(vc) #トップの回数(それぞれこの分だけデータ作成する)

        for i in range(1,self.num_arm+1):
            ##paramの中から行を見てトップが同じ列(またその他の列)を抽出
            df_maxidex_arm = df_p_maxidx[df_p_maxidx == 'a'+str(i)]#size = N
            df_non_maxidex_arm = df_p_maxidx[df_p_maxidx != 'a'+str(i)]#size = feature_dim - N
            top_index = df_maxidex_arm.index[:].tolist()
            other_index = df_non_maxidex_arm.index[:].tolist()

            df_maxidex_arm2 = df_p_maxidx2[df_p_maxidx2 == 'a'+str(i)]#size = N
            df_non_maxidex_arm2 = df_p_maxidx2[df_p_maxidx2 != 'a'+str(i)]#size = feature_dim - N
            top_index2 = df_maxidex_arm2.index[:].tolist()
            other_index2 = df_non_maxidex_arm2.index[:].tolist()

            ##平均0、標準偏差0.1の正規分布からN個の乱数を生成
            lam_seed = np.random.normal(0,0.1,(vc[i-1],self.N))
            ##総和で全て割って総和1になるようにする
            lam_sum_array = lam_seed.sum(1)
            lam_sum_array = lam_sum_array.reshape(-1,1)
            lam_seed = (lam_seed/lam_sum_array).tolist()

            ##平均0、標準偏差0.001の正規分布から、0になる特徴量の次元の分(feature_dim - N)の乱数を生成
            epsilon = np.random.normal(0, 0.001, (vc[i-1], self.feature_dim - self.N)).tolist()

            ##上記二つをまとめて重みλを生成
            lam = np.zeros((vc[i-1], self.feature_dim))
            top_index_total = [top_index[j] + self.feature_dim * k for k in range(vc[i-1]) for j in range(len(top_index))]
            other_index_total = [other_index[j] + self.feature_dim * k for k in range(vc[i-1]) for j in range(len(other_index))]
            np.put(lam, top_index_total, lam_seed)
            np.put(lam, other_index_total, epsilon)

            lam2 = np.zeros((vc2[i-1], self.feature_dim))
            top_index_total2 = [top_index2[j] + self.feature_dim * k for k in range(vc2[i-1]) for j in range(len(top_index2))]
            other_index_total2 = [other_index2[j] + self.feature_dim * k for k in range(vc2[i-1]) for j in range(len(other_index2))]
            np.put(lam2, top_index_total2, lam_seed)
            np.put(lam2, other_index_total2, epsilon)
            # print("lam : ")
            # print(lam)

            ##重みλを格納
            lam = lam.tolist()
            mixed_feature_data.extend(lam)
            lam2 = lam2.tolist()
            mixed_feature_data2.extend(lam2)

            ##sigmoidをかけて報酬確率の生成、格納
            p = [np.dot(lam[j], a_param).tolist() for j in range(vc[i-1])]
            p = self.sigmoid(np.array(p))#報酬確率の計算
            final_p.extend(p)#格納

            new_p = [np.dot(lam2[j], new_a_param).tolist() for j in range(vc2[i-1])]
            new_p = self.sigmoid(np.array(new_p))#報酬確率の計算
            new_final_p.extend(new_p)#格納
            
            # print("final_p: ")
            # print(final_p)
            # print("new_final_p: ")
            # print(new_final_p)
  
        #新しい混合特徴量と報酬確率を結合して返す
        # print("final_p",final_p)
        np.array(mixed_feature_data).reshape(self.data_size, self.feature_dim).tolist()
        np.array(mixed_feature_data2).reshape(self.data_size, self.feature_dim).tolist()
        np.array(final_p).reshape(self.data_size, self.num_arm).tolist()
        np.array(new_final_p).reshape(self.data_size, self.num_arm).tolist()
        feature_data = np.append(mixed_feature_data, final_p, axis=1).tolist()#結合
        new_feature_data = np.append(mixed_feature_data2, new_final_p, axis=1).tolist()#結合
        np.random.shuffle(feature_data)#datasetのシャッフル
        np.random.shuffle(new_feature_data)
        return feature_data, a_param, new_feature_data, new_a_param
    
    def data_saving(self, feature_data, param_data, new_feature_data, new_param_data):
        """data set をcsv形式で保存"""
        header1 = ['x' + str(i) for i in range(1,self.feature_dim+1)]
        header2 = ['p' + str(i) for i in range(1,self.num_arm+1)]
        header3 = ['a' + str(i) for i in range(1,self.num_arm+1)]
        header = header1 + header2

        if self.mixed == True:
            with open('./datasets/artificial_feature_data_mixed_' + str(self.aleph_opt) + '_'+ str(1) +'.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(feature_data)
            with open('./datasets/artificial_param_mixed_' + str(self.aleph_opt) + '_'+ str(1) +'.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header3)
                writer.writerows(param_data)
            with open('./datasets/artificial_feature_data_mixed_' + str(self.aleph_opt) + '_'+ str(2) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(new_feature_data)
            with open('./datasets/artificial_param_mixed_' + str(self.aleph_opt) + '_'+ str(2) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header3)
                writer.writerows(new_param_data)
                print("create data!")
        else:
            with open('./datasets/artificial_feature_data_' + str(self.aleph_opt) + '_'+ str(1) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(feature_data)
            with open('./datasets/artificial_param_' + str(self.aleph_opt) + '_'+ str(1) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header3)
                writer.writerows(param_data)
            with open('./datasets/artificial_feature_data_' + str(self.aleph_opt) + '_'+ str(2) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(new_feature_data)
            with open('./datasets/artificial_param_' + str(self.aleph_opt) + '_'+ str(2) + '.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header3)
                writer.writerows(new_param_data)
                print("create data!")

    def data_plot(self):
        """data set の中身確認"""
        #トップの腕が異なっているか可視化　ドユコト？
        if self.mixed == True:
            df_feature = pd.read_csv('./datasets/artificial_feature_data_mixed_' + str(self.aleph_opt) + '_'+ str(1) +'.csv')
            df_param = pd.read_csv('./datasets/artificial_param_mixed_' + str(self.aleph_opt) + '_'+ str(1) +'.csv') 
        else:
            df_feature = pd.read_csv('./datasets/artificial_feature_data_' + str(self.aleph_opt) + '_'+ str(1) +'.csv')
            df_param = pd.read_csv('./datasets/artificial_param_' + str(self.aleph_opt) + '_'+ str(1) +'.csv')
        df_p = df_feature.iloc[:, self.feature_dim:]
        df_p_maxidx = df_p.idxmax(axis = 1)
        vc = df_p_maxidx.value_counts()
        # print("top") #トップの回数
        # print(vc)



FLAG = True #1.one-hot特徴量の場合 False, 2.混合特徴量の場合 True
main = Main(FLAG)

if FLAG == True:
    feature_data, param_data, new_feature_data, new_param_data = main.mixed_data_sampling()
else:   
    feature_data, param_data, new_feature_data, new_param_data = main.one_hot_data_sampling()
main.data_saving(feature_data, param_data, new_feature_data, new_param_data)
main.data_plot()

