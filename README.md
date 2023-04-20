# 文脈付きバンディット問題のシミュレーション

 
# Requirement
* Python 3.10.9

# Real-World Datasets
データセットの情報 (腕の本数や特徴量) を setup_context.py で設定.
データのサンプリングを data_sampler.py で行う.
datasetsファイルに, 下記の実世界データ(3種類)が入っている. そのためdatasetの入手方法を示すが, 改めて入手する必要はない.なお, csvファイルの大きさの都合上, 人工データ(2種類×5つの水準)は含まれていないため各自で作成する必要がある.

* Mushroom data
    * キノコの22の特徴から可食を判別する.(n=8124) 特徴はone-hotベクトルに変換するため, 特徴ベクトルは117次元. 行動は食べるか食べないかの2種類. 食用キノコを食べると正の報酬が得られ, 毒キノコを食べると確率 p で正の報酬, 確率 1-p で大きな負の報酬が得られる. 食べない時の報酬は0. すべての報酬, および p の値はカスタマイズ可能. データセットは[UCI Machin Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)より取得できる.
    * 注意：こちらは未検証.
* Financial data(raw_stock data)
    * NYSEおよびNasdaqの上場企業 d = 21 社の過去14年間の株価からデータセットを作成(n = 3713). データの内容は各株式の日毎のセッションの開始時と終了時の価格差. 行動は k = 8 とし, 見込みのある有価証券を表す線型結合になるよう作成した. データは[こちら](https://storage.googleapis.com/bandits_datasets/raw_stock_contexts)から入手可能.
    * 注意：現在こちらは使っていない.
* Jester data
    * 合計73421人のユーザーからの100のジョークに対する[-10、10] の連続評価. データのうち, n = 19181 のユーザーが40個のジョークをすべて評価している. この評価のうち d = 32 をユーザーの特徴量として, 残りの k = 8 を行動として用いる. エージェントは1つのジョークを推薦し, 選択したジョークに対するユーザーの評価を報酬として取得する. データは[こちら](https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy)から入手可能.
    * 注意：こちらは未検証.

* 最適切な希求水準が一定となる人工データ
    * 腕の本数 8, 特徴量の次元 128, データサイズ 10万 でパターン1・2 で生成
    * パターン1 one-hot 特徴ベクトル h に対する報酬パラメータWの生成
        * datasize分のone-hotの特徴ベクトルを生成
        * 全ての腕のランキングの回数が同じになるような定数表(特徴次元数×腕の数だけある)を作る
        * 各腕へのランキングに応じた報酬パラメータW(sigmoid を噛ませると報酬確率になるもの)を設計する
            * [0, 定めた最適な希求水準 + 0.5] で均等に報酬確率 P が割り振られるように、各腕の報酬確率を設計
            * 報酬確率から sigmoid をかます前のパラメータWを推定
                * W = -log_e (1/P - 1) で求められる
    * パターン2 パターン1のデータセットを生成後、生成したデータセットを用いて混合特徴量(誤差項を追加した特徴量)の生成
        * 現在使用している人工データセットはこちら
        * 1番高い報酬確率を持つ腕が一致する one-hot特徴ベクトルを抽出
        * 一致する one-hot 特徴ベクトル群から混合係数 λ を生成
            * 一致する one-hot 特徴ベクトル群の中で 1 が入っている次元の箇所に混合係数 λ を割り当てる
                * 平均0、分散0.1 の正規分布から値をサンプリング
                * サンプリングした値の総和が1になるように、合計値でそれぞれの値を割る
                * 算出した値を one-hot の 1 の箇所に割り当てる
            * one-hot の 0 の箇所は微小ノイズを割り当てる
                * 平均0、分散0.001 の正規分布から値をサンプリング
                * その値をそのまま割り当てる
        * 新しく生成した混合特徴ベクトルから報酬確率も計算
            * トップの値が変わってないか確認 (トップの本数のみ確認済み)
    
    * 生成方法
        * 腕の本数 / 特徴量の次元 / 最適な希求水準 / データサイズ は artificial_data_generator_new.py 内で変更可能
        * パターン1 の特徴ベクトルは artificial_data_generator_new.py 内の下の方にある FLAG を以下のように設定して実行すると生成できる
        ```bash
        Flag = False
        ```
        実行コード
        ```bash
        python artificial_data_generator_new.py
        ```
        * パターン2 の特徴ベクトルは artificial_data_generator_new.py 内の下の方にある FLAG を以下のように設定して実行すると生成できる
            * artificial_feature_data_mixed_希求水準.csv と artificial_param_mixed_希求水準.csv が生成される
            * 注意：パターン1 のデータセットを生成後にパターン2 を生成する
        ```bash
        Flag = True
        ```
        実行コード
        ```bash
        python artificial_data_generator_new.py
        ```
        
        
# Usage
(1)実行
* 上記の artificial_data_generator_new.py で人工データセットを作成してから.
* 用いるアルゴリズムや各種パラメータなどシミュレーション設定は main.py 内で変更可能.  
定常環境・非定常環境は steady_flag で変更可能. 非定常環境で実行する場合, change_step で何 step で環境を変化させるか変更可能.
```bash
steady_flag = True # True: steady, False: unsteady
change_step = 4000 
```
行動価値の更新方法は Q_flag で切り替え可能. 制限版で実行する場合, data_batch_size で直近何件のデータを用いて行動価値を推定するかを変更可能.
```bash
Q_flag = True # True: 従来版, False: 制限版
data_batch_size = 100 
```

以下のコマンドで実行.
```bash
python main.py
```
基本的な結果はcsv, 生存率の計算に必要な結果はcsv_reward_count, 報酬期待値が同じ腕ごとの平均二乗誤差の結果はcsv_mseに格納される.

(2)基本的な結果のプロット
```bash
python plot/plot.py csv/結果が入っているディレクトリ
```
ex. csv/202303041234 の csv の内容を plot する場合
```bash
python plot/plot.py csv/202303041234
```
実行した結果はpngフォルダに保存される. 
出力される図

* regrets.png
* rewards.png
* greedy_rate.png
    * エージェントが最適だと思う行動を選択した割合(greedy率)
* accuracy.png
* errors.png
    * 平均誤差率 (MPE)
* entropy_of_reliability
    * LinRS 系の信頼度の推定値に対するエントロピー
    また,各アルゴリズムの 1 sim あたりの平均実行時間は 1 sim time: [ ] でprint される.

(3)生存率の結果のプロット
```bash
python plot/plot_survival_rate.py.py csv_reward_count/結果が入っているディレクトリ 生存ライン 1日のstep数
```
実行した結果はpng/survival_rateディレクトリに保存される. 

(4)真の報酬期待値が同じ腕ごとのMSEの平均の結果のプロット
```bash
python plot/plot_mse.py csv_mse/結果が入っているディレクトリ
```
実行した結果はpng_mseディレクトリに保存される. 
# Note
以下の点が整備しきれていないため注意が必要.

* 現状のコードでは artificial_data_generator_new.py で生成したデータセットのみ検証済み.
 
# Author
* 伊東 将吾
* 東京電機大学理工学部理工学科
* 20rd31@ms.dendai.ac.jp

