import csv
import random
import os

# 入力ファイル名、出力ファイル名、ランダムに選択する列の数を設定する
input_file_1 = "./datasets/artificial_feature_data_mixed_0.75_1.csv"
output_file_1 = "datasets/artificial_feature_data_mixed_0.75_2.csv"
output_file_2 = "datasets/artificial_feature_data_mixed_0.75_3.csv"
output_file_3 = "datasets/artificial_feature_data_mixed_0.75_4.csv"
n_arm = 8  # ランダムに選択する列の数を指定する 腕の数
n_feauture = 128 # 特徴量の数

start_col = n_feauture  # 入れ替える最初の列のインデックス 0~
end_col = n_feauture + n_arm -1 # 入れ替える最後の列のインデックス 0~

# 入力ファイル1を読み込む(feature_dataのファイルの列を交換)
with open(input_file_1, 'r') as input_csv_file_1:
    reader1 = csv.reader(input_csv_file_1)
    header1 = next(reader1)  # ヘッダー行を取得する
    rows1 = list(reader1)  # 各行をリストとして取得する
    # 列ごとにランダムな順序を作成する
    col_order_1 = list(range(len(header1)))
    # print(col_order_1)
    swap_cols_1 = random.sample(range(start_col, end_col+1), n_arm)
    while swap_cols_1 == [range(start_col, end_col+1)]:
        swap_cols_1 = random.sample(range(start_col, end_col+1), n_arm)
    # swap_cols_2 = swap_cols_1
    swap_col_order_1 = random.sample(swap_cols_1, len(swap_cols_1))
    for i, j in zip(swap_cols_1, swap_col_order_1):
        col_order_1[i], col_order_1[j] = col_order_1[j], col_order_1[i]
    with open(output_file_1, 'w') as output_csv_file_1:
        writer1 = csv.writer(output_csv_file_1)
        writer1.writerow([header1[i] for i in col_order_1])  # ヘッダー行を書き込む
        for row1 in rows1:
            new_row_1 = [row1[i] if i not in swap_cols_1 else row1[swap_col_order_1[swap_cols_1.index(i)]] for i in col_order_1]
            writer1.writerow(new_row_1)
            
    swap_cols_2 = random.sample(range(start_col, end_col+1), n_arm)
    while swap_cols_2 == [range(start_col, end_col+1)] or swap_cols_2 == swap_cols_1:
        swap_cols_2 = random.sample(range(start_col, end_col+1), n_arm)
    swap_cols_2 = random.sample(range(start_col, end_col+1), n_arm)
    swap_col_order_2 = random.sample(swap_cols_2, len(swap_cols_2))
    for i, j in zip(swap_cols_2, swap_col_order_2):
        col_order_1[i], col_order_1[j] = col_order_1[j], col_order_1[i]
    with open(output_file_2, 'w') as output_csv_file_2:
        writer1 = csv.writer(output_csv_file_2)
        writer1.writerow([header1[i] for i in col_order_1])  # ヘッダー行を書き込む
        for row1 in rows1:
            new_row_1 = [row1[i] if i not in swap_cols_2 else row1[swap_col_order_2[swap_cols_2.index(i)]] for i in col_order_1]
            writer1.writerow(new_row_1)
            
    swap_cols_3 = random.sample(range(start_col, end_col+1), n_arm)
    while swap_cols_3 == [range(start_col, end_col+1)] or swap_cols_3 == swap_cols_1 or swap_cols_3 == swap_cols_2:
        swap_cols_3 = random.sample(range(start_col, end_col+1), n_arm)
    swap_cols_3 = random.sample(range(start_col, end_col+1), n_arm)
    swap_col_order_3 = random.sample(swap_cols_3, len(swap_cols_2))
    for i, j in zip(swap_cols_3, swap_col_order_3):
        col_order_1[i], col_order_1[j] = col_order_1[j], col_order_1[i]
    with open(output_file_3, 'w') as output_csv_file_3:
        writer1 = csv.writer(output_csv_file_3)
        writer1.writerow([header1[i] for i in col_order_1])  # ヘッダー行を書き込む
        for row1 in rows1:
            new_row_1 = [row1[i] if i not in swap_cols_3 else row1[swap_col_order_3[swap_cols_3.index(i)]] for i in col_order_1]
            writer1.writerow(new_row_1)