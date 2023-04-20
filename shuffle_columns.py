import csv
import random
import os

# 入力ファイル名、出力ファイル名、ランダムに選択する列の数を設定する
input_file_1 = "./datasets/artificial_feature_data_mixed_0.75_1.csv"
output_file_1 = "datasets/artificial_feature_data_mixed_0.75_2.csv"
input_file_2 = "./datasets/artificial_param_mixed_0.75_1.csv"
output_file_2 = "datasets/artificial_param_mixed_0.75_2.csv"
# input_file_1 = "./datasets/artificial_feature_data_0.75_1.csv"
# output_file_1 = "datasets/artificial_feature_data_0.75_2.csv"
# input_file_2 = "./datasets/artificial_param_0.75_1.csv"
# output_file_2 = "datasets/artificial_param_0.75_2.csv"
num_arm = 8  # ランダムに選択する列の数を指定する 腕の数

start_col = -num_arm  # 入れ替える最初の列のインデックス 0~
end_col = -1 # 入れ替える最後の列のインデックス 0~

# 入力ファイル1を読み込む(feature_dataのファイルの列を交換)
with open(input_file_1, 'r') as input_csv_file_1:
    reader1 = csv.reader(input_csv_file_1)
    header1 = next(reader1)  # ヘッダー行を取得する
    rows1 = list(reader1)  # 各行をリストとして取得する
    # 列ごとにランダムな順序を作成する
    col_order_1 = list(range(len(header1)))
    swap_cols_1 = random.sample(range(start_col, end_col+1), num_arm)
    swap_col_order_1 = random.sample(swap_cols_1, len(swap_cols_1))
    for i, j in zip(swap_cols_1, swap_col_order_1):
        col_order_1[i], col_order_1[j] = col_order_1[j], col_order_1[i]
    with open(output_file_1, 'w') as output_csv_file_1:
        writer1 = csv.writer(output_csv_file_1)
        writer1.writerow([header1[i] for i in col_order_1])  # ヘッダー行を書き込む
        for row1 in rows1:
            new_row_1 = [row1[i] if i not in swap_cols_1 else row1[swap_col_order_1[swap_cols_1.index(i)]] for i in col_order_1]
            writer1.writerow(new_row_1)

# 入力ファイル2を読み込む(feature_dataのファイルの列を交換)
with open(input_file_2, 'r') as input_csv_file_2:
    reader2 = csv.reader(input_csv_file_2)
    header2 = next(reader2)  # ヘッダー行を取得する
    rows2 = list(reader2)  # 各行をリストとして取得する
    # 列ごとにランダムな順序を作成する
    col_order_2 = list(range(len(header2)))
    # swap_cols_2 = random.sample(range(start_col, end_col+1), num_arm)
    # swap_col_order_2 = random.sample(swap_cols_2, len(swap_cols_2))
    for i, j in zip(swap_cols_1, swap_col_order_1):
        col_order_2[i], col_order_2[j] = col_order_2[j], col_order_2[i]
    with open(output_file_2, 'w') as output_csv_file_2:
        writer2 = csv.writer(output_csv_file_2)
        writer2.writerow([header2[i] for i in col_order_2])  # ヘッダー行を書き込む
        for row2 in rows2:
            new_row_2 = [row2[i] if i not in swap_cols_1 else row2[swap_col_order_1[swap_cols_1.index(i)]] for i in col_order_2]
            writer2.writerow(new_row_2)