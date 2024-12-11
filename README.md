# RootCanalCTAnalysis
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# === Step 1: CTデータの読み込み ===
def load_ct_data(folder_path):
    """指定されたフォルダからDICOMファイルを読み込む"""
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_series)
    ct_image = reader.Execute()
    return sitk.GetArrayFromImage(ct_image), ct_image

# === Step 2: ROIの抽出 ===
def extract_roi(ct_data, threshold=(-1000, 400)):
    """CTデータの中から指定されたしきい値でROIを抽出"""
    roi = np.where((ct_data >= threshold[0]) & (ct_data <= threshold[1]), ct_data, 0)
    return roi

# === Step 3: 成功基準を設定 ===
def evaluate_success(pre_treatment, post_treatment):
    """治療前後のCTデータを比較して成功の可否を評価"""
    # 病変部位の体積を計算
    pre_volume = np.sum(pre_treatment > 0)
    post_volume = np.sum(post_treatment > 0)

    # 成功基準: 50%以上の体積減少
    success = post_volume / pre_volume <= 0.5
    return success, post_volume / pre_volume

# === Step 4: 統計処理と成功率計算 ===
def calculate_success_rate(data_folder):
    """フォルダ内の複数の患者データを解析し成功率を計算"""
    patient_folders = [os.path.join(data_folder, d) for d in os.listdir(data_folder)]
    results = []

    for patient in patient_folders:
        pre_treatment, _ = load_ct_data(os.path.join(patient, 'pre'))
        post_treatment, _ = load_ct_data(os.path.join(patient, 'post'))

        pre_roi = extract_roi(pre_treatment)
        post_roi = extract_roi(post_treatment)

        success, reduction_rate = evaluate_success(pre_roi, post_roi)
        results.append({'Patient': os.path.basename(patient), 'Success': success, 'ReductionRate': reduction_rate})

    results_df = pd.DataFrame(results)
    success_rate = results_df['Success'].mean() * 100
    return success_rate, results_df

# === 実行 ===
if __name__ == "__main__":
    # データフォルダのパスを指定
    data_folder = "./ct_patient_data"

    success_rate, results_df = calculate_success_rate(data_folder)
    print(f"Root Canal Treatment Success Rate: {success_rate:.2f}%")
    print(results_df)

    # 成果の可視化
    results_df['Success'].value_counts().plot(kind='bar', title='Treatment Success Distribution')
    plt.show()
git add root_canal_analysis.py
git commit -m "Add root canal CT analysis script"
git push origin main
# Root Canal Analysis Tool

## 必要条件
- Python 3.8以上
- 以下のPythonライブラリ：
  - numpy
  - pandas
  - SimpleITK
  - matplotlib
  - scikit-learn

## セットアップ手順
1. Zipファイルを解凍してください。
2. 解凍したフォルダ内でターミナルを開き、以下のコマンドを実行してライブラリをインストールしてください：

## 実行方法
1. CTデータフォルダの準備：
`ct_patient_data/` フォルダ内に、以下のように患者ごとにフォルダを用意してください：

2. コードを実行：
3. git clone https://github.com/your-username/RootCanalCTAnalysis.git
cd RootCanalCTAnalysis
git clone https://github.com/your-username/RootCanalCTAnalysis.git
cd RootCanalCTAnalysis
git clone https://github.com/your-username/RootCanalCTAnalysis.git
cd RootCanalCTAnalysis


4. 成果がコンソールに出力され、可視化されたグラフが表示されます。

## 注意
- 医療データを取り扱う際は、倫理規定とプライバシーに十分配慮してください。
