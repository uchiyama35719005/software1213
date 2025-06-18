import pandas as pd
import numpy as np
import scipy.stats as stats

df = pd.read_excel("lec05_sample_data.xlsx")

# === (3) 変数増加法による説明変数の選択 ===
y = df["価格"].to_numpy(dtype=float)
x1 = df["広さ"].to_numpy(dtype=float)
x2 = df["築年数"].to_numpy(dtype=float)
n = len(y)
Syy = np.sum((y - y.mean())**2)
phi_T = n - 1

# ステップ1-1: 「広さ」単独モデル
Z1 = x1 - x1.mean()
X1 = np.column_stack((np.ones(n), Z1))
beta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ y
y1_hat = X1 @ beta1
e1 = y - y1_hat
Se1 = np.sum(e1**2)
phi_e1 = n - 2
Ve1 = Se1 / phi_e1
F11 = (Syy - Se1) / (phi_T - phi_e1) / Ve1

# ステップ1-2: 「築年数」単独モデル
Z2 = x2 - x2.mean()
X2 = np.column_stack((np.ones(n), Z2))
beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y
y2_hat = X2 @ beta2
e2 = y - y2_hat
Se2 = np.sum(e2**2)
phi_e2 = n - 2
Ve2 = Se2 / phi_e2
F12 = (Syy - Se2) / (phi_T - phi_e2) / Ve2

print("\n=== (3) 変数増加法による説明変数選択 ===")
print(f"ステップ1-1: 広さ を追加すろと　F0 = {F11:.4f}")
print(f"ステップ1-2: 築年数 を追加すろと　F0 = {F12:.4f}")
if F11 >= 2 and F12 >= 2:
    if F11 > F12:
        print("よって　広さを追加")
        # ステップ2: 「築年数」追加
        Z2 = x2 - x2.mean()
        X2 = np.column_stack((np.ones(n), Z1, Z2))
        beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y
        y2_hat = X2 @ beta2
        e2 = y - y2_hat
        Se2 = np.sum(e2**2)
        phi_e2 = n - 3
        Ve2 = Se2 / phi_e2
        F22 = (Se1 - Se2) / (phi_e1 - phi_e2) / Ve2
        print(f"ステップ2: 築年数 を追加すると　F0 = {F22:.4f}")
        if F22 >= 2:
            print("よって　築年数を追加")
        else:
            print("追加無し")   
    else:
        print("よって　築年数を追加")
        # ステップ2: 「広さ」追加
        Z1 = x1 - x1.mean()
        X1 = np.column_stack((np.ones(n), Z2, Z1))
        beta1 = np.linalg.inv(X1.T @ X1) @ X1.T @ y
        y1_hat = X1 @ beta1
        e1 = y - y1_hat
        Se1 = np.sum(e1**2)
        phi_e1 = n - 3
        Ve1 = Se1 / phi_e1
        F21 = (Se2 - Se1) / (phi_e2 - phi_e1) / Ve1
        print(f"ステップ2: 広さ を追加すると　F0 = {F21:.4f}")
        if F21 >= 2:
            print("よって　広さを追加")
        else:
            print("追加無し")
else:
    print("追加なし")

# === (4) 予測値・標準化残差・テコ比 ===
beta_hat = beta2
y_hat = y2_hat
e = e2
Ve = Ve2

# 標準化残差
e_std = e / np.sqrt(Ve)

# テコ比 h
H = X2 @ np.linalg.inv(X2.T @ X2) @ X2.T
h = np.diag(H)

df_result = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "y": y,
    "y_hat": y_hat,
    "標準化残差e'": e_std,
    "テコ比h": h
})
print("\n=== (4) 各種指標（予測値、残差、標準化残差、テコ比） ===")
print(df_result)

# === (5) x0 = [70,10] の場合の予測値と区間推定 ===
x0 = np.array([70, 10])
z0 = np.array([x0[0] - x1.mean(), x0[1] - x2.mean()])
x0_design = np.insert(z0, 0, 1)

# マハラノビス距離 D0^2
cov_mat = np.cov(np.column_stack((x1, x2)).T, bias=True)
inv_cov_mat = np.linalg.inv(cov_mat)
D0_sq = z0 @ inv_cov_mat @ z0.T

# 区間推定
conf_term = np.sqrt((1/n) + (D0_sq / (n - 1)))
pred_term = np.sqrt(1 + (1/n) + (D0_sq / (n - 1)))
t_val = stats.t.ppf(0.975, df=n - 3)
CI_width = t_val * np.sqrt(Ve) * conf_term
PI_width = t_val * np.sqrt(Ve) * pred_term
y0_hat = x0_design @ beta_hat

confidence_interval = (y0_hat - CI_width, y0_hat + CI_width)
prediction_interval = (y0_hat - PI_width, y0_hat + PI_width)

print("\n=== (5) [70,10]の予測値と区間推定 ===")
print(f"予測値: {y0_hat:.4f}")
print(f"95% 信頼区間: {confidence_interval}")
print(f"95% 予測区間: {prediction_interval}")
# 部屋数（room_number）を説明変数に追加しました（Issue #1）