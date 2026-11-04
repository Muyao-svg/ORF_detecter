
# This part implements a non-HMM EM on unified big triplet dataframe.
# Non used in main pipeline, but may be useful for comparison and ablation.
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


EPS = 1e-12


def make_templates(rho=0.80):
    other = (1.0 - rho) / 2.0
    return {
        "B":  np.array([1/3, 1/3, 1/3], dtype=float),
        "F0": np.array([rho, other, other], dtype=float),
        "F1": np.array([other, rho, other], dtype=float),
        "F2": np.array([other, other, rho], dtype=float),
    }


def safe_prob(p, eps=EPS):
    p = np.asarray(p, dtype=float)
    p = np.maximum(p, eps)
    return p / p.sum()


# =========================================================
# 2. Multinomial log PMF
# =========================================================
def log_multinomial_pmf(x, prob, eps=EPS):
    x = np.asarray(x, dtype=float)
    prob = safe_prob(prob, eps=eps)

    if x.shape != (3,):
        raise ValueError(f"x must have shape (3,), got {x.shape}")
    if np.any(x < 0):
        raise ValueError("x must be non-negative.")

    n = x.sum()
    log_coeff = gammaln(n + 1.0) - np.sum(gammaln(x + 1.0))
    log_prob = np.sum(x * np.log(prob + eps))
    return float(log_coeff + log_prob)


# =========================================================
# 3. 背景 + 某个 frame 的混合模型
#    theta = (1-lambda) * B + lambda * Fk
# =========================================================
def mix_prob(lmbda, frame_name, templates, eps=EPS):
    theta = (1.0 - lmbda) * templates["B"] + lmbda * templates[frame_name]
    return safe_prob(theta, eps=eps)


# =========================================================
# 4. 单个位点/单个 triplet 拟合
#    输入 x = [x1, x2, x3]
# =========================================================
def fit_one_triplet(x, templates, criterion="BIC"):
    x = np.asarray(x, dtype=float)
    n = x.sum()

    # 全0直接判背景
    if n <= 0:
        return {
            "N": 0.0,
            "best_state_local": "B",
            "lambda_hat": 0.0,
            "logLik_B": np.nan,
            "logLik_F0": np.nan,
            "logLik_F1": np.nan,
            "logLik_F2": np.nan,
            "score_B": np.nan,
            "score_F0": np.nan,
            "score_F1": np.nan,
            "score_F2": np.nan,
            "support_B": 1.0,
            "support_F0": 0.0,
            "support_F1": 0.0,
            "support_F2": 0.0,
            "theta0": 1/3,
            "theta1": 1/3,
            "theta2": 1/3,
        }

    # ---- 背景模型 ----
    ll_B = log_multinomial_pmf(x, templates["B"])

    # ---- frame-specific 模型：只估 lambda ----
    def fit_frame(frame_name):
        def objective(lmbda):
            theta = mix_prob(lmbda, frame_name, templates)
            return -log_multinomial_pmf(x, theta)

        res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        lambda_hat = float(res.x)
        ll_hat = float(-res.fun)
        theta_hat = mix_prob(lambda_hat, frame_name, templates)
        return lambda_hat, ll_hat, theta_hat

    lambda_F0, ll_F0, theta_F0 = fit_frame("F0")
    lambda_F1, ll_F1, theta_F1 = fit_frame("F1")
    lambda_F2, ll_F2, theta_F2 = fit_frame("F2")

    ll_dict = {
        "B": ll_B,
        "F0": ll_F0,
        "F1": ll_F1,
        "F2": ll_F2,
    }

    # 参数个数：B=0, F0/F1/F2 各有1个 lambda
    k_dict = {
        "B": 0,
        "F0": 1,
        "F1": 1,
        "F2": 1,
    }

    if criterion.upper() == "AIC":
        score_dict = {
            s: -2.0 * ll_dict[s] + 2.0 * k_dict[s]
            for s in ll_dict
        }
    else:
        # 默认 BIC
        penalty = np.log(max(n, 2.0))
        score_dict = {
            s: -2.0 * ll_dict[s] + penalty * k_dict[s]
            for s in ll_dict
        }

    best_state_local = min(score_dict, key=score_dict.get) #拥有最小BIC的状态为local best state

    # 相对支持度（基于 logLik）
    ll_vec = np.array([ll_dict["B"], ll_dict["F0"], ll_dict["F1"], ll_dict["F2"]], dtype=float)  # 比较 loglike_B, loglike_F0,F1,F2
    rel = np.exp(ll_vec - np.nanmax(ll_vec))
    support = rel / rel.sum()

    if best_state_local == "B":
        lambda_hat = 0.0
        theta_hat = templates["B"]
    elif best_state_local == "F0":
        lambda_hat = lambda_F0
        theta_hat = theta_F0
    elif best_state_local == "F1":
        lambda_hat = lambda_F1
        theta_hat = theta_F1
    else:
        lambda_hat = lambda_F2
        theta_hat = theta_F2

    return {
        "N": float(n),
        "best_state_local": best_state_local,
        "lambda_hat": float(lambda_hat),

        "logLik_B": float(ll_B),
        "logLik_F0": float(ll_F0),
        "logLik_F1": float(ll_F1),
        "logLik_F2": float(ll_F2),

        # AIC/BIC 分数，越小越好
        "score_B": float(score_dict["B"]),
        "score_F0": float(score_dict["F0"]),
        "score_F1": float(score_dict["F1"]),
        "score_F2": float(score_dict["F2"]),

        # 相对支持度，基于 logLik 转换而来, 越接近1说明越有力支持该状态
        "support_B": float(support[0]),
        "support_F0": float(support[1]),
        "support_F1": float(support[2]),
        "support_F2": float(support[3]),

        # 最后选定状态的混合概率分布参数 theta N = (x1,x2,x3) ~ Multinomial(N, theta), theta = (theta0, theta1, theta2)
        "theta0": float(theta_hat[0]),
        "theta1": float(theta_hat[1]),
        "theta2": float(theta_hat[2]),
    }


# =========================================================
# 5. 局部 frame -> 全局 frame 校正
#    global = (local + frame_run) % 3
# =========================================================
def correct_state_by_frame(best_state_local, frame_run):
    if best_state_local == "B":
        return "B"
    local_num = int(best_state_local[1])   # F0/F1/F2 -> 0/1/2
    global_num = (local_num + int(frame_run)) % 3
    return f"F{global_num}"


def correct_supports_by_frame(frame_run, support_F0, support_F1, support_F2):
    """
    输入是局部 support_F0/F1/F2
    输出是校正后的全局 support_F0/F1/F2
    """
    local = np.array([support_F0, support_F1, support_F2], dtype=float)
    out = np.zeros(3, dtype=float)

    for local_idx in range(3):
        global_idx = (local_idx + int(frame_run)) % 3
        out[global_idx] = local[local_idx]

    return {
        "support_corrige_F0": float(out[0]),
        "support_corrige_F1": float(out[1]),
        "support_corrige_F2": float(out[2]),
    }


# =========================================================
# 6. 批量拟合整个 triplet_df
#    需要列:
#      x1, x2, x3, frame_run
# =========================================================
def fit_triplet_dataframe(triplet_df, rho=0.80, criterion="BIC"):
    required = {"x1", "x2", "x3", "frame_run"}
    missing = required - set(triplet_df.columns)
    if missing:
        raise ValueError(f"triplet_df is missing columns: {sorted(missing)}")

    templates = make_templates(rho=rho)
    records = []

    for _, row in triplet_df.iterrows():
        x = np.array([row["x1"], row["x2"], row["x3"]], dtype=float)

        fit_res = fit_one_triplet(x, templates=templates, criterion=criterion)

        best_state_global = correct_state_by_frame(
            fit_res["best_state_local"],
            row["frame_run"]
        )

        corr_support = correct_supports_by_frame(
            row["frame_run"],
            fit_res["support_F0"],
            fit_res["support_F1"],
            fit_res["support_F2"]
        )

        rec = dict(fit_res)
        rec["best_state_global"] = best_state_global
        rec.update(corr_support)

        records.append(rec)

    fit_df = pd.DataFrame(records)
    out = pd.concat([triplet_df.reset_index(drop=True), fit_df], axis=1)
    return out