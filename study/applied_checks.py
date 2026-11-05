"""应用 ML（CodeSignal 风格）题目的判题。只依赖 numpy / pandas，不需要 PyTorch。

每个检查是 fn(ns)，ns 是提交模块里的名字；assert 失败即未通过。
检查的是性质与手算结果，不对比标准实现，写法不同也能通过。
"""
import math

import numpy as np

CHECKS = {}


def problem(pid, entries, hints):
    def deco(fn):
        CHECKS.setdefault(pid, {"entries": entries, "hints": hints, "checks": []})["checks"].append(fn)
        return fn
    return deco


def reg(pid, entries, hints):
    CHECKS[pid] = {"entries": entries, "hints": hints, "checks": []}

    def add(fn):
        CHECKS[pid]["checks"].append(fn)
        return fn
    return add


# ---------------------------------------------------------------- A.1 metrics
add = reg("a1-metrics", ["confusion_matrix", "precision_recall_f1", "roc_auc"], [
    "confusion_matrix 返回 [[tn, fp], [fn, tp]]；用布尔掩码相乘再 sum，不要写循环里的一堆 if。",
    "precision = tp/(tp+fp)，recall = tp/(tp+fn)，f1 = 2pr/(p+r)；分母为 0 时返回 0.0，而不是 nan。",
    "AUC = 正样本得分高于负样本的概率。把得分排秩（并列取平均秩），再用 (正样本秩和 - P(P+1)/2) / (P*N)。",
])


@add
def check_confusion(ns):
    y, p = np.array([1, 0, 1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1, 0, 1])
    got = np.array(ns.confusion_matrix(y, p))
    assert got.tolist() == [[2, 1], [1, 3]], f"期望 [[tn,fp],[fn,tp]]=[[2,1],[1,3]]，得到 {got.tolist()}"


@add
def check_prf(ns):
    y, p = np.array([1, 0, 1, 1, 0, 0, 1]), np.array([1, 0, 0, 1, 1, 0, 1])
    pr, rc, f1 = ns.precision_recall_f1(y, p)
    assert np.allclose([pr, rc, f1], [0.75, 0.75, 0.75]), f"得到 {(pr, rc, f1)}，期望 (0.75, 0.75, 0.75)"


@add
def check_prf_zero_division(ns):
    r = ns.precision_recall_f1(np.array([1, 1, 0]), np.array([0, 0, 0]))
    assert not any(np.isnan(r)), "没有预测为正类时不能返回 nan"
    assert np.allclose(r, [0, 0, 0]), f"分母为 0 时应返回 0.0，得到 {r}"


@add
def check_auc_known(ns):
    y, s = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(ns.roc_auc(y, s) - 0.75) < 1e-9, f"经典例子 AUC=0.75，得到 {ns.roc_auc(y, s)}"


@add
def check_auc_ties_and_extremes(ns):
    assert abs(ns.roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-9, "得分并列时该算 0.5"
    assert abs(ns.roc_auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) - 1.0) < 1e-9
    assert abs(ns.roc_auc(np.array([1, 1, 0, 0]), np.array([.1, .2, .8, .9])) - 0.0) < 1e-9


@add
def check_auc_random(ns):
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 200), rng.normal(size=200)
    pos, neg = s[y == 1], s[y == 0]
    exp = ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
    assert abs(ns.roc_auc(y, s) - exp) < 1e-9, "与两两比较的定义不一致"


# ------------------------------------------------------------ A.2 preprocessing
add = reg("a2-preprocess", ["standardize", "min_max_scale", "one_hot", "impute_mean"], [
    "按列处理：axis=0。standardize 用总体标准差 (ddof=0)。",
    "常数列的 std 或 max-min 为 0：结果应当是全 0，而不是 nan。",
    "one_hot：categories 默认取排序后的唯一值；用 labels[:,None] == categories[None,:] 一行搞定。",
    "impute_mean：均值要忽略 NaN（np.nanmean），再用 np.where(np.isnan(X), mean, X) 填回。",
])


@add
def check_standardize(ns):
    X = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    Z = np.array(ns.standardize(X))
    assert np.allclose(Z.mean(0), 0) and np.allclose(Z.std(0), 1), "每列应均值 0、总体标准差 1"
    assert X[0, 0] == 1., "不要原地修改输入"


@add
def check_standardize_constant(ns):
    Z = np.array(ns.standardize(np.array([[5., 1.], [5., 2.], [5., 3.]])))
    assert not np.isnan(Z).any(), "常数列产生了 nan"
    assert np.allclose(Z[:, 0], 0)


@add
def check_minmax(ns):
    Z = np.array(ns.min_max_scale(np.array([[1., 5.], [3., 5.], [5., 5.]])))
    assert np.allclose(Z[:, 0], [0, .5, 1]) and np.allclose(Z[:, 1], 0), f"得到 {Z.tolist()}"


@add
def check_one_hot(ns):
    out = np.array(ns.one_hot(np.array(["b", "a", "c", "a"])))
    assert out.shape == (4, 3) and out.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]], \
        f"类别应按排序后的顺序 a,b,c 排列，得到 {out.tolist()}"
    out2 = np.array(ns.one_hot(np.array([2, 0]), categories=np.array([0, 1, 2])))
    assert out2.tolist() == [[0, 0, 1], [1, 0, 0]], "指定 categories 时应按其顺序和个数"


@add
def check_impute(ns):
    X = np.array([[1., np.nan], [3., 4.], [np.nan, 8.]])
    Z = np.array(ns.impute_mean(X))
    assert np.allclose(Z, [[1, 6], [3, 4], [2, 8]]), f"得到 {Z.tolist()}"
    assert np.isnan(X).sum() == 2, "不要原地修改输入"


# -------------------------------------------------------------- A.3 splits / CV
add = reg("a3-splits", ["train_test_split", "k_fold_indices"], [
    "先用 np.random.default_rng(seed).permutation(n) 得到打乱的下标，再切片。",
    "测试集大小 = ceil(n * test_size)（与 sklearn 相同）。",
    "k 折：把打乱下标 np.array_split 成 k 份，第 i 折的验证集是第 i 份，训练集是其余拼起来。",
    "同一个 seed 必须给出同样结果；X 与 y 必须用同一组下标，否则标签会错位。",
])


@add
def check_split_sizes(ns):
    X, y = np.arange(40).reshape(20, 2), np.arange(20)
    a, b, c, d = ns.train_test_split(X, y, test_size=0.25, seed=1)
    assert len(a) == 15 and len(b) == 5 and len(c) == 15 and len(d) == 5, "20 个样本、test_size=0.25 应为 15/5"
    a, b, c, d = ns.train_test_split(np.arange(10).reshape(10, 1), np.arange(10), test_size=0.25, seed=0)
    assert len(b) == 3, "n=10, test_size=0.25 → ceil(2.5)=3"


@add
def check_split_alignment_disjoint(ns):
    X, y = np.arange(60).reshape(30, 2), np.arange(30)
    Xtr, Xte, ytr, yte = ns.train_test_split(X, y, test_size=0.3, seed=3)
    assert np.array_equal(Xtr[:, 0] // 2, ytr) and np.array_equal(Xte[:, 0] // 2, yte), "X 与 y 错位了"
    assert not set(ytr) & set(yte) and len(set(ytr) | set(yte)) == 30, "训练/测试集必须不重叠且覆盖全部样本"


@add
def check_split_seed(ns):
    X, y = np.arange(40).reshape(20, 2), np.arange(20)
    r1, r2 = ns.train_test_split(X, y, seed=7), ns.train_test_split(X, y, seed=7)
    assert all(np.array_equal(p, q) for p, q in zip(r1, r2)), "同一 seed 应得到相同划分"
    r3 = ns.train_test_split(X, y, seed=8)
    assert not np.array_equal(r1[1], r3[1]), "不同 seed 应得到不同划分"


@add
def check_kfold(ns):
    folds = ns.k_fold_indices(23, 5, seed=0)
    assert len(folds) == 5
    val_all = np.concatenate([v for _, v in folds])
    assert sorted(val_all.tolist()) == list(range(23)), "各折验证集合起来应恰好覆盖每个样本一次"
    sizes = sorted(len(v) for _, v in folds)
    assert sizes[-1] - sizes[0] <= 1, f"折大小最多相差 1，得到 {sizes}"
    for tr, va in folds:
        assert not set(tr) & set(va) and len(tr) + len(va) == 23, "每折内训练与验证不能重叠"


# ------------------------------------------------------------------ A.4 linreg
add = reg("a4-linreg", ["mse", "linear_regression", "ridge_regression", "gradient_descent_linreg"], [
    "给 X 拼一列 1 当截距：Xb = np.c_[X, np.ones(n)]；用 np.linalg.lstsq 比手写求逆稳。",
    "岭回归：解 (XᵀX + αI)w = Xᵀy；截距那一项不要加惩罚（把 I 的最后一个对角元设为 0）。",
    "梯度下降：grad_w = 2/n · Xᵀ(Xw+b−y)，grad_b = 2/n · sum(Xw+b−y)。",
    "三个函数都返回 (w, b)，w 形状为 (d,)，b 是标量。",
])


def _lin_data(n=80, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    w = np.array([2.0, -1.0, 0.5])
    return X, X @ w + 3.0 + noise * rng.normal(size=n), w, 3.0


@add
def check_mse(ns):
    assert abs(ns.mse(np.array([1., 2., 3.]), np.array([1., 2., 5.])) - 4 / 3) < 1e-9


@add
def check_ols_recovers(ns):
    X, y, w, b = _lin_data()
    w_, b_ = ns.linear_regression(X, y)
    assert np.allclose(w_, w, atol=1e-6) and abs(b_ - b) < 1e-6, f"无噪声数据应精确恢复，得到 w={np.round(w_, 3)}, b={b_:.3f}"


@add
def check_ridge(ns):
    X, y, w, b = _lin_data(noise=0.1)
    w0, _ = ns.ridge_regression(X, y, 0.0)
    w1, b1 = ns.ridge_regression(X, y, 50.0)
    wo, _ = ns.linear_regression(X, y)
    assert np.allclose(w0, wo, atol=1e-6), "alpha=0 时应与普通最小二乘一致"
    assert np.linalg.norm(w1) < np.linalg.norm(w0), "alpha 变大，权重范数应变小"
    assert abs(b1 - b) < 0.3, "截距不应被惩罚：它应仍接近 3"


@add
def check_gd(ns):
    X, y, w, b = _lin_data(noise=0.05)
    w_, b_ = ns.gradient_descent_linreg(X, y, lr=0.1, n_iter=800)
    wo, bo = ns.linear_regression(X, y)
    assert np.allclose(w_, wo, atol=5e-2) and abs(b_ - bo) < 5e-2, "梯度下降结果应接近解析解"


# --------------------------------------------------------------- A.5 logistic
add = reg("a5-logreg", ["sigmoid", "log_loss", "fit_logistic", "predict_proba"], [
    "稳定 sigmoid：z>=0 用 1/(1+e^-z)，z<0 用 e^z/(1+e^z)；或 np.clip(z,-500,500)。",
    "梯度非常简洁：grad_w = Xᵀ(p−y)/n，grad_b = mean(p−y)；L2 项加 l2·w（不惩罚 b）。",
    "log_loss 先把 p 夹到 [1e-12, 1−1e-12]，否则 log(0) = −inf。",
    "fit_logistic 返回 (w, b)；predict_proba(X, w, b) 返回正类概率。",
])


@add
def check_sigmoid(ns):
    z = np.array([-1000., -1., 0., 1., 1000.])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            s = np.array(ns.sigmoid(z))
        except FloatingPointError:
            raise AssertionError("sigmoid 对大绝对值溢出了，请用稳定写法")
    assert np.allclose(s, [0, 1 / (1 + np.e), .5, np.e / (1 + np.e), 1], atol=1e-9) and not np.isnan(s).any()


@add
def check_logloss(ns):
    y = np.array([1., 0.])
    assert abs(ns.log_loss(y, np.array([.9, .1])) - (-np.log(.9))) < 1e-9
    v = ns.log_loss(y, np.array([1.0, 0.0]))
    assert np.isfinite(v), "p 恰为 0/1 时需要 clip，不能得到 inf"


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    X = np.r_[rng.normal(-2, 1, (100, 2)), rng.normal(2, 1, (100, 2))]
    return X, np.r_[np.zeros(100), np.ones(100)]


@add
def check_fit_separates(ns):
    X, y = _blobs()
    w, b = ns.fit_logistic(X, y, lr=0.2, n_iter=1500)
    acc = ((np.array(ns.predict_proba(X, w, b)) > .5) == y).mean()
    assert acc > 0.95, f"线性可分的两团数据准确率应 > 0.95，得到 {acc:.3f}"


@add
def check_l2_shrinks(ns):
    X, y = _blobs(1)
    w0, _ = ns.fit_logistic(X, y, lr=0.2, n_iter=1500, l2=0.0)
    w1, _ = ns.fit_logistic(X, y, lr=0.2, n_iter=1500, l2=1.0)
    assert np.linalg.norm(w1) < np.linalg.norm(w0), "加 L2 后权重范数应更小"


@add
def check_proba_range(ns):
    X, y = _blobs(2)
    w, b = ns.fit_logistic(X, y)
    p = np.array(ns.predict_proba(X, w, b))
    assert p.shape == (200,) and p.min() >= 0 and p.max() <= 1


# ---------------------------------------------------------------------- A.6 KNN
add = reg("a6-knn", ["knn_predict"], [
    "距离矩阵：((A[:,None,:]-B[None,:,:])**2).sum(-1)，形状 (n_test, n_train)，不需要开方。",
    "np.argsort(D, axis=1)[:, :k] 取最近 k 个下标；再对这些标签做多数表决。",
    "多数表决平票时取数值最小的标签（np.bincount(...).argmax() 天然如此）。",
])


@add
def check_knn_known(ns):
    Xtr = np.array([[0., 0.], [0., 1.], [1., 0.], [5., 5.], [5., 6.], [6., 5.]])
    ytr = np.array([0, 0, 0, 1, 1, 1])
    got = np.array(ns.knn_predict(Xtr, ytr, np.array([[0.2, 0.2], [5.5, 5.2]]), 3))
    assert got.tolist() == [0, 1], f"得到 {got.tolist()}"


@add
def check_knn_k1_self(ns):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 4)), rng.integers(0, 3, 30)
    assert np.array_equal(np.array(ns.knn_predict(X, y, X, 1)), y), "k=1 时在训练集上应完全还原标签"


@add
def check_knn_tie(ns):
    Xtr, ytr = np.array([[0.], [1.], [10.], [11.]]), np.array([3, 1, 3, 1])
    assert int(np.array(ns.knn_predict(Xtr, ytr, np.array([[0.5]]), 2))[0]) == 1, "平票时取最小标签"


# ----------------------------------------------------------- A.7 decision tree
add = reg("a7-tree", ["gini", "best_split"], [
    "gini(y) = 1 − Σ p_c²，p_c 是各类别占比。",
    "对每个特征，对排序后相邻不同取值取中点当候选阈值；规则：x <= thr 进左子树。",
    "增益 = gini(父) − (n_L/n)·gini(左) − (n_R/n)·gini(右)；选增益最大的 (feature, thr)。",
    "节点已纯或找不到任何划分时返回 (None, None, 0.0)。",
])


@add
def check_gini(ns):
    assert abs(ns.gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-9
    assert abs(ns.gini(np.array([1, 1, 1])) - 0.0) < 1e-9
    assert abs(ns.gini(np.array([0, 1, 2, 2])) - (1 - (1 / 16 + 1 / 16 + 4 / 16))) < 1e-9


@add
def check_split_simple(ns):
    X = np.array([[1.], [2.], [3.], [10.], [11.], [12.]])
    f, t, g = ns.best_split(X, np.array([0, 0, 0, 1, 1, 1]))
    assert f == 0 and 3 <= t < 10 and abs(g - 0.5) < 1e-9, f"得到 {(f, t, g)}"
    assert abs(t - 6.5) < 1e-9, "阈值应取相邻取值的中点 6.5"


@add
def check_split_picks_feature(ns):
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(40, 1))
    sig = np.r_[np.zeros(20) + rng.normal(0, .1, 20), np.ones(20) * 5 + rng.normal(0, .1, 20)][:, None]
    X = np.c_[noise, sig]
    f, t, g = ns.best_split(X, np.r_[np.zeros(20), np.ones(20)].astype(int))
    assert f == 1 and 0.5 < t < 4.5, f"应选有信息的第 1 个特征，得到 {(f, t)}"


@add
def check_split_pure(ns):
    f, t, g = ns.best_split(np.array([[1.], [2.]]), np.array([1, 1]))
    assert f is None and t is None and g == 0.0, "纯节点无需划分"


# ----------------------------------------------------------------------- A.8 PCA
add = reg("a8-pca", ["pca"], [
    "先减去每列均值；协方差 C = XcᵀXc / n。",
    "np.linalg.eigh(C) 给出升序特征值，要翻转成降序；特征向量按列排列，返回前需要转置成 (k, d)。",
    "解释方差比 = 特征值 / 全部特征值之和；投影 = Xc @ components.T。",
    "返回 (components, explained_variance_ratio, projected)。",
])


def _pca_data():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 3)) * np.array([5., 2., .5])
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    return Z @ R + np.array([10., -3., 1.])


@add
def check_pca_shapes_orth(ns):
    X = _pca_data()
    C, r, P = ns.pca(X, 2)
    C, r, P = np.array(C), np.array(r), np.array(P)
    assert C.shape == (2, 3) and r.shape == (2,) and P.shape == (200, 2), f"形状 {C.shape} {r.shape} {P.shape}"
    assert np.allclose(C @ C.T, np.eye(2), atol=1e-6), "主成分应当两两正交且单位长度"


@add
def check_pca_variance(ns):
    X = _pca_data()
    Xc = X - X.mean(0)
    ev = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / len(X)))[::-1]
    C, r, P = ns.pca(X, 3)
    assert np.allclose(np.array(r), ev / ev.sum(), atol=1e-6), "解释方差比与协方差矩阵特征值不符"
    assert abs(float(np.sum(r)) - 1) < 1e-6
    assert np.allclose(np.array(P).var(0), ev, atol=1e-6), "第 j 个主成分上的投影方差应等于第 j 个特征值"
    assert list(np.array(r)) == sorted(np.array(r), reverse=True), "要按解释方差从大到小排序"


@add
def check_pca_reconstruct(ns):
    X = _pca_data()
    C, r, P = [np.array(v) for v in ns.pca(X, 3)]
    assert np.allclose(P @ C + X.mean(0), X, atol=1e-6), "取满 3 个成分时应能无损还原"


# ---------------------------------------------------------------- A.9 pandas
add = reg("a9-pandas", ["top_k_per_group", "fill_group_mean", "merge_and_aggregate"], [
    "top_k_per_group：先 sort_values(value, ascending=False)，再 groupby(group).head(k)。",
    "fill_group_mean：df[col].fillna(df.groupby(group)[col].transform('mean'))。",
    "merge_and_aggregate：orders.merge(users, on='user_id', how='left') 之后 groupby(country)['amount'].agg(['sum','count']).reset_index()。",
    "注意不要原地修改传入的 DataFrame，返回新的。",
])


@add
def check_topk(ns):
    import pandas as pd
    df = pd.DataFrame({"g": list("aaabbb"), "v": [1, 5, 3, 9, 2, 7]})
    out = ns.top_k_per_group(df, "g", "v", 2)
    got = {(r.g, r.v) for r in out.itertuples()}
    assert got == {("a", 5), ("a", 3), ("b", 9), ("b", 7)}, f"得到 {sorted(got)}"
    assert list(df["v"]) == [1, 5, 3, 9, 2, 7], "不要修改传入的 DataFrame"


@add
def check_fill(ns):
    import pandas as pd
    df = pd.DataFrame({"g": list("aabb"), "x": [1.0, None, 10.0, None]})
    out = ns.fill_group_mean(df, "g", "x")
    assert out["x"].tolist() == [1.0, 1.0, 10.0, 10.0], f"得到 {out['x'].tolist()}"
    assert df["x"].isna().sum() == 2, "不要修改传入的 DataFrame"


@add
def check_merge(ns):
    import pandas as pd
    users = pd.DataFrame({"user_id": [1, 2, 3], "country": ["US", "US", "FR"]})
    orders = pd.DataFrame({"user_id": [1, 1, 2, 3, 9], "amount": [10, 20, 5, 7, 100]})
    out = ns.merge_and_aggregate(orders, users).sort_values("country").reset_index(drop=True)
    d = {r.country: (r.sum, r.count) for r in out.itertuples()}
    assert d.get("FR") == (7, 1) and d.get("US") == (35, 3), f"得到 {d}（country 缺失的订单 user 9 不应混入已知国家）"
    assert list(out.columns)[:1] == ["country"], "第一列应是 country（reset_index 之后）"


# ============================================================ 3.3 SmolLM 完整模型（PyTorch）
add = reg("s3-smollm-model", ["RMSNorm", "RopeAttention", "SwiGLU", "SmolLM", "greedy_generate"], [
    "解码层是两个“归一化 → 子层 → 残差相加”：x = x + Attn(RMSNorm(x))；x = x + MLP(RMSNorm(x))。",
    "SwiGLU：down( silu(gate(x)) * up(x) )，三个 Linear 都不带偏置，参数量 3·hidden·intermediate。",
    "SmolLM 需要有属性 embed_tokens（nn.Embedding）、layers（nn.ModuleList）、norm（最终 RMSNorm）、lm_head（nn.Linear，无偏置）。",
    "SmolLM.forward(input_ids, labels=None) 返回 (logits, loss)。next-token 损失要错开一位：logits[:, :-1] 对 labels[:, 1:]。",
    "权重共享：lm_head.weight = embed.weight；cfg.tie_word_embeddings 为 False 时不共享。",
    "因果 mask 是加性的：上三角（不含对角）为 -inf，形状 (1,1,S,S)，在 RopeAttention 里加到分数上。",
    "greedy_generate：循环 max_new_tokens 次，每次取最后一个位置 logits 的 argmax 追加到序列。",
])


def _lm_cfg(tie=True, layers=2, vocab=50, hidden=32, heads=4, kv=2):
    from types import SimpleNamespace
    return SimpleNamespace(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden, num_layers=layers,
                           num_heads=heads, kv_heads=kv, num_key_value_heads=kv, head_dim=hidden // heads,
                           rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=tie)


def _build(ns, **kw):
    import torch
    torch.manual_seed(0)
    return ns.SmolLM(_lm_cfg(**kw)).eval()


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


@add
def check_swiglu_shape_and_params(ns):
    import torch
    torch.manual_seed(0)
    H, I = 16, 40
    m = ns.SwiGLU(H, I)
    x = torch.randn(2, 5, H)
    out = m(x)
    assert tuple(out.shape) == (2, 5, H), f"SwiGLU 输出应为 (2,5,{H})，得到 {tuple(out.shape)}"
    assert _nparams(m) == 3 * H * I, f"无偏置的 SwiGLU 应有 3·H·I={3 * H * I} 个参数，得到 {_nparams(m)}"
    assert not torch.allclose(m(2 * x), 2 * out, atol=1e-4), "SwiGLU 是非线性的，f(2x) 不应等于 2f(x)"


@add
def check_forward_shapes(ns):
    import torch
    m = _build(ns)
    ids = torch.randint(0, 50, (3, 7))
    r = m(ids)
    assert isinstance(r, tuple) and len(r) == 2, "forward 应返回 (logits, loss)"
    logits, loss = r
    assert tuple(logits.shape) == (3, 7, 50), f"logits 应为 (3,7,50)，得到 {tuple(logits.shape)}"
    assert loss is None, "没传 labels 时 loss 应为 None"
    _, loss = m(ids, labels=ids)
    assert loss is not None and loss.dim() == 0 and torch.isfinite(loss), "传了 labels 后 loss 应为有限的标量"


@add
def check_loss_is_next_token(ns):
    import torch
    import torch.nn.functional as F
    m = _build(ns)
    torch.manual_seed(1)
    ids = torch.randint(0, 50, (2, 9))
    logits, loss = m(ids, labels=ids)
    exp = F.cross_entropy(logits[:, :-1].reshape(-1, 50), ids[:, 1:].reshape(-1))
    assert torch.allclose(loss, exp, atol=1e-5), f"loss={float(loss):.4f}，按“预测下一个 token”应为 {float(exp):.4f}（要错开一位）"


@add
def check_causal(ns):
    import torch
    m = _build(ns)
    torch.manual_seed(2)
    ids = torch.randint(0, 50, (2, 8))
    ids2 = ids.clone()
    ids2[:, -1] = (ids2[:, -1] + 7) % 50
    with torch.no_grad():
        a, _ = m(ids)
        b, _ = m(ids2)
    drift = (a[:, :-1] - b[:, :-1]).abs().max()
    assert drift < 1e-5, f"改最后一个 token 影响了前面的 logits（最大差 {float(drift):.2e}）：因果 mask 没生效"
    assert (a[:, -1] - b[:, -1]).abs().max() > 1e-6, "改最后一个 token 应该改变最后一个位置的输出"


@add
def check_weight_tying(ns):
    V, H = 50, 32
    tied, untied = _nparams(_build(ns, tie=True)), _nparams(_build(ns, tie=False))
    assert untied - tied == V * H, f"不共享比共享应多 V·H={V * H} 个参数，实际多 {untied - tied}"


@add
def check_layers_stack(ns):
    p1, p2, p3 = (_nparams(_build(ns, layers=n)) for n in (1, 2, 3))
    assert p2 > p1 and p2 - p1 == p3 - p2, f"每多一层应增加相同数量的参数，得到 {p1},{p2},{p3}"


@add
def check_residual_connections(ns):
    """把每个解码层的参数全部清零：若两处都有残差连接，每层都应是恒等映射，
    整个模型就等于 lm_head(norm(embed_tokens(ids)))。"""
    import torch
    m = _build(ns)
    for n in ("embed_tokens", "layers", "norm", "lm_head"):
        assert hasattr(m, n), f"SmolLM 需要有属性 `{n}`（沿用 Hugging Face Llama 的命名）"
    ids = torch.randint(0, 50, (2, 6))
    with torch.no_grad():
        for p in m.layers.parameters():
            p.zero_()
        got, _ = m(ids)
        exp = m.lm_head(m.norm(m.embed_tokens(ids)))
    assert torch.allclose(got, exp, atol=1e-5), "层参数清零后输出不等于“嵌入 → 最终归一化 → lm_head”：注意力和 MLP 两处都要有残差相加"


@add
def check_all_params_get_gradients(ns):
    import torch
    m = _build(ns)
    m.train()
    ids = torch.randint(0, 50, (2, 6))
    _, loss = m(ids, labels=ids)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"这些参数没有收到梯度: {dead[:4]}"


@add
def check_greedy_generate(ns):
    import torch
    m = _build(ns)
    torch.manual_seed(3)
    prompt = torch.randint(0, 50, (2, 4))
    out = ns.greedy_generate(m, prompt, max_new_tokens=5)
    assert tuple(out.shape) == (2, 9), f"应返回 (2, 4+5)，得到 {tuple(out.shape)}"
    assert torch.equal(out[:, :4], prompt), "返回序列的前缀应是原提示"
    with torch.no_grad():
        for t in range(4, 9):
            nxt = m(out[:, :t])[0][:, -1].argmax(-1)
            assert torch.equal(nxt, out[:, t]), f"第 {t - 4} 个新 token 不是上一步最后位置 logits 的 argmax"
    again = ns.greedy_generate(m, prompt, max_new_tokens=5)
    assert torch.equal(out, again), "贪心生成应是确定的"


# ============================================================ 流水线自检用（不出现在看板里）
add = reg("selftest-add", ["add"], ["add(a, b) 返回 a + b。这道题只用来检查流水线本身是否工作正常。"])


@add
def check_add_integers(ns):
    assert ns.add(2, 3) == 5 and ns.add(-1, 1) == 0, "整数加法不对"


@add
def check_add_floats(ns):
    assert abs(ns.add(0.5, 0.25) - 0.75) < 1e-12, "浮点加法不对"


# ============================================================ 4.1 KL 散度（PyTorch）
add = reg("s4-kl", ["kl_divergence", "kl_estimators"], [
    "精确 KL：先对两组 logits 做 log_softmax，再算 Σ p·(log p − log q)，沿最后一维求和。",
    "不要先 softmax 再取 log：logits 很大时会得到 log(0) = −inf。log_softmax 是稳定的。",
    "k1 = log π(x) − log π_ref(x)；k3 = r − log r − 1，其中 r = π_ref(x)/π(x) = exp(log π_ref − log π)。",
    "k3 ≥ 0 来自 x − 1 − log x ≥ 0；它和 k1 的期望（x 从 π 采样）都等于 KL(π‖π_ref)。",
])


@add
def check_kl_matches_definition(ns):
    import torch
    p = torch.tensor([[2.0, 0.0, -1.0]])
    q = torch.tensor([[0.0, 1.0, 0.0]])
    P, Q = p.softmax(-1), q.softmax(-1)
    exp = (P * (P.log() - Q.log())).sum(-1)
    got = ns.kl_divergence(p, q)
    assert tuple(got.shape) == (1,), f"(1,3) 的输入应输出形状 (1,)，得到 {tuple(got.shape)}"
    assert torch.allclose(got, exp, atol=1e-6), f"KL(P‖Q) 应为 {float(exp):.6f}，得到 {float(got):.6f}"


@add
def check_kl_properties(ns):
    import torch
    torch.manual_seed(0)
    p, q = torch.randn(4, 7, 11), torch.randn(4, 7, 11)
    kpq, kqp = ns.kl_divergence(p, q), ns.kl_divergence(q, p)
    assert tuple(kpq.shape) == (4, 7), f"(4,7,11) 应沿最后一维求和得到 (4,7)，得到 {tuple(kpq.shape)}"
    assert bool((kpq >= -1e-6).all()), "KL 不可能为负"
    assert torch.allclose(ns.kl_divergence(p, p), torch.zeros(4, 7), atol=1e-6), "两分布相同时 KL 应为 0"
    assert not torch.allclose(kpq, kqp, atol=1e-4), "KL 不对称：KL(P‖Q) 一般不等于 KL(Q‖P)，检查参数顺序"
    assert torch.allclose(ns.kl_divergence(p + 5.0, q - 3.0), kpq, atol=1e-5), "logits 整体加常数不改变分布，KL 也不应变"


@add
def check_kl_numerically_stable(ns):
    import torch
    p = torch.tensor([[1000.0, 0.0, -1000.0]])
    q = torch.tensor([[0.0, 1000.0, 0.0]])
    got = ns.kl_divergence(p, q)
    assert torch.isfinite(got).all(), "logits 很大时出现了 inf/nan：用 log_softmax，不要 softmax 后再 log"
    assert abs(float(got) - 1000.0) < 1e-2, f"这个例子的 KL 约为 1000，得到 {float(got):.3f}"


@add
def check_estimators_basic(ns):
    import torch
    torch.manual_seed(1)
    lp, lr = torch.randn(50) - 2.0, torch.randn(50) - 2.0
    k1, k3 = ns.kl_estimators(lp, lr)
    assert torch.allclose(k1, lp - lr, atol=1e-6), "k1 应为 log π − log π_ref（逐样本）"
    r = torch.exp(lr - lp)
    assert torch.allclose(k3, r - torch.log(r) - 1, atol=1e-5), "k3 应为 r − log r − 1，r = π_ref/π"
    assert bool((k3 >= -1e-7).all()), "k3 逐样本都不应为负"
    z1, z3 = ns.kl_estimators(lp, lp)
    assert torch.allclose(z1, torch.zeros(50)) and torch.allclose(z3, torch.zeros(50), atol=1e-7), "两者相同时估计值应为 0"


@add
def check_estimators_are_unbiased(ns):
    import torch
    g = torch.Generator().manual_seed(2)
    p_logits, q_logits = torch.tensor([1.0, 0.5, 0.0, -0.5]), torch.tensor([0.8, 0.6, 0.1, -0.4])
    lp_all, lq_all = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    exact = float((lp_all.exp() * (lp_all - lq_all)).sum())
    x = torch.multinomial(lp_all.exp(), 200000, replacement=True, generator=g)
    k1, k3 = ns.kl_estimators(lp_all[x], lq_all[x])
    for name, k in (("k1", k1), ("k3", k3)):
        assert abs(float(k.mean()) - exact) < 2e-3, f"{name} 的样本均值 {float(k.mean()):.5f} 应接近精确 KL {exact:.5f}（样本应来自 π）"
    assert float(k3.var()) < float(k1.var()), "两个分布接近时，k3 的方差应小于 k1"


# ============================================================ 4.3 SFT 数据与损失（PyTorch）
add = reg("s4-sft", ["build_sft_example", "collate", "sft_loss"], [
    "build_sft_example：input = prompt + response + [eos]；labels 与 input 等长，提示部分换成 ignore_index。",
    "collate：右侧补齐。input 用 pad_id，labels 用 ignore_index，attention_mask 真实 token 为 1、补齐处为 0。",
    "sft_loss：错开一位：logits[:, :-1] 对 labels[:, 1:]；用 F.cross_entropy(..., ignore_index=...) 对所有有效 token 求平均。",
    "没有任何有效 token 时返回 0（用 logits.sum() * 0 这类写法保持可反向传播）。",
])


@add
def check_build_example(ns):
    import torch
    inp, lab = ns.build_sft_example([5, 6, 7], [8, 9], eos_id=2, ignore_index=-100)
    inp, lab = torch.as_tensor(inp).tolist(), torch.as_tensor(lab).tolist()
    assert inp == [5, 6, 7, 8, 9, 2], f"input 应为 提示+回答+eos = [5,6,7,8,9,2]，得到 {inp}"
    assert lab == [-100, -100, -100, 8, 9, 2], f"labels 应把提示部分换成 -100、回答和 eos 保留，得到 {lab}"


@add
def check_collate(ns):
    import torch
    ex = [ns.build_sft_example([1, 2], [3], eos_id=0), ns.build_sft_example([4], [5, 6, 7], eos_id=0)]
    out = ns.collate(ex, pad_id=99, ignore_index=-100)
    assert isinstance(out, dict) and {"input_ids", "labels", "attention_mask"} <= set(out), "collate 应返回含 input_ids、labels、attention_mask 的 dict"
    ids, lab, am = (torch.as_tensor(out[k]) for k in ("input_ids", "labels", "attention_mask"))
    assert tuple(ids.shape) == (2, 5), f"按最长的 5 补齐，应为 (2,5)，得到 {tuple(ids.shape)}"
    assert ids[0].tolist() == [1, 2, 3, 0, 99] and lab[0].tolist() == [-100, -100, 3, 0, -100], f"第一条补齐不对：{ids[0].tolist()} / {lab[0].tolist()}"
    assert am.tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], f"attention_mask 不对：{am.tolist()}"


@add
def check_loss_only_on_response(ns):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(3)
    labels = torch.tensor([[-100, -100, 7, 8, 1], [-100, 4, 5, -100, -100]])
    logits = torch.randn(2, 5, 10)
    got = ns.sft_loss(logits, labels, ignore_index=-100)
    lg, lb = logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1)
    keep = lb != -100
    exp = F.cross_entropy(lg[keep], lb[keep])
    assert torch.allclose(torch.as_tensor(got), exp, atol=1e-5), f"应为所有有效 token 的平均交叉熵（错开一位），{float(exp):.5f} vs {float(got):.5f}"
    logits2 = logits.clone()
    logits2[0, 0] += 5.0          # 位置 0 预测位置 1 的提示 token：不应计入
    logits2[1, 3:] += 5.0         # 预测补齐/被忽略位置
    assert torch.allclose(torch.as_tensor(ns.sft_loss(logits2, labels, ignore_index=-100)), exp, atol=1e-5), "改动“预测提示/补齐 token”的 logits 改变了 loss：这些位置应被忽略"


@add
def check_token_level_average(ns):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(4)
    labels = torch.tensor([[-100, 1, 2, 3, 4, 5, 6], [-100, 7, -100, -100, -100, -100, -100]])
    logits = torch.randn(2, 7, 10)
    lg, lb = logits[:, :-1].reshape(-1, 10), labels[:, 1:].reshape(-1)
    tok = F.cross_entropy(lg, lb, ignore_index=-100)
    got = torch.as_tensor(ns.sft_loss(logits, labels, ignore_index=-100))
    assert torch.allclose(got, tok, atol=1e-5), "应按 token 平均（所有有效 token 等权），而不是先按序列平均再平均"


@add
def check_no_valid_tokens(ns):
    import torch
    logits = torch.randn(1, 4, 6, requires_grad=True)
    loss = torch.as_tensor(ns.sft_loss(logits, torch.full((1, 4), -100), ignore_index=-100))
    assert torch.isfinite(loss) and float(loss) == 0.0, f"没有有效 token 时应返回 0，得到 {float(loss)}"
    loss.backward()


@add
def check_gradients_masked(ns):
    import torch
    torch.manual_seed(5)
    labels = torch.tensor([[-100, -100, 3, 4]])
    logits = torch.randn(1, 4, 6, requires_grad=True)
    torch.as_tensor(ns.sft_loss(logits, labels, ignore_index=-100)).backward()
    g = logits.grad[0].abs().sum(-1)
    assert float(g[0]) == 0.0 and float(g[3]) == 0.0, f"预测提示 token 的位置 0 和最后一个位置不应有梯度，得到 {g.tolist()}"
    assert float(g[1]) > 0 and float(g[2]) > 0, "预测回答 token 的位置应有梯度"


# ============================================================ 6.2 MoE 路由与负载均衡（PyTorch）
add = reg("s6-moe-routing", ["topk_route", "load_balancing_loss", "moe_combine"], [
    "topk_route：torch.topk 取每个 token 最大的 k 个 logits，再只对这 k 个做 softmax（等价于把全量 softmax 的概率在这 k 个上重新归一化）。",
    "load_balancing_loss：P_i = 所有 token 在专家 i 上的平均路由概率（对全部专家做 softmax）；f_i = 被分到专家 i 的 (token, 槽位) 占全部 T·k 个分配的比例；loss = N·Σ f_i·P_i。",
    "f_i 来自 argmax/topk，不可导；梯度只经过 P_i 流回路由器。",
    "moe_combine：对每个专家 e，找出选中它的 (token, 槽位)，只把这些 token 送进专家 e，乘上对应权重后用 index_add_ 加回输出。",
])


def _route_data(T=12, N=6, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, N, generator=g)


@add
def check_topk_route(ns):
    import torch
    logits = _route_data()
    w, idx = ns.topk_route(logits, 2)
    assert tuple(w.shape) == (12, 2) and tuple(idx.shape) == (12, 2), f"应返回 (T,k) 的权重与下标，得到 {tuple(w.shape)}、{tuple(idx.shape)}"
    assert torch.allclose(w.sum(-1), torch.ones(12), atol=1e-6), "每个 token 的 k 个权重之和应为 1"
    top = logits.topk(2, dim=-1).indices
    assert torch.equal(idx.sort(-1).values, top.sort(-1).values), "选出的专家应是 logits 最大的 k 个"
    full = logits.softmax(-1).gather(-1, idx)
    assert torch.allclose(w, full / full.sum(-1, keepdim=True), atol=1e-6), "权重应等于全量 softmax 概率在这 k 个专家上重新归一化"
    w1, _ = ns.topk_route(logits, 1)
    assert torch.allclose(w1, torch.ones(12, 1)), "k=1 时权重应全为 1"


@add
def check_load_balancing_values(ns):
    import torch
    N, T = 4, 8
    logits = torch.zeros(T, N)
    idx = torch.arange(T).remainder(N).unsqueeze(-1)            # 完全均衡，k=1
    v = float(ns.load_balancing_loss(logits, idx, N))
    assert abs(v - 1.0) < 1e-6, f"均匀概率 + 均衡分配时 loss 应为 1，得到 {v:.4f}"
    peaked = torch.full((T, N), -20.0); peaked[:, 0] = 20.0
    v2 = float(ns.load_balancing_loss(peaked, torch.zeros(T, 1, dtype=torch.long), N))
    assert abs(v2 - N) < 1e-3, f"所有 token 都去专家 0、概率也全在专家 0 时 loss 应约为 N={N}，得到 {v2:.4f}"
    logits = _route_data(T=10, N=5, seed=3)
    idx = logits.topk(2, -1).indices
    P = logits.softmax(-1).mean(0)
    f = torch.zeros(5).index_add_(0, idx.reshape(-1), torch.ones(20)) / 20
    exp = 5 * (f * P).sum()
    got = torch.as_tensor(ns.load_balancing_loss(logits, idx, 5))
    assert torch.allclose(got, exp, atol=1e-6), f"top-2 时 f_i 按 T·k 个分配计算：应为 {float(exp):.5f}，得到 {float(got):.5f}"


@add
def check_load_balancing_gradient(ns):
    import torch
    logits = _route_data(seed=4).requires_grad_(True)
    idx = logits.detach().topk(2, -1).indices
    loss = ns.load_balancing_loss(logits, idx, logits.shape[1])
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, "梯度没有流回路由 logits：P_i 要由可导的 softmax 计算"


def _experts(N=4, D=6, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    Ws = [torch.randn(D, D, generator=g) for _ in range(N)]
    calls = [0] * N

    def make(i):
        def f(x):
            calls[i] += x.shape[0]
            return torch.tanh(x @ Ws[i])
        return f
    return [make(i) for i in range(N)], Ws, calls


@add
def check_moe_combine_matches_dense(ns):
    import torch
    experts, Ws, _ = _experts()
    g = torch.Generator().manual_seed(5)
    x, logits = torch.randn(10, 6, generator=g), torch.randn(10, 4, generator=g)
    out = ns.moe_combine(x, logits, experts, 2)
    assert tuple(out.shape) == (10, 6), f"输出应为 (T, D)=(10,6)，得到 {tuple(out.shape)}"
    w, idx = logits.topk(2, -1)
    w = w.softmax(-1)
    dense = torch.stack([torch.tanh(x @ W) for W in Ws], 1)            # (T, N, D)
    exp = (dense.gather(1, idx.unsqueeze(-1).expand(-1, -1, 6)) * w.unsqueeze(-1)).sum(1)
    assert torch.allclose(out, exp, atol=1e-5), "输出应为 Σ_j w_tj · E_{idx_tj}(x_t)"


@add
def check_moe_combine_is_sparse(ns):
    import torch
    experts, _, calls = _experts(seed=1)
    g = torch.Generator().manual_seed(6)
    x, logits = torch.randn(16, 6, generator=g), torch.randn(16, 4, generator=g)
    ns.moe_combine(x, logits, experts, 2)
    idx = logits.topk(2, -1).indices
    exp = [int((idx == e).sum()) for e in range(4)]
    assert calls == exp, f"每个专家只应处理分配给它的 token：期望各专家处理 {exp} 行，实际 {calls}"


@add
def check_moe_combine_gradients(ns):
    import torch
    experts, _, _ = _experts(seed=2)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(8, 6, generator=g, requires_grad=True)
    logits = torch.randn(8, 4, generator=g, requires_grad=True)
    ns.moe_combine(x, logits, experts, 2).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0, "梯度没有流回输入"
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0, "梯度没有流回路由 logits：权重要由可导的 softmax 得到"


# ============================================================ P 概率与统计（numpy / math，不需要 scipy）
def _a(x):
    return np.asarray(x, dtype=float)


# ---------- P.1 贝叶斯
add = reg("p1-bayes", ["posterior", "sequential_update", "bernoulli_nb_fit", "bernoulli_nb_predict"], [
    "posterior：逐元素相乘 prior·likelihood，再除以总和（总和就是证据 P(E)）。",
    "sequential_update：在对数空间累加 log prior + Σ_t log likelihood_t，最后减去 logsumexp 再取 exp，避免连乘下溢成 0。",
    "bernoulli_nb_fit：log_prior[c] = log(n_c / n)；θ[c, j] = (类 c 中特征 j 为 1 的个数 + α) / (n_c + 2α)，返回 (log_prior, log θ)。",
    "bernoulli_nb_predict：得分 = log_prior + X·log θᵀ + (1−X)·log(1−θ)ᵀ，取 argmax。log(1−θ) 用 np.log1p(-θ)。",
])


@add
def check_posterior_disease_test(ns):
    post = _a(ns.posterior(np.array([0.01, 0.99]), np.array([0.99, 0.05])))
    assert abs(post.sum() - 1) < 1e-9, "后验之和应为 1"
    assert abs(post[0] - 0.0099 / (0.0099 + 0.0495)) < 1e-9, f"患病率 1%、灵敏度 99%、假阳性 5% 时，阳性后患病概率应约为 0.1667，得到 {post[0]:.4f}"


@add
def check_sequential_update(ns):
    rng = np.random.default_rng(0)
    prior = np.array([0.2, 0.3, 0.5])
    L = rng.uniform(0.1, 0.9, size=(5, 3))
    exp = prior * L.prod(0); exp /= exp.sum()
    got = _a(ns.sequential_update(prior, L))
    assert np.allclose(got, exp, atol=1e-9), "5 次观测后的后验应等于先验乘以全部似然再归一化"
    assert np.allclose(_a(ns.sequential_update(prior, L[::-1])), exp, atol=1e-9), "观测的顺序不应影响最终后验"
    big = rng.uniform(1e-4, 1e-3, size=(2000, 3)); big[:, 2] *= 1.01
    out = _a(ns.sequential_update(prior, big))
    logp = np.log(prior) + np.log(big).sum(0); logp -= logp.max(); ref = np.exp(logp) / np.exp(logp).sum()
    assert np.isfinite(out).all() and abs(out.sum() - 1) < 1e-9, "2000 次很小的似然连乘下溢了：请在对数空间计算"
    assert np.allclose(out, ref, atol=1e-6), "对数空间的结果不对"


def _nb_data():
    X = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@add
def check_nb_fit(ns):
    X, y = _nb_data()
    lp, lt = (_a(v) for v in ns.bernoulli_nb_fit(X, y, alpha=1.0))
    assert lp.shape == (2,) and lt.shape == (2, 3), f"形状应为 (C,)=(2,) 与 (C,D)=(2,3)，得到 {lp.shape} {lt.shape}"
    assert np.allclose(np.exp(lp), [0.5, 0.5]), "两类各 3 个样本，先验应各为 0.5"
    exp0 = (np.array([2, 1, 3]) + 1) / (3 + 2)
    assert np.allclose(np.exp(lt[0]), exp0), f"类 0 的 θ 应为 (计数+1)/(3+2) = {exp0.tolist()}，得到 {np.exp(lt[0]).round(3).tolist()}"


@add
def check_nb_predict(ns):
    X, y = _nb_data()
    lp, lt = ns.bernoulli_nb_fit(X, y, alpha=1.0)
    pred = np.asarray(ns.bernoulli_nb_predict(X, lp, lt))
    assert (pred == y).mean() >= 5 / 6, f"在训练集上应基本分对，得到 {pred.tolist()}"
    Xn = np.array([[1, 0, 1], [0, 1, 0]])
    assert np.asarray(ns.bernoulli_nb_predict(Xn, lp, lt)).tolist() == [0, 1], "典型样本分错了"
    lt0 = _a(lt)
    assert np.isfinite(lt0).all() and (lt0 < 0).all(), "平滑之后 θ 必须严格在 (0,1) 内，log θ 不能是 −inf"


# ---------- P.2 常见分布
add = reg("p2-distributions", ["binomial_pmf", "poisson_pmf", "normal_pdf", "normal_cdf"], [
    "二项分布：C(n,k) p^k (1−p)^(n−k)。n 很大时用 math.lgamma 在对数空间算组合数，再 exp。",
    "泊松：e^{−λ} λ^k / k!，同样在对数空间算：−λ + k log λ − lgamma(k+1)。",
    "正态密度：exp(−(x−μ)²/(2σ²)) / (σ√(2π))；正态分布函数：½(1 + erf((x−μ)/(σ√2)))，用 math.erf。",
    "normal_pdf / normal_cdf 要能接受 numpy 数组（可以用 np.vectorize 包一下 math.erf）。",
])


@add
def check_binomial(ns):
    assert abs(ns.binomial_pmf(3, 10, 0.3) - 0.266827932) < 1e-8, "Binomial(10, 0.3) 在 k=3 处应约为 0.266828"
    assert abs(sum(ns.binomial_pmf(k, 20, 0.37) for k in range(21)) - 1) < 1e-9, "对所有 k 求和应为 1"
    v = ns.binomial_pmf(1000, 2000, 0.5)
    assert np.isfinite(v) and abs(v - 0.017839011) < 1e-7, f"n=2000 时不能溢出，应约为 0.017839，得到 {v}"
    assert ns.binomial_pmf(0, 5, 0.0) == 1.0 and ns.binomial_pmf(5, 5, 1.0) == 1.0, "p=0 或 p=1 的边界情况"


@add
def check_poisson(ns):
    assert abs(ns.poisson_pmf(2, 3.0) - 0.224041808) < 1e-8, "Poisson(3) 在 k=2 处应约为 0.224042"
    assert abs(sum(ns.poisson_pmf(k, 4.5) for k in range(80)) - 1) < 1e-9, "对 k 求和应为 1"
    mean = sum(k * ns.poisson_pmf(k, 4.5) for k in range(80))
    assert abs(mean - 4.5) < 1e-8, "泊松分布的均值应等于 λ"
    assert np.isfinite(ns.poisson_pmf(500, 480.0)), "k、λ 很大时不能溢出"


@add
def check_normal(ns):
    x = np.array([-1.0, 0.0, 1.96])
    pdf, cdf = _a(ns.normal_pdf(x, 0.0, 1.0)), _a(ns.normal_cdf(x, 0.0, 1.0))
    assert pdf.shape == (3,) and cdf.shape == (3,), "应支持数组输入"
    assert abs(pdf[1] - 0.398942280) < 1e-8 and abs(cdf[1] - 0.5) < 1e-12, "标准正态在 0 处：pdf≈0.398942，cdf=0.5"
    assert abs(cdf[2] - 0.975002105) < 1e-8, "Φ(1.96) 应约为 0.975002"
    assert abs(float(ns.normal_cdf(np.array([7.0]), 5.0, 2.0)[0]) - 0.841344746) < 1e-8, "N(5, 2²) 在 7 处的 cdf 应等于 Φ(1)"
    grid = np.linspace(-8, 8, 160001)
    assert abs(_a(ns.normal_pdf(grid, 0.0, 1.0)).sum() * (grid[1] - grid[0]) - 1) < 1e-6, "密度的积分应为 1"


# ---------- P.3 矩与协方差
add = reg("p3-moments", ["mean_var", "covariance_matrix", "correlation_matrix", "welford"], [
    "mean_var(x, ddof)：方差的分母是 n − ddof。",
    "covariance_matrix：先减去每列均值得到 Xc，再算 Xcᵀ Xc / (n − ddof)，结果形状 (D, D)。",
    "correlation_matrix：Corr = Cov / (σ σᵀ)，σ 是各列标准差；对角线应为 1。",
    "welford：一遍扫描，维护 n、mean、M2：delta = x − mean；mean += delta/n；M2 += delta·(x − mean)。最后 var = M2/n。",
])


@add
def check_mean_var(ns):
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    m, v = ns.mean_var(x, ddof=0)
    assert abs(m - 5) < 1e-12 and abs(v - 4) < 1e-12, f"这组数据均值 5、总体方差 4，得到 {m}, {v}"
    _, v1 = ns.mean_var(x, ddof=1)
    assert abs(v1 - 32 / 7) < 1e-12, "ddof=1 时分母是 n−1"


@add
def check_cov_corr(ns):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)) @ rng.normal(size=(4, 4))
    C = _a(ns.covariance_matrix(X, ddof=1))
    assert np.allclose(C, np.cov(X, rowvar=False)), "应与 np.cov(X, rowvar=False) 一致"
    assert np.allclose(C, C.T), "协方差矩阵应对称"
    assert np.linalg.eigvalsh(C).min() > -1e-10, "协方差矩阵应半正定"
    R = _a(ns.correlation_matrix(X))
    assert np.allclose(R, np.corrcoef(X, rowvar=False)) and np.allclose(np.diag(R), 1), "应与 np.corrcoef 一致，对角线为 1"


@add
def check_welford(ns):
    rng = np.random.default_rng(2)
    x = 1e9 + rng.normal(size=10000)
    m, v = ns.welford(iter(x.tolist()))
    assert abs(m - x.mean()) < 1e-6 and abs(v - x.var()) < 1e-6, f"在 1e9 的偏移上仍应精确（总体方差约为 1），得到 var={v:.6f}"
    naive = (x ** 2).mean() - x.mean() ** 2
    assert abs(naive - x.var()) > 1e-2, "（内部自检：朴素公式在这里应明显失准）"


# ---------- P.4 极大似然与 MAP
add = reg("p4-mle", ["gaussian_mle", "gaussian_nll", "bernoulli_map", "categorical_mle"], [
    "高斯的 MLE：μ̂ = 样本均值，σ̂² = (1/n)Σ(x−μ̂)²（分母是 n，不是 n−1）。",
    "gaussian_nll：平均负对数似然 = ½ log(2πσ²) + mean((x−μ)²) / (2σ²)。",
    "Beta(a, b) 先验 + k 次成功 / n 次试验的后验是 Beta(a+k, b+n−k)，MAP 是它的众数 (k+a−1)/(n+a+b−2)。",
    "categorical_mle(counts, alpha)：(c_i + α) / (N + Kα)，α=0 就是频率。",
])


@add
def check_gaussian_mle(ns):
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 3.0, size=500)
    mu, var = ns.gaussian_mle(x)
    assert abs(mu - x.mean()) < 1e-12 and abs(var - x.var()) < 1e-9, "MLE 的方差分母是 n"
    base = ns.gaussian_nll(x, mu, var)
    for dmu, dvar in ((0.1, 0), (-0.1, 0), (0, 0.5), (0, -0.5)):
        assert ns.gaussian_nll(x, mu + dmu, var + dvar) > base, "MLE 应当是负对数似然的最小点"
    assert abs(ns.gaussian_nll(np.array([0.0]), 0.0, 1.0) - 0.5 * math.log(2 * math.pi)) < 1e-12, "N(0,1) 在 0 处的 NLL 应为 ½log(2π)"


@add
def check_bernoulli_map(ns):
    assert abs(ns.bernoulli_map(7, 10, 1.0, 1.0) - 0.7) < 1e-12, "均匀先验 Beta(1,1) 时 MAP 等于 MLE"
    assert abs(ns.bernoulli_map(7, 10, 11.0, 11.0) - 17 / 30) < 1e-12, "强先验 Beta(11,11) 会把估计拉向 0.5：(7+10)/(10+20) = 0.567"
    assert abs(ns.bernoulli_map(0, 3, 2.0, 2.0) - 0.2) < 1e-12, "0 次成功时 MLE 是 0，Beta(2,2) 的 MAP 是 1/5"


@add
def check_categorical_mle(ns):
    c = np.array([3, 0, 1])
    assert np.allclose(_a(ns.categorical_mle(c, 0.0)), [0.75, 0, 0.25]), "α=0 时就是频率"
    s = _a(ns.categorical_mle(c, 1.0))
    assert np.allclose(s, np.array([4, 1, 2]) / 7) and abs(s.sum() - 1) < 1e-12, "加一平滑：(c+1)/(N+K)"


# ---------- P.5 置信区间与 bootstrap
add = reg("p5-ci", ["norm_ppf", "mean_ci", "bootstrap_ci"], [
    "norm_ppf：标准正态分布函数 Φ 单调递增，可以在 [−10, 10] 上二分，直到 Φ(z) 与 p 的差小于 1e-12。Φ 用 math.erf 写。",
    "mean_ci：x̄ ± z·s/√n，s 用 ddof=1 的样本标准差，z = norm_ppf(1 − (1−conf)/2)。",
    "bootstrap_ci：rng = np.random.default_rng(seed)；重复 n_boot 次有放回地抽 n 个样本、算统计量；取 (1−conf)/2 和 1−(1−conf)/2 两个分位数（np.quantile）。",
])


@add
def check_norm_ppf(ns):
    for p, z in ((0.975, 1.959963985), (0.5, 0.0), (0.05, -1.644853627), (0.995, 2.575829304)):
        assert abs(ns.norm_ppf(p) - z) < 1e-6, f"norm_ppf({p}) 应约为 {z}，得到 {ns.norm_ppf(p)}"


@add
def check_mean_ci(ns):
    x = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    lo, hi = ns.mean_ci(x, 0.95)
    half = 1.959963985 * x.std(ddof=1) / math.sqrt(5)
    assert abs(lo - (6 - half)) < 1e-6 and abs(hi - (6 + half)) < 1e-6, f"应为 6 ± {half:.4f}，得到 ({lo:.4f}, {hi:.4f})"
    rng = np.random.default_rng(4)
    cover = np.mean([(lambda c: c[0] <= 0 <= c[1])(ns.mean_ci(rng.normal(size=60), 0.95)) for _ in range(2000)])
    assert 0.93 <= cover <= 0.97, f"95% 区间在重复实验里应约有 95% 覆盖真值，得到 {cover:.3f}"


@add
def check_bootstrap(ns):
    rng = np.random.default_rng(5)
    x = rng.exponential(2.0, size=200)
    a = ns.bootstrap_ci(x, np.mean, 2000, 0.95, 7)
    b = ns.bootstrap_ci(x, np.mean, 2000, 0.95, 7)
    assert a[0] == b[0] and a[1] == b[1], "同一个 seed 应得到相同的区间"
    assert a[0] < x.mean() < a[1], "区间应包含样本均值"
    lo, hi = ns.mean_ci(x, 0.95)
    assert 0.8 < (a[1] - a[0]) / (hi - lo) < 1.25, "均值的 bootstrap 区间宽度应与正态近似相当"
    m = ns.bootstrap_ci(x, np.median, 1000, 0.9, 1)
    assert m[0] < np.median(x) < m[1], "对中位数也应适用"


# ---------- P.6 假设检验
add = reg("p6-testing", ["two_proportion_ztest", "welch_t", "bonferroni", "benjamini_hochberg"], [
    "两比例 z 检验：合并比例 p̂ = (x1+x2)/(n1+n2)，SE = √(p̂(1−p̂)(1/n1 + 1/n2))，z = (p1 − p2)/SE，双侧 p = 2(1 − Φ(|z|))。",
    "Welch t：t = (ā − b̄)/√(s_a²/n_a + s_b²/n_b)，样本方差用 ddof=1；自由度用 Welch–Satterthwaite 公式。",
    "Bonferroni：p_i ≤ α/m 才拒绝。",
    "BH：p 值升序排列，找最大的 k 使 p_(k) ≤ (k/m)·α，拒绝排名前 k 的全部假设（按原顺序返回布尔数组）。",
])


@add
def check_two_proportion(ns):
    z, p = ns.two_proportion_ztest(200, 1000, 250, 1000)
    assert abs(z - (-2.677397763)) < 1e-6, f"z 应约为 −2.6774，得到 {z}"
    assert abs(p - 0.007419649) < 1e-6, f"双侧 p 应约为 0.00742，得到 {p}"
    z0, p0 = ns.two_proportion_ztest(50, 100, 50, 100)
    assert abs(z0) < 1e-12 and abs(p0 - 1) < 1e-9, "两组比例相同时 z=0、p=1"


@add
def check_welch(ns):
    a = [5.1, 4.9, 5.6, 5.8, 6.0, 5.2]
    b = [4.1, 4.5, 4.0, 4.8, 4.4, 4.2, 4.6, 4.3]
    t, df = ns.welch_t(np.array(a), np.array(b))
    assert abs(t - 5.353005116) < 1e-6 and abs(df - 7.815014676) < 1e-6, f"应为 t≈5.3530、df≈7.8150，得到 {t}, {df}"


@add
def check_multiple_testing(ns):
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.20])
    assert np.asarray(ns.bonferroni(p, 0.05)).tolist() == [True, False, False, True, False], "Bonferroni 的阈值是 0.05/5 = 0.01"
    bh = np.asarray(ns.benjamini_hochberg(p, 0.05)).tolist()
    assert bh == [True, True, True, True, False], f"BH：排序后 0.04 ≤ 4/5·0.05 = 0.04，所以前 4 个都拒绝；得到 {bh}"
    assert np.asarray(ns.benjamini_hochberg(np.array([0.5, 0.6]), 0.05)).tolist() == [False, False], "都不显著时一个也不拒绝"
    gap = np.asarray(ns.benjamini_hochberg(np.array([0.03, 0.001, 0.035, 0.2]), 0.05)).tolist()
    assert gap == [True, True, True, False], (
        "排序后 0.03 > 2/4·0.05，但 0.035 ≤ 3/4·0.05：BH 要找满足条件的“最大”排名 k，再拒绝前 k 个，"
        f"不能在第一个不满足处停下；得到 {gap}")


# ---------- P.7 信息论
add = reg("p7-info", ["entropy", "cross_entropy", "mutual_information", "perplexity"], [
    "entropy：−Σ p log p，约定 0·log 0 = 0（先把 p=0 的项去掉）；base 决定单位（2 是比特）。",
    "cross_entropy(p, q)：−Σ p log q；它等于 H(p) + KL(p‖q)。",
    "mutual_information(joint)：I = Σ_{x,y} p(x,y) log( p(x,y) / (p(x)p(y)) )，边缘分布由行和、列和得到。",
    "perplexity(token_logprobs)：exp(−平均对数概率)，自然对数。",
])


@add
def check_entropy(ns):
    assert abs(ns.entropy(np.array([0.5, 0.5]), 2) - 1) < 1e-12, "公平硬币的熵是 1 比特"
    assert abs(ns.entropy(np.array([1.0, 0.0]), 2)) < 1e-12, "确定事件的熵是 0（0·log0 = 0）"
    assert abs(ns.entropy(np.ones(8) / 8, 2) - 3) < 1e-12, "8 个等概率结果的熵是 3 比特"
    assert abs(ns.entropy(np.array([0.5, 0.5]), math.e) - math.log(2)) < 1e-12, "以 e 为底时单位是 nats"


@add
def check_cross_entropy(ns):
    p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    kl = (p * np.log2(p / q)).sum()
    assert abs(ns.cross_entropy(p, q, 2) - (ns.entropy(p, 2) + kl)) < 1e-12, "交叉熵 = 熵 + KL"
    assert abs(ns.cross_entropy(p, p, 2) - ns.entropy(p, 2)) < 1e-12, "q = p 时交叉熵等于熵"


@add
def check_mutual_information(ns):
    indep = np.outer([0.3, 0.7], [0.4, 0.6])
    assert abs(ns.mutual_information(indep, 2)) < 1e-12, "独立时互信息为 0"
    same = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert abs(ns.mutual_information(same, 2) - 1) < 1e-12, "X = Y 且是公平硬币时互信息为 1 比特"
    j = np.array([[0.1, 0.2, 0.05], [0.25, 0.1, 0.3]])
    assert abs(ns.mutual_information(j, 2) - ns.mutual_information(j.T, 2)) < 1e-12, "互信息对称"
    assert ns.mutual_information(j, 2) > 0, "相关时互信息为正"


@add
def check_perplexity(ns):
    V = 50
    assert abs(ns.perplexity(np.full(20, -math.log(V))) - V) < 1e-9, "均匀分布在 V 个词上的困惑度就是 V"
    assert abs(ns.perplexity(np.zeros(5)) - 1) < 1e-12, "每个 token 概率都是 1 时困惑度为 1"


# ---------- P.8 采样
add = reg("p8-sampling", ["sample_exponential", "rejection_sample", "importance_estimate"], [
    "逆变换采样：若 U ~ Uniform(0,1)，则 F⁻¹(U) 服从分布 F。指数分布 F(x) = 1 − e^{−λx}，反解得到 x。",
    "拒绝采样：从提议分布抽 x、再抽 u ~ U(0,1)，若 u ≤ p(x)/(M·q(x)) 就接受；直到攒够 n 个。返回 (样本, 接受率)。",
    "重要性采样：E_p[f] ≈ mean(f(x)·w)，w = p(x)/q(x)，x 来自 q。自归一化版本用 Σ f·w / Σ w，p 只需已知到相差一个常数。",
])


@add
def check_exponential(ns):
    rng = np.random.default_rng(8)
    x = _a(ns.sample_exponential(2.0, 200000, rng))
    assert x.shape == (200000,) and (x >= 0).all(), "应返回 n 个非负样本"
    assert abs(x.mean() - 0.5) < 0.005 and abs(x.var() - 0.25) < 0.01, f"Exp(λ=2) 的均值 1/λ=0.5、方差 1/λ²=0.25，得到 {x.mean():.4f}, {x.var():.4f}"
    assert abs((x > 1.0).mean() - math.exp(-2.0)) < 0.004, "P(X>1) 应为 e^{−2}"


@add
def check_rejection(ns):
    rng = np.random.default_rng(9)
    target = lambda x: 6 * x * (1 - x)                        # Beta(2,2)
    samples, rate = ns.rejection_sample(target, lambda n, r: r.random(n), lambda x: np.ones_like(x), 1.5, 50000, rng)
    s = _a(samples)
    assert s.shape == (50000,), "应返回恰好 n 个被接受的样本"
    assert abs(s.mean() - 0.5) < 0.005 and abs(s.var() - 0.05) < 0.003, f"Beta(2,2) 的均值 0.5、方差 0.05，得到 {s.mean():.4f}, {s.var():.4f}"
    assert abs(rate - 1 / 1.5) < 0.01, f"接受率应约为 1/M = 0.667，得到 {rate:.3f}"


@add
def check_importance(ns):
    rng = np.random.default_rng(10)
    x = rng.normal(0, 2, size=200000)
    p = lambda t: np.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi)
    q = lambda t: np.exp(-t ** 2 / 8) / math.sqrt(8 * math.pi)
    f = lambda t: t ** 2
    est = ns.importance_estimate(f, x, p, q, False)
    assert abs(est - 1) < 0.02, f"用 N(0,2²) 的样本估计 N(0,1) 下的 E[x²]=1，得到 {est:.4f}"
    unnorm = lambda t: 5 * p(t)
    assert abs(ns.importance_estimate(f, x, unnorm, q, False) - 5) < 0.1, "p 乘了常数 5，普通重要性采样的结果也会乘 5"
    assert abs(ns.importance_estimate(f, x, unnorm, q, True) - 1) < 0.02, "自归一化版本应不受常数影响"


# ============================================================ B 数据加载（PyTorch，题面来自 v3/data-loading）
add = reg("b-streaming", ["StreamingDataset"], [
    "__iter__ 写成生成器：用 with open(...) 打开文件，for line in f 逐行读，每行 yield 一次，不要先 readlines()。",
    "每行是逗号分隔的浮点数，最后一列是 y：x = torch.tensor(values[:-1])，y = torch.tensor(values[-1])。",
    "多进程：worker_info = torch.utils.data.get_worker_info()；为 None 时是主进程；否则只处理 行号 % num_workers == worker_info.id 的行。",
])


def _csv(tmpdir, n=37, d=3):
    import os
    p = os.path.join(tmpdir, "data.csv")
    with open(p, "w") as f:
        for i in range(n):
            f.write(",".join(str(float(i * 10 + j)) for j in range(d)) + f",{float(i)}\n")
    return p


@add
def check_streaming_reads_every_row(ns):
    import tempfile, torch
    from torch.utils.data import IterableDataset
    with tempfile.TemporaryDirectory() as d:
        ds = ns.StreamingDataset(_csv(d))
        assert isinstance(ds, IterableDataset), "StreamingDataset 应继承 torch.utils.data.IterableDataset"
        rows = list(ds)
        assert len(rows) == 37, f"文件有 37 行，读到 {len(rows)} 行"
        x, y = rows[5]
        assert tuple(torch.as_tensor(x).shape) == (3,) and float(y) == 5.0, "每行应是 (x: 3 维张量, y: 标量)，最后一列是 y"
        assert torch.allclose(torch.as_tensor(x), torch.tensor([50.0, 51.0, 52.0])), "x 的取值不对"
        assert [float(r[1]) for r in rows] == [float(i) for i in range(37)], "单进程时应按文件顺序读出"
        assert len(list(ds)) == 37, "第二个 epoch 再迭代一次也应读出全部 37 行"


@add
def check_streaming_is_lazy(ns):
    import inspect
    it = ns.StreamingDataset.__iter__
    assert inspect.isgeneratorfunction(it), "__iter__ 应是一个生成器函数（用 yield），这样才是逐行读取，而不是先把整个文件读进内存"


@add
def check_streaming_multiworker_no_duplicates(ns):
    import tempfile
    from torch.utils.data import DataLoader
    with tempfile.TemporaryDirectory() as d:
        ds = ns.StreamingDataset(_csv(d))
        ys = [float(y) for _, y in DataLoader(ds, batch_size=None, num_workers=2)]
    assert sorted(ys) == [float(i) for i in range(37)], (
        f"num_workers=2 时读出了 {len(ys)} 行（应为 37 且不重复）：每个 worker 都会拿到数据集的一份拷贝，"
        "要用 get_worker_info() 让每个 worker 只处理属于自己的那部分行")


add = reg("b-shuffle-buffer", ["ShuffleBufferDataset"], [
    "先从底层迭代器取 buffer_size 个样本放进 buffer。",
    "之后每来一个新样本：随机选 buffer 里的一个位置 i，yield buffer[i]，再把新样本放到 buffer[i]。",
    "底层迭代器耗尽后，把 buffer 剩下的样本随机打乱后全部 yield 出去。",
])


class _Seq:
    def __init__(self, n):
        self.n, self.taken = n, 0

    def __iter__(self):
        self.taken = 0
        for i in range(self.n):
            self.taken += 1
            yield i


def _seq_ds(n):
    from torch.utils.data import IterableDataset

    class S(IterableDataset):
        def __init__(self):
            super().__init__()
            self.inner = _Seq(n)

        def __iter__(self):
            return iter(self.inner)
    return S()


@add
def check_buffer_complete_and_shuffled(ns):
    import random, torch
    random.seed(0); torch.manual_seed(0)
    out = [int(v) for v in ns.ShuffleBufferDataset(_seq_ds(500), 50)]
    assert sorted(out) == list(range(500)), f"每个样本应恰好出现一次，得到 {len(out)} 个、去重后 {len(set(out))} 个"
    assert out != list(range(500)), "输出顺序与输入完全相同：没有打乱"
    one = [int(v) for v in ns.ShuffleBufferDataset(_seq_ds(50), 1)]
    assert one == list(range(50)), "buffer_size=1 时没有可以选择的余地，输出顺序应与输入相同"


@add
def check_buffer_is_local(ns):
    import random, torch
    random.seed(1); torch.manual_seed(1)
    B = 20
    out = [int(v) for v in ns.ShuffleBufferDataset(_seq_ds(400), B)]
    early = [(p, i) for p, i in enumerate(out) if p < i - (B - 1)]
    assert not early, (f"样本 {early[0][1]} 出现在第 {early[0][0]} 个位置：buffer 里最多只有前 p+{B} 个样本，"
                       "一个样本不可能被提前这么多。是不是先把整个数据集读进来再打乱了？")


@add
def check_buffer_memory_bound(ns):
    import random, torch
    random.seed(2); torch.manual_seed(2)
    base = _seq_ds(300)
    it = iter(ns.ShuffleBufferDataset(base, 25))
    worst = 0
    for k, _ in enumerate(it, start=1):
        worst = max(worst, base.inner.taken - k)
    assert worst <= 25, f"某一时刻已读入但尚未输出的样本多达 {worst} 个，超过了 buffer_size=25：内存应当是 O(buffer_size)"


add = reg("b-dist-sampler", ["SimpleDistributedSampler"], [
    "num_samples = ceil(N / num_replicas)，total_size = num_samples × num_replicas。",
    "shuffle=True 时：g = torch.Generator(); g.manual_seed(seed + epoch); indices = torch.randperm(N, generator=g).tolist()；否则 indices = list(range(N))。",
    "补齐：indices += indices[:total_size − N]；再按 indices[rank::num_replicas] 取本 rank 的部分。",
    "所有 rank 必须用同一个随机排列（同样的 seed + epoch），否则各 rank 拿到的样本会重叠或遗漏。",
])


def _ds(n):
    class D:
        def __len__(self):
            return n

        def __getitem__(self, i):
            return i
    return D()


@add
def check_sampler_no_shuffle_layout(ns):
    got = [list(ns.SimpleDistributedSampler(_ds(10), 4, r, shuffle=False)) for r in range(4)]
    assert got == [[0, 4, 8], [1, 5, 9], [2, 6, 0], [3, 7, 1]], (
        f"N=10、4 个 rank、不打乱时，应先补齐成 [0..9, 0, 1]，再按 rank::4 交错取：得到 {got}")
    assert all(len(ns.SimpleDistributedSampler(_ds(10), 4, r, shuffle=False)) == 3 for r in range(4)), "__len__ 应为 ceil(10/4) = 3"


@add
def check_sampler_matches_pytorch(ns):
    from torch.utils.data import DistributedSampler
    for N, R in ((103, 4), (64, 8), (7, 3)):
        for epoch in (0, 3):
            for r in range(R):
                mine = ns.SimpleDistributedSampler(_ds(N), R, r, shuffle=True, seed=5)
                mine.set_epoch(epoch)
                ref = DistributedSampler(_ds(N), num_replicas=R, rank=r, shuffle=True, seed=5)
                ref.set_epoch(epoch)
                assert list(mine) == list(ref), (
                    f"N={N}、{R} 个 rank、rank={r}、epoch={epoch} 时与 torch.utils.data.DistributedSampler 不一致：")
                assert len(mine) == len(ref)


@add
def check_sampler_epochs(ns):
    s = ns.SimpleDistributedSampler(_ds(50), 2, 0, shuffle=True, seed=0)
    s.set_epoch(1); a = list(s)
    s.set_epoch(1); b = list(s)
    s.set_epoch(2); c = list(s)
    assert a == b, "同一个 seed + epoch 应得到相同的顺序"
    assert a != c, "换了 epoch 应得到不同的顺序（否则每个 epoch 数据顺序都一样）"
    union = set()
    for r in range(2):
        t = ns.SimpleDistributedSampler(_ds(50), 2, r, shuffle=True, seed=0); t.set_epoch(4)
        part = list(t)
        assert not (union & set(part)), "N 能被 rank 数整除时，各 rank 的样本不应重叠"
        union |= set(part)
    assert union == set(range(50)), "所有 rank 合起来应覆盖全部样本"
