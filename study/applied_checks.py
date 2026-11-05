"""应用 ML（CodeSignal 风格）题目的判题。只依赖 numpy / pandas，不需要 PyTorch。

每个检查是 fn(ns)，ns 是提交模块里的名字；assert 失败即未通过。
检查的是性质与手算结果，不对比标准实现，写法不同也能通过。
"""
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
