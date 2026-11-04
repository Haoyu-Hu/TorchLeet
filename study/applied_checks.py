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
