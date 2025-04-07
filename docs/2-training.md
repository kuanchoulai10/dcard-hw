# 特徵工程
經過探索性資料分析後，我們對訓練集又更認識了些。在進行特徵工程前，我們會先將我們的訓練集初步整理成以下格式：
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 793751 entries, 0 to 793750
Data columns (total 8 columns):
post_key           793751 non-null object
shared_count       793751 non-null int64
comment_count      793751 non-null int64
like_count         793751 non-null int64
collected_count    793751 non-null int64
weekday            793751 non-null int64
hour               793751 non-null int64
is_trending        793751 non-null int64
dtypes: bool(1), int64(6), object(1)
memory usage: 43.1+ MB
```

而在本節裡我們會談到整個 data pipeline 過程將會使用到的技術和模型，包括了 over/undersampling，多項式轉換、One-hot 編碼和 Tree-based 模型。

!!! info

    在訓練過程中，主要可分為三個階段：

    1. 重抽樣（resampling）
    2. 欄位轉換（column transformation）
    3. 分類（classification）

實際上，我們可將上述的三個階段表示為以下的 `Pipeline` 實體：

```python
cachedir = mkdtemp()
pipe = Pipeline(steps=[('resampler', 'passthrough'),
                       # ('columntransformer', 'passthrough'),
                       ('classifier', 'passthrough')],
                memory=cachedir)
```
在每個階段我們都會選擇兩到三種不同作法和幾組超參數設定，試圖找出最佳組合。

## 不平衡資料集的處理（STAGE 1）
首先，考慮一個二元分類問題，不平衡資料集指的是分類目標（$y$）絕大多數屬於某種類別（稱為 majority），僅有少部分屬於另種類別（稱為 minority）。

面對不平衡資料集時，若是不進行任何處理就直接進行訓練，則很有可能產出一個帶有偏見的（biased）模型，不分青紅皂白地預測絕大部分樣本為 majority 類別，而忽略了 minority 樣本的有用資訊。

一種可能的解決辦法就是重抽樣（resampling），又可分為 oversampling 和 undersampling 兩種：

- Oversampling 是增加 minority 樣本在資料集的比例。
- Undersampling 則是減少 majority 樣本在資料集的比例。

兩種方法都能使得模型在訓練階段時，更聽取 minority 樣本的意見。最直觀的作法就是透過「隨機抽樣」，刪去 majority 樣本或增加 minority 樣本。

實際上，`imblearn` 套件正提供了 `RandomOverSampler` 和 `RandomUnderSampler` 供我們使用。不僅如此，它也提供了其它的重抽樣方法的實作，其中我們會使用到 `SMOTE` 和 `NearMiss` 兩種。以下是簡介：

### SMOTE
SMOTE（Synthetic Minority Oversampling Technique）是 oversampling 技術的一種，它會在 minority 樣本之間合成新的 minority 樣本，從而增加 minority 類別的比例。示意圖如下所示：

![](https://taweihuang.hpd.io/wp-content/uploads/2018/12/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7-2019-01-06-%E4%B8%8B%E5%8D%8810%E3%80%8223%E3%80%8257.png)
[圖片來源](https://taweihuang.hpd.io/2018/12/30/imbalanced-data-sampling-techniques/)

### NearMiss
NearMiss 則是 undersampling 技術的一種，共有三種版本，其中我們只提第一版。NearMiss-1 會計算所有 majority 樣本前 $k$ 個 minority nearest neighbors 的平均距離，刪去那些最靠近 minority 樣本的 majority 樣本，直到兩種類別的比例為 1：1。示意圖如下：
<img src="https://imbalanced-learn.readthedocs.io/en/stable/_images/sphx_glr_plot_illustration_nearmiss_0011.png" height="350">
[圖片來源](https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html#mathematical-formulation)

## 多項式轉換和 One-hot 編碼（STAGE 2）
接著，我們透過 `sklearn` 的 `PolynomialFeatures` 和 `OneHotEncoder`，替不同欄位做特徵轉換：

- 針對 `shared_count`, `comment_count`, `liked_count`, `collected_count` 四個特徵，我們會進行二次方的多項式轉換，試圖捕捉特徵的非線性關係和彼此之間的交互作用。
- 針對 `weekday` 特徵，會從整數型態（`0` - `6`，依序代表星期一至星期六）轉換成 one-hot 格式，例如 `[1, 0, 0, 0, 0, 0, 0]` 代表的是星期一。

## Tree-based 集成模型（STAGE 3）
最後，分類模型我們主要以 tree-based 的集成（ensemble）模型作為我們的主要選擇．這當中包括了 `AdaBoostClassifier`, `GradientBoostingClassifier` 和 `XGBClassifier`。主要有幾點考量：

- 對特徵的單調性轉換（monotonic transformation）具有不變性（invariant），因此可以減少許多特徵轉換的工作。
- 也因為減少許多特徵轉換，因此模型的可解釋性高，對特徵的理解也相對好理解。
- 面對大型且複雜的資料集時，模型成效好，時常是 Kaggle 的常勝軍（特指 `XGBoost`, `LightGBM`, `CatBoost` 套件）。

集成學習（ensemble learning）主要可分為兩種：Bagging（bootstrap aggregating）和 Boosting。

### Bagging
Bagging 最有名的一種應用就是隨機森林（random forest），模型內會建立多顆決策樹，並透過自助式抽樣法（bootstrap sampling）和隨機挑選某幾個特徵欄位，讓每顆樹只學習到局部特徵，最後整合所有決策樹的觀點，做出最後預測結果。

### Boosting
#### Adaptive Boosting
Boosting 的應用相對 Bagging 則多了許多，最初期的就是 Adaptive boosting（AdaBoost），它的核心概念是依序建立 $T$ 個弱學習器（weak learner）$h_t(x)$，每個模型會更關注在前個模型分類錯誤的那些樣本上。不僅如此，每個模型都將會被賦予一個模型權重 $\alpha_t$，該權重必須反映兩件事：

- 模型權重越高，代表該模型效果較好。
- 模型權重越低，代表該模型效果較差。

如此一來，最終模型 $H(x)$ 就是蒐集這 $T$ 個弱學習器（weak learner）的觀點，做出最終分類。詳細內容可參考我過去所做的這篇[筆記](https://hackmd.io/@kcl10/HyXNoqOL8)。

#### Gradient Boosting
梯度提升技術通常會與決策樹搭配使用，它的核心概念是依序建立 $T$ 個模型 $h_t(x)$，每個階段的模型都是在預測前個階段所有樣本的梯度方向（虛擬殘差）。最終模型 $H(x)$ 就是將前幾個模型加總起來（additive model），做出最終分類。詳細數學式推導可參考我過去所做的這篇[筆記](https://hackmd.io/@kcl10/B1GKRg9L8)。

#### Extreme Gradient Boosting
XGBoost 是基於梯度提升技術的一個套件，其中又做了相當多的優化，包括 weighted quantile sketch, parallel learning, cache-aware access 等。詳細內容可參考這篇[論文](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)。




## 超參數優化（Hyperparameter Optimization）
大致上瞭解訓練過程將會使用的技術後，我們要開始設定每個階段可能的作法及其超參數組合。整理如下：

**Resampler**
- `passthrough`：不做任何重抽樣。
- `NearMiss`：使用預設參數。
- `SMOTE`：使用預設參數。

**Column Transformer**
- `passthrough`：不做任何特徵轉換。
- `col_trans`：針對不同欄位進行二項式轉換和 one-hot 編碼。

**Classifier**
- `AdaBoostClassifier`：使用預設參數，其中又會設定樹深限制在 `[1, 2, 3]` 層。
- `GradientBoostingClassifier`, `XGBClassifier`：使用預設參數，其中又會設定學習率在 `[0.025, 0.05, 0.1]`。

不論哪個分類器，我們都會設定其內部有 `[90, 100, 110 , 120]` 棵決策樹。並且另外設定細部的超參數：

三個階段交叉下來共有 216 種組合需要嘗試，個數有點過多，我們時間並不夠。實驗下來，經過特徵轉換的模型普遍效果不好，初步推估是因為前面 EDA 所提到的，變數之間的相關性普遍偏高，經過二次方多項式轉換其實成效並不高，甚至有可能降低成效。

因此，我們決定省略了「特徵轉換」階段，最終共有 **108** 種組合需要嘗試，透過 `GridSearchCV` 找尋最佳組合，設定 `cv=3`。

參數組合的程式碼參考如下：
```python
# poly_cols = ['shared_count', 'comment_count', 'liked_count', 'collected_count']
# col_trans = make_column_transformer((OneHotEncoder(dtype='int'), ['weekday']),
#                                     (PolynomialFeatures(include_bias=False), poly_cols),
#                                     remainder='passthrough')
param_grid_ada = {
    'resampler': ['passthrough', SMOTE(), NearMiss()],
    # 'columntransformer': ['passthrough', col_trans],
    'classifier': [AdaBoostClassifier()],
    'classifier__n_estimators': [90, 100, 110, 120],
    'classifier__base_estimator': [DecisionTreeClassifier(max_depth=1), 
                                   DecisionTreeClassifier(max_depth=2),
                                   DecisionTreeClassifier(max_depth=3)]
}
param_grid_gb = {
    'resampler': ['passthrough', SMOTE(), NearMiss()],
    # 'columntransformer': ['passthrough', col_trans],
    'classifier': [GradientBoostingClassifier(), XGBClassifier()],
    'classifier__n_estimators': [90, 100, 110, 120],
    'classifier__learning_rate': [0.025, 0.05, 0.1]
}
param_grid = [param_grid_ada, param_grid_gb]
```
