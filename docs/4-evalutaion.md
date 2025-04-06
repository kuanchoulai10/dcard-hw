# 結果與討論
## 衡量指標
討論結果之前，我們先回顧一些二元分類會使用的衡量指標，並以作業作為例子說明：

| Actual＼Predicted |          Negative          |          Positive           |
|:-----------------:|:--------------------------:|:---------------------------:|
|   **Negative**    |  $\color{red}{\text{TN}}$  |  $\color{blue}{\text{FP}}$  |
|   **Positive**    | $\color{green}{\text{FN}}$ | $\color{orange}{\text{TP}}$ |

$\text{Precision}$：衡量的是模型預測為熱門的那些文章當中，有多少比例實際上真的是熱門文章？**指標越大代表越能相信模型的熱門文章預測結果**。計算如下：
$$
\text{Precision} = \frac{\color{orange}{\text{TP}}}{\color{blue}{\text{FP}} + \color{orange}{\text{TP}}}
$$

$\text{Recall}$：衡量的是實際上是熱門的那些文章當中，有多少比例被模型預測出來？通常又被稱為 True Positive Rate（TPR）或 Sensitivity。**指標越大代表模型越能捕捉到實際上為熱門的文章**。計算如下：
$$
\text{Recall} = \dfrac{\color{orange}{\text{TP}}}{\color{green}{\text{FN}} + \color{orange}{\text{TP}}}
$$ 

$\text{Specificity}$：衡量的是實際上不是熱門的那些文章當中，有多少比例被模型預測出來？通常又被稱為 True Negative Rate（TNR）。**指標越大代表模型越能捕捉到實際上為非熱門的文章**。計算如下：
$$
\text{Specificity} = \dfrac{\color{red}{\text{TN}}}{\color{red}{\text{TN}}+\color{blue}{\text{FP}}}
$$

$\text{F1-score}$：$\text{Precision}$ 和 $\text{Recall}$ 的綜合性指標，介於 $0$ 到 $1$ 之間，越大越好。計算如下：
$$
\text{F1-score} = \dfrac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

$\text{Balanced Acc.}$： $\text{TPR}$ 和 $\text{TNR}$ 的綜合性指標，介於 $0$ 到 $1$ 之間，越大越好。計算如下。
$$
\text{Balanced Acc.} = \dfrac{\text{TNR} + \text{TPR}}{2}
$$

透過 `GridSearchCV` 搜尋最佳參數組合時，我們也會同時紀錄上述這五個指標，最終依據 f1-score 挑選最佳組合。程式碼參考如下：
```python
scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'specificity': make_scorer(specificity_score),
    'balanced_accuracy': 'balanced_accuracy',
    'f1_score': 'f1',
}
grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, refit='f1_score', 
                           n_jobs=-1, cv=3, return_train_score=True)
```
## 實驗結果
!!! info

    最佳模型是不做任何重抽樣的 `AdaBoostClassifier`，內部共有 100 棵樹深限制為 2 層的決策樹，交叉驗證的平均 f1-score 是 0.56。其在公開測試集的 f1-score 則是 0.53。預測的詳細資訊參考如下：

</br>

??? note

    ```
    ===================GETTING CONNECTOR START!==================
    ============================DONE!============================
    ====================GETTING TABLES START!====================
    posts_test                 總共有   225,986 筆資料和 3 個欄位
    post_shared_test           總共有    83,376 筆資料和 3 個欄位
    post_comment_created_test  總共有   607,251 筆資料和 3 個欄位
    post_liked_test            總共有   908,910 筆資料和 3 個欄位
    post_collected_test        總共有   275,073 筆資料和 3 個欄位
    ============================DONE!============================
    ====================MERGING TABLES START!====================
    ============================DONE!============================
    ================PREPROCESSING TOTAL_DF START!================
    ============================DONE!============================
    ==================PREDICTING TESTSET START!==================
    f1-score     = 0.53
    balanced acc = 0.70

                precision    recall  f1-score   support

            0       0.99      1.00      0.99    221479
            1       0.75      0.40      0.53      4507

        accuracy                           0.99    225986
    macro avg       0.87      0.70      0.76    225986
    weighted avg       0.98      0.99      0.98    225986

    ============================DONE!============================
    ```


接著我們就來看看實驗結果吧。（以下所有圖示皆為在交叉驗證時計算而來，並非整個訓練集或公開測試集上。）
#### Resampler
首先我們來看不同的重抽樣策略是否會對 f1-score 造成影響？
<img src="https://i.imgur.com/zEDEDf1.png" height="350">

!!! info

    不同的重抽樣策略確實會對 f1-score 造成影響：

    - NearMiss（undersampling）的 f1-score 最低：初步推估是因為刪除過多非熱門文章，損失太多 majority 類別的資訊。
    - SMOTE（oversampling）的 f1-score 居中。
    - 不做任何重抽樣的 f1-score 最高。

接著我們想要探究其原因，於是我們觀察不同的重抽樣策略到底對 precision 和 recall 造成什麼影響？

<img src="https://i.imgur.com/r24hqmW.png" height="350">

!!! info

- NearMiss 和 SMOTE 兩種重抽樣策略大大增加模型對 minority 類別的重視，因而在 recall 的表現上分別能有 0.91, 0.95 的良好表現。然而，取而代之的就是犧牲掉了 precision 的表現，分別只有 0.07 和 0.20。
- 換句話說，**重抽樣策略確實能捕捉到實際上為熱門的文章，但取而代之的就是我們不太能相信它所預測的熱門文章，實際上確實也是熱門文章。**

更進一步地，我們想觀察不同的重抽樣策略是否會與不同的分類器有交互作用，進而影響 f1-score？

<img src="https://i.imgur.com/Ge3583B.png" height="350">

!!! info

    - 在「SMOTE 重抽樣策略」以及「不做任何重抽樣」的情況底下，不同的分類器並不會造成 f1-score 大幅度的變化。
    - 然而，在 NearMiss 重抽樣策略的情況底下， `XGBClassifier` 的 f1-score 最高（0.18），而 `AdaBoostClassifier` 最低（0.07）。
        - `AdaBoostClassifier` 最低是因為它本身都是弱分類器，因此沒辦法在 majority 資訊過少的情況下做出有效的分類。
        - `XGBClassifier` 還勝過 `GradientBoostingClassifier` 是因為它是優化過後的 GBDT，理所當然毫不意外。

#### Classifier
另一方面，我們同樣也來看看不同的分類器是否會對 f1-score 造成影響？

<img src="https://i.imgur.com/4mALiwb.png" height="350">

!!! info

    - 不同的分類器對 f1-score 造成的影響並不大，平均而言 `XGBClassifier` 分數最高（0.35），源自於 NearMiss 重抽樣策略底下的分數較高所導致。

接著，我們來看看三種集成模型的內部分類器多寡是否會影響 f1-score？

<img src="https://i.imgur.com/GTb5KJq.png" height="350">

很明顯地，並不太會影響。不僅如此，若是進一步地觀察 `AdaBoostClassifier` 的樹深限制和另外兩種模型的學習率設定，同樣可以發現對 f1-score 的影響並不大（省略其圖示）。


# 後續發展
實驗結果大致如上，因時間考量並未加入其它嘗試。因此將未來可能方向統整於此：

#### 考慮其它重抽樣技術
重抽樣技術確實能增加模型對 minority 類別的重視，雖實驗結果不盡理想，但我們可以持續嘗試微調其超參數，甚至考慮其它重抽樣技術。可能的方向可參考 `imblearn` 套件的 User Guide 的 ["Over-sampling"](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html) 和 ["Under-sampling"](https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html)。

#### 考慮其它衡量指標
作業要求以 f1-score 作為模型衡量指標。然而，如果將 balanced accuracy 作為我們的模型衡量指標，最佳模型則會是透過 SMOTE 重抽樣策略來訓練，內部擁有 120 個分類器的、學習率為 0.1 的 `GradientBoostingClassifier` 模型，其 balanced accuracy 為 0.93。

以下是不同的重抽樣策略對 balanced accuracy 的影響：

<img src="https://i.imgur.com/08MiKXf.png" height="350">

!!! info

    可以發現 SMOTE 重抽樣策略的 balanced accuracy 最高。若我們模型的目的是**初步篩選那些可能為熱門的文章，送入後續的工作流**，那麼或許可以考慮 balanced accuracy 作為我們的模型衡量指標。

#### 考慮其它特徵轉換和分類器
實驗當中我們只考慮的 tree-based 的集成模型，它讓我們不需要做過多的特徵轉換即可建模。然而，我們可以往另個方向嘗試：考慮 logistic 迴歸、支持向量機、Poisson 迴歸等模型，加上操作得宜的特徵轉換，例如將`weekday` 和 `hour` 考慮轉換成循環的（circular）座標（可參考這篇[貼文](https://stats.stackexchange.com/questions/245866/is-hour-of-day-a-categorical-variable)的作法），進而提升模型成效。

#### 考慮序列資訊
實驗當中我們忽略了發文 10 小時內的分享數、評論數、愛心數和收藏數的「時間趨勢」。其中一種可能的方向是建立循環式神經網路（RNN, LSTM, GRU 等...），捕捉其「時間趨勢」和變數間的非線性關係。

最簡單的作法是將四種計次欄位整合成 4 維的向量（例如 `[4, 23, 17, 0]` 代表 4 個分享數、 23 個評論數，依此類推），序列長度為 10 。因此每一篇文章的時間序列資訊就是個 `(10, 4)` 的矩陣，接著送入模型訓練即可。

關於 LSTM 的模型細節可參考我過去所做的[筆記](https://hackmd.io/@kcl10/B1RoWCd0H)。

#### 考慮其它超參數優化方法
實驗當中我們是使用 `sklearn` 套件的 `GridSearchCV` 做超參數優化。然而，我們可以考慮另個 `RandomizedSearchCV`，在大量超參數組合需要優化時或許是個不錯的選擇，詳細內容可參考 2012 JMLR 的這篇[論文](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)。

不僅如此，甚至可以考慮 [`optuna`](https://optuna.org/) 或 [`hyperopt`](http://hyperopt.github.io/hyperopt/) 所提供的貝氏優化（bayesian optimization）實作。詳細內容可參考這部[影片](https://youtu.be/jtRPxRnOXnk)，兩個套件的比較可以參考這篇[文章](https://towardsdatascience.com/optuna-vs-hyperopt-which-hyperparameter-optimization-library-should-you-choose-ed8564618151)。


#### 考慮文本資料和使用者瀏覽行為
此次作業並未提供文本資料和使用者瀏覽行為，而找出潛藏熱門文章的真正目的是「更精準的推薦文章給使用者」。因此，可考慮加入 Latent Dirichlet Allocation（LDA） 主題模型，豐富一篇文章的主題資訊。關於 LDA 主題模型的細節可參考我過去所做的[簡報](https://hackmd.io/@kcl10/topic_model)。

不僅如此，我們甚至可以結合使用者瀏覽行為，做出更精緻的個人化文本推薦。詳細內容可參考這部[影片](https://www.youtube.com/watch?v=FkckgwMHP2s)和這篇[論文](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/ccd=r9lcr2/record?r1=1&h1=0)。


###### tags: `dcard`