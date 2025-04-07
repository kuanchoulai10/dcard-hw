# Results and Discussion
## Evaluation Metrics
Before discussing the results, let us revisit some commonly used metrics for binary classification, explained using this assignment as an example:

| Actual＼Predicted |          Negative          |          Positive           |
|:-----------------:|:--------------------------:|:---------------------------:|
|   **Negative**    |  $\color{red}{\text{TN}}$  |  $\color{blue}{\text{FP}}$  |
|   **Positive**    | $\color{green}{\text{FN}}$ | $\color{orange}{\text{TP}}$ |

$\text{Precision}$: Measures the proportion of articles predicted as popular that are actually popular. **Higher values indicate greater trust in the model's predictions for popular articles.** Formula:
$$
\text{Precision} = \frac{\color{orange}{\text{TP}}}{\color{blue}{\text{FP}} + \color{orange}{\text{TP}}}
$$

$\text{Recall}$: Measures the proportion of actual popular articles that are correctly predicted by the model. Also known as True Positive Rate (TPR) or Sensitivity. **Higher values indicate the model's ability to capture actual popular articles.** Formula:
$$
\text{Recall} = \dfrac{\color{orange}{\text{TP}}}{\color{green}{\text{FN}} + \color{orange}{\text{TP}}}
$$ 

$\text{Specificity}$: Measures the proportion of actual non-popular articles that are correctly predicted by the model. Also known as True Negative Rate (TNR). **Higher values indicate the model's ability to capture actual non-popular articles.** Formula:
$$
\text{Specificity} = \dfrac{\color{red}{\text{TN}}}{\color{red}{\text{TN}}+\color{blue}{\text{FP}}}
$$

$\text{F1-score}$: A harmonic mean of $\text{Precision}$ and $\text{Recall}$, ranging from $0$ to $1$, with higher values being better. Formula:
$$
\text{F1-score} = \dfrac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

$\text{Balanced Acc.}$: A combined metric of $\text{TPR}$ and $\text{TNR}$, ranging from $0$ to $1$, with higher values being better. Formula:
$$
\text{Balanced Acc.} = \dfrac{\text{TNR} + \text{TPR}}{2}
$$

When using `GridSearchCV` to find the best parameter combination, we record these five metrics and select the best combination based on the f1-score. Example code:
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
## Experimental Results
!!! info

    The best model is `AdaBoostClassifier` without any resampling, consisting of 100 decision trees with a maximum depth of 2. The average f1-score during cross-validation is 0.56, while the f1-score on the public test set is 0.53. Detailed prediction information is as follows:

</br>

??? note

    ```
    ===================GETTING CONNECTOR START!==================
    ============================DONE!============================
    ====================GETTING TABLES START!====================
    posts_test                 Total:   225,986 rows, 3 columns
    post_shared_test           Total:    83,376 rows, 3 columns
    post_comment_created_test  Total:   607,251 rows, 3 columns
    post_liked_test            Total:   908,910 rows, 3 columns
    post_collected_test        Total:   275,073 rows, 3 columns
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

Now, let us analyze the experimental results. (All figures below are based on cross-validation results, not the entire training set or public test set.)

#### Resampler
First, let us examine how different resampling strategies affect the f1-score:
<img src="https://i.imgur.com/zEDEDf1.png" height="350">

!!! info

    Different resampling strategies indeed affect the f1-score:

    - NearMiss (undersampling) has the lowest f1-score, likely due to excessive removal of non-popular articles, losing too much majority class information.
    - SMOTE (oversampling) achieves a moderate f1-score.
    - No resampling achieves the highest f1-score.

Next, we investigate how these resampling strategies impact precision and recall:

<img src="https://i.imgur.com/r24hqmW.png" height="350">

!!! info

- NearMiss and SMOTE significantly increase the model's focus on the minority class, resulting in excellent recall scores of 0.91 and 0.95, respectively. However, this comes at the cost of precision, which drops to 0.07 and 0.20, respectively.
- In other words, **resampling strategies can capture actual popular articles but reduce the trustworthiness of the predicted popular articles.**

We further explore whether resampling strategies interact with different classifiers to influence the f1-score:

<img src="https://i.imgur.com/Ge3583B.png" height="350">

!!! info

    - Under "SMOTE" and "No Resampling" strategies, different classifiers do not significantly affect the f1-score.
    - However, under the NearMiss strategy, `XGBClassifier` achieves the highest f1-score (0.18), while `AdaBoostClassifier` has the lowest (0.07).
        - `AdaBoostClassifier` performs poorly because it relies on weak classifiers, which struggle with limited majority class information.
        - `XGBClassifier` outperforms `GradientBoostingClassifier` due to its optimized GBDT implementation.

#### Classifier
Next, let us examine how different classifiers affect the f1-score:

<img src="https://i.imgur.com/4mALiwb.png" height="350">

!!! info

    - Different classifiers have minimal impact on the f1-score. On average, `XGBClassifier` achieves the highest score (0.35), primarily due to its performance under the NearMiss strategy.

Finally, we analyze whether the number of internal classifiers in ensemble models affects the f1-score:

<img src="https://i.imgur.com/GTb5KJq.png" height="350">

Clearly, the number of classifiers has little impact. Similarly, the tree depth for `AdaBoostClassifier` and the learning rate for the other two models also have minimal impact on the f1-score (figures omitted).

## Future Directions
The experimental results are summarized above. Due to time constraints, additional attempts were not included. Potential future directions are outlined below:

#### Explore Other Resampling Techniques
Resampling techniques can increase the model's focus on the minority class. Although the experimental results were not ideal, we can continue fine-tuning hyperparameters or exploring other resampling techniques. Refer to the ["Over-sampling"](https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html) and ["Under-sampling"](https://imbalanced-learn.readthedocs.io/en/stable/under_sampling.html) sections of the `imblearn` User Guide for potential directions.

#### Consider Other Evaluation Metrics
The assignment requires using f1-score as the evaluation metric. However, if we use balanced accuracy instead, the best model would be a `GradientBoostingClassifier` trained with SMOTE, consisting of 120 classifiers and a learning rate of 0.1, achieving a balanced accuracy of 0.93.

The impact of different resampling strategies on balanced accuracy is shown below:

<img src="https://i.imgur.com/08MiKXf.png" height="350">

!!! info

    SMOTE achieves the highest balanced accuracy. If the goal is **to preliminarily identify potentially popular articles for subsequent workflows**, balanced accuracy might be a better evaluation metric.

#### Explore Other Feature Transformations and Classifiers
The experiment only considered tree-based ensemble models, which require minimal feature transformation. However, we could explore logistic regression, support vector machines, Poisson regression, etc., combined with effective feature transformations. For example, converting `weekday` and `hour` into circular coordinates (refer to [this post](https://stats.stackexchange.com/questions/245866/is-hour-of-day-a-categorical-variable)) could improve model performance.

#### Incorporate Sequential Information
The experiment ignored the "time trends" of shares, comments, likes, and collections within 10 hours of posting. One potential direction is to use recurrent neural networks (RNN, LSTM, GRU, etc.) to capture these trends and nonlinear relationships between variables.

A simple approach is to combine the four count variables into a 4-dimensional vector (e.g., `[4, 23, 17, 0]` for 4 shares, 23 comments, etc.), with a sequence length of 10. Each article's sequential information would then be a `(10, 4)` matrix, which can be fed into the model for training.

For details on LSTM models, refer to my [notes](https://hackmd.io/@kcl10/B1RoWCd0H).

#### Explore Other Hyperparameter Optimization Methods
The experiment used `GridSearchCV` for hyperparameter optimization. However, `RandomizedSearchCV` might be a better choice for optimizing a large number of hyperparameter combinations. Refer to this [2012 JMLR paper](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) for details.

Additionally, consider Bayesian optimization implementations provided by [`optuna`](https://optuna.org/) or [`hyperopt`](http://hyperopt.github.io/hyperopt/). Watch this [video](https://youtu.be/jtRPxRnOXnk) for details, and compare the two libraries in this [article](https://towardsdatascience.com/optuna-vs-hyperopt-which-hyperparameter-optimization-library-should-you-choose-ed8564618151).

#### Incorporate Text Data and User Behavior
The assignment does not include text data or user behavior. Since the ultimate goal is to "recommend articles more accurately to users," consider incorporating Latent Dirichlet Allocation (LDA) topic modeling to enrich article topic information. Refer to my [presentation](https://hackmd.io/@kcl10/topic_model) for details on LDA.

Additionally, combining user behavior data could enable more refined personalized text recommendations. Refer to this [video](https://www.youtube.com/watch?v=FkckgwMHP2s) and this [paper](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/ccd=r9lcr2/record?r1=1&h1=0) for details.

###### tags: `dcard`
