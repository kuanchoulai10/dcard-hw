# Exploratory Data Analysis (EDA)

When we first receive a dataset, the initial step is to examine its details, including the number of records and columns in each table. Below is the dataset information as of the update on 2020/04/13:

```
posts_train                Contains 793,751 records and 3 columns
post_shared_train          Contains 304,260 records and 3 columns
post_comment_created_train Contains 2,372,228 records and 3 columns
post_liked_train           Contains 3,395,903 records and 3 columns
post_collected_train       Contains 1,235,126 records and 3 columns
```

```
posts_test                 Contains 225,986 records and 3 columns
post_shared_test           Contains 83,376 records and 3 columns
post_comment_created_test  Contains 607,251 records and 3 columns
post_liked_test            Contains 908,910 records and 3 columns
post_collected_test        Contains 275,073 records and 3 columns
```

The dataset is divided into training and testing sets. To avoid data leakage, only the training set is analyzed during EDA, leaving the testing set aside.

The training set covers posts from April 1, 2019, to the end of October 2019, spanning approximately seven months with around 793,000 posts. The goal is to build a predictive model that uses 10-hour post metrics (e.g., shares, comments, likes, and saves) to predict whether a post will receive 1,000 likes within 36 hours, classifying it as a "popular post."

Approximately 2.32% of the training posts are popular, equating to about 18,000 posts. This imbalance in the dataset necessitates techniques like over/undersampling during preprocessing and alternative evaluation metrics during model assessment.

## Problem Definition

The task can be approached in four ways, based on whether sequence information is considered and whether the problem is framed as regression or binary classification:

|                  | Regression | Binary Classification |
|:----------------:|------------|-----------------------|
| **With Sequence Info** | RNNs (e.g., GRU), traditional time series models (e.g., ARMA, ARIMA) | Same as left |
| **Without Sequence Info** | Poisson regression, SVM, tree-based models, etc. | Logistic regression, SVM, tree-based models, etc. |

The ideal approach is "with sequence info" and "regression," leveraging the 10-hour metrics' trends to predict the 36-hour total likes, then classifying posts based on a threshold (e.g., ≥1,000 likes). However, for simplicity and time constraints, we focus on "without sequence info" and "binary classification," aggregating 10-hour metrics and building a binary classification model to predict popular posts. The focus will be on handling imbalanced data, tree-based models, and subsequent discussions.

## Linear Relationships Between Variables

We simplify the dataset to include total shares, comments, likes, and saves within 10 hours and use a heatmap to observe their relationships with the total likes within 36 hours:

![](https://i.imgur.com/xvxqb6z.png)

!!! info

    Key observations from the heatmap:

    - Total likes within 36 hours moderately correlate with total likes within 10 hours (.58), shares (.36), and saves (.36), but weakly with comments (.17).
    - Total likes within 10 hours moderately correlate with shares (.63) and saves (.61).
    - Shares and saves within 10 hours moderately correlate (.48).

In simple terms, posts with more likes within 10 hours tend to have more shares and saves. However, the strongest predictor of total likes within 36 hours is the likes within 10 hours. Comments show little correlation with total likes.

## Heatmaps of Key Metrics

!!! danger

    To protect Dcard's proprietary information, color bars (`cbar=False`) are omitted, showing only relative relationships.

### Total Posts by Time

We examine whether the number of posts varies across different time periods:

![](https://i.imgur.com/LaPJRkL.png)

The x-axis represents 24 hours, and the y-axis represents days of the week.

!!! info

    Observations:

    - Posts are concentrated during midday, afternoon, and evening (12:00–18:00), with weekdays slightly higher than weekends.
    - The second-highest posting period is weekday mornings (05:00–12:00).
    - Posts are relatively fewer during evenings (18:00–01:00) on both weekdays and weekends.

These trends are reasonable, as students primarily post during the day. The relatively high number of early morning posts might be due to companies posting content before students wake up.

### Popular Post Proportion by Time

Next, we analyze whether certain time periods have a higher proportion of popular posts:

![](https://i.imgur.com/89isGTC.png)

!!! info

    Observations:

    - Posts during late-night and early-morning hours on weekends have a higher likelihood of being popular, likely due to increased user activity during these times.
    - The heatmap confirms that the proportion of popular posts varies by time.

### Average Likes Within 10 Hours by Time

We then examine the average likes within 10 hours for posts made at different times:

![](https://i.imgur.com/62gyK11.png)

!!! info

    Observations:

    - Posts made between 21:00–11:00 generally receive more likes within 10 hours.
    - Posts made between 11:00–21:00, especially during late afternoon and dinner hours, receive fewer likes on average.

This difference might be because students are less active during late afternoon and dinner hours but more active during the evening. Early morning posts are also visible to students the next day.

### Average Likes Within 36 Hours by Time

Finally, we analyze the average likes within 36 hours for posts made at different times:

![](https://i.imgur.com/1gm3nvR.png)

The trends are consistent with the 10-hour analysis and are not elaborated further.

### Summary

!!! info

    Key takeaways:

    - Variables are generally highly correlated. Polynomial transformations (`PolynomialFeatures`) may not yield significant improvements during feature engineering.
    - Posting time significantly impacts the proportion of popular posts and the number of likes, and this information should be incorporated into the model.
