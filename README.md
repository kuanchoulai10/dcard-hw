# 2020 Dcard Data Engineering Intern

Dcard is a popular social media platform in Taiwan, especially among college students and young adults. It was launched in 2011 as a university-only online forum, similar in spirit to how Facebook started within universities.

This project is a pre-interview assignment for the 2020 Dcard Data Engineer Internship Program.

On Dcard's app and website, there is an important section called "Trending Posts," where users can find the hottest discussion topics on the platform. **As data enthusiasts, we are also curious about which posts have the potential to become trending**. If we consider this factor in our recommendations, we might help users discover great posts faster. Therefore, in this assignment, **we aim to predict whether a post has the potential to appear in the "Trending Posts" section based on some data.**

To simplify the problem, **we define a trending post as one that receives at least 1000 likes within 36 hours of being posted.** During testing, we will calculate whether a post's like count exceeds 1000 within 36 hours to determine the ground truth or prediction benchmark.

!!! abstract

    ```shell
    $ tree
    .
    ├── eda_evaluation.ipynb: A Jupyter notebook used for generating visualizations.
    ├── outputs
    │   ├── best_model.h5: The best model obtained after training.
    │   ├── cv_results.csv: Cross-validation results.
    │   └── output.csv: Prediction results for the public dataset.
    ├── predict.py: A utility script for making predictions.
    ├── preprocessing.py: A shared utility script for database connections, preprocessing, and other common functions.
    ├── requirements.txt: A list of required Python packages and their versions.
    └── training.py: A utility script for training the model.
    ```

    The training dataset includes articles spanning from April 1, 2019, to the end of October 2019, covering approximately seven months. The dataset contains around 793,000 articles, of which about 2.32% (approximately 18,000 articles) are classified as popular. Through exploratory data analysis, we observed high correlations among variables. Additionally, the timing of article publication significantly influences the proportion of popular articles and the total number of likes within the first 36 hours of posting.

    We decided to use a "binary classification model without considering sequential information" as our primary approach, focusing on handling imbalanced datasets, tree-based ensemble models, and subsequent discussions. The training process was divided into three main stages:

    1. Resampling
    2. Column Transformation
    3. Classification

    After experimentation, we opted to omit the "feature transformation" stage. In total, 108 combinations were tested using `GridSearchCV` with `cv=3` to find the optimal configuration.

    Using the f1-score as the evaluation metric, the best-performing model was an `AdaBoostClassifier` without any resampling. This model consisted of 100 decision trees, each limited to a depth of 2. The average f1-score from cross-validation was 0.56, while the f1-score on the public test set was 0.53. Key findings from the experiments include:
    
    - Different resampling strategies significantly impact the f1-score.
    - Resampling strategies can effectively identify genuinely popular articles. However, this comes at the cost of reduced trust in the model's predictions of popular articles.
    - Under both "SMOTE resampling" and "no resampling" scenarios, the choice of classifier did not lead to substantial changes in the f1-score.
    - The choice of classifier had a relatively minor impact on the f1-score.

    Finally, we discussed several potential future directions, including exploring other resampling techniques, alternative evaluation metrics, and incorporating sequential information.

## Training Dataset

The training dataset covers posts from April 1, 2019, to the end of October 2019, approximately 7 months. It contains around 794,000 posts, of which about 2.32% (approximately 18,000 posts) are trending.

```
posts_train                Contains 793,751 records and 3 columns
post_shared_train          Contains 304,260 records and 3 columns
post_comment_created_train Contains 2,372,228 records and 3 columns
post_liked_train           Contains 3,395,903 records and 3 columns
post_collected_train       Contains 1,235,126 records and 3 columns
```

### Table: `posts_train`

| column_name        | data_type | description                                         |
|--------------------|-----------|-----------------------------------------------------|
| `post_key`         | string    | Unique identifier of the post                      |
| `created_at_hour`  | datetime  | The hour when the post was created                 |
| `like_count_36_hour` | integer | Number of likes the post received within 36 hours (only in train table) |

### Table: `post_shared_train`

| column_name        | data_type | description                                         |
|--------------------|-----------|-----------------------------------------------------|
| `post_key`         | string    | Unique identifier of the post                      |
| `created_at_hour`  | datetime  | The hour of the sharing activity                   |
| `count`            | integer   | Number of shares the post received in that hour    |


### Table: `post_comment_created_train`

| column_name        | data_type | description                                         |
|--------------------|-----------|-----------------------------------------------------|
| `post_key`         | string    | Unique identifier of the post                      |
| `created_at_hour`  | datetime  | The hour of the comment activity                   |
| `count`            | integer   | Number of comments the post received in that hour  |


### Table: `post_liked_train`

| column_name        | data_type | description                                         |
|--------------------|-----------|-----------------------------------------------------|
| `post_key`         | string    | Unique identifier of the post                      |
| `created_at_hour`  | datetime  | The hour of the like activity                      |
| `count`            | integer   | Number of likes the post received in that hour     |


### Table: `post_collected_train`

| column_name        | data_type | description                                         |
|--------------------|-----------|-----------------------------------------------------|
| `post_key`         | string    | Unique identifier of the post                      |
| `created_at_hour`  | datetime  | The hour of the collection activity                |
| `count`            | integer   | Number of times the post was bookmarked in that hour |


## Testing Dataset

```
posts_test                 Contains 225,986 records and 3 columns
post_shared_test           Contains 83,376 records and 3 columns
post_comment_created_test  Contains 607,251 records and 3 columns
post_liked_test            Contains 908,910 records and 3 columns
post_collected_test        Contains 275,073 records and 3 columns
```

## Evaluation Metrics

For offline evaluation, only the first 10 hours of data for each post will be used as input for prediction. The primary evaluation metric is the F1-score.

## Submission Requirements

Upon completing the assignment, you must submit at least the following four files. Failure to include any of these will be considered incomplete.

1. Report.pdf
    - Instructions on how to use your code
    - Methods and rationale
    - Evaluation results on the provided testing data
    - Experimental observations
2. train.py
3. predict.py
4. requirements.txt or Pipfile
5. (Optional) If your prediction requires a model file, please include it (we will not train it for you) and explain how to use it in Report.pdf.

We have some requirements for the program structure to facilitate testing:
- Training
    - The outermost layer should be wrapped in train.py.
    - The program should be executable as `python train.py {database_host} {model_filepath}`.
    - Example: `python train.py localhost:8080 ./model.h5`

- Prediction
    - The program should be executable as `python predict.py {database_host} {model_filepath} {output_filepath}`.
    - Specify where your model_filepath is located.
    - Example: `python predict.py localhost:8080 ./model.h5 ./sample_output.csv`
    - Your program must achieve the following during prediction:
        - Read data from the database. The data format will match the tables described in the next section. For evaluation, we will use our own test data.
        - Use another database's xxx_test tables as the test set during actual testing. Your predict.py should use these tables as input.
        - Output a CSV file with two columns as shown below, including a header (refer to the provided sample_output.csv):
            - post_key: string type
            - is_trending: bool type

## Usage Instructions

Environment:

- Operating System: Ubuntu 18.04 LTS Desktop
- Python version: Python 3.6.8
- Required Python packages and their versions are listed in `requirements.txt`.

As mentioned in `quiz.pdf`, the submission format must include hardcoded parameters such as `user` and `password` for database connections. **To ensure clarity, the command-line argument format has been slightly modified to meet usage requirements.** Below are the usage instructions for `training.py` and `predict.py`.

### `training.py`

The usage of `training.py` is as follows:

```
usage: training.py [-h] -u USERNAME -p PASSWORD --host HOSTNAME
                   [--port PORTNUMBER] [--protocol PROTOCOL]
                   DATABASE OUTPUT_PATH
```

At a minimum, you must provide five parameters: "username," "password," "host IP address," "database name," and "output path." **To train on the training set, use the following command:**

```bash
python training.py -u "USERNAME"\
                   -p "PASSWORD"\
                   --host "HOSTNAME"\
                   "DATABASE"\
                   "OUTPUT_PATH"
```

By default, the program connects to a PostgreSQL database on port 5432. If needed, you can use the `--protocol` and `--port` options to connect to other databases, such as MySQL:

??? note

    ```bash
    python training.py -u "USERNAME"\
                    -p "PASSWORD"\
                    --host "HOSTNAME"\
                    --port "3306"\
                    --protocol "mysql"\
                    "DATABASE"\
                    "OUTPUT_PATH"
    ```

!!! danger

    After training, the program generates two files: "best model" and "cross-validation results." The default filenames are `best_model.h5` and `cv_results.csv` (these cannot be changed). Therefore, when specifying `OUTPUT_PATH`, only the folder name is required.

For more details, use the `-h` or `--help` options:

??? note

    ```shell
    $ python training.py -h
    usage: training.py [-h] -u USERNAME -p PASSWORD --host HOSTNAME
                    [--port PORTNUMBER] [--protocol PROTOCOL]
                    DATABASE OUTPUT_PATH

    positional arguments:
    DATABASE             (Required) Database to use when connecting to server.
    OUTPUT_PATH          (Required) Best prediction model and cross validation
                        results outputs file path.

    optional arguments:
    -h, --help           show this help message and exit
    -u USERNAME          (Required) User for login if not current user.
    -p PASSWORD          (Required) Password to use when connecting to server.
    --host HOSTNAME      (Required) Host address to connect.
    --port PORTNUMBER    Port number to use for connection (default: 5432)
    --protocol PROTOCOL  Protocol to connect. (default: postgres)
    ```

### `predict.py`

The usage of `predict.py` is as follows:

```
usage: predict.py [-h] -u USERNAME -p PASSWORD --host HOSTNAME
                  [--port PORTNUMBER] [--protocol PROTOCOL] [-n]
                  DATABASE MODEL_NAME OUTPUT_PATH
```

Similar to `training.py`, you must provide five parameters, with an additional parameter for the "model path" used to predict trending posts. **To predict on the public test set, use the following command:**

```bash
python predict.py -u "USERNAME"\
                  -p "PASSWORD"\
                  --host "HOSTNAME"\
                  "DATABASE"\
                  "MODELNAME"\
                  "OUTPUT_PATH"
```

After execution, the program outputs `output.csv` (filename cannot be changed) to the specified folder. Note that the `MODEL_NAME` option must include the model file name, not the folder path.

As mentioned in the "Assignment Supplementary Notes and Corrections" email, the `posts_test` table in the private test set does not include the `like_count_36_hour` column. Therefore, you must use the `-n` option to indicate that this column is absent. **To predict on the private test set, use the following command:**

```bash
python predict.py -u "USERNAME"\
                  -p "PASSWORD"\
                  --host "HOSTNAME"\
                  -n\
                  "DATABASE"\
                  "MODELNAME"\
                  "OUTPUT_PATH"
```

If needed, you can also use the `--port` and `--protocol` options to connect to other databases.

For more details, use the `-h` or `--help` options:

??? note

    ```shell
    $ python predict.py -h
    usage: predict.py [-h] -u USERNAME -p PASSWORD --host HOSTNAME
                    [--port PORTNUMBER] [--protocol PROTOCOL] [-n]
                    DATABASE MODEL_NAME OUTPUT_PATH

    positional arguments:
    DATABASE             (Required) Database to use when connecting to server.
    MODEL_NAME           (Required) Prediction model name. If it is not in the
                        current directory, please specify where it is.
    OUTPUT_PATH          (Required) File path of predicted results.

    optional arguments:
    -h, --help           show this help message and exit
    -u USERNAME          (Required) User for login if not current user.
    -p PASSWORD          (Required) Password to use when connecting to server.
    --host HOSTNAME      (Required) Host address to connect.
    --port PORTNUMBER    Port number to use for connection (default: 5432)
    --protocol PROTOCOL  Protocol to connect. (default: postgres)
    -n                   No like_count_36_hour column when the option is given.
    ```
