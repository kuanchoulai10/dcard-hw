# 2020 Dcard DE Intern

## 為什麼會有這份作業？

2020年Dcard數據工程師實習計畫，面試前的作業

在 Dcard 上,有一個很重要的版面叫熱門文章,每天會有許多的使用者會看這個列表來得知 Dcard 站上最火熱的討論話題是什
麼。但身為做資料的人,我們也很想知道哪些文章是有潛力上熱門的,如果我們在推薦的時候考量這個因素進來也許能夠更快的讓
使用者知道這是一篇好的文章。所以在這個作業裡面,我們希望能夠根據一些資料,來預測某一篇文章是不是有機會上到熱門文章區塊

為了簡化問題複雜度,我們目前訂為在文章發出的 36 小時內愛心數 >= 1000 就是熱門文章。實際測試的時候我們會去計算 36 小
時內的某篇文章的愛心數是否超過 1000 來當做答案或是預測的基準。


## 訓練資料集

訓練集的文章涵蓋日期範圍從 2019 年的 4 月 1 日開始，持續到同年 10 月底，共 7 個月左右。總篇數有 79.4 萬篇左右，其中約有 2.32% 的文章是熱門文章，約莫是 1.8 萬篇。

```
posts_train                總共有   793,751 筆資料和 3 個欄位
post_shared_train          總共有   304,260 筆資料和 3 個欄位
post_comment_created_train 總共有 2,372,228 筆資料和 3 個欄位
post_liked_train           總共有 3,395,903 筆資料和 3 個欄位
post_collected_train       總共有 1,235,126 筆資料和 3 個欄位
```

posts_train

- post_key
- created_at_hour
- like_count_36_hour

post_shared_train

- post_key
- created_at_hour
- count

post_comment_created_train

- post_key
- created_at_hour
- count

post_liked_train

- post_key
- created_at_hour
- count

post_collected_train

- post_key
- created_at_hour
- count

## 測試資料集

```
posts_test                 總共有   225,986 筆資料和 3 個欄位
post_shared_test           總共有    83,376 筆資料和 3 個欄位
post_comment_created_test  總共有   607,251 筆資料和 3 個欄位
post_liked_test            總共有   908,910 筆資料和 3 個欄位
post_collected_test        總共有   275,073 筆資料和 3 個欄位
```

## Evaluation metrics

在做 offline evaluation 的時候只會使用每篇文章前 10 小時的資料當作是預測資料
以 f1-score 為主

## 回傳檔案要求

在完成作業後,你回傳的作業內容必須至少包含下列的 1 - 4 這四個檔案,未包含視同未完成。

1. Report.pdf
    - 怎麼使用你們的 code
    - 方法以及為什麼要這樣做
    - Evaluate 在我們提供的 testing data 的結果
    - 實驗觀察
2. train.py
3. predict.py
4. requirement.txt 或 pipfile
5. (Optional) 如果 predict 需要 model file 請務必附上 (我們不會幫你 train),並在 Report.pdf 裡說明如何執行

我們對這個 Task 的程式會有一些要求,所有的程式的 package structure 請自行規劃。但希望繳交上來的程式可以符合下面的規
格以利測試:
- training
    - 最外層用 train.py 包著
    - 實際會執行 python train.py {database_host} {model_filepath}
    - example: python train.py localhost:8080 ./model.h5

- predict
    - predict.py
    - 實際會執行 python predict.py {database_host} {model_filepath} {output_filepath}
    - 請告訴我們你的 model_filepath 放在哪裡
    - example: python predict.py localhost:8080 ./model.h5 ./sample_output.csv
    - 你的程式最終在預測的時候要能夠做到下列兩件事情
        - 從資料庫讀資料,資料的格式跟下一個 section 說的一樣,最後做 judgement 的時候我們會用自己做的另外的資料做測試。
        - 實際上我們會取另一個資料庫的 xxx_test tables 當測試集。所以 predict.py 裡面請吃這些 table當成是你的程式的 input。
        - 輸出成 CSV 格式,裡面有兩個 column 如下,需要輸出 header (請參照我們附上的 sample_output.csv)
            - post_key: string type
            - is_trending: bool type

## 使用方法

環境：

- Operatig System: Uuntu 18.04 LTS Desktop
- Python version: Python 3.6.8
- Python 所需套件及其版本已整理至 `requirement.txt`。

由於 `quiz.pdf` 裡所提到的繳交格式必須把格式寫死，例如連線至資料庫時的 `user`, `password` 等。**為求慎重，命令列參數的傳遞格式有稍作修改，以符合使用需求**。以下是 `training.py`, `predict.py` 的使用方法。

### `training.py`

`training.py` 的使用方法簡介如下：

```
usage: training.py [-h] -u USERNAME -p PASSWORD --host HOSTNAME
                   [--port PORTNUMBER] [--protocol PROTOCOL]
                   DATABASE OUTPUT_PATH
```

最低需求必須輸入「使用者名稱」、「密碼」、「主機IP位址」、「資料庫名稱」、「輸出路徑」五個參數。**對 training set 訓練的使用方法如下**：

```bash
python training.py -u "USERNAME"\
                   -p "PASSWORD"\
                   --host "HOSTNAME"\
                   "DATABASE"\
                   "OUTPUT_PATH"
```

預設連線至 PostgreSQL 資料庫，預設埠號為 5432。若有需求則可透過 `--protocol`, `--port` 選項更改，連線至其它資料庫如 MySQL：

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

    由於訓練過後是產出「最佳模型」和「交叉驗證結果」兩份檔案，預設名稱分別為 `best_model.h5`, `cv_results.csv`（無法更改），因此在傳遞 `OUTPUT_PATH` 時，只需給定「資料夾名稱」即可。

更多詳細資訊可透過 `-h`, `--help` 查看：

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

`predict.py` 的使用方法簡介如下：

```
usage: predict.py [-h] -u USERNAME -p PASSWORD --host HOSTNAME
                  [--port PORTNUMBER] [--protocol PROTOCOL] [-n]
                  DATABASE MODEL_NAME OUTPUT_PATH
```

與 `training.py` 類似，必須輸入五個參數，額外加上用來預測熱門文章的「模型路徑」。**對 public test set 預測的使用方法如下**：

```bash
python predict.py -u "USERNAME"\
                  -p "PASSWORD"\
                  --host "HOSTNAME"\
                  "DATABASE"\
                  "MODELNAME"\
                  "OUTPUT_PATH"
```
執行完成後，會輸出 `output.csv`（名稱無法更改）至指定的資料夾。必須注意的是，`MODEL_NAME` 選項必須帶入模型檔案名稱，而非資料夾路徑。

「作業補充說明及更正」信中有提到，在 private test 的時候，`posts_test` table 當中是沒有 `like_count_36_hour` 欄位的。因此，需要多給定 `-n` 選項，用來代表資料集當中並沒有該欄位。**對 private test set 預測的使用方法如下**：

```bash
python predict.py -u "USERNAME"\
                  -p "PASSWORD"\
                  --host "HOSTNAME"\
                  -n\
                  "DATABASE"\
                  "MODELNAME"\
                  "OUTPUT_PATH"
```
若有需要也可以透過 `--port`, `--protocol` 選項更改資料庫。

更多詳細資訊可透過 `-h`, `--help` 查看：

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
