#! /usr/bin/env python3

# 匯入內建套件
from argparse import ArgumentParser
from pathlib import Path, PurePath


def get_predict_parser():
    """
    初始化並回傳設定好的 `ArgumentParser` 實體。
    """
    parser = ArgumentParser()
    # 設定必要參數
    parser.add_argument('database', metavar='DATABASE',
                        help='(Required) Database to use when connecting to server.')
    parser.add_argument('model_name', metavar='MODEL_NAME',
                        help='(Required) Prediction model name. If it is not in the current directory, please specify where it is.')
    parser.add_argument('output_path', metavar='OUTPUT_PATH',
                        help='(Required) File path of predicted results.')
    parser.add_argument('-u', dest='user', required=True, metavar='USERNAME',
                        help='(Required) User for login if not current user.')
    parser.add_argument('-p', dest='password', required=True, metavar='PASSWORD',
                        help='(Required) Password to use when connecting to server.')
    parser.add_argument('--host', required=True, metavar='HOSTNAME',
                        help='(Required) Host address to connect.')
    # 設定選用參數
    parser.add_argument('--port', metavar='PORTNUMBER', default='5432',
                        help='Port number to use for connection (default: 5432)')
    parser.add_argument('--protocol', metavar='PROTOCOL', default='postgres',
                        help='Protocol to connect. (default: postgres)')
    parser.add_argument('-n', dest='has_like_count_36_hour', action='store_false',
                        help='No like_count_36_hour column when the option is given.')
    return parser


def main(args):
    # 取得連線引擎
    engine = preprocessing.get_connector(
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        database=args.database,
        protocol=args.protocol
    )
    # 取得預測階段所需的 tables，接著合併、預處理所有 tables 。
    table_names_test = ['posts_test', 'post_shared_test', 'post_comment_created_test',
                        'post_liked_test', 'post_collected_test']
    tables_test = preprocessing.get_tables(engine, table_names_test)
    total_df_test = preprocessing.merge_tables(tables_test, table_names_test, how='left')
    total_df_test = preprocessing.preprocess_total_df(total_df_test,
                                                      has_like_count_36_hour=args.has_like_count_36_hour)
    # 讀入預測模型
    preprocessing.print_info("PREDICTING TESTSET START!")
    model_name = Path(f'{args.model_name}')
    model = load(model_name)
    # 開始進行預測，根據收到的 tables 是否有 like_count_36_hour 欄位，做不同處理
    if args.has_like_count_36_hour:
        y_true = total_df_test['is_trending']
        total_df_test = total_df_test.drop('is_trending', axis=1)
        y_pred = model.predict(total_df_test)
        # 在 console 印出 evaluate 結果
        print(f"{format('f1-score', '12s')} = {f1_score(y_true, y_pred):.2f}")
        print(f"{format('balanced acc', '12s')} = {balanced_accuracy_score(y_true, y_pred):.2f}\n")
        print(classification_report(y_true, y_pred))
    else:
        y_pred = model.predict(total_df_test)
    # 輸出預測結果
    output_path = Path(f'{args.output_path}')
    output_name = Path('output.csv')
    output_df = pd.DataFrame(zip(total_df_test.index, y_pred), columns=['post_key', 'is_trending'])
    output_df.to_csv(str(output_path/output_name), index=False, encoding='utf8')
    preprocessing.print_info("DONE!")


if __name__ == '__main__':
    # 取得解析器實體並解析命令列參數
    parser = get_predict_parser()
    args = parser.parse_args()
    # 確保模型和輸出路徑正確
    assert Path(f'{args.model_name}').is_file(), "Model file doesn't exist."
    assert Path(f'{args.output_path}').is_dir(), "Output directory doesn't exist."
    # 匯入第三方套件
    import pandas as pd
    from joblib import load
    from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
    # 匯入自行維護的套件
    import preprocessing
    main(args)