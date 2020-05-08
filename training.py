#! /usr/bin/env python3

# 匯入內建套件
from argparse import ArgumentParser
from pathlib import Path, PurePath


def get_training_parser():
    """
    初始化並回傳設定好的 `ArgumentParser` 實體。
    """
    parser = ArgumentParser()
    # 設定必要參數
    parser.add_argument('database', metavar='DATABASE',
                        help='(Required) Database to use when connecting to server.')
    parser.add_argument('output_path', metavar='OUTPUT_PATH',
                        help='(Required) Best prediction model and cross validation results outputs file path.')
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
    # 取得訓練階段所需的 tables，接著合併、預處理所有 tables 。
    table_names_train = ['posts_train', 'post_shared_train', 'post_comment_created_train',
                         'post_liked_train', 'post_collected_train']
    tables_train = preprocessing.get_tables(engine, table_names_train)
    total_df_train = preprocessing.merge_tables(tables_train, table_names_train, how='left')
    total_df_train = preprocessing.preprocess_total_df(total_df_train, 
                                                       has_like_count_36_hour=True)
    # 開始訓練模型
    preprocessing.print_info("TRAINING START!")
    # STEP 1: 建立 Pipeline
    cachedir = mkdtemp()
    pipe = Pipeline(steps=[('resampler', 'passthrough'),
                           # ('columntransformer', 'passthrough'),
                           ('classifier', 'passthrough')],
                    memory=cachedir)
    # poly_cols = ['shared_count', 'comment_count', 'liked_count', 'collected_count']
    # col_trans = make_column_transformer((OneHotEncoder(dtype='int'), ['weekday']),
    #                                     (PolynomialFeatures(include_bias=False), poly_cols),
    #                                     remainder='passthrough')
    # STEP 2: 設定超參數空間以及衡量指標，建立 GridsearchCV
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
    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'specificity': make_scorer(specificity_score),
        'balanced_accuracy': 'balanced_accuracy',
        'f1_score': 'f1',
    }
    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, refit='f1_score', 
                               n_jobs=-1, cv=3, return_train_score=True)
    # STEP 3: 搜尋最佳超參數組合並印出所需時間
    start_time = time()
    grid_search.fit(total_df_train.drop('is_trending', axis=1), total_df_train['is_trending'])
    preprocessing.print_info(f"GRID SEARCH: {time()-start_time:.2f} secs")
    # STEP 4: 將最佳模型和交叉驗證結果儲存起來
    output_path = PurePath(f'{args.output_path}')
    model_name, results_name = PurePath('best_model.h5'), PurePath('cv_results.csv')
    dump(grid_search.best_estimator_, str(output_path/model_name))
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(str(output_path/results_name), index=False, encoding='utf8')
    rmtree(cachedir)
    preprocessing.print_info("DONE!")


if __name__ == '__main__':
    # 取得解析器實體並解析命令列參數
    parser = get_training_parser()
    args = parser.parse_args()
    # 確保輸出路徑正確
    assert Path(f'{args.output_path}').is_dir(), "Output directory doesn't exist."
    # 匯入內建套件
    from tempfile import mkdtemp
    from shutil import rmtree
    from time import time
    # 匯入第三方套件
    import pandas as pd
    from xgboost.sklearn import XGBClassifier
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.metrics import make_scorer
    from imblearn.pipeline import Pipeline
    from imblearn.metrics import specificity_score
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import NearMiss
    from joblib import dump
    # 匯入自行維護的套件
    import preprocessing
    main(args)