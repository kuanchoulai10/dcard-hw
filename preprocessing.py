# 匯入內建套件
import sys
from functools import reduce

# 匯入第三方套件
import sqlalchemy
import pandas as pd
import numpy as np


def print_info(info, width=61, fillchar='='):
    """
    印出格式化的資訊
    """
    temp_width = width - (width-len(info))//2
    print(info.rjust(temp_width, fillchar).ljust(width, fillchar))


def get_connector(user, host, database, password=None, port='5432', protocol='postgres'):
    """
    取得連線引擎，預設為連線至 PostgreSQL，埠號預設為 5432。
    """
    print_info("GETTING CONNECTOR START!")
    user_info = f'{user}:{password}' if password else user
    url = f'{protocol}://{user_info}@{host}:{port}/{database}'
    engine = sqlalchemy.create_engine(url, client_encoding='utf-8')
    print_info("DONE!")
    return engine


def get_tables(engine, table_names):
    """
    依照 `tables_names` 的順序，取得 tables，並依序儲存於 `list` 當中，回傳型態為 `list`，每個 element 為 `DataFrame`。
    """
    print_info("GETTING TABLES START!")
    rslt = []
    for tn in table_names:
        query = f'SELECT * FROM {tn}'
        exec(f'{tn} = pd.read_sql(query, engine)')
        # exec(f"{tn} = pd.read_csv('{tn}.csv', encoding='utf8')") # from current working directory
        print(
            f'{format(tn, "26s")} 總共有 {eval(f"{tn}.shape[0]"):9,} 筆資料和 {eval(f"{tn}.shape[1]")} 個欄位')
        exec(f'rslt.append({tn})')
    print_info("DONE!")
    return rslt


def merge_tables(tables, table_names, how):
    """
    合併所有 tables，回傳型態為 `DataFrame`。
    """
    print_info("MERGING TABLES START!")
    # 分別處理 post_{shared, comment_created, liked, collected}_{train, test} 四個 tables
    # groupby 每篇文章，將前十小時的分享數、評論數、愛心數、收藏數加總起來
    for idx, (table, tn) in enumerate(zip(tables, table_names)):
        if len(tn.split('_'))==2: continue                  # for handling posts_{train, test} table
        col_name = f"{tn.split('_')[1]}_count"              # tn.split('_')[1] is either {shared, comment, liked, collected}
        mapper = {'count': col_name}
        exec(f"tables[{idx}] = table.groupby(['post_key'], as_index=False).sum().rename(columns=mapper)")
    # 將 tables 合併起來並回傳。
    total_df = reduce(lambda left, right: pd.merge(left, right, on=['post_key'], how=how), tables)
    print_info("DONE!")
    return total_df


def preprocess_total_df(total_df, has_like_count_36_hour):
    """
    預處理剛合併好的 total_df 以符合後續建模需求，回傳型態為 `DataFrame`。
    """
    print_info("PREPROCESSING TOTAL_DF START!")
    total_df.set_index('post_key', inplace=True)                                    # post_key 欄位設為索引
    total_df['created_at_hour'] = pd.to_datetime(total_df['created_at_hour'])       # 將 created_at_hour 欄位轉換成 datetime 型態
    total_df['weekday'] = total_df['created_at_hour'].dt.dayofweek                  # 擷取出發文的 weekday
    total_df['hour'] = total_df['created_at_hour'].dt.hour                          # 擷取出發文的 hour
    total_df.fillna(0, inplace=True)                                                # NaN 值補 0
    # 根據收到的 tables 是否有 like_count_36_hour 欄位，做不同處理
    if has_like_count_36_hour:
        total_df['is_trending'] = 0+(total_df['like_count_36_hour']>=1000)          # 轉換成 is_trending 類別欄位
        total_df = total_df.drop(['created_at_hour', 'like_count_36_hour'], axis=1) # drop 掉不必要的欄位
    else:
        total_df = total_df.drop(['created_at_hour'], axis=1)                       # drop 掉不必要的欄位
    # 將計次欄位轉換成 int 型態
    col_names = ['shared_count', 'comment_count', 'liked_count', 'collected_count']
    for cn in col_names:
        total_df[cn] = total_df[cn].astype(dtype='int')
    print_info("DONE!")
    return total_df