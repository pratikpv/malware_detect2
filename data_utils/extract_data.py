import pandas as pd
import sqlalchemy

DATA_CSV_FILENAME = 'malware_data_raw.csv'


def save_tables_to_csv():
    engine = sqlalchemy.create_engine('mysql+pymysql://admin:Pratik@123@localhost:3306/new_schema')
    df = pd.read_sql_table('FILES', engine)
    df.to_csv(DATA_CSV_FILENAME)
    return df


def main():
    # save_tables_to_csv()
    df = pd.read_csv(DATA_CSV_FILENAME)
    print(df.keys())
    df = df[['SHA1', 'FAMILY']]
    print(df.groupby('FAMILY').count())


if __name__ == "__main__":
    main()
