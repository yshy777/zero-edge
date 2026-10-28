import os
import pandas as pd

def save_df_to_excel(df: pd.DataFrame, file_path: str, df_name: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        df.to_excel(file_path, sheet_name=df_name, index=False)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=df_name, index=False)
