import streamlit as st
import pandas as pd


st.title("Модуль очистки данных")

# Create file uploader object
upload_file = st.file_uploader('Загрузите неочищенные данные')


def load_data_calc_stats():
    if upload_file is not None:

        # Read the file to a dataframe using pandas
        df = pd.read_csv(upload_file)

        """вычисляем статистики"""
        # Create a section for the dataframe statistics
        st.header('Statistics of Dataframe')
        st.write(df.describe())

        return df


def setup_page_preview(data: pd.DataFrame):
    st.title("ssss")
    """Какой то текст с объяснением"""
    """Следуйте инструкции"""

    st.title("*1. Введите номер фичи.*")
    """еще описание"""
    #col = st.text_input("", '')
    col = st.selectbox("", data.columns.tolist())
    """Optional. You may set player's parameters in the sidebar in the left part of the screen.
    It may help you to understand the importance of different stats for player's salary. """
    st.title("*2. Нажмите кнопку если готовы *")
    return col


def main():

    df = load_data_calc_stats()

    setup_page_preview(df)


if __name__ == "__main__":
    main()