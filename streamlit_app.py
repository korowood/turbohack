import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Модуль очистки данных")

# Create file uploader object
upload_file = st.file_uploader('Загрузите неочищенные данные')


def load_data():
    if upload_file is not None:
        # Read the file to a dataframe using pandas
        df = pd.read_csv(upload_file, index_col='Параметр')
        """вычисляем статистики ..."""

        # st.header('Statistics of Dataframe')
        # st.write(df.describe())
        return df


def setup_page_preview(data: pd.DataFrame):
    st.title("ssss")
    """Какой то текст с объяснением"""
    """Следуйте инструкции"""

    st.title("*1. Введите номер фичи.*")
    """еще описание"""
    # col = st.text_input("", '')
    name_feat = st.selectbox("", data.columns.tolist())
    """Optional. You may set player's parameters in the sidebar in the left part of the screen.
    It may help you to understand the importance of different stats for player's salary. """
    st.title("*2. Нажмите кнопку если готовы *")
    return name_feat


# def setup_sidebar(player):
#     bar = st.sidebar
#     is_advanced_mode_checkbox = bar.checkbox('Show advanced settings')
#     bar.write(
#         """With using this option, you will be able to estimate
#         how different player's parameters affect fees cost.
#         """)
#     features = __create_features_initial_filling(player)
#     control_features = DEFAULT_PICKED_ADVANCED_FEATURES

def show_league_stat(data, name_feat):
    is_feat_select = st.checkbox('checkbox')
    if is_feat_select:
        left_col, right_col = st.columns(2)
        with left_col:
            fig = sns.boxplot(data=data[name_feat])
            st.write(fig)
        with right_col:
            fig = sns.histplot(data=data[name_feat])
            st.write(fig)


def main():
    df = load_data()

    name_feat = setup_page_preview(df)

    show_league_stat(df, name_feat)


if __name__ == "__main__":
    main()
