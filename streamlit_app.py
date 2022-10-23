import streamlit as st
import pandas as pd
import seaborn as sns
from task_1.streamlit_plot import *
from task_1.anomaly import anomaly_detect

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
        return df, True


def setup_page_preview(data: pd.DataFrame):
    st.title("ssss")
    """Какой то текст с объяснением"""
    """Следуйте инструкции"""

    st.title("*1. Введите номер фичи.*")
    """еще описание"""
    # col = st.text_input("", '')
    name_feat = st.selectbox("", data.columns.tolist())
    """Выбирайте признак по которому хотите посмотреть изменения"""
    st.title("*2. Нажмите кнопку если готовы *")
    return name_feat


def setup_sidebar(data):
    bar = st.sidebar
    with bar:
        result = st.button("Почистить даатсет?")

        if result:
            new_data = data.reset_index()
            outlier_col, outlier_list = anomaly_detect(new_data.loc[:, "х001":])
            new_data = new_data.drop(new_data.index[outlier_list])
            return result, new_data
        else:
            st.write("Данные еще не загружены")
            return result


def show_stat(data, name_feat):
    is_feat_select = st.checkbox('checkbox')
    if is_feat_select:
        one, two = st.columns([2, 2])
        with one:
            fig = plot_feat_hist(data, name_feat)
            # one.plotly_chart(fig)

            st.write(fig)
        with two:
            fig = plot_feat_boxplot(data, name_feat)
            # two.plotly_chart(fig)
            st.write(fig)

        fig = plot_feat_line(data.reset_index(), name_feat)
        # three.plotly_chart(fig)
        st.write(fig)


def show_diff(data, new_data, name_feat):
    is_feat_select = st.checkbox('Показать разницу')
    if is_feat_select:
        fig = plot_line_diff(data.reset_index(), new_data, name_feat)
        st.write(fig)


def main():
    if upload_file is not None:
        data, loaded = load_data()

        name_feat = setup_page_preview(data)

        show_stat(data, name_feat)

        if loaded:
            result, new_data = setup_sidebar(data)

        if result:
            show_diff(data, new_data, name_feat)

        # new_df = df.reset_index()
        #
        # outlier_col, outlier_list = anomaly_detect(new_df.loc[:, "х001":])
        #
        # new_df = new_df.drop(new_df.index[outlier_list])


if __name__ == "__main__":
    main()
