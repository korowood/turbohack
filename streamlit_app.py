import streamlit as st
import pandas as pd
# import openpyxl


from io import BytesIO
# from pyxlsb import open_workbook as open_xlsb

upload_file = st.file_uploader('Загрузите неочищенные данные')

@st.cache
def get_data_from_excel(filename, datasheet):

    df = pd.read_excel(
        io=filename,
        engine='openpyxl',
        sheet_name=datasheet,
    )


    #df.colums = ['Cable Length','Theta','No.']
    #df["hour"] = pd.to_datetime(df["Time"],format="%H:%M:%S").dt.hour
    return df


df = get_data_from_excel(upload_file, 'Sheet1')



# def to_excel(df):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, index=False, sheet_name='Sheet1')
#     workbook = writer.book
#     worksheet = writer.sheets['Sheet1']
#     format1 = workbook.add_format({'num_format': '0.00'})
#     worksheet.set_column('A:A', None, format1)
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data
#
#
#
#
#
#
#
# st.title("Модуль очистки данных")
#
# # Create file uploader object
# upload_file = st.file_uploader('Загрузите неочищенные данные')
# # df = pd.read_excel(upload_file, index_col='Параметр', engine="openpyxl")
#
# df_xlsx = to_excel(upload_file)
# st.download_button(label='📥 Download Current Result',
#                    data=df_xlsx ,
#                    file_name='df_test.xlsx')
#
# print("ok")
# bytesData = upload_file.getvalue()
# encoding = encodingUTF8
# s=str(bytesData,encoding)
# result = StringIO(s)



# def load_data_calc_stats():
#     if upload_file is not None:
#
#         # Read the file to a dataframe using pandas
#
#         # df = pd.read_excel(upload_file, index_col='Параметр', engine="openpyxl")
#         """вычисляем статистики"""
#         # Create a section for the dataframe statistics
#         st.header('Statistics of Dataframe')
#         st.write(df.describe())
#
#         return df
#
#
# def setup_page_preview(data: pd.DataFrame):
#     st.title("ssss")
#     """Какой то текст с объяснением"""
#     """Следуйте инструкции"""
#
#     st.title("*1. Введите номер фичи.*")
#     """еще описание"""
#     #col = st.text_input("", '')
#     # col = st.selectbox("", data.columns.tolist())
#     # """Optional. You may set player's parameters in the sidebar in the left part of the screen.
#     # It may help you to understand the importance of different stats for player's salary. """
#     # st.title("*2. Нажмите кнопку если готовы *")
#     # return col
#
#
# def main():
#
#     df = load_data_calc_stats()
#
#     # setup_page_preview(df)
#
#
# if __name__ == "__main__":
#     main()