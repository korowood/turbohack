def anomaly_detect(data):
    outlier_list = []
    outlier_col = []

    for col in data.columns:

        temp_df = data[(data[col] > data[col].mean() + data[col].std() * 10) |
                       (data[col] < data[col].mean() - data[col].std() * 10)]

        temp2_df = data[(data[col] > data[col].mean() + data[col].std() * 2) |
                        (data[col] < data[col].mean() - data[col].std() * 2)]
        if len(temp_df) > 0:
            outliers = temp_df.index.to_list()
            outlier_list.extend(outliers)
            outlier_col.append(col)
            # print(col, len(temp_df))
        elif len(temp2_df) > 0:
            outliers = temp2_df.index.to_list()
            outlier_list.extend(outliers)
            outlier_col.append(col)
            # print(col, len(temp2_df))

    outlier_list = list(set(outlier_list))
    # print(len(outlier_col), len(outlier_list))
    return outlier_col, outlier_list
