def filterDfByLambda(df, feature, block):
    filtered_values = filter(block, df[feature])
    return df[df[feature].isin(filtered_values)]
