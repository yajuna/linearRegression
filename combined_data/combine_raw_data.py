import pandas as pd

# Reorder tree data by temp with column name as new feature
tree = pd.read_csv('./raw_temp.csv')
tree = tree.melt(id_vars=['Date Time'], var_name='Temperature Source', value_name='Temperature')
# Format the date time column and set as index
tree['Date Time'] = pd.to_datetime(tree['Date Time'], format='%m/%d/%Y %H:%M')
tree.set_index('Date Time', inplace=True)
print(tree)

# Set the date time as a pd.datetime column and to the index
weather = pd.read_csv('./raw_weather.csv')
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather.set_index('datetime', inplace=True)
# Reindex weather to match the tree data frame
weather = weather.reindex(tree.index, method='nearest')
print(weather)

# Combine the data frames
combined = pd.concat([tree, weather], axis=1)
print(combined)

# clean the temperature source column
combined['Temperature Source'] = combined['Temperature Source'].apply(lambda x: x.replace('@', ' ')
                                                                      .replace('cm', '')
                                                                      .replace(',', ' ')
                                                                      .replace('m', ''))

combined['direction'] = combined['Temperature Source'].apply(lambda x: x.split(' ')[0][0])
combined['depth'] = combined['Temperature Source'].apply(lambda x: x.split(' ')[0][1:] if x.split(' ')[0][1:] != "_Ext_Tep" else 0)
combined['height'] = combined['Temperature Source'].apply(lambda x: x.split(' ')[1])

# Reorder the columns and drop redundant columns
combined = combined.drop('Temperature Source', axis=1)
cols = combined.columns.tolist()
cols = [cols[0]] + cols[-3:] + cols[1:7]
combined = combined[cols]
# print the first row to check the changes
print(combined.iloc[0])


# Save the cleaned data
combined.to_csv('./combined_data_temp.csv')
