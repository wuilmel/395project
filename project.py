
"""
Name: Wuilmel Wu Wu
Email: wuilmel.wuwu92@myhunter.cuny.edu
Resources:  books, online resources and class materials
Title: Strategic School Building Plan
URL: https://github.com/wuilmel/395project/blob/85a8f9a313424e455b6bdac12465e0648144bfb5/project.py
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium

df1 = pd.read_csv("https://data.cityofnewyork.us/resource/wg9x-4ke6.csv", usecols = ['system_code',
    'location_name', 'primary_address_line_1', 'state_code', 'x_coordinate',
    'y_coordinate', 'longitude', 'latitude', 'nta','geographical_district_code'])

df1.rename(columns = {'system_code': 'dbn'}, inplace=True)

df2 = pd.read_csv("https://data.cityofnewyork.us/resource/45j8-f6um.csv", usecols = ['dbn', 
    'school_name', 'year', 'total_enrollment', 'grade_pk_half_day_full_day', 'grade_k',
    'grade_1', 'grade_2', 'grade_3', 'grade_2',
    'grade_3', 'grade_4', 'grade_5', 'grade_6', 'grade_7', 'grade_8',
    'grade_9', 'grade_10', 'grade_11', 'grade_12',  
    'students_with_disabilities', 'english_language_learners', 'poverty', 'economic_need_index'])

merged_df = pd.merge(df2, df1, on=['dbn'])

df_merged = pd.merge(df2, df1, on=['dbn'])

print(merged_df)

#boxplot

df3 = merged_df.groupby(['school_name', 'year'])['total_enrollment'].apply(list)

df3 = df3.reset_index()

df4 = pd.DataFrame(df3['total_enrollment'].to_list())

df4['school_name'] = df3['school_name']

df4['year'] = df3['year']

df4 = pd.melt(df4, id_vars=['school_name', 'year'], value_name='total_enrollment')

sns.boxplot(x='school_name', y='total_enrollment', hue='year', data=df4)

plt.title('Total Enrollment by School and Year')

plt.xlabel('School Name')

plt.ylabel('Total Enrollment')

plt.xticks(rotation=90, ha='right', fontsize=5)

plt.subplots_adjust(bottom=0.25)

plt.show()

#Histogram

sns.histplot(data=merged_df, x='total_enrollment', hue='year', multiple='stack', bins=20)

plt.title('Total Enrollment by School and Year')

plt.xlabel('Total Enrollment')

plt.ylabel('Frequency')

plt.show()

#Scatter plot for one school

Roberto_school = merged_df[merged_df['school_name'] == 'P.S. 015 Roberto Clemente']

plt.scatter(Roberto_school ['year'], Roberto_school ['total_enrollment'])

plt.xlabel('Year')

plt.ylabel('Total Enrollment')

plt.title('Enrollment Trend for P.S. 015 Roberto Clemente')

plt.show()

#Scatter plot for all the shcool

df5 = merged_df.groupby('year')['total_enrollment'].sum()

plt.scatter(range(len(df5)), df5)

plt.xticks(range(len(df5)), df5.index)

plt.xlabel('Year')

plt.ylabel('Total Enrollment')

plt.title('Enrollment by Year')

plt.subplots_adjust(bottom=0.25)

plt.show()

#create Choropleth map with folium

nta_map = gpd.read_file("https://data.cityofnewyork.us/api/geospatial/cpf4-rkhq?method=export&format=GeoJSON")

merged_df['year_range'] = merged_df['year'].astype(str).str[:4] + '-' + merged_df['year'].astype(str).str[-2:]

df_2014_15 = merged_df[(merged_df["year"] >= "2014-15") & (merged_df["year"] <= "2015-16")]

df_2015_16 = merged_df[(merged_df["year"] >= "2015-16") & (merged_df["year"] <= "2016-17")]

df_2016_17 = merged_df[(merged_df["year"] >= "2016-17") & (merged_df["year"] <= "2017-18")]

df_2017_18 = merged_df[(merged_df["year"] >= "2017-18") & (merged_df["year"] <= "2018-19")]

df_2018_19 = merged_df[(merged_df["year"] >= "2018-19") & (merged_df["year"] <= "2019-20")]

#To check if I input the year correctly to the maps
print(merged_df['year'].unique())

map_2014_15 = folium.Map(location=[40.7128, -74.0060], zoom_start=15)

map_2015_16 = folium.Map(location=[40.7128, -74.0060], zoom_start=15)

map_2016_17 = folium.Map(location=[40.7128, -74.0060], zoom_start=15)

map_2017_18 = folium.Map(location=[40.7128, -74.0060], zoom_start=15)

map_2018_19 = folium.Map(location=[40.7128, -74.0060], zoom_start=15)

for map_obj, df_year in zip([map_2014_15, map_2015_16, map_2016_17, map_2017_18, map_2018_19], [df_2014_15, df_2015_16, df_2016_17, df_2017_18, df_2018_19]):

    enrollment = merged_df.groupby(["nta", "latitude", "longitude", "school_name"])["total_enrollment"].mean().reset_index()

    merged_df = nta_map.merge(enrollment, left_on="ntacode", right_on="nta", how="left")

    colorscale = [(0, 'white'), (0.07, 'purple'), (0.14, 'blue'), (0.21, 'cyan'),
                  (0.28, 'green'), (0.35, 'yellow'), (0.42, 'orange'),(0.49, 'red'),    
                  (0.56, 'maroon'), (0.63, 'pink'), (0.7, 'teal'), 
                  (0.77, 'lavender'), (0.84, 'beige'), (1, 'black')]

    folium.Choropleth(
        geo_data=merged_df,
        name='choropleth',
        data=merged_df,
        columns=['ntacode', 'total_enrollment'],
        key_on='feature.properties.ntacode',
        color_scale=colorscale,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Enrollment',
        highlight=True,
        threshold_scale=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        width='70%',
        legend_font_size=12,
        bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        reset=True,

    ).add_to(map_obj)

    merged_df = merged_df.dropna(subset=['latitude', 'longitude'])

    for lat, lng, name in zip(merged_df['latitude'], merged_df['longitude'], merged_df['school_name']):
        folium.Marker(
            location=[lat, lng],
            popup=name,
            icon=folium.Icon(icon='university', prefix='fa')
        ).add_to(map_obj)
    folium.LayerControl().add_to(map_obj)

map_2014_15.save('/Users/wuilm/Desktop/map_2014_15.html')

map_2015_16.save('/Users/wuilm/Desktop/map_2015_16.html')

map_2016_17.save('/Users/wuilm/Desktop/map_2016_17.html')

map_2017_18.save('/Users/wuilm/Desktop/map_2017_18.html')

map_2018_19.save('/Users/wuilm/Desktop/map_2018_19.html')

#Linear model

X = df_merged ['students_with_disabilities'].values.reshape(-1, 1)

y = df_merged ['total_enrollment'].values

model = LinearRegression()

model.fit(X, y)

print("Intercept: ", round(model.intercept_, 2))

print("Slope: ", round(model.coef_[0], 2))

X = df_merged ['students_with_disabilities']

y = df_merged ['total_enrollment']

fig = go.Figure()

fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data'))

fig.update_layout(title='Total Enrollment vs students with disabilities',
                  xaxis_title='students with Disabilities',
                  yaxis_title='Total Enrollment')

fig.add_trace(go.Scatter(x=X, y=model.predict(X.values.reshape(-1, 1)),
                         mode='lines', name='Linear Regression'))

fig.show()

#Multiple linear regression model

X = df_merged [['grade_pk_half_day_full_day', 'grade_k', 'grade_1', 'grade_2', 'grade_3', 'grade_2',
              'grade_3', 'grade_4', 'grade_5', 'grade_6', 'grade_7', 'grade_8', 'grade_9', 'grade_10', 'grade_11', 'grade_12']]

y = df_merged ['total_enrollment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', round(mse), 2)

print('Root Mean Squared Error:', round(rmse, 2))

print('R-squared:', r2)

y_pred = reg.predict(X_test)

#scatter plot with Multiple linear regression model
plt.scatter(y_test, y_pred)

plt.xlabel('Actual values')

plt.ylabel('Predicted values')

plt.title('Actual vs Predicted values')

plt.show()
