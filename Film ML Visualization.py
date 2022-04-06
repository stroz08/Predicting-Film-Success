# -*- coding: utf-8 -*-

# -- Sheet --

# # Predicting Film Financial and Critical Success using Graphical Analysis and Machine Learning Techniques


# ## Step 1: Gathering Data from Datasets
# We will begin by gathering the data from the 2017 and 2020 film datasets. We will gather the common columns so that we can later use them in the same ways. The CSV files will be converted into Pandas library dataframes.
# These datasets were gathered from Kaggle.com.
# - 2017 Dataset: https://www.kaggle.com/rounakbanik/the-movies-dataset
# - 2020 Dataset: https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset
# 
# The datasets have the following columns in common which will be collected and used later for graphing and developing our regression model:
# - Title
# - Genre
# - Budget
# - Global Revenue
# - Release Date
# - Runtime


# ### Importing Libraries


# Here we import the necessary libraries. We use Pandas for data management, Matplotlib and Seaborn
# for graphing, and some of Scikit-learn's features for machine learning.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ### Gathering 2017 Data


# We now gather the data we need from the 2017 dataset. This will be used for our machine learning models.
df_2017 = pd.read_csv('film_data/2017/movies_metadata.csv', low_memory=False)

# Afterward, we filter our dataframe to the desired columns.
df_2017 = df_2017[['title', 'genres', 'budget', 'revenue', 'release_date', 'runtime', 'id']]

# In order to make sure we can make accurate analysis, we will filter entries null values.
df_2017 = df_2017.dropna()

# We will also remove films that do not have a 0 value for budget, revenue, or runtime.
df_2017 = df_2017[(df_2017['budget'] != '0')]
df_2017 = df_2017[(df_2017['revenue'] != 0.0)]
df_2017 = df_2017[(df_2017['runtime'] != 0.0)]

# We are gather the ratingsd for each film.
df_2017_ratings = pd.read_csv('film_data/2017/ratings.csv')

# Since the ratings file contains several individual ratings from several users, which includes many ratings per movie id, 
# we are going to have to aggregate this data by the mean rating for each movie.
df_2017_ratings = round(df_2017_ratings.groupby('movieId')['rating'].mean(), 1)

# We will be merging these data frames using the 'movieId' and 'id' columns, which should be consistent through both files.
df_2017['id'] = pd.to_numeric(df_2017['id'])
df_2017_ratings = df_2017_ratings.to_frame().merge(df_2017, left_on='movieId', right_on='id')

# Converting budget to a float for the revenue model
df_2017_ratings['budget'] = pd.to_numeric(df_2017_ratings['budget'], downcast='float')

# The final result is shown below!
df_2017

# ### Gathering 2020 Data


# Now, we will gather the necessary data from the 2020 dataset. This will be used to measure the machine
# learning model's accuracy as well as graphing for the non-machine learning analysis.
df_2020 = pd.read_csv('film_data/2020/IMDb movies.csv', low_memory=False)

# We now filter to some of the columns which are shared with the 2017 dataset.
df_2020 = df_2020[['original_title', 'genre', 'budget', 'year', 'duration', 'worlwide_gross_income','imdb_title_id']]
# The dataset creator left us a typo to fix, it seems. We rename "worlwide" to "worlwide."
df_2020 = df_2020.rename(columns={'worlwide_gross_income' : 'worldwide_gross_income'})
# In order to make sure we can make accurate analysis, we will filter entries with missing values.
df_2020 = df_2020.dropna()

# Load in the 2020 movie ratings dataset
df_2020_ratings = pd.read_csv('film_data/2020/IMDb ratings.csv', low_memory=False)
# Filter to just the average vote since that is what we are interested in later.
df_2020_ratings = df_2020_ratings[['imdb_title_id', 'weighted_average_vote']]

# Merge on the 'imdb_title_id'
df_2020 = df_2020.merge(df_2020_ratings, left_on='imdb_title_id', right_on='imdb_title_id')

# The final result is shown below!
df_2020

# ### Modifying Data
# 
# We will be making modifications to our datasets so that the subsequent steps (such as Step 2) are more simple to accomplish.
# First, we simplify the dictionary of genres in the 2017 into a single genre.


# In order to plot the results for each genre, we will convert the dictionary of genres into a single genre.
# This solution is hard coded to work with the dictionary of genres that come from the 2017 dataset.
def genres_to_list(data):
    if len(data.split('\'')) >= 6:
        return data.split('\'')[5]
    # If the genre list is empty
    else:
        return None
 
# We apply this function to our 2017 dataframe in order to get the desired result.
df_2017['genres'] = df_2017['genres'].apply(genres_to_list)
df_2017['budget'] = pd.to_numeric(df_2017['budget'])
df_2017_ratings['genres'] = df_2017_ratings['genres'].apply(genres_to_list)

# ## Step 2: Genre, Rating, and Revenue Analysis
# Now that we have the necessary data, we can now analyze and visualize how genre and ratings affect revenue for films.
# We will be using the 2017 dataset for these graphs.


# ### Revenue vs Genre
# Films which fall under multiple genres will count the first genre listed as its type of genre.
# For example: an 'Action', 'Adventure' film would count as an 'Action' film


# This controls the palette for the rest of the plots of the document!
palette_name = "rocket_r"

# #### Mean Revenue per Genre


# We run a groupby to find the average revenue per genre of film. We now know on average how much each genre of film
# makes in terms of pure revenue.
genres = df_2017.groupby('genres')['revenue'].mean()
genres = pd.DataFrame(genres)
genres.columns = ['revenue']
genres = genres.sort_values('revenue', ascending=False)

# Graphing
fig, ax = plt.subplots(1, figsize=(20,10))
ax.set_facecolor('white')
# This function adds commas to the y axis [1].
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
sns.set()
sns.barplot(data=genres, x=genres.index, y='revenue', palette=palette_name, ax=ax)
plt.xticks(rotation=45)
plt.title("Mean Revenue per Genre", fontsize=22)
plt.xlabel("Genres", fontsize=16)
plt.ylabel("Revenue (USD)", fontsize=16)

# #### Ratio of Mean Revenue to Mean Budget per Genre


# We run a groupby to find the average revenue per genre of film, then divide it by the average budget per genre.
# This results in a ratio of how much money a film made versus how much it cost to produce.
genres = df_2017.groupby('genres')['revenue'].mean() / df_2017.groupby('genres')['budget'].mean()
genres = pd.DataFrame(genres)
genres.columns = ['revenue']
genres = genres.sort_values('revenue', ascending=False)

# Graphing
fig, ax = plt.subplots(1, figsize=(20,10))
ax.set_facecolor('white')
sns.set()
sns.barplot(data=genres, x=genres.index, y='revenue', palette=palette_name)
plt.xticks(rotation=45)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Ratio of Mean Revenue to Mean Budget per Genre", fontsize=22)
plt.xlabel("Genres", fontsize=16)
plt.ylabel("Revenue to Budget Ratio", fontsize=16)
# Creates a line to represent the point where budget is equal to revenue
plt.axhline(1, color="white", linewidth=10)

# **The line drawn on the y-axis indicates the point at which a film which breaks even on their budget**.
# 
# It seems that every genre, on average, makes its money back when compared its budget! Notice how although films in the Animation genre made the most revenue in the previous plot, they are very expensive so their position falls in this graph. Foreign films were last place in revenue, but they make a big jump in terms of revenue to budget ratio due to their on-average smaller budgets. The same phenomenon is apparent in an even more dramatic fashion with the TV Movie genre.


# ### Rating vs Genre


# #### Mean Rating per Genre
# 
# For this graph, we will find out what the average rating for each genre of film is.


# We run a groupby to find the average rating per genre of film.
ratings = df_2017_ratings.groupby('genres')['rating'].mean()
ratings = pd.DataFrame(ratings)
ratings.columns = ['rating']
ratings = ratings.sort_values('rating', ascending=False)

# Graphing
fig, ax = plt.subplots(1, figsize=(20,10))
ax.set_facecolor('white')
sns.set()
sns.barplot(data=ratings, x=ratings.index, y='rating', ax=ax, palette=palette_name)
plt.ylim(0,5)
plt.xticks(rotation=45)
plt.title("Mean Rating per Genre", fontsize=22)
plt.xlabel("Genres", fontsize=16)
plt.ylabel("Rating", fontsize=16)

# The graph displays that, on average, the most critically acclaimed genre is 'War.' It is interesting that although War films tend to not make much over their budget (as evident by the previous graph), they tend to be critically successful. The genre with the lowest average rating is 'History.' 


# ### Revenue vs Rating


# #### Mean Revenue per Rating
# 
# This graph will display the correlation between film ratings and the amount of revenue a film generates.


# We run a groupby to find the average revenue per rating of film.
ratings = df_2017_ratings.groupby('rating')['revenue'].mean()
ratings = pd.DataFrame(ratings)
ratings.columns = ['rating']

# Graphing
fig, ax = plt.subplots(1, figsize=(20,10))
ax.set_facecolor('white')
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
sns.set()
sns.regplot(data=ratings, x=ratings.index, y='rating', ax=ax, color='black')
plt.title("Mean Revenue per Rating", fontsize=22)
plt.xlabel("Rating", fontsize=16)
plt.ylabel("Revenue (USD)", fontsize=16)
df_2017

# The graph shows that there is a positive correlation between revenue and rating! Unsurprisingly, movies which are critically successful also tend to bring in large amounts of money. Keep in mind that revenue is NOT the same thing as profit.


# #### Mean Ratio of Revenue to Budget per Rating
# This graph displays the mean ratio of revenue to budget per each rating of film. This graph is less biased toward large budget, large revenue films than the previous graph because it is based on a ratio rather than the raw amount of revenue generated.


# We run a groupby to find the average revenue per rating of film.
ratings = df_2017_ratings.groupby('rating')['revenue'].mean() / df_2017_ratings.groupby('rating')['budget'].mean()
ratings = pd.DataFrame(ratings)
ratings.columns = ['rating']

# Graphing
fig, ax = plt.subplots(1, figsize=(20,10))
ax.set_facecolor('white')
sns.set()
sns.regplot(data=ratings, x=ratings.index, y='rating', ax=ax, color='black')
plt.title("Mean Ratio of Revenue to Budget per Rating", fontsize=22)
plt.xlabel("Rating", fontsize=16)
plt.ylabel("Revenue to Budget Ratio", fontsize=16)

# The results are very similar to the previous graph. The trend of films being more financially successful the more critically acclaimed they are holds true.


# ## Step 3: Generating a Regression Machine Learning Model for Ratings
# ### Generating the Model


# So now we have a merged dataframe with the average rating for each film title, we now are going to separate the significant
# columns into features and labels to set up our ratings regression model.
filtered_data = df_2017_ratings[['rating', 'budget', 'revenue', 'genres']]
features = filtered_data.loc[:, filtered_data.columns != 'rating']

# Onehot encode the genere feature since it is a string value
features_onehot = pd.get_dummies(features)
# Rating will be our label for this model
labels = filtered_data['rating']

# Crate our testing and training sets
features_train, features_test, labels_train, labels_test = \
    train_test_split(features_onehot, labels, test_size=0.2)

# Initialize and run and our regression model, then generate a prediction set
model = DecisionTreeRegressor()
model.fit(features_train, labels_train)
features_test_prediction = model.predict(features_test)

# Print the error betweeen the test labels and the predicted labels.
print(mean_squared_error(labels_test, features_test_prediction))

# ## Step 4: Generating a Regression Machine Learning Model for Revenue
# ### Generating the Model


# So now we have a merged dataframe with the average rating for each film title, we now are going to separate the significant
# columns into features and labels to set up our revenue regression model.
filtered_data_revenue = df_2017_ratings[['rating', 'budget', 'revenue', 'genres']]
features2 = filtered_data_revenue.loc[:, filtered_data_revenue.columns != 'revenue']

# Onehot encode the genre feature since it is a string value
features2_onehot = pd.get_dummies(features2)

# Revenue will be our label for this model
label = filtered_data_revenue['revenue']

# Crate our testing and training sets
features_train2, features_test2, label_train2, label_test2 = \
    train_test_split(features2_onehot, label, test_size=0.2)
    
# Initialize and run and our regression model, then generate a prediciton set
model2 = DecisionTreeRegressor()
model2.fit(features_train2, label_train2)
features_test_prediction2 = model2.predict(features_test2)

# Print the error betweeen the test labels and the predicted labels.
print(mean_squared_error(label_test2, features_test_prediction2))

# ## Step 5: Train a ML model with the 2017 data, and predict 2020 data!
# We now will generate another regression machine learning model with modified 2017 data so that we can use the 2020 data for testing our model. The error that we calculate from this process should give us an idea of how effective our 2017 model is at predicting films from 2017 to 2020.
# 
# Most of the difficulty of this process comes from filtering and modifying data from both sets so that they are compatible with each other.



'''
Regression Machine Learning Model for Ratings (2017 trained)
'''

# So now we have a merged dataframe with the average rating for each film title, we now are going to separate the significant
# columns into features and labels to set up our ratings regression model.
filtered_data_2017 = df_2017_ratings[['rating', 'budget', 'revenue', 'genres']]
#filtered_data_2017.drop(filtered_data_2017.tail(83).index, inplace=True)
features_2017 = filtered_data_2017.loc[:, filtered_data_2017.columns != 'rating']

# Onehot encode the genere feature since it is a string value
features_onehot_2017 = pd.get_dummies(features_2017)
features_onehot_2017 = features_onehot_2017.drop(columns=['genres_Foreign'], axis=1)
features_onehot_2017 = features_onehot_2017.drop(columns=['genres_Documentary'], axis=1)
# Rating will be our label for this model
labels_2017 = filtered_data_2017['rating']

# Crate our testing and training sets (2017)
features_train_2017, features_test_2017, labels_train_2017, labels_test_2017 = \
    train_test_split(features_onehot_2017, labels_2017, test_size=0.2)
# In order to make sure the shape is correct later for comparison against the 2020 model, we have to drop a few rows.
labels_test_2017.drop(labels_test_2017.tail(83).index,inplace=True)

# 2020 data prep
filtered_data_2020 = df_2020[['weighted_average_vote', 'budget', 'worldwide_gross_income', 'genre']]
# Change column names for consistency
filtered_data_2020 = filtered_data_2020.rename(columns={'weighted_average_vote' : 'rating', 'worldwide_gross_income' : 'revenue', 'genre': 'genres'})
# The two datasets use two different rating systems. Here we change the ratings to be in range 0-5 for our 2020 data.
filtered_data_2020['rating'] = filtered_data_2020['rating'] / 2

# Returns budgets and revenues which are in USD, and returns a None value if they are not in USD.
def currency_normalization(currency_str):
    if currency_str[0] == '$':
        return float(currency_str[1:])
    else:
        return None

# In the 2020 dataset, there are multiple types of currency! We have to drop all non-USD values so that we can
# compare against the 2017 data.
filtered_data_2020['budget'] = filtered_data_2020['budget'].apply(currency_normalization)
filtered_data_2020['revenue'] = filtered_data_2020['revenue'].apply(currency_normalization)
# Drop the values which weren't in USD.
filtered_data_2020.dropna(inplace=True)

# Filter to the genres in the 2017 dataset [2]
genre_list = ['War', 'Music', 'Thriller', 'Action', 'Western', 'Mystery', 'Fantasy', 'Comedy', 'Drama',
                                    'Horror', 'Adventure', 'Crime', 'Sci-Fi', 'Documentary', 'Family', 'Animation',
                                    'Romance', 'History']
filtered_data_2020 = filtered_data_2020[filtered_data_2020['genres'].isin(genre_list)]

# Since the 'revenue' from the 2020 dataset is actually income, we must convert income into revenue by adding
# the budget and income together. Income = Revenue - Budget, Revenue = Income + Budget.
filtered_data_2020['revenue'] =  filtered_data_2020['revenue'] + filtered_data_2020['budget']

features_2020 = filtered_data_2020.loc[:, filtered_data_2020.columns != 'rating']
# Onehot encode the genere feature since it is a string value
features_onehot_2020 = pd.get_dummies(features_2020)
features_onehot_2020 = features_onehot_2020.rename(columns={'genres_Sci-Fi' : 'genres_Science Fiction'})
# Rating will be our label for this model
labels_2020 = filtered_data_2020['rating']

# Crate our testing and training sets (2020)
features_train_2020, features_test_2020, labels_train_2020, labels_test_2020 = \
    train_test_split(features_onehot_2020, labels_2020, test_size=0.2)

# Initialize and run and our regression model, then generate a prediction set
model = DecisionTreeRegressor()
model.fit(features_train_2017, labels_train_2017)
features_test_prediction = model.predict(features_test_2020)

# Here we can see the accuracy of the model by finding the error between the 2017 labels and the 
# predicted values using the 2020 features.
print(mean_squared_error(labels_test_2020, features_test_prediction))

# # Testing
# During the process of developing our analysis, we had to test and check our data to ensure that we our code was working as intended. Listed below are the ways we tested our code, and how we analyzed our test results.


# Step 1
# - To make sure we had the correct data, we printed df_2017 and df_2020.
# 
# - Noticing the patterns in how df_2017 stored its genres, we were able to create the genre simplification function using a hard-coded approach.
#   
# Step 2
# - We visually assessed how reasonable the answers were using our graphs. Using the printed values from step 1, we were able to ensure that the values for the graphs were reasonable.
#   
# Step 3
# - We tested the accuracy of our model by printing out the mean error between predicted and test values. We were able to reduce the mean error from about 1.2 from earlier versions down to about 0.5 or 0.6 in the current version by modifying our features. We had to ensure that the features we used in this step would be compatible for the 2020 dataset, as well.
#   
# Step 4
# - We performed the same testing as the previous step. Initially the high error was concerning. However, we believe that this should be expected for such a volatile statistic such as raw revenue. 
# 
# Step 5
# - We compared the error between our labels produced from the 2017 model predicting using 2020 features and our 2017 labels. The error we received similar results to step 3, which indicates that our model was effective at predicting data from these films.


# # Sources
# [1] Comma plotting function: [User falsetru from StackOverFlow](https://stackoverflow.com/questions/25973581/how-do-i-format-axis-number-format-to-thousands-with-a-comma-in-matplotlib)
# 
# [2] Filtering method in pandas: [User atinjanki from StackOverFlow](https://stackoverflow.com/questions/59275119/how-can-i-filter-single-column-in-a-dataframe-on-multiple-values)
# 
# Datasets:
# - 2017 Dataset: https://www.kaggle.com/rounakbanik/the-movies-dataset
# - 2020 Dataset: https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset


