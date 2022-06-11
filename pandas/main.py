import pandas as pd

# What is the best wine I can buy for a given amount of money? Create a Series whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that 4.0 dollars is at the top and 3300.0 dollars is at the bottom).

reviews = pd.read_csv('../../test/wine-reviews.csv')

# best_rating_per_price = reviews.groupby(['price'])['points'].max().sort_values(by='price')

price_extremes = reviews.groupby(['variety'])['price'].agg([min, max])
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

reviewer_mean_ratings = reviews.groupby(['taster_name'])['points'].mean()

country_variety_counts = reviews.groupby(['country', 'vriety']).size().sort_values(ascending=False)
print(country_variety_counts)
