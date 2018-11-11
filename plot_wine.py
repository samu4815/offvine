import csv
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


wine_df = pd.read_csv("resources/winemag-data-with-title.csv")

no_price_outliers = wine_df[((wine_df.price - wine_df.price.mean()) / wine_df.price.std()).abs() < 3]

no_price_outliers['vintage'] = no_price_outliers['title'].str.extract('(\d\d\d\d)', expand=False).str.strip()

no_price_outliers = no_price_outliers[no_price_outliers.vintage != "NaN"]

relevant_cols = no_price_outliers[['country', 'vintage', 'points']]

not_low_age = relevant_cols['vintage'] > "1900"

not_high_age = relevant_cols['vintage'] < "2019"

relevant_cols_normal = relevant_cols[not_high_age & not_low_age]

avg_points_by_vintage_country = relevant_cols_normal.groupby(['vintage', 'country']).agg(np.mean)

avg_points_by_vintage_country = avg_points_by_vintage_country.reset_index()

median_points = avg_points_by_vintage_country['points'].agg(np.median)

# avg_points_by_vintage_country.to_csv('./output.csv')

avg_points_by_vintage_country['quality'] = np.where(avg_points_by_vintage_country['points'] > median_points, 1, 0)

# print avg_points_by_vintage_country