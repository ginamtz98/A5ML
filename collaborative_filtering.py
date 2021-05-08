#-------------------------------------------------------------------------
# AUTHOR: Gina Martinez
# FILENAME: collaborative_filtering.py
# SPECIFICATION: Reads the file trip_advisor_data.csv to make user-based recommendations and predict the ratings of user 100
# FOR: CS 4200- Assignment #5
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
userPred = df.lastRow[[-1]]
del userPred['User ID']
del userPred['galleries']
del userPred['restaurants']
r_userPred = 0
sim = []

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
for i, row in df.iteraterow():
    if i == len(df)-1:
        break
    vec1 = np.array([[row['dance clubs'], row['juice bars'], row['museums'], row['resorts'], row['parks/picnic spots'], row['beaches'], row['theaters'], row['religious institutions']]])
    vec2 = np.array([userPred['dance clubs'], userPred['juice bars'], userPred['museums'], userPred['resorts'], userPred['parks/picnic spots'], userPred['beaches'], userPred['theaters'], userPred['religious institutions']]).reshape(1,8)

    if i == 0:
        r_userPred = sum(vec2[0]/ len(vec2[0]))

    cosSim = cosine_similarity(vec1, vec2)
    sim.append((i, cosSim[0][0]))

   #find the top 10 similar users to the active user according to the similarity calculated before
   #--> add your Python code here
top10 = sorted(sim, key = lambda t: t[1], reverse=True)[:10]
print("(User ID, Similarity): ", top10)

   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here
weighted_galleries = []
weighted_restaurants = []
for t in top10:
    weighted_gallery = float(df.loc[t[0], 'galleries']) * t[1]
    weighted_restaurant = float(df.loc[t[0], 'restaurants']) * t[1]
    sum = sum + t[1]
    weighted_galleries.append(weighted_gallery)
    weighted_restaurants.append(weighted_restaurant)



