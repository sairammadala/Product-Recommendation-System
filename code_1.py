from flask import Flask, render_template ,url_for,request,session, redirect
app = Flask(__name__)

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


# Rating dataset
# Read And Handling missing values
ratings = pd.read_csv("rating.csv")

#read product and merging dataset with rating
product = pd.read_csv("product.csv")
df = pd.merge(ratings,product, on= "ProductId" , how="inner")


#product with over 20 ratings
agg_ratings= df.groupby("productName").agg(mean_ratings = ('rating' ,'mean'),
                                        num_of_rating=('rating' , 'count')).reset_index()

agg_ratings_GT20 = agg_ratings[agg_ratings['num_of_rating'] >1]
agg_ratings_GT20.sort_values(by= 'num_of_rating' , ascending=False)
df_GT20 = pd.merge(df,agg_ratings_GT20[['productName']] , on = 'productName' , how = 'inner')

#matrix
matrix = df_GT20.pivot_table(index= 'productName' , columns='UserId' , values= 'rating' )
print("/--------------------------------------------------------------------------------------------------")
#data normalization
matrix_norm = matrix.subtract(matrix.mean(axis= 1),axis = 'rows')
#identify similar user
item_similarity= matrix_norm.T.corr()

print(matrix_norm.columns)
picked_product= 'Apple'
# Loop through not purchase product
for picked_userid in matrix_norm.columns:
        if pd.notnull(matrix_norm.loc[picked_product, picked_userid]):
            # product that the target user has purchase
            picked_userid_purchase = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all') \
                                                 .sort_values(ascending=False)) \
                                    .reset_index() \
                                    .rename(columns={picked_userid: 'rating'})
        

import operator
# Function to recommend product based on a picked product
def recommend_product_based_on_product(picked_product, number_of_similar_items, number_of_recommendations):
    # Similarity score of the picked product with all other product
    picked_product_similarity_score = item_similarity[[picked_product]].reset_index().rename(columns={picked_product: 'similarity_score'})
    print(picked_product_similarity_score)
    
    rating_prediction = {}

    # Loop through not purchase product
    for picked_userid in matrix_norm.columns:
        if pd.notnull(matrix_norm.loc[picked_product, picked_userid]):
            # product that the target user has purchase
            picked_userid_purchase = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all') \
                                                 .sort_values(ascending=False)) \
                                    .reset_index() \
                                    .rename(columns={picked_userid: 'rating'})
            
            # Rank the similarities between the picked user purchase product and the picked product.
            picked_userid_purchase_similarity = pd.merge(left=picked_userid_purchase,
                                                        right=picked_product_similarity_score,
                                                        on='productName',
                                                        how='inner') \
                                                .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
            # Calculate the predicted rating using weighted average of similarity scores and the ratings from the picked user
            predicted_rating = round(np.average(picked_userid_purchase_similarity['rating'],
                                                weights=picked_userid_purchase_similarity['similarity_score']), 6)
            # Save the predicted rating in the dictionary
            rating_prediction[picked_userid] = predicted_rating

    # Sort the dictionary by predicted rating in descending order
    sorted_predictions = sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]

    # Retrieve the product names for the recommended product
    recommended_product = [(product.loc[product['ProductId'] == movie_id]['productName'].values[0], rating) for movie_id, rating in sorted_predictions]

    return recommended_product


users = {
    'muzamil': 'password123',
    'zubair': 'password456',
    'person': '12345'
}

@app.route('/')
def firstPage():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            # Successful login
            return redirect(url_for('index'))
        else:
            # Invalid credentials
            error = 'Invalid username or password. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/index' ,methods=['GET', 'POST'])
def search_recommand():
   if request.method == 'POST':
    search = request.form['search_text']
    picked_product= search
    product_images = {
      'Rice': url_for('static', filename='images/Rice.jpg'),
      'Yogurt': url_for('static', filename='images/Yogurt.jpg'),
      'Cucumber': url_for('static', filename='images/Cucumber.jpg'),
      'Carrot': url_for('static', filename='images/Carrot.jpg'),
      'Milk': url_for('static', filename='images/Milk.jpg'),
      'Spinach': url_for('static', filename='images/Spinach.jpg'),
      'Chicken': url_for('static', filename='images/Chicken.jpg'),
      'Lettuce': url_for('static', filename='images/Lettuce.jpg'),
       'Tomato': url_for('static', filename='images/Tomato.jpg'),
       'Banana': url_for('static', filename='images/Banana.jpg'),
        'Apple': url_for('static', filename='images/Apple.jpg'),
        'Onion': url_for('static', filename='images/Onion.webp'),
         'Bread': url_for('static', filename='images/Bread.webp') ,
         'Orange': url_for('static', filename='images/Orange.webp') ,
         'Potato': url_for('static', filename='images/Potato.jpg') 
        }
    recommended_product = recommend_product_based_on_product(picked_product=search, number_of_similar_items=2, number_of_recommendations=10)
    recommended_product_names = [movie[0] for movie in recommended_product]    
    return render_template('index.html',  picked_product=picked_product, recommended_product=recommended_product_names,product_images=product_images)
    
    return render_template('index.html')
    




if __name__ == '__main__':
   app.run(debug = True)

