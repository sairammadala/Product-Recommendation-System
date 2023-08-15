from flask import Flask, render_template ,url_for,request,session, redirect
app = Flask(__name__)
app.secret_key = "product_recommendation"

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Rating dataset
# Read And Handling missing values
#ratings = pd.read_csv("rating.csv")

#read product and merging dataset with rating
# product = pd.read_csv("product.csv")
# df = pd.merge(ratings,product, on= "ProductId" , how="inner")


# #product with over 20 ratings
# agg_ratings= df.groupby("productName").agg(mean_ratings = ('rating' ,'mean'),
#                                         num_of_rating=('rating' , 'count')).reset_index()

# agg_ratings_GT20 = agg_ratings[agg_ratings['num_of_rating'] >1]
# agg_ratings_GT20.sort_values(by= 'num_of_rating' , ascending=False)
# df_GT20 = pd.merge(df,agg_ratings_GT20[['productName']] , on = 'productName' , how = 'inner')

# #matrix
# matrix = df_GT20.pivot_table(index= 'productName' , columns='UserId' , values= 'rating' )
# print("/--------------------------------------------------------------------------------------------------")
# #data normalization
# matrix_norm = matrix.subtract(matrix.mean(axis= 1),axis = 'rows')
# #identify similar user
# item_similarity= matrix_norm.T.corr()

# #print(matrix_norm.columns)
# picked_product= 'Apple'
# # Loop through not purchase product
# for picked_userid in matrix_norm.columns:
#         if pd.notnull(matrix_norm.loc[picked_product, picked_userid]):
#             # product that the target user has purchase
#             picked_userid_purchase = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all') \
#                                                  .sort_values(ascending=False)) \
#                                     .reset_index() \
#                                     .rename(columns={picked_userid: 'rating'})
df = pd.read_csv("./archive/dataset-itembased.csv")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)    

import operator
# Function to recommend product based on a picked product
def recommend_product_based_on_product(picked_product):
    # Calculate TF-IDF for the entered keywords
    tfidf_matrix_keywords = tfidf_vectorizer.transform([picked_product])
    cosine_sim_keywords = linear_kernel(tfidf_matrix_keywords, tfidf_matrix).flatten()
    
    
    
    # Get the indices of items that match the entered keywords
    keyword_matching_indices = [[i,score] for i, score in enumerate(cosine_sim_keywords) if score > 0]
    keyword_matching_indices = sorted(keyword_matching_indices, key=lambda x:x[1], reverse=True)
    
    # Create a DataFrame to store recommended items
    recommended_items = pd.DataFrame(columns=df.columns)
    
    for item_idx in keyword_matching_indices:
        recommended_items.loc[len(recommended_items)] = df.iloc[item_idx[0]]

    #remaining code
            
    return recommended_items[:100]

    


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
            session['username'] = username
            return redirect(url_for('index'))
        else:
            # Invalid credentials
            error = 'Invalid username or password. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/index')
def index():

    dataframe = pd.read_csv('product.csv')
    product_ids_dict = dataframe.set_index('productName')['ProductId'].to_dict()


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
         'Orange': url_for('static', filename='images/Orange.webp'),
         'Potato': url_for('static', filename='images/Potato.jpg') 
        }
    
    return render_template('index.html', product_images=product_images)

@app.route('/index' ,methods=['GET', 'POST'])
def search_recommand():
   #reading product.csv to get ids
   dataframe = pd.read_csv('product.csv')
   product_ids_dict = dataframe.set_index('productName')['ProductId'].to_dict()
   #print(product_ids_dict)
   if request.method == 'POST':
    search = request.form['search_text']
    picked_product= search

    #handling user ratings
    if request.form.get("product"):
        rated_product = request.form.get("product")
        if request.form.get(f"{rated_product}-rating"):
                rating = int(request.form.get(f"{rated_product}-rating"))

                # Retrieve user identifier from the session
                user_identifier = session.get('username')

                # Update CSV with user's rating using user_identifier
                #update_csv(user_identifier, picked_product, rating)
                print(user_identifier, rated_product, rating)
                update_csv(1, product_ids_dict[rated_product], rating)
    else:
        print("oops")

    recommended_product = recommend_product_based_on_product(picked_product=search)

    product_images = {}
    recommended_product_names = {}
    print(len(recommended_product))
    items = []

    for index, row in recommended_product.iterrows():
        items.append(row['itemID'])
        product_images[row['itemID']] = row['img_link']
        recommended_product_names[row['itemID']] = row['product_name']
    #print(recommended_product_names)
    print(len(items))
    print(len(set(items)))

    #print(recommended_product_names)  
    return render_template('products.html', recommended_product=recommended_product_names,product_images=product_images, product_ids_dict=product_ids_dict)
    
   return render_template('index.html')
    


def update_csv(user_identifier, product, rating):
    csv_path = 'rating.csv'  # Update with the actual path to your CSV file

    # Read the existing CSV
    ratings_df = pd.read_csv(csv_path)

    # Check if the user has rated the product before
    existing_rating = ratings_df[(ratings_df['UserId'] == user_identifier) & (ratings_df['ProductId'] == product)]

    if existing_rating.empty:
        # User hasn't rated the product before, add a new row
        new_row = {'UserId': user_identifier, 'ProductId': product, 'rating': rating}
        new_row_df = pd.DataFrame([new_row])  # Convert the dictionary to a DataFrame
        ratings_df = pd.concat([ratings_df, new_row_df], ignore_index=True)  # Concatenate with existing DataFrame
    else:
        # Update the existing rating
        ratings_df.loc[existing_rating.index, 'rating'] = rating

    # Save the updated DataFrame back to the CSV
    ratings_df.to_csv(csv_path, index=False)









if __name__ == '__main__':
   app.run(debug = True)

