
from flask import Flask, render_template, url_for, request, session, redirect
from flask import jsonify

app = Flask(__name__)
app.secret_key = "product_recommendation"


import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic  # Using user-based k-Nearest Neighbors


# Rating datasetwu
# Read And Handling missing values
# ratings = pd.read_csv("rating.csv")

# read product and merging dataset with rating
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
df = pd.read_csv("./dataset-itembased.csv")
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["description"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

import operator


# Function to recommend product based on a picked product


def get_related_items(categories):
    related_items = pd.DataFrame(columns=df.columns)
    if categories == None:
        return related_items
    c = 0
    for index, row in df.iterrows():
        if c > 50:
            break
        if any(category in row["category"] for category in categories.split("|")):
            # related_items = related_items.append(row, ignore_index=True)
            # print(row)
            related_items.loc[len(related_items)] = row
            c += 1
    return related_items


def recommend_product_based_on_user(user_id):
    csv_file_path = "./dataset-userbased.csv"
    csv_file_path_item = "./dataset-itembased.csv"
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file_path)
    df_itembased = pd.read_csv(csv_file_path_item)
    reader = Reader(rating_scale=(1, 5))

    # Load the dataset
    dataset = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)

    # Train-test split
    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Create the model - Using user-based k-Nearest Neighbors
    model = KNNBasic(
        sim_options={"user_based": True}
    )  # User-based collaborative filtering

    # Train the model on the training set
    model.fit(trainset)

    # ... (same code for getting user_id and num_recommendations) ...
    # user_id = "AG3D6O4STAQKAY2UVGEUV46KN35Q"

    num_neighbors = 30

    # Find similar users to the target user
    similar_users = model.get_neighbors(trainset.to_inner_uid(user_id), k=num_neighbors)

    # Collect the items that the similar users have interacted with but the target user hasn't
    user_items = df[df["userID"] == user_id]["itemID"]
    items_to_recommend = set()
    for neighbor_user_id in similar_users:
        neighbor_items = df[df["userID"] == trainset.to_raw_uid(neighbor_user_id)][
            "itemID"
        ]
        items_to_recommend.update(neighbor_items)

    items_to_recommend = list(items_to_recommend - set(user_items))

    # Predict ratings for items and sort predictions
    predictions = [
        (item, model.predict(user_id, item).est) for item in items_to_recommend
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get the top recommended items
    top_recommendations = predictions[:]

    user_product_names = {}
    user_product_images = {}
    user_product_cost = {}

    # Print the top recommendations
    for item, estimated_rating in top_recommendations:
        user_product_names[item] = df_itembased.loc[
            df_itembased["itemID"] == item, "product_name"
        ].values[0]
        user_product_images[item] = df_itembased.loc[
            df_itembased["itemID"] == item, "img_link"
        ].values[0]
        user_product_cost[item] = df_itembased.loc[
            df_itembased["itemID"] == item, "cost"
        ].values[0][3:]

    return [user_product_images, user_product_names, user_product_cost]


def recommend_product_based_on_product(picked_product):
    # Calculate TF-IDF for the entered keywords
    tfidf_matrix_keywords = tfidf_vectorizer.transform([picked_product])
    cosine_sim_keywords = linear_kernel(tfidf_matrix_keywords, tfidf_matrix).flatten()

    # Get the indices of items that match the entered keywords
    keyword_matching_indices = [
        [i, score] for i, score in enumerate(cosine_sim_keywords) if score > 0
    ]

    keyword_matching_indices = sorted(keyword_matching_indices, key=lambda x: x[1], reverse=True)

    recommended_items = pd.DataFrame(columns=df.columns)

    for item_idx in keyword_matching_indices:
        recommended_items.loc[len(recommended_items)] = df.iloc[item_idx[0]]

    # remaining code
    keyword_matching_categories = []
    for i in keyword_matching_indices:
        keyword_matching_categories.append(df.iloc[i[0]]["category"])
    related_items = get_related_items(keyword_matching_categories[0])
    print(recommended_items)

    return [recommended_items[:100], related_items]


# recommend_product_based_on_product("boat")

# users = {"muzamil": "password123", "zubair": "password456", "person": "12345"}
user_name_id = {
    "person1": "AG3D6O4STAQKAY2UVGEUV46KN35Q",
    "person2": "AECPFYFQVRUWC3KGNLJIOREFP5LQ",
    "person3": "AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA",
}

user_id_pass = {
    "AG3D6O4STAQKAY2UVGEUV46KN35Q": "1234",
    "AECPFYFQVRUWC3KGNLJIOREFP5LQ": "1234",
    "AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA": "1234",
}


@app.route("/")
def firstPage():
    return render_template("login-new.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if (
            username in user_name_id
            and user_id_pass[user_name_id[username]] == password
        ):
            # Successful login
            session["username"] = username
            session["userid"] = user_name_id[username]
            return redirect(url_for("index"))
        else:
            # Invalid credentials
            error = "Invalid username or password. Please try again."
            return render_template("login-new.html", error=error)
    return render_template("login-new.html")


# @app.route("/signup")
# def form():
#     return render_template("signup.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    # UPDATING RATING IN ITEM BASED AND USERBASED
    if request.form.get("product_id"):
        rated_product = request.form.get("product_id")
        print("RATED PRODUCT")
        print(rated_product)
        if request.form.get(f"{rated_product}-rating"):
            rating = int(request.form.get(f"{rated_product}-rating"))

            # Retrieve user identifier from the session
            user_identifier = session.get("userid")

            update_csv(user_identifier, rated_product, rating)
    else:
        print(request.form)
        username = request.form.get("username")
        password = request.form.get("password")

        selected_checkboxes = []
        for checkbox_name in [
            "Electronics",
            "Cables&Accessories",
            "Computers&Accessories",
        ]:
            if checkbox_name in request.form:
                selected_checkboxes.append(request.form[checkbox_name])

        print("updating", [username, password, selected_checkboxes])
        update_user([username, password, "|".join(selected_checkboxes)])
        df = pd.read_csv("dataset-users.csv")

        session["userid"] = df.loc[df["username"] == username, "userID"].values[0]
        # GETTING RELATED ITEMS
        selected_categories = "|".join(selected_checkboxes)
        session["selected_categories"] = selected_categories
    related_df = get_related_items(session.get("selected_categories"))

    related_product_images = {}
    related_product_names = {}
    related_product_cost = {}

    for index, row in related_df.iterrows():
        related_product_images[row["itemID"]] = row["img_link"]
        related_product_names[row["itemID"]] = row["product_name"]
        related_product_cost[row["itemID"]] = row["cost"][3:]

    return render_template(
        "index.html",
        product_images=related_product_images,
        product_names=related_product_names,
        product_cost=related_product_cost,
    )


@app.route("/addtocart", methods=["GET", "POST"])
def addtocart():
    user_id = session.get("userid")
    product_id = request.form.get("product_id")
    product_name = request.form.get("product_name")
    product_img = request.form.get("product_image")
    product_cost = request.form.get("product_cost")
    update_cart(product_id, product_name, product_img, user_id, product_cost)
    return "", 204


@app.route("/cart_items", methods=["GET", "POST"])
def cart_items():
    df = pd.read_csv("./cart.csv")
    cart_item_ids = []
    product_images = {}
    product_names = {}
    product_cost = {}
    for index, row in df.iterrows():
        if row["user_id"] == session["userid"]:
            cart_item_ids.append(row["itemID"])
            product_images[row["itemID"]] = row["img_link"]
            product_names[row["itemID"]] = row["item_name"]
            product_cost[row["itemID"]] = row["cost"]

    return render_template(
        "cart_items.html",
        cart_item_ids=cart_item_ids,
        product_names=product_names,
        product_images=product_images,
        product_cost=product_cost,
    )


@app.route("/remove", methods=["GET", "POST"])
def remove():
    product_id = request.form.get("product_id")
    df = pd.read_csv("./cart.csv")
    mask = df["itemID"] == product_id
    df = df[~mask]
    df.to_csv("./cart.csv", index=False)
    return redirect(url_for("cart_items"))
    # return requests.get("http://127.0.0.1:5000/cart-items")


@app.route("/buy", methods=["GET", "POST"])
def buy():
    df_cart = pd.read_csv("./cart.csv")
    df_item = pd.read_csv("./dataset-itembased.csv")
    for index, row in df_cart.iterrows():
        row = df_item[df_item["itemID"] == row["itemID"]].iloc[0]
        new_row = {
            "userID": session["userid"],
            "itemID": row["itemID"],
            "rating": row["rating"],
            "product_name": row["product_name"],
            "category": row["category"],
            "cost": row["cost"],
            "description": row["description"],
            "img_link": row["img_link"],
        }
        new_item = pd.DataFrame([new_row])
        df_item = pd.concat([df_item, new_item], ignore_index=True)
        df_item.to_csv("./dataset-itembased.csv", index=False)
    print("Added cart items to item based")
    clear_cart(session["userid"])
    return render_template("buy.html")


@app.route("/categories", methods=["GET", "POST"])
def categories():
    # print(request)
    request_data = request.args.get("param1")
    # request_data = request.json
    # print(request_data.get("link"))
    print("requst data ", request_data)
    # print(request_data)

    clicked_link = request_data
    print("link name in cod1e", clicked_link)
    df = get_related_items(clicked_link)
    product_images = {}
    product_names = {}
    product_cost = {}
    for index, row in df.iterrows():
        product_images[row["itemID"]] = row["img_link"]
        product_names[row["itemID"]] = row["product_name"]
        product_cost[row["itemID"]] = row["cost"][3:]

    return render_template(
        "index.html",
        product_images=product_images,
        product_names=product_names,
        product_cost=product_cost,
    )
    # return "", 204


def clear_cart(user):
    df = pd.read_csv("./cart.csv")
    # df.drop(df.index, inplace=True)
    df = df[df["user_id"] != user]
    df.to_csv("./cart.csv", index=False)
    print("Successfully cleared cart")


def update_cart(product_id, product_name, product_img, user_id, product_cost):
    path = "./cart.csv"
    old_cart = pd.read_csv(path)
    new_row = {
        "itemID": product_id,
        "item_name": product_name,
        "img_link": product_img,
        "user_id": session.get("userid"),
        "cost": product_cost,
    }
    new_item = pd.DataFrame([new_row])
    new_cart = pd.concat([old_cart, new_item], ignore_index=True)
    new_cart.to_csv(path, index=False)


@app.route("/index", methods=["GET", "POST"])
def index():
    images, names, cost = recommend_product_based_on_user(session["userid"])

    if request.form.get("product_id"):
        rated_product = request.form.get("product_id")
        print("RATED PRODUCT")
        print(rated_product)
        if request.form.get(f"{rated_product}-rating"):
            rating = int(request.form.get(f"{rated_product}-rating"))

            # Retrieve user identifier from the session
            user_identifier = session.get("userid")

            # Update CSV with user's rating using user_identifier
            # update_csv(user_identifier, picked_product, rating)
            # print("RATING")
            # print(user_identifier, rated_product, rating)
            update_csv(user_identifier, rated_product, rating)
        else:
            print("oops")
    return render_template(
        "index.html", product_images=images, product_names=names, product_cost=cost
    )


@app.route("/search", methods=["GET", "POST"])
def search_recommand():
    # reading product.csv to get ids
    dataframe = pd.read_csv("product.csv")
    product_ids_dict = dataframe.set_index("productName")["ProductId"].to_dict()
    # print(product_ids_dict)
    if request.method == "POST":
        search = request.form["search_text"]
        print("search", search)
        picked_product = search

        # handling user ratings
        if request.form.get("product_id"):
            rated_product = request.form.get("product_id")
            print("RATED PRODUCT")
            print(rated_product)
            if request.form.get(f"{rated_product}-rating"):
                rating = int(request.form.get(f"{rated_product}-rating"))

                # Retrieve user identifier from the session
                user_identifier = session.get("userid")

                # Update CSV with user's rating using user_identifier
                # update_csv(user_identifier, picked_product, rating)
                print("RATING")
                print(user_identifier, rated_product, rating)
                update_csv(user_identifier, rated_product, rating)
        else:
            print("oops")

        recommended_product, related_products = recommend_product_based_on_product(
            picked_product=search
        )

        product_images = {}
        recommended_product_names = {}
        recommended_product_cost = {}
        print(len(recommended_product))
        items = []

        for index, row in recommended_product.iterrows():
            items.append(row["itemID"])
            product_images[row["itemID"]] = row["img_link"]
            recommended_product_names[row["itemID"]] = row["product_name"]
            recommended_product_cost[row["itemID"]] = row["cost"][3:]
        # print(recommended_product_names)
        print(len(items))
        print(len(set(items)))

        # RELATED PRODUCTS DIC BUILDING
        related_product_images = {}
        related_product_names = {}
        related_product_cost = {}
        for index, row in related_products.iterrows():
            related_product_images[row["itemID"]] = row["img_link"]
            related_product_names[row["itemID"]] = row["product_name"]
            related_product_cost[row["itemID"]] = row["cost"][3:]
        print("related", len(related_product_names))
        # print(recommended_product_names)
        return render_template(
            "products.html",
            recommended_product=recommended_product_names,
            product_images=product_images,
            product_ids_dict=product_ids_dict,
            related_product_images=related_product_images,
            related_product_names=related_product_names,
            picked_product=search,
            product_cost=recommended_product_cost,
            related_product_cost=related_product_cost,
        )

    return render_template("index.html")


def update_user(li):
    csv_path = "./dataset-users.csv"
    df = pd.read_csv(csv_path)
    id = chr(ord(df.loc[len(df) - 1]["userID"]) + 1)
    row = {
        "userID": id,
        "username": li[0],
        "password": li[1],
        "categories": li[2],
    }
    new_df = pd.DataFrame([row])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def update_csv(user_identifier, product, rating):
    csv_path_1 = (
        "./dataset-itembased.csv"  # Update with the actual path to your CSV file
    )
    csv_path_2 = "./dataset-userbased.csv"
    # Read the existing CSV
    ratings_df_1 = pd.read_csv(csv_path_1)
    ratings_df_2 = pd.read_csv(csv_path_2)
    # Check if the user has rated the product before
    existing_rating_1 = ratings_df_1[
        (ratings_df_1["userID"] == user_identifier)
        & (ratings_df_1["itemID"] == product)
    ]

    existing_rating_2 = ratings_df_2[
        (ratings_df_2["userID"] == user_identifier)
        & (ratings_df_2["itemID"] == product)
    ]

    if existing_rating_1.empty:
        # User hasn't rated the product before, add a new row

        # ITEMBASED
        new_row_1 = {
            "userID": user_identifier,
            "itemID": product,
            "rating": rating,
            "product_name": ratings_df_1.loc[
                ratings_df_1["itemID"] == product, "product_name"
            ].values[0],
            "category": ratings_df_1.loc[
                ratings_df_1["itemID"] == product, "category"
            ].values[0],
            "cost": ratings_df_1.loc[ratings_df_1["itemID"] == product, "cost"].values[
                0
            ],
            "description": ratings_df_1.loc[
                ratings_df_1["itemID"] == product, "description"
            ].values[0],
            "img_link": ratings_df_1.loc[
                ratings_df_1["itemID"] == product, "img_link"
            ].values[0],
        }
        new_row_df_1 = pd.DataFrame(
            [new_row_1]
        )  # Convert the dictionary to a DataFrame
        ratings_df_1 = pd.concat(
            [ratings_df_1, new_row_df_1], ignore_index=True
        )  # Concatenate with existing DataFrame
        print(new_row_1)

        # USERBASED
        new_row_2 = {
            "userID": user_identifier,
            "itemID": product,
            "rating": rating,
            "img_link": ratings_df_2.loc[
                ratings_df_2["itemID"] == product, "img_link"
            ].values[0],
        }
        new_row_df_2 = pd.DataFrame(
            [new_row_2]
        )  # Convert the dictionary to a DataFrame
        ratings_df_2 = pd.concat(
            [ratings_df_2, new_row_df_2], ignore_index=True
        )  # Concatenate with existing DataFrame
        print(new_row_2)
    else:
        # Update the existing rating
        ratings_df_1.loc[existing_rating_1.index, "rating"] = rating
        print("else", rating)

        ratings_df_2.loc[existing_rating_2.index, "rating"] = rating
        print("else 2", ratings_df_2.loc[existing_rating_2.index, "rating"])

    # Save the updated DataFrame back to the CSV
    ratings_df_1.to_csv(csv_path_1, index=False)
    ratings_df_2.to_csv(csv_path_2, index=False)


if __name__ == "__main__":
    app.run(debug=True)
