import streamlit as st
import pandas as pd
import numpy as np # Import numpy
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- Data Loading and Processing (from your notebook) ---
@st.cache_data # Cache data loading and processing
def load_and_process_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')

    # Pivot ratings to create the user-item matrix
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
    final_dataset.fillna(0, inplace=True)

    # Filter out movies and users with too few ratings
    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

    final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
    final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

    # Create the sparse matrix
    csr_data = csr_matrix(final_dataset.values)

    # Reset index to include movieId as a column for easier lookup
    final_dataset.reset_index(inplace=True)

    return movies, final_dataset, csr_data

movies, final_dataset, csr_data = load_and_process_data()

# --- Model Training (from your notebook) ---
@st.cache_resource # Cache the model training
def train_knn_model(_data):
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(_data)
    return knn

knn = train_knn_model(csr_data)


# --- Recommendation Function (updated for case-insensitivity) ---
def get_recommendations(movie_name):
    n_movies_to_reccomend = 10
    # Make the search case-insensitive
    movie_list = movies[movies['title'].str.contains(movie_name, case=False)]

    if len(movie_list):
        # Continue with the rest of your logic
        movie_idx = movie_list.iloc[0]['movieId']

        # Ensure the movie exists in the filtered dataset
        if movie_idx in final_dataset['movieId'].values:
            movie_idx_in_final_dataset = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
            distances, indices = knn.kneighbors(csr_data[movie_idx_in_final_dataset], n_neighbors=n_movies_to_reccomend+1)
            sorteded_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

            recommend_frame = []
            for val in sorteded_indices:
                movie_idx = final_dataset.iloc[val[0]]['movieId']
                idx = movies[movies['movieId'] == movie_idx].index
                recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
            df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend+1))
            return df
        else:
            return "Movie found in original data, but not in the filtered dataset (due to low ratings)."

    else:
        return "Movie Not Found"


# --- Streamlit App Interface ---
st.title("Movie Recommendation App")
movie_input = st.text_input("Enter a movie title:")

if movie_input:
    recommendations = get_recommendations(movie_input)
    if isinstance(recommendations, pd.DataFrame):
        st.write("Recommended Movies:")
        st.dataframe(recommendations)
    else:
        st.write(recommendations)