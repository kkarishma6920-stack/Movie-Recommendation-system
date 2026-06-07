import streamlit as st
import pandas as pd
from recommendation import recommend

movies = pd.read_csv("ml-latest-small/movies.csv")

st.title("Movie Recommendation System")

movie_list = movies["title"].values

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write(movie)