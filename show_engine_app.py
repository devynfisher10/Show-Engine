import streamlit as st
import pandas as pd
import numpy as np

# title for web app
st.title('Tv Show Similarity Engine')

# read in data
similarity_matrix = pd.read_csv("symmetric_similarity_matrix.csv")

# insert some graphics for visualization

st.header("Have you ever finished a great TV show and then wondered... what next?")
st.subheader("If you want a similar show to your past favorites, you could check any streaming service")
st.subheader("However, these services will only recommend their own shows")
st.subheader("This web app uses 1000+ of the most popular shows to recommend the most similar options using a universal sentence encoder and a number of other features including genres and ratings")
st.subheader("Simply select a show you like and look at the suggested options!")



@st.cache(ttl=180)
def most_similar(option):
    return similarity_matrix.columns[similarity_matrix.sort_values(option, ascending=False).iloc[1:17,0] + [1]*16]


num_shows_display = st.slider('How many similar shows would you like to see?', 1, 15, 3)


option = st.selectbox(
        'Select Show',
        similarity_matrix.columns[1:], index=0, help="To find a show, start typing or scroll!")


top_results = most_similar(option)


st.subheader("The most similar shows to " + option + str(" are:"))
for i in range(1, num_shows_display+1):
    st.text(top_results[i])





