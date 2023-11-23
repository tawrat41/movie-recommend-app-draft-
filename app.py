import streamlit as st  
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
import cv2
from PIL import Image
import pandas as pd
import json


st.set_page_config(
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown("""
    <style>
        # .stApp {
        # margin: auto;
        # text-align: center;
        # }
            body {
            font-family: 'Comic Sans MS', sans-serif !important;
            background-color: #fae0e4;  
        }
        p{
            text-align: justify;
        }        
        h1, h2, h3{
            text-align:center;
        }
        .stImage{
            display: flex;
            justify-content: center;
            # margin-left: auto;
            # margin-right: auto;
        }
        .stButton button{
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        # .stButton button:hover{
        #     background-color: #03045e;
        #     color: white;
        # }

    </style>


""", unsafe_allow_html=True)


session_state = st.session_state
if 'page_index' not in session_state:
    session_state.page_index = 0

st.sidebar.markdown("<h1>Movie Recommendation System</h1>", unsafe_allow_html=True)
section = st.sidebar.radio("Steps to follow - ", ["Introduction", "Types", "Content-based", "Visualize", "Step - 2","Training Initiation", "Machine Learning", "Setup the Model", "Training Parameters", "Train", "Re-Train (if required)", "Step - 3","Test", "Improve Accuracy", "Step - 4", "Conclusion"],  index=session_state.page_index)



if section == "Introduction":
    st.markdown('<div class="center"><h1>Recommendation System</h1></div>', unsafe_allow_html=True)
    image1 = Image.open('media/Screenshot 2023-11-23 115350.png')
    st.image(image1, caption='')

    st.markdown(
        """
                    <p>Hello kids! Today you're going to learn about Recommendation Systems! Imagine having someone with you who always knows exactly want. Someone who helps you decide what books you want to read, what song you want to listen to, or what ice-cream flavour you want to buy! 
                
                    That's a bit like how recommendation systems work! They're like little helpers on apps and websites, suggesting cool videos, games, or books based on what you enjoyed before. They pay attention to what you like, what your friends like, and even the special things that make you, you! 
                
                    Recommendation systems are algorithms that attempt to predict items (movies, music, books, news, web pages) that a user may be interested in. For example, when you watch a video on YouTube, it recommends other videos that might interest you. Similarly, when you buy a product on Amazon, it recommends to you other products that you might be interested in buying.</p> </div>
        """
        , unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        image1 = Image.open('media/youtube-home.png')
        st.image(image1, caption='YouTube Recommendation System')
    with col2:
        image1 = Image.open('media/amazon recommendation system 1.png')
        st.image(image1, caption='Amazon Recommendation System')
    with col3:
        image1 = Image.open('media/movie_recommender_system.jpg')
        st.image(image1, caption='Netflix Recommendation System')

    
elif section == "Types":
    st.markdown('<div class="center"><h1>Types of Recommendation Systems</h1></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("""
                <div>
                    <p>Recommendation systems can be of various types, the most commonly used types are:</p>    
                    <ol>
                        <li>Content-Based Filtering Systems</li>
                        <li>Collaborative Filtering Systems</li>
                    </ol>
                </div>


        """, unsafe_allow_html=True)


    st.markdown('<div class="center"><h3>Content-Based Filtering</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div><p>A Content-based recommender system filters items according to the content. It works on the principle that if a user likes one item, then he / she must like other items that are similar to it.
        In other words, it sees previous items that were liked by the user and suggests items that are similar to them. For example, if a user likes a particular book, then the recommendation system recommends similar books that the reader has not read before. </p></div>""", unsafe_allow_html=True)
    with col2:
        image1 = Image.open('media/Screenshot 2023-11-23 124534.png')
        st.image(image1, caption='Content-Based Filtering')

          
    st.markdown('<div class="center"><h3>Collaborative Filtering</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div><p>Collaborative Filtering filters items according to how similar they are to items selected by other users with similar preferences. It works on the principle that if two or more users have liked similar items, it must mean that they have similar preferences.
        For example if user1 likes apples and mangos and user2 likes strawberries and mangoes. Then the recommendation system suggests strawberries to user1 and apples to user2.</p></div>""", unsafe_allow_html=True)


    with col2:
        image1 = Image.open('media/Screenshot 2023-11-23 131048.png')
        st.image(image1, caption='Collaborative Filtering')

elif section == "Content-based":
    st.markdown("""<div class="center"><h1>Let's Create a Content-based Recommendation System</h1></div>""", unsafe_allow_html=True)
    st.markdown("""<div><p>Who doesn't like movies? If you ever watched a movie or TV show on Netflix, you might have noticed that once you're done watching a movie, Netflix recommends you a list of new movies that it thinks you might be interested in. Ever wondered how they do that? Well here's your chance to find out!
    Let's go ahead and learn how a movie recommendation system works by creating one yourself!</p></div>""", unsafe_allow_html=True)    
    image1 = Image.open('media/cover_Netflix2_1600px_web-1280x640 (1).jpg')
    # st.image(image1, caption='Collaborative Filtering')

    st.markdown('<div class="center"><h2>Loading and Exploring the Movie Dataset </h2></div>', unsafe_allow_html=True)
    st.markdown("""<div><p>We will use the <a href="https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"> TMDB Dataset </a>to create our movie recommendation system. This dataset contains data about the top 10,000 Top-rated movies. But before we actually start building the system, we need to first load and explore the data. Click the button below to upload the TMDB dataset:</p></div>""", unsafe_allow_html=True)    

    if st.button("Upload Data"):
        df = pd.read_csv("tmdb_5000_movies.csv")
        df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)])
        column_to_display = ['id', 'title','genres', 'original_language', 'overview', 'popularity', 'release_date', 'vote_average', 'vote_count']
        st.dataframe(df[column_to_display].head())

        st.markdown("""<p>As we can see, the dataset consists of the following data for each movie:</p>""", unsafe_allow_html=True)    


        st.markdown("<ol>", unsafe_allow_html=True)
        st.markdown("<li><b>Index (idx):</b> Also known as the index, this is a unique integer that is used to identify the movie</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Title:</b> This is the title of the movie</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Genre:</b> This is the genre or type of movie (crime, adventure, etc.)</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Original Language:</b> Original language in which the movie is released</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Overview:</b> Summary of the movie</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Popularity:</b> Movie Popularity. This is a numeric value. The higher the value, the more popular the movie</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Release Date:</b> Movie release date</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Vote Average:</b> Average number of votes for the movie</li>", unsafe_allow_html=True)
        st.markdown("<li><b>Vote Count:</b> Number of people who voted</li>", unsafe_allow_html=True)
        st.markdown("</ol>", unsafe_allow_html=True)

        st.markdown("""<p>The dataset that we used here consists of data for 5000 movies!</p>""", unsafe_allow_html=True)   


elif section == "Visualize":

    # st.set_page_config(layout="centered")


    st.markdown("""<div class="center"><h1>Let's visualize some of the data to understand it better </h1></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="center"><p>It's time to now explore the data to get a good understanding of it. Let's first visualize the top movies based on popularity. You can interact with this graph to get more information: </p></div>""", unsafe_allow_html=True)

    df = pd.read_csv("tmdb_5000_movies.csv")
    column_to_display = ['id', 'title','genres', 'original_language', 'overview', 'popularity', 'release_date', 'vote_average', 'vote_count']

    df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)])

    # User input for the number of top movies to display
    col1, col2, col3 = st.columns([2,1,2])

    with col1:
        pass
    with col2:
        num_top_movies = st.number_input('Select the number of top movies to display:', min_value=1, max_value=len(df), value=5, format="%d")
    with col3:
        pass


    # Display the top movies based on popularity
    top_movies = df.nlargest(num_top_movies, 'popularity')
    st.dataframe(top_movies[column_to_display].reset_index(drop=True))
