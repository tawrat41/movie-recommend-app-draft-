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
from datetime import datetime, date
import matplotlib.pyplot as plt


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
        h1, h2, h3, h4, h5{
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
section = st.sidebar.radio("Steps to follow - ", ["Introduction", "Types", "Content-based", "Visualize", "Visualize (Contd)", "How does a Movie Recommendation System Work?"],  index=session_state.page_index)



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


    st.markdown('<div class="center"><h3>Content-Based Filtering</h 3></div>', unsafe_allow_html=True)
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

    st.markdown("""<div class="center"><p>Now let's see do a year-wise analysis of movie releases! Visualize movies between years.     
    </p></div>""", unsafe_allow_html=True)

    # Convert 'release_date' column to datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')



    # Define the layout with columns
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col1:
        pass

    with col2:
        start_year = st.date_input('Select starting year:', min_value=df['release_date'].min().date(), max_value=df['release_date'].max().date(), value=df['release_date'].min().date())

    with col3:
        end_year = st.date_input('Select ending year:', min_value=df['release_date'].min().date(), max_value=df['release_date'].max().date(), value=df['release_date'].max().date())

    with col4:
        pass

    filtered_df = df[(df['release_date'].dt.date >= start_year) & (df['release_date'].dt.date <= end_year)]

    # st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        pass
    with col2:
        if not filtered_df.empty:
            fig, ax = plt.subplots()
            ax.bar(filtered_df['release_date'].dt.year.value_counts().sort_index().index, filtered_df['release_date'].dt.year.value_counts().sort_index())
            st.pyplot(fig)  # Display the chart
        else:
            st.warning('No movies in the selected date range.')
    with col3:
        pass

    st.markdown("""<div class="center"><p>From the above graph, we can easily see which year had the most releases of top-rated movies 
    If you want to see how the number of top-rated movies varied over the years, you can take a look at the line chart below:</div>""", unsafe_allow_html=True)

    # Group by release year and calculate the average rating
    col1, col2, col3 = st.columns(3)

    with col1:
        pass

    with col2:
        movies_by_year = df.groupby(df['release_date'].dt.year).size()

        # Create line chart
        fig, ax = plt.subplots()
        ax.plot(movies_by_year.index, movies_by_year)
        ax.set_xlabel('Release Year')
        ax.set_ylabel('Total Movies Released')

        # Display the line chart in Streamlit
        st.pyplot(fig)
    with col3:
        pass

    st.markdown("""<div class="center"><h3>Can you tell which years made the highest number of top-rated movies?</h3></div>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2,1,1,2])


    with col1:
        pass
    with col2:
        year_range_start = st.number_input('Enter the start year for the range:', min_value=min(df['release_date'].dt.year), max_value=max(df['release_date'].dt.year))
    with col3:
        year_range_end = st.number_input('Enter the end year for the range:', min_value=min(df['release_date'].dt.year), max_value=max(df['release_date'].dt.year))
    with col4:
        pass

    if st.button("Submit"):
        if 2000 <= year_range_start <= 2017 and 2000 <= year_range_end <= 2017 and year_range_start <= year_range_end:
            st.success(f'That\'s correct! The selected year range is {year_range_start}-{year_range_end}.')
        else:
            st.warning('Wrong answer! The correct year range is 2000-2017.')
        

elif section == "Visualize (Contd)":
    st.markdown("""<div class="center"><h1>Visualization of Movie Data (Contd)</h1></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="center"><p>Let's look at a pie chart representing the percentage of movies in different languages:</p></div>""", unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        df = pd.read_csv("tmdb_5000_movies.csv")

        # Count the number of movies in each language
        language_counts = df['original_language'].value_counts()

        # Consider only the top 5 languages and group the rest as "Others"
        top_languages = language_counts.head(7)
        other_languages_count = language_counts[7:].sum()
        top_languages['Others'] = other_languages_count

        # Create a pie chart with adjusted label size
        fig, ax = plt.subplots()
        ax.pie(top_languages, labels=top_languages.index, startangle=0, autopct='', textprops={'fontsize': 6}, labeldistance=1.1)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


        # Display the pie chart in Streamlit
        st.pyplot(fig)
    
    with col2:
        st.markdown("""<div class="center"><p>From the above pie chart, which language movies have the largest share in the dataset? (top 3)
                    </p></div>""", unsafe_allow_html=True)
        
        # Allow users to input their guesses for the top 3 languages
        guess1 = st.selectbox('Guess 1:', [''] + list(top_languages.index), key='guess1')
        if guess1 == 'en':
                st.image(Image.open('media/correct.png').resize((30, 30)), use_column_width=False)
        else:
            st.image(Image.open('media/cross.png').resize((30, 30)), use_column_width=False)

        guess2 = st.selectbox('Guess 2:', [''] + list(top_languages.index), key='guess2')
        if guess1 == 'en':
                st.image(Image.open('media/correct.png').resize((30, 30)), use_column_width=False)
        else:
            st.image(Image.open('media/cross.png').resize((30, 30)), use_column_width=False)

        guess3 = st.selectbox('Guess 3:', [''] + list(top_languages.index), key='guess3')
        if guess1 == 'en':
                st.image(Image.open('media/correct.png').resize((30, 30)), use_column_width=False)
        else:
            st.image(Image.open('media/cross.png').resize((30, 30)), use_column_width=False)


    st.markdown("""<div class="center"><h5>You can also view names of movies from a given genre. Select a genre from dropdown to see which movies match it: </h5></div>""", unsafe_allow_html=True)

    df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)])
    # Allow users to select a genre
    

    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        pass
    with col2:
        all_genres = set(genre for genres_list in df['genres'] for genre in genres_list)
        selected_genre = st.selectbox('Select a genre:', [''] + list(all_genres))
    # Display the DataFrame with the selected genre
    if selected_genre:
        filtered_df = df[df['genres'].apply(lambda genres: selected_genre in genres)]
        st.dataframe(filtered_df[['id', 'title', 'genres', 'original_language', 'overview', 'popularity', 'release_date', 'vote_average', 'vote_count']].head())
    with col3:
        pass

    st.markdown("""<div class="center"><h5>Finally, you can see the summary of any movie from the list. Select a movie title to view its summary:</h5></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        pass
    with col2:
        # Allow users to select a movie
        selected_movie = st.selectbox('Select a movie:', [''] + list(df['title']))

    # Display the 'overview' of the selected movie
    if selected_movie:
        overview = df[df['title'] == selected_movie]['overview'].iloc[0]
        st.write(f'Overview for {selected_movie}:\n{overview}')
    with col3:
        pass


elif section == "How does a Movie Recommendation System Work?":
    st.markdown("""<div class="center"><h1>How does a Movie Recommendation System Work?</h1></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="center"><p>
    Data plays a very important role in ML projects, including the movie recommendation system, so it's important to properly understand the dataset. The best way to understand the data is to explore the dataset, just the way you did up till now. </p>
    <p>Now that we have explored the data, we are ready to start building the Movie Recommendation System! </p>
    <p>The primary goal of movie recommendation systems is to filter and predict movies that a given user is most likely to want to watch. One way is to filter the dataset based on contents of the movie that the user liked, and then find other movies with similar content. We can get a good idea about the contents of a movie from its overview (description). </p>
    <p>However, as we just saw, the overview is in an unstructured text format. However, the computer needs the data in a structured format in order to work with it. So, we need to transform this textual data into numerical representations, which can be used to train the system. This process of converting text into numbers is called vectorization. </p></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="center"><h2>Vectorization</h2></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="center"><p>
    Vectorization is the process of mapping words into vectors of real numbers. In this case, a vector is a sort of list of numbers used to represent a given text.  </p>
    <ol>Once we get the text data converted into vectors, we can easily use this to:
        <li>Find similar or related words. </li>
        <li>Find relationships between words.</li>
        <li>Measure how similar two words are.</li>
    </ol>
    <p>The Bag of Words model is one such way that helps convert words in a document to numerical representations. It provides a way to extract the important features of a text document so that it is useful for the training.  How does it do this? By specifying the number of times each unique word appears in a document. </p>
    <p>Here’s how a bag of words is created. Say you have a document that has a single sentence:</p>
    <p>We first need to tokenize the document (separate the sentence into individual words). After tokenization, each word is called a token.</p>
    <p>This document can be tokenized into the following bag of words, along with how many times each word appears in the document:</p>                    
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(Image.open('media/2023-11-09_20-44-42 1.jpg'))
    with col2:
        st.image(Image.open('media/61c9a99f4e761d37f3a5cf5f_bag of words.png'))