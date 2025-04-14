# Import the required libraries
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import pandas as pd 
import numpy as np 
import plotly.express as px
import datetime as dt 
import dash
from dash import dcc, html, Input, Output
from io import BytesIO
import base64
import re
from PIL import Image
import matplotlib.pyplot as plt
import random

#import the datasets
df = pd.read_csv('https://raw.githubusercontent.com/AY-Khalid/crimes-in-California/refs/heads/main/word_data.csv')  # Your dataset with columns: word, frequency, year


# remove some unnecessary symbols or words or characters
def clean_word(word):
    return re.sub(r'[^\w\s]', '', word)  # Remove $, (, ), etc.
df['word'] = df['word'].apply(clean_word)
numeric_words = df[df["word"].str.match(r'^\(?\$?\d+(\.\d{1,2})?\)?$', na=False)]["word"].unique()
numeric_words = np.append(numeric_words, ["(17yrs", "from", "with", "over", "see", "in", "under", ".", "simple", "church", "plain", "all"
                                         ,"aggravated", "partner", "motor", "weapon", "deadly","petty","and", "bike", "letters", "person", "identity",
                                         "intimate", "order","17yrs", "yrs", "etc"])
df = df[~df["word"].isin(numeric_words)]
df['word'] = df['word'].replace({'vandalisms':'vandalism', 'overexcptgunsfowllivestkprod':'theft'})

# using stopwords to remove meaningless words
word_stop = set(STOPWORDS)

# update the stopword with some specific words
word_stop.update("simple", "petty", "all", "attempt", "older")


# map of califonia 
image = Image.open("califonia.png").convert("L") 
# image = image.resize((800, 800))  
mask = np.array(image)
mask = np.where(mask > 128, 255, 0).astype(np.uint8) 


#building the dash app 
app = dash.Dash(__name__)
server = app.server  # For deployment

# Get unique years for dropdown
years = sorted(df['year'].unique())

# using custom colors 
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = [
        "#800080",  # purple
        "#00008B",  # dark blue
        "#000000",  # black
        "#8A2BE2",  # blue purple (blueviolet)
        "#FF69B4"   # pink (hotpink)
    ]
    return random.choice(colors)

# Layout
app.layout = html.Div([
    html.H1("Interactive wc (crimes by year)", style={'textAlign': 'center'}),
    
    html.Div([
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in years],
            value=years[0],
            clearable=False,
            style={'width': '50%'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Img(id='wordcloud-img')
    ], style={'textAlign': 'center'})
])

# Callback to update word cloud based on year
@app.callback(
    Output('wordcloud-img', 'src'),
    Input('year-dropdown', 'value')
)
def update_wordcloud(selected_year):
    data = df[df['year'] == selected_year]
    
    # Generate word frequencies dictionary
    freq_dict = dict(zip(data['word'], data['frequency']))
    
    # Create WordCloud with the image mask
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        mask=mask,
        contour_color='white',
        contour_width=1, 
        color_func=custom_color_func # color function defined above
    ).generate_from_frequencies(freq_dict)
    
    # Convert image to base64
    buffer = BytesIO()
    wc.to_image().save(buffer, format='PNG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    
    return f'data:image/png;base64,{encoded_image}'

# Run the app
if __name__ == '__main__':
    app.run_server(debung=True)
