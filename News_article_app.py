import pandas as pd
import streamlit as st
import os
import pickle
import re
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

def extract_url_category(data, url_column):
    """
    Extracts a category from a URL column based on a list of predefined categories.
    If no category is found, attempts to extract the third segment dynamically.

    Args:
        data (pd.DataFrame): The input DataFrame.
        url_column (str): The column name containing URLs.

    Returns:
        pd.DataFrame: The updated DataFrame with a new 'Url_Category' column.
    """
    import re

    # List of predefined categories
    categories = ['business', 'education', 'entertainment', 'technology', 'sports']

    # Regex to match predefined categories
    category_pattern = r'(' + '|'.join(categories) + r')'

    # Extract categories based on the predefined list
    data['Url_Category'] = data[url_column].str.extract(category_pattern, expand=False)

    # Define fallback regex to extract the third segment dynamically
    fallback_pattern = r'^(?:https?://)?[^/]+/[^/]+/([^/]+)/'

    # Apply fallback regex only to rows where Url_Category is still NaN
    data['Url_Category'] = data.apply(
        lambda row: re.search(fallback_pattern, row[url_column]).group(1)
        if pd.isna(row['Url_Category']) and re.search(fallback_pattern, row[url_column])
        else row['Url_Category'],
        axis=1
    )

    # Fill any remaining NaN values with 'url-web'
    data['Url_Category'] = data['Url_Category'].fillna('url-web')

    # Drop the original URL column
    data = data.drop(columns=[url_column])

    # Rearrange columns for consistency
    data = data[['Headlines', 'Description', 'Content', 'Url_Category']]

    return data

def clean_text(text):
    """
    Clean a given text by performing the following operations:
    - Convert the text to lowercase.
    - Remove all non-alphabetic characters (including digits, punctuation, and special characters).
    - Remove extra spaces.
    - Strip leading and trailing spaces.

    Parameters:
    text (str): The raw text that needs to be cleaned.

    Returns:
    str: The cleaned text.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove alphanumeric characters, special characters, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

def tokenize_text(text):
    """
    Tokenizes a given text into words (tokens).
    
    This function uses the `word_tokenize` method from the NLTK library
    to split the input text into individual words, removing punctuation 
    and separating the text into meaningful chunks for further processing.

    Parameters:
    text (str): The text to be tokenized.
    
    Returns:
    list: A list of tokens (words) extracted from the input text.
    """
    # Tokenize the input text into words
    tokens = word_tokenize(text)
    
    # Return the list of tokens
    return tokens

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.
    
    Stopwords are common words that do not carry significant meaning 
    for text analysis (e.g., "the", "and", "is"). This function filters 
    out those words from the input token list.

    Parameters:
    tokens (list): A list of tokens (words) from which stopwords will be removed.
    
    Returns:
    list: A list of tokens with stopwords removed.
    """

    # Load the set of stopwords for the English language
    stop_words = set(stopwords.words('english'))

    # Filter out stopwords from the list of tokens
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Return the filtered list of tokens
    return filtered_tokens

def lemmatize_tokens(tokens):
    """
    Lemmatizes a list of tokens (words) to their base form.

    Parameters:
    tokens (list): A list of words (strings) to be lemmatized.

    Returns:
    list: A list of lemmatized words.
    
    Example:
    >>> lemmatize_tokens(["running", "better", "cats"])
    ['run', 'better', 'cat']
    """
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize each word in the tokens list and return the results in a new list
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    """
    Preprocesses the input text by performing tokenization, stopword removal,
    and lemmatization in sequence.

    The steps include:
    1. Tokenizing the text into individual words.
    2. Removing stopwords (common words with little meaning for analysis).
    3. Lemmatizing the remaining tokens to their base forms.

    Parameters:
    text (str): The raw text that needs to be processed.

    Returns:
    list: A list of preprocessed tokens (words).
    """
    # Tokenize the text into words
    tokens = tokenize_text(text)
    
    # Remove stopwords from the list of tokens
    tokens = remove_stopwords(tokens)
    
    # Lemmatize the remaining tokens
    tokens = lemmatize_tokens(tokens)
    
    # Return the preprocessed list of tokens
    return tokens

def predict(headline,description,content,url,model):
    input_data = [[headline, description, content, url]]
    column_name = ["Headlines", "Description", "Content", "Url"]

    df = pd.DataFrame(data = input_data, columns = column_name)
    
    df = extract_url_category(df, 'Url')

    df["Headlines"] = df["Headlines"].apply(clean_text)  
    df["Description"] = df["Description"].apply(clean_text)  
    df["Content"] = df["Content"].apply(clean_text)  
    df["Url_Category"] = df["Url_Category"].apply(clean_text)

    df["Headlines"] = df["Headlines"].apply(preprocess_text)  
    df["Description"] = df["Description"].apply(preprocess_text)  
    df["Content"] = df["Content"].apply(preprocess_text)  
    df["Url_Category"] = df["Url_Category"].apply(preprocess_text)

    df["Headlines"] = df["Headlines"].apply(lambda tokens: ' '.join(tokens))
    df["Description"] = df["Description"].apply(lambda tokens: ' '.join(tokens))
    df["Content"] = df["Content"].apply(lambda tokens: ' '.join(tokens))
    df["Url_Category"] = df["Url_Category"].apply(lambda tokens: ' '.join(tokens))

    with open('Vectorizer_vocabulary.txt', 'r') as file:
        vocab = [v.strip() for v in file]

    vectorizer = TfidfVectorizer(vocabulary = vocab)

    df['combined'] = df["Headlines"] + df["Description"] + df["Content"] + df["Url_Category"]

    input_set = vectorizer.fit_transform(df['combined']).toarray()

    category_num_nb = models[0].predict(input_set)[0]
    category_num_svm = models[1].predict(input_set)[0]
    category_num_nn = models[2].predict(input_set)[0]

    cat_list = [category_num_nb, category_num_svm, category_num_nn]
    cat_mode = statistics.mode(cat_list)
    
    category_dict = { 0: "Business",1: "Education",2: "Entertainment",3: "Sports",4: "Technology"}

    category = category_dict[cat_mode ]

    return category

model_filenames = ['models/naive_bayes_model.pkl', 'models/svm_model.pkl', 'models/neural_networks_model.pkl']
models = []

for filename in model_filenames:
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        models.append(model)

st.markdown(
    """
    <div style="background-color: black; color: white; padding: 10px; text-align: center;">
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(" ", ["Home", "About the models","About the data", "Category Predictor","About the team"])

# Set up content for each page
if page == "Home":
    home_page_image = "Article_collage.jpg"
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>News Article Category Predictor</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 15, 1])  # Create 3 columns with a larger middle column
    with col2:
        st.image(home_page_image, width=600)
    
    st.write('Welcome to our category predictor! Here you can classify the category of any of your favourite news articles using our sophisticated machine learning models. The predictor takes in your article details, namely the:')
    st.markdown("""
    * Headline: The title of the article that appears above the news story.
    * Description: A brief summary of the key facts mentioned in the article.
    * Content: The entire news story.
    * Url: The web address that directly leads to the news article.
    """)
    st.write('These details are ran by our 3 models simultaneoulsy and the modal category is given as the final result. If each model predicts a different category, the result of the first model will be given. Additionally, they can only identify articles that are of the follwing categories:')
    st.markdown("""
    * Business
    * Education
    * Technology
    * Entertainment
    * Sports
    """)
    st.markdown("Give it a try on the page Category Predictor")
    
elif page == "About the models":
    st.title("About the models")
    st.write("The predictor utilizes 3 efficient machine learning models, namely Support vector machine, Neural networks and Naive bayes classifier. It functions by separately predicting the article category using the models and finding the mode from the results and outputting it as the determined category. Each model was optimised using an optimisation process that identifies suitable hyperparameters through cross validation. Find more information about the models below, along with how they primarily function.")
    
    st.subheader("Support vector machine classifier")
    st.write("This type of model is characterised by a soft margin  that separates the p dimensional feature space into the number of classes available. The hyperplane is of a shape that is one dimension less than the number of features in the feature space. In our case, the hyperplane is of 5 999 dimensions. This division hardly separates the data  perfectly.  The observations that fit on or fall within the separating margin are called support vectors. Adjustment of these vectors alters the shape of the margin. The tuning parameter , C, controls the width of the margin and consequently the  violations allowed by the model. Typically, this model is used in binary classification problems but can be modified to  extend to multiclass problems like our predictor. This is achieved  by considering them as  one vs all. ")
    
    st.subheader("Neural network classifier")
    st.write("It is a model that makes decisions using a series of interconnected decision nodes where values are transformed using activation functions.  These nodes are called neurons and are arranged into multiple layers all the way from the input layer to the output layer. The number of neurons in the input layer are equivalent to the number of features in the training data. The activation functions introduce nonlinearity to the values, ensuring they exist in the bounds of [0,1] before migrating to the next neuron layer. Additionally, they are particular to the layer. In our classifier, each neuron in a  layer is fully connected to the neurons in the subsequent  layer. The model sees each feature # times before casting a prediction.")
    
    st.subheader("Naive bayes classifier")
    st.write("Naive Bayes is a family of probabilistic classifiers based on Bayes' Theorem, which calculates the probability of a class given the input features. This model is built on the assumption that  features are independent given their class lables. This assumption simplifies the model and allows for efficient calculations, even with large datasets. The sub type that is utilised is the Multinomial Naive Bayes model, which is commonly used for text classification. The model predicts the category of a news article based on the frequency of words appearing in the article. The key advantage of Naive Bayes is its speed and simplicity, which makes it well-suited for applications where speed is crucial and the features are largely independent of each other.")
    

elif page == "About the data":
    st.title("About the data")
    st.write("The models were trained on a dataset that consisted of 5284 news articles, with each article belonging to one of the five categories: ")
    st.markdown("""
    * **Business**
    * **Education**
    * **Technology**
    * **Entertainment**
    * **Sports**
    """)
    st.write("The news articles were nearly equally split amongst the categories. Education had the largest composition of 27% and sports had the lowest with 12%. The image below demonstates the count and percentage composition of each category")
    per_comp_image = "data_comp.png"
    st.image(per_comp_image)
    
elif page =="Category Predictor":
    st.title("Category Predictor")
    st.write("Input your news article details below and find out what category it belongs to!")
    
    input_headline = st.text_input("Enter the  headline:")
    input_description = st.text_input("Enter the description:")
    input_content = st.text_input("Enter the content:")
    input_url = st.text_input("Enter the url:")
                
    if st.button("Predict"):
        if input_headline and input_description and input_content and input_url:
            result = predict(input_headline,input_description,input_content,input_url,models)
            st.write("Category:", result)
        else:
            st.warning("Please fill all the fields.")
        
        
elif page == "About the team":
    st.title("Meet the team!")
    st.write("This application was designed by a team of diligent data science students.")
    team_image = "team.png"
    st.image(team_image)

st.markdown(
    """
    <div style="background-color: black; color: white; padding: 10px; text-align: center;">
    </div>
    """,
    unsafe_allow_html=True
)



