import streamlit as st
import praw
import pandas as pd
import re
from thefuzz import process
import wbdata
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import country_converter as coco
from io import StringIO

# Set Streamlit page configuration
st.set_page_config(page_title="Reddit Migration Analysis", layout="wide")

# Sidebar Inputs
st.sidebar.header("Reddit API Credentials")
client_id = st.sidebar.text_input("Client ID", type="password")
client_secret = st.sidebar.text_input("Client Secret", type="password")
user_agent = st.sidebar.text_input("User Agent", value="Reddit Migration Analysis App")

st.sidebar.header("Data Extraction Settings")
post_limit = st.sidebar.slider("Number of Reddit Posts to Fetch", min_value=100, max_value=2000, value=500, step=100)

st.sidebar.header("Country Extraction")
extraction_method = st.sidebar.selectbox(
    "Select Country Extraction Method",
    options=["Before Arrow (Origin)", "After Arrow (Destination)"]
)

st.sidebar.header("Outlier Removal")
remove_gdp_outliers = st.sidebar.checkbox("Remove Outliers in GDP per Capita", value=True)
remove_occurrence_outliers = st.sidebar.checkbox("Remove Outliers in Occurrences", value=False)

# Fetch Reddit Posts
@st.cache_data(ttl=3600)
def fetch_reddit_posts(client_id, client_secret, user_agent, subreddit_name, limit):
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        for submission in subreddit.new(limit=limit):
            posts.append(submission.title)
        return posts
    except Exception as e:
        st.error(f"An error occurred while fetching Reddit posts: {e}")
        return []

# Check if credentials are provided
if client_id and client_secret and user_agent:
    with st.spinner("Fetching Reddit posts..."):
        posts = fetch_reddit_posts(client_id, client_secret, user_agent, "IWantOut", post_limit)
    if posts:
        st.success("Reddit posts fetched successfully!")
        st.write(f"Total posts fetched: {len(posts)}")
    else:
        st.warning("No posts fetched. Please check your credentials and try again.")
        st.stop()
else:
    st.warning("Please enter your Reddit API credentials in the sidebar to proceed.")
    st.stop()

# Define extraction functions
def extract_countries_before_arrow(title):
    try:
        parts = title.split('->')
        if len(parts) < 2:
            return []
        before_arrow = parts[0]
        before_arrow = before_arrow.replace('[IWantOut]', '').strip()
        countries = re.findall(r'\b[A-Za-z\s]+\b', before_arrow)
        country_list = []
        for part in countries:
            split_countries = [c.strip() for c in part.split(',')]
            country_list.extend(split_countries)
        country_list = [c for c in country_list if c]
        return country_list
    except Exception as e:
        st.error(f"Error parsing title: {title}\nError: {e}")
        return []

def extract_countries_after_arrow(title):
    try:
        parts = title.split('->')
        if len(parts) < 2:
            return []
        after_arrow = parts[1]
        after_arrow = after_arrow.replace('[IWantOut]', '').strip()
        countries = re.findall(r'\b[A-Za-z\s]+\b', after_arrow)
        country_list = []
        for part in countries:
            split_countries = [c.strip() for c in part.split(',')]
            country_list.extend(split_countries)
        country_list = [c for c in country_list if c]
        return country_list
    except Exception as e:
        st.error(f"Error parsing title: {title}\nError: {e}")
        return []

# Extract countries based on user selection
def extract_countries(posts, method):
    if method == "Before Arrow (Origin)":
        extractor = extract_countries_before_arrow
    else:
        extractor = extract_countries_after_arrow
    all_countries = []
    for title in posts:
        extracted = extractor(title)
        all_countries.extend(extracted)
    return all_countries

all_countries = extract_countries(posts, extraction_method)

# Initialize country converter
cc = coco.CountryConverter()

def standardize_country(name):
    try:
        standardized = cc.convert(names=name, to='name_short')
        if standardized == 'not found' or standardized is None:
            return None
        return standardized
    except Exception as e:
        st.error(f"Error standardizing country name '{name}': {e}")
        return None

# Standardize extracted countries
standardized_countries = [standardize_country(c) for c in all_countries]
standardized_countries = [c for c in standardized_countries if isinstance(c, str)]

# Fetch GDP Data
@st.cache_data(ttl=86400)
def fetch_gdp_data(indicator):
    try:
        gdp_data = wbdata.get_dataframe(
            indicators=indicator,
            keep_levels=False
        )
        gdp_data.reset_index(inplace=True)
        
        # Convert 'date' to datetime
        if not pd.api.types.is_datetime64_any_dtype(gdp_data['date']):
            gdp_data['date'] = pd.to_datetime(gdp_data['date'], errors='coerce')
        
        # Drop rows with invalid dates
        gdp_data = gdp_data.dropna(subset=['date'])
        
        # Sort and drop duplicates to keep the most recent
        gdp_data = gdp_data.sort_values(['country', 'date'], ascending=[True, False]).drop_duplicates(subset=['country'], keep='first')
        
        # Select relevant columns
        gdp_data = gdp_data[['country', 'GDP_per_Capita']]
        gdp_data.columns = ['Country', 'GDP_per_Capita']
        
        return gdp_data
    except Exception as e:
        st.error(f"An error occurred while fetching GDP data: {e}")
        return pd.DataFrame()

indicator = {'NY.GDP.PCAP.CD': 'GDP_per_Capita'}

with st.spinner("Fetching GDP per Capita data..."):
    gdp_data = fetch_gdp_data(indicator)
if not gdp_data.empty:
    st.success("GDP per Capita data fetched successfully!")
else:
    st.warning("GDP data is empty. Please try again later.")

# Function to perform fuzzy matching
def match_country(name, official_list, threshold=80):
    if not isinstance(name, str):
        return None
    match = process.extractOne(name, official_list)
    if match and match[1] >= threshold:
        return match[0]
    return None

# Apply fuzzy matching
matched_countries = [match_country(c, gdp_data['Country'].tolist()) for c in standardized_countries]
matched_countries = [c for c in matched_countries if c]

# Count occurrences
country_counts = pd.Series(matched_countries).value_counts().reset_index()
country_counts.columns = ['Country', 'Occurrences']

# Merge with GDP data
merged_data = pd.merge(
    country_counts,
    gdp_data,
    on='Country',
    how='left'
)

# Drop rows with missing GDP per capita
merged_data = merged_data.dropna(subset=['GDP_per_Capita'])

# Remove outliers based on user selection
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

if remove_gdp_outliers:
    merged_data = remove_outliers_iqr(merged_data, 'GDP_per_Capita')
if remove_occurrence_outliers:
    merged_data = remove_outliers_iqr(merged_data, 'Occurrences')

st.write(f"Data points after outlier removal: {merged_data.shape[0]}")

# Visualization
def create_scatter_plot(df):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    sns.scatterplot(
        data=df,
        x='GDP_per_Capita',
        y='Occurrences',
        s=100,
        color='blue',
        edgecolor='w',
        alpha=0.7
    )
    
    sns.regplot(
        data=df,
        x='GDP_per_Capita',
        y='Occurrences',
        scatter=False,
        color='red',
        line_kws={'linewidth':2}
    )
    
    plt.title('Reddit Post Occurrences vs. GDP per Capita', fontsize=16)
    plt.xlabel('GDP per Capita (USD)', fontsize=14)
    plt.ylabel('Number of Occurrences in Reddit Posts', fontsize=14)
    
    # Optional: Log scale
    # plt.xscale('log')
    # plt.yscale('log')
    
    plt.grid(True)
    st.pyplot(plt)

create_scatter_plot(merged_data)

# Display the data
st.header("Processed Data")
st.dataframe(merged_data)

# Download option
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(merged_data)

st.download_button(
    label="Download Processed Data as CSV",
    data=csv_data,
    file_name='reddit_gdp_data_processed.csv',
    mime='text/csv',
)
