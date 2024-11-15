import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import nltk

# Download NLTK resources if not already available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Replace with your Alpha Vantage API key
API_KEY = 'Z4TYQLKEWMNN6YJF'

def get_stock_sentiment(stock_ticker):
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': stock_ticker,
        'apikey': API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'feed' in data:
            sentiments = []
            all_text = ""
            for article in data['feed']:
                sentiment_score = article.get('overall_sentiment_score', 0.0)
                summary = article.get('summary', '')
                
                # Accumulate sentiment score
                sentiments.append(sentiment_score)
                
                # Accumulate text for word cloud
                all_text += " " + summary.lower()
                
            return sentiments, all_text
        else:
            print('No sentiment data found for this stock.')
            return [], ""
    else:
        print(f'Error fetching data: {response.status_code}')
        return [], ""

def classify_sentiments(sentiments):
    positive = sum(1 for score in sentiments if score > 0.05)
    negative = sum(1 for score in sentiments if score < -0.05)
    neutral = sum(1 for score in sentiments if -0.05 <= score <= 0.05)
    return positive, negative, neutral

def plot_sentiment_pie(positive, negative, neutral):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive, negative, neutral]
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution')
    plt.show()

def extract_meaningful_words(text):
    # Define stop words, including finance-specific irrelevant words
    stopwords = set([
        'the', 'and', 'of', 'in', 'to', 'for', 'on', 'a', 'is', 'with', 'that', 'by', 'as', 'at',
        'from', 'it', 'an', 'or', 'be', 'this', 'are', 'has', 'have', 'was', 'were', 'company', 
        'market', 'share', 'price', 'business', 'stock', 'growth', 'industry', 'financial', 'investors'
    ])
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabet characters
    words = [word for word in text.split() if word not in stopwords and len(word) > 3]
    
    # Extract nouns and adjectives for more meaningful terms
    words = [word for (word, pos) in nltk.pos_tag(words) if pos in ('NN', 'NNS', 'JJ')]
    
    # Count word frequencies and filter out words that are not frequent
    word_counts = Counter(words)
    common_words = {word: count for word, count in word_counts.items() if count > 2}  # Only include words that appear more than twice

    return common_words

def create_wordcloud(common_words):
    # Generate and display the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(common_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Meaningful Words in News Articles')
    plt.show()

if __name__ == '__main__':
    stock_ticker = input('Enter the stock ticker (e.g., AAPL for Apple): ').upper()
    sentiment_data, all_text = get_stock_sentiment(stock_ticker)
    
    if sentiment_data:
        # Sentiment analysis and pie chart
        positive, negative, neutral = classify_sentiments(sentiment_data)
        print(f"Sentiment Summary for {stock_ticker}:\nPositive: {positive}\nNegative: {negative}\nNeutral: {neutral}")
        plot_sentiment_pie(positive, negative, neutral)
        
        # Word cloud for common, impactful words
        if all_text:
            common_words = extract_meaningful_words(all_text)
            create_wordcloud(common_words)
