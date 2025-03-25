import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)

def article_agent_summary(df):
    """Summarize the dataset at the article and agent level."""
    summary_records = []
    for _, row in df.iterrows():
        transcript_id = row['transcript_id']
        article_url = row['article_url']
        agent_counts = {}
        agent_sentiments = {}
        for msg in row['content']:
            agent = msg.get('agent')
            sentiment = msg.get('sentiment')
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            agent_sentiments.setdefault(agent, []).append(sentiment)
        summary_records.append({
            'transcript_id': transcript_id,
            'article_url': article_url,
            'agent_counts': agent_counts,
            'agent_sentiments': agent_sentiments
        })
    summary_df = pd.DataFrame(summary_records)
    logging.info("Article-agent summary created.")
    return summary_df

def plot_message_counts(df):
    """Bar plot of total messages sent per agent."""
    agent_message_counts = {}
    for _, row in df.iterrows():
        for msg in row['content']:
            agent = msg.get('agent')
            agent_message_counts[agent] = agent_message_counts.get(agent, 0) + 1

    agents = list(agent_message_counts.keys())
    counts = list(agent_message_counts.values())
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=agents, y=counts, palette="viridis")
    plt.title("Message Counts per Agent")
    plt.xlabel("Agent")
    plt.ylabel("Message Count")
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df):
    """Count plot for sentiment distribution per agent."""
    sentiment_records = []
    for _, row in df.iterrows():
        for msg in row['content']:
            sentiment_records.append({
                'agent': msg.get('agent'),
                'sentiment': msg.get('sentiment')
            })
    sentiment_df = pd.DataFrame(sentiment_records)
    
    plt.figure(figsize=(10, 5))
    sns.countplot(x='sentiment', hue='agent', data=sentiment_df, palette="magma")
    plt.title("Sentiment Distribution per Agent")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Agent")
    plt.tight_layout()
    plt.show()

def plot_message_length_distribution(df):
    """Plot the distribution of message lengths per agent."""
    lengths = []
    for _, row in df.iterrows():
        for msg in row['content']:
            lengths.append({
                'agent': msg.get('agent'),
                'message_length': msg.get('message_length', 0)
            })
    lengths_df = pd.DataFrame(lengths)
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='agent', y='message_length', data=lengths_df, palette="coolwarm")
    plt.title("Message Length Distribution per Agent")
    plt.xlabel("Agent")
    plt.ylabel("Message Length (words)")
    plt.tight_layout()
    plt.show()

def generate_wordcloud(text, title):
    """Generate and display a word cloud from a given text."""
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False)
    wordcloud = wc.generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_wordclouds_per_agent(df):
    """Generate word clouds for each agent from processed messages."""
    agent_texts = {}
    for _, row in df.iterrows():
        for msg in row['content']:
            agent = msg.get('agent')
            processed = msg.get('processed_message', '')
            agent_texts.setdefault(agent, "").append(processed)
    for agent, texts in agent_texts.items():
        full_text = " ".join(texts)
        generate_wordcloud(full_text, f"Word Cloud for {agent}")

def correlation_matrix(df):
    """
    If you have numeric features (e.g., config_encoded, average message length per transcript),
    generate a correlation matrix.
    """
    # Example: compute average message length per transcript
    df['avg_message_length'] = df['content'].apply(lambda msgs: np.mean([msg.get('message_length', 0) for msg in msgs]))
    num_features = df[['config_encoded', 'avg_message_length']]
    corr = num_features.corr()
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def summarize_transcript(transcript):
    """
    Given a transcript (as a dictionary), return:
      - Article URL
      - Message counts per agent
      - Overall sentiment per agent (using mode of sentiments)
    
    If the transcript is nested under the "transcript" key, then extract it.
    """
    if "transcript" in transcript:
        transcript = transcript["transcript"]

    from collections import Counter
    agent_counts = {}
    agent_sentiments = {}
    
    for msg in transcript.get('content', []):
        agent = msg.get('agent')
        sentiment = msg.get('sentiment')
        agent_counts[agent] = agent_counts.get(agent, 0) + 1
        agent_sentiments.setdefault(agent, []).append(sentiment)
    
    overall_sentiments = {}
    for agent, sentiments in agent_sentiments.items():
        overall_sentiments[agent] = Counter(sentiments).most_common(1)[0][0]
    
    summary = {
        'article_url': transcript.get('article_url'),
        'message_counts': agent_counts,
        'overall_sentiments': overall_sentiments
    }
    return summary

