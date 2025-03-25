from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from wordcloud import WordCloud
import pandas as pd

from data_processing import DataLoader, DataCleaner, DataTransformer
from eda import summarize_transcript, article_agent_summary

app = FastAPI(title="BiztelAI DS API")
logging.basicConfig(level=logging.INFO)

templates = Jinja2Templates(directory="templates")

# Load and process the dataset at startup
DATA_FILE = "BiztelAI_DS_Dataset_Mar'25.json"
data_loader = DataLoader(DATA_FILE)
raw_df = data_loader.load_data()
cleaner = DataCleaner(raw_df)
cleaned_df = cleaner.clean_dataframe()
transformer = DataTransformer(cleaned_df)
processed_df = transformer.transform_content()
processed_df = transformer.encode_categories()

# Create dataset summary for Endpoint 1
dataset_summary = article_agent_summary(processed_df).to_dict(orient='records')

# Define request models
class RawTextInput(BaseModel):
    text: str

class TranscriptInput(BaseModel):
    transcript: dict

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dataset_summary", response_class=HTMLResponse)
async def dataset_summary_html(request: Request):
    return templates.TemplateResponse("dataset_summary.html", {
        "request": request,
        "summary_data": dataset_summary  # Assuming dataset_summary is defined earlier
    })



@app.post("/preprocess")
async def preprocess_raw_text(raw_input: RawTextInput):
    """Endpoint 2: Transform raw text into its preprocessed form."""
    try:
        transformer = DataTransformer(None)  # Only using text preprocessing
        processed_text = transformer.preprocess_text(raw_input.text)
        return {"processed_text": processed_text}
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing text.")

@app.post("/insights")
async def get_transcript_insights(transcript_input: TranscriptInput):
    """Endpoint 3: Return transcript insights including message counts and overall sentiment."""
    try:
        summary = summarize_transcript(transcript_input.transcript)
        return {"transcript_summary": summary}
    except Exception as e:
        logging.error(f"Transcript insight error: {e}")
        raise HTTPException(status_code=500, detail="Error generating insights.")

# Endpoints for form submissions from the interactive UI
@app.post("/preprocess_form", response_class=HTMLResponse)
async def preprocess_form(request: Request):
    form_data = await request.form()
    text = form_data.get("text")
    try:
        transformer = DataTransformer(None)
        processed_text = transformer.preprocess_text(text)
        return templates.TemplateResponse("result.html", {"request": request, "result": f"Processed Text: {processed_text}"})
    except Exception as e:
        logging.error(f"Preprocessing form error: {e}")
        return templates.TemplateResponse("result.html", {"request": request, "error": "Error processing text."})


@app.post("/insights_form", response_class=HTMLResponse)
async def insights_form(request: Request):
    form_data = await request.form()
    transcript_json = form_data.get("transcript")
    try:
        transcript = json.loads(transcript_json)
        summary = summarize_transcript(transcript)
        
        # If the transcript is nested under "transcript", extract that inner dictionary
        if "transcript" in transcript:
            transcript_inner = transcript["transcript"]
        else:
            transcript_inner = transcript
        
        content = transcript_inner.get("content", [])
        
        # Graph 1: Message Counts per Agent
        agent_message_counts = {}
        for msg in content:
            agent = msg.get("agent")
            agent_message_counts[agent] = agent_message_counts.get(agent, 0) + 1
        agents = list(agent_message_counts.keys())
        counts = list(agent_message_counts.values())
        fig1 = plt.figure(figsize=(6,4))
        sns.barplot(x=agents, y=counts, palette="viridis")
        plt.title("Message Counts per Agent")
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png")
        buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode("utf-8")
        plt.close(fig1)
        
        # Graph 2: Sentiment Distribution per Agent
        sentiment_records = []
        for msg in content:
            sentiment_records.append({
                'agent': msg.get('agent'),
                'sentiment': msg.get('sentiment')
            })
        sentiment_df = pd.DataFrame(sentiment_records)
        fig2 = plt.figure(figsize=(6,4))
        sns.countplot(x='sentiment', hue='agent', data=sentiment_df, palette="magma")
        plt.title("Sentiment Distribution per Agent")
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png")
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode("utf-8")
        plt.close(fig2)
        
        # Graph 3: Message Length Distribution per Agent
        lengths = []
        for msg in content:
            lengths.append({
                'agent': msg.get('agent'),
                'message_length': msg.get('message_length', 0)
            })
        lengths_df = pd.DataFrame(lengths)
        fig3 = plt.figure(figsize=(6,4))
        sns.boxplot(x='agent', y='message_length', data=lengths_df, palette="coolwarm")
        plt.title("Message Length Distribution per Agent")
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format="png")
        buf3.seek(0)
        img3 = base64.b64encode(buf3.read()).decode("utf-8")
        plt.close(fig3)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "summary": summary,
            "img1": img1,
            "img2": img2,
            "img3": img3
        })
    except Exception as e:
        logging.error(f"Insights form error: {e}")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "error": "Error generating insights."
        })



# EDA page displaying interactive visualizations
@app.get("/eda", response_class=HTMLResponse)
async def eda_page(request: Request):
    # Plot 1: Message Counts per Agent
    fig1 = plt.figure(figsize=(8, 5))
    agent_message_counts = {}
    for _, row in processed_df.iterrows():
        for msg in row['content']:
            agent = msg.get('agent')
            agent_message_counts[agent] = agent_message_counts.get(agent, 0) + 1
    agents = list(agent_message_counts.keys())
    counts = list(agent_message_counts.values())
    sns.barplot(x=agents, y=counts, palette="viridis")
    plt.title("Message Counts per Agent")
    plt.xlabel("Agent")
    plt.ylabel("Message Count")
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png")
    buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close(fig1)

    # Plot 2: Sentiment Distribution per Agent
    fig2 = plt.figure(figsize=(10, 5))
    sentiment_records = []
    for _, row in processed_df.iterrows():
        for msg in row['content']:
            sentiment_records.append({
                'agent': msg.get('agent'),
                'sentiment': msg.get('sentiment')
            })
    sentiment_df = pd.DataFrame(sentiment_records)
    sns.countplot(x='sentiment', hue='agent', data=sentiment_df, palette="magma")
    plt.title("Sentiment Distribution per Agent")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    img2 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    # Plot 3: Message Length Distribution per Agent
    fig3 = plt.figure(figsize=(10, 5))
    lengths = []
    for _, row in processed_df.iterrows():
        for msg in row['content']:
            lengths.append({
                'agent': msg.get('agent'),
                'message_length': msg.get('message_length', 0)
            })
    lengths_df = pd.DataFrame(lengths)
    sns.boxplot(x='agent', y='message_length', data=lengths_df, palette="coolwarm")
    plt.title("Message Length Distribution per Agent")
    plt.xlabel("Agent")
    plt.ylabel("Message Length (words)")
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png")
    buf3.seek(0)
    img3 = base64.b64encode(buf3.read()).decode("utf-8")
    plt.close(fig3)

    # Plot 4: Combined Word Cloud for All Processed Messages
    all_texts = []
    for _, row in processed_df.iterrows():
        for msg in row['content']:
            all_texts.append(msg.get('processed_message', ''))
    combined_text = " ".join(all_texts)
    fig4 = plt.figure(figsize=(10, 5))
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False)
    wordcloud = wc.generate(combined_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Combined Word Cloud")
    buf4 = io.BytesIO()
    fig4.savefig(buf4, format="png")
    buf4.seek(0)
    img4 = base64.b64encode(buf4.read()).decode("utf-8")
    plt.close(fig4)

    # Plot 5: Correlation Matrix
    processed_df['avg_message_length'] = processed_df['content'].apply(
        lambda msgs: np.mean([msg.get('message_length', 0) for msg in msgs])
    )
    num_features = processed_df[['config_encoded', 'avg_message_length']]
    corr = num_features.corr()
    fig5 = plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    buf5 = io.BytesIO()
    fig5.savefig(buf5, format="png")
    buf5.seek(0)
    img5 = base64.b64encode(buf5.read()).decode("utf-8")
    plt.close(fig5)

    return templates.TemplateResponse("eda.html", {
        "request": request,
        "img1": img1,
        "img2": img2,
        "img3": img3,
        "img4": img4,
        "img5": img5
    })



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
