import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter
from transformers import pipeline
from typing import List, Optional
from datetime import datetime, timezone
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VibeAI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model sekali saat startup
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class Comment(BaseModel):
    id: str # Cast ke string dari DB
    content: str
    article_id: str
    updated_at: str

class Article(BaseModel):
    id: str
    title: str
    views: int = 0
    likes: int = 0
    updated_at: str

class AnalysisRequest(BaseModel):
    articles: List[Article]
    comments: List[Comment]

@app.post("/analyze")
async def analyze_vibe(request: AnalysisRequest):
    if not request.articles:
        return {"insights": [], "recommendations": []}

    # 1. Sentiment Analysis pada Komentar
    candidate_labels = ["Technical Detail", "Deployment/AWS", "UI/UX Design", "Performance", "General Praise"]
    struggle_topics = []

    for comment in request.comments:
        res = sentiment_pipe(comment.content)[0]
        if res['label'] == 'NEGATIVE' or res['score'] < 0.6:
            topic = classifier(comment.content, candidate_labels)
            struggle_topics.append(topic['labels'][0])

    top_struggle = Counter(struggle_topics).most_common(1)
    
    # 2. Impact & Surge Detection (Time-Series)
    df = pd.DataFrame([a.dict() for a in request.articles])
    now = datetime.now(timezone.utc)
    df['updated_at'] = pd.to_datetime(df['updated_at']).dt.tz_localize(None).dt.tz_localize(timezone.utc)
    df['age_hours'] = (now - df['updated_at']).dt.total_seconds() / 3600
    
    # Surge velocity & Trend Weighting
    df['surge_velocity'] = df['likes'] / (df['age_hours'] + 1)
    df['trend_weight'] = np.exp(-np.log(2) * df['age_hours'] / 72)
    df['impact_score'] = (df['views'] + (df['likes'] * 10) + (df['surge_velocity'] * 50)) * df['trend_weight']

    # 3. Hasil Akhir
    trending = df.sort_values(by='impact_score', ascending=False).head(3)
    
    insights = [{"name": r['title'], "views": int(r['impact_score'])} for _, r in trending.iterrows()]
    
    recommendations = []
    if top_struggle:
        recommendations.append(f"Reader Alert: Banyak pembaca kesulitan dengan '{top_struggle[0][0]}'. Buatlah panduan mendalam tentang topik ini.")
    
    for _, row in trending.head(2).iterrows():
        recommendations.append(f"Trending Surge: Topik '{row['title']}' sedang naik daun. Pertimbangkan membuat sekuel atau update.")

    return {"insights": insights, "recommendations": recommendations}