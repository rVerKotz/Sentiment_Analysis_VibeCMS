import os
import httpx
import asyncio
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from collections import Counter
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

# Configuration for Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN", "")
SENTIMENT_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
CLASSIFIER_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class Comment(BaseModel):
    id: str
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

async def query_hf_api(client, url, payload):
    """Helper to query Hugging Face API with retries for model loading"""
    for _ in range(3):
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)
        result = response.json()
        
        # If model is still loading, wait 2 seconds and retry
        if isinstance(result, dict) and "estimated_time" in result:
            await asyncio.sleep(2)
            continue
        return result
    return None

@app.post("/analyze")
async def analyze_vibe(request: AnalysisRequest):
    if not request.articles:
        return {"insights": [], "recommendations": []}

    struggle_topics = []

    async with httpx.AsyncClient() as client:
        # 1. Process Sentiments and Classification
        for comment in request.comments:
            # Sentiment Analysis
            sentiment_res = await query_hf_api(client, SENTIMENT_URL, {"inputs": comment.content})
            
            if sentiment_res and isinstance(sentiment_res, list) and len(sentiment_res) > 0:
                # API format is [[{'label': 'POSITIVE', 'score': 0.99}]]
                top_res = sentiment_res[0][0] if isinstance(sentiment_res[0], list) else sentiment_res[0]
                
                if top_res['label'] == 'NEGATIVE' or top_res['score'] < 0.6:
                    # Zero-Shot Classification
                    candidate_labels = ["Technical Detail", "Deployment/AWS", "UI/UX Design", "Performance", "General Praise"]
                    payload = {
                        "inputs": comment.content,
                        "parameters": {"candidate_labels": candidate_labels}
                    }
                    class_res = await query_hf_api(client, CLASSIFIER_URL, payload)
                    
                    if class_res and "labels" in class_res:
                        struggle_topics.append(class_res['labels'][0])

    top_struggle = Counter(struggle_topics).most_common(1)
    
    # 2. Impact & Surge Detection (Pandas logic remains the same)
    try:
        df = pd.DataFrame([a.model_dump() for a in request.articles])
        now = datetime.now(timezone.utc)
        df['updated_at'] = pd.to_datetime(df['updated_at']).dt.tz_localize(None).dt.tz_localize(timezone.utc)
        df['age_hours'] = (now - df['updated_at']).dt.total_seconds() / 3600
        
        df['surge_velocity'] = df['likes'] / (df['age_hours'] + 1)
        df['trend_weight'] = np.exp(-np.log(2) * df['age_hours'] / 72)
        df['impact_score'] = (df['views'] + (df['likes'] * 10) + (df['surge_velocity'] * 50)) * df['trend_weight']

        trending = df.sort_values(by='impact_score', ascending=False).head(3)
        insights = [{"name": r['title'], "views": int(r['impact_score'])} for _, r in trending.iterrows()]
    except Exception as e:
        # Fallback if pandas processing fails
        insights = []

    # 3. Final Recommendations
    recommendations = []
    if top_struggle:
        recommendations.append(f"Reader Alert: Banyak pembaca kesulitan dengan '{top_struggle[0][0]}'. Buatlah panduan mendalam tentang topik ini.")
    
    for _, row in trending.head(2).iterrows():
        recommendations.append(f"Trending Surge: Topik '{row['title']}' sedang naik daun. Pertimbangkan membuat sekuel atau update.")

    return {"insights": insights, "recommendations": recommendations}