import os
import httpx
import asyncio
import numpy as np
import pandas as pd
from pydantic import BaseModel
from collections import Counter
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VibeAI Engine", redirect_slashes=False)

# 1. Standard CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration for Hugging Face Inference API
HF_TOKEN = os.getenv("HF_TOKEN")
SENTIMENT_URL = "https://router.huggingface.co/hf-inference/models/distilbert-base-uncased-finetuned-sst-2-english"
CLASSIFIER_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
KEYWORDS_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

# Explicit Preflight Handler
# Some proxies/browsers prefer an explicit handler for OPTIONS
@app.options("/{rest_of_path:path}")
async def preflight_handler():
    return JSONResponse(content="OK", status_code=200)

# Warn immediately if the token is missing during cold start
if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable is not set. AI features will be disabled.")

class Comment(BaseModel):
    id: str
    content: str
    article_id: str
    updated_at: str

class Article(BaseModel):
    id: str
    title: str
    content: Optional[str] = ""
    views: int = 0
    likes: int = 0
    updated_at: str

class AnalysisRequest(BaseModel):
    articles: List[Article]
    comments: List[Comment]

async def query_hf_api(client, url, payload):
    """
    Helper to query Hugging Face API with retries for model loading.
    Constructs headers dynamically to ensure token validity.
    """
    if not HF_TOKEN:
        return {"error": "HF_TOKEN_MISSING"}
    
    api_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for i in range(3):
        try:
            response = await client.post(url, headers=api_headers, json=payload, timeout=30.0)
            
            # If we hit rate limits or unauthorized
            if response.status_code == 401:
                print("Error: Unauthorized. Check your HF_TOKEN.")
                return None
            
            if not response.text:
                print(f"Empty response from HF API (Attempt {i+1})")
                await asyncio.sleep(1)
                continue
                
            result = response.json()
            
            # If model is still loading, the API returns an estimated_time
            if isinstance(result, dict) and "estimated_time" in result:
                wait_time = min(result.get("estimated_time", 2), 5)
                print(f"Model loading... waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
                
            return result
        except Exception as e:
            print(f"API Attempt {i+1} failed: {str(e)}")
            await asyncio.sleep(1)
    return None

async def extract_dynamic_labels(client, articles: List[Article]):
    """
    Dynamically generates categories based on current articles with huggingface.
    This replaces Internal NLP for better topical relevance and current production environment.
    """
    if not articles:
        return ["Technology", "General", "Personal", "Insight"]

    # Increased snippet length to 300 to provide better context for the label generator
    text_samples = []
    for a in articles[:3]:
        content_val = getattr(a, 'content', '') or ''
        text_samples.append(f"{a.title}: {content_val[:300]}")
        
    sample_text = " ".join(text_samples)
    
    payload = {
        "inputs": f"Summarize the main topics of these articles into 5 distinct categories: {sample_text}",
        "parameters": {"max_length": 40}
    }
    
    res = await query_hf_api(client, KEYWORDS_URL, payload)
    
    if res and isinstance(res, list) and len(res) > 0:
        summary = res[0].get('summary_text', "")
        # Clean and split the summary into individual labels
        labels = [l.strip().title() for l in summary.replace(".", "").replace("and", "").split(",") if len(l.strip()) > 2]
        if len(labels) >= 2: return labels[:5]
    
    return ["Performance", "Design", "Technical", "Usability", "Security"]

@app.get("/")
async def health():
    return {
        "status": "online",
        "hf_token_configured": bool(HF_TOKEN),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/analyze")
async def analyze_vibe(request: AnalysisRequest):
    print(f"Received request with {len(request.articles)} articles")
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured on server")
    
    if not request.articles:
        return {"insights": [], "recommendations": []}

    struggle_topics = []

    async with httpx.AsyncClient() as client:
        dynamic_labels = await extract_dynamic_labels(client, request.articles)
        print(dynamic_labels)
        for comment in request.comments:
            sentiment_res = await query_hf_api(client, SENTIMENT_URL, {"inputs": comment.content})
            
            # Process results if valid list returned
            if sentiment_res and isinstance(sentiment_res, list):
                try:
                    # API returns nested list: [[{'label': '...', 'score': ...}]]
                    top_res = sentiment_res[0][0] if isinstance(sentiment_res[0], list) else sentiment_res[0]
                    
                    if top_res['label'] == 'NEGATIVE' or top_res['score'] < 0.6:
                        class_res = await query_hf_api(client, CLASSIFIER_URL, {
                            "inputs": comment.content,
                            "parameters": {"candidate_labels": dynamic_labels}
                        })
                        if class_res and "labels" in class_res:
                            struggle_topics.append(class_res['labels'][0])
                except Exception as e:
                    print(f"Sentiment parsing error: {e}")

    top_struggle = Counter(struggle_topics).most_common(1)
    
    try:
        # Compatibility fix: Use model_dump if available (Pydantic v2), else dict (v1)
        data = []
        for a in request.articles:
            data.append(a.model_dump() if hasattr(a, 'model_dump') else a.dict())
            
        df = pd.DataFrame(data)
        now = datetime.now(timezone.utc)
        df['updated_at'] = pd.to_datetime(df['updated_at']).dt.tz_localize(None).dt.tz_localize(timezone.utc)
        df['age_hours'] = (now - df['updated_at']).dt.total_seconds() / 3600
        
        # Trend calculation logic
        df['surge_velocity'] = df['likes'] / (df['age_hours'] + 1)
        df['trend_weight'] = np.exp(-np.log(2) * df['age_hours'] / 72)
        df['impact_score'] = (df['views'] + (df['likes'] * 10) + (df['surge_velocity'] * 50)) * df['trend_weight']

        trending = df.sort_values(by='impact_score', ascending=False).head(3)
        insights = [{"name": r['title'], "views": int(r['impact_score'])} for _, r in trending.iterrows()]
    except Exception as e:
        print(f"Pandas processing error: {e}")
        insights = []

    # Final recommendations construction
    recommendations = []
    if top_struggle:
        recommendations.append(f"Reader Alert: Banyak pembaca menemui kendala pada '{top_struggle[0][0]}'.")
    
    try:
        for _, row in trending.head(2).iterrows():
            recommendations.append(f"Trending: Topik '{row['title']}' sedang populer.")
    except:
        pass

    return {"insights": insights, "recommendations": recommendations}