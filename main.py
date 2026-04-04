import os
import re
import jwt
import time
import math
import httpx
import asyncio
import numpy as np
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Union
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Response, Request, Header, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI(title="VibeAI Engine", redirect_slashes=False)

# 1. Standard CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

load_dotenv(dotenv_path=".env.local")
HF_TOKEN = os.getenv("HF_TOKEN")
VIBE_AI_INTERNAL_TOKEN = os.getenv("VIBE_AI_INTERNAL_TOKEN")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
LIMIT_PER_MINUTE = 5
rate_limit_store = defaultdict(list)

# Model URLs
SENTIMENT_URL = "https://router.huggingface.co/hf-inference/models/lxyuan/distilbert-base-multilingual-cased-sentiments-student"
TRANSLATION_URL = "https://router.huggingface.co/hf-inference/models/facebook/mbart-large-50-many-to-many-mmt"
CLASSIFIER_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
KEYWORDS_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

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
    
@app.options("/{rest_of_path:path}")
async def preflight_handler():
    return JSONResponse(content="OK", status_code=200)

def check_rate_limit(user_id: str):
    now = time.time()
    # Bersihkan data lama (> 60 detik)
    rate_limit_store[user_id] = [t for t in rate_limit_store[user_id] if now - t < 60]
    if len(rate_limit_store[user_id]) >= LIMIT_PER_MINUTE:
        return False
    rate_limit_store[user_id].append(now)
    return True

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT token with an expiration time. Default is 30 days."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Default 30 hari
        expire = datetime.now(timezone.utc) + timedelta(days=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

async def verify_vibe_access(
    x_vibe_token: Optional[str] = Header(None),
    token: Optional[str] = Depends(oauth2_scheme)
):
    """
    Middleware Gabungan:
    1. Check X-Vibe-Token (Master/Unlimited)
    2. Check JWT Bearer (Limited/Rate Limited)
    """
    # Master Token (Used by VibeCMS Next.js Server via /api/analyze)
    if x_vibe_token == VIBE_AI_INTERNAL_TOKEN:
        return {"access": "unlimited", "user": "system_internal"}

    # JWT Token (For client/user calling directly, subject to rate limits)
    if token:
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token payload")
            
            # Check Rate Limit for limited access
            if not check_rate_limit(user_id):
                raise HTTPException(
                    status_code=429, 
                    detail=f"Rate limit exceeded. Max {LIMIT_PER_MINUTE} requests per minute."
                )
            
            return {"access": "limited", "user": user_id}
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired. Please renew after 30 days.")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Could not validate credentials")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Valid access token or Master header required",
    )

async def query_hf_api(client, url, payload):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN_MISSING"}
    
    api_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for i in range(3):
        try:
            response = await client.post(url, headers=api_headers, json=payload, timeout=30.0)
            if response.status_code != 200:
                if response.status_code == 503:
                    await asyncio.sleep(5) 
                    continue
                # Log non-retryable errors to help debugging
                print(f"DEBUG: API {url.split('/')[-1]} error: {response.status_code} - {response.text}")
                return None
            return response.json()
        except Exception as e:
            await asyncio.sleep(1)
    return None

@app.post("/token")
async def generate_token(user_id: str = "guest_user"):
    """Endpoint for generating a JWT token (Valid for 30 days)"""
    # Di sini biasanya ada validasi user (login)
    access_token = create_access_token(data={"sub": user_id})
    return {"access_token": access_token, "token_type": "bearer", "expires_in_days": 30}

async def extract_dynamic_labels(client, articles: List[Article]):
    """
    Extracts topics using AI with deep context and punctuation cleaning.
    """
    if not articles:
        return ["Technology", "General", "Insight", "Feedback"]

    context = ". ".join([f"{a.title}: {a.content[:200]}" for a in articles[:3]])
    prompt = f"Categorize these articles into 5 short distinct tags: {context}. Tags:"
    
    res = await query_hf_api(client, KEYWORDS_URL, {
        "inputs": prompt,
        "parameters": {"max_length": 60, "do_sample": False}
    })
    
    if res and isinstance(res, list) and len(res) > 0:
        raw_output = res[0].get('summary_text', "").lower()
        instruction_trigger = "categorize these articles into 5 short distinct tags"
        if instruction_trigger in raw_output:
            raw_output = raw_output.split("tags:")[-1] if "tags:" in raw_output else raw_output.replace(instruction_trigger, "")
        
        raw_labels = re.split(r'[,:\n\s/]', raw_output)
        cleaned_labels = []
        for l in raw_labels:
            clean = re.sub(r'[^a-zA-Z0-9]', '', l).strip().title()
            if 2 < len(clean) < 20:
                cleaned_labels.append(clean)
        
        unique_labels = list(dict.fromkeys(cleaned_labels))
        if len(unique_labels) >= 2:
            return unique_labels[:6]
    
    print("DEBUG: AI Labeling failed or echoed. Using semantic fallback.")
    smart_fallback = ["Performance", "Architecture", "Cloud Service", "Technical Guide"]
    for a in articles:
        words = [re.sub(r'[^a-zA-Z]', '', w).title() for w in a.title.split() if len(w) > 5]
        smart_fallback.extend(words)
    return list(dict.fromkeys(smart_fallback))[:6]

@app.post("/analyze")
async def analyze_vibe(request: AnalysisRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN missing")

    async with httpx.AsyncClient() as client:
        dynamic_labels = await extract_dynamic_labels(client, request.articles)
        print(f"Log: Labels generated: {dynamic_labels}")

        struggle_topics = []
        
        # KEYWORDS: Indicators of pivots from praise to criticism
        #TO DO: Refine this list based on actual comment patterns & Different Languages. This is a critical part of the logic to identify the "pivot" point in comments where sentiment shifts.
        struggle_en = ["but", "however", "lack", "missing", "confusing", "hard", "difficult", "error", "problem", "issue", "clearer", "explain", "not clear"]
        struggle_id = ["tapi", "namun", "kurang", "bingung", "sulit", "susah", "error", "kendala", "perjelas", "jelasin", "tidak jelas"]

        for comment in request.comments:
            content_lower = comment.content.lower()
            
            # 1. Global Translation (Initial detection pass)
            translation_res = await query_hf_api(client, TRANSLATION_URL, {
                "inputs": comment.content,
                "parameters": {"src_lang": "id_ID", "tgt_lang": "en_XX"}
            })
            
            translated_full = content_lower
            if translation_res and isinstance(translation_res, list) and len(translation_res) > 0:
                item = translation_res[0]
                if isinstance(item, dict) and "translation_text" in item:
                    translated_full = item["translation_text"].lower()
                elif isinstance(item, str):
                    translated_full = item.lower()

            # 2. Sentiment Check
            sentiment_res = await query_hf_api(client, SENTIMENT_URL, {"inputs": comment.content})
            
            should_classify = False
            if sentiment_res and isinstance(sentiment_res, list):
                try:
                    data = sentiment_res[0]
                    scores = {item['label']: item['score'] for item in data}
                    top_label = max(scores, key=scores.get)
                    
                    has_indicator_en = any(word in translated_full for word in struggle_en)
                    has_indicator_id = any(word in content_lower for word in struggle_id)
                    
                    if top_label in ['negative', 'neutral']:
                        should_classify = True
                    elif has_indicator_en or has_indicator_id:
                        should_classify = True

                    if should_classify:
                        # 3. Extract the "Criticism" segment (Pivot logic)
                        target_text_id = comment.content
                        for pivot in struggle_id:
                            if pivot in content_lower:
                                parts = re.split(f"\\b{pivot}\\b", content_lower, flags=re.IGNORECASE)
                                if len(parts) > 1:
                                    target_text_id = parts[-1].strip()
                                    break
                        
                        # 4. TRANSLATE the specific criticism to English
                        final_classification_input = target_text_id
                        trans_target = await query_hf_api(client, TRANSLATION_URL, {
                            "inputs": target_text_id,
                            "parameters": {"src_lang": "id_ID", "tgt_lang": "en_XX"}
                        })
                        
                        if trans_target and isinstance(trans_target, list) and len(trans_target) > 0:
                            item = trans_target[0]
                            if isinstance(item, dict) and "translation_text" in item:
                                final_classification_input = item["translation_text"]
                            elif isinstance(item, str):
                                final_classification_input = item
                            print(f"Log: Classified segment translated to: '{final_classification_input[:50]}'")

                        # 5. Classify the English-translated criticism
                        class_res = await query_hf_api(client, CLASSIFIER_URL, {
                            "inputs": final_classification_input,
                            "parameters": {
                                "candidate_labels": dynamic_labels
                            }
                        })
                        
                        # UPDATED: Handle both dict response and list response (based on your log)
                        if class_res:
                            label = None
                            score = 0
                            
                            # Case: Response is a list of dictionaries [{'label': ..., 'score': ...}, ...]
                            if isinstance(class_res, list) and len(class_res) > 0:
                                top_item = class_res[0]
                                label = top_item.get('label')
                                score = top_item.get('score', 0)
                                
                            # Case: Response is a dict with 'labels' and 'scores' keys
                            elif isinstance(class_res, dict) and "labels" in class_res:
                                label = class_res['labels'][0]
                                score = class_res.get('scores', [0])[0]

                            if label:
                                print(f"Log: Top Label: {label} (Score: {round(score, 3)})")
                                if score > 0.1:
                                    struggle_topics.append(label)
                            else:
                                print(f"DEBUG: Classifier failed to parse labels. Response: {class_res}")

                except Exception as e:
                    print(f"Log: Processing error: {e}")

    # 3. Trending & Scoring Logic
    scored_articles = []
    now = datetime.now(timezone.utc)
    for a in request.articles:
        try:
            dt = pd.to_datetime(a.updated_at).to_pydatetime()
            age_hours = max((now - dt).total_seconds() / 3600, 0.01)
            surge_velocity = a.likes / (age_hours + 1)
            trend_weight = math.exp(-math.log(2) * age_hours / 72)
            impact_score = (a.views + (a.likes * 10) + (surge_velocity * 50)) * trend_weight
            scored_articles.append({
                "name": a.title, 
                "score": int(round(impact_score)),
                "is_active": (a.views > 0 or a.likes > 0),
                "is_new": age_hours < 24
            })
        except Exception as e: 
            print(f"Log: Error occurred while processing article '{a.title}': {e}")
            continue

    scored_articles.sort(key=lambda x: x['score'], reverse=True)
    insights_data = [{"name": i['name'], "views": i['score']} for i in scored_articles[:3]]

    # 4. Recommendation Output
    top_struggle = Counter(struggle_topics).most_common(1)
    print(f"Log: Final struggle counter: {dict(Counter(struggle_topics))}")
    
    recommendations = []
    if top_struggle:
        recommendations.append(f"Reader Alert: Issues reported regarding '{top_struggle[0][0]}'. Consider providing more detail.")
    else:
        recommendations.append("Insight: Reader sentiment is stable. Content quality is well-received.")
    
    trending_added = 0
    for sa in scored_articles:
        if sa['is_active'] and sa['score'] > 5 and trending_added < 2:
            recommendations.append(f"Trending: '{sa['name']}' is gaining traction.")
            trending_added += 1
        elif sa['is_new'] and trending_added < 2:
            recommendations.append(f"Fresh: '{sa['name']}' is newly released.")
            trending_added += 1

    return {
        "insights": insights_data,
        "recommendations": recommendations
    }