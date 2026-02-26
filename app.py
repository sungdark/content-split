#!/usr/bin/env python3
"""
ContentSplit — AI Content Repurposer API

Turns long-form content into platform-specific social media posts.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from middleware import validate_api_key, track_usage, get_usage_stats, get_or_create_key, PLANS
from scoring import QualityScorer

app = FastAPI(
    title="ContentSplit",
    description="AI-powered content repurposing API. Turn long-form content into platform-optimized social posts.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────────

class RepurposeRequest(BaseModel):
    content: str = Field(..., min_length=50, max_length=50000, description="Source content to repurpose")
    source_type: str = Field(default="blog", description="Type of source: blog, article, notes, transcript")
    targets: list[str] = Field(
        default=["twitter_thread", "linkedin", "nostr"],
        description="Target platforms"
    )
    tone: str = Field(default="professional", description="Tone: professional, casual, witty, technical")
    max_tweets: int = Field(default=8, ge=2, le=20)
    include_hashtags: bool = Field(default=True)
    language: str = Field(default="en", description="Output language: en, pt, es")


class RepurposeResponse(BaseModel):
    id: str
    source_length: int
    targets_generated: list[str]
    results: dict
    hashtags: Optional[dict] = None
    created_at: str


class QualityScoreRequest(BaseModel):
    submission: str = Field(..., min_length=1, description="Submission content: JSON, markdown, code, or text")


class QualityBenchmarkRequest(BaseModel):
    submissions: list[str] = Field(..., min_length=1, description="Batch submissions for benchmark")


# ── Content Generation (using prompts, model-agnostic) ────────────────────

PLATFORM_PROMPTS = {
    "twitter_thread": {
        "system": "You are an expert social media copywriter. Create engaging Twitter/X threads.",
        "user": """Turn this content into a Twitter thread of {max_tweets} tweets.

Rules:
- First tweet must hook the reader (start with a bold statement or question)
- Each tweet ≤ 280 characters
- Use line breaks for readability
- End with a CTA or takeaway
- Tone: {tone}
- Thread numbering: 1/, 2/, etc.

Content:
{content}

Return as JSON array of strings."""
    },
    "linkedin": {
        "system": "You are a LinkedIn content expert. Write professional, engaging posts.",
        "user": """Turn this content into a LinkedIn post.

Rules:
- Hook in first line (before "see more" fold)
- Use short paragraphs and line breaks
- Include relevant emojis sparingly
- End with a question to drive engagement
- 1300 characters max
- Tone: {tone}

Content:
{content}

Return the post text."""
    },
    "nostr": {
        "system": "You are a content creator for decentralized social media (NOSTR/Bluesky).",
        "user": """Turn this content into a NOSTR note.

Rules:
- Concise but substantive (300-500 chars ideal)
- Include 3-5 relevant hashtags
- Casual, authentic tone
- Tone adjustment: {tone}

Content:
{content}

Return the note text."""
    },
    "email_newsletter": {
        "system": "You are an email marketing expert writing newsletter content.",
        "user": """Turn this content into a newsletter section.

Rules:
- Compelling subject line suggestion
- Brief intro (2-3 sentences)
- 3-5 key takeaways as bullet points
- CTA at the end
- Tone: {tone}

Content:
{content}

Return as JSON with keys: subject_line, intro, takeaways (array), cta"""
    },
    "video_script": {
        "system": "You are a video scriptwriter for short-form content (YouTube Shorts, TikTok, Reels).",
        "user": """Turn this content into a 60-second video script.

Rules:
- Hook in first 3 seconds
- Clear structure: hook → context → value → CTA
- Natural speaking style
- Include visual/B-roll suggestions in [brackets]
- Tone: {tone}

Content:
{content}

Return the script."""
    },
    "summary": {
        "system": "You are a content summarizer.",
        "user": """Summarize this content in 2-3 sentences.

Tone: {tone}

Content:
{content}

Return the summary."""
    },
}

HASHTAG_PROMPTS = {
    "twitter": "Suggest 5 relevant Twitter hashtags for this content. Return as JSON array.",
    "linkedin": "Suggest 3 relevant LinkedIn hashtags. Return as JSON array.",
    "nostr": "Suggest 5 relevant hashtags for NOSTR. Return as JSON array.",
}


async def generate_content(platform: str, content: str, tone: str, max_tweets: int = 8, model: str = "gpt-4o-mini") -> str:
    """Generate repurposed content using AI. Supports OpenAI and Anthropic."""
    
    template = PLATFORM_PROMPTS.get(platform)
    if not template:
        raise ValueError(f"Unknown platform: {platform}")
    
    user_prompt = template["user"].format(
        content=content[:10000],  # Truncate very long content
        tone=tone,
        max_tweets=max_tweets,
    )
    
    # Try OpenAI first, fall back to Anthropic
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        return await _generate_openai(template["system"], user_prompt, openai_key, model)
    elif anthropic_key:
        return await _generate_anthropic(template["system"], user_prompt, anthropic_key)
    else:
        # Fallback: simple extractive repurposing (no AI)
        return _fallback_repurpose(platform, content, tone, max_tweets)


async def _generate_openai(system: str, user: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Generate via OpenAI API."""
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


async def _generate_anthropic(system: str, user: str, api_key: str) -> str:
    """Generate via Anthropic API."""
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-haiku-20240307",
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": 2000,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


def _fallback_repurpose(platform: str, content: str, tone: str, max_tweets: int) -> str:
    """Simple rule-based repurposing when no AI API is available."""
    sentences = [s.strip() for s in content.replace("\n", " ").split(".") if s.strip()]
    
    if platform == "twitter_thread":
        tweets = []
        for i, sent in enumerate(sentences[:max_tweets]):
            tweet = f"{i+1}/ {sent}."
            if len(tweet) > 280:
                tweet = tweet[:277] + "..."
            tweets.append(tweet)
        return json.dumps(tweets)
    
    elif platform == "linkedin":
        intro = sentences[0] + "." if sentences else ""
        body = "\n\n".join(f"→ {s}." for s in sentences[1:5])
        return f"{intro}\n\n{body}\n\nWhat do you think? 💬"
    
    elif platform == "nostr":
        key_points = " ".join(sentences[:3])
        return f"{key_points[:450]}\n\n#content #insights"
    
    elif platform == "email_newsletter":
        return json.dumps({
            "subject_line": sentences[0][:60] if sentences else "Update",
            "intro": " ".join(sentences[:2]),
            "takeaways": [f"{s}." for s in sentences[2:6]],
            "cta": "Read the full article →"
        })
    
    elif platform == "video_script":
        return f"[HOOK] {sentences[0]}.\n\n" + "\n".join(f"[POINT] {s}." for s in sentences[1:5]) + "\n\n[CTA] Follow for more."
    
    else:
        return " ".join(sentences[:3])


# ── API Routes ─────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str = Field(..., description="Email address for API key")
    plan: str = Field(default="free", description="Plan: free, starter, pro, enterprise")


class SignupResponse(BaseModel):
    api_key: str
    plan: str
    monthly_limit: int
    message: str


@app.post("/api/signup", response_model=SignupResponse)
async def signup(req: SignupRequest):
    """Get an API key. Free tier: 50 requests/month, no credit card required."""
    if req.plan not in PLANS:
        raise HTTPException(400, f"Invalid plan. Available: {list(PLANS.keys())}")
    
    api_key = get_or_create_key(req.email, req.plan)
    plan_config = PLANS[req.plan]
    
    return SignupResponse(
        api_key=api_key,
        plan=req.plan,
        monthly_limit=plan_config["limit"],
        message=f"API key created! Add 'X-API-Key: {api_key}' header to your requests.",
    )


@app.get("/api/usage")
async def usage(user: dict = Depends(validate_api_key)):
    """Check your API usage and limits."""
    return get_usage_stats(user["key"])


@app.get("/api/pricing")
async def pricing():
    """View available plans and pricing."""
    return {
        "plans": {
            name: {
                "requests_per_month": config["limit"],
                "rate_per_minute": config["rate_per_min"],
                "platforms": config["platforms"],
                "price_usd": config.get("price", 0),
            }
            for name, config in PLANS.items()
        }
    }


@app.post("/api/repurpose", response_model=RepurposeResponse)
async def repurpose_content(req: RepurposeRequest, user: dict = Depends(validate_api_key)):
    """Repurpose content for multiple platforms."""
    
    valid_targets = set(PLATFORM_PROMPTS.keys())
    invalid = set(req.targets) - valid_targets
    if invalid:
        raise HTTPException(400, f"Invalid targets: {invalid}. Valid: {valid_targets}")
    
    results = {}
    for target in req.targets:
        try:
            result = await generate_content(
                platform=target,
                content=req.content,
                tone=req.tone,
                max_tweets=req.max_tweets,
            )
            # Try to parse JSON results
            try:
                results[target] = json.loads(result)
            except json.JSONDecodeError:
                results[target] = result
        except Exception as e:
            results[target] = {"error": str(e)}
    
    # Generate hashtags if requested
    hashtags = None
    if req.include_hashtags:
        hashtags = {}
        for target in req.targets:
            platform_key = target.split("_")[0]  # twitter_thread → twitter
            if platform_key in HASHTAG_PROMPTS:
                # Simple keyword extraction as fallback
                words = req.content.lower().split()
                common_tags = [w.strip(".,!?") for w in words if len(w) > 5][:5]
                hashtags[platform_key] = [f"#{t}" for t in set(common_tags)]
    
    # Track usage
    track_usage(user.get("key", "anonymous"))
    
    response_id = f"cs_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(req.content[:50]) % 10000:04d}"
    
    return RepurposeResponse(
        id=response_id,
        source_length=len(req.content),
        targets_generated=req.targets,
        results=results,
        hashtags=hashtags,
        created_at=datetime.now().isoformat(),
    )


@app.get("/api/platforms")
async def list_platforms():
    """List available target platforms."""
    return {
        "platforms": list(PLATFORM_PROMPTS.keys()),
        "tones": ["professional", "casual", "witty", "technical", "friendly"],
        "languages": ["en", "pt", "es"],
    }


quality_scorer = QualityScorer()


@app.post("/api/quality/score")
async def quality_score(req: QualityScoreRequest):
    return quality_scorer.score(req.submission)


@app.post("/api/quality/benchmark")
async def quality_benchmark(req: QualityBenchmarkRequest):
    return quality_scorer.benchmark(req.submissions)


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Landing page."""
    landing = Path(__file__).parent / "landing.html"
    if landing.exists():
        return HTMLResponse(landing.read_text())
    return HTMLResponse("<h1>ContentSplit API</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "ai_available": bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
