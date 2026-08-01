"""
AI Creator Trend Intelligence API Router

Provides location-based social media trend analysis, country intelligence,
AI content generation, and real-time notifications for Instagram Reels and YouTube Shorts.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/creator-intelligence", tags=["creator-intelligence"])

# -----------------------------------------------------------------------------
# Models / Schemas
# -----------------------------------------------------------------------------

class CountrySummary(BaseModel):
    id: str
    name: str
    code: str
    flag: str
    population: str
    languages: List[str]
    time_zone: str
    major_platforms: List[str]
    current_trend_score: int
    total_active_trends: int
    coordinates: Dict[str, float]  # {"lat": float, "lng": float}

class PastTrendItem(BaseModel):
    id: str
    title: str
    category: str
    platform: str
    trend_strength: int  # percentage
    peak_date: str
    duration_days: int
    status: str  # "Ended", "Declining"
    why_viral: str
    hashtags: List[str]
    audio_sample: str
    meme_format: str

class CurrentTrendItem(BaseModel):
    id: str
    title: str
    category: str
    platform: str  # "Instagram Reels", "YouTube Shorts", "Both"
    engagement: str
    viral_score: int
    growth_pct: int
    trend_strength: int  # percentage
    expected_duration: str
    status: str  # "Growing", "Stable", "Peak", "Declining"
    thumbnail_url: str
    hashtags: List[str]
    audio_track: str
    keywords: List[str]
    target_audience: str

class TrendAnalysisRequest(BaseModel):
    location_name: str
    country_code: str
    radius_km: str  # "5 km", "10 km", "25 km", "50 km", "100 km", "District", "State", "Country", "Continent"
    lat: Optional[float] = None
    lng: Optional[float] = None

class GeneratedContentResponse(BaseModel):
    trend_id: str
    trend_title: str
    original_script: str
    better_hook: str
    better_cta: str
    storyboard: List[Dict[str, str]]
    scene_breakdown: List[Dict[str, str]]
    camera_angles: List[str]
    bgm_suggestions: List[str]
    voice_over_style: str
    thumbnail_idea: str
    seo_title: str
    viral_keywords: List[str]
    suggested_duration: str

class NotificationItem(BaseModel):
    id: str
    title: str
    message: str
    type: str  # "growth", "duration", "location", "audio", "viral"
    timestamp: str
    read: bool = False

# -----------------------------------------------------------------------------
# Mock Database / Country Datasets
# -----------------------------------------------------------------------------

COUNTRIES_DATA: Dict[str, CountrySummary] = {
    "IN": CountrySummary(
        id="IN",
        name="India",
        code="IN",
        flag="🇮🇳",
        population="1.43 Billion",
        languages=["Hindi", "English", "Kannada", "Tamil", "Telugu", "Bengali"],
        time_zone="IST (UTC+5:30)",
        major_platforms=["Instagram Reels", "YouTube Shorts", "Moj"],
        current_trend_score=94,
        total_active_trends=184,
        coordinates={"lat": 20.5937, "lng": 78.9629}
    ),
    "IN-KA": CountrySummary(
        id="IN-KA",
        name="Karnataka, India",
        code="IN-KA",
        flag="🇮🇳",
        population="68 Million",
        languages=["Kannada", "English", "Tulu"],
        time_zone="IST (UTC+5:30)",
        major_platforms=["Instagram Reels", "YouTube Shorts"],
        current_trend_score=95,
        total_active_trends=142,
        coordinates={"lat": 15.3173, "lng": 75.7139}
    ),
    "IN-TN": CountrySummary(
        id="IN-TN",
        name="Tamil Nadu, India",
        code="IN-TN",
        flag="🇮🇳",
        population="76 Million",
        languages=["Tamil", "English"],
        time_zone="IST (UTC+5:30)",
        major_platforms=["Instagram Reels", "YouTube Shorts"],
        current_trend_score=93,
        total_active_trends=138,
        coordinates={"lat": 11.1271, "lng": 78.6569}
    ),
    "IN-MH": CountrySummary(
        id="IN-MH",
        name="Maharashtra, India",
        code="IN-MH",
        flag="🇮🇳",
        population="123 Million",
        languages=["Marathi", "Hindi", "English"],
        time_zone="IST (UTC+5:30)",
        major_platforms=["Instagram Reels", "YouTube Shorts"],
        current_trend_score=96,
        total_active_trends=175,
        coordinates={"lat": 19.7515, "lng": 75.7139}
    ),
    "US": CountrySummary(
        id="US",
        name="United States",
        code="US",
        flag="🇺🇸",
        population="335 Million",
        languages=["English", "Spanish"],
        time_zone="EST/PST (UTC-5 to -8)",
        major_platforms=["YouTube Shorts", "Instagram Reels", "TikTok"],
        current_trend_score=98,
        total_active_trends=230,
        coordinates={"lat": 37.0902, "lng": -95.7129}
    ),
    "JP": CountrySummary(
        id="JP",
        name="Japan",
        code="JP",
        flag="🇯🇵",
        population="125 Million",
        languages=["Japanese"],
        time_zone="JST (UTC+9:00)",
        major_platforms=["YouTube Shorts", "X (Twitter)", "Instagram Reels"],
        current_trend_score=89,
        total_active_trends=112,
        coordinates={"lat": 36.2048, "lng": 138.2529}
    ),
    "GB": CountrySummary(
        id="GB",
        name="United Kingdom",
        code="GB",
        flag="🇬🇧",
        population="67 Million",
        languages=["English"],
        time_zone="GMT/BST (UTC+0)",
        major_platforms=["Instagram Reels", "YouTube Shorts", "TikTok"],
        current_trend_score=91,
        total_active_trends=145,
        coordinates={"lat": 55.3781, "lng": -3.4360}
    ),
    "DE": CountrySummary(
        id="DE",
        name="Germany",
        code="DE",
        flag="🇩🇪",
        population="84 Million",
        languages=["German", "English"],
        time_zone="CET (UTC+1:00)",
        major_platforms=["Instagram Reels", "YouTube Shorts"],
        current_trend_score=86,
        total_active_trends=98,
        coordinates={"lat": 51.1657, "lng": 10.4515}
    ),
    "BR": CountrySummary(
        id="BR",
        name="Brazil",
        code="BR",
        flag="🇧🇷",
        population="214 Million",
        languages=["Portuguese"],
        time_zone="BRT (UTC-3:00)",
        major_platforms=["Instagram Reels", "YouTube Shorts", "Kwai"],
        current_trend_score=96,
        total_active_trends=195,
        coordinates={"lat": -14.2350, "lng": -51.9253}
    ),
    "FR": CountrySummary(
        id="FR",
        name="France",
        code="FR",
        flag="🇫🇷",
        population="68 Million",
        languages=["French"],
        time_zone="CET (UTC+1:00)",
        major_platforms=["Instagram Reels", "YouTube Shorts", "TikTok"],
        current_trend_score=88,
        total_active_trends=104,
        coordinates={"lat": 46.2276, "lng": 2.2137}
    ),
    "KR": CountrySummary(
        id="KR",
        name="South Korea",
        code="KR",
        flag="🇰🇷",
        population="51 Million",
        languages=["Korean"],
        time_zone="KST (UTC+9:00)",
        major_platforms=["YouTube Shorts", "Instagram Reels"],
        current_trend_score=97,
        total_active_trends=160,
        coordinates={"lat": 35.9078, "lng": 127.7669}
    ),
}

PAST_TRENDS_BY_COUNTRY: Dict[str, List[PastTrendItem]] = {
    "IN": [
        PastTrendItem(
            id="past-in-1",
            title="Protest Movement & Student Voice",
            category="News & Activism",
            platform="Instagram Reels",
            trend_strength=95,
            peak_date="14 June",
            duration_days=12,
            status="Ended",
            why_viral="High emotional resonance across major university campuses.",
            hashtags=["#StudentVoice", "#CampusTrend", "#YouthPower"],
            audio_sample="Acoustic Protest Anthem",
            meme_format="Text overlay on crowd footage"
        ),
        PastTrendItem(
            id="past-in-2",
            title="Monsoon Street Food Challenge",
            category="Food & Vlogs",
            platform="YouTube Shorts",
            trend_strength=88,
            peak_date="02 July",
            duration_days=18,
            status="Ended",
            why_viral="Seasonal rain cravings combined with local vendor highlights.",
            hashtags=["#MonsoonEats", "#StreetFoodIndia", "#RainyDayVibes"],
            audio_sample="Lofi Rain LoFi Beats",
            meme_format="Split screen reaction"
        ),
        PastTrendItem(
            id="past-in-3",
            title="Traditional Fusion Dance Reel",
            category="Dance & Culture",
            platform="Instagram Reels",
            trend_strength=92,
            peak_date="20 May",
            duration_days=14,
            status="Ended",
            why_viral="High energy choreography set to classic bollywood remaps.",
            hashtags=["#FusionDance", "#ReelsIndia", "#DanceChallenge"],
            audio_sample="Classic Folk Trap Beat",
            meme_format="Transition transformation reel"
        ),
    ],
    "US": [
        PastTrendItem(
            id="past-us-1",
            title="AI Productivity Workflow Breakdown",
            category="Tech & Business",
            platform="YouTube Shorts",
            trend_strength=96,
            peak_date="10 July",
            duration_days=21,
            status="Ended",
            why_viral="Solopreneurs showcasing 10x speed automation tools.",
            hashtags=["#AIWorkflows", "#TechHacks", "#Productivity"],
            audio_sample="Futuristic Cyberpunk Synth",
            meme_format="Screen recording fast cuts"
        ),
        PastTrendItem(
            id="past-us-2",
            title="Summer Music Festival Outfits",
            category="Fashion & Lifestyle",
            platform="Instagram Reels",
            trend_strength=90,
            peak_date="25 June",
            duration_days=10,
            status="Ended",
            why_viral="High aesthetics during major festival weekends.",
            hashtags=["#FestivalFit", "#OOTD", "#SummerVibes"],
            audio_sample="Upbeat Indie Pop Track",
            meme_format="Walk into frame fit check"
        )
    ],
    "IN-KA": [
        PastTrendItem(
            id="past-ka-1",
            title="Kantara & Coastal Karnataka Cultural Lore",
            category="Culture & Cinema",
            platform="Instagram Reels",
            trend_strength=96,
            peak_date="18 May",
            duration_days=25,
            status="Ended",
            why_viral="Deep cultural reverence and traditional Kola folk beats.",
            hashtags=["#KarnatakaCulture", "#KolaFolk", "#NammaKarnataka"],
            audio_sample="Traditional Yakshagana Folk Beats",
            meme_format="Cinematic slow-motion transition"
        ),
        PastTrendItem(
            id="past-ka-2",
            title="Bengaluru Tech Park & Metro Commute Vlogs",
            category="Lifestyle & Tech",
            platform="YouTube Shorts",
            trend_strength=90,
            peak_date="10 June",
            duration_days=15,
            status="Ended",
            why_viral="Relatable IT worker humor and traffic memes.",
            hashtags=["#BengaluruTraffic", "#NammaMetro", "#TechParkLife"],
            audio_sample="Upbeat Kannada Rap Beat",
            meme_format="Pov day in the life vlog"
        ),
    ],
}

CURRENT_TRENDS_BY_COUNTRY: Dict[str, List[CurrentTrendItem]] = {
    "IN-KA": [
        CurrentTrendItem(
            id="curr-ka-1",
            title="Kannada Cinema Blockbuster Teasers & Music",
            category="Entertainment",
            platform="Instagram Reels",
            engagement="5.4M Views",
            viral_score=96,
            growth_pct=48,
            trend_strength=96,
            expected_duration="12 Days",
            status="Growing",
            thumbnail_url="https://images.unsplash.com/photo-1518770660439-4636190af475?w=500&auto=format&fit=crop&q=60",
            hashtags=["#Sandalwood", "#KannadaSongs", "#ReelsKarnataka"],
            audio_track="Cinematic Kannada Mass Beat",
            keywords=["Kannada Teaser", "Mass BGM", "Hero Entry", "Sandalwood"],
            target_audience="Youth, Cinema Fans, Karnataka Audience"
        ),
        CurrentTrendItem(
            id="curr-ka-2",
            title="Coorg & Chikmagalur Coffee Plantation Treks",
            category="Travel & Lifestyle",
            platform="YouTube Shorts",
            engagement="3.8M Views",
            viral_score=91,
            growth_pct=36,
            trend_strength=91,
            expected_duration="10 Days",
            status="Growing",
            thumbnail_url="https://images.unsplash.com/photo-1506461883276-594a12b11cf3?w=500&auto=format&fit=crop&q=60",
            hashtags=["#CoorgDiaries", "#Chikmagalur", "#MonsoonKarnataka"],
            audio_track="Soothing Acoustic Kannada Melody",
            keywords=["Coffee Estate", "Homestay", "Foggy Hills", "Weekend Getaway"],
            target_audience="Travelers, Photographers, Weekend Seekers"
        ),
        CurrentTrendItem(
            id="curr-ka-3",
            title="Bengaluru Street Food & CTR Dosa Review",
            category="Food & Vlogs",
            platform="Instagram Reels",
            engagement="4.2M Views",
            viral_score=93,
            growth_pct=29,
            trend_strength=93,
            expected_duration="14 Days",
            status="Stable",
            thumbnail_url="https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=500&auto=format&fit=crop&q=60",
            hashtags=["#BengaluruFoodie", "#BenneDosa", "#NammaBengaluru"],
            audio_track="Upbeat Lofi Filter Coffee Groove",
            keywords=["Butter Dosa", "Filter Coffee", "Malleswaram", "VV Puram"],
            target_audience="Foodies, Local Explorers"
        ),
    ],
    "IN": [
        CurrentTrendItem(
            id="curr-in-1",
            title="Spider-Man Movie & Fan Theories",
            category="Entertainment",
            platform="Instagram Reels",
            engagement="4.8M Views",
            viral_score=92,
            growth_pct=34,
            trend_strength=92,
            expected_duration="7 Days",
            status="Growing",
            thumbnail_url="https://images.unsplash.com/photo-1635805737707-575885ab0820?w=500&auto=format&fit=crop&q=60",
            hashtags=["#SpiderMan", "#MarvelIndia", "#MovieTheory"],
            audio_track="Orchestral Hero Theme (Remix)",
            keywords=["Spider-Man", "Easter Eggs", "Multiverse", "Trailer Breakdown"],
            target_audience="Gen Z, Movie Buffs, Comics Fans"
        ),
        CurrentTrendItem(
            id="curr-in-2",
            title="Festival Prep & Ethnic Aesthetics",
            category="Culture & Lifestyle",
            platform="YouTube Shorts",
            engagement="8.2M Views",
            viral_score=97,
            growth_pct=42,
            trend_strength=97,
            expected_duration="15 Days",
            status="Stable",
            thumbnail_url="https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=500&auto=format&fit=crop&q=60",
            hashtags=["#FestiveVibes", "#EthnicWear", "#IndiaDiaries"],
            audio_track="Traditional Shehnai & Bass",
            keywords=["Festival Outfit", "Decor Ideas", "Festive Glow"],
            target_audience="Fashion Enthusiasts, Families"
        ),
        CurrentTrendItem(
            id="curr-in-3",
            title="Budget Travel Hacks in Western Ghats",
            category="Travel & Adventure",
            platform="Instagram Reels",
            engagement="2.9M Views",
            viral_score=85,
            growth_pct=18,
            trend_strength=81,
            expected_duration="5 Days",
            status="Declining",
            thumbnail_url="https://images.unsplash.com/photo-1506461883276-594a12b11cf3?w=500&auto=format&fit=crop&q=60",
            hashtags=["#WesternGhats", "#BudgetTravel", "#MonsoonTrek"],
            audio_track="Acoustic Chill Acoustic",
            keywords=["Hidden Waterfall", "Backpacking", "Homestay"],
            target_audience="Solo Travelers, Hikers"
        ),
        CurrentTrendItem(
            id="curr-in-4",
            title="10-Minute AI Video Editing Hacks",
            category="Education & Tech",
            platform="YouTube Shorts",
            engagement="3.5M Views",
            viral_score=89,
            growth_pct=28,
            trend_strength=89,
            expected_duration="12 Days",
            status="Growing",
            thumbnail_url="https://images.unsplash.com/photo-1518770660439-4636190af475?w=500&auto=format&fit=crop&q=60",
            hashtags=["#AITools", "#VideoEditing", "#CreatorHacks"],
            audio_track="Upbeat Lo-Fi Tech Beat",
            keywords=["CapCut AI", "Auto Captions", "Viral Transitions"],
            target_audience="Content Creators, Editors"
        )
    ],
    "US": [
        CurrentTrendItem(
            id="curr-us-1",
            title="AI Agents Building Apps Live",
            category="Technology",
            platform="YouTube Shorts",
            engagement="6.1M Views",
            viral_score=98,
            growth_pct=52,
            trend_strength=98,
            expected_duration="14 Days",
            status="Growing",
            thumbnail_url="https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=500&auto=format&fit=crop&q=60",
            hashtags=["#AIAgents", "#CodingChallenge", "#BuildInPublic"],
            audio_track="Tech Synthwave Beats",
            keywords=["Claude 3.5", "Antigravity AI", "No Code", "SaaS in 1 hour"],
            target_audience="Developers, Founders, Tech Enthusiasts"
        ),
        CurrentTrendItem(
            id="curr-us-2",
            title="Cold Plunge Morning Routine",
            category="Health & Fitness",
            platform="Instagram Reels",
            engagement="3.9M Views",
            viral_score=88,
            growth_pct=15,
            trend_strength=84,
            expected_duration="8 Days",
            status="Stable",
            thumbnail_url="https://images.unsplash.com/photo-1517838277536-f5f99be501cd?w=500&auto=format&fit=crop&q=60",
            hashtags=["#Biohacking", "#ColdPlunge", "#MorningRoutine"],
            audio_track="Deep Ambient Focus sound",
            keywords=["Dopamine Reset", "Dopamine Detox", "Ice Bath"],
            target_audience="Fitness Enthusiasts, Productivity Seekers"
        )
    ]
}

DEFAULT_PAST_TRENDS: List[PastTrendItem] = [
    PastTrendItem(
        id="past-def-1",
        title="Global AI Art Breakdown",
        category="Creative Tech",
        platform="Instagram Reels",
        trend_strength=88,
        peak_date="05 July",
        duration_days=14,
        status="Ended",
        why_viral="Stunning side-by-side render comparisons.",
        hashtags=["#AIArt", "#DigitalArt", "#CreativeTech"],
        audio_sample="Cinematic Ambient Drone",
        meme_format="Before and after transformation"
    )
]

DEFAULT_CURRENT_TRENDS: List[CurrentTrendItem] = [
    CurrentTrendItem(
        id="curr-def-1",
        title="Short-Form Storytelling & Hooks",
        category="Content Creation",
        platform="YouTube Shorts",
        engagement="5.1M Views",
        viral_score=91,
        growth_pct=31,
        trend_strength=91,
        expected_duration="10 Days",
        status="Growing",
        thumbnail_url="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=500&auto=format&fit=crop&q=60",
        hashtags=["#Storytelling", "#CreatorEconomy", "#HookMastery"],
        audio_track="Upbeat Modern Pulse",
        keywords=["Viral Hook", "Story Arc", "Retention Hack"],
        target_audience="Creators, Marketers, Vloggers"
    ),
    CurrentTrendItem(
        id="curr-def-2",
        title="Micro-Vlog 15-Second Day in the Life",
        category="Lifestyle",
        platform="Instagram Reels",
        engagement="4.2M Views",
        viral_score=87,
        growth_pct=22,
        trend_strength=87,
        expected_duration="6 Days",
        status="Stable",
        thumbnail_url="https://images.unsplash.com/photo-1492691527719-9d1e07e534b4?w=500&auto=format&fit=crop&q=60",
        hashtags=["#MicroVlog", "#DayInMyLife", "#AestheticVlog"],
        audio_track="Chilled Chillhop Beat",
        keywords=["Daily Vlog", "Morning Coffee", "Desk setup"],
        target_audience="Gen Z, Lifestyle Lovers"
    )
]

NOTIFICATIONS_LIST: List[NotificationItem] = [
    NotificationItem(
        id="notif-1",
        title="🔥 AI Productivity Surge",
        message="AI Productivity videos increased by 28% in the past 24 hours.",
        type="growth",
        timestamp="10 mins ago"
    ),
    NotificationItem(
        id="notif-2",
        title="🚀 Spider-Man Lifespan Alert",
        message="Spider-Man trend expected to stay active for 6 more days.",
        type="duration",
        timestamp="1 hour ago"
    ),
    NotificationItem(
        id="notif-3",
        title="📈 Tamil Nadu Regional Surge",
        message="Travel content rising rapidly in Tamil Nadu region (+42%).",
        type="location",
        timestamp="3 hours ago"
    ),
    NotificationItem(
        id="notif-4",
        title="🎵 New Trending Audio Detected",
        message="New acoustic lo-fi remix detected gaining 50k reels/hour.",
        type="audio",
        timestamp="5 hours ago"
    ),
    NotificationItem(
        id="notif-5",
        title="🔥 Festival Reels Momentum",
        message="Festival-related reels gaining major traction across South Asia.",
        type="viral",
        timestamp="1 day ago"
    ),
]

# -----------------------------------------------------------------------------
# Router Endpoints
# -----------------------------------------------------------------------------

@router.get("/countries", response_model=List[CountrySummary])
def get_supported_countries():
    """Return all supported country metadata for the world map and search."""
    return list(COUNTRIES_DATA.values())

@router.get("/country/{country_code}")
def get_country_intelligence(country_code: str):
    """
    Get detailed country intelligence data for the route:
    /creator-intelligence/country/{country}
    """
    code = country_code.upper()
    country = COUNTRIES_DATA.get(code)
    if not country:
        country = CountrySummary(
            id=code,
            name=country_code.replace("-", " ").title(),
            code=code,
            flag="🌍",
            population="Dynamic Population",
            languages=["Local", "English"],
            time_zone="Local Time Zone",
            major_platforms=["Instagram Reels", "YouTube Shorts"],
            current_trend_score=85,
            total_active_trends=95,
            coordinates={"lat": 20.0, "lng": 0.0}
        )

    past_trends = PAST_TRENDS_BY_COUNTRY.get(code, DEFAULT_PAST_TRENDS)
    current_trends = CURRENT_TRENDS_BY_COUNTRY.get(code, DEFAULT_CURRENT_TRENDS)

    top_20 = current_trends + [
        CurrentTrendItem(
            id=f"top20-{i}",
            title=f"Viral Highlight #{i+3}: {country.name} Regional Beat",
            category="Regional & Culture",
            platform="Instagram Reels" if i % 2 == 0 else "YouTube Shorts",
            engagement=f"{3.0 + i*0.4:.1f}M Views",
            viral_score=88 - i,
            growth_pct=25 - i,
            trend_strength=88 - i,
            expected_duration=f"{10 - i} Days",
            status="Growing" if i < 5 else "Stable",
            thumbnail_url="https://images.unsplash.com/photo-1518770660439-4636190af475?w=500&auto=format&fit=crop&q=60",
            hashtags=[f"#{country.name}Vibes", "#ViralToday"],
            audio_track="Regional Trending Sound",
            keywords=["Regional", "Culture", "Viral"],
            target_audience="Local & Global Audience"
        )
        for i in range(1, 17)
    ]

    return {
        "overview": country,
        "past_trends": past_trends,
        "current_trends": current_trends,
        "top_20_trends": top_20,
        "historical_timeline": [
            {"period": "Last Week", "total_viral": 42, "top_category": "Entertainment", "growth": "+18%"},
            {"period": "Last Month", "total_viral": 184, "top_category": "Lifestyle & Tech", "growth": "+34%"},
            {"period": "Last 3 Months", "total_viral": 512, "top_category": "Music & Dance", "growth": "+52%"},
            {"period": "Last Year", "total_viral": 1420, "top_category": "Vlogs & Shorts", "growth": "+110%"},
        ],
        "trend_forecast": [
            {"timeframe": "Next 7 Days", "predicted_top_trend": "AI Creative Tools & Cinematic Shorts", "confidence": "94%"},
            {"timeframe": "Next 30 Days", "predicted_top_trend": "Autumn/Festive Lifestyle Reels", "confidence": "88%"},
            {"timeframe": "Next 90 Days", "predicted_top_trend": "Year-End Recap & Tech Unboxing", "confidence": "82%"},
        ],
        "trending_reels": [t for t in top_20 if t.platform in ["Instagram Reels", "Both"]],
        "trending_shorts": [t for t in top_20 if t.platform in ["YouTube Shorts", "Both"]],
        "ai_recommendations": {
            "content_ideas": [
                f"Behind-the-scenes breakdown of {current_trends[0].title}",
                f"3-step quick tutorial leveraging high growth keywords in {country.name}",
                f"Reacting to peak regional viral trends with an original spin"
            ],
            "titles": [
                f"Why {current_trends[0].title} is Taking Over {country.name}",
                f"The 15-Second Secret to Going Viral in {country.name} Right Now",
                f"Don't Miss This Trend Before It Ends in 7 Days"
            ],
            "hooks": [
                "Stop scrolling if you want to know what everyone in " + country.name + " is talking about...",
                "I tested this viral trend so you don't have to, and here is what happened...",
                "Here is the exact formula behind today's #1 trending reel..."
            ],
            "cta": "Drop a comment with your thoughts and hit follow for daily trend intelligence!",
            "best_platform": country.major_platforms[0],
            "best_upload_time": "6:30 PM - 8:30 PM (Local Time)",
            "hashtags": current_trends[0].hashtags + ["#TrendIntelligence", "#CreatorStudio"],
            "keywords": current_trends[0].keywords,
            "target_audience": current_trends[0].target_audience,
            "suggested_duration": "15-30 Seconds"
        }
    }

@router.post("/analyze")
def analyze_trends(request: TrendAnalysisRequest):
    """
    Perform trend analysis for specified location and radius.
    Returns 30% Past Trends and 70% Current Trends.
    """
    country_code = request.country_code.upper()
    past = PAST_TRENDS_BY_COUNTRY.get(country_code, DEFAULT_PAST_TRENDS)
    current = CURRENT_TRENDS_BY_COUNTRY.get(country_code, DEFAULT_CURRENT_TRENDS)

    return {
        "location": request.location_name,
        "radius": request.radius_km,
        "country_code": country_code,
        "last_updated": "Today 6:42 PM IST",
        "past_trends_30_pct": past,
        "current_trends_70_pct": current,
    }

@router.post("/generate-content", response_model=GeneratedContentResponse)
def generate_similar_content(trend_id: str, trend_title: str):
    """
    Generate original non-plagiarized content script, hook, CTA, camera angles,
    BGM suggestions, and storyboard based on a trend.
    """
    return GeneratedContentResponse(
        trend_id=trend_id,
        trend_title=trend_title,
        original_script=(
            f"[SCENE START]\n"
            f"VISUAL: Dynamic push-in shot of creator holding a phone with dramatic lighting.\n"
            f"AUDIO: Energetic beat drops in sync with text animation.\n\n"
            f"HOOK: 'Everyone is talking about {trend_title}, but nobody is showing you this hidden detail...'\n\n"
            f"BODY: 'In the next 20 seconds, here are 3 quick insights that turn this trend into your next viral piece of content without copying anyone else.'\n\n"
            f"CTA: 'Save this post for your next shoot and follow for daily creator intel!'"
        ),
        better_hook=f"What if I told you the real reason {trend_title} went viral isn't what you think?",
        better_cta=f"Tag a fellow creator who needs to try this spin before the trend ends!",
        storyboard=[
            {"frame": "1", "shot": "Extreme Close-Up", "visual": "Eyes focused on screen, dramatic rim light", "audio": "Whisper hook"},
            {"frame": "2", "shot": "Medium Quick Cut", "visual": "Fast transition showing 3 key screen points", "audio": "Upbeat synth drop"},
            {"frame": "3", "shot": "Wide Dynamic Zoom", "visual": "Creator pointing to bold animated gold text", "audio": "Punchy sound effect"}
        ],
        scene_breakdown=[
            {"time": "0:00 - 0:03", "action": "Hook execution with instant text overlay"},
            {"time": "0:03 - 0:15", "action": "3 fast-paced core insights"},
            {"time": "0:15 - 0:25", "action": "Actionable takeaway & call to action"}
        ],
        camera_angles=["Low Angle Whip Pan", "Macro Lens Detail Shot", "Over-the-shoulder POV"],
        bgm_suggestions=["Cyberpunk Lofi Bass Drop", "Orchestral Trap Hybrid", "Upbeat Modern Tech Pulse"],
        voice_over_style="Confident, fast-paced, enthusiastic tone with crisp pauses.",
        thumbnail_idea=f"Creator with surprised reaction overlaying text: 'THE {trend_title.upper()} SECRET'",
        seo_title=f"How to Leverage {trend_title} for 10x Reach (Original Content Guide)",
        viral_keywords=[trend_title, "Creator Guide", "Viral Script", "Reels Hack", "Shorts Strategy"],
        suggested_duration="22 Seconds"
    )

@router.get("/notifications", response_model=List[NotificationItem])
def get_notifications():
    """Fetch live creator intelligence notifications."""
    return NOTIFICATIONS_LIST
