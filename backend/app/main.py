import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analysis, auth, creator_intelligence, dashboard, payments, scripts, studio
from app.core.config import settings
from app.database import Base, engine

# Import all models so SQLAlchemy creates every table.
from app.models import (  # noqa: F401
    analysis_result,
    character,
    dialogue,
    generated_artifact,
    scene,
    scene_emotion,
    script,
    storyboard_panel,
    user,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _warm_ai_models() -> None:
    """
    Load both ML models once at process startup rather than on the first
    user request. This makes the *first* script upload after a deploy take
    the same amount of time as every subsequent one, instead of silently
    stalling the first user's request for 10-30s while the models load.

    If model loading fails (e.g. weights haven't been downloaded yet, see
    ml_models/README.md), we log it and let the app keep starting --
    individual pipeline runs will raise a clear error instead of the whole
    API failing to boot.
    """
    try:
        from app.ai import emotion as emotion_ai

        emotion_ai._get_pipeline()
        logger.info("Emotion model warmed.")
    except Exception:
        logger.exception("Could not warm the emotion model at startup.")

    try:
        from app.ai import ner as ner_ai

        ner_ai.get_nlp()
        logger.info("NER model warmed.")
    except Exception:
        logger.exception("Could not warm the NER model at startup.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _warm_ai_models()
    yield
    # No teardown needed: SQLAlchemy's engine/connection pool and the AI
    # model singletons in app/ai/ don't hold resources that need explicit
    # release on shutdown.


app = FastAPI(
    title="AI Pre-Production Studio",
    description=(
        "Backend API for AI Pre-Production Studio. Upload a screenplay and the full "
        "NLP/AI pipeline (character tracking, emotion analysis, shot suggestions, BGM "
        "recommendations, storyboard, scene importance, cost estimation) runs "
        "automatically -- see GET /scripts/{script_id}/dashboard for the unified result."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# The CORS spec forbids the "*" wildcard together with credentials, and
# browsers reject that combination outright. This API authenticates with
# bearer tokens in the Authorization header (not cookies), so credentials
# are only enabled when a concrete origin allowlist is configured.
_cors_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(scripts.router)
app.include_router(analysis.router)
app.include_router(dashboard.router)
app.include_router(payments.router)
app.include_router(studio.router)
app.include_router(creator_intelligence.router)


@app.get("/")
def root():
    return {"message": "AI Pre-Production Studio API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
