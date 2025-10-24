"""
main_api.py
=========================
FastAPI entry point for model inference service.

Features
--------
- Centralized API for model predictions
- Integrated logging (Loguru)
- Input validation via Pydantic
- Ready for Docker deployment

Author: Rostand Surel
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#from obesity_predictor.config.logger_config import logger
from obesity_predictor.api.routers.prediction_router import router as prediction_router


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Obesity Predictor API",
        description="Predict obesity class using trained ML models.",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(prediction_router, prefix="/api/v1/predict", tags=["Prediction"])

    @app.get("/")
    def health_check():
        """
        Health check endpoint to verify the API is running.
        """
        #logger.info("[API] Health check called.")
        return {"status": "ok", "message": "Obesity Predictor API is live 🚀"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("obesity_predictor.api.main_api:app", host="0.0.0.0", port=8000, reload=True)