import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import PORT, POSES_DIR
from app.fingerspell import register as register_fingerspell


def create_app() -> FastAPI:
    if not POSES_DIR.is_dir():
        raise RuntimeError(f"Pose directory not found: {POSES_DIR}")

    app = FastAPI(title="Avatar pose files", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Count", "X-ILSA-Ms"],
    )

    @app.middleware("http")
    async def time_gets(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-ILSA-Ms"] = f"{ms:.1f}"
        if request.url.path.startswith("/tposes/"):
            print(f"[tpose] GET {request.url.path} {ms:.1f} ms", flush=True)
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    register_fingerspell(app)
    app.mount("/tposes", StaticFiles(directory=str(POSES_DIR)), name="tposes")
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
