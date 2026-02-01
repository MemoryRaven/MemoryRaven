from __future__ import annotations

import asyncio

import uvicorn

from .app import create_app
from .db import DB
from .settings import settings


async def _main() -> None:
    db = await DB.connect(settings.postgres_dsn)
    app = create_app(db)

    config = uvicorn.Config(
        app,
        host=settings.bind_host,
        port=settings.bind_port,
        log_level=(settings.log_level or "info").lower(),
        loop="uvloop",
        http="httptools",
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    finally:
        await db.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
