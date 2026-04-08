"""FastAPI app for SupportOps OpenEnv."""

from openenv.core.env_server.http_server import create_app

from support_env.env import SupportOpsEnvironment
from support_env.models import SupportAction, SupportObservation


app = create_app(
    SupportOpsEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
