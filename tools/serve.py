import os
import subprocess

import click

from core import settings


@click.command()
@click.option("--port", default=settings.MODEL_SERVE_PORT, show_default=True, help="Port to serve the Mlflow model on.")
def main(port):
    """
    Serve the latest registered Mlflow model locally.
    """
    os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI

    click.echo(f"Starting Mlflow model server on port {port} ...")

    subprocess.run(
        ["mlflow", "models", "serve", "-m", f"models:/{settings.BEST_MODEL_REGISTRY_NAME}/latest", "--port", str(port)]
    )

    click.echo("Mlflow model server stopped.")


if __name__ == "__main__":
    main()
