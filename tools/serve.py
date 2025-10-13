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

    click.secho(f"Starting Mlflow model server on port {port} ...", fg="cyan")

    try:
        result = subprocess.run(
            [
                "mlflow",
                "models",
                "serve",
                "-m",
                f"models:/{settings.BEST_MODEL_REGISTRY_NAME}/latest",
                "--port",
                str(port),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            click.secho("Mlflow model server exited successfully.", fg="green")
        else:
            click.secho("Mlflow model serving failed!", fg="red", bold=True)
            click.echo(f"\nExit Code: {result.returncode}")
            click.echo(f"Stdout:\n{result.stdout.strip() or '(empty)'}")
            click.echo(f"Stderr:\n{result.stderr.strip() or '(empty)'}")

    except FileNotFoundError:
        click.secho("Error: Mlflow CLI not found. Make sure Mlflow is installed.", fg="red", bold=True)
    except Exception as e:
        click.secho(f"Unexpected error: {e}", fg="red", bold=True)


if __name__ == "__main__":
    main()
