import os
import shutil
import signal
import subprocess
import sys
import time

import click

from core import settings


@click.command()
@click.option("--port", default=settings.MODEL_SERVE_PORT, show_default=True, help="Port to serve the MLflow model on.")
def main(port: int) -> None:
    """Serve the latest registered MLflow model locally (w/error handling & shutdown)."""
    os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI

    # Pre-flight: ensure `mlflow` is available on PATH
    if shutil.which("mlflow") is None:
        click.secho("Error: 'mlflow' CLI not found on PATH. Install mlflow or adjust your PATH.", fg="red", bold=True)
        sys.exit(127)

    click.secho(f"Starting MLflow model server on port {port} ...", fg="cyan")

    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        f"models:/{settings.BEST_MODEL_REGISTRY_NAME}/latest",
        "--port",
        str(port),
    ]

    # Cross-platform process-group setup so we can signal the whole tree
    is_windows = os.name == "nt"
    popen_kwargs = {}

    if is_windows:
        # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK_EVENT
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # start_new_session=True -> new process group; we can send signals to pgid
        popen_kwargs["start_new_session"] = True

    process = None
    try:
        # Inherit parent stdio so logs stream live to console
        process = subprocess.Popen(cmd, **popen_kwargs)
        # Wait until child exits (or we get Ctrl+C)
        return_code = process.wait()
        if return_code == 0:
            click.secho("MLflow model server exited successfully.", fg="green")
        else:
            click.secho(f"MLflow server exited with code {return_code}.", fg="red", bold=True)
        sys.exit(return_code)

    except FileNotFoundError:
        click.secho("Error: 'mlflow' command not found (disappeared after pre-check).", fg="red", bold=True)
        sys.exit(127)

    except KeyboardInterrupt:
        click.secho("\nStopping MLflow model server...", fg="yellow")

        # Try graceful shutdown first
        try:
            if process and process.poll() is None:
                if is_windows:
                    # Prefer CTRL_BREAK_EVENT when we created a new process group
                    try:
                        process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    except Exception:
                        process.terminate()
                else:
                    # Send SIGINT to the whole process group
                    os.killpg(process.pid, signal.SIGINT)
                # Give it a moment to exit cleanly
                _grace_deadline = time.time() + 8
                while time.time() < _grace_deadline and process.poll() is None:
                    time.sleep(0.1)

            # If still alive, escalate
            if process and process.poll() is None:
                click.secho("Graceful stop timed out; terminating...", fg="yellow")
                process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    click.secho("Terminate timed out; killing...", fg="red")
                    process.kill()
                    process.wait(timeout=5)

        except Exception as e:
            click.secho(f"Shutdown encountered an issue: {e}", fg="red")

        finally:
            click.secho("Server stopped.", fg="green")
            # Mirror typical Ctrl+C exit code
            sys.exit(130)

    except Exception as e:
        click.secho(f"Unexpected error while starting MLflow server: {e}", fg="red", bold=True)
        # If the child started but failed quickly, propagate its code if available
        if process and process.poll() is not None:
            sys.exit(process.returncode)
        sys.exit(1)


if __name__ == "__main__":
    main()
