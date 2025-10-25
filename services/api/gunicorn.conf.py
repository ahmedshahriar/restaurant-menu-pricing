workers = 2  # start small; set ~= CPU cores
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
timeout = 30  # keep ≤ AML timeout
graceful_timeout = 30
keepalive = 5
accesslog = "-"  # stdout
errorlog = "-"
loglevel = "info"
