import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    from waitress import serve
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    if os.getenv("FLASK_DEBUG", "0") == "1":
        app.run(host=host, port=port, debug=True)
    else:
        serve(app, host=host, port=port, threads=int(os.getenv("WEB_THREADS", "8")))
