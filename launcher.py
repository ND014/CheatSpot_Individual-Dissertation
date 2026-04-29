import threading

import webview

from app import app  # your Flask app


def start_flask():
    app.run(port=5001, debug=False, use_reloader=False)


if __name__ == "__main__":
    t = threading.Thread(target=start_flask, daemon=True)
    t.start()

    window = webview.create_window(
        "CheatSpot - Exam Cheating Detection",
        "http://localhost:5001",
        width=1280,
        height=800,
        resizable=True,
    )
    webview.start()