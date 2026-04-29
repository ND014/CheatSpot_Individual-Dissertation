import threading
import webview
from app import app

# ========================
# CONFIG
# ========================
APP_URL = "http://127.0.0.1:5001"
WINDOW_TITLE = "CheatSpot - Exam Cheating Detection"


# ========================
# FLASK SERVER
# ========================
def start_flask():
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )


# ========================
# MAIN
# ========================
def main():
    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Create desktop window
    webview.create_window(
        title=WINDOW_TITLE,
        url=APP_URL,
        width=1280,
        height=800,
        resizable=True
    )

    webview.start()


if __name__ == "__main__":
    main()