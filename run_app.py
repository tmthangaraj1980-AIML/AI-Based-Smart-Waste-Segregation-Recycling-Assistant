import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import webbrowser

STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
CHROME_NAMES = ["chrome", "google-chrome", "chromium", "chromium-browser"]


def find_chrome_executable():
    if sys.platform.startswith("win"):
        paths = [
            os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\%USERNAME%\AppData\Local"), "Google", "Chrome", "Application", "chrome.exe"),
        ]
        for path in paths:
            if os.path.exists(path):
                return path
    elif sys.platform.startswith("darwin"):
        path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(path):
            return path
    else:
        for name in CHROME_NAMES:
            path = shutil.which(name)
            if path:
                return path

    return None


def wait_for_server(host: str, port: int, timeout: float = 15.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.5)
    return False


def open_in_chrome(url: str) -> None:
    chrome_path = find_chrome_executable()
    if chrome_path:
        try:
            webbrowser.register("chrome", None, webbrowser.BackgroundBrowser(chrome_path))
            webbrowser.get("chrome").open(url, new=2)
            return
        except Exception:
            pass

    print("Chrome not found or failed to launch. Opening in the default browser instead.")
    webbrowser.open(url, new=2)


if __name__ == "__main__":
    command = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(STREAMLIT_PORT), "--server.headless", "true"]
    print("Starting Streamlit app...")
    process = subprocess.Popen(command)

    if wait_for_server("127.0.0.1", STREAMLIT_PORT, timeout=20):
        print(f"Streamlit server is ready at {STREAMLIT_URL}")
        open_in_chrome(STREAMLIT_URL)
    else:
        print(f"Unable to confirm Streamlit server on port {STREAMLIT_PORT}. You can still open {STREAMLIT_URL} manually.")

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        print("Streamlit server stopped.")
