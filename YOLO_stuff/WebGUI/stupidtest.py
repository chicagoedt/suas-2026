from flask import Flask, jsonify, render_template
import pyautogui

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/cursor")
def cursor_position():
    x, y = pyautogui.position()
    return jsonify({
        "x": x,
        "y": y
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=67, debug=True)