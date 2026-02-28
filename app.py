from __future__ import annotations

from flask import Flask, jsonify, render_template

from nifty_analyzer import AnalysisService

app = Flask(__name__)
service = AnalysisService()


@app.route("/")
def dashboard():
    try:
        data = service.run()
        return render_template("index.html", data=data, error=None)
    except Exception as exc:
        return render_template("index.html", data=None, error=str(exc))


@app.route("/api/analyze")
def analyze_api():
    data = service.run()
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
