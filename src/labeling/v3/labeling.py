from flask import Flask, request, send_from_directory, jsonify, Response
import os
import re

app = Flask(__name__)

# 여러 데이터셋 지원
DATASETS = {
    "NICE평가정보": {
        "dir": os.path.join("dataset", "NICE평가정보"),
        "img_pattern": re.compile(r"^(?:captcha|([A-Za-z0-9]{6}))_([A-Za-z0-9]{4})\.png$"),
    },
    "NHN_KCP": {
        "dir": os.path.join("dataset", "NHN_KCP"),
        "img_pattern": re.compile(r"^(?:captcha|([A-Za-z0-9]{5}))_([A-Za-z0-9]{4})\.png$"),
    },
    "한국모바일인증": {
        "dir": os.path.join("dataset", "한국모바일인증1"),
        "img_pattern": re.compile(r"^(?:captcha|([A-Za-z0-9]{5}))_([A-Za-z0-9]{4})\.png$"),
    },
    "sci": {
        "dir": os.path.join("dataset", "sci"),
        "img_pattern": re.compile(r"^(?:captcha|([A-Za-z0-9]{6}))_([A-Za-z0-9]{4})\.png$"),
    },
}


@app.route("/")
def labeling_tool():
    with open(os.path.join(os.path.dirname(__file__), "labeling.html"), "r") as f:
        return f.read()


@app.route("/main.js")
def serve_main_js():
    return send_from_directory(os.path.dirname(__file__), "main.js")


@app.route("/main.css")
def serve_main_css():
    return send_from_directory(os.path.dirname(__file__), "main.css")


@app.route("/api/datasets")
def api_datasets():
    return jsonify({"datasets": list(DATASETS.keys())})


@app.route("/api/images")
def api_images():
    dataset = request.args.get("dataset", "")
    if dataset not in DATASETS:
        return jsonify({"images": []})
    files = os.listdir(DATASETS[dataset]["dir"])
    unlabeled = [f for f in files if f.startswith("captcha_") and f.endswith(".png")]
    return jsonify({"images": unlabeled})


@app.route("/api/image/<dataset>/<img_name>")
def api_image(dataset, img_name):
    if dataset not in DATASETS:
        return Response("Not found", status=403)

    path = os.path.join(DATASETS[dataset]["dir"], img_name)
    if not os.path.exists(path):
        return Response("Not found", status=403)

    dir_path = os.path.abspath(DATASETS[dataset]["dir"])
    return send_from_directory(dir_path, img_name)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    data = request.get_json()
    img_name = data.get("img_name", "")
    label = data.get("label", "")
    dataset = data.get("dataset", "")
    if dataset not in DATASETS:
        return jsonify({"ok": False, "error": "Invalid dataset"})
    m = DATASETS[dataset]["img_pattern"].match(img_name)
    if not m or not label:
        return jsonify({"ok": False, "error": "Invalid filename or label"})
    hash_part = m.group(2)
    new_name = f"{label}_{hash_part}.png"
    src = os.path.join(DATASETS[dataset]["dir"], img_name)
    dst = os.path.join(DATASETS[dataset]["dir"], new_name)
    if not os.path.exists(src):
        return jsonify({"ok": False, "error": "File not found"})
    if os.path.exists(dst):
        return jsonify({"ok": False, "error": "Target file already exists"})
    os.rename(src, dst)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=3000)
