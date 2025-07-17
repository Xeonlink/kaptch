from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import requests
import random
import string
import cv2
import numpy as np

app = Flask(__name__)

DATASET_DIR = os.path.join("dataset")

# AUTHCOMS 정의: name을 key로 하는 dict
AUTHCOMS = {
    "NICE": {
        "captcha_url": "https://nice.checkplus.co.kr/cert/captcha/image/c8edf290ed9767fd61776573a9880268d3fb12b489943ccd3cd6a1c3a5f00ff9",
        "folder": "NICE평가정보",
        "label_length": 6,
    },
    # 여기에 추가 authcom을 넣을 수 있음
}

# 앱 시작 시 폴더 미리 생성
for authcom in AUTHCOMS.values():
    folder = os.path.join(DATASET_DIR, authcom["folder"])
    os.makedirs(folder, exist_ok=True)


@app.route("/labeling/main.css")
def serve_main_css():
    return send_from_directory(os.path.dirname(__file__), "main.css")


@app.route("/labeling/main.js")
def serve_main_js():
    return send_from_directory(os.path.dirname(__file__), "main.js")


@app.route("/api/authcoms")
def api_authcoms():
    # parsing
    # (no input)

    # validating
    # (no validation needed)

    # business logic
    return jsonify(
        {
            "authcoms": [
                {
                    "name": name,
                    "folder": t["folder"],
                    "label_length": t["label_length"],
                    "captcha_url": t["captcha_url"],
                }
                for name, t in AUTHCOMS.items()
            ]
        }
    )


@app.route("/")
def labeling_tool():
    path = os.path.join("labeling", "labeling.html")
    with open(path, "r") as f:
        return f.read()


@app.route("/api/authcoms/<authcom>/labels", methods=["POST"])
def api_authcom_labels_post(authcom):
    # parsing
    data = request.get_json()
    label = data.get("label", "").strip()
    img_b64 = data.get("img_b64", "")
    authcom_name = authcom

    # validating
    t = AUTHCOMS.get(authcom_name)
    if not t:
        return jsonify({"ok": False, "error": "authcom not found"}), 400
    if not label or not img_b64:
        return jsonify({"ok": False, "error": "라벨 또는 이미지 없음"}), 400
    if len(label) != t["label_length"]:
        return jsonify({"ok": False, "error": f"라벨 길이는 {t['label_length']}자여야 합니다."}), 400
    folder = os.path.join(DATASET_DIR, t["folder"])

    # business logic
    for _ in range(10):
        h = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        fname = f"{label}_{h}.png"
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                cv2.imwrite(fpath, img)
                return jsonify({"ok": True})
            else:
                return jsonify({"ok": False, "error": "이미지 디코딩 실패"}), 400
    return jsonify({"ok": False, "error": "파일 저장 실패(중복 hash)"}), 500


@app.route("/api/authcoms/<authcom>/labels", methods=["GET"])
def api_authcom_labels_get(authcom):
    # parsing
    authcom_name = authcom

    # validating
    t = AUTHCOMS.get(authcom_name)
    if not t:
        return jsonify({"count": 0})
    folder = os.path.join(DATASET_DIR, t["folder"])

    # business logic
    try:
        count = len([f for f in os.listdir(folder) if f.endswith(".png")])
    except Exception:
        count = 0
    return jsonify({"count": count})


@app.route("/api/authcoms/<authcom>/image")
def api_authcom_image(authcom):
    """
    Proxy endpoint to fetch captcha image from external URL and return as base64 (for CORS bypass)
    """
    t = AUTHCOMS.get(authcom)
    if not t:
        return jsonify({"ok": False, "error": "authcom not found"}), 400
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://nice.checkplus.co.kr/",
        }
        resp = requests.get(t["captcha_url"], headers=headers, timeout=5, cookies={"JSESSIONID": "1234567890"})
        if resp.status_code != 200:
            return jsonify({"ok": False, "error": "이미지 요청 실패"}), 500
        img_bytes = resp.content
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return jsonify({"ok": True, "img_b64": img_b64})
    except Exception as e:
        return jsonify({"ok": False, "error": f"이미지 요청 예외: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=3000)
