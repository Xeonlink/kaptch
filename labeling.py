from flask import Flask, request, send_from_directory, jsonify, Response
import os
import csv
from typing import Optional

app = Flask(__name__)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TRAIN_CSV = os.path.join(DATASET_DIR, "train_list.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test_list.csv")


@app.route("/")
def labeling_tool():
    with open("labeling.html", "r") as f:
        return f.read()


def get_csv_path(csv_type: str) -> Optional[str]:
    if csv_type == "train":
        return TRAIN_CSV
    elif csv_type == "test":
        return TEST_CSV
    return None


@app.route("/api/csv")
def api_csv():
    csv_type = request.args.get("type", "train")
    csv_path = get_csv_path(csv_type)
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"rows": [], "labelColIdx": 1})
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    label_col_idx = 1  # 항상 두 번째 컬럼이 라벨
    return jsonify({"rows": rows, "labelColIdx": label_col_idx})


@app.route("/api/image/<path:img_path>")
def api_image(img_path):
    # img_path는 train/0000/0000.png 등 상대경로
    abs_path = os.path.join(DATASET_DIR, img_path)
    if not os.path.exists(abs_path):
        return Response("Not found", status=404)
    dir_name = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    return send_from_directory(dir_name, file_name)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    print("test")
    data = request.get_json()
    csv_type = data.get("type", "train")
    idx = int(data.get("idx", 0))
    label = data.get("label", "")
    csv_path = get_csv_path(csv_type)
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"ok": False})
    # CSV 파일을 읽고 수정 후 저장
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if 0 <= idx < len(rows):
        rows[idx][1] = label  # 항상 두 번째 컬럼이 라벨
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=3000)
