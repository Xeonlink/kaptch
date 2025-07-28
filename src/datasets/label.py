from flask import Flask, request, send_from_directory, jsonify
import csv
import os
from src.constants import DATA_CSV


app = Flask(__name__)


@app.route("/")
def root():
    """루트 경로에서 index.html을 서빙합니다.

    Returns:
        Response: index.html 파일
    """
    home_dir = os.path.join(os.path.dirname(__file__), "label")
    return send_from_directory(home_dir, "index.html")


@app.route("/<path:name>")
def home(name):
    """홈 디렉토리의 정적 파일들을 서빙합니다.

    Parameters:
        name (str): 요청된 파일 경로

    Returns:
        Response: 요청된 파일 또는 404 에러
    """
    home_dir = os.path.join(os.path.dirname(__file__), "label")
    return send_from_directory(home_dir, name)


@app.route("/api/datasets")
def get_datasets():
    datasets = os.listdir("dataset")
    datasets = list(filter(lambda x: os.path.isdir(os.path.join("dataset", x)), datasets))
    return jsonify({"datasets": datasets})


@app.route("/api/datasets/<dataset>")
def get_dataset(dataset: str):
    """dataset의 전체 데이터수와 label이 없는 이미지의 index배열을 반환합니다.

    Parameters:
        dataset (str): 데이터셋 이름

    Returns:
        dict: 데이터셋 정보를 포함한 JSON 응답
    """

    csv_path = os.path.join("dataset", dataset, DATA_CSV)
    if not os.path.exists(csv_path):
        return jsonify({"error": "Dataset not found"}), 404

    unlabeled_indices: list[int] = []
    total_count = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            total_count += 1
            # 2열(인덱스 1)이 비어있으면 라벨이 없는 것
            if len(row) < 2 or not row[1].strip():
                unlabeled_indices.append(i)

    return jsonify({"total_count": total_count, "unlabeled_indices": unlabeled_indices})


@app.route("/api/datasets/<dataset>/images/<int:index>")
def get_images(dataset: str, index: int):
    """특정 인덱스의 이미지를 반환합니다.

    Parameters:
        dataset (str): 데이터셋 이름
        index (int): 이미지 인덱스

    Returns:
        Response: 이미지 파일 또는 에러 응답
    """
    csv_path = os.path.join("dataset", dataset, DATA_CSV)
    if not os.path.exists(csv_path):
        return jsonify({"error": "Dataset not found"}), 404

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            # index번째 행까지 건너뛰기
            for _ in range(index):
                next(reader, None)

            row = next(reader)

            return send_from_directory(os.path.join(os.getcwd(), "dataset", dataset), row[0])
        except StopIteration:
            return jsonify({"error": "Index not found"}), 404


@app.route("/api/datasets/<dataset>/labels/<int:index>", methods=["POST"])
def post_labels(dataset: str, index: int):
    """특정 인덱스의 라벨을 업데이트합니다.

    Parameters:
        dataset (str): 데이터셋 이름
        index (int): 업데이트할 행의 인덱스

    Returns:
        Response: 업데이트 결과를 포함한 JSON 응답
    """
    data = request.get_json()
    label = data.get("label", "")
    csv_path = os.path.join("dataset", dataset, DATA_CSV)
    if not os.path.exists(csv_path):
        return jsonify({"error": "Dataset not found"}), 404

    # 전체 파일을 읽어서 메모리에 저장
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    # 인덱스가 유효한지 확인
    if index >= len(rows):
        return jsonify({"error": "Index out of range"}), 404

    # 해당 행의 라벨 업데이트
    rows[index][1] = label

    # 전체 파일을 다시 쓰기
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return jsonify({"ok": True, "label": label, "index": index})


if __name__ == "__main__":
    debug = True
    port = 3000
    app.run(debug=debug, port=port)
