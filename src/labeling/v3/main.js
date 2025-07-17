// main.js for labeling tool (v3)
// All logic previously in labeling.html is moved here and refactored for multi-dataset support

document.addEventListener('DOMContentLoaded', () => {
    const imagePreview = document.getElementById('image-preview');
    const labelInput = document.getElementById('label-input');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const progress = document.getElementById('progress');
    const step = document.getElementById('step');
    const datasetSelect = document.getElementById('dataset-select');

    let unlabeledImages = [];
    let curUnlabeled = 0;
    let currentDataset = '';

    function fetchDatasets() {
        fetch('/api/datasets')
            .then(r => r.json())
            .then(data => {
                datasetSelect.innerHTML = '';
                data.datasets.forEach(ds => {
                    const opt = document.createElement('option');
                    opt.value = ds;
                    opt.textContent = ds;
                    datasetSelect.appendChild(opt);
                });
                if (data.datasets.length > 0) {
                    currentDataset = data.datasets[0];
                    datasetSelect.value = currentDataset;
                    fetchAndInit();
                }
            });
    }

    function fetchAndInit() {
        if (!currentDataset) return;
        fetch(`/api/images?dataset=${encodeURIComponent(currentDataset)}`)
            .then(r => r.json())
            .then(data => {
                unlabeledImages = data.images;
                curUnlabeled = 0;
                if (unlabeledImages.length > 0) {
                    showImage();
                    labelInput.style.display = '';
                    imagePreview.style.display = '';
                    prevBtn.disabled = false;
                    nextBtn.disabled = false;
                    labelInput.focus();
                    step.textContent = '라벨을 입력하고 Enter를 누르세요.';
                } else {
                    imagePreview.style.display = 'none';
                    labelInput.style.display = 'none';
                    progress.textContent = '라벨이 필요한 이미지가 없습니다!';
                    step.textContent = '완료!';
                }
            });
    }

    function showImage() {
        if (unlabeledImages.length === 0) return;
        const imgName = unlabeledImages[curUnlabeled];
        fetch(`/api/image/${encodeURIComponent(currentDataset)}/${encodeURIComponent(imgName)}`)
            .then(r => {
                if (!r.ok) throw new Error('이미지 없음');
                return r.blob();
            })
            .then(blob => {
                const url = URL.createObjectURL(blob);
                imagePreview.src = url;
                imagePreview.onload = () => URL.revokeObjectURL(url);
            })
            .catch(() => {
                imagePreview.style.display = 'none';
                progress.textContent = '이미지 파일이 없습니다: ' + imgName;
            });
        labelInput.value = '';
        progress.textContent = `${unlabeledImages.length}`;
        prevBtn.disabled = curUnlabeled === 0;
        nextBtn.disabled = curUnlabeled === unlabeledImages.length - 1;
    }

    labelInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            saveLabelAndNext();
            e.preventDefault();
        }
    });

    function saveLabelAndNext() {
        if (unlabeledImages.length === 0) return;
        const imgName = unlabeledImages[curUnlabeled];
        const label = labelInput.value;
        fetch('/api/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                img_name: imgName,
                label: label,
                dataset: currentDataset
            })
        }).then(r => r.json()).then(res => {
            if (res.ok) {
                unlabeledImages.splice(curUnlabeled, 1);
                if (curUnlabeled >= unlabeledImages.length) {
                    curUnlabeled = unlabeledImages.length - 1;
                }
                if (unlabeledImages.length > 0) {
                    showImage();
                    labelInput.focus();
                } else {
                    fetchAndInit();
                }
            } else {
                alert('라벨 저장 실패: ' + (res.error || ''));
            }
        });
    }

    prevBtn.addEventListener('click', () => {
        if (curUnlabeled > 0) {
            curUnlabeled--;
            showImage();
            labelInput.focus();
        }
    });
    nextBtn.addEventListener('click', () => {
        if (curUnlabeled < unlabeledImages.length - 1) {
            curUnlabeled++;
            showImage();
            labelInput.focus();
        }
    });
    datasetSelect.addEventListener('change', () => {
        currentDataset = datasetSelect.value;
        fetchAndInit();
    });

    fetchDatasets();
});
