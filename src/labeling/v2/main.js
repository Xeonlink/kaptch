let authcoms = [];
let currentAuthcom = '';
let labelLength = 6;
let currentImgElement = null;

function updateLabelCount() {
    fetch(`/api/authcoms/${encodeURIComponent(currentAuthcom)}/labels`)
        .then(r => r.json())
        .then(data => {
            document.getElementById('label-count').textContent = '라벨링된 이미지 수: ' + (data.count ?? 0);
        });
}

function fetchImage() {
    setStatus('이미지 불러오는 중...');
    fetch(`/api/authcoms/${encodeURIComponent(currentAuthcom)}/image`)
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                const img = document.getElementById('image-preview');
                img.onload = () => setStatus('');
                img.onerror = () => setStatus('이미지 로딩 실패');
                img.src = 'data:image/png;base64,' + data.img_b64;
                img.style.display = '';
                document.getElementById('label-input').value = '';
                document.getElementById('label-input').focus();
                currentImgElement = img;
                updateLabelCount();
                console.log('fetchImage called, base64 length:', data.img_b64.length);
            } else {
                setStatus(data.error || '이미지 불러오기 실패');
            }
        })
        .catch(err => {
            setStatus('이미지 요청 오류: ' + err);
            console.error('fetchImage error:', err);
        });
}

function getCanvasBase64WithWhiteBG(imgElement) {
    const w = imgElement.naturalWidth;
    const h = imgElement.naturalHeight;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(imgElement, 0, 0, w, h);
    return canvas.toDataURL('image/png').replace(/^data:image\/png;base64,/, '');
}

function submitLabel() {
    const labelInput = document.getElementById('label-input');
    const label = labelInput.value.trim();
    if (!label) {
        setStatus('라벨을 입력하세요.');
        return;
    }
    if (label.length !== labelLength) {
        setStatus('라벨 길이는 ' + labelLength + '자여야 합니다.');
        return;
    }
    const img = document.getElementById('image-preview');
    let imgB64 = '';
    if (img && img.naturalWidth && img.naturalHeight) {
        imgB64 = getCanvasBase64WithWhiteBG(img);
    }
    setStatus('저장 중...');
    console.log('submitLabel called, label:', label, 'imgB64.length:', imgB64.length);
    fetch(`/api/authcoms/${encodeURIComponent(currentAuthcom)}/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, img_b64: imgB64 })
    })
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                fetchImage();
            } else {
                setStatus(data.error || '저장 실패');
                updateLabelCount();
            }
        })
        .catch(err => {
            setStatus('저장 중 오류 발생: ' + err);
            console.error('submitLabel fetch error:', err);
        });
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

function onAuthcomChange() {
    const sel = document.getElementById('authcom-select');
    currentAuthcom = sel.value;
    const t = authcoms.find(x => x.name === currentAuthcom);
    labelLength = t ? t.label_length : 6;
    const labelInput = document.getElementById('label-input');
    labelInput.setAttribute('maxlength', labelLength);
    labelInput.setAttribute('placeholder', '라벨을 입력하세요 (' + labelLength + '자)');
    updateLabelCount();
    fetchImage();
}

document.addEventListener('DOMContentLoaded', () => {
    fetch('/api/authcoms')
        .then(r => r.json())
        .then(data => {
            authcoms = data.authcoms;
            const sel = document.getElementById('authcom-select');
            sel.innerHTML = '';
            authcoms.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t.name;
                opt.textContent = t.name + ' (' + t.folder + ', ' + t.label_length + '자)';
                sel.appendChild(opt);
            });
            currentAuthcom = authcoms[0].name;
            labelLength = authcoms[0].label_length;
            const labelInput = document.getElementById('label-input');
            labelInput.setAttribute('maxlength', labelLength);
            labelInput.setAttribute('placeholder', '라벨을 입력하세요 (' + labelLength + '자)');
            updateLabelCount();
            fetchImage();
            // Enter 이벤트 바인딩
            labelInput.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') {
                    submitLabel();
                }
            });
        });
    document.getElementById('authcom-select').addEventListener("change", () => {
        onAuthcomChange();
    });
    document.getElementById('fetch-btn').addEventListener("click", () => {
        fetchImage();
    });
    document.getElementById('submit-btn').addEventListener("click", () => {
        submitLabel();
    });
});