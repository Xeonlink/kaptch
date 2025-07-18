/**
 * Dataset Labeling Tool (v3) - Simplified Version
 * 라벨링 도구의 메인 JavaScript 모듈
 * 히스토리 기능 제거 및 최적화된 버전
 */

// ==================== STATE ====================

/**
 * 애플리케이션 상태
 */
const state = {
    unlabeledIndices: [],
    curUnlabeled: 0,
    currentDataset: '',
    totalCount: 0,
    currentImage: null,
    currentLabel: '',
    isLoading: false,
    error: null
};

// ==================== DOM ELEMENTS ====================

/**
 * DOM 요소 참조
 */
const elements = {
    imagePreview: document.getElementById('image-preview'),
    labelInput: document.getElementById('label-input'),
    prevBtn: document.getElementById('prev-btn'),
    nextBtn: document.getElementById('next-btn'),
    progress: document.getElementById('progress'),
    step: document.getElementById('step'),
    datasetSelect: document.getElementById('dataset-select')
};

// ==================== UI UPDATE FUNCTIONS ====================

/**
 * 진행 상황 업데이트
 */
function updateProgress() {
    const labeledCount = state.totalCount - state.unlabeledIndices.length;
    elements.progress.textContent = `${labeledCount} / ${state.totalCount}`;
}

/**
 * 네비게이션 버튼 상태 업데이트
 */
function updateNavigationButtons() {
    // 이전 버튼은 항상 비활성화
    elements.prevBtn.disabled = true;
    
    // 다음 버튼은 라벨링할 이미지가 있을 때만 활성화
    const nextDisabled = state.curUnlabeled === state.unlabeledIndices.length - 1;
    elements.nextBtn.disabled = nextDisabled;
}

/**
 * 이미지 프리뷰 업데이트
 * @param {string|null} imageUrl - 표시할 이미지 URL 또는 null
 */
function updateImagePreview(imageUrl) {
    if (imageUrl) {
        elements.imagePreview.src = imageUrl;
        elements.imagePreview.style.display = '';
    } else {
        elements.imagePreview.style.display = 'none';
    }
}

/**
 * 라벨 입력 필드 업데이트
 * @param {string} label - 입력할 라벨 텍스트
 */
function updateLabelInput(label) {
    if (elements.labelInput.value !== label) {
        elements.labelInput.value = label || '';
    }
}

/**
 * 로딩 상태 업데이트
 * @param {boolean} isLoading - 로딩 상태 여부
 */
function updateLoadingState(isLoading) {
    if (isLoading) {
        elements.step.textContent = '로딩 중...';
    }
}

/**
 * 에러 상태 업데이트
 * @param {string|null} error - 에러 메시지 또는 null
 */
function updateErrorState(error) {
    if (error) {
        alert(error);
        state.error = null; // 에러 표시 후 초기화
    }
}

/**
 * UI 초기화
 */
function initializeUI() {
    elements.labelInput.style.display = '';
    elements.imagePreview.style.display = '';
    elements.labelInput.focus();
}

/**
 * 완료 상태 표시
 */
function showCompletion() {
    elements.imagePreview.style.display = 'none';
    elements.labelInput.style.display = 'none';
    elements.progress.textContent = '라벨이 필요한 이미지가 없습니다!';
    elements.step.textContent = '완료!';
}

// ==================== BUSINESS LOGIC ====================

/**
 * 사용 가능한 데이터셋 목록을 가져와서 셀렉트 박스에 표시
 */
async function fetchDatasets() {
    try {
        state.isLoading = true;
        updateLoadingState(true);
        
        const response = await fetch('/api/datasets');
        const data = await response.json();
        
        elements.datasetSelect.innerHTML = '';
        data.datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds;
            opt.textContent = ds;
            elements.datasetSelect.appendChild(opt);
        });
        
        if (data.datasets.length > 0) {
            state.currentDataset = data.datasets[0];
            await fetchAndInit();
        }
    } catch (error) {
        state.error = '데이터셋 목록을 불러올 수 없습니다.';
        console.error('데이터셋 목록을 가져오는데 실패했습니다:', error);
    } finally {
        state.isLoading = false;
        updateLoadingState(false);
    }
}

/**
 * 현재 데이터셋의 라벨링 정보를 가져와서 초기화
 */
async function fetchAndInit() {
    if (!state.currentDataset) return;
    
    try {
        state.isLoading = true;
        updateLoadingState(true);
        
        const response = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}`);
        const data = await response.json();
        
        if (data.error) {
            state.error = '데이터셋을 불러올 수 없습니다: ' + data.error;
            return;
        }
        
        state.unlabeledIndices = data.unlabeled_indices;
        state.totalCount = data.total_count;
        state.curUnlabeled = 0;
        
        if (state.unlabeledIndices.length > 0) {
            await showCurrentImage();
            initializeUI();
        } else {
            showCompletion();
        }
        
        updateProgress();
        updateNavigationButtons();
    } catch (error) {
        state.error = '데이터셋을 불러올 수 없습니다.';
        console.error('데이터셋 초기화에 실패했습니다:', error);
    } finally {
        state.isLoading = false;
        updateLoadingState(false);
    }
}

/**
 * 현재 라벨링할 이미지를 화면에 표시
 */
async function showCurrentImage() {
    if (state.unlabeledIndices.length === 0) return;
    
    const index = state.unlabeledIndices[state.curUnlabeled];
    
    try {
        const response = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/images/${index}`);
        if (!response.ok) throw new Error('이미지 없음');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        state.currentImage = url;
        state.currentLabel = '';
        
        updateImagePreview(url);
        updateLabelInput('');
    } catch (error) {
        state.currentImage = null;
        state.error = '이미지 파일이 없습니다: 인덱스 ' + index;
        updateImagePreview(null);
    }
}

/**
 * 현재 이미지에 라벨을 저장하고 다음 이미지로 이동
 */
async function saveLabelAndNext() {
    if (state.unlabeledIndices.length === 0) return;
    
    const index = state.unlabeledIndices[state.curUnlabeled];
    const label = elements.labelInput.value.trim();
    
    if (label === '') {
        state.error = '라벨을 입력해주세요.';
        return;
    }

    try {
        const response = await fetch(`/api/datasets/${encodeURIComponent(state.currentDataset)}/labels/${index}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label })
        });
        
        const res = await response.json();
        if (res.ok) {
            // 현재 인덱스를 라벨링 완료된 목록에서 제거
            state.unlabeledIndices.splice(state.curUnlabeled, 1);
            if (state.curUnlabeled >= state.unlabeledIndices.length) {
                state.curUnlabeled = Math.max(0, state.unlabeledIndices.length - 1);
            }
            
            // 라벨 입력 필드 초기화
            state.currentLabel = '';
            updateLabelInput('');
            
            if (state.unlabeledIndices.length > 0) {
                await showCurrentImage();
                elements.labelInput.focus();
            } else {
                // 모든 이미지가 라벨링 완료되면 다시 초기화
                await fetchAndInit();
            }
            
            updateProgress();
            updateNavigationButtons();
        } else {
            state.error = '라벨 저장 실패: ' + (res.error || '');
        }
    } catch (error) {
        state.error = '라벨 저장에 실패했습니다.';
        console.error('라벨 저장에 실패했습니다:', error);
    }
}

/**
 * 다음 버튼 클릭 처리
 */
async function handleNextButton() {
    if (state.curUnlabeled < state.unlabeledIndices.length - 1) {
        state.curUnlabeled++;
        await showCurrentImage();
        elements.labelInput.focus();
        updateProgress();
        updateNavigationButtons();
    }
}

/**
 * 라벨 입력 필드 키보드 이벤트 처리
 * @param {KeyboardEvent} e - 키보드 이벤트 객체
 */
async function handleLabelInputKeydown(e) {
    if (e.key === 'Enter') {
        await saveLabelAndNext();
        e.preventDefault();
    }
}

/**
 * 데이터셋 선택 변경 처리
 */
async function handleDatasetChange() {
    state.currentDataset = elements.datasetSelect.value;
    await fetchAndInit();
}

// ==================== INITIALIZATION ====================

/**
 * 애플리케이션 초기화
 */
function initializeApp() {
    // 이벤트 리스너 등록
    elements.labelInput.addEventListener('keydown', async (e) => await handleLabelInputKeydown(e));
    elements.nextBtn.addEventListener('click', async () => await handleNextButton());
    elements.datasetSelect.addEventListener('change', async () => await handleDatasetChange());
    
    // 애플리케이션 시작
    fetchDatasets();
}

// 애플리케이션 시작 (DOM이 로드된 후 실행)
document.addEventListener('DOMContentLoaded', initializeApp);