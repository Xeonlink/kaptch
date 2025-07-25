# Kaptch - 한국 모바일인증 캡챠 자동 인식 파이프라인

한국 모바일인증 서비스의 캡챠를 자동으로 인식하는 머신러닝 파이프라인입니다. CRNN(Convolutional Recurrent Neural Network) 모델을 사용하여 캡챠 이미지의 숫자를 정확하게 인식합니다.

## 🚀 주요 기능

- **자동 데이터 수집**: Playwright를 사용한 웹 크롤링
- **웹 기반 라벨링**: 브라우저를 통한 편리한 라벨링 인터페이스
- **모델 훈련**: CRNN 모델을 사용한 캡챠 인식 훈련
- **체크포인트 관리**: 훈련 과정의 체크포인트 저장 및 관리
- **ONNX 변환**: 추론 최적화를 위한 ONNX 모델 변환

## 📋 지원하는 인증업체

- **NHN KCP**: NHN KCP 모바일인증
- **NICE 평가정보**: NICE 본인확인
- **SCI 평가정보**: SCI 본인확인
- **KMCERT**: KMCERT 본인확인

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 기본 사용법
```bash
# 메인 CLI 실행
python main.py --help

# 데이터셋 생성
python main.py datasets create sci

# 캡챠 수집
python main.py datasets crawl sci

# 라벨링 서버 시작
python main.py datasets label

# 모델 훈련
python main.py train sci

# 체크포인트 확인
python main.py checkpoints list sci

# ONNX 변환
python main.py misc torch2onnx sci ep4.pkl
```

## 📖 상세 사용법

### 데이터셋 관리

#### 데이터셋 생성
```bash
python main.py datasets create <데이터셋명>
```
새로운 데이터셋을 생성합니다. 하이픈은 자동으로 언더스코어로 변환됩니다.

#### 캡챠 수집
```bash
python main.py datasets crawl <데이터셋명> [--goal 1100] [--headless]
```
웹사이트에서 캡챠 이미지를 자동으로 수집합니다.

#### 라벨링
```bash
python main.py datasets label [--port 3000]
```
웹 브라우저를 통해 캡챠 이미지에 라벨을 붙일 수 있는 서버를 시작합니다.

### 모델 훈련

#### 기본 훈련
```bash
python main.py train <데이터셋명>
```

#### 고급 훈련 옵션
```bash
python main.py train <데이터셋명> \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --patience 10 \
  --train-size 2000 \
  --test-size 200
```

### 체크포인트 관리

#### 체크포인트 목록 확인
```bash
python main.py checkpoints list <데이터셋명>
python main.py checkpoints list all  # 모든 데이터셋
```

#### 체크포인트 삭제
```bash
python main.py checkpoints remove <데이터셋명>
```

### 유틸리티

#### ONNX 변환
```bash
python main.py misc torch2onnx <데이터셋명> <체크포인트명> [--output captcha.onnx]
```

## 🏗️ 프로젝트 구조

```
kaptch/
├── main.py                 # 메인 CLI 진입점
├── src/
│   ├── datasets/          # 데이터셋 관리
│   │   ├── __init__.py   # 데이터셋 CLI
│   │   ├── label.py      # 웹 라벨링 서버
│   │   └── pom.py        # 페이지 오브젝트 모델
│   ├── train/            # 모델 훈련
│   │   ├── __init__.py   # 훈련 CLI
│   │   ├── model.py      # CRNN 모델 정의
│   │   ├── dataset.py    # 데이터셋 클래스
│   │   └── checkpoint.py # 체크포인트 관리
│   ├── checkpoints.py    # 체크포인트 CLI
│   ├── misc.py          # 기타 유틸리티
│   └── constants.py     # 상수 정의
├── dataset/              # 데이터셋 저장소
├── checkpoints/          # 체크포인트 저장소
└── captcha.onnx         # 변환된 ONNX 모델
```

## 🔧 기술 스택

- **Python 3.12+**: 메인 프로그래밍 언어
- **PyTorch**: 딥러닝 프레임워크
- **Typer**: CLI 프레임워크
- **Rich**: 터미널 출력 라이브러리
- **Playwright**: 웹 자동화
- **Flask**: 웹 라벨링 서버
- **OpenCV**: 이미지 처리
- **ONNX**: 모델 변환 및 추론

## 📊 모델 아키텍처

- **CNN**: 이미지 특징 추출 (Depthwise Separable Convolution 사용)
- **RNN**: 시퀀스 모델링 (Bidirectional GRU)
- **CTC Loss**: 가변 길이 출력 처리
- **Early Stopping**: 과적합 방지
- **Learning Rate Scheduling**: 최적화 성능 향상

## ⚠️ 주의사항

1. **법적 고려사항**: 웹 크롤링 시 해당 웹사이트의 이용약관을 준수하세요.
2. **데이터 보안**: 수집된 데이터의 안전한 관리가 필요합니다.
3. **리소스 관리**: GPU 메모리 사용량을 모니터링하세요.
4. **백업**: 중요한 체크포인트는 정기적으로 백업하세요.

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 상업적 사용 시 관련 법규를 준수하시기 바랍니다.

## 📞 문의

프로젝트에 대한 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해 주세요.
