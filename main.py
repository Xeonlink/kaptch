from yaspin import yaspin

with yaspin(text="Module Loading..."):
    from pathlib import Path

    import pyfiglet
    import typer

    from src.checkpoints import app as checkpoints_app
    from src.constants import APP_NAME
    from src.datasets import app as datasets_app
    from src.misc import app as misc_app
    from src.train import app as train_app

DATASET_ROOT = Path("dataset")


def main():
    pyfiglet.print_figlet(APP_NAME)

    app = typer.Typer(name=APP_NAME, help="한국 모바일인증 캡챠 자동 인식 파이프라인", rich_markup_mode="rich")
    app.add_typer(datasets_app, name="datasets", help="데이터셋 관리 (생성, 크롤링, 라벨링)")
    app.add_typer(train_app, help="모델 훈련 및 평가")
    app.add_typer(checkpoints_app, name="checkpoints", help="체크포인트 관리 (목록, 삭제)")
    app.add_typer(misc_app, name="misc", help="기타 유틸리티 (ONNX 변환, 모델 검증)")
    app()


if __name__ == "__main__":
    main()
