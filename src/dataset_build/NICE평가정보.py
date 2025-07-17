"""
NICE평가정보 데이터셋 생성 스크립트

데이터셋 구조:
- dataset/NICE평가정보/captcha_XXXX.png
"""

from playwright.sync_api import Playwright, sync_playwright, expect, Page, BrowserContext
import random
import string
import os

DATASET_DIR: str = os.path.join("dataset", "NICE평가정보")
MAX_DATA_COUNT: int = 1_000


def create_image_path() -> str:
    h = "".join(random.choices(string.ascii_letters + string.digits, k=4))
    return os.path.join(DATASET_DIR, f"captcha_{h}.png")


def main(p: Playwright) -> None:
    data_count = len(os.listdir(DATASET_DIR))
    if data_count == MAX_DATA_COUNT:
        print(f"dataset이 완성되었습니다.")
        return
    if data_count > MAX_DATA_COUNT:
        print(f"데이터 수가 최대 데이터 수를 초과했습니다. ({MAX_DATA_COUNT})")
        return

    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    page.goto("https://www.seoul.go.kr/member/userlogin/loginCheck.do")
    page.get_by_role("link", name="본인확인 로그인").click()
    with context.expect_page() as page_info:
        page.get_by_role("button", name="휴대폰").click()
        page = page_info.value

    page.locator("#telcomKT").click()
    page.locator("button[value='SMS']").click()
    page.locator("#mobileCertAgree").check()
    page.locator("#btnMobileCertStart").click()
    name_locator = page.locator("#userName")
    name_locator.fill("홍길동")
    name_locator.press("Enter")
    page.locator("#btnSubmit").click()
    num1_locator = page.locator("#myNum1")
    num1_locator.fill("821203")
    num1_locator.press("Enter")
    num2_locator = page.locator("#myNum2")
    num2_locator.fill("1")
    num2_locator.press("Enter")
    mobile_no_locator = page.locator("#mobileNo")
    mobile_no_locator.fill("01012341234")
    mobile_no_locator.press("Enter")

    image = page.locator("#simpleCaptchaImg")
    reload_btn = page.locator("#btnSimpleCaptchaReload")

    print(f"데이터 수: {data_count:,}/{MAX_DATA_COUNT:,}", end="", flush=True)
    while data_count < MAX_DATA_COUNT:
        reload_btn.click()
        image.screenshot(path=create_image_path())
        data_count += 1
        print(f"\r데이터 수: {data_count:,}/{MAX_DATA_COUNT:,}", end="", flush=True)

    print(f"\ndataset 구성을 완료하였습니다.")


if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)

    with sync_playwright() as p:
        main(p)
