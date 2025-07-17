"""
NHN_KCP 데이터셋 생성 스크립트

데이터셋 구조:
- dataset/NHN_KCP/captcha_XXXX.png
"""

from playwright.sync_api import Playwright, sync_playwright, expect, Page, BrowserContext
import random
import string
import os

DATASET_DIR: str = os.path.join("dataset", "NHN_KCP")
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

    page.goto("https://user.cafe24.com/join/hosting/general/?page=step1&landTime=1751035289")
    page.locator("#agreeAll").check(force=True)
    with context.expect_page() as page_info:
        page.get_by_role("link", name=" 휴대폰 인증").click()
        page = page_info.value

    page.locator("label[for='agency-kt']").click()
    page.locator("label[for='agree_all']").first.click()
    page.locator("#btnSms").click()

    image = page.locator("#CAPTCHA_CaptchaImage")
    reload_btn = page.locator("#CAPTCHA_ReloadLink")

    print(f"데이터 수: {data_count:,}/{MAX_DATA_COUNT:,}", end="", flush=True)
    while data_count < MAX_DATA_COUNT:
        reload_btn.click()
        image.screenshot(path=create_image_path(), type="png")
        data_count += 1
        print(f"\r데이터 수: {data_count:,}/{MAX_DATA_COUNT:,}", end="", flush=True)

    print(f"\ndataset 구성을 완료하였습니다.")


if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)

    with sync_playwright() as p:
        main(p)
