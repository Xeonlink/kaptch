from playwright.sync_api import BrowserContext, Page, Locator
from typing import override
import time
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO


class Pom:
    context: BrowserContext
    page: Page
    gate_url: str

    image_locator: Locator | None = None
    reload_btn_locator: Locator | None = None

    def __init__(self, context: BrowserContext, page: Page):
        self.context = context
        self.page = page

    def goto(self):
        self.page.goto(self.gate_url)

    def prepare(self):
        pass

    def __get_image_base64(self) -> str:
        """이미지 엘리먼트의 base64 데이터를 반환합니다."""
        img_elem = self.image_locator.element_handle()
        if img_elem is None:
            raise ValueError("이미지 엘리먼트를 찾을 수 없습니다.")

        return self.page.evaluate(
            """image => {
                const canvas = document.createElement('canvas');
                canvas.width = image.naturalWidth;
                canvas.height = image.naturalHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(image, 0, 0);
                return canvas.toDataURL('image/png');
            }""",
            img_elem,
        )

    def __wait_for_b64_change(self, old_b64: str) -> str:
        timeout = 5  # seconds
        start = time.time()
        while True:
            new_b64 = self.__get_image_base64()
            time.sleep(0.05)
            if new_b64 != old_b64:
                return new_b64

            if time.time() - start > timeout:
                raise TimeoutError("Image base64 did not change after reload")

    def save_captcha(self, path: str):
        if self.reload_btn_locator is None:
            raise ValueError("reload_btn_locator is None")
        if self.image_locator is None:
            raise ValueError("image_locator is None")

        self.page.wait_for_load_state("networkidle")
        old_b64 = self.__get_image_base64()
        self.reload_btn_locator.click()

        new_b64_bytes = b""
        while len(new_b64_bytes) == 0:
            new_b64 = self.__wait_for_b64_change(old_b64)
            new_b64_bytes = base64.b64decode(new_b64.split(",")[1])

        image = Image.open(BytesIO(new_b64_bytes))
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            image = image.convert("RGB")

        image.save(path)


class nhnkcp(Pom):
    gate_url = "https://user.cafe24.com/join/hosting/general/?page=step1&landTime=1751035289"

    @override
    def prepare(self):
        self.page.locator("#agreeAll").check(force=True)
        with self.context.expect_page() as page_info:
            self.page.get_by_role("link", name=" 휴대폰 인증").click()
            self.page = page_info.value

        self.page.locator("label[for='agency-kt']").click()
        self.page.locator("label[for='agree_all']").first.click()
        self.page.locator("#btnSms").click()

        self.image_locator = self.page.locator("#CAPTCHA_CaptchaImage")
        self.reload_btn_locator = self.page.locator("#CAPTCHA_ReloadLink")


class nice(Pom):
    gate_url = "https://www.seoul.go.kr/member/userlogin/loginCheck.do"

    @override
    def prepare(self):
        self.page.get_by_role("link", name="본인확인 로그인").click()

        with self.context.expect_page() as page_info:
            self.page.get_by_role("button", name="휴대폰").click()
            self.page = page_info.value

        self.page.locator("#telcomKT").click()
        self.page.locator("button[value='SMS']").click()
        self.page.locator("#mobileCertAgree").check()
        self.page.locator("#btnMobileCertStart").click()
        name_locator = self.page.locator("#userName")
        name_locator.fill("홍길동")
        name_locator.press("Enter")
        self.page.locator("#btnSubmit").click()
        num1_locator = self.page.locator("#myNum1")
        num1_locator.fill("821203")
        num1_locator.press("Enter")
        num2_locator = self.page.locator("#myNum2")
        num2_locator.fill("1")
        num2_locator.press("Enter")
        mobile_no_locator = self.page.locator("#mobileNo")
        mobile_no_locator.fill("01012341234")
        mobile_no_locator.press("Enter")

        self.image_locator = self.page.locator("#simpleCaptchaImg")
        self.reload_btn_locator = self.page.locator("#btnSimpleCaptchaReload")


class sci(Pom):
    gate_url = "https://www.lotteimall.com/member/regist/forward.MemberRegist.lotte"

    @override
    def prepare(self):
        self.page.locator("#info_chk").click()
        with self.context.expect_page() as page_info:
            self.page.locator(".btn_box.phone").click()
            self.page = page_info.value

        self.page.locator("li.tel_item").first.click()
        self.page.locator("li.cert_item").nth(2).click()
        self.page.locator("#check_txt").check()
        self.page.get_by_role("button", name="다음").click()
        name_locator = self.page.get_by_placeholder("이름")
        name_locator.fill("홍길동")
        name_locator.press("Enter")
        self.page.locator(".btnUserName").click()
        num1_locator = self.page.get_by_placeholder("생년월일 6자리")
        num1_locator.fill("821203")
        num1_locator.press("Enter")
        num2_locator = self.page.locator(".myNum2")
        num2_locator.fill("1")
        num2_locator.press("Enter")
        mobile_no_locator = self.page.get_by_placeholder("휴대폰번호")
        mobile_no_locator.fill("01012341234")
        mobile_no_locator.press("Enter")

        self.image_locator = self.page.locator("#simpleCaptchaImg")
        self.reload_btn_locator = self.page.locator("a[title='새로고침']")


class kmcert(Pom):
    gate_url = "https://state.gwd.go.kr/portal/minwon/epeople/counsel"

    @override
    def prepare(self):
        with self.context.expect_page() as page_info:
            self.page.frame_locator("iframe[title='민원상담신청']").locator("a.be_03").click()
            self.page = page_info.value

        self.page.get_by_role("button", name="KT").first.click()
        self.page.get_by_role("button", name="문자(SMS) 인증").click()
        self.page.locator("#mobileCertAgree").check()
        self.page.locator("#btnCertAuthStart").click()
        name_locator = self.page.locator("#userName")
        name_locator.fill("홍길동")
        name_locator.press("Enter")
        self.page.locator(".btnUserName").click()
        num1_locator = self.page.locator("#myNum1")
        num1_locator.fill("821203")
        num1_locator.press("Enter")
        num2_locator = self.page.locator("#myNum2")
        num2_locator.fill("1")
        num2_locator.press("Enter")
        mobile_no_locator = self.page.locator("#mobileNo")
        mobile_no_locator.fill("01012341234")
        mobile_no_locator.press("Enter")

        self.image_locator = self.page.locator("#simpleCaptchaImg")
        self.reload_btn_locator = self.page.locator("#simpleCaptchaBtnReload")


class dream(Pom):
    gate_url = "https://www.makeshop.co.kr/newmakeshop/home/create_shop.html"

    @override
    def prepare(self):
        with self.context.expect_page() as page_info:
            self.page.get_by_role("button", name="본인인증").click()
            self.page = page_info.value

        self.page.locator("button[data-telco='kt']").first.click()
        self.page.locator("li[data-sign='pass'] > button").click()
        self.page.locator("#user_agree_checkbox").click()
        self.page.locator("button.btn_selsign").click()
        name_locator = self.page.locator("#userName_pass")
        name_locator.fill("홍길동")
        name_locator.press("Enter")
        self.page.locator("button.btnUserName_pass").click()
        mobile_no_locator = self.page.locator("#mobileNo_pass")
        mobile_no_locator.fill("01012341234")
        mobile_no_locator.press("Enter")

        self.image_locator = self.page.locator("#simpleCaptchaImg").filter(visible=True)
        self.reload_btn_locator = self.page.locator("#btnSimpleCaptchaReload")


class kgmobilians(Pom):
    gate_url = "https://accounts.yanolja.com/?service=yanolja"

    @override
    def prepare(self):
        self.page.get_by_role("button", name="이메일로 시작하기").click()
        self.page.get_by_role("button", name="이메일로 가입하기").click()
        self.page.get_by_role("button", name="전체 동의").click()

        with self.context.expect_page() as page_info:
            self.page.get_by_role("button", name="본인 인증하기").click()
            self.page = page_info.value

        self.page.evaluate(
            "(section) => { section.style.display = 'block'; }",
            self.page.locator("#smsStep1").element_handle(),
        )

        self.image_locator = self.page.get_by_alt_text("보안문자 숫자 6자리").filter(visible=True)
        self.reload_btn_locator = self.page.locator("input.reLoad").filter(visible=True)
