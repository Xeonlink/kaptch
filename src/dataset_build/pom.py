from playwright.sync_api import BrowserContext, Page, Locator
from typing import Literal, Type
import time

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

    def _get_image_base64(self) -> str:
        """이미지 엘리먼트의 base64 데이터를 반환합니다."""
        img_elem = self.image_locator.element_handle()
        if img_elem is None:
            raise ValueError("이미지 엘리먼트를 찾을 수 없습니다.")
        return self.page.evaluate(
            """img => {
                const c = document.createElement('canvas');
                c.width = img.naturalWidth;
                c.height = img.naturalHeight;
                c.getContext('2d').drawImage(img, 0, 0);
                return c.toDataURL();
            }""",
            img_elem
        )


    def save_captcha(self, path: str):
        if self.reload_btn_locator is None:
            raise ValueError("reload_btn_locator is None")
        if self.image_locator is None:
            raise ValueError("image_locator is None")
        
        self.page.wait_for_load_state("networkidle")
        old_b64 = self._get_image_base64()
        self.reload_btn_locator.click()

        # base64 데이터가 바뀔 때까지 polling
        timeout = 5  # seconds
        start = time.time()
        while True:
            new_b64 = self._get_image_base64()
            if new_b64 != old_b64:
                break
            if time.time() - start > timeout:
                raise TimeoutError("Image base64 did not change after reload")
            time.sleep(0.05)

        # 새 이미지가 완전히 로드될 때까지 기다림
        self.page.wait_for_function(
            "(img) => img.complete && img.naturalWidth > 0",
            arg=self.image_locator.element_handle(),
            timeout=1000 * 5
        )
        self.image_locator.screenshot(path=path, type="png")


class NHN_KCP_Page(Pom):
    gate_url = "https://user.cafe24.com/join/hosting/general/?page=step1&landTime=1751035289"

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


class NICE평가정보_Page(Pom):
    gate_url = "https://www.seoul.go.kr/member/userlogin/loginCheck.do"

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


class SCI평가정보_Page(Pom):
    gate_url = "https://www.lotteimall.com/member/regist/forward.MemberRegist.lotte"

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


class KMCERT_Page(Pom):
    gate_url = "https://state.gwd.go.kr/portal/minwon/epeople/counsel"

    def prepare(self):
        with self.context.expect_page() as page_info:
            self.page.frame_locator("iframe[title='민원상담신청']").locator("a.be_06").click()
            self.page = page_info.value

        # self.page.locator()

        self.image_locator = self.page.locator("#simpleCaptchaImg")
        self.reload_btn_locator = self.page.locator("a[title='새로고침']")


class PomFactory:
    type Authcom = Literal["nhn_kcp", "nice", "sci", "kmcert"]
    authcom_list: list[Authcom] = ["nhn_kcp", "nice", "sci", "kmcert"]

    @staticmethod
    def create(authcom: Authcom, context: BrowserContext, page: Page) -> Type[Pom]:
        if authcom == "nhn_kcp":
            return NHN_KCP_Page(context, page)
        elif authcom == "nice":
            return NICE평가정보_Page(context, page)
        elif authcom == "sci":
            return SCI평가정보_Page(context, page)
        elif authcom == "kmcert":
            return KMCERT_Page(context, page)

    @staticmethod
    def authcom_type(authcom: str) -> str:
        if authcom in PomFactory.authcom_list:
            return authcom
        else:
            raise ValueError(f"Unsupported authcom: {authcom}")
