from collections.abc import Awaitable
import numpy as np
import aiofiles
import asyncio
import aiohttp
import time
import cv2
import csv
import os


def __get_image_url() -> str:
    """
    캡차 이미지를 가져올 수 있는 URL을 반환합니다.

    Returns:
        str: 이미지 URL
    """
    return "https://www.kmcert.com/kmcis/comm_v2/captcha/kmcisCaptchaImage.jsp?id=51909422-fd45-40f0-a0fa-38aab3740261"


async def __download_image(session: aiohttp.ClientSession, url: str) -> np.ndarray:
    """
    주어진 URL에서 이미지를 비동기로 다운로드하여 np.ndarray로 반환합니다.

    Parameters:
        session (aiohttp.ClientSession): aiohttp 세션
        url (str): 이미지 URL
    Returns:
        np.ndarray: 디코딩된 이미지 배열
    """
    async with session.get(url) as resp:
        data: bytes = await resp.read()
        image: np.ndarray = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        return image


async def __save_image(image: np.ndarray, path: str) -> None:
    """
    이미지를 PNG로 인코딩하여 지정된 경로에 비동기로 저장합니다.

    Parameters:
        image (np.ndarray): 저장할 이미지 배열
        path (str): 저장 경로
    Returns:
        None
    Raises:
        RuntimeError: 인코딩 실패 시 발생
    """
    success: bool
    encoded: np.ndarray
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image")
    async with aiofiles.open(path, "wb") as f:
        await f.write(encoded.tobytes())


async def __download_and_save(session: aiohttp.ClientSession, url: str, path: str) -> None:
    """
    이미지를 비동기로 다운로드한 뒤, 지정된 경로에 비동기로 저장합니다.

    Parameters:
        session (aiohttp.ClientSession): aiohttp 세션
        url (str): 이미지 URL
        path (str): 저장 경로
    Returns:
        None
    """
    image: np.ndarray = await __download_image(session, url)
    await __save_image(image, path)


async def __build_dataset(train_image_count: int, test_image_count: int) -> None:
    """
    지정된 개수만큼 학습/테스트 이미지를 비동기로 다운로드 및 저장하고, 리스트 CSV를 생성합니다.

    Parameters:
        train_image_count (int): 학습 이미지 개수
        test_image_count (int): 테스트 이미지 개수
    Returns:
        None
    """
    current_dir: str = os.path.dirname(__file__)
    dataset_dir: str = os.path.join(current_dir, "dataset")
    train_dir: str = os.path.join(dataset_dir, "train")
    test_dir: str = os.path.join(dataset_dir, "test")

    image_url: str = __get_image_url()
    os.makedirs(dataset_dir, exist_ok=True)

    train_list_path: str = os.path.join(dataset_dir, "train_list.csv")
    train_rows: list[list[str]] = []
    train_tasks: list[Awaitable[None]] = []
    async with aiohttp.ClientSession() as session:
        for count in range(train_image_count):
            subfolder: str = f"{count // 1000:04d}"
            subfolder_path: str = os.path.join(train_dir, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            filename: str = f"{count % 1000:04d}.png"
            rel_path: str = f"train/{subfolder}/{filename}"
            abs_path: str = os.path.join(subfolder_path, filename)
            train_rows.append([rel_path, ""])
            train_tasks.append(__download_and_save(session, image_url, abs_path))
        await asyncio.gather(*train_tasks)
    with open(train_list_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

    test_list_path: str = os.path.join(dataset_dir, "test_list.csv")
    test_rows: list[list[str]] = []
    test_tasks: list[Awaitable[None]] = []
    async with aiohttp.ClientSession() as session:
        for count in range(test_image_count):
            subfolder: str = f"{count // 1000:04d}"
            subfolder_path: str = os.path.join(test_dir, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            filename: str = f"{count % 1000:04d}.png"
            rel_path: str = f"test/{subfolder}/{filename}"
            abs_path: str = os.path.join(subfolder_path, filename)
            test_rows.append([rel_path, ""])
            test_tasks.append(__download_and_save(session, image_url, abs_path))
        await asyncio.gather(*test_tasks)
    with open(test_list_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(test_rows)


def main() -> None:
    """
    전체 데이터셋 생성 과정을 실행하고, 소요 시간을 출력합니다.

    Returns:
        None
    """
    start_time: float = time.time()
    asyncio.run(
        __build_dataset(
            train_image_count=10_000,
            test_image_count=1_000,
        )
    )
    end_time: float = time.time()
    print("Done")
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
