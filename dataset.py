""" realtime image process """
import os
import glob
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# --- 원본 utils.py의 커널 생성 함수들을 NumPy 기반으로 구현 ---
def cubic(x):
    """Matlab의 bicubic interpolation에 사용되는 cubic 함수를 구현합니다."""
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + \
        np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2))
    return f

def get_bicubic_kernel(size=20, factor=4):
    """
    Matlab의 imresize bicubic 커널과 동일한 4x4 커널을 생성
    논문의 설명대로, 15x15 가우시안 커널과 합성곱하기 위해 20x20으로 제로패딩
    """
    # 20x20 크기의 0으로 채워진 배열을 만듭니다.
    k = np.zeros((size, size))

    # 1D bicubic 커널 값을 계산합니다.
    bicubic_k_1d = cubic(np.arange(-2, 2) / factor + 1e-8)
    # 합이 1이 되도록 정규화합니다.
    bicubic_k_1d = bicubic_k_1d / np.sum(bicubic_k_1d)
    # 외적(outer product)을 통해 1D 커널 두 개를 곱하여 2D 4x4 커널을 만듭니다.
    bicubic_k_2d = np.outer(bicubic_k_1d, bicubic_k_1d.T)

    # 20x20 배열의 중앙에 4x4 커널을 배치합니다.
    pad = (size - 4) // 2
    k[pad:pad + 4, pad:pad + 4] = bicubic_k_2d
    return k

def random_anisotropic_gaussian_kernel(size=15, sig_min=0.2, sig_max=4.0):
    """랜덤한 회전 각도와 표준편차를 가진 15x15 비등방성 가우시안 커널을 생성합니다."""
    # 랜덤한 회전 각도 (0 ~ 180도)
    theta = np.random.uniform(0, np.pi)
    # x, y 방향으로 랜덤한 표준편차
    sig_x = np.random.uniform(sig_min, sig_max)
    sig_y = np.random.uniform(sig_min, sig_max)

    # 랜덤 값을 기반으로 공분산 행렬을 계산하여 가우시안의 모양과 방향을 결정합니다.
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    covariance_matrix = rotation_matrix @ np.diag([sig_x ** 2, sig_y ** 2]) @ rotation_matrix.T
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)

    # 2D 가우시안 함수를 이용해 커널을 생성합니다.
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.stack([xx, yy], axis=2)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inv_covariance_matrix) * xy, 2))

    # 합이 1이 되도록 정규화합니다.
    return kernel / np.sum(kernel)

class ToNumpy:
    def __call__(self, pil_img):
        return np.array(pil_img)

# --- 논문의 Degradation을 수행하는 커스텀 Transform 클래스 ---
# PyTorch의 transform 파이프라인에 포함시키기 위해 클래스로 구현합니다.
class GenerateLR:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.kernel_size = 20
        self.pad_size = (self.kernel_size - self.scale_factor) // 2

    # __call__ 메소드를 구현하면 이 클래스의 객체를 함수처럼 호출할 수 있습니다. (예: transform(image))
    def __call__(self, hr_img_np):
        # 1. 20x20 Bicubic 커널과 15x15 Gaussian 커널을 랜덤하게 생성
        bicubic_k = get_bicubic_kernel(factor=self.scale_factor)
        gaussian_k = random_anisotropic_gaussian_kernel()
        # 2. 두 커널을 합성곱하여 20x20 크기의 최종 다운샘플링 커널(k_d)을 생성
        downsampling_k = convolve2d(bicubic_k, gaussian_k, mode='same')
        # 3. HR 이미지의 각 채널(R, G, B)에 다운샘플링 커널을 적용
        lr_channels = []
        for i in range(hr_img_np.shape[2]):  # R, G, B 채널 순회
            # 컨볼루션 전에 대칭 패딩 추가
            # np.pad를 사용해 상하(pad_size), 좌우(pad_size)에 대칭(symmetric) 패딩을 추가
            hr_channel = hr_img_np[:, :, i]
            hr_channel_padded = np.pad(hr_channel, self.pad_size, 'symmetric')
            # HR 이미지와 커널을 합성곱
            lr_channel = convolve2d(hr_channel_padded, downsampling_k, mode='valid')
            # scale_factor 만큼의 간격으로 픽셀을 샘플링하여 다운샘플링 수행
            lr_channel = lr_channel[::self.scale_factor, ::self.scale_factor]
            lr_channels.append(lr_channel)
        # 4. 각 채널 결과를 합쳐서 최종 LR 이미지를 생성
        lr_img_np = np.stack(lr_channels, axis=2)

        return lr_img_np

# --- 최종 데이터셋 클래스 ---
class SISRDataset(Dataset):
    def __init__(self, hr_dir, scale_factor, patch_size_lr):
        super().__init__()
        self.hr_image_paths = sorted(glob.glob(os.path.join(hr_dir, '*')))
        self.scale_factor = scale_factor
        self.patch_size_hr = patch_size_lr * scale_factor

        # HR 이미지 전처리: 랜덤 크롭 후 NumPy 배열로 변환
        self.hr_base_transform = transforms.Compose([
            transforms.RandomCrop(self.patch_size_hr),
            ToNumpy()
        ])

        # LR 이미지 생성기: 위에서 정의한 커스텀 Transform 클래스
        self.lr_generator = GenerateLR(self.scale_factor)

        # 텐서 변환 및 정규화: NumPy 배열을 텐서로 바꾸고 [-1, 1] 범위로 정규화
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # ❗️ Sampler가 준 index는 무시하고, 매번 전체 800개 이미지 중 하나를 랜덤으로 선택
        random_hr_path = random.choice(self.hr_image_paths)

        # 1. HR 이미지를 PIL 형태로 불러옵니다.
        hr_image_pil = Image.open(self.hr_image_paths[index]).convert('RGB')

        # 2. HR 이미지에서 HR 패치를 랜덤하게 잘라내고 NumPy 배열로 변환합니다.
        hr_patch_np = self.hr_base_transform(hr_image_pil)

        # 3. HR 패치(NumPy)를 입력으로 하여 LR 패치(NumPy)를 생성합니다.
        lr_patch_np = self.lr_generator(hr_patch_np)

        # 텐서로 변환하기 전, NumPy 배열 타입을 float32로 명시합니다.
        # error 해결: TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
        hr_patch_np = hr_patch_np.astype(np.float32) / 255.0
        lr_patch_np = lr_patch_np.astype(np.float32) / 255.0

        # 4. 두 NumPy 배열을 각각 PyTorch 텐서로 변환하고 [-1, 1]로 정규화합니다.
        hr_patch = self.tensor_transform(hr_patch_np)
        lr_patch = self.tensor_transform(lr_patch_np)

        return lr_patch, hr_patch

    def __len__(self):
        return len(self.hr_image_paths)


# """ using preprocessed patch """
# import os
# import glob
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
#
# # 사전 생성된 (LR, HR) 패치 이미지 쌍을 읽어오는 데이터셋
# class SISRDataset(Dataset):
#
#     def __init__(self, lr_dir, hr_dir):
#         super().__init__()
#         # LR 패치 파일 경로를 기준으로 파일 목록을 가져옴
#         self.lr_image_paths = sorted(glob.glob(os.path.join(lr_dir, '*')))
#         self.hr_dir = hr_dir
#
#         # 정규화
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#
#     def __getitem__(self, index):
#         # 1. index에 해당하는 LR 패치 경로를 가져옵니다.
#         lr_path = self.lr_image_paths[index]
#
#         # 2. LR 파일 이름으로 HR 파일 경로를 유추합니다.
#         # 예: lr/000001.png -> hr/000001.png
#         filename = os.path.basename(lr_path)
#         hr_path = os.path.join(self.hr_dir, filename)
#
#         # 3. LR, HR 패치 이미지를 불러옵니다.
#         lr_img = Image.open(lr_path).convert('RGB')
#         hr_img = Image.open(hr_path).convert('RGB')
#
#         # 4. 두 이미지를 텐서로 변환하고 [-1, 1]로 정규화합니다.
#         lr_tensor = self.transform(lr_img)
#         hr_tensor = self.transform(hr_img)
#
#         return lr_tensor, hr_tensor
#
#     def __len__(self):
#         return len(self.lr_image_paths)