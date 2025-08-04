import torch
import torch.nn as nn
import torch.nn.functional as F

def kernel_normalize_pytorch(kernel, kernel_size):
    """
    커널의 합을 1로 정규화합니다.
    kernel shape: [B, k*k, s*s, H, W]
    """
    # k*k 차원(dim=1)을 기준으로 평균을 계산합니다.
    kernel_mean = torch.mean(kernel, dim=1, keepdim=True)
    kernel = kernel - kernel_mean
    kernel = kernel + 1.0 / (kernel_size * kernel_size)
    return kernel

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.res_block(x)


class UpsamplingBaselineNetwork(nn.Module):
    def __init__(self, scale_factor=4, channels=3, num_res_blocks=12):
        super().__init__()

        self.scale_factor = scale_factor
        self.up_kernel_size = 5  # 원본 up_kernel의 크기는 5
        ch = 64

        # 1. 공유되는 몸통 부분: conv -> res 12개 -> relu
        self.shared_body = nn.Sequential(
            nn.Conv2d(channels, ch, kernel_size=3, padding=1),
            *[ResBlock(ch) for _ in range(num_res_blocks)],
            nn.ReLU()
        )

        # 2. 잔차 이미지 브랜치 (rgb residual image branch)
        # 원본 코드: relu -> conv -> ps -> (relu -> conv -> ps) -> conv
        residual_layers = [
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1), # 128 채널
            nn.ReLU(),
            nn.PixelShuffle(2), # 32 채널
        ]
        if scale_factor == 4:
            residual_layers.extend([
                nn.Conv2d(ch // 2, ch, kernel_size=3, padding=1), # 64 채널
                nn.ReLU(),
                nn.PixelShuffle(2) # 16 채널
            ])
        residual_layers.append(nn.Conv2d(ch, channels, kernel_size=3, padding=1)) # 64 채널
        self.residual_branch = nn.Sequential(*residual_layers)

        # 3. 업샘플링 커널 브랜치 (upsampling kernel branch)
        self.kernel_branch = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch * 2, self.up_kernel_size * self.up_kernel_size * scale_factor * scale_factor, kernel_size=3, padding=1) # 5x5xsxs 채널
        )

    def forward(self, x):
        # 1. 공유 몸통 부분을 통과시켜 특징맵 생성
        shared_features = self.shared_body(x)

        # 2. 두 브랜치로 특징맵을 각각 전달
        residual_img = self.residual_branch(shared_features)
        upsampling_kernel = self.kernel_branch(shared_features)


        """ local filtering -> SR 생성 구현 필요 """
        # 3. local filtering 실행 및 잔차 이미지 더하기
        # 주의: local_conv_us는 원본 LR 이미지(x)를 입력으로 받음

        upsampled_img = local_conv_us(x, upsampling_kernel, self.scale_factor, self.up_kernel_size)
        sr_image = upsampled_img + residual_img

        return sr_image