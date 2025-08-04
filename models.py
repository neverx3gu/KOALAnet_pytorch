import torch
import torch.nn as nn
import torch.nn.functional as F

def kernel_normalize(kernel, kernel_size):
    """
    커널의 합을 1로 정규화합니다.
    kernel shape: [B, k*k, s*s, H, W]
    """
    # k*k 차원(dim=1)을 기준으로 평균을 계산합니다.
    kernel_mean = torch.mean(kernel, dim=1, keepdim=True)
    kernel = kernel - kernel_mean
    kernel = kernel + 1.0 / (kernel_size * kernel_size)
    return kernel

def local_conv_us(img, kernel, scale, kernel_size):
    """
    PyTorch 버전의 local_conv_us (커널 정규화 포함).
    img: [B, C, H, W]       (예: [B, 3, 64, 64])
    kernel: [B, k*k*s*s, H, W] (예: [B, 400, 64, 64])
    scale: s                (예: 4)
    kernel_size: k          (예: 5)
    """
    batch_size, channels, height, width = img.shape

    """ 이미지 준비 """
    """ 이미지는 출력 정보를 가지지 않아도 됨. 따라서 커널과 달리 s*s 를 가지는 축이 없음. """
    # 1. 패치 추출 (Sliding window)
    # 결과: [B, C*k*k, H*W] (예: [B, 75, 4096])
    img_patches = F.unfold(img, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    # 2. 패치 재구성 (정보 분리)
    # 결과: [B, C, k*k, H, W] (예: [B, 3, 25, 64, 64])
    img_patches = img_patches.view(batch_size, channels, kernel_size * kernel_size, height, width)

    """ 커널 준비 """
    # 1. 커널 재구성 (정보 분리)
    # 결과: [B, k*k, s*s, H, W] (예: [B, 25, 16, 64, 64])
    kernel = kernel.view(batch_size, kernel_size * kernel_size, scale * scale, height, width)

    # 2. 커널 정규화 (형태는 변하지 않음)
    kernel = kernel_normalize(kernel, kernel_size)

    # 3. 채널에 맞게 커널 복제
    # [B, k*k, s*s, H, W] -> unsqueeze(1) -> [B, 1, k*k, s*s, H, W]
    # -> repeat -> [B, C, k*k, s*s, H, W] (예: [B, 3, 25, 16, 64, 64])
    kernel = kernel.unsqueeze(1).repeat(1, channels, 1, 1, 1, 1)

    """ 로컬 필터링 (핵심 연산) """
    # 곱셈을 위해 img_patches의 차원을 임시로 확장합니다: [B, C, k*k, 1, H, W]
    ## 여기서, TF 코드의 경우 명시적으로 완전히 같은 형태로(16으로) 맞춰주지만 파이토치는 그렇지 않음.
    ## 파이토치에서는 "브로드캐스팅"이라는 암시적 변환을 통해 두 텐서의 모양이 정확하지 않아도 알아서 16으로 복제해준다고 함.
    ## 메모리에 직접 저장하는 게 아니라, 연산할 때만 임시로 늘리는 거라 메모리 효율도 좋음!
    # 곱셈 결과 텐서의 형태: [B, C, k*k, s*s, H, W]
    # dim=2 (k*k 축) 기준으로 합산하여 차원을 축소합니다.
    # 결과: [B, C, s*s, H, W] (예: [B, 3, 16, 64, 64])
    output_patches = torch.sum(img_patches.unsqueeze(3) * kernel, dim=2)

    """ 업샘플링 (수동 PixelShuffle) """
    # 1. s*s 축을 맨 뒤로 보냅니다 (permute).
    # 결과: [B, C, H, W, s*s] (예: [B, 3, 64, 64, 16])
    output_patches = output_patches.permute(0, 1, 3, 4, 2).contiguous()

    # 2. s*s 축을 s, s 두 개의 축으로 나눕니다 (view).
    # 결과: [B, C, H, W, s, s] (예: [B, 3, 64, 64, 4, 4])
    output = output_patches.view(batch_size, channels, height, width, scale, scale)

    # 3. H 옆에 s, W 옆에 s를 위치시킵니다 (permute).
    # 결과: [B, C, H, s, W, s] (예: [B, 3, 64, 4, 64, 4])
    output = output.permute(0, 1, 2, 4, 3, 5).contiguous()

    # 4. H와 s, W와 s 축을 합쳐 최종 이미지 형태로 만듭니다 (view).
    # 결과: [B, C, H*s, W*s] (예: [B, 3, 256, 256])
    output = output.view(batch_size, channels, height * scale, width * scale)

    return output

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
        residual_layers.append(nn.Conv2d(16, channels, kernel_size=3, padding=1)) # 3 채널
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

        # 3. local filtering 실행 및 잔차 이미지 더하기
        # 주의: local_conv_us는 원본 LR 이미지(x)를 입력으로 받음
        upsampled_img = local_conv_us(x, upsampling_kernel, self.scale_factor, self.up_kernel_size)
        sr_image = upsampled_img + residual_img

        return sr_image