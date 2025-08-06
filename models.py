import torch
import torch.nn as nn
import torch.nn.functional as F

def kernel_normalize(kernel, kernel_size):
    """
    커널의 합을 1로 정규화
    kernel shape: [B, k*k, s*s, H, W]
    """
    # k*k 차원(dim=1)을 기준으로 평균을 계산합니다.
    kernel_mean = torch.mean(kernel, dim=1, keepdim=True)
    kernel = kernel - kernel_mean
    kernel = kernel + 1.0 / (kernel_size * kernel_size)
    return kernel

""" LR에 F_u로 local filtering -> HR prediction 만들기 """
def local_conv_us(img, kernel, scale, kernel_size):
    """
    PyTorch 버전의 local_conv_us (커널 정규화 포함)
    img: [B, C, H, W]       (예: [B, 3, 64, 64])
    kernel: [B, k*k*s*s, H, W] (예: [B, 400, 64, 64])
    scale: s                (예: 4)
    kernel_size: k          (예: 5)
    """
    batch_size, channels, height, width = img.shape

    """ 이미지 준비 """
    """ 이미지는 출력 정보를 가지지 않아도 됨. 따라서 커널과 달리 s*s 를 가지는 축이 없음. """
    # 1. 패치 추출 (Sliding window), 결과: [B, C*k*k, H*W] (예: [B, 75, 4096])
    img_patches = F.unfold(img, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    # 2. 패치 재구성 (정보 분리), 결과: [B, C, k*k, H, W] (예: [B, 3, 25, 64, 64])
    img_patches = img_patches.view(batch_size, channels, kernel_size * kernel_size, height, width)

    """ 커널 준비 """
    # 1. 커널 재구성 (정보 분리), 결과: [B, k*k, s*s, H, W] (예: [B, 25, 16, 64, 64])
    kernel = kernel.view(batch_size, kernel_size * kernel_size, scale * scale, height, width)

    # 2. 커널 정규화
    kernel = kernel_normalize(kernel, kernel_size)

    # 3. 채널에 맞게 커널 복제
    # [B, k*k, s*s, H, W] -> unsqueeze(1) -> [B, 1, k*k, s*s, H, W] -> repeat -> [B, C, k*k, s*s, H, W] (예: [B, 3, 25, 16, 64, 64])
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

""" HR에 F_d로 local filtering -> LR prediction 만들기"""
def local_conv_ds(img, kernel, scale, kernel_size):
    batch_size, channels, height, width = img.shape

    # local filtering 이므로 extract_patches와 동일하게 unfold 사용. 단 stride를 scale만큼 주어 다운샘플링
    img_patches = F.unfold(img, kernel_size=kernel_size, padding=(kernel_size - scale) // 2, stride=scale)
    img_patches = img_patches.view(batch_size, channels, kernel_size * kernel_size, height // scale, width // scale) # [B, C, k*k, H, W]

    # 커널은 각 위치마다 하나씩 있으므로 공간 차원을 추가
    # [B, k*k, H_lr, W_lr] -> [B, 1, k*k, H_lr, W_lr]
    kernel = kernel.view(batch_size, 1, kernel_size * kernel_size, height // scale, width // scale)

    # 이미지 패치와 커널을 곱하고, 패치 차원(dim=2)으로 sum
    result = torch.sum(img_patches * kernel, dim=2)

    return result

""" koala module 내부에서 parameter k 생성할 때 로컬 필터링 """
def local_conv_feat(img, kernel, kernel_size):
    batch_size, channels, height, width = img.shape

    # stride=1, padding='SAME'으로 패치 추출
    img_patches = F.unfold(img, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
    img_patches = img_patches.view(batch_size, channels, kernel_size * kernel_size, height, width)

    # 커널은 채널별로 동일하게 적용되므로, 채널 축을 추가하여 브로드캐스팅 준비
    # [B, k*k, H, W] -> [B, 1, k*k, H, W]
    kernel = kernel.view(batch_size, 1, kernel_size * kernel_size, height, width)

    # 이미지 패치와 커널을 곱하고, 패치 차원(dim=2)으로 합산
    result = torch.sum(img_patches * kernel, dim=2)
    return result

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

class EncoderBlock(nn.Module):
    """U-Net의 인코더 한 단계를 구성하는 블록 (Conv -> Res -> Res -> Relu -> Pool)"""
    def __init__(self, in_channels, out_channels): # in_channels = 3, out_channels = 64
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_block1 = ResBlock(out_channels)
        self.res_block2 = ResBlock(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.relu(x)
        skip_connection = x  # 디코더로 전달할 연결
        pooled_x = self.pool(x)
        return pooled_x, skip_connection

class DecoderBlock(nn.Module):
    """U-Net의 디코더 한 단계를 구성하는 블록 (Deconv -> Concat -> Conv -> Res -> Res -> Relu)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose2d는 Deconvolution이라고도 불리며, 이미지 크기를 키웁니다.
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Concat 이후 채널 수가 2배가 되므로, Conv의 입력 채널도 2배가 됩니다.
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.res_block1 = ResBlock(out_channels)
        self.res_block2 = ResBlock(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, skip_connection): # 입력 두 개 받음 (skip 포함)
        x = self.upconv(x)
        # skip_connection을 채널(dim=1) 방향으로 합칩니다.
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.relu(x)
        return x

""" KOALA moudle """
class KOALAModule(nn.Module):
    def __init__(self, feat_channels=64, kernel_feat_channels=64, local_kernel_size=7):
        super().__init__()

        # 1. 입력 특징(x)을 처리하는 ResBlock 스타일의 경로
        self.feature_path = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        )

        # 2. 커널 특징(f_d)으로부터 곱셈 파라미터(m)를 생성하는 경로
        self.multiplicative_path = nn.Sequential(
            nn.Conv2d(kernel_feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        )

        # 3. 커널 특징(f_d)으로부터 로컬 필터(k)를 생성하는 경로
        self.local_filter_path = nn.Sequential(
            # 1x1 Conv를 사용하여 공간적 정보를 섞지 않음
            nn.Conv2d(kernel_feat_channels, feat_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_channels, local_kernel_size ** 2, kernel_size=1)
            # normalize는? forward에서 수행
        )

        self.local_kernel_size = local_kernel_size

    def forward(self, x, kernel_features):
        # 입력 특징(x)과 커널 특징(f_d)을 각각의 경로로 전달
        feature_processed = self.feature_path(x)
        multiplicative_params = self.multiplicative_path(kernel_features)
        local_filter_params = self.local_filter_path(kernel_features)
        local_filter_params = kernel_normalize(local_filter_params, self.local_kernel_size) # 커널 정규화

        # 1. 곱셈(element-wise multiplication)으로 특징 조정
        feature_multiplied = feature_processed * multiplicative_params

        # 2. 로컬 필터링으로 특징 조정
        feature_filtered = local_conv_feat(feature_multiplied, local_filter_params, self.local_kernel_size)

        # 3. 잔차 연결 (Residual Connection)
        return x + feature_filtered

""" Training Stage 1 """
class DownsamplingNetwork(nn.Module):
    """
    LR 이미지로부터 열화 커널을 예측하는 U-Net 기반 네트워크.
    """

    def __init__(self, in_channels=3, base_channels=64, kernel_size=20):
        super().__init__()

        # 인코더 부분
        self.enc1 = EncoderBlock(in_channels, base_channels)  # 64 channels
        self.enc2 = EncoderBlock(base_channels, base_channels * 2)  # 128 channels

        # 병목(Bottleneck) 부분: 5 conv layers
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1), # 첫 번째 conv는 in_channel = 128
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1), # 나머지는 256
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),  # 나머지는 256
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),  # 나머지는 256
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),  # 나머지는 256
            nn.ReLU()
        )

        # 디코더 부분
        self.dec1 = DecoderBlock(base_channels * 4, base_channels * 2)  # 128 channels
        self.dec2 = DecoderBlock(base_channels * 2, base_channels)  # 64 channels

        # 최종 커널 예측을 위한 헤드(Head)
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            # 최종 출력 채널은 k*k (20*20=400)
            nn.Conv2d(base_channels, kernel_size * kernel_size, kernel_size=3, padding=1)

            # 마지막 정규화는 loss 계산할 때만 쓰고, 이미지 만들 때는 안씀...
        )

    def forward(self, x):
        # 인코더 경로
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)

        # 병목
        x = self.bottleneck(x)

        # 디코더 경로 (skip connection 사용)
        x = self.dec1(x, skip2)
        x = self.dec2(x, skip1)

        # 최종 커널 맵 출력
        kernel_map = self.head(x)
        return kernel_map

""" Training Stage 2 """
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
        if scale_factor == 4:
            final_in_channels = 16
        else:
            final_in_channels = 32
        residual_layers.append(nn.Conv2d(final_in_channels, channels, kernel_size=3, padding=1)) # 3 채널
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

""" Training Stage 3 """
class UpsamplingNetwork(nn.Module):
    def __init__(self, scale_factor=4, channels=3, num_res_blocks=12):
        super().__init__()
        self.scale_factor = scale_factor
        self.up_kernel_size = 5
        ch = 64

        # 1. Downsampling Network의 커널 특징을 처리하는 부분
        self.kernel_feature_extractor = nn.Sequential(
            nn.Conv2d(20 ** 2, ch, kernel_size=3, padding=1), # DOWNSAMPLING_KERNEL_SIZE = 20
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 2. LR 이미지의 초기 특징 추출
        self.initial_conv = nn.Conv2d(channels, ch, kernel_size=3, padding=1)

        # 3. 5개의 KOALA 모듈과 7개의 ResBlock
        self.koala_blocks = nn.ModuleList([KOALAModule() for _ in range(5)])
        self.res_blocks = nn.ModuleList([ResBlock(ch) for _ in range(7)])

        # 4. 브랜치가 나뉘기 전 마지막 ReLU
        self.final_relu = nn.ReLU()

        # 5. 잔차 이미지 브랜치 (UpsamplingBaselineNetwork와 동일)
        final_in_channels = 16 if scale_factor == 4 else 32
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=final_in_channels, out_channels=channels, kernel_size=3, padding=1)
        )

        # 6. 업샘플링 커널 브랜치 (UpsamplingBaselineNetwork와 동일)
        output_channels = self.up_kernel_size * self.up_kernel_size * scale_factor * scale_factor
        self.kernel_branch = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch * 2, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, lr_img, downsampling_kernel_map):
        # Downsampling Network가 예측한 커널 맵에서 특징(f_d) 추출
        kernel_features = self.kernel_feature_extractor(downsampling_kernel_map)

        # LR 이미지에서 초기 특징(x) 추출
        net = self.initial_conv(lr_img)

        # 5개의 KOALA 모듈 통과
        for i in range(5):
            net = self.koala_blocks[i](net, kernel_features)

        # 7개의 ResBlock 통과
        for i in range(7):
            net = self.res_blocks[i](net)

        shared_features = self.final_relu(net)

        # 이후는 UpsamplingBaselineNetwork와 동일
        residual_img = self.residual_branch(shared_features)
        upsampling_kernel = self.kernel_branch(shared_features)
        upsampled_img = local_conv_us(lr_img, upsampling_kernel, self.scale_factor, self.up_kernel_size)
        sr_image = upsampled_img + residual_img

        return sr_image