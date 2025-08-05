# python3 test.py --input_dir ./testset/DIV2K/LR/X4/imgs --hr_dir ./testset/DIV2K/HR --output_dir ./test_results
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import structural_similarity as ssim

# models.py에서 Stage 3에 필요한 모델 구조를 가져옵니다.
from models import DownsamplingNetwork, UpsamplingNetwork


def calculate_psnr(img1, img2):
    """PSNR을 계산하는 함수 (Y 채널 기준)"""
    img1_y = rgb2ycbcr(img1)[:, :, 0]
    img2_y = rgb2ycbcr(img2)[:, :, 0]
    mse = np.mean((img1_y.astype(np.float64) - img2_y.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """SSIM을 계산하는 함수 (Y 채널 기준)"""
    img1_y = rgb2ycbcr(img1)[:, :, 0]
    img2_y = rgb2ycbcr(img2)[:, :, 0]
    return ssim(img1_y, img2_y, data_range=255)


def main(args):
    # --- 1. 설정 및 모델 로딩 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    downsampling_net = DownsamplingNetwork(kernel_size=args.down_kernel_size).to(device)
    upsampling_net = UpsamplingNetwork(scale_factor=args.scale).to(device)

    print(f"가중치 파일 로딩 중: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    downsampling_net.load_state_dict(checkpoint['downsampling_net'])
    upsampling_net.load_state_dict(checkpoint['upsampling_net'])
    print("가중치 로딩 완료!")

    downsampling_net.eval()
    upsampling_net.eval()

    # --- 2. 전처리/후처리 정의 ---
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    postprocess = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.ToPILImage()
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    lr_image_paths = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    # --- 3. 추론 및 평가 루프 ---
    with torch.no_grad():
        for lr_path in tqdm(lr_image_paths, desc="테스트 진행률"):
            filename = os.path.basename(lr_path)

            # LR 읽기
            lr_img = Image.open(lr_path).convert('RGB')
            lr_tensor = preprocess(lr_img).unsqueeze(0).to(device)
            _, _, h, w = lr_tensor.shape

            # 3.1 4의 배수 맞춤 패딩
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            if pad_h or pad_w:
                # (left, right, top, bottom)
                lr_tensor = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='reflect')

            # 3.2 Down / Up 네트워크 추론
            pred_kernel_map = downsampling_net(lr_tensor)
            sr_tensor = upsampling_net(lr_tensor, pred_kernel_map)

            # 3.3 크롭 크기 결정
            if args.hr_dir:
                hr_path = os.path.join(args.hr_dir, filename)
                if os.path.exists(hr_path):
                    hr_img = Image.open(hr_path).convert('RGB')
                    hr_w, hr_h = hr_img.size  # PIL: (width, height)
                    crop_h, crop_w = hr_h, hr_w
                else:
                    crop_h, crop_w = h * args.scale, w * args.scale
            else:
                crop_h, crop_w = h * args.scale, w * args.scale

            # 3.4 SR 결과 크롭
            sr_tensor = sr_tensor[:, :, :crop_h, :crop_w]

            # 3.5 PIL 변환 및 저장
            sr_img = postprocess(sr_tensor.squeeze(0).cpu())
            sr_img.save(os.path.join(args.output_dir, filename))

            # 3.6 PSNR/SSIM 평가
            if args.hr_dir and os.path.exists(os.path.join(args.hr_dir, filename)):
                hr_np = np.array(hr_img)
                sr_np = np.array(sr_img)
                psnr_val = calculate_psnr(hr_np, sr_np)
                ssim_val = calculate_ssim(hr_np, sr_np)

                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1
                tqdm.write(f"{filename}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")

    # --- 4. 평균 지표 출력 ---
    if count > 0:
        print(f"\n평균 PSNR: {total_psnr / count:.2f} dB")
        print(f"평균 SSIM: {total_ssim / count:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KOALAnet Stage 3 테스트 스크립트')
    parser.add_argument('--weights', type=str, default='./koalanet_stage3_final.pth',
                        help='학습된 Stage 3 모델 가중치 (.pth)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='저해상도(LR) 이미지 폴더 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='결과(SR) 이미지 저장 폴더 경로')
    parser.add_argument('--scale', type=int, default=4,
                        help='업샘플 배율')
    parser.add_argument('--down_kernel_size', type=int, default=20,
                        help='Downsampling Network 커널 크기')
    parser.add_argument('--hr_dir', type=str, default=None,
                        help='(선택) 고해상도(HR) 이미지 폴더 경로 (평가용)')
    args = parser.parse_args()

    main(args)
