# python3 test.py --weights ./koalanet_baseline_x4_final.pth --input_dir ./test_images/lr --output_dir ./test_results --hr_dir ./test_images/hr

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse
import numpy as np
from skimage.color import rgb2ycbcr

# models.py에서 우리가 만든 모델 구조를 가져옵니다.
from models import UpsamplingBaselineNetwork


def calculate_psnr(img1, img2):
    """PSNR을 계산하는 함수 (Y 채널 기준)"""
    # img1, img2: [0, 255] 범위의 NumPy 배열

    # YCbCr 변환 후 Y 채널(밝기)만 사용
    img1_y = rgb2ycbcr(img1)[:, :, 0]
    img2_y = rgb2ycbcr(img2)[:, :, 0]

    mse = np.mean((img1_y - img2_y) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def main(args):
    # --- 1. 설정 및 모델 로딩 ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    # 모델 구조를 인스턴스화하고 장치로 보냅니다.
    model = UpsamplingBaselineNetwork(scale_factor=args.scale).to(device)

    # 저장된 가중치(state_dict)를 불러와 모델에 적용합니다.
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # 모델을 추론(evaluation) 모드로 설정합니다. (Dropout, BatchNorm 등의 동작을 비활성화)
    model.eval()

    # --- 2. 이미지 처리 ---
    # LR 이미지를 텐서로 변환하고 [-1, 1]로 정규화하는 전처리
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # SR 이미지를 저장 가능한 PIL 이미지로 변환하는 후처리
    postprocess = transforms.Compose([
        # [-1, 1] -> [0, 1]로 역정규화
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        # 텐서를 PIL 이미지로
        transforms.ToPILImage()
    ])

    # 결과 저장 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)

    lr_image_paths = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    total_psnr = 0.0

    # --- 3. 추론 루프 ---
    # torch.no_grad() 컨텍스트 안에서 실행하여, 불필요한 그래디언트 계산을 방지합니다.
    with torch.no_grad():
        for lr_path in tqdm(lr_image_paths, desc="테스트 진행률"):
            filename = os.path.basename(lr_path)

            # LR 이미지 불러오기 및 전처리
            lr_img = Image.open(lr_path).convert('RGB')
            lr_tensor = preprocess(lr_img).unsqueeze(0).to(device)  # 배치 차원 추가

            # 모델 추론 (SR 이미지 생성)
            sr_tensor = model(lr_tensor)

            # 후처리 및 저장
            sr_img = postprocess(sr_tensor.squeeze(0).cpu())  # 배치 차원 제거 및 CPU로 이동
            sr_img.save(os.path.join(args.output_dir, filename))

            # --- 4. (선택) 성능 평가 (PSNR) ---
            if args.hr_dir:
                hr_path = os.path.join(args.hr_dir, filename)
                if os.path.exists(hr_path):
                    hr_img = Image.open(hr_path).convert('RGB')

                    # PIL 이미지를 NumPy 배열로 변환
                    sr_np = np.array(sr_img)
                    hr_np = np.array(hr_img)

                    psnr = calculate_psnr(hr_np, sr_np)
                    total_psnr += psnr
                    print(f"{filename}: PSNR = {psnr:.2f} dB")

    if args.hr_dir and len(lr_image_paths) > 0:
        avg_psnr = total_psnr / len(lr_image_paths)
        print(f"\n평균 PSNR: {avg_psnr:.2f} dB")


if __name__ == '__main__':
    # argparse를 사용해 터미널에서 인자를 받습니다.
    parser = argparse.ArgumentParser(description='KOALAnet Baseline Model Testing')
    parser.add_argument('--weights', type=str, required=True, help='학습된 모델 가중치 파일 경로 (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='저화질(LR) 테스트 이미지가 있는 폴더')
    parser.add_argument('--output_dir', type=str, required=True, help='결과(SR) 이미지를 저장할 폴더')
    parser.add_argument('--scale', type=int, default=4, help='업샘플링 배율')
    parser.add_argument('--hr_dir', type=str, default=None, help='(선택) 고화질(HR) 정답 이미지가 있는 폴더 (PSNR 계산용)')
    args = parser.parse_args()

    main(args)