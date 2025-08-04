import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 저희가 직접 만든 모듈들을 불러옵니다.
from models import UpsamplingBaselineNetwork
from dataset import SISRDataset

# --- 1. 하이퍼파라미터 및 설정 ---
# 논문과 최대한 유사하게 설정합니다.
HR_DATA_DIR = './data/DIV2K_train_HR'
MODEL_SAVE_PATH = './koalanet_baseline_x4_final.pth'

SCALE_FACTOR = 4
PATCH_SIZE_LR = 64
BATCH_SIZE = 8
EPOCHS = 200  # 논문에서는 200k iteration을 제안했지만, 우선 200 에포크로 설정합니다.
LEARNING_RATE = 1e-4

# 논문의 학습률 감소 정책 (80%, 90% 지점에서 1/10로 감소)
# 200 에포크 기준: 160, 180 에포크
LR_DECAY_MILESTONES = [0.8*EPOCHS, 0.9*EPOCHS]
LR_DECAY_FACTOR = 0.1


def main():
    # --- 2. 훈련 준비 ---
    # GPU 가속 장치 확인 (Apple Silicon MPS, NVIDIA CUDA 순서로 확인)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Apple Silicon GPU (MPS)를 사용합니다.")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("NVIDIA GPU (CUDA)를 사용합니다.")
    else:
        device = torch.device('cpu')
        print("CPU를 사용합니다.")

    # 데이터셋 및 데이터로더 설정
    print("데이터셋 로딩 중... (첫 실행 시 시간이 걸릴 수 있습니다)")
    dataset = SISRDataset(hr_dir=HR_DATA_DIR, scale_factor=SCALE_FACTOR, patch_size_lr=PATCH_SIZE_LR)
    # num_workers: 데이터를 불러올 때 사용할 프로세스 수. 컴퓨터 환경에 맞게 조절하세요.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    print("데이터셋 로딩 완료!")

    # 모델 인스턴스 생성 및 장치로 이동
    model = UpsamplingBaselineNetwork(scale_factor=SCALE_FACTOR).to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.L1Loss()  # 원본 논문처럼 L1 손실(MAE)을 사용
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습률 스케줄러(Learning Rate Scheduler) 정의
    # MultiStepLR은 지정된 에포크(milestones)에 도달할 때마다 학습률에 gamma를 곱해줍니다.
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DECAY_MILESTONES, gamma=LR_DECAY_FACTOR)

    # --- 3. 훈련 루프 ---
    print("훈련을 시작합니다...")
    for epoch in range(EPOCHS):
        model.train()  # 모델을 훈련 모드로 설정

        # tqdm을 사용하여 배치 진행률 표시
        progress_bar = tqdm(data_loader, desc=f"에포크 [{epoch + 1}/{EPOCHS}] LR={scheduler.get_last_lr()[0]:.1e}")

        total_loss = 0.0

        for lr_imgs, hr_imgs in progress_bar:
            # 데이터를 설정된 장치(GPU 또는 CPU)로 이동
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # 옵티마이저의 기울기를 0으로 초기화
            optimizer.zero_grad()

            # 순전파(Forward pass): 모델에 LR 이미지를 입력하여 SR 이미지 생성
            sr_imgs = model(lr_imgs)

            # 손실 계산
            loss = criterion(sr_imgs, hr_imgs)

            # 역전파(Backward pass): 손실에 대한 기울기 계산
            loss.backward()

            # 옵티마이저 실행: 계산된 기울기를 바탕으로 모델의 가중치 업데이트
            optimizer.step()

            total_loss += loss.item()

            # 진행률 표시줄에 현재 평균 손실 업데이트
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        # 에포크가 끝나면 스케줄러를 업데이트합니다.
        scheduler.step()

    # --- 4. 훈련 종료 및 모델 저장 ---
    print("훈련 완료!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    main()