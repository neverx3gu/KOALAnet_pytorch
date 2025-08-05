import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Sampler

from models import UpsamplingBaselineNetwork
from dataset import SISRDataset

# 1 epoch 당 정해진 수의 샘플만 무작위로 추출하는 샘플러 (TensorFlow의 'updates_per_epoch' 개념을 구현)
class EpochBasedSampler(Sampler):
    def __init__(self, data_source, num_samples_per_epoch):
        self.data_source = data_source
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self):
        # 전체 데이터셋 인덱스에서 필요한 만큼만 무작위로 뽑음 (중복 허용)
        indices = torch.randint(high=len(self.data_source), size=(self.num_samples_per_epoch,)).tolist()
        return iter(indices)

    def __len__(self):
        # DataLoader가 1 에포크의 길이를 알 수 있도록 설정
        return self.num_samples_per_epoch

# --- 1. 하이퍼파라미터 및 설정 ---
# preprocssed image dataset 쓸 때는 안 쓰는 변수
HR_DATA_DIR = './data/DIV2K_train_HR'

# with preprocessed patch
# LR_PREPROCESSED_DIR = './data/train_preprocessed/lr'
# HR_PREPROCESSED_DIR = './data/train_preprocessed/hr'
MODEL_SAVE_PATH = './koalanet_stage2.pth'

SCALE_FACTOR = 4
PATCH_SIZE_LR = 64
BATCH_SIZE = 8
EPOCHS = 200 # 논문에서는 총 iteration이 200k
UPDATES_PER_EPOCH = 100 # TF 원본 코드의 1 에포크당 업데이트 횟수
LEARNING_RATE = 1e-4

# 논문의 학습률 감소 정책 (80%, 90% 지점에서 1/10로 감소)
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
    dataset = SISRDataset(hr_dir=HR_DATA_DIR, scale_factor=SCALE_FACTOR, patch_size_lr=PATCH_SIZE_LR) # preprocessed image data set 사용할 때는 X
    # dataset = SISRDataset(lr_dir=LR_PREPROCESSED_DIR, hr_dir=HR_PREPROCESSED_DIR)

    # 원본 TF 코드의 1 에포크당 업데이트 횟수(100)를 기준으로 샘플러 생성
    # 1 에포크당 100번 업데이트 * 배치크기 8 = 800개 샘플
    num_samples_per_epoch = 100 * BATCH_SIZE
    epoch_sampler = EpochBasedSampler(dataset, num_samples_per_epoch)

    # DataLoader에 sampler를 지정하고, shuffle=False로 설정
    # (Sampler가 이미 셔플링을 담당하므로 shuffle은 False여야 합니다.)
    # mps는 pin_memory = True 지원하지 않음
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=epoch_sampler, # 커스텀 샘플러 지정
        shuffle=False, # Sampler 사용 시 반드시 False
        num_workers=12,
        pin_memory=False,
        persistent_workers=True
    )

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
        progress_bar = tqdm(data_loader, desc=f"epoch [{epoch + 1}/{EPOCHS}] LR={scheduler.get_last_lr()[0]:.1e}")

        total_loss = 0.0

        for lr_imgs, hr_imgs in progress_bar:
            # 데이터를 설정된 장치(GPU 또는 CPU)로 이동
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # 옵티마이저의 기울기를 0으로 초기화
            optimizer.zero_grad()

            # MPS 장치에 대해 autocast 활성화
            with torch.autocast(device_type="mps"):
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

        # 에포크가 끝나면 스케줄러를 업데이트
        scheduler.step()

    # --- 4. 훈련 종료 및 모델 저장 ---
    print("훈련 완료!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델이 '{MODEL_SAVE_PATH}' 파일로 저장되었습니다.")


if __name__ == '__main__':
    main()