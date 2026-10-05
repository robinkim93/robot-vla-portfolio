# LeRobotDataset 클래스 임포트
# LeRobotDataset 클래스는 데이터셋을 관리하는 클래스
from lerobot.datasets import LeRobotDataset 

# LeRobotDataset 클래스 인스턴스 생성
# LeRobotDataset 클래스 인스턴스 생성 시 데이터셋 경로와 시연 번호를 지정
ds = LeRobotDataset("lerobot/svla_so101_pickplace", episodes=[0])

# fps, 시연 수, 장면 수, 데이터 속성 출력
print("fps:", ds.fps)
print("시연 수", ds.num_episodes, "/ 장면 수", ds.num_frames)
print("데이터 속성:", list(ds.features))
print()

# 첫 번째 장면 데이터 출력
frame = ds[0]
for key, value in frame.items():
    shape = tuple(value.shape) if hasattr(value, 'shape') else None
    if shape is not None and value.numel() > 6:
        print(f"{key:30s} 모양={shape}")
    else:
        print(f"{key:30s} {value}")

print()
print("state: ", frame["observation.state"])
print("action:", frame["action"])