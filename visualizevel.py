import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def numeric_key(fname):
    """ 파일 이름에서 숫자 값을 추출하여 정렬 기준으로 사용 """
    base = os.path.basename(fname)
    prefix = base.split('_')[0]
    return float(prefix)


def load_velocity_data(folder_path, nx, ny):
    """ 주어진 폴더에서 ux, uy 속도 데이터를 로드하여 리스트로 반환 """
    file_list = glob.glob(os.path.join(folder_path, "*.txt"))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No velocity data files found in folder: {folder_path}")

    file_list.sort(key=numeric_key)  # 파일 정렬

    num_files = len(file_list)
    print(f"Found {num_files} velocity files in {folder_path}.")

    data_list = []
    labels = []

    for idx, file in enumerate(file_list):
        data = np.loadtxt(file, delimiter=",", skiprows=1)  # CSV 형식 가정 (ux, uy)
        if data.shape[1] != 2:
            raise ValueError(f"Invalid data format in file: {file}")

        ux, uy = data[:, 0], data[:, 1]  # ux, uy 분리
        magnitude = np.sqrt(ux**2 + uy**2)  # 속도 크기 계산

        magnitude_2d = magnitude.reshape((nx, ny))  # 2D 배열로 변환
        data_list.append(magnitude_2d)

        label = file.split('/')[-1].split('_')[0]  # 파일명에서 시간 추출
        labels.append(label)

    return data_list, labels, num_files


def create_velocity_animation(data_list, labels, nu, num_files, nx, ny, Lx, Ly, save_filename="velocity_magnitude_animation.gif", interval_ms=100):
    """ 속도 크기(magnitude) 애니메이션 생성 및 저장 (그리드 & 도메인 크기 자동 적용) """

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(4, 3))  # ✅ 크기 줄임
    im = ax.imshow(
        data_list[0],
        origin='lower',
        cmap='inferno',  # ✅ 강렬한 시각 효과 적용
        extent=[0, Lx, 0, Ly],  # ✅ 도메인 크기 자동 적용
        animated=True
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)  # ✅ 컬러바 크기 최적화
    cbar.set_label("Velocity Magnitude |U|", fontsize=8)  # ✅ 컬러바 라벨 크기 축소
    cbar.ax.tick_params(labelsize=7)  # ✅ 컬러바 숫자 크기 축소

    ax.set_xlabel("X", fontsize=8)  # ✅ x축 라벨 크기 조정
    ax.set_ylabel("Y", fontsize=8)  # ✅ y축 라벨 크기 조정

    time_text = ax.text(0.95, 0.05, f"t = {labels[0]} s", color="white",
                        fontsize=6, fontweight="light", ha="right", va="bottom",
                        transform=ax.transAxes, bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.2"))  # ✅ 박스 크기 축소

    ax.set_title(f"Velocity Magnitude (nu={nu}), {nx}x{ny}", fontsize=9)  # ✅ 제목 크기 조정

    def init():
        im.set_data(data_list[0])
        time_text.set_text(f"t = {labels[0]} s")
        return [im, time_text]

    def update(frame):
        im.set_data(data_list[frame])
        time_text.set_text(f"t = {labels[frame]} s")
        return [im, time_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=num_files,
        init_func=init,
        interval=interval_ms,
        blit=True
    )

    plt.tight_layout()
    ani.save(save_filename, writer="pillow", fps=10)
    print(f"Animation saved as {save_filename}")

    plt.show()


# ✅ 실행 예제 (값만 수정하면 자동 적용)
if __name__ == "__main__":
    folder_path = "../proceed_data/proceed_data_v_half/processed_data_u/"  # ✅ 데이터 폴더 경로 설정
    nx, ny = 1024, 1024  # ✅ 원하는 그리드 크기 설정
    Lx, Ly = 6.283, 6.283  # ✅ 원하는 도메인 크기 설정
    interval_ms = 10
    nu = 0.000005  # ✅ 점성 계수

    # ✅ 데이터 로드
    data_list, labels, num_files = load_velocity_data(folder_path, nx, ny)

    # ✅ 애니메이션 생성 및 저장
    create_velocity_animation(data_list, labels, nu, num_files, nx, ny, Lx, Ly, save_filename="[x0.5]Velocity.gif", interval_ms=10)
