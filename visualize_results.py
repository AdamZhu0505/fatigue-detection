import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# ========== 参数路径 ==========
ear_csv_path = "out/ear_log.csv"
images_root = "data/images"
save_closed_dir = "out/closed_frames"
plot_output_dir = "out/plots"

# EAR 阈值
ear_thresh = 0.2

# ========== 创建输出目录 ==========
os.makedirs(save_closed_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

# ========== 读取 CSV ==========
df = pd.read_csv(ear_csv_path)
df["frame"] = df["frame"].astype(str).str.zfill(5)  # 保证编号格式

# ========== 闭眼帧图像保存 ==========
for seq in df["seq"].unique():
    seq_df = df[df["seq"] == seq]
    closed_df = seq_df[seq_df["avg_ear"] < ear_thresh]

    if closed_df.empty:
        continue

    # 创建目标文件夹
    seq_out = os.path.join(save_closed_dir, seq)
    os.makedirs(seq_out, exist_ok=True)

    for _, row in closed_df.iterrows():
        frame = row["frame"]
        left_img = os.path.join(images_root, seq, f"left_{frame}.png")
        right_img = os.path.join(images_root, seq, f"right_{frame}.png")
        dst_left = os.path.join(seq_out, f"left_{frame}.png")
        dst_right = os.path.join(seq_out, f"right_{frame}.png")

        if os.path.exists(left_img) and os.path.exists(right_img):
            shutil.copyfile(left_img, dst_left)
            shutil.copyfile(right_img, dst_right)

    print(f"✅ 保存闭眼图像：{seq} 共 {len(closed_df)} 对")

# ========== EAR 可视化 ==========
for seq in df["seq"].unique():
    seq_df = df[df["seq"] == seq].sort_values("frame")
    x = range(len(seq_df))
    y = seq_df["avg_ear"].values
    closed_x = [i for i, v in enumerate(y) if v < ear_thresh]
    closed_y = [v for v in y if v < ear_thresh]

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="avg EAR")
    plt.scatter(closed_x, [y[i] for i in closed_x], color="red", label="Closed (EAR < 0.2)", s=20)
    plt.axhline(y=ear_thresh, color="gray", linestyle="--", label="Threshold = 0.2")
    plt.title(f"EAR 曲线图 - {seq}")
    plt.xlabel("Frame")
    plt.ylabel("avg EAR")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(plot_output_dir, f"{seq}_ear_plot.png")
    plt.savefig(save_path)
    plt.close()

    print(f"📈 保存 EAR 曲线图：{save_path}")

print("🏁 所有闭眼图像保存 + EAR 图像生成完成。")
