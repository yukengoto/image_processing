import os
import sys
from PIL import Image
import torch
import open_clip
import numpy as np
from sklearn.cluster import KMeans

# モデル・前処理
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()

# 入力フォルダ
img_dir = "." # current directory
image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg','.JPG','.png'))]

# print paths in image_paths
if not image_paths:
    print("エラー: 画像フォルダが空です。画像を追加してください。")
    exit()  # 処理を中断してエラーを明確にする
else:
    print(f"画像フォルダにある画像の数: {len(image_paths)}")

# 画像ベクトル化
features = []
for path in image_paths:
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(image)
        features.append(feat[0].numpy())
        # show progress by percentage at bottom line
        if len(features) % 10 == 0:  # 10枚ごとに進捗を表示
            sys.stdout.write("\r")
            sys.stdout.write(f"進捗: {len(features)}/{len(image_paths)} ({(len(features) / len(image_paths)) * 100:.2f}%)")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n中断されました。")
        raise
    except: 
        continue

# features を生成するコードの後に
print(f"featuresの長さ: {len(features)}")
if len(features) == 0:
    print("エラー: featuresが空です。データの読み込みまたは特徴量抽出に問題がある可能性があります。")
    exit() # 処理を中断してエラーを明確にする

# numpy配列に変換
features_np = np.array(features)

# 配列の次元を確認
print(f"features_npの次元数: {features_np.ndim}")
print(f"features_npの形状: {features_np.shape}")


# クラスタリング（例：10クラス）
kmeans = KMeans(n_clusters=10)
labels = kmeans.fit_predict(np.array(features))

# フォルダ分類
for label, path in zip(labels, image_paths):
    dest = os.path.join("output", f"cluster_{label}")
    os.makedirs(dest, exist_ok=True)
    os.rename(path, os.path.join(dest, os.path.basename(path)))


    