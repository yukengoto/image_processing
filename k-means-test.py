import os
import sys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import open_clip

# 画像読み込み & 特徴量抽出用関数（再利用しやすくする）
def extract_features(image_paths, model, preprocess):
    features = []
    for i, path in enumerate(image_paths, 1):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(image)
            features.append(feat[0].numpy())

            # 一行で進捗表示
            if i % 10 == 0 or i == len(image_paths):
                sys.stdout.write(f"\r進捗: {i}/{len(image_paths)} ({(i / len(image_paths)) * 100:.2f}%)")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n中断されました。")
            raise
        except:
            continue
    print()  # 改行
    return features

# 最適クラスタ数を自動判定（シルエットスコア）
def find_best_k(features_np, k_range=range(2, 21)):
    best_score = -1
    best_k = 2
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(features_np)
            score = silhouette_score(features_np, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    return best_k

# ラベルに名前をつける（中心ベクトルに最も近い画像のファイル名を参考に）
def describe_clusters(features_np, image_paths, labels, kmeans):
    from sklearn.metrics import pairwise_distances_argmin_min
    cluster_names = {}
    for i in range(kmeans.n_clusters):
        indices = [j for j, label in enumerate(labels) if label == i]
        if not indices:
            cluster_names[i] = f"cluster_{i}"
            continue
        cluster_center = kmeans.cluster_centers_[i].reshape(1, -1)
        cluster_features = features_np[indices]
        closest_idx, _ = pairwise_distances_argmin_min(cluster_center, cluster_features)
        best_idx = indices[closest_idx[0]]
        basename = os.path.splitext(os.path.basename(image_paths[best_idx]))[0]
        cluster_names[i] = basename[:20] or f"cluster_{i}"
    return cluster_names

# メイン処理
def main():
    img_dir = "."
    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.endswith(('.jpg', '.JPG', '.png'))]

    if not image_paths:
        print("エラー: 画像フォルダが空です。画像を追加してください。")
        return

    print(f"画像フォルダにある画像の数: {len(image_paths)}")

    # take first N of image_paths for testing
    test_max = 100000
    if len(image_paths) > test_max:
        print(f"画像が多すぎるため、最初の{test_max}枚のみを使用します。")
        image_paths = image_paths[:test_max]

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()

    features = extract_features(image_paths, model, preprocess)
    if not features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        return

    features_np = np.array(features)
    print(f"\n特徴量の形状: {features_np.shape}")

    print("\n最適なクラスタ数を探索中...")
    best_k = find_best_k(features_np)
    print(f"→ 最適なクラスタ数は {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=0)
    labels = kmeans.fit_predict(features_np)

    cluster_names = describe_clusters(features_np, image_paths, labels, kmeans)

    print("\n画像をクラスタごとに移動中...")
    os.makedirs("output", exist_ok=True)
    total = len(image_paths)

    for i, (label, path) in enumerate(zip(labels, image_paths), 1):
        name = cluster_names[label]
        dest = os.path.join("output", name)
        os.makedirs(dest, exist_ok=True)

        try:
            os.rename(path, os.path.join(dest, os.path.basename(path)))
        except FileExistsError:
            continue
        # 一行で進捗表示
        if i % 10 == 0 or i == total:
            sys.stdout.write(f"\r分類中: {i}/{total} ({(i / total) * 100:.2f}%)")
            sys.stdout.flush()

    print("\n分類完了。output/ フォルダ内を確認してください。")

main()

