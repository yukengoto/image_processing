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
    size_features = []

    for path in image_paths:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            img = Image.open(path).convert("RGB")
            w, h = img.size
            size_feat = [w, h, w / h]  # 幅, 高さ, アスペクト比
            image = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image).cpu().numpy()[0]
            features.append(feat)
            size_features.append(size_feat)

            # 進捗表示
            if len(features) % 10 == 0:
                sys.stdout.write(f"\r進捗: {len(features)}/{len(image_paths)}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n中断されました。")
            raise
        except Exception:
            continue
    return features, size_features

# 最適クラスタ数を自動判定（シルエットスコア）
def find_best_k(features_np, k_range=range(5, 15)):
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
    # take first N of image_paths for testing
    test_max = 100000
    img_dir = "."
    #
    print("searching for images in the current directory...")
    sys.stdout.flush()
    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.endswith(('.jpg', '.JPG', '.png'))]

    if not image_paths:
        print("エラー: 画像フォルダが空です。画像を追加してください。")
        return

    print(f"画像フォルダにある画像の数: {len(image_paths)}")

    if len(image_paths) > test_max:
        print(f"画像が多すぎるため、最初の{test_max}枚のみを使用します。")
        image_paths = image_paths[:test_max]

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()

    features, size_features = extract_features(image_paths, model, preprocess)
    if not features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        return
    if not size_features:
        print("エラー: size_featuresが空です。画像の読み込みに失敗した可能性があります。")
        return

    # numpy 配列に変換
    features_np = np.array(features)
    size_np = np.array(size_features)

    # サイズ情報を正規化（スケーリング）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    size_np_scaled = scaler.fit_transform(size_np)

    # CLIP特徴と連結
    combined_features = np.concatenate([features_np, size_np_scaled], axis=1)

    print(f"\n特徴量の形状: {features_np.shape}")
    print(f"\n特徴量の形状: {combined_features.shape}")

    print("\n最適なクラスタ数を探索中...")
    best_k = find_best_k(combined_features)
    print(f"→ 最適なクラスタ数は {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=0)
    labels = kmeans.fit_predict(combined_features)

    cluster_names = describe_clusters(combined_features, image_paths, labels, kmeans)

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

