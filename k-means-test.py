import os
import sys
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import open_clip

# 画像読み込み & 特徴量抽出用関数（再利用しやすくする）
def extract_features(image_paths, model, preprocess):
    features = []
    cnt = 0
    for path in image_paths:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            image = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image).cpu().numpy()[0]
            features.append(feat)
            cnt += 1
            if cnt % 10 == 0:
                sys.stdout.write(f"\r進捗: {cnt}/{len(image_paths)}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n中断されました。")
            raise
        except Exception:
            continue

    if not features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        exit(1)

    # numpy 配列に変換
    features_np = np.array(features)
    return features_np

# 画像読み込み & 特徴量抽出用関数（再利用しやすくする）
def extract_sizes(image_paths, model, preprocess):
    size_features = []
    cnt = 0
    for path in image_paths:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            if True:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                # 面積は対数を取ると極端な差を緩和できる
                #size_feat = [np.log1p(w * h), w / h]
                #size_feat = [w, h, w / h]  # 幅, 高さ, アスペクト比
                size_feat = [np.log1p(w), np.log1p(h)]
                size_features.append(size_feat)
            cnt += 1
            if cnt % 10 == 0:
                sys.stdout.write(f"\r進捗: {cnt}/{len(image_paths)}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n中断されました。")
            raise
        except Exception:
            continue

    if not size_features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        exit(1)

    # numpy 配列に変換
    size_np = np.array(size_features)

    # サイズ情報を正規化（スケーリング）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    size_np_scaled = scaler.fit_transform(size_np)
    return size_np_scaled

# 最適クラスタ数を自動判定（シルエットスコア）
def find_best_k(features_np, k_range=range(5, 13)):
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

# 画像パスを取得する関数（再帰的にディレクトリを探索）
def get_image_paths(img_dir, recursive=False):
    exts = ('.jpg', '.png')  # 小文字で統一しておく
    image_paths = []

    if recursive:
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(exts):
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(img_dir):
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(img_dir, file))

    return image_paths

# 空のディレクトリを削除する関数
def remove_empty_dirs(root_dir):
    for current_dir, dirs, files in os.walk(root_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(current_dir)
                print(f"空フォルダを削除: {current_dir}")
            except OSError:
                # 権限や他のプロセスによる使用中などで削除できない場合
                continue

# メイン処理
def main():
    parser = argparse.ArgumentParser(description="画像クラスタリング")
    parser.add_argument('--max', type=int, default=100000,
                        help='読み込む画像の最大枚数（デフォルト: 100000）')
    parser.add_argument('--dir', type=str, default='.',
                        help='画像フォルダのパス（デフォルト: カレントディレクトリ）')
    parser.add_argument('--recursive', action='store_true',
                        help='サブフォルダも再帰的に検索する場合は指定')
    parser.add_argument('--no-picture', action='store_false', dest='check_picture',
                        help='CLIP特徴を取得しない場合は指定')
    parser.add_argument('--no-size', action='store_false', dest='check_size',
                        help='サイズ情報を取得しない場合は指定')
    args = parser.parse_args()
    img_dir = args.dir
    n_max = args.max
    recursive = args.recursive
    check_picture = args.check_picture  # CLIP特徴を取得するかどうか
    check_size = args.check_size  # サイズ情報を取得するかどうか
    #
    #
    print("searching for images in the current directory...")
    sys.stdout.flush()
    image_paths = get_image_paths(img_dir, recursive)

    if not image_paths:
        print("エラー: 画像フォルダが空です。画像を追加してください。")
        return

    print(f"画像フォルダにある画像の数: {len(image_paths)}")

    if len(image_paths) > n_max:
        print(f"画像が多すぎるため、最初の{n_max}枚のみを使用します。")
        image_paths = image_paths[:n_max]

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()

    if check_picture:
        features_np = extract_features(image_paths, model, preprocess)
        print(f"\n特徴量の形状: {features_np.shape}")

        print("\n最適なクラスタ数を探索中...")
        best_k = find_best_k(features_np)
        print(f"→ 最適なクラスタ数は {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=0)
        labels = kmeans.fit_predict(features_np)

        cluster_names = describe_clusters(features_np, image_paths, labels, kmeans)
    elif check_size:
        features_np = extract_sizes(image_paths, model, preprocess)
        print(f"\n特徴量の形状: {features_np.shape}")


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

    remove_empty_dirs("output")

    print("\n分類完了。output/ フォルダ内を確認してください。")


if __name__ == "__main__":
    main()