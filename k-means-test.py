import os
import sys
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import open_clip
# from sklearn.preprocessing import StandardScaler # 追加

# 画像読み込み & CLIP特徴量抽出用関数
def extract_features(image_paths, model, preprocess):
    features = []
    cnt = 0
    # デバイス設定をループの外に出す
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB") # ここで画像を読み込む
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
        except Exception as e: # エラー内容も表示
            print(f"\nエラー発生（{path}）：{e}")
            continue

    if not features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        exit(1)

    # numpy 配列に変換
    features_np = np.array(features)
    return features_np

# 画像サイズ情報抽出用関数（元のextract_sizesだが、ここではKMeansのためだけに特徴量を抽出する）
# def extract_sizes(image_paths): # model, preprocess 引数を削除
#     size_features = []
#     cnt = 0
#     for path in image_paths:
#         try:
#             img = Image.open(path).convert("RGB")
#             w, h = img.size
#             # 幅と高さの対数を使用
#             size_feat = [np.log1p(w), np.log1p(h)]
#             size_features.append(size_feat)
#             cnt += 1
#             # プログレスバーはメインループで管理
#             if cnt % 10 == 0:
#                 sys.stdout.write(f"\r進捗: {cnt}/{len(image_paths)}")
#                 sys.stdout.flush()
#         except KeyboardInterrupt:
#             print("\n中断されました。")
#             raise
#         except Exception as e:
#             # print(f"\nエラー発生（{path}）：{e}") # 大量のエラー表示を避けるためコメントアウト
#             continue

#     if not size_features:
#         print("エラー: size_featuresが空です。画像の読み込みに失敗した可能性があります。")
#         exit(1)

#     # numpy 配列に変換
#     size_np = np.array(size_features)

#     # サイズ情報を正規化（スケーリング）
#     scaler = StandardScaler()
#     size_np_scaled = scaler.fit_transform(size_np)
#     return size_np_scaled

# 固定閾値でサイズ分類を行う関数
def assign_fixed_size_labels(image_paths):
    labels = []
    cluster_names = {}

    # 閾値定義
    THRESHOLD_AREA_SMALL = 200 * 200
    THRESHOLD_AREA_MEDIUM = 2000 * 1100
    THRESHOLD_ASPECT_PORTRAIT = 0.98
    THRESHOLD_ASPECT_SQUARE = 1.02

    # クラスタ名マッピング用
    label_map = {}
    current_label_id = 0

    # 画像サイズごとの出現回数をカウント
    from collections import Counter, defaultdict
    size_counter = Counter()
    size_to_paths = defaultdict(list)
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            w, h = img.size
            size_counter[(w, h)] += 1
            size_to_paths[(w, h)].append(path)
        except Exception:
            continue
        if i % 10 == 0 or i == len(image_paths) - 1:
            sys.stdout.write(f"\rSearching for popular sizes: {i}/{len(image_paths)} ({(i / len(image_paths)) * 100:.2f}%)")
            sys.stdout.flush()

    # 多数派サイズの閾値（例: 画像数の5%以上かつ10枚以上）
    min_count = max(10, int(len(image_paths) * 0.05))
    popular_sizes = {size for size, count in size_counter.items() if count >= min_count}

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            w, h = img.size
            area = w * h
            aspect_ratio = w / h if h != 0 else 0

            # 多数派サイズなら専用ラベル
            if (w, h) in popular_sizes:
                folder_name = f"{w}x{h}"
            else:
                if area < THRESHOLD_AREA_SMALL:
                    size_category = "Small"
                elif area < THRESHOLD_AREA_MEDIUM:
                    size_category = "Medium"
                else:
                    size_category = "Large"

                if aspect_ratio < THRESHOLD_ASPECT_PORTRAIT:
                    aspect_category = "Portrait"
                elif aspect_ratio < THRESHOLD_ASPECT_SQUARE:
                    aspect_category = "Square"
                else:
                    aspect_category = "Landscape"

                folder_name = f"{size_category}_{aspect_category}"

            if folder_name not in label_map:
                label_map[folder_name] = current_label_id
                cluster_names[current_label_id] = folder_name
                current_label_id += 1

            labels.append(label_map[folder_name])

            if i % 10 == 0 or i == len(image_paths) - 1:
                sys.stdout.write(f"\r進捗: {i}/{len(image_paths)} ({(i / len(image_paths)) * 100:.2f}%)")
                sys.stdout.flush()

        except Exception as e:
            print(f"\nWarning: Could not process {path} due to error: {e}. Skipping.")
            labels.append(-1)

    return np.array(labels), cluster_names


# 最適クラスタ数を自動判定（シルエットスコア）
def find_best_k(features_np, k_range=range(2, 13)): # k_rangeを2からに修正（シルエットスコアの最小要件）
    best_score = -1
    best_k = 2
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto') # n_init='auto' を追加して警告を回避
            labels = kmeans.fit_predict(features_np)
            # クラスタが1つしかない場合や、データ点がクラスタ数より少ない場合はシルエットスコア計算不可
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(features_np, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e: # エラー内容も表示
            print(f"\nKMeans (k={k}) error: {e}")
            continue
    return best_k

# ラベルに名前をつける（中心ベクトルに最も近い画像のファイル名を参考に）
# 固定閾値分類の場合はこの関数は使用しないか、別の目的で使う
def describe_clusters(features_np, image_paths, labels, kmeans):
    from sklearn.metrics import pairwise_distances_argmin_min
    from collections import Counter, defaultdict
    cluster_names = {}
    for i in range(kmeans.n_clusters):
        indices = [j for j, label in enumerate(labels) if label == i]
        if not indices:
            cluster_names[i] = f"cluster_{i}"
            continue
        cluster_center = kmeans.cluster_centers_[i].reshape(1, -1)
        cluster_features = features_np[indices]
        
        if cluster_features.shape[0] == 0:
            cluster_names[i] = f"cluster_{i}"
            continue

        closest_idx, _ = pairwise_distances_argmin_min(cluster_center, cluster_features)
        best_idx_in_cluster = closest_idx[0] 
        best_idx = indices[best_idx_in_cluster]

        try:
            with Image.open(image_paths[best_idx]) as img:
                w, h = img.size
                # クラスタ名を代表画像のアスペクト比とサイズで付ける
                cluster_names[i] = f"{w}x{h}_ratio_{w/h:.2f}"
        except Exception:
            basename = os.path.splitext(os.path.basename(image_paths[best_idx]))[0]
            cluster_names[i] = basename[:20] or f"cluster_{i}"
    return cluster_names

# 画像パスを取得する関数（再帰的にディレクトリを探索）
def get_image_paths(img_dir, recursive=False):
    exts = ('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp') # 拡張子を増やすと良い
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
    parser.add_argument('--use-picture', action='store_true', dest='check_picture', # デフォルトをFalseに変更
                        help='CLIP特徴を取得して分類する場合は指定') # オプション名を変更
    parser.add_argument('--use-size-rule', action='store_true', dest='check_size_rule', # 新しいオプション
                        help='固定閾値でサイズ分類する場合は指定')
    parser.add_argument('--k', type=int, default=None,
                        help='K-Meansのクラスタ数を手動で指定する場合（デフォルト: 自動判定）')
    args = parser.parse_args()
    img_dir = args.dir
    n_max = args.max
    recursive = args.recursive
    check_picture = args.check_picture      # CLIP特徴を使うか
    check_size_rule = args.check_size_rule  # 固定閾値のサイズ分類を使うか

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

    # CLIPモデルのロードはcheck_pictureがTrueの場合のみ
    model, _, preprocess = None, None, None # 初期化
    if check_picture:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()

    labels = []
    cluster_names = {}

    if check_picture: # CLIP特徴量でK-Meansクラスタリング
        print("\nCLIP特徴量でクラスタリング中...")
        features_np = extract_features(image_paths, model, preprocess)
        print(f"\n特徴量の形状: {features_np.shape}")

        if args.k is None:
            print("\n最適なクラスタ数を探索中...")
            best_k = find_best_k(features_np)
            print(f"→ 最適なクラスタ数は {best_k}")
        else:
            best_k = args.k
            print(f"クラスタ数 (手動指定): {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(features_np)
        cluster_names = describe_clusters(features_np, image_paths, labels, kmeans)
        output_dir = "output_by_clip_kmeans"

    elif check_size_rule: # 固定閾値でサイズ分類
        print("\n固定閾値でサイズ分類中...")
        labels, cluster_names = assign_fixed_size_labels(image_paths)
        if len(set(labels)) <= 1: # 分類が機能しなかった場合（すべて同じカテゴリなど）
            print("警告: 固定閾値による分類が効果的ではありませんでした。")
        output_dir = "output_by_fixed_size_rule"

    else: # どちらの分類も指定されていない場合
        print("エラー: 分類方法 (--use-picture または --use-size-rule) を指定してください。")
        return

    print("\n画像をクラスタごとに移動中...")
    os.makedirs(output_dir, exist_ok=True)
    total = len(image_paths)

    # ラベルが -1 (スキップされた画像) のものや、labels の長さが image_paths と異なる場合の対処
    if len(labels) != len(image_paths):
        print("警告: 処理された画像の数とラベルの数が一致しません。スキップされた画像がある可能性があります。")
        # ここで labels と image_paths の対応関係を再構築するロジックが必要になる場合があります
        # 例として、ラベルがない場合は 'unknown' フォルダに移動するなどの対策
        #temp_labels = [-1] * len(image_paths)
        #valid_image_paths = []
        #valid_labels = []
        for i, path in enumerate(image_paths):
            try:
                # assign_fixed_size_labels 内でエラーでスキップされた画像を特定
                # ここでは単純に既存の labels と cluster_names に頼る
                # もし assign_fixed_size_labels が一部の画像をスキップした場合、
                # labels の長さが image_paths より短くなる可能性があるので注意
                
                # assign_fixed_size_labels はエラーが発生しても labels には追加する（ここでは -1）ので
                # 長さは一致すると仮定し、-1 は 'Unknown' フォルダに入れる
                if labels[i] == -1:
                    name = "Unknown_Error"
                    if name not in cluster_names.values():
                        cluster_names[max(cluster_names.keys()) + 1 if cluster_names else 0] = name
                    label_for_unknown = [k for k, v in cluster_names.items() if v == name][0]
                    labels[i] = label_for_unknown # -1を実際のラベルIDに変換
            except IndexError:
                # labelsがimage_pathsより短い場合に発生
                print(f"致命的なエラー: ラベルと画像の数が一致しません。{len(labels)} vs {len(image_paths)}")
                return # 処理を中断
                
    for i, path in enumerate(image_paths, 1): # iを1からカウント開始
        label = labels[i-1] # labelsは0-indexedなのでi-1
        name = cluster_names.get(label, "Unknown_Category") # ラベルが見つからない場合のデフォルト名
        
        # ファイル名に使えない文字を置換
        name = name.replace(":", "_").replace("/", "_").replace("\\", "_").replace("*", "_").replace("?", "_").replace('"', '_').replace("<", "_").replace(">", "_").replace("|", "_")
        dest = os.path.join(output_dir, name)
        os.makedirs(dest, exist_ok=True)

        try:
            base_name = os.path.basename(path)
            new_path = os.path.join(dest, base_name)
            counter = 1
            while os.path.exists(new_path):
                name_parts = os.path.splitext(base_name)
                new_path = os.path.join(dest, f"{name_parts[0]}_{counter}{name_parts[1]}")
                counter += 1
            
            os.rename(path, new_path)
        except Exception as e:
            print(f"\nファイルの移動中にエラーが発生: {path} -> {new_path}: {e}")
            continue
        
        # 一行で進捗表示
        if i % 10 == 0 or i == total:
            sys.stdout.write(f"\r分類中: {i}/{total} ({(i / total) * 100:.2f}%)")
            sys.stdout.flush()

    remove_empty_dirs(img_dir)
    print(f"\n分類完了。{output_dir}/ フォルダ内を確認してください。")


if __name__ == "__main__":
    main()

