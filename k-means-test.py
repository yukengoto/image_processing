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
    processed_indices = [] # 正常に処理された画像のインデックス
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for i, path in enumerate(image_paths): # インデックスも取得
        try:
            img = Image.open(path).convert("RGB")
            image = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image).cpu().numpy()[0]
            features.append(feat)
            processed_indices.append(i) # 正常処理されたインデックスを記録
            
            if (i + 1) % 10 == 0 or (i + 1) == len(image_paths): # プログレスバーの表示を修正
                sys.stdout.write(f"\r進捗 (特徴量抽出): {i + 1}/{len(image_paths)}")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n中断されました。")
            raise
        except Exception as e:
            print(f"\nエラー発生（{path}）：{e} - この画像をスキップします。")
            # エラーが発生した場合は、その画像を処理済みとして扱わないが、
            # 後でラベルを生成する際には対応する位置を考慮する必要がある。
            # ここでは features には追加しないまま続行し、後で処理
            continue # このままcontinueで良い

    if not features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        exit(1)

    features_np = np.array(features)
    return features_np, processed_indices # 正常に処理されたインデックスも返す

# 固定閾値でサイズ分類を行う関数
def assign_fixed_size_labels(image_paths):
    labels = []
    cluster_names = {}

    # 閾値定義
    THRESHOLD_AREA_SMALL = 250 * 250
    THRESHOLD_AREA_MEDIUM = 1100 * 900
    THRESHOLD_ASPECT_PORTRAIT = 0.95
    THRESHOLD_ASPECT_SQUARE = 1.05

    # クラスタ名マッピング用
    label_map = {}
    current_label_id = 0

    # 画像サイズごとの出現回数をカウントし、サイズ情報も保存
    from collections import Counter, defaultdict
    size_counter = Counter()
    size_to_paths = defaultdict(list)
    size_info = []  # (path, w, h) のリスト

    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as img:
                w, h = img.size
                size_counter[(w, h)] += 1
                size_to_paths[(w, h)].append(path)
                size_info.append((path, w, h))
        except Exception:
            size_info.append((path, None, None))
            continue
        if i % 10 == 0 or i == len(image_paths) - 1:
            sys.stdout.write(f"\rSearching for popular sizes: {i}/{len(image_paths)} ({(i / len(image_paths)) * 100:.2f}%)")
            sys.stdout.flush()

    # 多数派サイズの閾値（例: 画像数の2%以上かつ5枚以上）
    min_count = max(5, int(len(image_paths) * 0.02))
    if min_count > 100: min_count = 20  # 上限を設定しておくと良い
    popular_sizes = {size for size, count in size_counter.items() if count >= min_count}

    print()

    for i, (path, w, h) in enumerate(size_info):
        try:
            if w is None or h is None:
                raise ValueError("Image size not available")
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

            if i % 10 == 0 or i == len(size_info) - 1:
                sys.stdout.write(f"\r進捗: {i}/{len(size_info)} ({(i / len(size_info)) * 100:.2f}%)")
                sys.stdout.flush()

        except Exception as e:
            print(f"\nWarning: Could not process {path} due to error: {e}. Skipping.")
            labels.append(-1)

    return np.array(labels), cluster_names


# 最適クラスタ数を自動判定（シルエットスコア）
def find_best_k(features_np, k_range=range(10, 21)):
    best_score = -1
    best_k = 2
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10) # n_initを整数に修正
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
# describe_clusters 関数（変更案）
def describe_clusters(labels, kmeans, clip_model, clip_tokenizer):
    from sklearn.metrics import pairwise_distances_argmin_min
    import torch
    import hashlib
    #
    cluster_names = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

    # 事前に定義するキーワードリスト (例: ユーザーが期待するカテゴリや一般的な画像内容)
    # これらをCLIPでエンコードし、画像特徴量と比較する
    # あなたの画像群の内容に合わせて、このリストを調整してください
    keywords = [
        "a picture of a person",
        "an outdoor landscape",
        "a building",
        "an animal",
        "a close up of a face",
        "a scene with many objects",
        "a digital art illustration",
        "a cartoon character",
        "a professional photo",
        "a blurred background",
        "a text document",
        "a screenshot",
        "a natural scene",
        "a city view",
        "a food item",
        "a vehicle",
        "a plant or flower",
        "a abstract image",
        "a black and white photo",
        "a vibrant colorful image",
        # さらに具体的なキーワードを追加する
        "an anime character",
        "a game screenshot",
        "a CD album cover",
        "a travel destination",
        "a glamour photo",
        "a portrait",
        "a group of people",
        "a landscape with water",
        "a mountain view",
        "an indoor scene",
        "a foot",
        "a leg",
        "a sole",
        "a toe",
        "a heel",
        "a vagina",
        "a penis",
        "kissing",
        "two or more women",
        "a foot and a penis",
        "an ugly face",
        "a breast",
        "a nipple",
        "a butt",
        "a face",
        "a mouth",
        "a mouth and penis",
        "a face and penis",
        "a girl",
        "a shoe",
        "a socks",
        "a hand",
        "a female body",
        "a landscape"
    ]
    
    # キーワードのCLIPテキスト特徴量を事前に計算
    print("\nキーワードのCLIPテキスト特徴量を計算中...")
    text_features = []
    with torch.no_grad():
        for kw in keywords:
            text = clip_tokenizer(kw).to(device)
            text_feat = clip_model.encode_text(text).cpu().numpy()[0]
            text_features.append(text_feat / np.linalg.norm(text_feat)) # 正規化
    text_features_np = np.array(text_features)
    print("キーワード特徴量の計算完了。")

    for i in range(kmeans.n_clusters):
        indices = [j for j, label in enumerate(labels) if label == i]
        if not indices:
            cluster_names[i] = f"empty_cluster_{i}"
            continue

        cluster_center_feature = kmeans.cluster_centers_[i]
        
        # クラスタ中心と最も類似するキーワードを見つける
        # cosine similarity (内積) を計算するために正規化が必要
        normalized_cluster_center = cluster_center_feature / np.linalg.norm(cluster_center_feature)
        
        similarities = np.dot(normalized_cluster_center, text_features_np.T)
        best_keyword_idx = np.argmax(similarities)
        best_keyword_similarity = similarities[best_keyword_idx]
        # キーワードから先頭の "a picture of ", "an " などを除去してクラスタ名にする
        base_name = keywords[best_keyword_idx].replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()

        center_hash = hashlib.sha256(cluster_center_feature.tobytes()).hexdigest()
        # フォルダ名として適切な形に整形 (例: スペースをアンダースコアに)
        # 類似度も表示すると、ラベリングの信頼性が分かる
        # ハッシュ値の一部をフォルダ名として使用（長すぎると不便なので、先頭8文字程度）
        cluster_names[i] = f"{base_name.replace(' ', '_')}_sim{best_keyword_similarity:.2f}_{center_hash[:8]}"
    return cluster_names

def describe_clusters_simple(features_np, image_paths, labels, kmeans):
    import hashlib
    cluster_names = {}
    for i in range(kmeans.n_clusters):
        cluster_center = kmeans.cluster_centers_[i]
        
        # クラスタ中心のベクトルをバイト列に変換し、SHA256ハッシュを計算
        # 高精度で一意性を保つため、float64のバイト表現を使用
        center_hash = hashlib.sha256(cluster_center.tobytes()).hexdigest()
        
        # ハッシュ値の一部をフォルダ名として使用（長すぎると不便なので、先頭10文字程度）
        cluster_names[i] = f"cluster_{center_hash[:10]}" 
        
        # オプション：より短くしたい場合は、より短いハッシュアルゴリズムを使うか、
        # さらに短く切り詰めることも可能ですが、衝突の可能性が高まります
        # 例：f"cl_{center_hash[:6]}"
        
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
    parser = argparse.ArgumentParser(description="画像クラスタリング:画像ファイルをフォルダに振り分ける")
    parser.add_argument('--max', type=int, default=100000,
                        help='読み込む画像の最大枚数（デフォルト: 100000）')
    parser.add_argument('--dir', type=str, default='.',
                        help='画像フォルダのパス（デフォルト: カレントディレクトリ）')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='サブフォルダも再帰的に検索する場合は指定')
    #parser.add_argument('--use-picture', action='store_true', dest='check_picture', # デフォルトをFalseに変更
    #                    help='CLIP特徴を取得して分類する場合は指定') # オプション名を変更
    parser.add_argument('--size', action='store_true', dest='check_size_rule', # 新しいオプション
                        help='サイズ分類する場合は指定')
    parser.add_argument('--k', '-k', type=int, default=None,
                        help='K-Meansのクラスタ数を手動で指定する場合（デフォルト: 自動判定）')
    args = parser.parse_args()
    img_dir = args.dir
    n_max = args.max
    recursive = args.recursive
    check_size_rule = args.check_size_rule  # 固定閾値のサイズ分類を使うか
    check_picture = not check_size_rule # どちらか一方を使う

    print("Searching for images in the directory...")
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
        # extract_featuresは正常に処理された画像のfeaturesと、その元のインデックスを返す
        features_np, processed_indices = extract_features(image_paths, model, preprocess)
        
        # K-Meansの入力とする画像パスを再構築
        processed_image_paths_for_clustering = [image_paths[idx] for idx in processed_indices]
        # original_indices_for_clustering = processed_indices # 後の Unknown 処理のために保持
        npics = len(processed_image_paths_for_clustering)
        print(f"\n特徴量の形状: {features_np.shape} (処理対象の画像数: {npics})")

        if args.k is None:
            print("\n最適なクラスタ数を探索中...")
            min_k = int(npics/1000) # 最小クラスタ数を画像数の1/500に設定
            max_k = int(npics/100) # 最大クラスタ数を画像数の1/100に設定
            if min_k < 8: min_k = 10 # 最小クラスタ数を10に制限
            elif min_k > 30: min_k = 30 # 最小クラスタ数を30に制限
            if max_k > 30: max_k = 30
            elif max_k <= min_k + 4: max_k = min_k + 4
            print(f"→ クラスタ数の範囲: {min_k} 〜 {max_k}")
            best_k = find_best_k(features_np, k_range=range(min_k, max_k))
            print(f"→ 最適なクラスタ数は {best_k}")
        else:
            best_k = args.k
            print(f"クラスタ数 (手動指定): {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
        kmeans_labels = kmeans.fit_predict(features_np)
        # kmeans_cluster_names = describe_clusters_simple(features_np, processed_image_paths_for_clustering, kmeans_labels, kmeans)
        # describe_clusters に clip_model と clip_tokenizer を渡す
        kmeans_cluster_names = describe_clusters(
            kmeans_labels, 
            kmeans,
            model,          # CLIPモデルを渡す
            open_clip.get_tokenizer('ViT-B-32') # トークナイザーをここで取得して渡す
        )

        # 全ての画像パスに対応するlabelsリストを構築
        labels = np.full(len(image_paths), -1, dtype=int) # 全体を-1 (Unknown)で初期化
        current_unknown_label_id = max(kmeans_cluster_names.keys()) + 1 if kmeans_cluster_names else 0
        kmeans_cluster_names[current_unknown_label_id] = "Unknown_Error" # Unknownカテゴリを追加

        for idx, cluster_label in zip(processed_indices, kmeans_labels):
            labels[idx] = cluster_label # 処理された画像にはK-Meansのラベルを割り当てる
        
        # 処理されなかった画像（-1のままの画像）にUnknown_Errorラベルを割り当てる
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = current_unknown_label_id
        
        cluster_names = kmeans_cluster_names # K-Meansのクラスタ名とUnknown_Errorを追加したものを最終的なcluster_namesとする
        output_dir = "clusters_by_clip"
       
    elif check_size_rule: # 固定閾値でサイズ分類
        print("\n固定閾値でサイズ分類中...")
        labels, cluster_names = assign_fixed_size_labels(image_paths)
        if len(set(labels)) <= 1: # 分類が機能しなかった場合（すべて同じカテゴリなど）
            print("警告: 固定閾値による分類が効果的ではありませんでした。")
        output_dir = "clusters_by_size"

    else: # どちらの分類も指定されていない場合
        print("エラー: 分類方法 (--use-picture または --use-size-rule) を指定してください。")
        return

    print("\n画像をクラスタごとに移動中...")
    os.makedirs(output_dir, exist_ok=True)
    total = len(image_paths)
    for i, path in enumerate(image_paths, 0): # iは0-indexed
        label = labels[i]
        name = cluster_names.get(label, "Unknown_Category_Fallback") # 念のためデフォルトも
        
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
        
        if (i + 1) % 10 == 0 or (i + 1) == total: # プログレスバーの表示を修正
            sys.stdout.write(f"\r分類中: {i + 1}/{total} ({(i + 1) / total * 100:.2f}%)")
            sys.stdout.flush()

    remove_empty_dirs(img_dir)  # 空のディレクトリを削除

    print(f"\n分類完了。{output_dir}/ フォルダ内を確認してください。")

if __name__ == "__main__":
    main()

