import os
import sys
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import open_clip
import hashlib # ハッシュ値の生成に必要

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
            continue

    if not features:
        print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。")
        exit(1)

    features_np = np.array(features)
    return features_np, processed_indices # 正常に処理されたインデックスも返す

# 固定閾値でサイズ分類を行う関数
def assign_fixed_size_labels(image_paths):
    labels = [None] * len(image_paths) # 初期化をNoneで埋める
    cluster_names = {}
    
    THRESHOLD_AREA_SMALL = 250 * 250
    THRESHOLD_AREA_MEDIUM = 1100 * 900
    
    THRESHOLD_ASPECT_PORTRAIT = 0.95
    THRESHOLD_ASPECT_SQUARE = 1.05    
    
    # クラスタ名マッピング用
    label_map = {}
    current_label_id = 0
    
    # "Unknown_Error" のためのラベルIDを事前に確保
    unknown_folder_name = "Unknown_Error"
    if unknown_folder_name not in label_map:
        label_map[unknown_folder_name] = current_label_id
        cluster_names[current_label_id] = unknown_folder_name
        current_label_id += 1

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
            # continue # エラー時もリストに追加して長さを維持

        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            sys.stdout.write(f"\rSearching for popular sizes: {i+1}/{len(image_paths)} ({(i+1) / len(image_paths) * 100:.2f}%)")
            sys.stdout.flush()

    # 多数派サイズの閾値（例: 画像数の2%以上かつ5枚以上）
    min_count = max(5, int(len(image_paths) * 0.02))
    if min_count > 100: min_count = 20 # 上限を設定しておくと良い
    popular_sizes = {size for size, count in size_counter.items() if count >= min_count}

    print() # 改行

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

            labels[i] = label_map[folder_name] # 正しい位置にラベルを格納

        except Exception as e:
            print(f"\nWarning: Could not process {path} due to error in assign_fixed_size_labels: {e}. Assigning to '{unknown_folder_name}'.")
            labels[i] = label_map[unknown_folder_name] # エラー時はUnknown_Errorカテゴリに割り当てる
            
        if (i + 1) % 10 == 0 or (i + 1) == len(size_info):
            sys.stdout.write(f"\r進捗 (サイズ分類): {i+1}/{len(size_info)} ({(i+1) / len(size_info) * 100:.2f}%)")
            sys.stdout.flush()

    # 全ての画像パスに対応するラベルが割り当てられていることを確認
    if None in labels:
        print("致命的なエラー: 未割り当てのラベルが存在します。処理を中断します。")
        sys.exit(1)

    return np.array(labels), cluster_names


# 最適クラスタ数を自動判定（シルエットスコア）
def find_best_k(features_np, k_range=range(10, 21)):
    best_score = -1
    best_k = 2
    if len(features_np) < 2: # データ点が少ない場合はクラスタリングできない
        return 1
    
    # k_rangeをデータ点数で調整
    if len(features_np) < k_range.stop:
        k_range = range(k_range.start, max(2, len(features_np))) # データ点数まで
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(features_np)
            if len(set(labels)) < 2: # クラスタが1つしかない場合はシルエットスコア計算不可
                continue
            score = silhouette_score(features_np, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"\nKMeans (k={k}) error: {e}")
            continue
    return best_k

# describe_clusters 関数をK-Meansの結果にキーワードベースの命名を行う関数として維持
def describe_clusters(kmeans_labels, kmeans_model, clip_model, clip_tokenizer, features_for_kmeans):
    cluster_names = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 以前のdescribe_clustersで使っていたキーワードリスト
    keywords = [
        "a picture of a person", "an outdoor landscape", "a building", "an animal",
        "a close up of a face", "a scene with many objects", "a digital art illustration",
        "a cartoon character", "a professional photo", "a blurred background",
        "a text document", "a screenshot", "a natural scene", "a city view",
        "a food item", "a vehicle", "a plant or flower", "a abstract image",
        "a black and white photo", "a vibrant colorful image",
        "an anime character", "a game screenshot", "a CD album cover",
        "a travel destination", "a glamour photo", "a portrait",
        "a group of people", "a landscape with water", "a mountain view",
        "an indoor scene", "a foot", "a leg", "a sole", "a toe", "a heel",
        "a vagina", "a penis", "kissing", "two or more women", "a foot and a penis",
        "an ugly face", "a breast", "a nipple", "a butt", "a face", "a mouth",
        "a mouth and penis", "a face and penis", "a girl", "a shoe", "a socks",
        "a hand", "a female body", "a landscape"
    ]
    
    print("\nキーワードのCLIPテキスト特徴量を計算中 (K-Meansクラスタ命名用)...")
    text_features = []
    with torch.no_grad():
        for kw in keywords:
            text = clip_tokenizer(kw).to(device)
            text_feat = clip_model.encode_text(text).cpu().numpy()[0]
            text_features.append(text_feat / np.linalg.norm(text_feat)) # 正規化
    text_features_np = np.array(text_features)
    print("キーワード特徴量の計算完了。")

    for i in range(kmeans_model.n_clusters):
        # features_for_kmeans からこのクラスタに属する特徴量を抽出
        indices_in_kmeans_data = np.where(kmeans_labels == i)[0]
        if not indices_in_kmeans_data.size:
            cluster_names[i] = f"empty_cluster_{i}"
            continue

        cluster_center_feature = kmeans_model.cluster_centers_[i]
        
        normalized_cluster_center = cluster_center_feature / np.linalg.norm(cluster_center_feature)
        
        similarities = np.dot(normalized_cluster_center, text_features_np.T)
        sorted_indices = np.argsort(similarities)[::-1]
        
        best_keyword_idx_1 = sorted_indices[0]
        best_keyword_idx_2 = sorted_indices[1]
        
        best_keyword_1 = keywords[best_keyword_idx_1]
        best_keyword_2 = keywords[best_keyword_idx_2]
        
        best_similarity_1 = similarities[best_keyword_idx_1]
        best_similarity_2 = similarities[best_keyword_idx_2]

        base_name_1 = best_keyword_1.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
        base_name_2 = best_keyword_2.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
        
        # クラスタ中心のハッシュ値も追加して安定性を確保
        center_hash = hashlib.sha256(cluster_center_feature.tobytes()).hexdigest()

        cluster_names[i] = (
            f"{base_name_1.replace(' ', '_')}_{base_name_2.replace(' ', '_')}_"
            f"sim{best_similarity_1:.2f}_{best_similarity_2:.2f}_id{center_hash[:8]}"
        )
        
    return cluster_names

# 画像パスを取得する関数（再帰的にディレクトリを探索）
def get_image_paths(img_dir, recursive=False):
    exts = ('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
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
    parser.add_argument('--size', action='store_true', dest='check_size_rule',
                        help='サイズ分類する場合は指定')
    parser.add_argument('--k', '-k', type=int, default=None,
                        help='K-Meansのクラスタ数を手動で指定する場合（デフォルト: 自動判定）')
    parser.add_argument('--hybrid', action='store_true', dest='use_hybrid_clip',
                        help='キーワード分類後に残りをK-Meansで分類するハイブリッドモードを使用') # 新しいオプション
    
    args = parser.parse_args()
    img_dir = args.dir
    n_max = args.max
    recursive = args.recursive
    check_size_rule = args.check_size_rule
    use_hybrid_clip = args.use_hybrid_clip

    # 少なくとも1つの分類方法が指定されているか確認
    if not (check_size_rule or use_hybrid_clip):
        print("エラー: 少なくとも1つの分類方法 (--size または --hybrid) を指定してください。")
        parser.print_help()
        return

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

    # CLIPモデルのロードはハイブリッドモードで必要
    model, _, preprocess = None, None, None
    if use_hybrid_clip:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model.eval()

    # 最終的なラベルとクラスタ名
    final_labels = np.full(len(image_paths), -1, dtype=object) # カテゴリ名文字列やK-Meansラベルを格納するためdtype=object
    final_cluster_names = {}
    
    # 未処理の画像用の"Unknown_Error"カテゴリを定義
    unknown_label_key = "Unknown_Error_System_Key" # 内部的なキーとして使う
    final_cluster_names[unknown_label_key] = "Unknown_Error" # フォルダ名

    if use_hybrid_clip:
        print("\nハイブリッド分類 (キーワード & K-Means) を実行中...")
        
        # 1. 全ての画像からCLIP特徴量を抽出
        features_all_np, processed_indices_all = extract_features(image_paths, model, preprocess)
        
        # 各画像の元のインデックスを特徴量に対応させるマップ
        original_idx_map = {idx_in_features: original_img_idx for idx_in_features, original_img_idx in enumerate(processed_indices_all)}

        # CLIPトークナイザーもここで取得
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # キーワードの定義 (describe_clusters と同じものを使用)
        keywords_for_classification = {
            "Landscape": ["an outdoor landscape", "a natural scene", "a mountain view"],
            "Person": ["a picture of a person", "a portrait", "a group of people"],
            "Building": ["a building", "a city view", "architecture"],
            "Animal": ["an animal", "a pet", "wildlife"],
            "Art_Illustration": ["a digital art illustration", "a cartoon character", "an abstract image"],
            "Text_Document": ["a text document", "a screenshot"],
            "Object_Misc": ["a scene with many objects", "a food item", "a vehicle", "a hand"],
            "Anime": ["an anime character", "a game screenshot", "a CD album cover"],
            "Other_Visual": ["a blurred background", "a black and white photo", "a vibrant colorful image", "a girl"],
            # 37
            "a4": ["foot", "barefoot", "calf", "barefoot soles", "leg", "sole", "toes", "toenails", "heel", "a foot and a penis", "penis massage by feet" "shoes", "sandals", "socks"],
            "10005": ["a vagina", "between legs",],
            "face": ["a fimale face", "a female portrait"],
            "mouth": ["mouth", "teeth", "lips", "tongue"],
            "bj": ["a mouth and penis", "a face and penis", "blow job", "oral sex"],
            "fin": ["ejaculation", "semen", "sperm"],
            "sv": ["penis massage by hand, a hand and a penis", "submissive", "obedience", "slavish"],
            "soft":["underware", "swimsuit", "bra"],
            "1011": ["little girl"],
            "1010": ["toilet", "japanese-style toilet", "using japanese-style toilet", "peeing", "excretion", "feces"],
            "fk": ["people having sex", "a girl getting fucked"],
            "multi" :["multiple girls"],
            "body" :["female body","a breast", "a nipple", "a butt"],
            "chu": ["kissing"]
        }
        
        # キーワードのCLIPテキスト特徴量を事前に計算
        keyword_features_map = {} # {カテゴリ名: そのカテゴリの代表キーワード特徴量}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for category, kw_list in keywords_for_classification.items():
            text = clip_tokenizer(kw_list[0]).to(device) # 各カテゴリの最初のキーワードを代表とする
            with torch.no_grad():
                text_feat = model.encode_text(text).cpu().numpy()[0]
            keyword_features_map[category] = text_feat / np.linalg.norm(text_feat) # 正規化
        
        keyword_categories = list(keyword_features_map.keys())
        keyword_features_np = np.array(list(keyword_features_map.values()))

        # 2. キーワードによる直接分類
        print("\nキーワードに最も近い画像を分類中...")
        classified_by_keyword_mask = np.zeros(len(features_all_np), dtype=bool) # キーワード分類されたかどうかのマスク
        keyword_classification_threshold = 0.20 # この類似度以上でキーワードに分類
        
        # ユニークなラベルIDを生成するためのカウンター
        current_unique_label_id = 0 

        for i, feat in enumerate(features_all_np):
            original_img_idx = original_idx_map[i] # 元のimage_pathsでのインデックス
            
            normalized_image_feat = feat / np.linalg.norm(feat)
            similarities = np.dot(normalized_image_feat, keyword_features_np.T)
            
            best_category_idx = np.argmax(similarities)
            best_similarity = similarities[best_category_idx]
            best_category_name = keyword_categories[best_category_idx]

            if best_similarity >= keyword_classification_threshold:
                # フォルダ名として適切な形に整形
                clean_category_name = best_category_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                label_key = f"KW_{clean_category_name}"
                # label_key = f"KW_{clean_category_name}_sim{best_similarity:.2f}"
                
                # final_labelsに割り当てるためのユニークなIDを生成
                if label_key not in final_cluster_names:
                    final_cluster_names[label_key] = label_key # フォルダ名として使う
                
                final_labels[original_img_idx] = label_key
                classified_by_keyword_mask[i] = True
            
            if (i + 1) % 10 == 0 or (i + 1) == len(features_all_np):
                sys.stdout.write(f"\r進捗 (キーワード分類): {i+1}/{len(features_all_np)}")
                sys.stdout.flush()
        print("\nキーワード分類完了。")

        # 3. キーワード分類されなかった画像をK-Meansでクラスタリング
        unclassified_features_np = features_all_np[~classified_by_keyword_mask]
        unclassified_original_indices = [original_idx_map[i] for i in np.where(~classified_by_keyword_mask)[0]]

        if len(unclassified_features_np) > 0:
            print(f"\nキーワード分類されなかった画像 ({len(unclassified_features_np)}枚) をK-Meansでクラスタリング中...")

            min_k = int(len(unclassified_features_np) / 1000)
            max_k = int(len(unclassified_features_np) / 100)
            if min_k < 8: min_k = 10
            elif min_k > 30: min_k = 30
            if max_k > 30: max_k = 30
            elif max_k <= min_k + 4: max_k = min_k + 4
            
            print(f"→ K-Meansクラスタ数の範囲: {min_k} 〜 {max_k}")
            
            best_k_for_kmeans = find_best_k(unclassified_features_np, k_range=range(min_k, max_k))
            print(f"→ K-Meansの最適なクラスタ数は {best_k_for_kmeans}")

            if best_k_for_kmeans > 0: # クラスタリング可能な場合
                kmeans = KMeans(n_clusters=best_k_for_kmeans, random_state=0, n_init='auto')
                kmeans_labels_unclassified = kmeans.fit_predict(unclassified_features_np)
                
                # K-Meansクラスタにキーワードベースの命名
                kmeans_cluster_names = describe_clusters(
                    kmeans_labels_unclassified, 
                    kmeans, 
                    model, 
                    clip_tokenizer, 
                    unclassified_features_np # features_for_kmeans を渡す
                )

                # K-Meansのラベルを最終的なラベルリストにマッピング
                # K-Meansのラベルとキーワード分類のラベルが重複しないように、ユニークなキーを生成
                for i, kmeans_label in enumerate(kmeans_labels_unclassified):
                    original_img_idx = unclassified_original_indices[i]
                    kmeans_cluster_key = f"KMeans_{kmeans_label}_{kmeans_cluster_names[kmeans_label]}" # ユニークなキー
                    
                    if kmeans_cluster_key not in final_cluster_names:
                        final_cluster_names[kmeans_cluster_key] = kmeans_cluster_names[kmeans_label]
                    
                    final_labels[original_img_idx] = kmeans_cluster_key
            else:
                print("警告: 残りの画像数が少なすぎるため、K-Meansクラスタリングは行われませんでした。")

        # 4. 処理されなかった画像 (Noneのままの画像) にUnknown_Errorラベルを割り当てる
        for i in range(len(final_labels)):
            if final_labels[i] == -1: # まだ分類されていない画像
                final_labels[i] = unknown_label_key # Unknown_Error_System_Key を割り当てる

        labels = final_labels
        cluster_names = final_cluster_names
        output_dir = "clusters_by_hybrid_clip"

    elif check_size_rule: # 固定閾値でサイズ分類
        print("\n固定閾値でサイズ分類中...")
        labels, cluster_names = assign_fixed_size_labels(image_paths)
        if len(set(labels)) <= 1 and "Unknown_Error" not in cluster_names.values():
             print("警告: 固定閾値による分類が効果的ではありませんでした（すべての画像が同じカテゴリかもしれません）。")
        output_dir = "clusters_by_size"

    else: # ここには到達しないはずだが念のため
        print("エラー: 分類方法が指定されていません。")
        return

    print("\n画像をカテゴリごとに移動中...")
    os.makedirs(output_dir, exist_ok=True)
    total = len(image_paths)
    for i, path in enumerate(image_paths, 0):
        # labelが文字列キーになる可能性があるので、getで取得
        label_key = labels[i] 
        name = cluster_names.get(label_key, "Unknown_Category_Fallback")
        
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
        
        if (i + 1) % 10 == 0 or (i + 1) == total:
            sys.stdout.write(f"\r分類中: {i + 1}/{total} ({(i + 1) / total * 100:.2f}%)")
            sys.stdout.flush()

    remove_empty_dirs(img_dir)

    print(f"\n分類完了。{output_dir}/ フォルダ内を確認してください。")

if __name__ == "__main__":
    main()

