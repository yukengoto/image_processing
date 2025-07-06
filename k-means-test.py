# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import hashlib
from collections import Counter, defaultdict


# --- 1. グローバル設定とキーワード定義 ---
# 将来的に外部ファイルから読み込むことを想定。
# 現状はここで直接定義し、修正の容易性を確保。
# キー: 最終的なフォルダ名 (カテゴリ名)
# 値: そのカテゴリを表すキーワードのリスト。最初のキーワードが代表として使われる。
# キーワードの定義
KEYWORDS_FOR_CLASSIFICATION = {
            "Landscape": ["an outdoor landscape", "a natural scene", "a mountain view"],
            #"Person": ["a picture of a person", "a portrait", "a group of people"],
            "Building": ["a building", "a city view", "architecture"],
            #"Animal": ["an animal", "a pet", "wildlife"],
            #"Art_Illustration": ["a digital art illustration", "a cartoon character", "an abstract image"],
            "Text_Document": ["a text document", "a screenshot"],
            #"Object_Misc": ["a scene with many objects", "a food item", "a vehicle", "a hand"],
            #"Anime": ["an anime character", "a game screenshot", "a CD album cover"],
            #"Other_Visual": ["a blurred background", "a black and white photo", "a vibrant colorful image"],
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
KEYWORDS_FOR_CLASSIFICATION2 = {
    "Landscape": ["an outdoor landscape", "a natural scene", "a mountain view", "a forest"],
    "Person": ["a picture of a person", "a portrait", "a group of people", "a close up of a face"],
    "Building": ["a building", "a city view", "an urban landscape", "architecture"],
    "Animal": ["an animal", "a pet", "wildlife"],
    "Art_Illustration": ["a digital art illustration", "a cartoon character", "an abstract image", "a painting"],
    "Text_Document": ["a text document", "a screenshot", "a document with text"],
    "Foot": ["a foot", "a leg", "a sole", "a toe", "a heel"],
    "Genitals": ["a vagina", "a penis", "a breast", "a nipple", "a butt"],
    "Face": ["a face", "an ugly face"],
    "Mouth": ["a mouth", "a mouth and penis", "a face and penis"],
    "Combined_Scenes": ["a foot and a penis", "kissing", "two or more women"], # 複合的なキーワード
    "Object_Misc": ["a scene with many objects", "a food item", "a vehicle", "a shoe", "a socks", "a hand"],
    "Other_Visual": ["a blurred background", "a black and white photo", "a vibrant colorful image", "a female body", "a girl"]
}

# describe_kmeans_clusters (K-Meansクラスタ命名用) で使用するキーワードのフラットリスト
# 上記の辞書のすべての値を結合して作成
KEYWORDS_FOR_KMEANS_NAMING = [item for sublist in KEYWORDS_FOR_CLASSIFICATION.values() for item in sublist]

# ハイブリッド分類におけるキーワード分類の類似度閾値
KEYWORD_CLASSIFICATION_THRESHOLD = 0.25

# 特徴量データ出力ファイル名
FEATURE_LOG_FILENAME = "image_features_log.npy"

# --- 2. ヘルパー関数群 ---

def get_image_paths(img_dirs, recursive=False):
    """
    指定されたディレクトリから画像ファイルのパスを再帰的に取得します。

    Args:
        img_dirs (list): 画像を検索するディレクトリパスのリスト。
        recursive (bool): サブディレクトリを再帰的に検索するかどうか。

    Returns:
        list: 見つかった画像ファイルのパスのリスト。
    """
    exts = ('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    image_paths = []

    for img_dir in img_dirs:
        if not os.path.isdir(img_dir):
            print(f"警告: ディレクトリ '{img_dir}' が見つかりません。スキップします。", file=sys.stderr)
            continue

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

def remove_empty_dirs(root_dir):
    """
    指定されたルートディレクトリ以下の空のディレクトリを削除します。

    Args:
        root_dir (str): 削除対象のルートディレクトリ。
    """
    for current_dir, dirs, files in os.walk(root_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(current_dir)
                print(f"空フォルダを削除: {current_dir}")
            except OSError:
                continue

# --- 3. ImageSorter クラス定義 ---

class ImageSorter:
    def __init__(self, clip_model_name='ViT-B-32', pretrained_dataset='laion2b_s34b_b79k'):
        self.device = "cuda" if "cuda" in sys.modules and torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.clip_model_name = clip_model_name
        self.pretrained_dataset = pretrained_dataset
        self.clip_tokenizer = None

        self.keyword_classification_map = {}
        self.keyword_categories = []
        self.keyword_features_np = None
        self.kmeans_naming_text_features_np = None
        
    def _ensure_clip_loaded(self):
        """CLIPモデルがロードされていることを確認し、必要ならロードします。"""
        if self.model is None:
            print(f"CLIPモデル ({self.clip_model_name}, {self.pretrained_dataset}) をロード中...")
            # 遅延インポート
            import torch
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.pretrained_dataset
            )
            self.model.eval().to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            print("CLIPモデルのロード完了。")

            self._prepare_keyword_features()

    def _prepare_keyword_features(self):
        """分類キーワードとK-Means命名用キーワードのCLIP特徴量を準備します。"""
        if self.keyword_features_np is not None:
            return # 既に準備済みならスキップ

        import torch # _ensure_clip_loadedでインポート済みのはずだが念のため

        for category, kw_list in KEYWORDS_FOR_CLASSIFICATION.items():
            text = self.clip_tokenizer(kw_list[0]).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(text).cpu().numpy()[0]
            self.keyword_classification_map[category] = text_feat / np.linalg.norm(text_feat)
        
        self.keyword_categories = list(self.keyword_classification_map.keys())
        self.keyword_features_np = np.array(list(self.keyword_classification_map.values()))
        print(f"分類キーワード {len(self.keyword_categories)}個の特徴量準備完了。")

        self.kmeans_naming_text_features = []
        with torch.no_grad():
            for kw in KEYWORDS_FOR_KMEANS_NAMING:
                text = self.clip_tokenizer(kw).to(self.device)
                text_feat = self.model.encode_text(text).cpu().numpy()[0]
                self.kmeans_naming_text_features.append(text_feat / np.linalg.norm(text_feat))
        self.kmeans_naming_text_features_np = np.array(self.kmeans_naming_text_features)
        print(f"K-Means命名用キーワード {len(KEYWORDS_FOR_KMEANS_NAMING)}個の特徴量準備完了。")


    def extract_features(self, image_paths):
        """
        画像パスのリストからCLIP特徴量を抽出し、正常に処理された画像の元のインデックスを返します。
        """
        self._ensure_clip_loaded() # CLIPモデルのロードを保証

        features = []
        processed_indices = []
        
        # 遅延インポート
        from PIL import Image
        import torch

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                image = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(image).cpu().numpy()[0]
                features.append(feat)
                processed_indices.append(i)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                    sys.stdout.write(f"\r進捗 (特徴量抽出): {i + 1}/{len(image_paths)}")
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("\n中断されました。")
                raise
            except Exception as e:
                print(f"\nエラー発生（{path}）：{e} - この画像をスキップします。", file=sys.stderr)
                continue

        if not features:
            print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。", file=sys.stderr)
            return np.array([]), []

        features_np = np.array(features)
        return features_np, processed_indices

    def assign_fixed_size_labels(self, image_paths):
        """
        画像サイズに基づいて画像を分類します。
        """
        # 遅延インポート
        from PIL import Image

        labels = [None] * len(image_paths)
        cluster_names = {}
        
        THRESHOLD_AREA_SMALL = 250 * 250
        THRESHOLD_AREA_MEDIUM = 1100 * 900
        THRESHOLD_ASPECT_PORTRAIT = 0.95
        THRESHOLD_ASPECT_SQUARE = 1.05    
        
        label_map = {}
        current_label_id = 0
        
        unknown_folder_name = "Unknown_Error_Size"
        if unknown_folder_name not in label_map:
            label_map[unknown_folder_name] = current_label_id
            cluster_names[current_label_id] = unknown_folder_name
            current_label_id += 1

        size_counter = Counter()
        size_info = []

        for i, path in enumerate(image_paths):
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    size_counter[(w, h)] += 1
                    size_info.append((path, w, h))
            except Exception:
                size_info.append((path, None, None))
            
            if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                sys.stdout.write(f"\rSearching for popular sizes: {i+1}/{len(image_paths)} ({(i+1) / len(image_paths) * 100:.2f}%)")
                sys.stdout.flush()

        min_count = max(5, int(len(image_paths) * 0.02))
        if min_count > 100: min_count = 20
        popular_sizes = {size for size, count in size_counter.items() if count >= min_count}

        print() # 改行

        for i, (path, w, h) in enumerate(size_info):
            try:
                if w is None or h is None:
                    raise ValueError("Image size not available")
                area = w * h
                aspect_ratio = w / h if h != 0 else 0

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

                labels[i] = label_map[folder_name]

            except Exception as e:
                print(f"\n警告: 画像サイズ処理エラー {path}: {e}。'{unknown_folder_name}'に割り当てます。", file=sys.stderr)
                labels[i] = label_map[unknown_folder_name]
                
            if (i + 1) % 10 == 0 or (i + 1) == len(size_info):
                sys.stdout.write(f"\r進捗 (サイズ分類): {i+1}/{len(size_info)} ({(i+1) / len(size_info) * 100:.2f}%)")
                sys.stdout.flush()

        if None in labels:
            print("致命的なエラー: 未割り当てのラベルが存在します。処理を中断します。", file=sys.stderr)
            sys.exit(1)

        return np.array(labels), cluster_names

    def find_best_k(self, features_np, k_range=range(10, 21)):
        """最適なK-Meansクラスタ数をシルエットスコアに基づいて決定します。"""
        # 遅延インポート
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_score = -1
        best_k = 2
        if len(features_np) < 2:
            return 1 
        
        if len(features_np) < k_range.stop:
            k_range = range(k_range.start, max(2, len(features_np) + 1))
        
        if k_range.start >= len(features_np):
            return max(1, len(features_np) // 2)

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(features_np)
                if len(set(labels)) < 2 or len(set(labels)) > len(features_np):
                    continue
                score = silhouette_score(features_np, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"\nKMeans (k={k}) error: {e}", file=sys.stderr)
                continue
        return best_k if best_k > 0 else max(2, len(features_np) // 5)

    def describe_kmeans_clusters(self, kmeans_labels, kmeans_model, features_for_kmeans):
        """K-Meansクラスタにキーワードベースの命名を行います。"""
        cluster_names = {}
        
        for i in range(kmeans_model.n_clusters):
            indices_in_kmeans_data = np.where(kmeans_labels == i)[0]
            if not indices_in_kmeans_data.size:
                cluster_names[i] = f"empty_cluster_{i}"
                continue

            cluster_center_feature = kmeans_model.cluster_centers_[i]
            
            normalized_cluster_center = cluster_center_feature / np.linalg.norm(cluster_center_feature)
            
            similarities = np.dot(normalized_cluster_center, self.kmeans_naming_text_features_np.T)
            sorted_indices = np.argsort(similarities)[::-1]
            
            best_keyword_idx_1 = sorted_indices[0]
            best_keyword_idx_2 = sorted_indices[1]
            
            best_keyword_1 = KEYWORDS_FOR_KMEANS_NAMING[best_keyword_idx_1]
            best_keyword_2 = KEYWORDS_FOR_KMEANS_NAMING[best_keyword_idx_2]
            
            base_name_1 = best_keyword_1.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
            base_name_2 = best_keyword_2.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
            
            center_hash = hashlib.sha256(cluster_center_feature.tobytes()).hexdigest()

            cluster_names[i] = (
                f"{base_name_1.replace(' ', '_')}_{base_name_2.replace(' ', '_')}_id{center_hash[:8]}"
            )
            
        return cluster_names

    def run_hybrid_classification(self, image_paths, args_k):
        """
        キーワード分類後に残りをK-Meansで分類するハイブリッドモードを実行します。
        """
        # 遅延インポート
        from sklearn.cluster import KMeans
        import torch

        print("\nハイブリッド分類 (キーワード & K-Means) を実行中...")
        
        # 特徴量抽出
        features_all_np, processed_indices_all = self.extract_features(image_paths)
        if len(features_all_np) == 0:
            print("処理できる画像がありません。終了します。", file=sys.stderr)
            return np.array([]), {}, [] # 特徴量ログのために空のリストも返す

        original_idx_map = {idx_in_features: original_img_idx for idx_in_features, original_img_idx in enumerate(processed_indices_all)}

        final_labels = np.full(len(image_paths), None, dtype=object)
        final_cluster_names = {}
        processed_image_features = [None] * len(image_paths) # 特徴量ログ用のリスト
        
        # 未処理の画像用の"Unknown_Error"カテゴリを定義
        unknown_label_key = "Unknown_Error_System_Key" 
        final_cluster_names[unknown_label_key] = "Unknown_Error"

        # 1. キーワードによる直接分類
        print("\nキーワードに最も近い画像を分類中...")
        classified_by_keyword_mask = np.zeros(len(features_all_np), dtype=bool) 
        
        for i, feat in enumerate(features_all_np):
            original_img_idx = original_idx_map[i]
            processed_image_features[original_img_idx] = feat # 特徴量を保存
            
            normalized_image_feat = feat / np.linalg.norm(feat)
            similarities = np.dot(normalized_image_feat, self.keyword_features_np.T)
            
            best_category_idx = np.argmax(similarities)
            best_similarity = similarities[best_category_idx]
            best_category_name = self.keyword_categories[best_category_idx]

            if best_similarity >= KEYWORD_CLASSIFICATION_THRESHOLD:
                clean_category_name = best_category_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                # フォルダ名に類似度を含めない
                label_key = f"KW_{clean_category_name}" 
                
                if label_key not in final_cluster_names:
                    final_cluster_names[label_key] = label_key 
                
                final_labels[original_img_idx] = label_key
                classified_by_keyword_mask[i] = True
            
            if (i + 1) % 10 == 0 or (i + 1) == len(features_all_np):
                sys.stdout.write(f"\r進捗 (キーワード分類): {i+1}/{len(features_all_np)}")
                sys.stdout.flush()
        print("\nキーワード分類完了。")

        # 2. キーワード分類されなかった画像をK-Meansでクラスタリング
        unclassified_features_np = features_all_np[~classified_by_keyword_mask]
        unclassified_original_indices = [original_idx_map[i] for i in np.where(~classified_by_keyword_mask)[0]]

        if len(unclassified_features_np) > 0:
            print(f"\nキーワード分類されなかった画像 ({len(unclassified_features_np)}枚) をK-Meansでクラスタリング中...")

            min_k_val = int(len(unclassified_features_np) / 1000)
            max_k_val = int(len(unclassified_features_np) / 100)
            if min_k_val < 8: min_k_val = 10
            elif min_k_val > 30: min_k_val = 30
            if max_k_val > 30: max_k_val = 30
            elif max_k_val <= min_k_val + 4: max_k_val = min_k_val + 4
            
            print(f"→ K-Meansクラスタ数の範囲: {min_k_val} 〜 {max_k_val}")
            
            best_k_for_kmeans = args_k if args_k is not None else self.find_best_k(unclassified_features_np, k_range=range(min_k_val, max_k_val))
            print(f"→ K-Meansのクラスタ数: {best_k_for_kmeans}")

            if best_k_for_kmeans > 0:
                kmeans = KMeans(n_clusters=best_k_for_kmeans, random_state=0, n_init='auto')
                kmeans_labels_unclassified = kmeans.fit_predict(unclassified_features_np)
                
                kmeans_cluster_names = self.describe_kmeans_clusters(
                    kmeans_labels_unclassified, 
                    kmeans, 
                    unclassified_features_np
                )

                for i, kmeans_label in enumerate(kmeans_labels_unclassified):
                    original_img_idx = unclassified_original_indices[i]
                    kmeans_cluster_key = f"KMeans_{kmeans_cluster_names[kmeans_label]}"
                    
                    if kmeans_cluster_key not in final_cluster_names:
                        final_cluster_names[kmeans_cluster_key] = kmeans_cluster_names[kmeans_label]
                    
                    final_labels[original_img_idx] = kmeans_cluster_key
            else:
                print("警告: 残りの画像数が少なすぎるため、K-Meansクラスタリングは行われませんでした。", file=sys.stderr)

        # 3. 未分類の画像をUnknown_Errorに割り当てる (Noneのままの画像)
        for i in range(len(final_labels)):
            if final_labels[i] is None:
                final_labels[i] = unknown_label_key

        return final_labels, final_cluster_names, processed_image_features

    def run_kmeans_only_classification(self, image_paths, args_k):
        """
        K-Meansのみで画像を分類します。
        """
        # 遅延インポート
        from sklearn.cluster import KMeans
        import torch

        print("\nK-Meansのみでクラスタリング中...")
        
        features_np, processed_indices = self.extract_features(image_paths)
        if len(features_np) == 0:
            print("処理できる画像がありません。終了します。", file=sys.stderr)
            return np.array([]), {}, []

        original_idx_map = {idx_in_features: original_img_idx for idx_in_features, original_img_idx in enumerate(processed_indices)}

        min_k_val = int(len(features_np) / 1000)
        max_k_val = int(len(features_np) / 100)
        if min_k_val < 8: min_k_val = 10
        elif min_k_val > 30: min_k_val = 30
        if max_k_val > 30: max_k_val = 30
        elif max_k_val <= min_k_val + 4: max_k_val = min_k_val + 4
        
        print(f"→ K-Meansクラスタ数の範囲: {min_k_val} 〜 {max_k_val}")
        
        best_k = args_k if args_k is not None else self.find_best_k(features_np, k_range=range(min_k_val, max_k_val))
        print(f"→ K-Meansのクラスタ数: {best_k}")

        final_labels = np.full(len(image_paths), None, dtype=object)
        final_cluster_names = {}
        processed_image_features = [None] * len(image_paths) # 特徴量ログ用のリスト
        
        unknown_label_key = "Unknown_Error_System_Key" 
        final_cluster_names[unknown_label_key] = "Unknown_Error"

        if best_k > 0:
            kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
            kmeans_labels = kmeans.fit_predict(features_np)
            
            kmeans_cluster_names = self.describe_kmeans_clusters(
                kmeans_labels, 
                kmeans, 
                features_np
            )

            for i, kmeans_label in enumerate(kmeans_labels):
                original_img_idx = original_idx_map[i]
                processed_image_features[original_img_idx] = features_np[i] # 特徴量を保存
                kmeans_cluster_key = f"KMeans_{kmeans_cluster_names[kmeans_label]}"
                
                if kmeans_cluster_key not in final_cluster_names:
                    final_cluster_names[kmeans_cluster_key] = kmeans_cluster_names[kmeans_label]
                
                final_labels[original_img_idx] = kmeans_cluster_key
        else:
            print("警告: 画像数が少なすぎるため、K-Meansクラスタリングは行われませんでした。", file=sys.stderr)
        
        # 未分類の画像をUnknown_Errorに割り当てる
        for i in range(len(final_labels)):
            if final_labels[i] is None:
                final_labels[i] = unknown_label_key

        return final_labels, final_cluster_names, processed_image_features

    def run_size_classification(self, image_paths):
        """
        画像サイズに基づいて画像を分類します。
        このモードではCLIP特徴量を使用しないため、特徴量ログは空になります。
        """
        print("\n固定閾値でサイズ分類中...")
        labels, cluster_names = self.assign_fixed_size_labels(image_paths)
        if len(set(labels)) <= 1 and "Unknown_Error_Size" not in cluster_names.values():
             print("警告: 固定閾値による分類が効果的ではありませんでした（すべての画像が同じカテゴリかもしれません）。", file=sys.stderr)
        
        # サイズ分類では特徴量を抽出しないため、空のリストを返す
        return labels, cluster_names, [None] * len(image_paths)


    def organize_images(self, image_paths, labels, cluster_names, processed_image_features, output_dir):
        """
        分類結果に基づいて画像をフォルダに移動し、特徴量データをファイル出力します。
        """
        print(f"\n画像をカテゴリごとに移動中... 出力先: {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        total = len(image_paths)

        # 特徴量ログデータを格納するリスト
        feature_log_data = []

        for i, path in enumerate(image_paths, 0):
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

                # 移動後の新しいパスと特徴量をログに追加
                if processed_image_features[i] is not None:
                    feature_log_data.append({
                        'filepath': new_path,
                        'feature': processed_image_features[i]
                    })

            except Exception as e:
                print(f"\nファイルの移動中にエラーが発生: {path} -> {new_path}: {e}", file=sys.stderr)
                # エラー発生時でも、特徴量が取得できていればログに追加
                if processed_image_features[i] is not None:
                    feature_log_data.append({
                        'filepath': new_path, # 移動に失敗しても、意図した新しいパスを記録
                        'feature': processed_image_features[i]
                    })
                continue
            
            if (i + 1) % 10 == 0 or (i + 1) == total:
                sys.stdout.write(f"\r分類中: {i + 1}/{total} ({(i + 1) / total * 100:.2f}%)")
                sys.stdout.flush()

        print(f"\n分類完了。{output_dir}/ フォルダ内を確認してください。")

        # 特徴量データをファイルに出力
        if feature_log_data:
            np.save(FEATURE_LOG_FILENAME, np.array(feature_log_data, dtype=object))
            print(f"特徴量データを '{FEATURE_LOG_FILENAME}' に保存しました。")
        else:
            print("保存する特徴量データがありませんでした。")


# --- 4. メイン処理 ---

def main():
    parser = argparse.ArgumentParser(
        description="画像分類ツール:画像ファイルをフォルダに振り分ける。デフォルトはハイブリッド分類。",
        formatter_class=argparse.RawTextHelpFormatter # ヘルプメッセージの整形を保持
    )
    parser.add_argument('img_dirs', metavar='DIR', type=str, nargs='*', default=['.'],
                        help='画像フォルダのパス（複数指定可、デフォルト: カレントディレクトリ）\n例: python script.py ./my_photos /mnt/data/images')
    parser.add_argument('--max', type=int, default=100000,
                        help='読み込む画像の最大枚数（デフォルト: 100000）')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='サブフォルダも再帰的に検索する場合は指定')
    parser.add_argument('--size', action='store_true',
                        help='サイズ分類する場合は指定')
    parser.add_argument('--kmeans-only', action='store_true', dest='kmeans_only',
                        help='K-Meansのみで分類する場合は指定（CLIP特徴量を使用）')
    parser.add_argument('--k', '-k', type=int, default=None,
                        help='K-Meansのクラスタ数を手動で指定する場合（デフォルト: 自動判定）')
    
    args = parser.parse_args()
    img_dirs = args.img_dirs
    n_max = args.max
    recursive = args.recursive
    
    print("Searching for images in the specified directories...")
    sys.stdout.flush()
    image_paths = get_image_paths(img_dirs, recursive)

    if not image_paths:
        print("エラー: 指定されたフォルダに画像が見つかりませんでした。", file=sys.stderr)
        return

    print(f"画像フォルダにある画像の総数: {len(image_paths)}")

    if len(image_paths) > n_max:
        print(f"画像が多すぎるため、最初の{n_max}枚のみを使用します。")
        image_paths = image_paths[:n_max]

    sorter = ImageSorter() # ImageSorterインスタンスを作成

    labels = np.array([])
    cluster_names = {}
    processed_image_features = [] # 特徴量ログ用のデータ
    output_dir = "" 

    if args.size:
        labels, cluster_names, processed_image_features = sorter.run_size_classification(image_paths)
        output_dir = "clusters_by_size"
    elif args.kmeans_only:
        labels, cluster_names, processed_image_features = sorter.run_kmeans_only_classification(image_paths, args.k)
        output_dir = "clusters_by_kmeans_only"
    else: # デフォルト動作: ハイブリッドモード
        labels, cluster_names, processed_image_features = sorter.run_hybrid_classification(image_paths, args.k)
        output_dir = "clusters_by_hybrid_clip"

    # 画像の移動と特徴量ログの保存
    sorter.organize_images(image_paths, labels, cluster_names, processed_image_features, output_dir)

    # 処理元ディレクトリの空ディレクトリを削除
    for img_dir in img_dirs:
        remove_empty_dirs(img_dir)

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()
