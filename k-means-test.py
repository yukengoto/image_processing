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
KEYWORDS_FOR_CLASSIFICATION2 = {
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
KEYWORDS_FOR_CLASSIFICATION = {
    # 自然・風景
    "Landscape": ["an outdoor landscape", "a natural scene", "a mountain view", "a forest", "a beach", "a sunset", "a river", "a field", "a sky"],
    "Nature_Elements": ["a flower", "a tree", "a plant", "clouds", "water", "a rock", "snow", "ice"],

    # 人物関連
    "Person_Portrait": ["a close up of a person", "a portrait", "a human face", "headshot of a person", "selfie"],
    "Person_FullBody": ["a person standing", "a person walking", "a person sitting", "a person from behind", "multiple people", "a crowd"],
    "Person_Action": ["a person running", "a person jumping", "a person dancing", "a person playing sports", "people interacting"],
    
    # 都市・建築
    "Building": ["a building", "a city view", "an urban landscape", "architecture", "a house", "a skyscraper", "a bridge", "a street", "an interior of a building"],
    "Vehicle": ["a car", "a truck", "a motorcycle", "an airplane", "a train", "a boat", "a bicycle"], # 乗り物を独立カテゴリに

    # 動物
    "Animal": ["an animal", "a pet", "wildlife", "a dog", "a cat", "a bird", "a fish", "an insect"],

    # アート・デザイン
    "Art_Illustration": ["a digital art illustration", "a cartoon character", "an abstract image", "a painting", "a drawing", "a sculpture", "a graphic design", "a sketch"],
    "Text_Document": ["a text document", "a screenshot", "a document with text", "a book page", "a sign with text", "a newspaper"],

    # 物体・その他
    "Food_Drink": ["a food item", "a dish", "fruit", "vegetables", "a drink", "a meal", "dessert"], # 食べ物を独立カテゴリに
    "Object_Household": ["furniture", "a chair", "a table", "a bed", "kitchenware", "home decor"], # 家庭用品を追加
    "Object_Misc": ["a scene with many objects", "a tool", "electronics", "a bag", "clothes", "a shoe", "a socks", "a hand", "money"],
    
    # NSFW 
    "Adult_Nudity": ["a nude body", "exposed breasts", "genitals", "a naked person", "pornographic image", "explicit content", "sexual act"],
    "Adult_Implied": ["implied nudity", "suggestive pose", "lingerie", "revealing clothing", "buttocks", "cleavage"],
    
    # その他視覚的特徴
    "Abstract_Pattern": ["an abstract pattern", "a texture", "a blurry image", "a background", "a design"], # 抽象的なものやパターン
    "Black_White": ["a black and white photo", "monochrome image"],
    "Vibrant_Color": ["a vibrant colorful image", "a colorful abstract"],
    "Blurred_Image": ["a blurred image", "out of focus background"],
    "Watermark_Overlay": ["an image with a watermark", "text overlay on image"], # ウォーターマークなども識別
    "Other_Visual": ["a strange image", "a distorted image", "a placeholder image", "an empty image"] # どうしても分類できないもの
}

KEYWORDS_FOR_KMEANS_NAMING = [item for sublist in KEYWORDS_FOR_CLASSIFICATION.values() for item in sublist]

KEYWORD_CLASSIFICATION_THRESHOLD = 0.25

FEATURE_LOG_FILENAME = "image_features_log.npy"

# --- 2. ヘルパー関数群 ---

def get_image_paths(img_dirs, recursive=False):
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
    for current_dir, dirs, files in os.walk(root_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(current_dir)
                print(f"空フォルダを削除: {current_dir}")
            except OSError:
                continue

# --- 3. ImageSorter クラス定義 ---

class ImageSorter:
    def __init__(self, clip_model_name='ViT-B-32', pretrained_dataset='laion2b_s34b_b79k', feature_mode='full'):
        # feature_mode: 'full' (CLIPロードと特徴量抽出), 'load_only' (CLIPロードのみ、特徴量抽出はしない), 'none' (CLIPロードしない)
        self.device = "cuda" if "cuda" in sys.modules and torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.clip_model_name = clip_model_name
        self.pretrained_dataset = pretrained_dataset
        self.clip_tokenizer = None
        self.feature_mode = feature_mode

        self.keyword_classification_map = {}
        self.keyword_categories = []
        self.keyword_features_np = None
        self.kmeans_naming_text_features_np = None
        
        if self.feature_mode != 'none': # CLIP関連の処理が必要な場合のみロードを試みる
            self._ensure_clip_loaded()

    def _ensure_clip_loaded(self):
        """CLIPモデルがロードされていることを確認し、必要ならロードします。"""
        if self.model is None:
            if self.feature_mode == 'load_only':
                print(f"CLIPモデル ({self.clip_model_name}, {self.pretrained_dataset}) をロード中 (特徴量抽出はスキップ)...")
            else: # feature_mode == 'full'
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


    def extract_features(self, image_paths, pre_extracted_features=None):
        """
        画像パスのリストからCLIP特徴量を抽出、または事前抽出された特徴量を利用します。
        """
        if pre_extracted_features is not None:
            print("事前抽出された特徴量を使用します。")
            features = [item['feature'] for item in pre_extracted_features]
            processed_indices = list(range(len(features)))
            return np.array(features), processed_indices
        
        self._ensure_clip_loaded()

        features = []
        processed_indices = []
        
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
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best_score = -1
        best_k = 2
        if len(features_np) < 2: return 1
        if len(features_np) < k_range.stop: k_range = range(k_range.start, max(2, len(features_np) + 1))
        if k_range.start >= len(features_np): return max(1, len(features_np) // 2)
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                labels = kmeans.fit_predict(features_np)
                if len(set(labels)) < 2 or len(set(labels)) > len(features_np): continue
                score = silhouette_score(features_np, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"\nKMeans (k={k}) error: {e}", file=sys.stderr)
                continue
        return best_k if best_k > 0 else max(2, len(features_np) // 5)

    def describe_kmeans_clusters(self, kmeans_labels, kmeans_model, features_for_kmeans):
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
            best_keyword_1 = KEYWORDS_FOR_KMEANS_NAMING[sorted_indices[0]]
            best_keyword_2 = KEYWORDS_FOR_KMEANS_NAMING[sorted_indices[1]]
            base_name_1 = best_keyword_1.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
            base_name_2 = best_keyword_2.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
            center_hash = hashlib.sha256(cluster_center_feature.tobytes()).hexdigest()
            cluster_names[i] = (
                f"{base_name_1.replace(' ', '_')}_{base_name_2.replace(' ', '_')}_id{center_hash[:8]}"
            )
        return cluster_names
    
    def run_hybrid_classification(self, image_paths, args_k, pre_extracted_features=None):
        from sklearn.cluster import KMeans
        import torch
        print("\nハイブリッド分類 (キーワード & K-Means) を実行中...")
        
        features_all_np, processed_indices_all = self.extract_features(image_paths, pre_extracted_features)
        if len(features_all_np) == 0:
            print("処理できる画像がありません。終了します。", file=sys.stderr)
            return np.array([]), {}, []

        original_idx_map = {idx_in_features: original_img_idx for idx_in_features, original_img_idx in enumerate(processed_indices_all)}

        final_labels = np.full(len(image_paths), None, dtype=object)
        final_cluster_names = {}
        processed_image_features_for_log = [None] * len(image_paths) 
        
        unknown_label_key = "Unknown_Error_System_Key" 
        final_cluster_names[unknown_label_key] = "Unknown_Error"

        print("\nキーワードに最も近い画像を分類中...")
        classified_by_keyword_mask = np.zeros(len(features_all_np), dtype=bool) 
        
        for i, feat in enumerate(features_all_np):
            original_img_idx = original_idx_map[i]
            processed_image_features_for_log[original_img_idx] = feat 
            
            normalized_image_feat = feat / np.linalg.norm(feat)
            similarities = np.dot(normalized_image_feat, self.keyword_features_np.T)
            
            best_category_idx = np.argmax(similarities)
            best_similarity = similarities[best_category_idx]
            best_category_name = self.keyword_categories[best_category_idx]

            if best_similarity >= KEYWORD_CLASSIFICATION_THRESHOLD:
                clean_category_name = best_category_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                label_key = f"KW_{clean_category_name}" 
                
                if label_key not in final_cluster_names:
                    final_cluster_names[label_key] = label_key 
                
                final_labels[original_img_idx] = label_key
                classified_by_keyword_mask[i] = True
            
            if (i + 1) % 10 == 0 or (i + 1) == len(features_all_np):
                sys.stdout.write(f"\r進捗 (キーワード分類): {i+1}/{len(features_all_np)}")
                sys.stdout.flush()
        print("\nキーワード分類完了。")

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

        for i in range(len(final_labels)):
            if final_labels[i] is None:
                final_labels[i] = unknown_label_key

        return final_labels, final_cluster_names, processed_image_features_for_log

    def run_kmeans_only_classification(self, image_paths, args_k, pre_extracted_features=None):
        from sklearn.cluster import KMeans
        import torch
        print("\nK-Meansのみでクラスタリング中...")
        
        features_np, processed_indices = self.extract_features(image_paths, pre_extracted_features)
        if len(features_np) == 0:
            print("処理できる画像がありません。終了します。", file=sys.stderr)
            return np.array([]), {}, []

        original_idx_map = {idx_in_features: original_img_idx for idx_in_features, original_img_idx in enumerate(processed_indices)}

        min_k_val = int(len(features_np) / 1000)
        max_k_val = int(len(features_np) / 100)
        if min_k_val < 8: min_k_val = 10
        elif min_k_val > 30: max_k_val = 30
        if max_k_val > 30: max_k_val = 30
        elif max_k_val <= min_k_val + 4: max_k_val = min_k_val + 4
        
        print(f"→ K-Meansクラスタ数の範囲: {min_k_val} 〜 {max_k_val}")
        
        best_k = args_k if args_k is not None else self.find_best_k(features_np, k_range=range(min_k_val, max_k_val))
        print(f"→ K-Meansのクラスタ数: {best_k}")

        final_labels = np.full(len(image_paths), None, dtype=object)
        final_cluster_names = {}
        processed_image_features_for_log = [None] * len(image_paths) 
        
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
                processed_image_features_for_log[original_img_idx] = features_np[i] 
                kmeans_cluster_key = f"KMeans_{kmeans_label}_{kmeans_cluster_names[kmeans_label]}"
                
                if kmeans_cluster_key not in final_cluster_names:
                    final_cluster_names[kmeans_cluster_key] = kmeans_cluster_names[kmeans_label]
                
                final_labels[original_img_idx] = kmeans_cluster_key
        else:
            print("警告: 画像数が少なすぎるため、K-Meansクラスタリングは行われませんでした。", file=sys.stderr)
        
        for i in range(len(final_labels)):
            if final_labels[i] is None:
                final_labels[i] = unknown_label_key

        return final_labels, final_cluster_names, processed_image_features_for_log

    def run_size_classification(self, image_paths):
        print("\n固定閾値でサイズ分類中...")
        labels, cluster_names = self.assign_fixed_size_labels(image_paths)
        if len(set(labels)) <= 1 and "Unknown_Error_Size" not in cluster_names.values():
             print("警告: 固定閾値による分類が効果的ではありませんでした（すべての画像が同じカテゴリかもしれません）。", file=sys.stderr)
        return labels, cluster_names, [None] * len(image_paths)

    def organize_images(self, image_paths, labels, cluster_names, processed_image_features, output_dir):
        print(f"\n画像をカテゴリごとに移動中... 出力先: {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        total = len(image_paths)
        feature_log_data = []
        for i, path in enumerate(image_paths, 0):
            label_key = labels[i] 
            name = cluster_names.get(label_key, "Unknown_Category_Fallback")
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
                if processed_image_features[i] is not None:
                    feature_log_data.append({
                        'filepath': new_path,
                        'feature': processed_image_features[i]
                    })
            except Exception as e:
                print(f"\nファイルの移動中にエラーが発生: {path} -> {new_path}: {e}", file=sys.stderr)
                if processed_image_features[i] is not None:
                    feature_log_data.append({
                        'filepath': new_path,
                        'feature': processed_image_features[i]
                    })
                continue
            if (i + 1) % 10 == 0 or (i + 1) == total:
                sys.stdout.write(f"\r分類中: {i + 1}/{total} ({(i + 1) / total * 100:.2f}%)")
                sys.stdout.flush()
        print(f"\n分類完了。{output_dir}/ フォルダ内を確認してください。")
        if feature_log_data:
            np.save(FEATURE_LOG_FILENAME, np.array(feature_log_data, dtype=object))
            print(f"特徴量データを '{FEATURE_LOG_FILENAME}' に保存しました。")
        else:
            print("保存する特徴量データがありませんでした。")


# --- 4. メイン処理 ---

def main():
    parser = argparse.ArgumentParser(
        description="画像分類ツール:画像ファイルをフォルダに振り分ける。デフォルトはハイブリッド分類。\n"
                    "特徴量ファイルからの再分類 (--reclassify) もサポート。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 通常のディレクトリ指定（再分類モードでは使用しない）
    parser.add_argument('img_dirs', metavar='DIR', type=str, nargs='*', default=['.'],
                        help='画像フォルダのパス（複数指定可、デフォルト: カレントディレクトリ）\n'
                             '再分類モードではこの引数は無視されます。\n'
                             '例: python script.py ./my_photos /mnt/data/images')
    
    parser.add_argument('--max', type=int, default=100000,
                        help='読み込む/処理する画像の最大枚数（デフォルト: 100000）')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='サブフォルダも再帰的に検索する場合は指定（通常モードのみ）')
    parser.add_argument('--size', action='store_true',
                        help='サイズ分類する場合は指定')
    parser.add_argument('--kmeans-only', '--km', action='store_true', dest='kmeans_only',
                        help='K-Meansのみで分類する場合は指定（CLIP特徴量を使用）')
    parser.add_argument('--k', '-k', type=int, default=None,
                        help='K-Meansのクラスタ数を手動で指定する場合（デフォルト: 自動判定）')
    
    # 再分類用オプション
    parser.add_argument('--reclassify', action='store_true',
                        help='保存された特徴量データから画像を再分類します。このオプションを使用すると、DIR引数は無視されます。')
    parser.add_argument('--feature-file', type=str, default=FEATURE_LOG_FILENAME,
                        help=f'再分類時に読み込む特徴量データファイル（デフォルト: {FEATURE_LOG_FILENAME}）。\n'
                             f'--reclassify と併用します。')
    parser.add_argument('--output-suffix', type=str, default='_reclassified',
                        help='再分類時の出力フォルダ名に付加するサフィックス（デフォルト: _reclassified）。\n'
                             '--reclassify と併用します。')
    
    args = parser.parse_args()

    labels = np.array([])
    cluster_names = {}
    processed_image_features = [] # 特徴量ログ用のデータ
    output_dir = "" 
    
    if args.reclassify:
        # --- 再分類モード ---
        feature_file = args.feature_file
        print(f"再分類モード: 特徴量ファイル '{feature_file}' を読み込み中...")
        if not os.path.exists(feature_file):
            print(f"エラー: 特徴量ファイル '{feature_file}' が見つかりません。", file=sys.stderr)
            return

        try:
            loaded_data = np.load(feature_file, allow_pickle=True)
        except Exception as e:
            print(f"エラー: 特徴量ファイルの読み込みに失敗しました: {e}", file=sys.stderr)
            return

        if not loaded_data.size > 0:
            print("エラー: 読み込んだ特徴量データが空です。", file=sys.stderr)
            return

        print(f"読み込んだ特徴量データ数: {len(loaded_data)}")

        # 既存のパスと特徴量を準備
        image_paths = [item['filepath'] for item in loaded_data]
        features_from_log = [item['feature'] for item in loaded_data]

        if len(image_paths) > args.max:
            print(f"画像が多すぎるため、最初の{args.max}枚のみを使用します。")
            image_paths = image_paths[:args.max]
            features_from_log = features_from_log[:args.max]
        
        # サイズ分類は特徴量を必要としないため、再分類モードでは非対応（意味がないため）
        if args.size:
            print("警告: --reclassify モードでは --size 分類はサポートされていません。通常モードで実行してください。", file=sys.stderr)
            return

        # ImageSorterを、CLIPモデルをロードしてキーワード特徴量を準備するが、画像からの特徴量抽出はしないモードで初期化
        sorter = ImageSorter(feature_mode='load_only')

        if args.kmeans_only:
            labels, cluster_names, processed_image_features = sorter.run_kmeans_only_classification(
                image_paths, args.k, pre_extracted_features=loaded_data # loaded_dataをそのまま渡す
            )
            base_output_dir = "clusters_by_kmeans_only"
        else: # デフォルト動作: ハイブリッドモード
            labels, cluster_names, processed_image_features = sorter.run_hybrid_classification(
                image_paths, args.k, pre_extracted_features=loaded_data # loaded_dataをそのまま渡す
            )
            base_output_dir = "clusters_by_hybrid_clip"

        output_dir = f"{base_output_dir}{args.output_suffix}"

    else:
        # --- 通常の分類モード (特徴量抽出から開始) ---
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

        # ImageSorterをフルモードで初期化（CLIPモデルロードと特徴量抽出）
        sorter = ImageSorter(feature_mode='full') 

        if args.size:
            labels, cluster_names, processed_image_features = sorter.run_size_classification(image_paths)
            output_dir = "clusters_by_size"
        elif args.kmeans_only:
            labels, cluster_names, processed_image_features = sorter.run_kmeans_only_classification(image_paths, args.k)
            output_dir = "clusters_by_kmeans_only"
        else: # デフォルト動作: ハイブリッドモード
            labels, cluster_names, processed_image_features = sorter.run_hybrid_classification(image_paths, args.k)
            output_dir = "clusters_by_hybrid_clip"
        
        # 通常モードの場合、処理元ディレクトリの空ディレクトリを削除
        for img_dir in img_dirs:
            remove_empty_dirs(img_dir)

    # 画像の移動と特徴量ログの保存（両モードで共通）
    if len(labels) > 0 and len(cluster_names) > 0:
        sorter.organize_images(image_paths, labels, cluster_names, processed_image_features, output_dir)
    else:
        print("分類結果が空のため、画像の移動はスキップされました。", file=sys.stderr)

    # 出力先ディレクトリ内の空ファイルを削除(reclassfyモードで以前の結果が移動されて空ディレクトリが残る可能性があるため)
    if args.reclassify and output_dir:
        remove_empty_dirs(output_dir)

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()
    

