# image_classifier.py

import numpy as np
import hashlib
from collections import Counter, defaultdict
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image # サイズ分類に必要

# clip_feature_extractorから必要な定義をインポート
from clip_feature_extractor import KEYWORDS_FOR_CLASSIFICATION, KEYWORDS_FOR_KMEANS_NAMING

KEYWORD_CLASSIFICATION_THRESHOLD = 0.25 # 定数をここに移動

class ImageClassifier:
    def __init__(self, feature_extractor=None):
        # CLIPFeatureExtractorのインスタンスを受け取る
        self.feature_extractor = feature_extractor
        if feature_extractor:
            self.keyword_features_np = feature_extractor.get_keyword_features()
            self.keyword_categories = feature_extractor.get_keyword_categories()
            self.kmeans_naming_text_features_np = feature_extractor.get_kmeans_naming_features()
        else:
            # feature_extractorがない場合のフォールバック（K-Meansのみ、または事前抽出特徴量を使う場合）
            self.keyword_features_np = None
            self.keyword_categories = []
            self.kmeans_naming_text_features_np = None

    def find_best_k(self, features_np, k_range=range(10, 21)):
        """K-Meansの最適なクラスタ数Kを見つけます。"""
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
        """K-Meansクラスタに人間が読める名前を割り当てます。"""
        if self.kmeans_naming_text_features_np is None:
            raise RuntimeError("K-Means命名用特徴量がロードされていません。")

        cluster_names = {}
        for i in range(kmeans_model.n_clusters):
            # クラスタセンターを基準に命名
            cluster_center_feature = kmeans_model.cluster_centers_[i]
            normalized_cluster_center = cluster_center_feature / np.linalg.norm(cluster_center_feature)
            similarities = np.dot(normalized_cluster_center, self.kmeans_naming_text_features_np.T)
            sorted_indices = np.argsort(similarities)[::-1]
            best_keyword_1 = KEYWORDS_FOR_KMEANS_NAMING[sorted_indices[0]]
            best_keyword_2 = KEYWORDS_FOR_KMEANS_NAMING[sorted_indices[1]]
            
            # 代表キーワードから不要な接頭辞を除去
            base_name_1 = best_keyword_1.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
            base_name_2 = best_keyword_2.replace("a picture of ", "").replace("an ", "").replace("a ", "").strip()
            
            # クラスタの一意なIDを生成
            center_hash = hashlib.sha256(cluster_center_feature.tobytes()).hexdigest()
            cluster_names[i] = (
                f"{base_name_1.replace(' ', '_')}_{base_name_2.replace(' ', '_')}_id{center_hash[:8]}"
            )
        return cluster_names

    def run_hybrid_classification(self, image_paths, args_k, pre_extracted_features_map=None):
        """
        キーワードとK-Meansを組み合わせたハイブリッド分類を実行します。
        pre_extracted_features_map: {original_index: feature_array} の辞書
        """
        print("\nハイブリッド分類 (キーワード & K-Means) を実行中...")
        if self.feature_extractor is None and pre_extracted_features_map is None:
            raise ValueError("特徴量抽出器がないか、事前抽出された特徴量が提供されていません。")

        # 特徴量の取得（新規抽出 or 事前抽出）
        if pre_extracted_features_map is None:
            # 新規抽出の場合、image_pathsから特徴量を抽出
            features_all_np, processed_indices_all = self.feature_extractor.extract_features_from_paths(image_paths)
            processed_image_features_map = {idx: features_all_np[i_feat] for i_feat, idx in enumerate(processed_indices_all)}
        else:
            # 事前抽出された特徴量を使用
            # pre_extracted_features_mapは元のimage_pathsのインデックスと特徴量のマッピングを持つ
            # ここでは features_all_np は map の値、processed_indices_all は map のキー となる
            processed_image_features_map = pre_extracted_features_map
            features_all_np = np.array(list(pre_extracted_features_map.values()))
            processed_indices_all = list(pre_extracted_features_map.keys())


        if len(features_all_np) == 0:
            print("処理できる画像がありません。終了します。", file=sys.stderr)
            return np.array([]), {}, {}

        # 最終的な結果を格納するための配列
        final_labels = np.full(len(image_paths), None, dtype=object)
        final_cluster_names = {} # 最終的な表示名マッピング
        
        unknown_label_key = "Unknown_Error_System_Key" 
        final_cluster_names[unknown_label_key] = "Unknown_Error"

        print("\nキーワードに最も近い画像を分類中...")
        # キーワードで分類された画像の、processed_indices_all におけるインデックスのマスク
        classified_by_keyword_mask = np.zeros(len(features_all_np), dtype=bool) 
        
        # Keyword featuresがロードされているか確認
        if self.keyword_features_np is None:
            raise RuntimeError("分類キーワードの特徴量がロードされていません。CLIPFeatureExtractorを適切に初期化してください。")

        for i_feat, feat in enumerate(features_all_np):
            original_img_idx = processed_indices_all[i_feat] # processed_indices_allは元のimage_pathsのインデックス
            
            normalized_image_feat = feat / np.linalg.norm(feat)
            similarities = np.dot(normalized_image_feat, self.keyword_features_np.T)
            
            best_category_idx = np.argmax(similarities)
            best_similarity = similarities[best_category_idx]
            best_category_name = self.keyword_categories[best_category_idx]

            if best_similarity >= KEYWORD_CLASSIFICATION_THRESHOLD:
                # タグ名は "KW_カテゴリ名" の形式で保存し、表示名は "カテゴリ名" とする
                tag_name_for_db = f"KW_{best_category_name.replace(' ', '_').replace('/', '_').replace('\\', '_')}"
                final_labels[original_img_idx] = tag_name_for_db
                final_cluster_names[tag_name_for_db] = best_category_name # 表示用の名前
                classified_by_keyword_mask[i_feat] = True
            
            if (i_feat + 1) % 10 == 0 or (i_feat + 1) == len(features_all_np):
                sys.stdout.write(f"\r進捗 (キーワード分類): {i_feat+1}/{len(features_all_np)}")
                sys.stdout.flush()
        print("\nキーワード分類完了。")

        # キーワードで分類されなかった画像の特徴量と元のインデックスを取得
        unclassified_features_np = features_all_np[~classified_by_keyword_mask]
        unclassified_original_indices = [original_idx for i, original_idx in enumerate(processed_indices_all) if not classified_by_keyword_mask[i]]

        if len(unclassified_features_np) > 0:
            print(f"\nキーワード分類されなかった画像 ({len(unclassified_features_np)}枚) をK-Meansでクラスタリング中...")

            # K-Meansのクラスタ数決定ロジック
            min_k_val = max(2, int(len(unclassified_features_np) / 1000))
            max_k_val = int(len(unclassified_features_np) / 100)
            if min_k_val < 8: min_k_val = 8 # 最小値を8に変更（K-Meansクラスタリングが意味を持つように）
            if max_k_val <= min_k_val: max_k_val = min_k_val + 4 # 範囲が狭すぎる場合は調整
            if max_k_val > 30: max_k_val = 30 # 最大値を30に制限

            print(f"→ K-Meansクラスタ数の範囲: {min_k_val} 〜 {max_k_val}")
            
            best_k_for_kmeans = args_k if args_k is not None else self.find_best_k(unclassified_features_np, k_range=range(min_k_val, max_k_val))
            print(f"→ K-Meansのクラスタ数: {best_k_for_kmeans}")

            if best_k_for_kmeans > 0 and len(unclassified_features_np) >= best_k_for_kmeans:
                kmeans = KMeans(n_clusters=best_k_for_kmeans, random_state=0, n_init='auto')
                kmeans_labels_unclassified = kmeans.fit_predict(unclassified_features_np)
                
                kmeans_cluster_naming_map = self.describe_kmeans_clusters(
                    kmeans_labels_unclassified, 
                    kmeans, 
                    unclassified_features_np
                )

                for i, kmeans_label in enumerate(kmeans_labels_unclassified):
                    original_img_idx = unclassified_original_indices[i]
                    # タグ名は KMeans_クラスタ名_IDハッシュ の形式でデータベースに保存
                    tag_name_for_db = f"KMeans_{kmeans_cluster_naming_map[kmeans_label]}"
                    # 表示名は IDハッシュを除いた部分
                    display_name = kmeans_cluster_naming_map[kmeans_label].split('_id')[0].replace('_', ' ')
                    
                    final_labels[original_img_idx] = tag_name_for_db
                    final_cluster_names[tag_name_for_db] = display_name
            else:
                print("警告: 残りの画像数が少なすぎるため、K-Meansクラスタリングは行われませんでした。", file=sys.stderr)
        
        # どの分類にも属さなかった画像にUnknown_Error_System_Keyを割り当てる
        for i in range(len(final_labels)):
            if final_labels[i] is None:
                final_labels[i] = unknown_label_key

        return final_labels, final_cluster_names, processed_image_features_map # processed_image_features_map を返す

    def run_kmeans_only_classification(self, image_paths, args_k, pre_extracted_features_map=None):
        """K-Meansのみでクラスタリングを実行します。"""
        print("\nK-Meansのみでクラスタリング中...")
        if self.feature_extractor is None and pre_extracted_features_map is None:
            raise ValueError("特徴量抽出器がないか、事前抽出された特徴量が提供されていません。")
            
        # 特徴量の取得（新規抽出 or 事前抽出）
        if pre_extracted_features_map is None:
            features_np, processed_indices_all = self.feature_extractor.extract_features_from_paths(image_paths)
            processed_image_features_map = {idx: features_np[i_feat] for i_feat, idx in enumerate(processed_indices_all)}
        else:
            processed_image_features_map = pre_extracted_features_map
            features_np = np.array(list(pre_extracted_features_map.values()))
            processed_indices_all = list(pre_extracted_features_map.keys())

        if len(features_np) == 0:
            print("処理できる画像がありません。終了します。", file=sys.stderr)
            return np.array([]), {}, {}

        min_k_val = max(2, int(len(features_np) / 1000))
        max_k_val = int(len(features_np) / 100)
        if min_k_val < 8: min_k_val = 8
        if max_k_val <= min_k_val: max_k_val = min_k_val + 4
        if max_k_val > 30: max_k_val = 30
        
        print(f"→ K-Meansクラスタ数の範囲: {min_k_val} 〜 {max_k_val}")
        
        best_k = args_k if args_k is not None else self.find_best_k(features_np, k_range=range(min_k_val, max_k_val))
        print(f"→ K-Meansのクラスタ数: {best_k}")

        final_labels = np.full(len(image_paths), None, dtype=object)
        final_cluster_names = {}
        
        unknown_label_key = "Unknown_Error_System_Key" 
        final_cluster_names[unknown_label_key] = "Unknown_Error"

        if best_k > 0 and len(features_np) >= best_k:
            kmeans = KMeans(n_clusters=best_k, random_state=0, n_init='auto')
            kmeans_labels = kmeans.fit_predict(features_np)
            
            kmeans_cluster_naming_map = self.describe_kmeans_clusters(
                kmeans_labels, 
                kmeans, 
                features_np
            )

            for i_feat, kmeans_label in enumerate(kmeans_labels):
                original_img_idx = processed_indices_all[i_feat]
                tag_name_for_db = f"KMeans_{kmeans_cluster_naming_map[kmeans_label]}"
                display_name = kmeans_cluster_naming_map[kmeans_label].split('_id')[0].replace('_', ' ')
                
                final_labels[original_img_idx] = tag_name_for_db
                final_cluster_names[tag_name_for_db] = display_name
        else:
            print("警告: 画像数が少なすぎるため、K-Meansクラスタリングは行われませんでした。", file=sys.stderr)
        
        for i in range(len(final_labels)):
            if final_labels[i] is None:
                final_labels[i] = unknown_label_key

        return final_labels, final_cluster_names, processed_image_features_map

    def assign_fixed_size_labels(self, image_paths):
        """画像のサイズに基づいてタグを割り当てます。"""
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

                labels[i] = folder_name # サイズ分類では直接フォルダ名をラベルとする
            except Exception as e:
                print(f"\n警告: 画像サイズ処理エラー {path}: {e}。'{unknown_folder_name}'に割り当てます。", file=sys.stderr)
                labels[i] = unknown_folder_name
                
            if (i + 1) % 10 == 0 or (i + 1) == len(size_info):
                sys.stdout.write(f"\r進捗 (サイズ分類): {i+1}/{len(size_info)} ({(i+1) / len(size_info) * 100:.2f}%)")
                sys.stdout.flush()

        if None in labels:
            print("致命的なエラー: 未割り当てのラベルが存在します。処理を中断します。", file=sys.stderr)
            sys.exit(1)

        # サイズ分類ではfeaturesは抽出されないので空の辞書を返す
        return np.array(labels, dtype=object), cluster_names, {} 

# テスト用
if __name__ == '__main__':
    # ダミー特徴量とパスを作成 (テスト用)
    dummy_features = np.random.rand(50, 512)
    dummy_image_paths = [f"path/to/image_{i}.jpg" for i in range(50)]
    dummy_pre_extracted_map = {i: dummy_features[i] for i in range(50)}

    # CLIPFeatureExtractorのモック、または実際のインスタンス
    class MockFeatureExtractor:
        def get_keyword_features(self):
            # ダミーキーワード特徴量
            return np.random.rand(len(KEYWORDS_FOR_CLASSIFICATION), 512)
        def get_keyword_categories(self):
            return list(KEYWORDS_FOR_CLASSIFICATION.keys())
        def get_kmeans_naming_features(self):
            # ダミーK-Means命名用特徴量
            return np.random.rand(len(KEYWORDS_FOR_KMEANS_NAMING), 512)
        def extract_features_from_paths(self, paths):
            # ダミーの特徴量抽出
            feats = np.random.rand(len(paths), 512)
            indices = list(range(len(paths)))
            return feats, indices

    mock_extractor = MockFeatureExtractor()
    classifier = ImageClassifier(feature_extractor=mock_extractor)

    # ハイブリッド分類テスト
    labels, cluster_names, features_map = classifier.run_hybrid_classification(dummy_image_paths, args_k=5)
    print(f"\nハイブリッド分類ラベルの例: {labels[:5]}")
    print(f"クラスタ名の例: {list(cluster_names.items())[:5]}")
    print(f"抽出された特徴量の数: {len(features_map)}")

    # K-Meansのみ分類テスト
    labels_km, cluster_names_km, features_map_km = classifier.run_kmeans_only_classification(dummy_image_paths, args_k=5)
    print(f"\nK-Meansのみ分類ラベルの例: {labels_km[:5]}")

    # サイズ分類テスト (ダミーパスでは正確なテストは難しいが、構造確認)
    labels_size, cluster_names_size, features_map_size = classifier.assign_fixed_size_labels(dummy_image_paths)
    print(f"\nサイズ分類ラベルの例: {labels_size[:5]}")
    