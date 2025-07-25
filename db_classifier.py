# db_classifier.py (修正後)

import os
import argparse
import sys

# 新しく作成したモジュールをインポート
from db_manager import DBManager, blob_to_numpy
from clip_feature_extractor import CLIPFeatureExtractor
from image_classifier import ImageClassifier


# --- ヘルパー関数 (既存のまま、またはget_image_pathsは削除しても良い) ---
def get_image_paths_from_fs(img_dirs, recursive=False, max_count=None):
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
                    if file.lower().endswith(exts):
                        image_paths.append(os.path.join(img_dir, file))
    
    # 実際に存在するファイルのみにフィルタリング
    filtered_paths = []
    for p in image_paths:
        if os.path.exists(p):
            filtered_paths.append(p)
        else:
            print(f"警告: ディスク上にファイルが見つかりません、スキップします: {p}", file=sys.stderr)

    if max_count and max_count > 0 and len(filtered_paths) > max_count:
        print(f"--maxオプションにより、最初の {max_count} 枚の画像に制限します。")
        filtered_paths = filtered_paths[:max_count]
            
    return filtered_paths


# --- メイン分類とタグ付けロジック (大幅修正) ---
def classify_and_tag_images(db_manager: DBManager, image_paths: list, args, classifier: ImageClassifier):
    """
    指定された引数に基づいて分類を実行し、分類名をデータベースにタグとして追加します。
    さらに、CLIP特徴量をデータベースに保存します。
    """
    print("\n画像分類とタグ付けを開始します...")
    
    labels = []
    cluster_names = {}
    processed_image_features_map = {} # {original_index: feature_array}

    # 分類モードの決定と実行
    if args.size:
        print("サイズベースの分類を使用します。")
        labels, cluster_names, processed_image_features_map = classifier.assign_fixed_size_labels(image_paths)
    elif args.kmeans_only:
        print("K-Meansのみの分類を使用します。")
        labels, cluster_names, processed_image_features_map = classifier.run_kmeans_only_classification(
            image_paths, args.k, 
            pre_extracted_features_map={i: feat for i, feat in enumerate(processed_image_features_map.values())} # 既存のfeatures_mapは空の可能性があるので注意
        )
    else: # デフォルト: ハイブリッド (キーワード + K-Means) 分類
        print("ハイブリッド (キーワード + K-Means) 分類を使用します。")
        labels, cluster_names, processed_image_features_map = classifier.run_hybrid_classification(
            image_paths, args.k, 
            pre_extracted_features_map={i: feat for i, feat in enumerate(processed_image_features_map.values())} # 既存のfeatures_mapは空の可能性があるので注意
        )
    
    if len(labels) == 0:
        print("ラベルが生成されませんでした。分類はスキップされます。")
        return

    # CLIP特徴量をデータベースに保存
    if processed_image_features_map: # 特徴量が取得できた場合のみ保存
        # processed_image_features_map は既に {original_image_path_index: feature_array} 形式なので、
        # file_path に変換して渡す
        features_to_save_by_path = {}
        for original_idx, feature_array in processed_image_features_map.items():
            features_to_save_by_path[image_paths[original_idx]] = feature_array
        
        db_manager.save_clip_features_to_db(features_to_save_by_path)

    print("\n分類ラベルをデータベースにタグとして適用します...")
    tagged_count = 0
    total_images = len(image_paths)

    for i, file_path in enumerate(image_paths):
        label_key = labels[i]
        
        # ImageClassifierが返すクラスタ名 (表示用) を取得
        # cluster_names には "KMeans_KMeans_name_idHASH" -> "name idHASH" のようなマッピングが入っている
        # ここでは db_classifier.py がタグを付けるので、label_key を直接タグ名とする
        # 表示名が必要なら cluster_names[label_key] を使う
        tag_name_for_db = label_key
        
        if tag_name_for_db:
            if db_manager.add_tag_to_file(file_path, tag_name_for_db.strip()):
                tagged_count += 1
            
        if (i + 1) % 10 == 0 or (i + 1) == total_images:
            sys.stdout.write(f"\r...進捗 (タグ付け): {i + 1}/{total_images}")
            sys.stdout.flush()
            
    print(f"\nタグ付け完了。成功: {tagged_count} ファイル")
    print(f"スキップ/エラー: {total_images - tagged_count} ファイル")


# --- メイン実行 ---
def main():
    parser = argparse.ArgumentParser(
        description="CLIP特徴量を使用して画像を分類し、SQLiteにタグ付けします。\n"
                    "db_classifier.py は、既存のデータベースと画像を連携させ、タグ付けと特徴量保存を行います。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('db_file', type=str,
                        help='SQLiteデータベースファイルのパス (例: C:/Users/yuken/OneDrive/ドキュメント/FileDirector/HISTORY.txt.db)')
    
    parser.add_argument('--max', type=int, default=100000,
                        help='読み込む/処理する画像の最大枚数（デフォルト: 100000）')
    parser.add_argument('--size', action='store_true',
                        help='サイズベースの分類を使用します。')
    parser.add_argument('--kmeans-only', '--km', action='store_true', dest='kmeans_only',
                        help='K-Meansのみの分類を使用します（CLIP特徴量が必要です）。')
    parser.add_argument('--k', '-k', type=int, default=None,
                        help='K-Meansのクラスタ数を手動で指定する場合（デフォルト: 自動判定）。')
    
    parser.add_argument('--list-paths', action='store_true',
                        help='データベースから画像ファイルのパスのみをリスト表示して終了します（テスト用）。')

    # 新規追加: --extract-features-only (タグ付けせず特徴量だけをDBに保存)
    parser.add_argument('--extract-features-only', action='store_true',
                        help='画像を分類せず、CLIP特徴量のみを抽出しデータベースに保存します。')

    # 新規追加: --load-features-only (DBから特徴量をロードして分類に使用)
    parser.add_argument('--load-features-only', action='store_true',
                        help='データベースに保存されたCLIP特徴量をロードし、それを使って画像を分類します（特徴量抽出をスキップ）。')

    args = parser.parse_args()

    # DB Managerの初期化
    db_manager = DBManager(args.db_file)
    if not db_manager.connect():
        print("データベース接続に失敗しました。終了します。")
        return

    try:
        # --- テストオプション: パスをリスト表示するだけ ---
        if args.list_paths:
            print("\n--- データベースからの画像パスリスト (user_sort_orderでソート済み) ---")
            for path in db_manager.get_image_paths_from_db():
                print(f"パス: {path}") # user_sort_orderはdb_managerから直接取得できないので省略
            print("--------------------------------------------------")
            return # パス表示後に終了

        # DBから処理対象の画像パスを取得
        all_image_paths_from_db = db_manager.get_image_paths_from_db()

        # 実際にファイルシステムに存在する画像パスをフィルタリング
        # (get_image_paths_from_fsは不要になったので、直接フィルタリング)
        image_file_paths = []
        for p in all_image_paths_from_db:
            if any(p.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                if os.path.exists(p):
                    image_file_paths.append(p)
                else:
                    print(f"警告: ディスク上にファイルが見つかりません、スキップします: {p}", file=sys.stderr)

        if args.max > 0 and len(image_file_paths) > args.max:
            print(f"--maxオプションにより、最初の {args.max} 枚の画像に制限します。")
            image_file_paths = image_file_paths[:args.max]

        if not image_file_paths:
            print("フィルタリング後、処理する画像ファイルが見つかりませんでした。終了します。")
            return

        print(f"\n処理対象の画像ファイル数: {len(image_file_paths)}")

        # CLIPFeatureExtractorとImageClassifierの初期化
        # --size の場合はCLIPモデル不要なので、feature_extractor=None
        if args.size:
            feature_extractor = None
            classifier = ImageClassifier(feature_extractor=feature_extractor)
        else:
            feature_extractor = CLIPFeatureExtractor()
            classifier = ImageClassifier(feature_extractor=feature_extractor)

        # --extract-features-only モード
        if args.extract_features_only:
            if args.size:
                print("警告: --extract-features-only と --size は併用できません。CLIP特徴量はサイズ分類では生成されません。", file=sys.stderr)
                return
            print("CLIP特徴量のみを抽出し、データベースに保存します。")
            features_np, processed_indices = feature_extractor.extract_features_from_paths(image_file_paths)
            
            features_to_save_by_path = {}
            for i_feat, original_idx in enumerate(processed_indices):
                features_to_save_by_path[image_file_paths[original_idx]] = features_np[i_feat]
            
            db_manager.save_clip_features_to_db(features_to_save_by_path)
            print("特徴量の抽出と保存が完了しました。")
            return

        # --load-features-only モード
        pre_extracted_features_map = None
        if args.load_features_only:
            if args.size:
                print("警告: --load-features-only と --size は併用できません。", file=sys.stderr)
                return
            print("データベースから既存のCLIP特徴量をロードします。")
            pre_extracted_features_map = db_manager.get_clip_features_from_db(image_file_paths)
            if not pre_extracted_features_map:
                print("データベースに既存の特徴量が見つかりませんでした。通常の分類を行います。", file=sys.stderr)
                pre_extracted_features_map = None # 通常分類にフォールバック
            else:
                # ロードした特徴量の中から、image_file_paths に対応するものを filtered_processed_features_map に格納
                # run_hybrid_classification/run_kmeans_only_classification に渡す形式に合わせる
                # {original_index: feature_array} の形式に変換
                temp_features = {}
                for original_idx, path in enumerate(image_file_paths):
                    if path in pre_extracted_features_map:
                        temp_features[original_idx] = pre_extracted_features_map[path]
                
                # temp_features が空の場合も考慮
                if not temp_features:
                    print("警告: ロードされた特徴量と画像パスが一致しませんでした。通常の分類を行います。", file=sys.stderr)
                    pre_extracted_features_map = None
                else:
                    pre_extracted_features_map = temp_features
                
                print(f"{len(pre_extracted_features_map)} 個の特徴量をデータベースからロードしました。")

        # ここで classify_and_tag_images を呼び出す際に、
        # 事前抽出された特徴量があればそれを渡す
        if pre_extracted_features_map:
            # 事前抽出特徴量がある場合は、classifierに直接渡すためのラッパー関数を定義するか、
            # classify_and_tag_images の引数として渡す
            # 既存のclassify_and_tag_imagesのロジックを修正して対応
            
            # --- ここを修正 ---
            # classify_and_tag_images に直接 pre_extracted_features_map を渡すのではなく、
            # classifier のメソッド呼び出し時に渡すように、classify_and_tag_images 自体を修正する
            labels, cluster_names, result_features_map = None, None, None
            if args.size:
                labels, cluster_names, result_features_map = classifier.assign_fixed_size_labels(image_file_paths)
            elif args.kmeans_only:
                labels, cluster_names, result_features_map = classifier.run_kmeans_only_classification(image_file_paths, args.k, pre_extracted_features_map=pre_extracted_features_map)
            else:
                labels, cluster_names, result_features_map = classifier.run_hybrid_classification(image_file_paths, args.k, pre_extracted_features_map=pre_extracted_features_map)

            # 結果に基づいてDBに保存
            if labels is not None and len(labels) > 0:
                print("\n分類ラベルをデータベースにタグとして適用します...")
                tagged_count = 0
                for i, file_path in enumerate(image_file_paths):
                    tag_name_for_db = labels[i] # labelsはすでにDB保存用の形式
                    if tag_name_for_db:
                        if db_manager.add_tag_to_file(file_path, tag_name_for_db.strip()):
                            tagged_count += 1
                        
                    if (i + 1) % 10 == 0 or (i + 1) == len(image_file_paths):
                        sys.stdout.write(f"\r...進捗 (タグ付け): {i + 1}/{len(image_file_paths)}")
                        sys.stdout.flush()
                print(f"\nタグ付け完了。成功: {tagged_count} ファイル")
                print(f"スキップ/エラー: {len(image_file_paths) - tagged_count} ファイル")
                
                # 特徴量も保存
                if result_features_map:
                    features_to_save_by_path = {}
                    for original_idx, feature_array in result_features_map.items():
                        features_to_save_by_path[image_file_paths[original_idx]] = feature_array
                    db_manager.save_clip_features_to_db(features_to_save_by_path)

        else: # --load-features-only が指定されていない、または特徴量がロードできなかった場合
            # 通常の分類とタグ付けを実行 (特徴量抽出から開始)
            classify_and_tag_images(db_manager, image_file_paths, args, classifier)

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # エラートレースバックも表示

    finally:
        db_manager.close()
        print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()
