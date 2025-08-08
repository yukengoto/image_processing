# image_organizer.py

import os
import sys
import argparse
import numpy as np
from collections import Counter # assign_fixed_size_labels で使用する場合
 

# モジュールのインポート
from clip_feature_extractor import CLIPFeatureExtractor
from image_classification_engine import ImageClassifier

FEATURE_LOG_FILENAME = "image_features_log.npy" # 特徴量ログのファイル名もここに移動

# --- ヘルパー関数 (既存のまま) ---
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

# --- ImageSorter クラス (名称を ImageOrganizer に変更し、役割を縮小) ---
# このクラスは主に画像のファイルシステム上の整理と、特徴量ログの管理を行う
class ImageOrganizer:
    def __init__(self):
        pass # 特徴量抽出や分類のロジックは別のクラスに移譲

    def organize_images(self, image_paths, labels, cluster_names, processed_image_features_map, output_dir):
        """
        画像をカテゴリごとに移動し、特徴量ログを保存します。
        labels: image_paths と同じインデックスのリスト（DB保存用のタグ名形式）
        cluster_names: label_key -> 表示名 のマッピング
        processed_image_features_map: {original_index: feature_array} の辞書
        """
        print(f"\n画像をカテゴリごとに移動中... 出力先: {output_dir}/")
        os.makedirs(output_dir, exist_ok=True)
        total = len(image_paths)
        feature_log_data = [] # 新しいパスと特徴量のペアを記録

        for i, path in enumerate(image_paths, 0):
            label_key = labels[i] # これはDBに保存する形式のタグ名
            
            # 実際のフォルダ名として使うのは cluster_names にマップされた表示名
            # label_keyが "KW_カテゴリ名" や "KMeans_クラスタ名_IDハッシュ" の形式の場合
            # cluster_names にはそれに対応する表示名 ("カテゴリ名" や "クラスタ名") が格納されている想定
            display_name = cluster_names.get(label_key, "Unknown_Category_Fallback")
            
            # ファイルシステムで安全な名前に変換
            safe_folder_name = display_name.replace(" ", "_").replace(":", "_").replace("/", "_").replace("\\", "_").replace("*", "_").replace("?", "_").replace('"', '_').replace("<", "_").replace(">", "_").replace("|", "_")
            
            dest = os.path.join(output_dir, safe_folder_name)
            os.makedirs(dest, exist_ok=True)
            try:
                base_name = os.path.basename(path)
                new_path = os.path.join(dest, base_name)
                counter = 1
                while os.path.exists(new_path): # ファイル名が重複する場合の処理
                    name_parts = os.path.splitext(base_name)
                    new_path = os.path.join(dest, f"{name_parts[0]}_{counter}{name_parts[1]}")
                    counter += 1
                os.rename(path, new_path)
                
                # processed_image_features_mapは original_index -> feature_array のマップ
                # ここでは new_path と feature_array を紐付けて保存
                if i in processed_image_features_map and processed_image_features_map[i] is not None:
                    feature_log_data.append({
                        'filepath': new_path,
                        'feature': processed_image_features_map[i]
                    })
            except Exception as e:
                print(f"\nファイルの移動中にエラーが発生: {path} -> {new_path}: {e}", file=sys.stderr)
                # エラー発生時でも元のパスと特徴量をログに残す場合
                if i in processed_image_features_map and processed_image_features_map[i] is not None:
                     feature_log_data.append({
                        'filepath': path, # 移動失敗時は元のパス
                        'feature': processed_image_features_map[i]
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


# --- 4. メイン処理 (classify_image.py のメインロジック) ---

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
    processed_image_features_map = {} # {original_index: feature_array}
    output_dir = "" 
    
    # ImageOrganizer を初期化
    organizer = ImageOrganizer()

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
        
        # processed_image_features_map を {original_index: feature_array} 形式で作成
        for i, item in enumerate(loaded_data):
            processed_image_features_map[i] = item['feature']

        if len(image_paths) > args.max:
            print(f"画像が多すぎるため、最初の{args.max}枚のみを使用します。")
            image_paths = image_paths[:args.max]
            # processed_image_features_map もそれに合わせて調整が必要
            processed_image_features_map = {i: processed_image_features_map[i] for i in range(args.max)}
        
        # サイズ分類は特徴量を必要としないため、再分類モードでは非対応（意味がないため）
        if args.size:
            print("警告: --reclassify モードでは --size 分類はサポートされていません。通常モードで実行してください。", file=sys.stderr)
            return

        # CLIPFeatureExtractorをロードのみのモードで初期化
        feature_extractor = CLIPFeatureExtractor() # デフォルトでモデルをロードし、キーワード特徴量も準備
        classifier = ImageClassifier(feature_extractor=feature_extractor)

        if args.kmeans_only:
            labels, cluster_names, _ = classifier.run_kmeans_only_classification(
                image_paths, args.k, pre_extracted_features_map=processed_image_features_map
            )
            base_output_dir = "clusters_by_kmeans_only"
        else: # デフォルト動作: ハイブリッドモード
            labels, cluster_names, _ = classifier.run_hybrid_classification(
                image_paths, args.k, pre_extracted_features_map=processed_image_features_map
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
        if args.size:
            feature_extractor = None # サイズ分類では不要
            classifier = ImageClassifier(feature_extractor=feature_extractor)
            labels, cluster_names, _ = classifier.assign_fixed_size_labels(image_paths)
            processed_image_features_map = {} # サイズ分類では特徴量は生成されない
            output_dir = "clusters_by_size"
        elif args.kmeans_only:
            feature_extractor = CLIPFeatureExtractor()
            classifier = ImageClassifier(feature_extractor=feature_extractor)
            labels, cluster_names, processed_image_features_map = classifier.run_kmeans_only_classification(image_paths, args.k)
            output_dir = "clusters_by_kmeans_only"
        else: # デフォルト動作: ハイブリッドモード
            feature_extractor = CLIPFeatureExtractor()
            classifier = ImageClassifier(feature_extractor=feature_extractor)
            labels, cluster_names, processed_image_features_map = classifier.run_hybrid_classification(image_paths, args.k)
            output_dir = "clusters_by_hybrid_clip"
        
        # 通常モードの場合、処理元ディレクトリの空ディレクトリを削除
        for img_dir in img_dirs:
            remove_empty_dirs(img_dir)

    # 画像の移動と特徴量ログの保存（両モードで共通）
    if len(labels) > 0 and len(cluster_names) > 0:
        organizer.organize_images(image_paths, labels, cluster_names, processed_image_features_map, output_dir)
    else:
        print("分類結果が空のため、画像の移動はスキップされました。", file=sys.stderr)

    # 出力先ディレクトリ内の空ファイルを削除(reclassifyモードで以前の結果が移動されて空ディレクトリが残る可能性があるため)
    if args.reclassify and output_dir:
        remove_empty_dirs(output_dir)

    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()
    