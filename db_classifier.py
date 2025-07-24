import sqlite3
import os
import json
import argparse # コマンドライン引数用
import numpy as np
import io
import sys

# ImageSorterモジュールをインポート
# classify_image.py を image_sorter_module.py にリネームして、このスクリプトと同じディレクトリに置いてください
from classify_image import ImageSorter 

# --- NumPy配列 <-> BLOB変換ヘルパー関数 ---
def numpy_to_blob(arr):
    """NumPy配列をSQLite BLOBとして保存可能なバイト列に変換します。"""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def blob_to_numpy(blob_data):
    """SQLite BLOBデータからNumPy配列に変換します。"""
    if blob_data is None:
        return None
    out = io.BytesIO(blob_data)
    out.seek(0)
    return np.load(out)

# --- データベース初期化 ---
def initialize_database(db_path):
    """SQLiteデータベースを初期化し、必要なテーブルが存在することを確認します。"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # file_metadata テーブル (C++側で管理されていることを想定、user_sort_orderが存在することを確認)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                file_size INTEGER,
                checksum TEXT,
                time_created TEXT,
                time_modified TEXT,
                part_sum TEXT,
                counter_value INTEGER DEFAULT 0,
                user_sort_order INTEGER DEFAULT 0
            );
        """)
        # file_tags テーブル (タグ用)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_tags (
                file_path TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (file_path, tag),
                FOREIGN KEY (file_path) REFERENCES file_metadata(file_path)
            );
        """)
        # file_attributes テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_attributes (
                file_path TEXT,
                attr_key TEXT,
                attr_value TEXT,
                is_user_attr INTEGER, -- 0: 通常属性, 1: ユーザー定義属性
                PRIMARY KEY (file_path, attr_key),
                FOREIGN KEY (file_path) REFERENCES file_metadata (file_path)
            );
        """)

        # clip_features テーブル (CLIP埋め込み用、オプション)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clip_features (
                file_path TEXT PRIMARY KEY,
                clip_feature_blob BLOB,
                FOREIGN KEY (file_path) REFERENCES file_metadata (file_path)
            );
        """)

        conn.commit()
        print(f"データベースとテーブルが初期化/検証されました: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"データベース初期化エラー: {e}")
        if conn:
            conn.close()
        return None

# --- タグ管理 ---
def add_tag_to_file(conn, file_path, tag_name):
    """file_tagsテーブルに分類タグを追加します。"""
    cursor = conn.cursor()
    try:
        # このタグがこのファイルに既に存在するかチェック
        cursor.execute("""
            SELECT 1 FROM file_tags
            WHERE file_path = ? AND tag = ?
        """, (file_path, tag_name))
        if cursor.fetchone():
            # print(f"タグ '{tag_name}' は {file_path} に既に存在します。スキップします。")
            return True # タグが既に存在する場合は成功とみなす
        
        # 新しいタグを挿入
        cursor.execute("""
            INSERT INTO file_tags (file_path, tag)
            VALUES (?, ?)
        """, (file_path, tag_name))
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"ファイル {file_path} にタグ '{tag_name}' を追加中にエラーが発生しました: {e}")
        return False

# db_classifier.py の新しい関数として追加 (numpy_to_blob の下あたり、または classify_and_tag_images の上あたりに)
def save_clip_features_to_db(conn, processed_image_features):
    """
    辞書形式のCLIP特徴量をデータベースのclip_featuresテーブルに保存します。
    """
    if not processed_image_features:
        print("保存すべきCLIP特徴量がありません。")
        return

    print("\nデータベースにCLIP特徴量を保存中...")
    cursor = conn.cursor()
    saved_count = 0
    total_to_save = len(processed_image_features)

    for i, (file_path, feature_array) in enumerate(processed_image_features.items()):
        # numpy_to_blob を使ってNumPy配列をBLOBに変換
        feature_blob = numpy_to_blob(feature_array)
        
        try:
            # clip_features テーブルに挿入 (既に存在する場合は更新)
            cursor.execute("""
                INSERT OR REPLACE INTO clip_features (file_path, clip_feature_blob)
                VALUES (?, ?)
            """, (file_path, feature_blob))
            saved_count += 1
        except sqlite3.Error as e:
            print(f"エラー: 特徴量の保存中に問題が発生しました - {file_path}: {e}", file=sys.stderr)
            
        if (i + 1) % 50 == 0 or (i + 1) == total_to_save:
            sys.stdout.write(f"\r...進捗 (特徴量保存): {i + 1}/{total_to_save}")
            sys.stdout.flush()

    conn.commit() # ここで変更をコミット
    print(f"\nCLIP特徴量の保存とコミットが完了しました。成功: {saved_count} ファイル")

# classify_and_tag_images 関数を以下のように修正:
# --- メイン分類とタグ付けロジック ---
def classify_and_tag_images(conn, image_paths, args, sorter):
    """
    指定された引数に基づいて分類を実行し、分類名をデータベースにタグとして追加します。
    さらに、CLIP特徴量をデータベースに保存します。
    """
    print("\n画像分類とタグ付けを開始します...")
    labels = np.array([])
    cluster_names = {}
    # processed_image_features を受け取る変数を追加
    processed_image_features_dict = {} # ImageSorterから返される特徴量辞書を格納

    # 分類モードの決定
    if args.size:
        print("サイズベースの分類を使用します。")
        # size分類は特徴量を返さないため、空の辞書を代入
        labels, cluster_names, _ = sorter.run_size_classification(image_paths)
        processed_image_features_dict = {} # サイズ分類では特徴量は抽出されない
    elif args.kmeans_only:
        print("K-Meansのみの分類を使用します。")
        # ここで `_` ではなく `processed_image_features_dict` を受け取る
        labels, cluster_names, processed_image_features_dict = sorter.run_kmeans_only_classification(image_paths, args.k)
    else: # デフォルト: ハイブリッド (キーワード + K-Means) 分類
        print("ハイブリッド (キーワード + K-Means) 分類を使用します。")
        # ここで `_` ではなく `processed_image_features_dict` を受け取る
        labels, cluster_names, processed_image_features_dict = sorter.run_hybrid_classification(image_paths, args.k)

    if len(labels) == 0:
        print("ラベルが生成されませんでした。分類はスキップされます。")
        return

    # CLIP特徴量をデータベースに保存
    if processed_image_features_dict: # 特徴量が取得できた場合のみ保存
        save_clip_features_to_db(conn, {image_paths[i]: feature for i, feature in enumerate(processed_image_features_dict)})
        # processed_image_features_dict は classify_image.py の processed_image_features_for_log で、
        # これはリスト形式で返されるため、辞書形式に変換し直す必要がある。
        # ImageSorterのrun_xxx_classificationは、リスト `processed_image_features_for_log` を返すため、
        # `image_paths` と組み合わせて辞書形式に変換する。
        # 正しい変換:
        processed_image_features_map = {}
        for i, path in enumerate(image_paths):
            if i < len(processed_image_features_dict) and processed_image_features_dict[i] is not None:
                processed_image_features_map[path] = processed_image_features_dict[i]
        save_clip_features_to_db(conn, processed_image_features_map)


    print("\n分類ラベルをデータベースにタグとして適用します...")
    # ... (この後のタグ付けロジックは変更なし) ...
    tagged_count = 0
    for i, file_path in enumerate(image_paths):
        label_key = labels[i]
        
        tag_name = ""
        if label_key.startswith("KW_"):
            tag_name = label_key
        elif label_key.startswith("KMeans_"):
            # ImageSorterのdescribe_kmeans_clustersが生成するcluster_namesのキーに依存
            # run_hybrid_classificationやrun_kmeans_only_classificationで返される
            # final_cluster_namesのキーは "KMeans_名前_idHASH" のような形式になっている
            # ここでは `labels` がそのキー形式を持っているはず
            
            # `labels` に格納されているキー (例: "KMeans_mountain_view_tree_id1234abcd") から
            # `cluster_names` 辞書を使って適切なタグ名を取得
            # `cluster_names` には "KMeans_mountain_view_tree_id1234abcd" -> "mountain_view_tree_id1234abcd" が入っている
            # そのさらに前にある `KMeans_` を取り除く必要があるため、少しロジックを変更
            
            # labels[i] の形式: "KMeans_KMeans_name_idHASH" または "KMeans_name_idHASH"
            # final_cluster_names は "KMeans_name_idHASH" -> "name_idHASH" のマッピングを持つ
            
            # label_keyが "KMeans_名前_idHASH" の形式の場合
            # cluster_names には "KMeans_名前_idHASH" -> "名前_idHASH" が入っている
            # タグとして使いたいのは "名前" の部分なので、ここを少し調整する
            
            full_cluster_key_from_label = labels[i] # 例: KMeans_mountain_view_tree_id1234abcd
            
            # final_cluster_names から対応する名前を取得
            if full_cluster_key_from_label in cluster_names:
                # cluster_names[full_cluster_key_from_label] は "mountain_view_tree_id1234abcd" のような形式
                tag_name_raw = cluster_names[full_cluster_key_from_label]
                # "_id" 以降を削除してカテゴリ名のみを取得
                if '_id' in tag_name_raw:
                    tag_name = tag_name_raw.split('_id')[0]
                else:
                    tag_name = tag_name_raw # IDがない場合
            else:
                # 予期しない形式の場合のフォールバック
                print(f"警告: 予期しないKMeansラベルキー形式 '{label_key}'。フォールバック名を使用します。", file=sys.stderr)
                tag_name = f"KMeans_Cluster_{label_key.replace('KMeans_', '').replace('_id', '_')}" # KMeans_や_idを除去
        else: # サイズ分類の場合
            tag_name = cluster_names.get(label_key, "Unknown_Category_Size")


        if tag_name:
            if add_tag_to_file(conn, file_path, tag_name.strip()):
                tagged_count += 1
            
        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            sys.stdout.write(f"\r...進捗 (タグ付け): {i + 1}/{len(image_paths)}")
            sys.stdout.flush()
            
    print(f"\nタグ付け完了。成功: {tagged_count} ファイル")
    print(f"スキップ/エラー: {len(image_paths) - tagged_count} ファイル")


def classify_and_tag_images2(conn, image_paths, args, sorter):
    """
    指定された引数に基づいて分類を実行し、分類名をデータベースにタグとして追加します。
    """
    print("\n画像分類とタグ付けを開始します...")
    labels = np.array([])
    cluster_names = {}
    
    # 分類モードの決定
    if args.size:
        print("サイズベースの分類を使用します。")
        labels, cluster_names, _ = sorter.run_size_classification(image_paths)
    elif args.kmeans_only:
        print("K-Meansのみの分類を使用します。")
        labels, cluster_names, _ = sorter.run_kmeans_only_classification(image_paths, args.k)
    else: # デフォルト: ハイブリッド (キーワード + K-Means) 分類
        print("ハイブリッド (キーワード + K-Means) 分類を使用します。")
        labels, cluster_names, _ = sorter.run_hybrid_classification(image_paths, args.k)

    if len(labels) == 0:
        print("ラベルが生成されませんでした。分類はスキップされます。")
        return

    print("\n分類ラベルをデータベースにタグとして適用します...")
    tagged_count = 0
    for i, file_path in enumerate(image_paths):
        label_key = labels[i]
        
        # ImageSorterのlabel_keyから人間が読めるタグ名を抽出
        # label_keyは通常 "KW_Landscape" または "KMeans_0_mountain_view_id1234abcd" のような形式
        # 'Landscape' や 'mountain_view' の部分が欲しい
        
        tag_name = ""
        if label_key.startswith("KW_"):
            tag_name = label_key#[3:] #.replace("_", " ") # "KW_" を削除し、アンダースコアをスペースに置換
        elif label_key.startswith("KMeans_"):
            tag_name = label_key.replace("KMeans", "KM") # "KMeans_" を削除
            # 例: KMeans_0_mountain_view_id1234abcd -> "mountain_view" を抽出
            # ImageSorterのdescribe_kmeans_clustersが生成する名前の形式に依存
            # cluster_names ディクショナリから直接取得するのが最も安全
            #if label_key in cluster_names:
            #    full_cluster_name = cluster_names[label_key]
                # ハッシュID部分があれば削除 (例: "name_id1234abcd" -> "name")
                #if '_id' in full_cluster_name:
                #    tag_name = full_cluster_name.split('_id')[0]#.replace("_", " ")
                #else:
            #    tag_name = full_cluster_name#.replace("_", " ") # アンダースコアをスペースに置換
            #else:
                # 予期しない形式の場合のフォールバック
            #    print(f"警告: 予期しないKMeansラベルキー形式 '{label_key}'。フォールバック名を使用します。")
            #    tag_name = f"KMeans_Cluster_{label_key}"#.replace("_", " ")
        else:
            # その他の分類 (例: サイズ分類) の場合は、cluster_namesから直接取得
            tag_name = cluster_names.get(label_key, "Unknown_Category")#.replace("_", " ")

        if tag_name:
            if add_tag_to_file(conn, file_path, tag_name.strip()):
                tagged_count += 1
            
        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            print(f"...進捗 (タグ付け): {i + 1}/{len(image_paths)}")
            
    print(f"\nタグ付け完了。成功: {tagged_count} ファイル")
    print(f"スキップ/エラー: {len(image_paths) - tagged_count} ファイル")


# --- メイン実行 ---
def main():
    parser = argparse.ArgumentParser(
        description="CLIP特徴量を使用して画像を分類し、SQLiteにタグ付けします。",
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

    args = parser.parse_args()

    # --- テストオプション: パスをリスト表示するだけ ---
    if args.list_paths:
        conn_test = None
        try:
            conn_test = sqlite3.connect(args.db_file)
            cursor_test = conn_test.cursor()
            # user_sort_order でソートして表示
            cursor_test.execute("SELECT file_path, user_sort_order FROM file_metadata ORDER BY user_sort_order ASC")
            print("\n--- データベースからの画像パスリスト (user_sort_orderでソート済み) ---")
            for row in cursor_test.fetchall():
                print(f"パス: {row[0]}, 順序: {row[1]}")
            print("--------------------------------------------------")
        except sqlite3.Error as e:
            print(f"パスのリスト表示中にエラーが発生しました: {e}")
        finally:
            if conn_test:
                conn_test.close()
        return # パス表示後に終了

    # --- メイン処理 ---
    conn = initialize_database(args.db_file)
    if not conn:
        print("データベース接続に失敗しました。終了します。")
        return

    cursor = conn.cursor()
    try:
        # file_metadata からすべての画像ファイルのパスを取得し、user_sort_order でソート
        cursor.execute("SELECT file_path FROM file_metadata ORDER BY user_sort_order ASC")
        file_paths_from_db = [row[0] for row in cursor.fetchall()]

        # 実際の画像ファイルをフィルタリングし、--max制限を適用
        image_file_paths = []
        for p in file_paths_from_db:
            # 一般的な画像拡張子でフィルタリング
            if any(p.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']):
                if os.path.exists(p): # ファイルが実際にディスク上に存在するか確認
                    image_file_paths.append(p)
                else:
                    print(f"警告: ディスク上にファイルが見つかりません、スキップします: {p}")
            # else:
            #     print(f"スキップ: 画像ファイルではありません - {p}") # 冗長になる場合があるのでコメントアウト

        if args.max > 0 and len(image_file_paths) > args.max:
            print(f"--maxオプションにより、最初の {args.max} 枚の画像に制限します。")
            image_file_paths = image_file_paths[:args.max]

        if not image_file_paths:
            print("フィルタリング後、処理する画像ファイルが見つかりませんでした。終了します。")
            return

        print(f"\n分類とタグ付けのために {len(image_file_paths)} 個の画像ファイルを処理します。")

        # 分類タイプに基づいてImageSorterを初期化
        # CLIPベースの分類には'full'モードが必要
        # サイズ分類の場合、'none'モードの方が効率的だが、run_hybrid/kmeans_onlyで必要になるため'full'でも安全
        sorter = ImageSorter(feature_mode='full') 

        # 分類とタグ付けを実行
        classify_and_tag_images(conn, image_file_paths, args, sorter)

    except sqlite3.Error as e:
        print(f"データベース操作エラー: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    finally:
        if conn:
            conn.close()
        print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()
