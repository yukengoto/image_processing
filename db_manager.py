import sqlite3
#import os
#import json # for recent_db_paths.json
import numpy as np
import io
import sys

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
    try:
        return np.load(out)
    except Exception as e:
        print(f"警告: BLOBデータをNumPy配列に変換できませんでした: {e}", file=sys.stderr)
        return None

# --- データベース初期化とDBManagerクラス ---
class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.initialize_database()

    def connect(self):
        """データベースに接続します。"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row # カラム名をキーとして結果を取得できるようにする

    def close(self):
        """データベース接続を閉じます。"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_database(self):
        """SQLiteデータベースを初期化し、必要なテーブルとカラムが存在することを確認します。"""
        try:
            cursor = self.conn.cursor()

            # file_metadata テーブル (C++側で管理されていることを想定)
            # 既存のテーブル構造に合わせて、不足しているカラムを追加するALTER TABLE文を使用
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    last_modified REAL,
                    size INTEGER,
                    creation_time REAL,
                    clip_feature_blob BLOB,
                    size_category TEXT,
                    kmeans_category TEXT,
                    user_sort_order INTEGER DEFAULT 0
                )
            """)
            
            # tags カラムが存在しない場合のみ追加
            try:
                cursor.execute("SELECT tags FROM file_metadata LIMIT 1")
            except sqlite3.OperationalError:
                cursor.execute("ALTER TABLE file_metadata ADD COLUMN tags TEXT DEFAULT ''")
                print("Added 'tags' column to file_metadata table.")

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"データベースの初期化エラー: {e}", file=sys.stderr)
            raise # エラーを上位に伝える

    def insert_or_update_file_metadata(self, file_path, last_modified, size, creation_time, clip_feature=None, size_category=None, kmeans_category=None, user_sort_order=0):
        """
        ファイルメタデータを挿入または更新します。
        既存のファイルパスがあれば更新、なければ新規挿入。
        """
        cursor = self.conn.cursor()
        clip_feature_blob = numpy_to_blob(clip_feature) if clip_feature is not None else None

        # 既存のレコードがあるか確認
        cursor.execute("SELECT id FROM file_metadata WHERE file_path = ?", (file_path,))
        existing_id = cursor.fetchone()

        if existing_id:
            # 更新
            update_sql = """
                UPDATE file_metadata SET
                    last_modified = ?,
                    size = ?,
                    creation_time = ?,
                    clip_feature_blob = COALESCE(?, clip_feature_blob), -- NULLでない場合にのみ更新
                    size_category = COALESCE(?, size_category),
                    kmeans_category = COALESCE(?, kmeans_category),
                    user_sort_order = ?
                WHERE file_path = ?
            """
            cursor.execute(update_sql, (
                last_modified, size, creation_time,
                clip_feature_blob, size_category, kmeans_category, user_sort_order,
                file_path
            ))
        else:
            # 挿入
            insert_sql = """
                INSERT INTO file_metadata (file_path, last_modified, size, creation_time, clip_feature_blob, size_category, kmeans_category, user_sort_order)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, (
                file_path, last_modified, size, creation_time, clip_feature_blob, size_category, kmeans_category, user_sort_order
            ))
        self.conn.commit()
    
    def get_all_file_metadata(self):
        """file_metadataテーブルから全てのファイルパスと関連情報を取得します。
           clip_feature_blobもBLOBとして返します。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path, clip_feature_blob, size_category, kmeans_category, tags FROM file_metadata")
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            # tags カラムが存在しなかった場合のためにデフォルト値
            if 'tags' not in row_dict:
                row_dict['tags'] = '' 
            results.append(row_dict)
        return results

    def get_total_image_count(self):
        """file_metadataテーブルの全エントリ数を取得します。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM file_metadata")
        return cursor.fetchone()[0]

    def get_file_paths_without_clip_features(self):
        """clip_feature_blobがNULLのfile_pathを全て取得します。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path FROM file_metadata WHERE clip_feature_blob IS NULL")
        return [row[0] for row in cursor.fetchall()]

    def update_file_metadata(self, file_path, data_to_update):
        """
        特定のファイルパスのメタデータを更新します。
        data_to_updateは辞書形式で、{'column_name': value, ...}
        """
        if not data_to_update:
            return

        set_clauses = []
        values = []
        for col, val in data_to_update.items():
            set_clauses.append(f"{col} = ?")
            if col == 'clip_feature_blob' and isinstance(val, np.ndarray):
                values.append(numpy_to_blob(val))
            else:
                values.append(val)
        
        values.append(file_path) # WHERE句の値

        sql = f"UPDATE file_metadata SET {', '.join(set_clauses)} WHERE file_path = ?"
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, tuple(values))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"ファイルメタデータの更新エラー ({file_path}): {e}", file=sys.stderr)
            raise # エラーを上位に伝える

    def get_file_metadata_by_path(self, file_path):
        """特定のファイルパスのメタデータを取得します。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM file_metadata WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        return dict(row) if row else None


# if __name__ == '__main__':
#     # テスト用DBファイル
#     test_db_path = 'test_image_features.db'
#     if os.path.exists(test_db_path):
#         os.remove(test_db_path)

#     db_manager = DBManager(test_db_path)

#     # ダミーデータの挿入
#     dummy_feature1 = np.random.rand(512).astype(np.float32)
#     dummy_feature2 = np.random.rand(512).astype(np.float32)
#     dummy_feature3 = np.random.rand(512).astype(np.float32) # 特徴量なし

#     db_manager.insert_or_update_file_metadata(
#         file_path='C:/path/to/image1.jpg',
#         last_modified=1678886400.0,
#         size=1024,
#         creation_time=1678886000.0,
#         clip_feature=dummy_feature1,
#         size_category='Large',
#         kmeans_category='ClusterA'
#     )
#     db_manager.insert_or_update_file_metadata(
#         file_path='C:/path/to/image2.png',
#         last_modified=1678886500.0,
#         size=512,
#         creation_time=1678886100.0,
#         clip_feature=dummy_feature2,
#         size_category='Medium',
#         kmeans_category='ClusterB'
#     )
#     db_manager.insert_or_update_file_metadata(
#         file_path='C:/path/to/image3.gif',
#         last_modified=1678886600.0,
#         size=256,
#         creation_time=1678886200.0,
#         clip_feature=None, # 特徴量がNULLのケース
#         size_category='Small',
#         kmeans_category='ClusterC'
#     )

#     print(f"総画像数: {db_manager.get_total_image_count()}")

#     # 特徴量がないファイルを取得
#     no_feature_files = db_manager.get_file_paths_without_clip_features()
#     print(f"特徴量がないファイル: {no_feature_files}")

#     # 特徴量がないファイルに特徴量を追加するシミュレーション
#     if no_feature_files:
#         print(f"特徴量がないファイルに特徴量を更新中: {no_feature_files[0]}")
#         new_feature = np.random.rand(512).astype(np.float32)
#         db_manager.update_file_metadata(no_feature_files[0], {'clip_feature_blob': new_feature})
#     
#     # タグの追加テスト
#     db_manager.update_file_metadata('C:/path/to/image1.jpg', {'tags': 'tag1,tag2'})
#     db_manager.update_file_metadata('C:/path/to/image2.png', {'tags': 'tag2,tag3'})
    
#     # 全メタデータの取得テスト
#     all_metadata = db_manager.get_all_file_metadata()
#     print("\n全ファイルメタデータ:")
#     for meta in all_metadata:
#         print(meta)

#     db_manager.close()
#     print(f"テストDB '{test_db_path}' を閉じました。")