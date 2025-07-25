# db_manager.py

import sqlite3
import numpy as np
import io
import sys
import os

# --- NumPy配列 <-> BLOB変換ヘルパー関数 (移動) ---
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

class DBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """SQLiteデータベースに接続し、必要なテーブルが存在することを確認します。"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()

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

            self.conn.commit()
            print(f"データベースとテーブルが初期化/検証されました: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"データベース初期化エラー: {e}", file=sys.stderr)
            if self.conn:
                self.conn.close()
            self.conn = None
            return False

    def close(self):
        """データベース接続を閉じます。"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("データベース接続を閉じました。")

    def add_tag_to_file(self, file_path, tag_name):
        """file_tagsテーブルに分類タグを追加します。"""
        if not self.conn:
            print("エラー: データベースに接続していません。", file=sys.stderr)
            return False
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT 1 FROM file_tags
                WHERE file_path = ? AND tag = ?
            """, (file_path, tag_name))
            if cursor.fetchone():
                # print(f"タグ '{tag_name}' は {file_path} に既に存在します。スキップします。")
                return True # タグが既に存在する場合は成功とみなす
            
            cursor.execute("""
                INSERT INTO file_tags (file_path, tag)
                VALUES (?, ?)
            """, (file_path, tag_name))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"ファイル {file_path} にタグ '{tag_name}' を追加中にエラーが発生しました: {e}", file=sys.stderr)
            return False

    def save_clip_features_to_db(self, processed_image_features: dict):
        """
        辞書形式のCLIP特徴量をデータベースのclip_featuresテーブルに保存します。
        processed_image_features: {file_path: feature_array} の辞書
        """
        if not self.conn:
            print("エラー: データベースに接続していません。", file=sys.stderr)
            return
        if not processed_image_features:
            print("保存すべきCLIP特徴量がありません。")
            return

        print("\nデータベースにCLIP特徴量を保存中...")
        cursor = self.conn.cursor()
        saved_count = 0
        total_to_save = len(processed_image_features)

        for i, (file_path, feature_array) in enumerate(processed_image_features.items()):
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

        self.conn.commit() # ここで変更をコミット
        print(f"\nCLIP特徴量の保存とコミットが完了しました。成功: {saved_count} ファイル")

    def get_image_paths_from_db(self, max_count=None):
        """データベースから画像ファイルのパスを取得します。"""
        if not self.conn:
            print("エラー: データベースに接続していません。", file=sys.stderr)
            return []
        cursor = self.conn.cursor()
        query = "SELECT file_path FROM file_metadata ORDER BY user_sort_order ASC"
        if max_count and max_count > 0:
            query += f" LIMIT {max_count}"
        
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]

    def get_clip_features_from_db(self, file_paths: list) -> dict:
        """
        指定されたファイルパスのCLIP特徴量をデータベースから読み込みます。
        戻り値: {file_path: feature_array} の辞書
        """
        if not self.conn:
            print("エラー: データベースに接続していません。", file=sys.stderr)
            return {}
        if not file_paths:
            return {}

        features_map = {}
        cursor = self.conn.cursor()
        
        # IN句の最大数に注意。SQLiteは1000がデフォルト。多数のパスがある場合は分割してクエリ発行が必要
        # 簡単のためここでは直接IN句を使うが、実際の使用ではチャンクに分けるべき
        placeholders = ','.join('?' * len(file_paths))
        query = f"SELECT file_path, clip_feature_blob FROM clip_features WHERE file_path IN ({placeholders})"
        
        try:
            cursor.execute(query, file_paths)
            for row in cursor.fetchall():
                file_path = row[0]
                feature_blob = row[1]
                if feature_blob:
                    features_map[file_path] = blob_to_numpy(feature_blob)
        except sqlite3.Error as e:
            print(f"データベースからの特徴量読み込みエラー: {e}", file=sys.stderr)
        
        return features_map

# テスト用
if __name__ == '__main__':
    dummy_db_path = "test_db.db"
    if os.path.exists(dummy_db_path):
        os.remove(dummy_db_path)

    db_manager = DBManager(dummy_db_path)
    if db_manager.connect():
        # ダミーデータ挿入
        cursor = db_manager.conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO file_metadata (file_path) VALUES (?)", ("test_image_1.jpg",))
        cursor.execute("INSERT OR IGNORE INTO file_metadata (file_path) VALUES (?)", ("test_image_2.jpg",))
        db_manager.conn.commit()

        # タグ追加テスト
        db_manager.add_tag_to_file("test_image_1.jpg", "landscape")
        db_manager.add_tag_to_file("test_image_1.jpg", "mountain")
        
        # 特徴量保存テスト
        dummy_feature_1 = np.random.rand(512)
        dummy_feature_2 = np.random.rand(512)
        features_to_save = {
            "test_image_1.jpg": dummy_feature_1,
            "test_image_2.jpg": dummy_feature_2
        }
        db_manager.save_clip_features_to_db(features_to_save)

        # パス取得テスト
        paths = db_manager.get_image_paths_from_db()
        print(f"DBから取得したパス: {paths}")

        # 特徴量読み込みテスト
        loaded_features = db_manager.get_clip_features_from_db(paths)
        print(f"DBから読み込んだ特徴量の数: {len(loaded_features)}")
        if "test_image_1.jpg" in loaded_features:
            print(f"test_image_1.jpg の特徴量形状: {loaded_features['test_image_1.jpg'].shape}")
            print(f"元の特徴量と一致するか: {np.allclose(dummy_feature_1, loaded_features['test_image_1.jpg'])}")

        db_manager.close()
        os.remove(dummy_db_path)
