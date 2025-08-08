import sqlite3
import numpy as np
import io
import sys
import os
import json

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
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            # 外部キー制約を有効化
            self.conn.execute("PRAGMA foreign_keys=ON;")

    def close(self):
        """データベース接続を閉じます。"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_database(self):
        """C++コードと同じテーブル構造でデータベースを初期化します。"""
        try:
            cursor = self.conn.cursor()

            # file_metadata テーブル（C++と同じ構造）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_size INTEGER,
                    checksum TEXT,
                    time_created TEXT,
                    time_modified TEXT,
                    part_sum TEXT,
                    counter_value INTEGER,
                    user_sort_order INTEGER DEFAULT 0,
                    normal_attributes JSON
                )
            """)

            # file_tags テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_tags (
                    file_path TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (file_path, tag),
                    FOREIGN KEY (file_path) REFERENCES file_metadata(file_path) ON DELETE CASCADE
                )
            """)

            # user_attributes テーブル
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_attributes (
                    file_path TEXT NOT NULL,
                    attr_key TEXT NOT NULL,
                    attr_value TEXT,
                    PRIMARY KEY (file_path, attr_key),
                    FOREIGN KEY (file_path) REFERENCES file_metadata(file_path) ON DELETE CASCADE
                )
            """)

            # インデックスの作成
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag ON file_tags (tag)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_attributes_key ON user_attributes (attr_key)")

            # 既存のテーブルにCLIP特徴量用のカラムを追加（互換性のため）
            try:
                cursor.execute("ALTER TABLE file_metadata ADD COLUMN clip_feature_blob BLOB")
                print("Added 'clip_feature_blob' column to file_metadata table.")
            except sqlite3.OperationalError:
                # カラムが既に存在する場合は無視
                pass

            self.conn.commit()
            print("Database initialized with C++ compatible structure.")
        except sqlite3.Error as e:
            print(f"データベースの初期化エラー: {e}", file=sys.stderr)
            raise

    def insert_or_update_file_metadata(self, file_path, file_size=None, checksum=None, 
                                     time_created=None, time_modified=None, part_sum=None,
                                     counter_value=None, user_sort_order=None, 
                                     normal_attributes=None, clip_feature_blob=None):
        """
        C++のsaveFileInfoSingle相当の機能
        ファイルメタデータをfile_metadataテーブルに挿入または更新します。
        """
        try:
            cursor = self.conn.cursor()
            
            # normal_attributesが辞書の場合はJSON文字列に変換
            if normal_attributes and isinstance(normal_attributes, dict):
                normal_attributes_json = json.dumps(normal_attributes, ensure_ascii=False, separators=(',', ':'))
            else:
                normal_attributes_json = normal_attributes

            # INSERT OR REPLACE を使用
            query = """
                INSERT OR REPLACE INTO file_metadata
                (file_path, file_size, checksum, time_created, time_modified, 
                 part_sum, counter_value, user_sort_order, normal_attributes, clip_feature_blob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.execute(query, (
                file_path, file_size, checksum, time_created, time_modified,
                part_sum, counter_value, user_sort_order, normal_attributes_json,
                clip_feature_blob
            ))
            
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"ファイルメタデータの挿入/更新エラー: {e}", file=sys.stderr)
            return False

    def load_file_info(self, file_path):
        """
        C++のloadFileInfo相当の機能
        指定されたファイルパスの完全な情報を取得します。
        """
        try:
            cursor = self.conn.cursor()
            
            # メタデータを取得
            cursor.execute("""
                SELECT file_size, checksum, time_created, time_modified, part_sum,
                       counter_value, user_sort_order, normal_attributes, clip_feature_blob
                FROM file_metadata WHERE file_path = ?
            """, (file_path,))
            
            meta_row = cursor.fetchone()
            if not meta_row:
                return None
            
            result = {
                'file_path': file_path,
                'file_size': meta_row['file_size'],
                'checksum': meta_row['checksum'],
                'time_created': meta_row['time_created'],
                'time_modified': meta_row['time_modified'],
                'part_sum': meta_row['part_sum'],
                'counter_value': meta_row['counter_value'],
                'user_sort_order': meta_row['user_sort_order'],
                'normal_attributes': {},
                'tags': set(),
                'user_attributes': {},
                'clip_feature_blob': meta_row['clip_feature_blob']
            }
            
            # normal_attributesをJSONから辞書に変換
            if meta_row['normal_attributes']:
                try:
                    result['normal_attributes'] = json.loads(meta_row['normal_attributes'])
                except (json.JSONDecodeError, TypeError):
                    result['normal_attributes'] = {}
            
            # タグを取得
            cursor.execute("SELECT tag FROM file_tags WHERE file_path = ?", (file_path,))
            for tag_row in cursor.fetchall():
                result['tags'].add(tag_row['tag'])
            
            # ユーザー属性を取得
            cursor.execute("SELECT attr_key, attr_value FROM user_attributes WHERE file_path = ?", (file_path,))
            for attr_row in cursor.fetchall():
                result['user_attributes'][attr_row['attr_key']] = attr_row['attr_value']
            
            return result
            
        except sqlite3.Error as e:
            print(f"ファイル情報の読み込みエラー ({file_path}): {e}", file=sys.stderr)
            return None

    def add_tag_to_file(self, file_path, tag_name):
        """
        ファイルにタグを追加します（file_tagsテーブルを使用）
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO file_tags (file_path, tag)
                VALUES (?, ?)
            """, (file_path, tag_name))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"タグ追加エラー ({file_path}, {tag_name}): {e}", file=sys.stderr)
            return False

    def remove_tag_from_file(self, file_path, tag_name):
        """
        ファイルからタグを削除します
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM file_tags WHERE file_path = ? AND tag = ?
            """, (file_path, tag_name))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"タグ削除エラー ({file_path}, {tag_name}): {e}", file=sys.stderr)
            return False

    # db_manager.py の get_file_tags メソッドも確認用に修正版を提供
    def get_file_tags(self, file_path):
        """
        指定されたファイルのタグを取得します
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT tag FROM file_tags WHERE file_path = ?", (file_path,))
            tags = [row['tag'] for row in cursor.fetchall()]
            return tags
        except sqlite3.Error as e:
            print(f"タグ取得エラー ({file_path}): {e}", file=sys.stderr)
            return []

    def get_file_tags2(self, file_path):
        """
        指定されたファイルのタグを取得します
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT tag FROM file_tags WHERE file_path = ?", (file_path,))
            return [row['tag'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"タグ取得エラー ({file_path}): {e}", file=sys.stderr)
            return []

    def set_user_attribute(self, file_path, attr_key, attr_value):
        """
        ユーザー属性を設定します（user_attributesテーブルを使用）
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_attributes (file_path, attr_key, attr_value)
                VALUES (?, ?, ?)
            """, (file_path, attr_key, attr_value))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"ユーザー属性設定エラー ({file_path}, {attr_key}): {e}", file=sys.stderr)
            return False

    def get_user_attribute(self, file_path, attr_key):
        """
        ユーザー属性を取得します
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT attr_value FROM user_attributes 
                WHERE file_path = ? AND attr_key = ?
            """, (file_path, attr_key))
            row = cursor.fetchone()
            return row['attr_value'] if row else None
        except sqlite3.Error as e:
            print(f"ユーザー属性取得エラー ({file_path}, {attr_key}): {e}", file=sys.stderr)
            return None

    def get_all_file_metadata(self):
        """
        全てのファイルメタデータを取得します（user_sort_order順）
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT file_path, file_size, checksum, time_created, time_modified,
                       part_sum, counter_value, user_sort_order, normal_attributes, clip_feature_blob
                FROM file_metadata ORDER BY user_sort_order ASC
            """)
            
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                # normal_attributesをJSONから辞書に変換
                if row_dict['normal_attributes']:
                    try:
                        row_dict['normal_attributes'] = json.loads(row_dict['normal_attributes'])
                    except (json.JSONDecodeError, TypeError):
                        row_dict['normal_attributes'] = {}
                else:
                    row_dict['normal_attributes'] = {}
                results.append(row_dict)
            
            return results
        except sqlite3.Error as e:
            print(f"全ファイルメタデータ取得エラー: {e}", file=sys.stderr)
            return []

    def get_image_paths_from_db(self, max_count=None):
        """データベースから画像ファイルのパスを取得します（user_sort_order順）"""
        try:
            cursor = self.conn.cursor()
            query = "SELECT file_path FROM file_metadata ORDER BY user_sort_order ASC"
            if max_count and max_count > 0:
                query += f" LIMIT {max_count}"
            
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"画像パス取得エラー: {e}", file=sys.stderr)
            return []

    def save_clip_features_to_db(self, processed_image_features: dict):
        """
        CLIP特徴量をfile_metadataテーブルのclip_feature_blobカラムに保存します
        processed_image_features: {file_path: feature_array} の辞書
        """
        if not self.conn:
            print("エラー: データベースに接続していません。", file=sys.stderr)
            return
        if not processed_image_features:
            print("保存すべきCLIP特徴量がありません。")
            return

        print("\nデータベースにCLIP特徴量を保存中...")
        saved_count = 0
        total_to_save = len(processed_image_features)

        try:
            cursor = self.conn.cursor()
            for i, (file_path, feature_array) in enumerate(processed_image_features.items()):
                try:
                    blob_data = numpy_to_blob(feature_array)
                    cursor.execute("""
                        UPDATE file_metadata 
                        SET clip_feature_blob = ?
                        WHERE file_path = ?
                    """, (blob_data, file_path))
                    saved_count += 1
                except sqlite3.Error as e:
                    print(f"エラー: 特徴量の保存中に問題が発生しました - {file_path}: {e}", file=sys.stderr)
                    
                if (i + 1) % 50 == 0 or (i + 1) == total_to_save:
                    sys.stdout.write(f"\r...進捗 (特徴量保存): {i + 1}/{total_to_save}")
                    sys.stdout.flush()
            
            self.conn.commit()
            print(f"\nCLIP特徴量の保存が完了しました。成功: {saved_count} ファイル")
        except sqlite3.Error as e:
            print(f"CLIP特徴量保存エラー: {e}", file=sys.stderr)

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
        
        try:
            # IN句を使用してバッチで取得
            placeholders = ','.join('?' * len(file_paths))
            query = f"SELECT file_path, clip_feature_blob FROM file_metadata WHERE file_path IN ({placeholders})"
            
            cursor.execute(query, file_paths)
            for row in cursor.fetchall():
                file_path = row[0]
                feature_blob = row[1]
                if feature_blob:
                    features_map[file_path] = blob_to_numpy(feature_blob)
        except sqlite3.Error as e:
            print(f"データベースからの特徴量読み込みエラー: {e}", file=sys.stderr)
        
        return features_map

    def get_files_without_clip_features(self):
        """CLIP特徴量が設定されていないファイルパスを取得します"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM file_metadata WHERE clip_feature_blob IS NULL")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"特徴量なしファイル取得エラー: {e}", file=sys.stderr)
            return []

    def get_total_image_count(self):
        """file_metadataテーブルの全エントリ数を取得します。"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM file_metadata")
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"総数取得エラー: {e}", file=sys.stderr)
            return 0

    def search_files_by_tags(self, tags: list, match_all=True):
        """
        タグによってファイルを検索します
        match_all=True: 全てのタグを持つファイル
        match_all=False: いずれかのタグを持つファイル
        """
        if not tags:
            return []
        
        try:
            cursor = self.conn.cursor()
            if match_all:
                # 全てのタグを持つファイルを検索
                placeholders = ','.join('?' * len(tags))
                query = f"""
                    SELECT file_path FROM file_tags 
                    WHERE tag IN ({placeholders})
                    GROUP BY file_path 
                    HAVING COUNT(DISTINCT tag) = ?
                """
                cursor.execute(query, tags + [len(tags)])
            else:
                # いずれかのタグを持つファイルを検索
                placeholders = ','.join('?' * len(tags))
                query = f"""
                    SELECT DISTINCT file_path FROM file_tags 
                    WHERE tag IN ({placeholders})
                """
                cursor.execute(query, tags)
            
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"タグ検索エラー: {e}", file=sys.stderr)
            return []