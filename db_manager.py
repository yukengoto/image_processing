import sqlite3
import numpy as np
import io
import sys
import os # os.statなど、ファイルシステム操作のために追加

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
            self.conn.execute("PRAGMA journal_mode=WAL;") # WALモードは高い並行性を実現
            self.conn.execute("PRAGMA synchronous=NORMAL;") # パフォーマンス向上

    def close(self):
        """データベース接続を閉じます。"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_database(self):
        """SQLiteデータベースを初期化し、必要なテーブルとカラムが存在することを確認します。"""
        try:
            cursor = self.conn.cursor()

            # file_metadata テーブルが存在しない場合は、必要なカラム全てを含むテーブルを作成
            # size_category, kmeans_category は削除した状態
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    last_modified REAL,
                    size INTEGER,
                    creation_time REAL,
                    clip_feature_blob BLOB,
                    user_sort_order INTEGER DEFAULT 0,
                    tags TEXT DEFAULT ''
                )
            """)
            
            # 既存のテーブルに不足しているカラムを追加するためのALTER TABLE文
            # PRAGMA table_info を使用して、より確実にカラムの存在を確認
            column_additions = {
                'clip_feature_blob': 'BLOB',
                'user_sort_order': 'INTEGER DEFAULT 0',
                'tags': 'TEXT DEFAULT ''',
            }

            # 現在のテーブルのカラム情報を取得
            cursor.execute("PRAGMA table_info(file_metadata)")
            existing_columns = [row[1] for row in cursor.fetchall()] # row[1]はカラム名

            for col_name, col_type in column_additions.items():
                if col_name not in existing_columns:
                    try:
                        # カラムが存在しない場合は追加
                        cursor.execute(f"ALTER TABLE file_metadata ADD COLUMN {col_name} {col_type}")
                        print(f"Added '{col_name}' column to file_metadata table.")
                    except sqlite3.OperationalError as op_e:
                        print(f"Operational error adding column '{col_name}': {op_e}", file=sys.stderr)
                    except Exception as ex:
                        print(f"Error adding column '{col_name}': {ex}", file=sys.stderr)

            self.conn.commit()
        except sqlite3.Error as e:
            print(f"データベースの初期化エラー: {e}", file=sys.stderr)
            raise # エラーを上位に伝える

    def insert_or_update_file_metadata(self, file_path, last_modified, size, creation_time, 
                                       clip_feature_blob=None, tags=""): # size_category, kmeans_category を削除
        """
        ファイルメタデータ（とオプションでCLIP特徴量）をデータベースに挿入または更新します。
        file_pathが既に存在する場合は更新、存在しない場合は新規挿入します。
        """
        try:
            cursor = self.conn.cursor()
            
            # 既存のレコードがあるかチェック
            cursor.execute("SELECT file_path FROM file_metadata WHERE file_path = ?", (file_path,))
            existing_record = cursor.fetchone()

            if existing_record:
                # レコードが存在する場合、更新
                update_query = """
                    UPDATE file_metadata
                    SET last_modified = ?, size = ?, creation_time = ?,
                        clip_feature_blob = COALESCE(?, clip_feature_blob), -- ?がNULLでなければ更新、NULLなら既存値を保持
                        tags = ?
                    WHERE file_path = ?
                """
                cursor.execute(update_query, (
                    last_modified, size, creation_time, clip_feature_blob,
                    tags, file_path
                ))
            else:
                # レコードが存在しない場合、新規挿入
                insert_query = """
                    INSERT INTO file_metadata
                    (file_path, last_modified, size, creation_time, clip_feature_blob, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    file_path, last_modified, size, creation_time, clip_feature_blob,
                    tags
                ))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"データベース操作エラー: {e}", file=sys.stderr)
            return False

    def get_all_file_metadata(self):
        """file_metadataテーブルから全てのファイルパスと関連情報を取得します。
           clip_feature_blobもBLOBとして返します。"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT file_path, clip_feature_blob, tags FROM file_metadata")
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            # 存在しない可能性のあるカラムに対してデフォルト値を提供
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
        
        # update_file_metadataから除外するカラムを定義
        # ここではsize_categoryとkmeans_categoryがテーブルに存在しないことを想定
        excluded_columns = ['size_category', 'kmeans_category']

        set_clauses = []
        values = []
        for col, val in data_to_update.items():
            if col in excluded_columns: # 除外するカラムであればスキップ
                continue
            set_clauses.append(f"{col} = ?")
            if col == 'clip_feature_blob' and isinstance(val, np.ndarray):
                values.append(numpy_to_blob(val))
            else:
                values.append(val)
        
        if not set_clauses: # 更新対象のカラムがなければ何もしない
            return

        values.append(file_path) # WHERE句の値

        sql = f"UPDATE file_metadata SET {', '.join(set_clauses)} WHERE file_path = ?"
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql, tuple(values))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"ファイルメタデータの更新エラー ({file_path}): {e}", file=sys.stderr)
            raise # エラーを上位に伝える

    def get_file_metadata(self, file_path: str): # get_file_metadata_by_path をリネーム
        """
        特定のファイルパスのメタデータを取得します。
        見つからない場合はNoneを返します。
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM file_metadata WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        return dict(row) if row else None


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

    def save_clip_features_to_db(self, processed_image_features: dict):
        """
        辞書形式のCLIP特徴量をデータベースに保存します。
        この関数はfile_metadataのclip_feature_blobを更新するように修正されています。
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

        for i, (file_path, feature_array) in enumerate(processed_image_features.items()):
            try:
                # update_file_metadata を呼び出して clip_feature_blob を更新
                self.update_file_metadata(file_path, {'clip_feature_blob': feature_array})
                saved_count += 1
            except sqlite3.Error as e:
                print(f"エラー: 特徴量の保存中に問題が発生しました - {file_path}: {e}", file=sys.stderr)
                
            if (i + 1) % 50 == 0 or (i + 1) == total_to_save:
                sys.stdout.write(f"\r...進捗 (特徴量保存): {i + 1}/{total_to_save}")
                sys.stdout.flush()

        print(f"\nCLIP特徴量の保存が完了しました。成功: {saved_count} ファイル")

    def add_tag_to_file(self, file_path, tag_name):
        """tagsカラムを更新するように修正"""
        if not self.conn:
            print("エラー: データベースに接続していません。", file=sys.stderr)
            return False
        
        try:
            # 既存のタグを取得
            metadata = self.get_file_metadata(file_path) # get_file_metadata を使用
            current_tags_str = metadata.get('tags', '') if metadata else '' # metadataがNoneの場合も考慮
            current_tags = [t.strip() for t in current_tags_str.split(',') if t.strip()]
            
            if tag_name not in current_tags:
                current_tags.append(tag_name)
                new_tags_str = ','.join(sorted(current_tags)) # ソートして保存
                self.update_file_metadata(file_path, {'tags': new_tags_str})
                return True
            else:
                return True # 既にタグが存在する場合は成功
        except Exception as e:
            print(f"ファイル {file_path} にタグ '{tag_name}' を追加中にエラーが発生しました: {e}", file=sys.stderr)
            return False

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
        placeholders = ','.join('?' * len(file_paths))
        query = f"SELECT file_path, clip_feature_blob FROM file_metadata WHERE file_path IN ({placeholders})"
        
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