import sys
import os
import json
import sqlite3
import numpy as np
import mimetypes
from pathlib import Path
from PIL import Image, ExifTags  # PILライブラリを追加

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableView, QLineEdit, QHeaderView,
    QStatusBar, QAbstractItemView, QMessageBox, QInputDialog, QMenu,
    QLabel, QComboBox, QProgressDialog, QListView, QStackedWidget, QStyle
)
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QSize,
    QThreadPool, QRunnable, Signal, QObject, QUrl
)
from PySide6.QtGui import QPixmap, QDesktopServices, QImage

# --- 新しいモジュールのインポート ---
from db_manager import DBManager, blob_to_numpy, numpy_to_blob
from clip_feature_extractor import CLIPFeatureExtractor

# --- 1. サムネイル生成をバックグラウンドで行うためのQRunnableとシグナルエミッター ---
class ThumbnailSignalEmitter(QObject):
    """QRunnableからQAbstractTableModelにシグナルを送るためのヘルパークラス"""
    thumbnail_ready = Signal(QModelIndex, QPixmap, bool)  # bool: is_preview
    error = Signal(str)

# image_feature_manager.py の安全性向上のための修正
# === ファイル形式チェック用のユーティリティクラス ===
class FileTypeValidator:
    """ファイル形式の検証とカテゴリ分類を行うクラス"""
    
    # 対応する画像形式
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg'}
    
    # 対応する動画形式（将来的な拡張用）
    SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    
    # サムネイル生成可能な形式（QPixmapで読み込み可能）
    THUMBNAIL_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico'}
    
    # CLIP特徴量抽出可能な形式
    CLIP_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    @classmethod
    def is_image_file(cls, file_path):
        """画像ファイルかどうかを判定"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_IMAGE_EXTENSIONS
    
    @classmethod
    def is_video_file(cls, file_path):
        """動画ファイルかどうかを判定"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_VIDEO_EXTENSIONS
    
    @classmethod
    def supports_thumbnail(cls, file_path):
        """サムネイル生成に対応しているかを判定"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.THUMBNAIL_SUPPORTED_EXTENSIONS
    
    @classmethod
    def supports_clip_features(cls, file_path):
        """CLIP特徴量抽出に対応しているかを判定"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.CLIP_SUPPORTED_EXTENSIONS
    
    @classmethod
    def get_file_category(cls, file_path):
        """ファイルのカテゴリを取得"""
        if cls.is_image_file(file_path):
            return "image"
        elif cls.is_video_file(file_path):
            return "video"
        else:
            return "other"
    
    @classmethod
    def validate_file_integrity(cls, file_path):
        """ファイルの整合性を基本チェック"""
        try:
            if not os.path.exists(file_path):
                return False, "ファイルが存在しません"
            
            if not os.path.isfile(file_path):
                return False, "ディレクトリです"
            
            if os.path.getsize(file_path) == 0:
                return False, "ファイルサイズが0です"
            
            # MIMEタイプによる基本検証
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and not mime_type.startswith(('image/', 'video/')):
                if not cls.is_image_file(file_path) and not cls.is_video_file(file_path):
                    return False, f"サポートされていないファイル形式: {mime_type}"
            
            return True, "OK"
        except Exception as e:
            return False, f"検証エラー: {e}"


# === 改良されたThumbnailGenerator ===
#class SafeThumbnailGenerator(QRunnable):
class ThumbnailGenerator(QRunnable):
    """安全性を向上させたサムネイル生成タスク"""
    #class ThumbnailGenerator(QRunnable):
    def _load_and_rotate_image(self, image_path, is_preview=True):
        """EXIFの回転情報とサムネイルを考慮して画像を読み込む"""
        try:
            with Image.open(image_path) as img:
                # プレビュー用の場合は埋め込みサムネイルを試す
                if is_preview:
                    try:
                        if hasattr(img, '_getexif') and img._getexif():
                            exif = dict(img._getexif().items())
                            if 0x0201 in exif:  # JPEGInterchangeFormat
                                jpeg_start = exif[0x0201]
                                if 0x0202 in exif:  # JPEGInterchangeFormatLength
                                    jpeg_length = exif[0x0202]
                                    with open(image_path, 'rb') as f:
                                        f.seek(jpeg_start)
                                        thumbnail_data = f.read(jpeg_length)
                                        if thumbnail_data:
                                            try:
                                                thumbnail_img = Image.open(io.BytesIO(thumbnail_data))
                                                if thumbnail_img.mode != 'RGB':
                                                    thumbnail_img = thumbnail_img.convert('RGB')
                                                thumb_data = thumbnail_img.tobytes('raw', 'RGB')
                                                qimg = QImage(thumb_data, thumbnail_img.width, thumbnail_img.height, 
                                                            thumbnail_img.width * 3, QImage.Format.Format_RGB888)
                                                return QPixmap.fromImage(qimg)
                                            except:
                                                pass
                    except:
                        pass

                # プレビュー用の場合は高速化のために小さいサイズで読み込む
                if is_preview:
                    # 元画像の1/4サイズで読み込む
                    img.thumbnail((img.width // 4, img.height // 4))

                # 回転情報の取得と適用
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] == 'Orientation':
                            break
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = dict(img._getexif().items())
                        if orientation in exif:
                            if exif[orientation] == 2:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            elif exif[orientation] == 3:
                                img = img.transpose(Image.ROTATE_180)
                            elif exif[orientation] == 4:
                                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            elif exif[orientation] == 5:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                            elif exif[orientation] == 6:
                                img = img.transpose(Image.ROTATE_270)
                            elif exif[orientation] == 7:
                                img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                            elif exif[orientation] == 8:
                                img = img.transpose(Image.ROTATE_90)
                except:
                    pass

                img_data = img.convert('RGB').tobytes('raw', 'RGB')
                qimg = QImage(img_data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
                return QPixmap.fromImage(qimg)

        except Exception as e:
            print(f"画像の読み込み処理エラー ({os.path.basename(image_path)}): {e}")
            return QPixmap(image_path)

    def run(self):
        try:
            # ファイルの事前検証
            is_valid, error_msg = FileTypeValidator.validate_file_integrity(self.image_path)
            if not is_valid:
                self.signal_emitter.error.emit(f"ファイル検証エラー ({os.path.basename(self.image_path)}): {error_msg}")
                return

            # プレビュー用の低画質サムネイルを生成
            preview_pixmap = self._load_and_rotate_image(self.image_path, is_preview=True)
            if not preview_pixmap.isNull():
                preview_thumb = preview_pixmap.scaled(
                    self.size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation  # 高速な変換を使用
                )
                self.signal_emitter.thumbnail_ready.emit(self.index, preview_thumb, True)

            # 高画質版のサムネイルを生成
            original_pixmap = self._load_and_rotate_image(self.image_path, is_preview=False)
            if not original_pixmap.isNull():
                final_thumb = original_pixmap.scaled(
                    self.size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.signal_emitter.thumbnail_ready.emit(self.index, final_thumb, False)

        except Exception as e:
            error_msg = f"サムネイル生成中の予期せぬエラー ({os.path.basename(self.image_path)}): {e}"
            print(error_msg, file=sys.stderr)
            self.signal_emitter.error.emit(error_msg)


    def __init__(self, image_path, size: QSize, index, signal_emitter):
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.index = index
        self.signal_emitter = signal_emitter

# --- 全画像データロード用のQRunnableとシグナルエミッター ---
class AllImagesLoadSignalEmitter(QObject):
    """全画像データのロードタスクからメインスレッドにシグナルを送るためのヘルパークラス"""
    progress_update = Signal(int, int, str)  # (current, total, message)
    finished = Signal(list, int)  # (data_list, total_count)
    error = Signal(str)

class AllImagesLoader(QRunnable):
    """データベースから全画像データとタグ情報を取得するタスク"""
    def __init__(self, db_path: str, signal_emitter: AllImagesLoadSignalEmitter, max_display_count: int = None):
        super().__init__()
        self.db_path = db_path
        self.signal_emitter = signal_emitter
        self.max_display_count = max_display_count
        self.db_manager = None

    # AllImagesLoader.run() メソッドの修正（タグ取得部分）
    def run(self):
        try:
            self.db_manager = DBManager(self.db_path)
            
            # 全画像のメタデータを取得
            self.signal_emitter.progress_update.emit(0, 0, "画像データを読み込み中...")
            all_db_data = self.db_manager.get_all_file_metadata()
            total_count = len(all_db_data)
            
            if total_count == 0:
                self.signal_emitter.finished.emit([], 0)
                return
            
            # 各ファイルのタグ情報を取得（バッチ処理で効率化）
            self.signal_emitter.progress_update.emit(0, total_count, "タグ情報を読み込み中...")
            
            # タグ情報を一括取得するためのSQL
            cursor = self.db_manager.conn.cursor()
            file_paths = [item['file_path'] for item in all_db_data]
            
            # IN句を使って一括でタグを取得
            placeholders = ','.join('?' * len(file_paths))
            cursor.execute(f"""
                SELECT file_path, tag FROM file_tags 
                WHERE file_path IN ({placeholders})
                ORDER BY file_path, tag
            """, file_paths)
            
            # タグ情報を辞書にまとめる
            tags_dict = {}
            for row in cursor.fetchall():
                file_path = row[0]
                tag = row[1]
                if file_path not in tags_dict:
                    tags_dict[file_path] = set()
                tags_dict[file_path].add(tag)
            
            # デバッグ情報：タグ辞書の内容を確認
            print(f"DEBUG: Found tags for {len(tags_dict)} files")
            for file_path, tags in list(tags_dict.items())[:3]:  # 最初の3ファイル分だけ表示
                print(f"DEBUG: {os.path.basename(file_path)}: {tags}")
            
            # 各アイテムにタグ情報とスコア（None）を追加
            for i, item in enumerate(all_db_data):
                item['score'] = None  # 検索ではないのでスコアはNone
                file_path = item.get('file_path')
                item['tags'] = tags_dict.get(file_path, set())
                
                # デバッグ：最初の数ファイルのタグ情報を出力
                if i < 5:
                    print(f"DEBUG: Item {i} - {os.path.basename(file_path)}: tags = {item['tags']} (type: {type(item['tags'])})")
                
                # 進捗更新（100件ごと）
                if (i + 1) % 100 == 0 or (i + 1) == total_count:
                    self.signal_emitter.progress_update.emit(
                        i + 1, total_count, f"データ整理中: {i + 1}/{total_count}"
                    )
            
            # 表示件数制限があれば適用
            if self.max_display_count and self.max_display_count < len(all_db_data):
                display_data = all_db_data[:self.max_display_count]
            else:
                display_data = all_db_data
            
            self.signal_emitter.finished.emit(display_data, total_count)
            
        except Exception as e:
            error_msg = f"全画像データの読み込み中にエラーが発生しました: {e}"
            print(error_msg, file=sys.stderr)
            import traceback
            traceback.print_exc()  # デバッグ用：スタックトレースを出力
            self.signal_emitter.error.emit(error_msg)
        finally:
            if self.db_manager:
                self.db_manager.close()

# --- 特徴量取得をバックグラウンドで行うためのQRunnableとシグナルエミッター ---
class FeatureExtractionSignalEmitter(QObject):
    """特徴量取得タスクからメインスレッドにシグナルを送るためのヘルパークラス"""
    progress_update = Signal(int, int, str)  # (current, total, filename)
    finished = Signal(int)  # 処理されたファイル数
    error = Signal(str)

# === 改良されたFeatureExtractor ===
class FeatureExtractor(QRunnable):
#class SafeFeatureExtractor(QRunnable):
    """安全性を向上させた特徴量抽出タスク"""
    
    def __init__(self, db_path: str, files_without_features: list, signal_emitter: FeatureExtractionSignalEmitter):
        super().__init__()
        self.db_path = db_path
        self.files_without_features = files_without_features
        self.signal_emitter = signal_emitter
        self.db_manager = None
        self.clip_feature_extractor = None

    def run(self):
        try:
            self.db_manager = DBManager(self.db_path)
            self.clip_feature_extractor = CLIPFeatureExtractor()
            
            # 対応ファイルのみをフィルタリング
            valid_files = []
            skipped_count = 0
            
            for file_path in self.files_without_features:
                is_valid, error_msg = FileTypeValidator.validate_file_integrity(file_path)
                if is_valid and FileTypeValidator.supports_clip_features(file_path):
                    valid_files.append(file_path)
                else:
                    skipped_count += 1
                    if not is_valid:
                        print(f"スキップ ({os.path.basename(file_path)}): {error_msg}", file=sys.stderr)
                    else:
                        print(f"スキップ (非対応形式): {os.path.basename(file_path)}", file=sys.stderr)
            
            total_files = len(valid_files)
            
            if total_files == 0:
                if skipped_count > 0:
                    self.signal_emitter.error.emit(f"{skipped_count} 個のファイルがスキップされました（非対応形式または破損ファイル）")
                self.signal_emitter.finished.emit(0)
                return

            if skipped_count > 0:
                self.signal_emitter.progress_update.emit(0, total_files, f"{skipped_count} 個のファイルをスキップしました")

            # 特徴量抽出の進捗を手動で追跡
            extracted_features = []
            processed_indices = []
            
            for i, file_path in enumerate(valid_files):
                try:
                    self.signal_emitter.progress_update.emit(
                        i + 1, 
                        total_files, 
                        f"特徴量抽出中: {os.path.basename(file_path)}"
                    )
                    
                    # 個別に特徴量を抽出
                    features, indices = self.clip_feature_extractor.extract_features_from_paths([file_path])
                    if len(features) > 0:
                        extracted_features.append(features[0])
                        processed_indices.append(i)
                    else:
                        print(f"特徴量抽出失敗: {os.path.basename(file_path)}", file=sys.stderr)
                    
                except Exception as e:
                    error_msg = f"特徴量抽出エラー ({os.path.basename(file_path)}): {e}"
                    print(error_msg, file=sys.stderr)
                    continue
            
            if len(extracted_features) == 0:
                self.signal_emitter.finished.emit(0)
                return

            # データベース更新フェーズ
            updated_count = 0
            for i, (feature, original_index) in enumerate(zip(extracted_features, processed_indices)):
                file_path = valid_files[original_index]
                
                try:
                    feature_blob = numpy_to_blob(feature)
                    self.db_manager.insert_or_update_file_metadata(
                        file_path=file_path,
                        clip_feature_blob=feature_blob
                    )
                    updated_count += 1
                    
                    self.signal_emitter.progress_update.emit(
                        i + 1, 
                        len(extracted_features), 
                        f"DB更新中: {os.path.basename(file_path)}"
                    )
                except Exception as e:
                    error_msg = f"特徴量更新エラー ({os.path.basename(file_path)}): {e}"
                    print(error_msg, file=sys.stderr)
            
            self.signal_emitter.finished.emit(updated_count)
            
        except Exception as e:
            error_msg = f"特徴量抽出処理中にエラーが発生しました: {e}"
            print(error_msg, file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.signal_emitter.error.emit(error_msg)
        finally:
            if self.db_manager:
                self.db_manager.close()

# ビュー切り替えの列挙型を追加
from enum import Enum

class ViewMode(Enum):
    TABLE = 0
    ICON = 1

class ImageViewModel(QAbstractTableModel):
    def __init__(self, parent=None, initial_thumbnail_size=100):
        super().__init__(parent)
        self.view_mode = ViewMode.TABLE
        self._data = []
        self._total_image_count = 0
        self._headers = ["", "ファイル名", "類似度", "タグ", "パス"]
        self.thumbnail_cache = {}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(os.cpu_count() or 1)
        
        self.thumbnail_signal_emitter = ThumbnailSignalEmitter()
        self.thumbnail_signal_emitter.thumbnail_ready.connect(self.update_thumbnail)
        self.thumbnail_signal_emitter.error.connect(self.parent().statusBar().showMessage)
        
        self.thumbnail_size = QSize(initial_thumbnail_size, initial_thumbnail_size)

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        # アイコンビューモードの場合は1列
        if self.view_mode == ViewMode.ICON:
            return 1
        # テーブルビューモードの場合は通常通り
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DecorationRole:
            # サムネイルは最初の列のみに表示
            if index.column() != 0:
                return None
                
            # サムネイルの表示
            file_path = self._data[index.row()].get('file_path', '')
            if not file_path:
                return QPixmap()

            # キャッシュにあればそれを返す
            if file_path in self.thumbnail_cache:
                return self.thumbnail_cache[file_path]
            
            # サムネイル生成をリクエスト
            if FileTypeValidator.supports_thumbnail(file_path):
                generator = ThumbnailGenerator(
                    image_path=file_path,
                    size=self.thumbnail_size,
                    index=index,
                    signal_emitter=self.thumbnail_signal_emitter
                )
                self.thread_pool.start(generator)
            return QPixmap()

        elif role == Qt.ItemDataRole.DisplayRole:
            row = index.row()
            col = index.column()
            if col == 0:
                return ""  # サムネイル列は空文字列
            elif col == 1:
                return os.path.basename(self._data[row].get('file_path', ''))
            elif col == 2:
                score = self._data[row].get('score')
                return f"{score:.4f}" if score is not None else ""
            elif col == 3:
                tags = self._data[row].get('tags', set())
                return ', '.join(sorted(tags)) if tags else ""
            elif col == 4:
                return self._data[row].get('file_path', '')

        elif role == Qt.ItemDataRole.ToolTipRole:
            file_info = self._data[index.row()]
            tags = ', '.join(sorted(file_info.get('tags', set())))
            return f"ファイル名: {os.path.basename(file_info.get('file_path', ''))}\nタグ: {tags}"

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if self.view_mode == ViewMode.ICON:
            return None
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def set_data(self, data, total_count):
        self.beginResetModel()
        self._data = data
        self._total_image_count = total_count
        self.thumbnail_cache.clear()
        self.endResetModel()

    def set_current_thumbnail_size(self, size_int: int):
        new_size = QSize(size_int, size_int) if size_int > 0 else QSize(0, 0)
        if self.thumbnail_size != new_size:
            self.thumbnail_size = new_size
            self.thumbnail_cache.clear()
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, 0),
                [Qt.ItemDataRole.DecorationRole]
            )

    def set_view_mode(self, mode: ViewMode):
        self.view_mode = mode
        self.layoutChanged.emit()

    def update_thumbnail(self, index: QModelIndex, pixmap: QPixmap, is_preview: bool):
        file_path = self._data[index.row()].get('file_path')
        current_pixmap = self.thumbnail_cache.get(file_path)
        
        if is_preview and current_pixmap is not None:
            return
            
        self.thumbnail_cache[file_path] = pixmap
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])

    def get_row_data(self, row: int):
        if 0 <= row < len(self._data):
            return self._data[row]
        return None

    def get_total_image_count(self):
        return self._total_image_count


# --- 3. データベースへの画像追加をバックグラウンドで行うためのQRunnableとシグナルエミッター ---
class ImageAddSignalEmitter(QObject):
    """QRunnableからメインスレッドにシグナルを送るためのヘルパークラス"""
    progress_update = Signal(str)
    finished = Signal(int) # 追加されたファイルの数を返す
    error = Signal(str)

class ImageAdder(QRunnable):
    """画像ファイルをデータベースに追加するタスク"""
    def __init__(self, db_path: str, image_paths: list, signal_emitter: ImageAddSignalEmitter):
        super().__init__()
        self.db_path = db_path # DBManagerインスタンスではなくパスを受け取る
        self.image_paths = image_paths
        self.signal_emitter = signal_emitter
        self.db_manager = None # 各スレッドで個別にインスタンス化

    # === ImageAdder の修正 ===
    def run(self):
    #def safe_image_adder_run(self):
        """安全性を向上させたImageAdder.runメソッド"""
        added_count = 0
        skipped_count = 0
        total_files = len(self.image_paths)
        
        self.signal_emitter.progress_update.emit(f"データベースに {total_files} 個のファイルを追加中...")

        try:
            self.db_manager = DBManager(self.db_path)
            
            for i, file_path in enumerate(self.image_paths):
                # ファイル検証
                is_valid, error_msg = FileTypeValidator.validate_file_integrity(file_path)
                if not is_valid:
                    self.signal_emitter.progress_update.emit(f"スキップ: {os.path.basename(file_path)} ({error_msg})")
                    skipped_count += 1
                    continue
                
                # 画像ファイルかチェック
                if not FileTypeValidator.is_image_file(file_path):
                    self.signal_emitter.progress_update.emit(f"スキップ (非画像ファイル): {os.path.basename(file_path)}")
                    skipped_count += 1
                    continue
                
                try:
                    # ファイルのメタデータを取得
                    stat_info = os.stat(file_path)
                    file_size = stat_info.st_size
                    creation_time = str(stat_info.st_ctime)
                    last_modified_time = str(stat_info.st_mtime)

                    self.db_manager.insert_or_update_file_metadata(
                        file_path=file_path,
                        file_size=file_size,
                        time_created=creation_time,
                        time_modified=last_modified_time,
                        clip_feature_blob=None
                    )
                    added_count += 1
                    self.signal_emitter.progress_update.emit(f"追加中: {added_count}/{total_files - skipped_count} ({os.path.basename(file_path)})")
                    
                except Exception as e:
                    error_msg = f"ファイル '{file_path}' の追加中にエラーが発生しました: {e}"
                    print(error_msg, file=sys.stderr)
                    self.signal_emitter.error.emit(f"ファイル追加エラー ({os.path.basename(file_path)}): {e}")
                    skipped_count += 1
            
            if skipped_count > 0:
                self.signal_emitter.progress_update.emit(f"完了: {added_count} 個追加、{skipped_count} 個スキップ")
            
            self.signal_emitter.finished.emit(added_count)
            
        except Exception as e:
            error_msg = f"データベース処理中にエラーが発生しました: {e}"
            print(error_msg, file=sys.stderr)
            self.signal_emitter.error.emit(error_msg)
        finally:
            if self.db_manager:
                self.db_manager.close()

# image_feature_manager.py に追加するコード

from PySide6.QtWidgets import (
    # 既存のインポートに以下を追加
    QDialog, QCheckBox, QScrollArea, QGridLayout, QGroupBox
)

# === タグ選択ダイアログクラス ===
class TagSelectionDialog(QDialog):
    """既存タグの選択と新規タグの入力を行うダイアログ"""
    
    def __init__(self, parent=None, db_manager=None, title="タグの選択", 
                 current_tags=None, allow_new_tags=True):
        super().__init__(parent)
        self.db_manager = db_manager
        self.current_tags = current_tags or set()
        self.allow_new_tags = allow_new_tags
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(400, 500)
        self.selected_existing_tags = set()
        self.new_tags_text = ""
        self.tag_checkboxes = {}
        self._init_ui()
    def _make_tristate_checkbox(self, tag):
        checkbox = QCheckBox(tag)
        checkbox.setTristate(True)
        checkbox.setCheckState(Qt.PartiallyChecked)
        def on_click():
            state = checkbox.checkState()
            if state == Qt.Unchecked:
                checkbox.setCheckState(Qt.Checked)
            elif state == Qt.Checked:
                checkbox.setCheckState(Qt.PartiallyChecked)
            else:
                checkbox.setCheckState(Qt.Unchecked)

        checkbox.clicked.connect(on_click)
        return checkbox

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        # 既存タグ選択エリア
        existing_tags_group = QGroupBox("既存のタグから選択")
        main_layout.addWidget(existing_tags_group)
        existing_layout = QVBoxLayout(existing_tags_group)
        # スクロールエリア
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        existing_layout.addWidget(self.scroll_area)
        # チェックボックスを配置するウィジェット
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QGridLayout(self.checkbox_widget)
        self.scroll_area.setWidget(self.checkbox_widget)
        self.tag_checkboxes = {}
        # 既存タグをロード
        if self.db_manager:
            try:
                cursor = self.db_manager.conn.cursor()
                cursor.execute("SELECT DISTINCT tag FROM file_tags ORDER BY tag")
                existing_tags = [row[0] for row in cursor.fetchall()]
                row = 0
                col = 0
                for tag in existing_tags:
                    checkbox = self._make_tristate_checkbox(tag)
                    self.checkbox_layout.addWidget(checkbox, row, col)
                    self.tag_checkboxes[tag] = checkbox
                    col += 1
                    if col >= 2:
                        col = 0
                        row += 1
            except Exception as e:
                error_label = QLabel(f"タグの読み込みエラー: {e}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.checkbox_layout.addWidget(error_label, 0, 0, 1, 2)
        # 新規タグ入力エリア（オプション）
        if self.allow_new_tags:
            new_tags_group = QGroupBox("新しいタグを追加（カンマ区切り）")
            main_layout.addWidget(new_tags_group)
            new_tags_layout = QVBoxLayout(new_tags_group)
            self.new_tags_input = QLineEdit()
            self.new_tags_input.setPlaceholderText("新しいタグ1, 新しいタグ2, ...")
            new_tags_layout.addWidget(self.new_tags_input)
        # ボタン
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        self.select_all_button = QPushButton("すべて選択")
        self.select_all_button.clicked.connect(self._select_all_tags)
        button_layout.addWidget(self.select_all_button)
        self.clear_all_button = QPushButton("すべて解除")
        self.clear_all_button.clicked.connect(self._clear_all_tags)
        button_layout.addWidget(self.clear_all_button)
        button_layout.addStretch()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
    # _load_existing_tagsは不要になりました
        if self.allow_new_tags:
            new_tags_group = QGroupBox("新しいタグを追加（カンマ区切り）")
            main_layout.addWidget(new_tags_group)
            
            new_tags_layout = QVBoxLayout(new_tags_group)
            self.new_tags_input = QLineEdit()
            self.new_tags_input.setPlaceholderText("新しいタグ1, 新しいタグ2, ...")
            new_tags_layout.addWidget(self.new_tags_input)
            
    def _load_existing_tags(self):
        """データベースから既存のタグを読み込んでチェックボックスを作成"""
        if not self.db_manager:
            return
        
        try:
            # データベースからすべてのタグを取得
            cursor = self.db_manager.conn.cursor()
            cursor.execute("SELECT DISTINCT tag FROM file_tags ORDER BY tag")
            existing_tags = [row[0] for row in cursor.fetchall()]
            
            if not existing_tags:
                no_tags_label = QLabel("既存のタグがありません")
                no_tags_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.checkbox_layout.addWidget(no_tags_label, 0, 0, 1, 2)
                return
            
            # チェックボックスを2列で配置
            row = 0
            col = 0
            
            for tag in existing_tags:
                checkbox = QCheckBox(tag)
                
                # 現在のタグが選択されている場合はチェック
                if tag in self.current_tags:
                    checkbox.setChecked(True)
                    self.selected_existing_tags.add(tag)
                
                checkbox.stateChanged.connect(
                    lambda state, t=tag: self._on_tag_checkbox_changed(state, t)
                )
                
                self.tag_checkboxes[tag] = checkbox
                self.checkbox_layout.addWidget(checkbox, row, col)
                
                col += 1
                if col >= 2:  # 2列で配置
                    col = 0
                    row += 1
        
        except Exception as e:
            error_label = QLabel(f"タグの読み込みエラー: {e}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.checkbox_layout.addWidget(error_label, 0, 0, 1, 2)
    
    def _on_tag_checkbox_changed(self, state, tag):
        # 3状態: チェック(選択), 未チェック(除外), 中間(無視)
        pass  # 状態はget_selected_tagsで取得するのでここでは何もしない
    
    def _select_all_tags(self):
        """すべてのタグを選択"""
        for checkbox in self.tag_checkboxes.values():
            checkbox.setChecked(True)
    
    def _clear_all_tags(self):
        """すべてのタグの選択を解除"""
        for checkbox in self.tag_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_tags(self):
        """3状態チェックボックスの状態を返す
        戻り値: (include_tags, exclude_tags)
        include_tags: Qt.Checked(選択)のタグ集合
        exclude_tags: Qt.Unchecked(除外)のタグ集合
        """
        include_tags = set()
        exclude_tags = set()
        for tag, checkbox in self.tag_checkboxes.items():
            state = checkbox.checkState()
            if state == Qt.Checked:
                include_tags.add(tag)
            elif state == Qt.Unchecked:
                exclude_tags.add(tag)
            # Qt.PartiallyCheckedは無視
        # 新規タグ(追加時のみ)
        if self.allow_new_tags and hasattr(self, 'new_tags_input'):
            new_tags_text = self.new_tags_input.text().strip()
            if new_tags_text:
                for tag in [t.strip() for t in new_tags_text.split(',') if t.strip()]:
                    include_tags.add(tag)
        return include_tags, exclude_tags

# image_feature_manager.py への追加コード
# ファイル種別フィルター機能の実装

from PySide6.QtWidgets import (
    # 既存のインポートに追加
    QButtonGroup, QRadioButton
)

# === ファイル種別フィルター用のダイアログクラス ===
class FileTypeFilterDialog(QDialog):
    """ファイル種別によるフィルタリングを行うダイアログ"""
    
    def __init__(self, parent=None, db_manager=None):
        super().__init__(parent)
        self.db_manager = db_manager
        
        self.setWindowTitle("ファイル種別フィルター")
        self.setModal(True)
        self.resize(400, 350)
        
        self._init_ui()
        self._load_file_type_statistics()
    
    def _init_ui(self):
        """UIの初期化"""
        main_layout = QVBoxLayout(self)
        
        # 説明ラベル
        description_label = QLabel("表示するファイル種別を選択してください：")
        main_layout.addWidget(description_label)
        
        # ファイル種別選択エリア
        filter_group = QGroupBox("ファイル種別")
        main_layout.addWidget(filter_group)
        
        filter_layout = QVBoxLayout(filter_group)
        
        # ラジオボタングループ
        self.filter_button_group = QButtonGroup(self)
        
        # 全て表示
        self.all_files_radio = QRadioButton("すべてのファイル")
        self.all_files_radio.setChecked(True)
        self.filter_button_group.addButton(self.all_files_radio, 0)
        filter_layout.addWidget(self.all_files_radio)
        
        # 画像ファイルのみ
        self.image_files_radio = QRadioButton("画像ファイルのみ")
        self.filter_button_group.addButton(self.image_files_radio, 1)
        filter_layout.addWidget(self.image_files_radio)
        
        # 動画ファイルのみ
        self.video_files_radio = QRadioButton("動画ファイルのみ")
        self.filter_button_group.addButton(self.video_files_radio, 2)
        filter_layout.addWidget(self.video_files_radio)
        
        # サムネイル対応ファイルのみ
        self.thumbnail_supported_radio = QRadioButton("サムネイル表示可能ファイル")
        self.filter_button_group.addButton(self.thumbnail_supported_radio, 3)
        filter_layout.addWidget(self.thumbnail_supported_radio)
        
        # CLIP対応ファイルのみ
        self.clip_supported_radio = QRadioButton("CLIP特徴量対応ファイル")
        self.filter_button_group.addButton(self.clip_supported_radio, 4)
        filter_layout.addWidget(self.clip_supported_radio)
        
        # 統計情報表示エリア
        self.stats_group = QGroupBox("ファイル統計")
        main_layout.addWidget(self.stats_group)
        
        stats_layout = QVBoxLayout(self.stats_group)
        self.stats_label = QLabel("統計情報を読み込み中...")
        stats_layout.addWidget(self.stats_label)
        
        # 拡張子別詳細フィルター
        extension_group = QGroupBox("拡張子別フィルター（オプション）")
        main_layout.addWidget(extension_group)
        
        extension_layout = QVBoxLayout(extension_group)
        
        # 拡張子チェックボックス用のスクロールエリア
        self.extension_scroll_area = QScrollArea()
        self.extension_scroll_area.setWidgetResizable(True)
        self.extension_scroll_area.setMaximumHeight(150)
        extension_layout.addWidget(self.extension_scroll_area)
        
        self.extension_widget = QWidget()
        self.extension_layout = QGridLayout(self.extension_widget)
        self.extension_scroll_area.setWidget(self.extension_widget)
        
        self.extension_checkboxes = {}
        
        # すべて選択/解除ボタン
        extension_button_layout = QHBoxLayout()
        self.select_all_ext_button = QPushButton("すべて選択")
        self.select_all_ext_button.clicked.connect(self._select_all_extensions)
        extension_button_layout.addWidget(self.select_all_ext_button)
        
        self.clear_all_ext_button = QPushButton("すべて解除")
        self.clear_all_ext_button.clicked.connect(self._clear_all_extensions)
        extension_button_layout.addWidget(self.clear_all_ext_button)
        
        extension_button_layout.addStretch()
        extension_layout.addLayout(extension_button_layout)
        
        # メインボタン
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        
        button_layout.addStretch()
        
        self.apply_button = QPushButton("フィルター適用")
        self.apply_button.clicked.connect(self.accept)
        button_layout.addWidget(self.apply_button)
        
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
    
    def _load_file_type_statistics(self):
        """データベースからファイル種別の統計情報を読み込み"""
        if not self.db_manager:
            self.stats_label.setText("データベースが開かれていません")
            return
        
        try:
            # 全ファイル情報を取得
            all_files = self.db_manager.get_all_file_metadata()
            
            if not all_files:
                self.stats_label.setText("データベースにファイルがありません")
                return
            
            # 種別ごとの統計を計算
            stats = {
                'total': len(all_files),
                'image': 0,
                'video': 0,
                'thumbnail_supported': 0,
                'clip_supported': 0,
                'other': 0,
                'extensions': {}
            }
            
            for file_info in all_files:
                file_path = file_info.get('file_path', '')
                
                # 拡張子統計
                ext = Path(file_path).suffix.lower()
                if ext:
                    stats['extensions'][ext] = stats['extensions'].get(ext, 0) + 1
                
                # 種別統計
                if FileTypeValidator.is_image_file(file_path):
                    stats['image'] += 1
                elif FileTypeValidator.is_video_file(file_path):
                    stats['video'] += 1
                else:
                    stats['other'] += 1
                
                if FileTypeValidator.supports_thumbnail(file_path):
                    stats['thumbnail_supported'] += 1
                
                if FileTypeValidator.supports_clip_features(file_path):
                    stats['clip_supported'] += 1
            
            # 統計情報の表示
            stats_text = f"""全ファイル数: {stats['total']} 件
画像ファイル: {stats['image']} 件
動画ファイル: {stats['video']} 件  
その他: {stats['other']} 件
サムネイル対応: {stats['thumbnail_supported']} 件
CLIP対応: {stats['clip_supported']} 件"""
            
            self.stats_label.setText(stats_text)
            
            # 拡張子チェックボックスを作成
            self._create_extension_checkboxes(stats['extensions'])
            
        except Exception as e:
            self.stats_label.setText(f"統計情報の読み込みエラー: {e}")
    
    def _create_extension_checkboxes(self, extension_stats):
        """拡張子別チェックボックスを作成"""
        # 拡張子を使用頻度順でソート
        sorted_extensions = sorted(extension_stats.items(), key=lambda x: x[1], reverse=True)
        
        row = 0
        col = 0
        
        for ext, count in sorted_extensions:
            checkbox = QCheckBox(f"{ext} ({count})")
            checkbox.setChecked(True)  # デフォルトで全て選択
            
            self.extension_checkboxes[ext] = checkbox
            self.extension_layout.addWidget(checkbox, row, col)
            
            col += 1
            if col >= 3:  # 3列で配置
                col = 0
                row += 1
    
    def _select_all_extensions(self):
        """すべての拡張子を選択"""
        for checkbox in self.extension_checkboxes.values():
            checkbox.setChecked(True)
    
    def _clear_all_extensions(self):
        """すべての拡張子の選択を解除"""
        for checkbox in self.extension_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_filter_settings(self):
        """選択されたフィルター設定を取得"""
        selected_button_id = self.filter_button_group.checkedId()
        
        # 選択された拡張子を取得
        selected_extensions = set()
        for ext, checkbox in self.extension_checkboxes.items():
            if checkbox.isChecked():
                selected_extensions.add(ext)
        
        return {
            'filter_type': selected_button_id,  # 0:全て, 1:画像, 2:動画, 3:サムネイル対応, 4:CLIP対応
            'selected_extensions': selected_extensions
        }

class FilterSidePanel(QWidget):
    """タグとファイル種別フィルターのサイドパネル"""
    
    filter_changed = Signal()  # フィルター条件が変更されたときのシグナル
    
    def __init__(self, parent=None, db_manager=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.selected_tags = set()
        self.excluded_tags = set()
        self.selected_file_types = set()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # ファイル種別フィルター
        file_type_group = QGroupBox("ファイル種別")
        file_type_layout = QVBoxLayout(file_type_group)
        
        self.file_type_checkboxes = {}
        file_types = [
            ("画像ファイル", "image"),
            ("動画ファイル", "video"),
            ("サムネイル対応", "thumbnail"),
            ("CLIP対応", "clip")
        ]
        
        for label, type_id in file_types:
            cb = QCheckBox(label)
            cb.stateChanged.connect(self._on_filter_changed)
            self.file_type_checkboxes[type_id] = cb
            file_type_layout.addWidget(cb)
        
        layout.addWidget(file_type_group)
        
        # タグフィルター
        tag_group = QGroupBox("タグフィルター")
        tag_layout = QVBoxLayout(tag_group)
        
        # タグ検索
        self.tag_search = QLineEdit()
        self.tag_search.setPlaceholderText("タグを検索...")
        self.tag_search.textChanged.connect(self._filter_tag_list)
        tag_layout.addWidget(self.tag_search)
        
        # タグリスト（スクロール可能）
        self.tag_list = QScrollArea()
        self.tag_list.setWidgetResizable(True)
        self.tag_widget = QWidget()
        self.tag_layout = QVBoxLayout(self.tag_widget)
        self.tag_list.setWidget(self.tag_widget)
        self.tag_list.setMinimumHeight(300)
        tag_layout.addWidget(self.tag_list)
        
        # タグ選択ボタン
        button_layout = QHBoxLayout()
        self.select_all_tags = QPushButton("すべて選択")
        self.clear_all_tags = QPushButton("選択解除")
        self.select_all_tags.clicked.connect(self._select_all_tags)
        self.clear_all_tags.clicked.connect(self._clear_all_tags)
        button_layout.addWidget(self.select_all_tags)
        button_layout.addWidget(self.clear_all_tags)
        tag_layout.addLayout(button_layout)
        
        layout.addWidget(tag_group)
        
        # フィルタークリアボタン
        self.clear_filters_button = QPushButton("フィルターをクリア")
        self.clear_filters_button.clicked.connect(self._clear_all_filters)
        layout.addWidget(self.clear_filters_button)
        
        layout.addStretch()

    def update_tag_list(self):
        """データベースから最新のタグリストを取得して表示を更新"""
        if not self.db_manager:
            return
        
        # 既存のタグチェックボックスをクリア
        for i in reversed(range(self.tag_layout.count())):
            widget = self.tag_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # データベースから全タグを取得
        try:
            cursor = self.db_manager.conn.cursor()
            cursor.execute("""
                SELECT tag, COUNT(DISTINCT file_path) as count 
                FROM file_tags 
                GROUP BY tag 
                ORDER BY count DESC, tag
            """)
            
            self.tag_checkboxes = {}
            for tag, count in cursor.fetchall():
                checkbox = QCheckBox(f"{tag} ({count})")
                checkbox.setTristate(True)
                checkbox.tag_name = tag  # タグ名を保存
                checkbox.stateChanged.connect(self._on_filter_changed)
                self.tag_checkboxes[tag] = checkbox
                self.tag_layout.addWidget(checkbox)
            
            self.tag_layout.addStretch()
            
        except Exception as e:
            print(f"タグリストの更新エラー: {e}")

    def get_current_filters(self):
        """現在のフィルター設定を取得"""
        file_types = set()
        for type_id, cb in self.file_type_checkboxes.items():
            if cb.isChecked():
                file_types.add(type_id)
        
        include_tags = set()
        exclude_tags = set()
        for tag, cb in self.tag_checkboxes.items():
            if cb.checkState() == Qt.Checked:
                include_tags.add(tag)
            elif cb.checkState() == Qt.Unchecked:
                exclude_tags.add(tag)
        
        return {
            'file_types': file_types,
            'include_tags': include_tags,
            'exclude_tags': exclude_tags
        }

    def _on_filter_changed(self):
        """フィルター条件が変更されたときの処理"""
        self.filter_changed.emit()

    def _filter_tag_list(self, search_text):
        """タグリストを検索テキストでフィルタリング"""
        search_text = search_text.lower()
        for tag, checkbox in self.tag_checkboxes.items():
            checkbox.setVisible(search_text in tag.lower())

    def _select_all_tags(self):
        """すべてのタグを選択"""
        for checkbox in self.tag_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setCheckState(Qt.Checked)

    def _clear_all_tags(self):
        """すべてのタグの選択を解除"""
        for checkbox in self.tag_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setCheckState(Qt.PartiallyChecked)

    def _clear_all_filters(self):
        """すべてのフィルターをクリア"""
        # ファイル種別フィルターをクリア
        for checkbox in self.file_type_checkboxes.values():
            checkbox.setChecked(False)
        
        # タグフィルターをクリア
        for checkbox in self.tag_checkboxes.values():
            checkbox.setCheckState(Qt.PartiallyChecked)
        
        self.filter_changed.emit()

class FilterSidePanel(QWidget):
    """タグとファイル種別フィルターのサイドパネル"""
    
    filter_changed = Signal()  # フィルター条件が変更されたときのシグナル
    
    def __init__(self, parent=None, db_manager=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.selected_tags = set()
        self.excluded_tags = set()
        self.selected_file_types = set()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # ファイル種別フィルター
        file_type_group = QGroupBox("ファイル種別")
        file_type_layout = QVBoxLayout(file_type_group)
        
        self.file_type_checkboxes = {}
        file_types = [
            ("画像ファイル", "image"),
            ("動画ファイル", "video"),
            ("サムネイル対応", "thumbnail"),
            ("CLIP対応", "clip")
        ]
        
        for label, type_id in file_types:
            cb = QCheckBox(label)
            cb.stateChanged.connect(self._on_filter_changed)
            self.file_type_checkboxes[type_id] = cb
            file_type_layout.addWidget(cb)
        
        layout.addWidget(file_type_group)
        
        # タグフィルター
        tag_group = QGroupBox("タグフィルター")
        tag_layout = QVBoxLayout(tag_group)
        
        # タグ検索
        self.tag_search = QLineEdit()
        self.tag_search.setPlaceholderText("タグを検索...")
        self.tag_search.textChanged.connect(self._filter_tag_list)
        tag_layout.addWidget(self.tag_search)
        
        # タグリスト（スクロール可能）
        self.tag_list = QScrollArea()
        self.tag_list.setWidgetResizable(True)
        self.tag_widget = QWidget()
        self.tag_layout = QVBoxLayout(self.tag_widget)
        self.tag_list.setWidget(self.tag_widget)
        self.tag_list.setMinimumHeight(300)

        # チェックボックスのスタイルを設定
        checkbox_style = """
            QCheckBox::indicator:indeterminate {
                background-color: #ffcccc;
                border: 2px solid #ff0000;
            }
        """
        self.tag_widget.setStyleSheet(checkbox_style)

        tag_layout.addWidget(self.tag_list)
        
        # タグ選択ボタン
        button_layout = QHBoxLayout()
        self.select_all_tags = QPushButton("すべて選択")
        self.clear_all_tags = QPushButton("選択解除")
        self.select_all_tags.clicked.connect(self._select_all_tags)
        self.clear_all_tags.clicked.connect(self._clear_all_tags)
        button_layout.addWidget(self.select_all_tags)
        button_layout.addWidget(self.clear_all_tags)
        tag_layout.addLayout(button_layout)
        
        layout.addWidget(tag_group)
        
        # フィルタークリアボタン
        self.clear_filters_button = QPushButton("フィルターをクリア")
        self.clear_filters_button.clicked.connect(self._clear_all_filters)
        layout.addWidget(self.clear_filters_button)
        
        layout.addStretch()

    def update_tag_list(self):
        """データベースから最新のタグリストを取得して表示を更新"""
        if not self.db_manager:
            return
        
        # 既存のタグチェックボックスをクリア
        for i in reversed(range(self.tag_layout.count())):
            widget = self.tag_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # データベースから全タグを取得
        try:
            cursor = self.db_manager.conn.cursor()
            cursor.execute("""
                SELECT tag, COUNT(DISTINCT file_path) as count 
                FROM file_tags 
                GROUP BY tag 
                ORDER BY count DESC, tag
            """)
            
            self.tag_checkboxes = {}
            for tag, count in cursor.fetchall():
                checkbox = QCheckBox(f"{tag} ({count})")
                checkbox.setTristate(True)
                checkbox.tag_name = tag  # タグ名を保存
                checkbox.stateChanged.connect(self._on_filter_changed)
                self.tag_checkboxes[tag] = checkbox
                self.tag_layout.addWidget(checkbox)
            
            self.tag_layout.addStretch()
            
        except Exception as e:
            print(f"タグリストの更新エラー: {e}")

    def get_current_filters(self):
        """現在のフィルター設定を取得"""
        file_types = set()
        for type_id, cb in self.file_type_checkboxes.items():
            if cb.isChecked():
                file_types.add(type_id)
        
        include_tags = set()
        exclude_tags = set()
        for tag, cb in self.tag_checkboxes.items():
            if cb.checkState() == Qt.Checked:
                include_tags.add(tag)
            elif cb.checkState() == Qt.PartiallyChecked: # Hide if intermediate state
                exclude_tags.add(tag)
        
        return {
            'file_types': file_types,
            'include_tags': include_tags,
            'exclude_tags': exclude_tags
        }

    def _on_filter_changed(self):
        """フィルター条件が変更されたときの処理"""
        self.filter_changed.emit()

    def _filter_tag_list(self, search_text):
        """タグリストを検索テキストでフィルタリング"""
        search_text = search_text.lower()
        for tag, checkbox in self.tag_checkboxes.items():
            checkbox.setVisible(search_text in tag.lower())

    def _select_all_tags(self):
        """すべてのタグを選択"""
        for checkbox in self.tag_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setCheckState(Qt.Checked)

    def _clear_all_tags(self):
        """すべてのタグの選択を解除"""
        for checkbox in self.tag_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setCheckState(Qt.PartiallyChecked)

    def _clear_all_filters(self):
        """すべてのフィルターをクリア"""
        # ファイル種別フィルターをクリア
        for checkbox in self.file_type_checkboxes.values():
            checkbox.setChecked(False)
        
        # タグフィルターをクリア
        for checkbox in self.tag_checkboxes.values():
            checkbox.setCheckState(Qt.Unchecked)
        
        self.filter_changed.emit()


# --- 4. メインアプリケーションウィンドウ ---
class ImageFeatureViewerApp(QMainWindow):
    SETTINGS_FILE = "image_feature_manager.config.json"
    SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("画像特徴量検索＆管理")
        self.setGeometry(100, 100, 1200, 800)

        self.db_path = None
        self.db_manager = None # メインスレッド用のDBManagerインスタンス
        self.clip_feature_extractor = None

        self.top_n_display_count = 1000 
        self.similarity_threshold = 0.25
        self.recent_db_paths = []
        self.window_x = 100
        self.window_y = 100
        self.thumbnail_size = 100 # Default thumbnail size

        self._load_settings()

        self.move(self.window_x, self.window_y)

        # Gemini suggested code but don't know if this is needed
        # # 起動時にDBパスが設定されている場合、ここでモデルをロード
        # if self.db_path:
        #     progress_dialog = QProgressDialog(
        #         "CLIPモデルをロード中です...", 
        #         "キャンセル", 0, 0, self
        #     )
        #     progress_dialog.setWindowTitle("初期化")
        #     progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        #     progress_dialog.setCancelButton(None)
        #     progress_dialog.show()
        #     QApplication.processEvents() # UIを更新するために必要
        #     try:
        #         self.clip_feature_extractor = CLIPFeatureExtractor()
        #     except Exception as e:
        #         QMessageBox.critical(self, "エラー", f"CLIPモデルのロードに失敗しました:\n{e}")
        #         self.clip_feature_extractor = None
        #     progress_dialog.close()

        self._init_ui()
        self._init_menu()
        
        self.setAcceptDrops(True)
        self.thread_pool = QThreadPool() # アプリケーション全体でスレッドプールを使用
        self.thread_pool.setMaxThreadCount(os.cpu_count() * 2 or 2) # スレッド数を調整

    # ImageFeatureViewerApp クラスに追加するメソッドとUI要素
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # 垂直から水平レイアウトに変更

        # サイドパネルの追加
        self.filter_panel = FilterSidePanel(self, self.db_manager)
        self.filter_panel.filter_changed.connect(self._apply_current_filters)
        self.filter_panel.setMaximumWidth(300)  # パネルの最大幅を設定
        main_layout.addWidget(self.filter_panel)

        # メインコンテンツ領域
        content_layout = QVBoxLayout()
        
        # 検索コントロール
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("検索キーワードを入力...")
        self.search_input.returnPressed.connect(self._perform_search)
        self.search_button = QPushButton("検索")
        self.search_button.clicked.connect(self._perform_search)
        
        # 全画像表示ボタンを追加
        self.show_all_button = QPushButton("全画像表示")
        self.show_all_button.clicked.connect(self._display_all_images_from_db)
        self.show_all_button.setEnabled(False)
        
        self.acquire_features_button = QPushButton("特徴量を取得")
        self.acquire_features_button.clicked.connect(self._acquire_missing_features)
        self.acquire_features_button.setEnabled(False)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.show_all_button)
        search_layout.addWidget(self.acquire_features_button)
        
        # サムネイルサイズコントロール
        search_layout.addStretch(1)
        search_layout.addWidget(QLabel("サムネイルサイズ:"))
        
        self.thumbnail_size_combo = QComboBox()
        self.thumbnail_size_combo.addItem("非表示", 0)
        self.thumbnail_size_combo.addItem("50px", 50)
        self.thumbnail_size_combo.addItem("100px", 100)
        self.thumbnail_size_combo.addItem("200px", 200)
        self.thumbnail_size_combo.addItem("300px", 300)
        self.thumbnail_size_combo.addItem("400px", 400)
        
        # 現在のサムネイルサイズに基づいて選択を設定
        if self.thumbnail_size == 0:
            self.thumbnail_size_combo.setCurrentIndex(0)
        elif self.thumbnail_size <= 50:
            self.thumbnail_size_combo.setCurrentIndex(1)
        elif self.thumbnail_size <= 100:
            self.thumbnail_size_combo.setCurrentIndex(2)
        elif self.thumbnail_size <= 200:
            self.thumbnail_size_combo.setCurrentIndex(3)
        elif self.thumbnail_size <= 300:
            self.thumbnail_size_combo.setCurrentIndex(4)
        else:
            self.thumbnail_size_combo.setCurrentIndex(5)
        
        self.thumbnail_size_combo.currentIndexChanged.connect(self._on_thumbnail_size_changed)
        search_layout.addWidget(self.thumbnail_size_combo)
        
        content_layout.addLayout(search_layout)



        #content_layout.addWidget(self.table_view)
        
        # メインコンテンツをメインレイアウトに追加
        main_layout.addLayout(content_layout)
        main_layout.setStretch(1, 1)  # メインコンテンツ領域を広げる

        # ステータスバー
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了。DBファイルを開いてください。")

        # ビュー切り替えボタンを追加
        view_mode_layout = QHBoxLayout()
        self.table_view_button = QPushButton()
        self.table_view_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.table_view_button.setCheckable(True)
        self.table_view_button.setChecked(True)
        self.table_view_button.clicked.connect(lambda: self._switch_view_mode(ViewMode.TABLE))
        
        self.icon_view_button = QPushButton()
        self.icon_view_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView))
        self.icon_view_button.setCheckable(True)
        self.icon_view_button.clicked.connect(lambda: self._switch_view_mode(ViewMode.ICON))
        
        view_mode_layout.addWidget(self.table_view_button)
        view_mode_layout.addWidget(self.icon_view_button)
        view_mode_layout.addStretch()
        search_layout.addLayout(view_mode_layout)

        # スタックウィジェットでビューを切り替え
        self.view_stack = QStackedWidget()

        # テーブルビュー
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.doubleClicked.connect(self._open_file_on_double_click)

        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        self.table_view.setColumnWidth(0, self.thumbnail_size)
        self.table_view.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 5)
        self.view_stack.addWidget(self.table_view)
        
        # リストビュー
        self.list_view = QListView()
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setSpacing(10)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setMovement(QListView.Movement.Static)
        self.view_stack.addWidget(self.list_view)
        
        # 共通のモデルを設定
        self.model = ImageViewModel(self, initial_thumbnail_size=self.thumbnail_size)
        self.table_view.setModel(self.model)
        self.list_view.setModel(self.model)

        content_layout.addWidget(self.view_stack)

    def _switch_view_mode(self, mode: ViewMode):
        self.model.set_view_mode(mode)
        self.table_view_button.setChecked(mode == ViewMode.TABLE)
        self.icon_view_button.setChecked(mode == ViewMode.ICON)
        self.view_stack.setCurrentIndex(mode.value)
        
        if mode == ViewMode.ICON:
            # アイコンサイズを設定
            self.list_view.setIconSize(QSize(self.thumbnail_size, self.thumbnail_size))
            self.list_view.setGridSize(QSize(self.thumbnail_size + 30, self.thumbnail_size + 50))

    def _apply_current_filters(self):
        """サイドパネルの現在のフィルター設定を適用"""
        if not self.db_manager:
            return

        try:
            self.status_bar.showMessage("フィルターを適用中...")

            # フィルター設定を取得
            filters = self.filter_panel.get_current_filters()
            
            # 全ファイル情報を取得
            all_files = self.db_manager.get_all_file_metadata()
            if not all_files:
                return

            filtered_files = []
            for file_info in all_files:
                file_path = file_info.get('file_path', '')
                
                # ファイル種別フィルター
                if filters['file_types']:
                    include_by_type = False
                    for file_type in filters['file_types']:
                        if (file_type == 'image' and FileTypeValidator.is_image_file(file_path)) or \
                        (file_type == 'video' and FileTypeValidator.is_video_file(file_path)) or \
                        (file_type == 'thumbnail' and FileTypeValidator.supports_thumbnail(file_path)) or \
                        (file_type == 'clip' and FileTypeValidator.supports_clip_features(file_path)):
                            include_by_type = True
                            break
                    if not include_by_type:
                        continue

                # タグ情報を取得
                try:
                    file_tags = set(self.db_manager.get_file_tags(file_path))
                except Exception:
                    file_tags = set()
                file_info['tags'] = file_tags

                # タグフィルター
                if filters['include_tags'] and not filters['include_tags'].issubset(file_tags):
                    continue
                if filters['exclude_tags'] and filters['exclude_tags'].intersection(file_tags):
                    continue

                file_info['score'] = None
                filtered_files.append(file_info)

            # 表示件数制限を適用
            display_files = filtered_files[:self.top_n_display_count]
            
            # モデルを更新
            self.model.set_data(display_files, len(filtered_files))

            # フィルター情報を作成
            filter_info = []
            if filters['file_types']:
                filter_info.append(f"ファイル種別: {len(filters['file_types'])}種")
            if filters['include_tags']:
                filter_info.append(f"含むタグ: {len(filters['include_tags'])}個")
            if filters['exclude_tags']:
                filter_info.append(f"除外タグ: {len(filters['exclude_tags'])}個")

            status = f"フィルター適用: {', '.join(filter_info) if filter_info else '未設定'}"
            status += f" - {len(display_files)}/{len(filtered_files)} 件表示"
            self.status_bar.showMessage(status)

        except Exception as e:
            error_msg = f"フィルター適用中にエラーが発生しました: {e}"
            print(error_msg, file=sys.stderr)
            self.status_bar.showMessage(error_msg)

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("ファイル")

        create_new_db_action = file_menu.addAction("新しいDBを作成...")
        create_new_db_action.triggered.connect(self._create_new_db_file)

        open_action = file_menu.addAction("DBを開く...")
        open_action.triggered.connect(self._open_db_file_dialog)

        self.recent_files_menu = QMenu("最近開いたファイル", self)
        file_menu.addMenu(self.recent_files_menu)
        self.recent_files_menu.aboutToShow.connect(self._populate_recent_files_menu)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("終了")
        exit_action.triggered.connect(self.close)

        settings_menu = menubar.addMenu("設定")
        set_display_count_action = settings_menu.addAction("表示件数を設定...")
        set_display_count_action.triggered.connect(self._set_display_count)
        set_threshold_action = settings_menu.addAction("類似度閾値を設定...")
        set_threshold_action.triggered.connect(self._set_similarity_threshold)

        tag_menu = menubar.addMenu("タグ")
        self.add_tag_action = tag_menu.addAction("選択したファイルにタグを追加...")
        self.add_tag_action.triggered.connect(self._add_tags_to_selected_files)
        self.add_tag_action.setEnabled(False)

        self.filter_by_tag_action = tag_menu.addAction("タグでフィルタリング...")
        self.filter_by_tag_action.triggered.connect(self._filter_files_by_tags)
        self.filter_by_tag_action.setEnabled(False)

        # 既存のメニュー初期化の後に追加
        # フィルターメニューを追加
        filter_menu = menubar.addMenu("フィルター")
        
        self.file_type_filter_action = filter_menu.addAction("ファイル種別でフィルター...")
        self.file_type_filter_action.triggered.connect(self._show_file_type_filter_dialog)
        self.file_type_filter_action.setEnabled(False)  # DB接続時に有効化
        
        filter_menu.addSeparator()
        
        self.clear_filters_action = filter_menu.addAction("フィルターをクリア")
        self.clear_filters_action.triggered.connect(self._clear_all_filters)
        self.clear_filters_action.setEnabled(False)


    def _populate_recent_files_menu(self):
        self.recent_files_menu.clear()
        if not self.recent_db_paths:
            self.recent_files_menu.addAction("最近開いたファイルはありません").setEnabled(False)
            return

        for path in self.recent_db_paths:
            action = self.recent_files_menu.addAction(os.path.basename(path))
            action.setData(path)
            action.triggered.connect(lambda checked, p=path: self._open_db(p))

    def _load_settings(self):
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.top_n_display_count = settings.get('top_n_display_count', self.top_n_display_count)
                    self.similarity_threshold = settings.get('similarity_threshold', self.similarity_threshold)
                    self.recent_db_paths = settings.get('recent_db_paths', [])
                    self.window_x = settings.get('window_x', self.window_x)
                    self.window_y = settings.get('window_y', self.window_y)
                    self.thumbnail_size = settings.get('thumbnail_size', self.thumbnail_size) # Load thumbnail size

                    last_path = settings.get('last_opened_db_path')
                    if last_path and os.path.exists(last_path):
                        self.db_path = last_path 

            except (IOError, json.JSONDecodeError) as e:
                print(f"設定ファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)

    def _save_settings(self):
        current_pos = self.pos()
        self.window_x = current_pos.x()
        self.window_y = current_pos.y()

        settings = {
            'top_n_display_count': self.top_n_display_count,
            'similarity_threshold': self.similarity_threshold,
            'recent_db_paths': self.recent_db_paths,
            'last_opened_db_path': self.db_path,
            'window_x': self.window_x,
            'window_y': self.window_y,
            'thumbnail_size': self.thumbnail_size, # Save thumbnail size
        }
        try:
            with open(self.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"設定ファイルの保存中にエラーが発生しました: {e}", file=sys.stderr)

    def _add_recent_db_path(self, path):
        if path in self.recent_db_paths:
            self.recent_db_paths.remove(path)
        self.recent_db_paths.insert(0, path)
        self.recent_db_paths = self.recent_db_paths[:10]  # 最大10件

    def _create_new_db_file(self):
        options = QFileDialog.Option.DontUseNativeDialog
        db_path, _ = QFileDialog.getSaveFileName(self, "新しいDBファイルを作成", "", "SQLite Databases (*.db);;All Files (*)", options=options)
        if db_path:
            if not db_path.lower().endswith(".db"):
                db_path += ".db"

            if os.path.exists(db_path):
                reply = QMessageBox.question(self, "確認", 
                                             f"指定されたDBファイルは既に存在します。\n'{os.path.basename(db_path)}'\n上書きしますか？",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return

            self._open_db(db_path)
            self.status_bar.showMessage(f"新しいデータベース '{os.path.basename(db_path)}' を作成し、開きました。")

    def _open_db_file_dialog(self):
        options = QFileDialog.Option.DontUseNativeDialog
        db_path, _ = QFileDialog.getOpenFileName(self, "DBファイルを開く", "", "SQLite Databases (*.db);;All Files (*)", options=options)
        if db_path:
            self._open_db(db_path)

        
    def _open_db(self, db_path):
        if not os.path.exists(db_path) and not db_path.lower().endswith(".db"):
            QMessageBox.warning(self, "エラー", f"指定されたDBファイルが見つかりません:\n{db_path}")
            return

        try:
            if self.db_manager:
                self.db_manager.close()
            
            self.db_path = db_path
            # メインスレッド用のDBManagerインスタンス
            self.db_manager = DBManager(self.db_path) 

            # サイドパネルのデータベース参照を更新
            self.filter_panel.db_manager = self.db_manager
            self.filter_panel.update_tag_list()

            if self.clip_feature_extractor is None:
                # プログレスダイアログの表示... Not working
                progress_dialog = QProgressDialog(
                    "CLIPモデルをロード中です...", 
                    "キャンセル", 0, 0, self
                )
                progress_dialog.setWindowTitle("初期化")
                progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                progress_dialog.setCancelButton(None) # キャンセルボタンを無効化
                progress_dialog.show()
                #
                # CLIPモデルの初期化
                #
                self.clip_feature_extractor = CLIPFeatureExtractor()
                # ダイアログを閉じる
                progress_dialog.close()

            self._add_recent_db_path(self.db_path)
            
            self.search_button.setEnabled(True)
            self.show_all_button.setEnabled(True)
            self.acquire_features_button.setEnabled(True)
            self.add_tag_action.setEnabled(True)
            self.filter_by_tag_action.setEnabled(True)

            total_count = self.db_manager.get_total_image_count()
            self.status_bar.showMessage(f"データベース '{os.path.basename(self.db_path)}' を開きました。全画像数: {total_count}")
            
            # データベース開示時に自動で全画像を表示
            self._display_all_images_from_db_async()
            
        except sqlite3.Error as e:
            QMessageBox.critical(self, "DB接続エラー", f"データベースへの接続に失敗しました:\n{e}")
            self.db_path = None
            self.db_manager = None
            self.status_bar.showMessage("DB未接続。")
            self.search_button.setEnabled(False)
            self.show_all_button.setEnabled(False)
            self.acquire_features_button.setEnabled(False)
            self.add_tag_action.setEnabled(False)
            self.filter_by_tag_action.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "初期化エラー", f"アプリケーションの初期化中にエラーが発生しました:\n{e}")

        # ファイル種別フィルターアクションを有効化
        if hasattr(self, 'file_type_filter_action'):
            self.file_type_filter_action.setEnabled(True)

    def _on_thumbnail_size_changed(self, index=None):
        """サムネイルサイズのコンボボックスが変更されたときのハンドラ"""
        new_size = self.thumbnail_size_combo.currentData()
        
        if new_size != self.thumbnail_size:
            self.thumbnail_size = new_size
            
            if self.thumbnail_size == 0:
                # 非表示の場合
                self.status_bar.showMessage("サムネイルを非表示に設定しました。")
                self.table_view.setColumnWidth(0, 0)  # サムネイル列を非表示
                self.table_view.setColumnHidden(0, True)  # 列を完全に隠す
                self.table_view.verticalHeader().setDefaultSectionSize(25)  # 行の高さを最小に
            else:
                # サイズ指定の場合
                self.status_bar.showMessage(f"サムネイルサイズを {self.thumbnail_size}px に設定しました。")
                self.table_view.setColumnHidden(0, False)  # 列を表示
                self.table_view.setColumnWidth(0, self.thumbnail_size)
                self.table_view.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 5)
                
                # リストビューのアイコンとグリッドサイズを更新
                icon_size = QSize(self.thumbnail_size, self.thumbnail_size)
                self.list_view.setIconSize(icon_size)
                
                # グリッドサイズを計算（アイコンサイズに余白を加算）
                grid_width = self.thumbnail_size + 60  # テキスト用の余白
                grid_height = self.thumbnail_size + 40  # ラベル用の余白
                self.list_view.setGridSize(QSize(grid_width, grid_height))
                
                # リストビューの更新を強制
                self.list_view.reset()

            # モデルにサイズ変更を通知
            self.model.set_current_thumbnail_size(self.thumbnail_size)

    def _on_thumbnail_size_changed2(self, index=None):
        """サムネイルサイズのコンボボックスが変更されたときのハンドラ"""
        new_size = self.thumbnail_size_combo.currentData()
        
        if new_size != self.thumbnail_size:
            self.thumbnail_size = new_size
            
            if self.thumbnail_size == 0:
                # 非表示の場合
                self.status_bar.showMessage("サムネイルを非表示に設定しました。")
                self.table_view.setColumnWidth(0, 0)  # サムネイル列を非表示
                self.table_view.setColumnHidden(0, True)  # 列を完全に隠す
                self.table_view.verticalHeader().setDefaultSectionSize(25)  # 行の高さを最小に
            else:
                # サイズ指定の場合
                self.status_bar.showMessage(f"サムネイルサイズを {self.thumbnail_size}px に設定しました。")
                self.table_view.setColumnHidden(0, False)  # 列を表示
                self.table_view.setColumnWidth(0, self.thumbnail_size)
                self.table_view.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 5)

            # モデルにサイズ変更を通知
            self.model.set_current_thumbnail_size(self.thumbnail_size)

    def _set_display_count(self):
        new_count, ok = QInputDialog.getInt(self, "表示件数を設定", "表示する画像の最大数:",
                                             self.top_n_display_count, 1, 10000)
        if ok:
            self.top_n_display_count = new_count
            self.status_bar.showMessage(f"表示件数を {self.top_n_display_count} に設定しました。")

    def _set_similarity_threshold(self):
        new_threshold, ok = QInputDialog.getDouble(self, "類似度閾値を設定", "類似度閾値 (0.0 - 1.0):",
                                                    self.similarity_threshold, 0.0, 1.0, 2)
        if ok:
            self.similarity_threshold = new_threshold
            self.status_bar.showMessage(f"類似度閾値を {self.similarity_threshold:.2f} に設定しました。")

    # さらに、_perform_search メソッドでもタグ取得を確実に行う修正
    def _perform_search(self):
        if not self.db_manager:
            self.status_bar.showMessage("エラー: DBが選択されていません。")
            return

        query_text = self.search_input.text().strip()
        if not query_text:
            self.status_bar.showMessage("検索キーワードを入力してください。")
            return

        try:
            self.status_bar.showMessage("検索中...")
            
            if self.clip_feature_extractor is None:
                QMessageBox.critical(self, "エラー", "CLIPモデルが初期化されていません。アプリケーションを再起動してください。")
                return

            search_feature = self.clip_feature_extractor.extract_features_from_text(query_text)
            if np.linalg.norm(search_feature) > 0:
                search_feature = search_feature / np.linalg.norm(search_feature)
            else:
                self.status_bar.showMessage("検索キーワードの特徴量が無効です。")
                return

            results = []
            all_db_data = self.db_manager.get_all_file_metadata()

            for item in all_db_data:
                file_path = item.get('file_path')
                clip_feature_blob = item.get('clip_feature_blob')

                if clip_feature_blob:
                    image_feature = blob_to_numpy(clip_feature_blob)
                    if image_feature is not None:
                        if np.linalg.norm(image_feature) > 0:
                            image_feature = image_feature / np.linalg.norm(image_feature)
                            similarity = np.dot(search_feature, image_feature.T)
                            
                            item['score'] = float(similarity)
                            # タグ情報を確実に取得して追加
                            try:
                                tags = self.db_manager.get_file_tags(file_path)
                                item['tags'] = set(tags) if tags else set()
                            except Exception as tag_error:
                                print(f"DEBUG: Tag retrieval error for {file_path}: {tag_error}")
                                item['tags'] = set()
                            results.append(item)

            total_image_count = len(all_db_data)
            
            filtered_results = [r for r in results if r['score'] >= self.similarity_threshold]
            sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)
            display_data = sorted_results[:self.top_n_display_count]
            
            self.model.set_data(display_data, total_image_count)
            self.status_bar.showMessage(f"検索完了。上位 {len(display_data)} 件を表示中。({total_image_count} 件中)")

        except sqlite3.Error as e:
            self.status_bar.showMessage(f"検索中のデータベースエラー: {e}")
            QMessageBox.critical(self, "データベースエラー", f"検索中にデータベースエラーが発生しました:\n{e}")
        except Exception as e:
            self.status_bar.showMessage(f"検索中にエラーが発生しました: {e}")
            QMessageBox.critical(self, "検索エラー", f"検索中に予期せぬエラーが発生しました:\n{e}")
            import traceback
            traceback.print_exc()  # デバッグ用

    def _acquire_missing_features(self):
        if not self.db_manager or not self.clip_feature_extractor:
            self.status_bar.showMessage("エラー: DBが選択されていないか、CLIPモデルが初期化されていません。")
            return
        
        try:
            files_without_features = self.db_manager.get_files_without_clip_features()
            
            if not files_without_features:
                self.status_bar.showMessage("特徴量がないファイルは見つかりませんでした。")
                QMessageBox.information(self, "情報", "特徴量がないファイルは見つかりませんでした。")
                return

            # 進捗ダイアログを作成
            self.progress_dialog = QProgressDialog("特徴量を抽出中...", "キャンセル", 0, len(files_without_features), self)
            self.progress_dialog.setWindowTitle("特徴量取得")
            self.progress_dialog.setModal(True)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.show()

            # シグナルエミッターを作成
            self.feature_extraction_emitter = FeatureExtractionSignalEmitter()
            self.feature_extraction_emitter.progress_update.connect(self._on_feature_extraction_progress)
            self.feature_extraction_emitter.finished.connect(self._on_feature_extraction_finished)
            self.feature_extraction_emitter.error.connect(self._on_feature_extraction_error)

            # 特徴量抽出タスクを開始
            feature_extractor = FeatureExtractor(
                self.db_path, 
                files_without_features, 
                self.feature_extraction_emitter
            )
            self.thread_pool.start(feature_extractor)
            
            self.status_bar.showMessage(f"{len(files_without_features)} 個のファイルの特徴量抽出を開始しました...")
            
        except Exception as e:
            self.status_bar.showMessage(f"特徴量取得の初期化中にエラーが発生しました: {e}")
            QMessageBox.critical(self, "特徴量取得エラー", f"特徴量取得の初期化中に予期せぬエラーが発生しました:\n{e}")

    def _on_feature_extraction_progress(self, current: int, total: int, message: str):
        """特徴量抽出の進捗更新"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(current)
            self.progress_dialog.setMaximum(total)
            self.progress_dialog.setLabelText(message)
            
            if total > 0:
                percentage = (current / total) * 100
                self.status_bar.showMessage(f"{message} ({current}/{total}, {percentage:.1f}%)")

    def _on_feature_extraction_finished(self, updated_count: int):
        """特徴量抽出完了"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            delattr(self, 'progress_dialog')
        
        self.status_bar.showMessage(f"{updated_count} 個のファイルにCLIP特徴量を追加しました。")
        QMessageBox.information(self, "完了", f"{updated_count} 個のファイルにCLIP特徴量を追加しました。")
        
        # 表示を更新
        self._display_all_images_from_db_async()

    def _on_feature_extraction_error(self, error_message: str):
        """特徴量抽出エラー"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            delattr(self, 'progress_dialog')
        
        self.status_bar.showMessage(f"特徴量取得中にエラーが発生しました: {error_message}")
        QMessageBox.critical(self, "特徴量取得エラー", f"特徴量取得中にエラーが発生しました:\n{error_message}")

    # === ImageFeatureViewerApp クラスのメソッド修正 ===
    def _add_tags_to_selected_files(self):
        """選択されたファイルにタグを追加（改良版）"""
        if not self.db_manager:
            QMessageBox.warning(self, "エラー", "DBが開かれていません。")
            return
        
        selected_indexes = self.table_view.selectionModel().selectedRows()
        if not selected_indexes:
            QMessageBox.information(self, "情報", "タグを追加するファイルを選択してください。")
            return

        # 既存のタグ選択ダイアログを表示
        dialog = TagSelectionDialog(
            parent=self,
            db_manager=self.db_manager,
            title="タグの追加",
            allow_new_tags=True
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_tags = dialog.get_selected_tags()
            
            if not selected_tags:
                QMessageBox.information(self, "情報", "タグが選択されていません。")
                return

            updated_count = 0
            for index in selected_indexes:
                row_data = self.model.get_row_data(index.row())
                if row_data and 'file_path' in row_data:
                    file_path = row_data['file_path']
                    
                    try:
                        for tag in selected_tags:
                            self.db_manager.add_tag_to_file(file_path, tag)
                        updated_count += 1
                    except Exception as e:
                        print(f"タグ更新エラー ({file_path}): {e}")
                        self.status_bar.showMessage(f"タグ更新エラー ({os.path.basename(file_path)}): {e}")

            if updated_count > 0:
                QMessageBox.information(self, "完了", f"{updated_count} 個のファイルに {len(selected_tags)} 個のタグを追加しました。")
                # 表示を再読み込み
                self._display_all_images_from_db_async()
            else:
                QMessageBox.warning(self, "警告", "タグの追加に失敗しました。")

    def _filter_files_by_tags(self):
        """タグによるファイルフィルタリング（3状態対応版）"""
        if not self.db_manager:
            QMessageBox.warning(self, "エラー", "DBが開かれていません。")
            return
        dialog = TagSelectionDialog(
            parent=self,
            db_manager=self.db_manager,
            title="タグでフィルタリング",
            allow_new_tags=False
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            include_tags, exclude_tags = dialog.get_selected_tags()
            try:
                # まずinclude_tagsでフィルタ
                if include_tags:
                    file_paths = self.db_manager.search_files_by_tags(list(include_tags), match_all=True)
                else:
                    # タグ指定なしなら全ファイル
                    file_paths = [f['file_path'] for f in self.db_manager.get_all_file_metadata()]
                # 除外タグでさらにフィルタ
                if exclude_tags:
                    filtered_paths = []
                    for file_path in file_paths:
                        tags = set(self.db_manager.get_file_tags(file_path))
                        if not tags.intersection(exclude_tags):
                            filtered_paths.append(file_path)
                    file_paths = filtered_paths
                if not file_paths:
                    QMessageBox.information(self, "検索結果", "指定された条件に合致するファイルは見つかりませんでした。")
                    return
                results = []
                for file_path in file_paths:
                    file_info = self.db_manager.load_file_info(file_path)
                    if file_info:
                        file_info['score'] = None
                        results.append(file_info)
                self.model.set_data(results[:self.top_n_display_count], len(results))
                msg = f"タグフィルター完了。"
                if include_tags:
                    msg += f"選択: {', '.join(include_tags)} "
                if exclude_tags:
                    msg += f"除外: {', '.join(exclude_tags)} "
                msg += f"{len(results)} 件のファイルが見つかりました。"
                self.status_bar.showMessage(msg)
            except Exception as e:
                QMessageBox.critical(self, "タグフィルターエラー", f"タグフィルター中にエラーが発生しました:\n{e}")
    # === 使用方法 ===
    # 上記のクラスとメソッドを image_feature_manager.py の該当箇所に追加・置換してください
    # 
    # 主な変更点：
    # 1. TagSelectionDialog クラスを追加
    # 2. _add_tags_to_selected_files メソッドを置換
    # 3. _filter_files_by_tags メソッドを置換
    # 4. インポート文に QDialog, QCheckBox, QScrollArea, QGridLayout, QGroupBox を追加

    def _open_file_on_double_click(self, index: QModelIndex):
        if index.isValid():
            row_data = self.model.get_row_data(index.row())
            if row_data and 'file_path' in row_data:
                file_path = row_data['file_path']
                if os.path.exists(file_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                else:
                    QMessageBox.warning(self, "ファイルが見つかりません", f"ファイルが見つかりません:\n{file_path}")

    def _display_all_images_from_db_async(self):
        """バックグラウンドで全画像データを非同期ロード"""
        if not self.db_manager:
            return
        
        # 進捗ダイアログを作成
        self.all_images_progress_dialog = QProgressDialog("画像データを読み込み中...", "キャンセル", 0, 0, self)
        self.all_images_progress_dialog.setWindowTitle("データ読み込み")
        self.all_images_progress_dialog.setModal(True)
        self.all_images_progress_dialog.setMinimumDuration(500)  # 500ms後に表示
        
        # シグナルエミッターを作成
        self.all_images_load_emitter = AllImagesLoadSignalEmitter()
        self.all_images_load_emitter.progress_update.connect(self._on_all_images_load_progress)
        self.all_images_load_emitter.finished.connect(self._on_all_images_load_finished)
        self.all_images_load_emitter.error.connect(self._on_all_images_load_error)
        
        # 全画像ローダータスクを開始
        all_images_loader = AllImagesLoader(
            self.db_path,
            self.all_images_load_emitter,
            self.top_n_display_count
        )
        self.thread_pool.start(all_images_loader)
        
        self.status_bar.showMessage("画像データを読み込み中...")

    def _display_all_images_from_db(self):
        """同期版：全画像表示（ボタンから呼び出される）"""
        self._display_all_images_from_db_async()

    def _on_all_images_load_progress(self, current: int, total: int, message: str):
        """全画像ロードの進捗更新"""
        if hasattr(self, 'all_images_progress_dialog'):
            if total > 0:
                self.all_images_progress_dialog.setMaximum(total)
                self.all_images_progress_dialog.setValue(current)
                percentage = (current / total) * 100
                self.status_bar.showMessage(f"{message} ({current}/{total}, {percentage:.1f}%)")
            else:
                self.all_images_progress_dialog.setRange(0, 0)  # 不定長進捗バー
                self.status_bar.showMessage(message)
            
            self.all_images_progress_dialog.setLabelText(message)

    def _on_all_images_load_finished(self, data_list: list, total_count: int):
        """全画像ロード完了"""
        if hasattr(self, 'all_images_progress_dialog'):
            self.all_images_progress_dialog.close()
            delattr(self, 'all_images_progress_dialog')
        
        self.model.set_data(data_list, total_count)
        
        displayed_count = len(data_list)
        if displayed_count < total_count:
            self.status_bar.showMessage(f"全画像を表示しました。{displayed_count}/{total_count} 件表示（表示件数制限: {self.top_n_display_count}）")
        else:
            self.status_bar.showMessage(f"全画像を表示しました。合計 {total_count} 件")

    def _on_all_images_load_error(self, error_message: str):
        """全画像ロードエラー"""
        if hasattr(self, 'all_images_progress_dialog'):
            self.all_images_progress_dialog.close()
            delattr(self, 'all_images_progress_dialog')
        
        self.status_bar.showMessage(f"画像データの読み込み中にエラーが発生しました: {error_message}")
        QMessageBox.critical(self, "データ読み込みエラー", f"画像データの読み込み中にエラーが発生しました:\n{error_message}")

    # === ImageFeatureViewerApp クラスへの追加メソッド ===

    def _show_file_type_filter_dialog(self):
        """ファイル種別フィルターダイアログを表示"""
        if not self.db_manager:
            QMessageBox.warning(self, "エラー", "DBが開かれていません。")
            return
        
        dialog = FileTypeFilterDialog(self, self.db_manager)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filter_settings = dialog.get_filter_settings()
            self._apply_file_type_filter(filter_settings)

    def _apply_file_type_filter(self, filter_settings):
        """ファイル種別フィルターを適用"""
        if not self.db_manager:
            return
        
        try:
            self.status_bar.showMessage("ファイル種別フィルターを適用中...")
            
            # 全ファイル情報を取得
            all_files = self.db_manager.get_all_file_metadata()
            
            if not all_files:
                QMessageBox.information(self, "情報", "データベースにファイルがありません。")
                return
            
            filtered_files = []
            filter_type = filter_settings['filter_type']
            selected_extensions = filter_settings['selected_extensions']
            
            for file_info in all_files:
                file_path = file_info.get('file_path', '')
                ext = Path(file_path).suffix.lower()
                
                # 拡張子フィルターをチェック
                if selected_extensions and ext not in selected_extensions:
                    continue
                
                # ファイル種別フィルターをチェック
                if filter_type == 0:  # すべてのファイル
                    include_file = True
                elif filter_type == 1:  # 画像ファイルのみ
                    include_file = FileTypeValidator.is_image_file(file_path)
                elif filter_type == 2:  # 動画ファイルのみ
                    include_file = FileTypeValidator.is_video_file(file_path)
                elif filter_type == 3:  # サムネイル対応ファイル
                    include_file = FileTypeValidator.supports_thumbnail(file_path)
                elif filter_type == 4:  # CLIP対応ファイル
                    include_file = FileTypeValidator.supports_clip_features(file_path)
                else:
                    include_file = True
                
                if include_file:
                    # タグ情報を追加
                    try:
                        tags = self.db_manager.get_file_tags(file_path)
                        file_info['tags'] = set(tags) if tags else set()
                    except Exception:
                        file_info['tags'] = set()
                    
                    file_info['score'] = None  # 検索ではないのでスコアはNone
                    filtered_files.append(file_info)
            
            # 表示件数制限を適用
            display_files = filtered_files[:self.top_n_display_count]
            
            # モデルを更新
            self.model.set_data(display_files, len(filtered_files))
            
            # フィルター情報を作成
            filter_type_names = {
                0: "すべて",
                1: "画像ファイル", 
                2: "動画ファイル",
                3: "サムネイル対応",
                4: "CLIP対応"
            }
            
            filter_desc = filter_type_names.get(filter_type, "不明")
            
            if selected_extensions:
                ext_desc = f"拡張子: {', '.join(sorted(selected_extensions))}"
                if len(selected_extensions) > 5:
                    ext_desc = f"拡張子: {len(selected_extensions)}種類選択"
            else:
                ext_desc = "全拡張子"
            
            self.status_bar.showMessage(
                f"フィルター適用完了: {filter_desc}, {ext_desc} - "
                f"{len(display_files)}/{len(filtered_files)} 件表示"
            )
            
            # フィルタークリアボタンを有効化
            if hasattr(self, 'clear_filters_action'):
                self.clear_filters_action.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "フィルターエラー", f"ファイル種別フィルター中にエラーが発生しました:\n{e}")
            self.status_bar.showMessage(f"フィルターエラー: {e}")

    def _clear_all_filters(self):
        """すべてのフィルターをクリアして全画像を表示"""
        self._display_all_images_from_db_async()
        if hasattr(self, 'clear_filters_action'):
            self.clear_filters_action.setEnabled(False)


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.db_manager:
            QMessageBox.warning(self, "エラー", "DBが開かれていません。先にDBを開くか作成してください。")
            event.ignore()
            return
            
        if event.mimeData().hasUrls():
            dropped_paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    dropped_paths.append(url.toLocalFile())
            
            if not dropped_paths:
                event.ignore()
                return

            image_paths_to_add = []
            folders_to_process = []

            for path in dropped_paths:
                if os.path.isfile(path):
                    if path.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                        image_paths_to_add.append(path)
                elif os.path.isdir(path):
                    folders_to_process.append(path)
            
            if folders_to_process:
                reply = QMessageBox.question(
                    self,
                    "フォルダの処理",
                    f"{len(folders_to_process)} 個のフォルダがドロップされました。サブフォルダ内の画像も追加しますか？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )

                if reply == QMessageBox.StandardButton.Cancel:
                    event.ignore()
                    return

                recursive = (reply == QMessageBox.StandardButton.Yes)
                
                self.status_bar.showMessage("フォルダ内の画像ファイルをスキャン中...")
                QApplication.processEvents()

                for folder_path in folders_to_process:
                    for root, _, files in os.walk(folder_path):
                        for file_name in files:
                            if file_name.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS):
                                full_path = os.path.join(root, file_name)
                                if not self.db_manager.load_file_info(full_path): 
                                    image_paths_to_add.append(full_path)
                        if not recursive:
                            break

            if image_paths_to_add:
                image_paths_to_add = list(set(image_paths_to_add))
                if not image_paths_to_add:
                    self.status_bar.showMessage("追加する新しい画像ファイルは見つかりませんでした。")
                    event.acceptProposedAction()
                    return

                adder_emitter = ImageAddSignalEmitter()
                adder_emitter.progress_update.connect(self.status_bar.showMessage)
                adder_emitter.finished.connect(self._on_image_add_finished)
                adder_emitter.error.connect(lambda msg: QMessageBox.warning(self, "ファイル追加エラー", msg))

                adder = ImageAdder(self.db_path, image_paths_to_add, adder_emitter)
                self.thread_pool.start(adder)
                self.status_bar.showMessage(f"{len(image_paths_to_add)} 個の画像のデータベースへの追加を開始しました...")

            else:
                self.status_bar.showMessage("ドロップされたファイルの中にサポートされている画像ファイルは見つかりませんでした。")
            
            event.acceptProposedAction()
        else:
            event.ignore()

    def _on_image_add_finished(self, added_count: int):
        """ImageAdderからの完了シグナルを受け取った際の処理"""
        self.status_bar.showMessage(f"データベースに {added_count} 個の画像を追加しました。")
        QMessageBox.information(self, "画像追加完了", f"データベースに {added_count} 個の画像を新規追加しました。")
        
        total_count = self.db_manager.get_total_image_count() 
        self.status_bar.showMessage(f"データベース更新完了。全画像数: {total_count}")
        
        # 表示を更新
        self._display_all_images_from_db_async()

    def closeEvent(self, event):
        self._save_settings()
        if self.db_manager:
            self.db_manager.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageFeatureViewerApp()
    window.show()
    sys.exit(app.exec())






    