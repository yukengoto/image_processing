import sys
import os
import json
import sqlite3
import numpy as np
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableView, QLineEdit, QHeaderView,
    QStatusBar, QAbstractItemView, QMessageBox, QInputDialog, QMenu,
    QSlider, QLabel
)
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QSize,
    QThreadPool, QRunnable, Signal, QObject, QUrl
)
from PySide6.QtGui import QPixmap, QImage, QDesktopServices, QIntValidator # QIntValidatorを追加

# --- 新しいモジュールのインポート ---
from db_manager import DBManager, blob_to_numpy, numpy_to_blob
from clip_feature_extractor import CLIPFeatureExtractor

# --- 1. サムネイル生成をバックグラウンドで行うためのQRunnableとシグナルエミッター ---
class ThumbnailSignalEmitter(QObject):
    """QRunnableからQAbstractTableModelにシグナルを送るためのヘルパークラス"""
    thumbnail_ready = Signal(QModelIndex, QPixmap)
    error = Signal(str)

class ThumbnailGenerator(QRunnable):
    """画像を読み込み、サムネイルを生成するタスク (ディスクキャッシュなし)"""
    def __init__(self, image_path, size: QSize, index, signal_emitter):
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.index = index
        self.signal_emitter = signal_emitter

    def run(self):
        try:
            if not os.path.exists(self.image_path):
                error_msg = f"ERROR: サムネイル生成を試みましたが、ファイルが見つかりません: {self.image_path}"
                print(error_msg, file=sys.stderr)
                self.signal_emitter.error.emit(f"ファイルが見つかりません: {os.path.basename(self.image_path)}")
                return

            original_pixmap = QPixmap(self.image_path)
            
            if original_pixmap.isNull():
                raise ValueError(f"画像をロードできませんでした (QPixmap.isNull()): {self.image_path}")

            pixmap = original_pixmap.scaled(
                self.size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            if pixmap.isNull():
                raise ValueError(f"QPixmapのリサイズに失敗しました: {self.image_path}")

            self.signal_emitter.thumbnail_ready.emit(self.index, pixmap)
        except Exception as e:
            error_msg = f"ERROR: サムネイル生成中に予期せぬエラーが発生しました ({self.image_path}): {e}"
            print(error_msg, file=sys.stderr)
            self.signal_emitter.error.emit(f"サムネイル生成エラー ({os.path.basename(self.image_path)}): {e}")

# --- 2. データベースのデータと連携するカスタムテーブルモデル ---
class ImageTableModel(QAbstractTableModel):
    def __init__(self, parent=None, initial_thumbnail_size=100):
        super().__init__(parent)
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

    def set_data(self, data, total_count):
        self.beginResetModel()
        self._data = data
        self._total_image_count = total_count
        self.thumbnail_cache.clear()
        self.endResetModel()

    def set_current_thumbnail_size(self, size_int: int):
        new_size = QSize(size_int, size_int)
        if self.thumbnail_size != new_size:
            self.thumbnail_size = new_size
            self.thumbnail_cache.clear() # Clear cache to force re-render
            # Notify view to redraw decorations (thumbnails)
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1), [Qt.ItemDataRole.DecorationRole])

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        item_data = self._data[row]

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return ""
            elif col == 1:
                return os.path.basename(item_data.get('file_path', ''))
            elif col == 2:
                score = item_data.get('score')
                return f"{score:.4f}" if score is not None else ""
            elif col == 3:
                tags = item_data.get('tags', '')
                return tags if tags else ""
            elif col == 4:
                return item_data.get('file_path', '')

        elif role == Qt.ItemDataRole.DecorationRole and col == 0:
            file_path = item_data.get('file_path')
            if file_path in self.thumbnail_cache:
                return self.thumbnail_cache[file_path]
            else:
                if file_path and os.path.exists(file_path):
                    generator = ThumbnailGenerator(
                        image_path=file_path,
                        size=self.thumbnail_size, # Use current thumbnail size
                        index=index,
                        signal_emitter=self.thumbnail_signal_emitter
                    )
                    self.thread_pool.start(generator)
                return QPixmap()
        
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col == 0:
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignLeft

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        return None

    def update_thumbnail(self, index: QModelIndex, pixmap: QPixmap):
        file_path = self._data[index.row()].get('file_path')
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

    def run(self):
        added_count = 0
        total_files = len(self.image_paths)
        self.signal_emitter.progress_update.emit(f"データベースに {total_files} 個の画像を追加中...")

        try:
            self.db_manager = DBManager(self.db_path) # スレッド内でDBManagerをインスタンス化
            
            for i, file_path in enumerate(self.image_paths):
                if not os.path.exists(file_path):
                    self.signal_emitter.progress_update.emit(f"警告: ファイルが見つかりません。スキップします: {file_path}")
                    continue
                
                try:
                    # ファイルのメタデータを取得
                    stat_info = os.stat(file_path)
                    file_size = stat_info.st_size
                    creation_time = stat_info.st_ctime
                    last_modified_time = stat_info.st_mtime

                    # データベースに挿入または更新 (CLIP特徴量はここではNone)
                    self.db_manager.insert_or_update_file_metadata(
                        file_path=file_path,
                        last_modified=last_modified_time,
                        size=file_size,
                        creation_time=creation_time,
                        clip_feature_blob=None, # ここでは特徴量を追加しない
                        tags="", # 新規追加時はタグは空
                    )
                    added_count += 1
                    self.signal_emitter.progress_update.emit(f"追加中: {added_count}/{total_files} ({os.path.basename(file_path)})")
                except Exception as e:
                    error_msg = f"ERROR: ファイル '{file_path}' の追加中にエラーが発生しました: {e}"
                    print(error_msg, file=sys.stderr)
                    self.signal_emitter.error.emit(f"ファイル追加エラー ({os.path.basename(file_path)}): {e}")
            
            self.signal_emitter.finished.emit(added_count)
        except Exception as e:
            # DBManagerインスタンス化またはループ中の致命的なエラー
            error_msg = f"データベース処理の初期化または実行中にエラーが発生しました: {e}"
            print(error_msg, file=sys.stderr)
            self.signal_emitter.error.emit(error_msg)
        finally:
            if self.db_manager:
                self.db_manager.close() # 処理終了時に接続を閉じる


# --- 4. メインアプリケーションウィンドウ ---
class ImageFeatureViewerApp(QMainWindow):
    SETTINGS_FILE = "image_feature_manager.config.json"
    SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("画像特徴量検索＆管理")
        self.setGeometry(100, 100, 1200, 800)

        self.db_path = None
        self.db_manager = None # メインスレッドでのDBManagerインスタンス
        self.clip_feature_extractor = None

        self.top_n_display_count = 1000 
        self.similarity_threshold = 0.25
        self.recent_db_paths = []
        self.window_x = 100
        self.window_y = 100
        self.thumbnail_size = 100 # Default thumbnail size

        self._load_settings()

        self.move(self.window_x, self.window_y)

        self._init_ui()
        self._init_menu()
        
        self.setAcceptDrops(True)
        self.thread_pool = QThreadPool() # アプリケーション全体でスレッドプールを使用
        self.thread_pool.setMaxThreadCount(os.cpu_count() * 2 or 2) # スレッド数を調整

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Search and DB controls layout
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("検索キーワードを入力...")
        self.search_button = QPushButton("検索")
        self.search_button.clicked.connect(self._perform_search)
        
        # Removed self.open_db_button from here as it's in the menu
        # self.open_db_button = QPushButton("DBを開く")
        # self.open_db_button.clicked.connect(self._open_db_file_dialog)

        self.acquire_features_button = QPushButton("特徴量がないファイルを取得")
        self.acquire_features_button.clicked.connect(self._acquire_missing_features)
        self.acquire_features_button.setEnabled(False)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        # search_layout.addWidget(self.open_db_button) # Removed
        search_layout.addWidget(self.acquire_features_button)
        
        # Add thumbnail size controls to the far right of search_layout
        search_layout.addStretch(1) # Pushes subsequent widgets to the right
        search_layout.addWidget(QLabel("サムネイルサイズ:"))
        self.thumbnail_size_slider = QSlider(Qt.Horizontal)
        self.thumbnail_size_slider.setRange(50, 400) # Min 50, Max 400 pixels
        self.thumbnail_size_slider.setSingleStep(50) # Step 50
        self.thumbnail_size_slider.setTickPosition(QSlider.TicksBelow) # Show ticks
        self.thumbnail_size_slider.setTickInterval(50) # Ticks at 50, 100, etc.
        self.thumbnail_size_slider.setValue(self.thumbnail_size)
        self.thumbnail_size_slider.setFixedWidth(120) # Make slider smaller
        self.thumbnail_size_slider.valueChanged.connect(self._on_thumbnail_size_changed)
        
        self.thumbnail_size_input = QLineEdit(str(self.thumbnail_size))
        self.thumbnail_size_input.setFixedWidth(40) # Make input smaller
        self.thumbnail_size_input.setValidator(QIntValidator(50, 400)) # Ensure valid input range
        self.thumbnail_size_input.editingFinished.connect(self._on_thumbnail_size_changed)

        search_layout.addWidget(self.thumbnail_size_slider)
        search_layout.addWidget(self.thumbnail_size_input)
        
        main_layout.addLayout(search_layout)


        self.model = ImageTableModel(self, initial_thumbnail_size=self.thumbnail_size) # Pass initial size
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_view.doubleClicked.connect(self._open_file_on_double_click)

        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed) # Thumbnail column fixed
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive) # ファイル名
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive) # 類似度
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive) # タグ
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive) # パス
        self.table_view.setColumnWidth(0, self.thumbnail_size) # Set initial thumbnail column width

        self.table_view.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 5) # Set initial row height

        main_layout.addWidget(self.table_view)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了。DBファイルを開いてください。")

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
        max_recent = 10
        if len(self.recent_db_paths) > max_recent:
            self.recent_db_paths = self.recent_db_paths[:max_recent]

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
            if self.clip_feature_extractor is None:
                self.clip_feature_extractor = CLIPFeatureExtractor()

            self.status_bar.showMessage(f"データベース '{os.path.basename(self.db_path)}' を開きました。")
            self._add_recent_db_path(self.db_path)
            
            self.search_button.setEnabled(True)
            self.acquire_features_button.setEnabled(True)
            self.add_tag_action.setEnabled(True)
            self.filter_by_tag_action.setEnabled(True)

            total_count = self.db_manager.get_total_image_count()
            self.status_bar.showMessage(f"データベース '{os.path.basename(self.db_path)}' を開きました。全画像数: {total_count}")
            self.model.set_data([], total_count)
            
        except sqlite3.Error as e:
            QMessageBox.critical(self, "DB接続エラー", f"データベースへの接続に失敗しました:\n{e}")
            self.db_path = None
            self.db_manager = None
            self.status_bar.showMessage("DB未接続。")
            self.search_button.setEnabled(False)
            self.acquire_features_button.setEnabled(False)
            self.add_tag_action.setEnabled(False)
            self.filter_by_tag_action.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "初期化エラー", f"アプリケーションの初期化中にエラーが発生しました:\n{e}")

    def _on_thumbnail_size_changed(self, value=None):
        """サムネイルサイズのスライダーまたは入力フィールドが変更されたときのハンドラ"""
        new_size = self.thumbnail_size

        if isinstance(value, int): # Value from QSlider
            new_size = value
        else: # Value from QLineEdit (editingFinished)
            try:
                line_edit_value = int(self.thumbnail_size_input.text())
                # Use updated range for validation
                if 50 <= line_edit_value <= 400: 
                    new_size = line_edit_value
                else:
                    QMessageBox.warning(self, "入力エラー", "サムネイルサイズは50から400の範囲で入力してください。")
                    self.thumbnail_size_input.setText(str(self.thumbnail_size)) # Revert to current value
                    return
            except ValueError:
                QMessageBox.warning(self, "入力エラー", "有効な数値を入力してください。")
                self.thumbnail_size_input.setText(str(self.thumbnail_size)) # Revert to current value
                return
        
        # Ensure consistency across UI elements
        if self.thumbnail_size != new_size:
            self.thumbnail_size = new_size
            self.thumbnail_size_slider.setValue(self.thumbnail_size)
            self.thumbnail_size_input.setText(str(self.thumbnail_size))
            self.status_bar.showMessage(f"サムネイルサイズを {self.thumbnail_size}px に設定しました。")

            # Update model and view
            self.model.set_current_thumbnail_size(self.thumbnail_size)
            self.table_view.setColumnWidth(0, self.thumbnail_size)
            self.table_view.verticalHeader().setDefaultSectionSize(self.thumbnail_size + 5)


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
            all_db_data = self.db_manager.get_all_file_metadata() # メインスレッドからアクセス

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

    def _acquire_missing_features(self):
        if not self.db_manager or not self.clip_feature_extractor:
            self.status_bar.showMessage("エラー: DBが選択されていないか、CLIPモデルが初期化されていません。")
            return
        
        self.status_bar.showMessage("特徴量がないファイルを検索し、取得中です...")
        QApplication.processEvents()
        
        try:
            # メインスレッドからDBManagerを使ってファイルパスを取得
            files_without_features = self.db_manager.get_file_paths_without_clip_features()
            
            if not files_without_features:
                self.status_bar.showMessage("特徴量がないファイルは見つかりませんでした。")
                return

            self.status_bar.showMessage(f"{len(files_without_features)} 個のファイルの特徴量を抽出中...")
            QApplication.processEvents()

            # 特徴量抽出はCLIPFeatureExtractorが内部でスレッドセーフに処理すると仮定
            # または、必要であればこれもQRunnableでラップする必要があるかもしれません
            extracted_features, processed_indices = self.clip_feature_extractor.extract_features_from_paths(files_without_features)
            
            if len(extracted_features) == 0:
                self.status_bar.showMessage("特徴量の抽出に成功したファイルはありませんでした。")
                return

            updated_count = 0
            # メインスレッドのDBManagerを使って更新
            # 大量更新の場合は、DBManagerにバッチ更新メソッドを追加すると効率的
            for i, original_index in enumerate(processed_indices):
                file_path = files_without_features[original_index]
                feature_blob = numpy_to_blob(extracted_features[i])
                
                self.db_manager.update_file_metadata(file_path, {'clip_feature_blob': feature_blob})
                updated_count += 1
            
            self.status_bar.showMessage(f"{updated_count} 個のファイルにCLIP特徴量を追加しました。")
            QMessageBox.information(self, "完了", f"{updated_count} 個のファイルにCLIP特徴量を追加しました。")
            
            # DBに画像が追加され、特徴量も追加された可能性があるため、テーブルを再表示
            self._display_all_images_from_db()

        except Exception as e:
            self.status_bar.showMessage(f"特徴量取得中にエラーが発生しました: {e}")
            QMessageBox.critical(self, "特徴量取得エラー", f"特徴量取得中に予期せぬエラーが発生しました:\n{e}")

    def _add_tags_to_selected_files(self):
        if not self.db_manager:
            QMessageBox.warning(self, "エラー", "DBが開かれていません。")
            return
        
        selected_indexes = self.table_view.selectionModel().selectedRows()
        if not selected_indexes:
            QMessageBox.information(self, "情報", "タグを追加するファイルを選択してください。")
            return

        tags_text, ok = QInputDialog.getText(self, "タグの追加", "追加するタグをカンマ区切りで入力してください:")
        
        if ok and tags_text:
            new_tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            if not new_tags:
                QMessageBox.warning(self, "警告", "有効なタグが入力されませんでした。")
                return

            updated_count = 0
            for index in selected_indexes:
                row_data = self.model.get_row_data(index.row())
                if row_data and 'file_path' in row_data:
                    file_path = row_data['file_path']
                    current_tags_str = row_data.get('tags', '')
                    current_tags = set([t.strip() for t in current_tags_str.split(',') if t.strip()])
                    
                    updated_tags = current_tags.union(set(new_tags))
                    updated_tags_str = ','.join(sorted(list(updated_tags)))
                    
                    try:
                        self.db_manager.update_file_metadata(file_path, {'tags': updated_tags_str})
                        updated_count += 1
                    except Exception as e:
                        print(f"タグ更新エラー ({file_path}): {e}")
                        self.status_bar.showMessage(f"タグ更新エラー ({os.path.basename(file_path)}): {e}")

            if updated_count > 0:
                QMessageBox.information(self, "完了", f"{updated_count} 個のファイルにタグを追加しました。")
                self._perform_search()
            else:
                QMessageBox.warning(self, "警告", "タグの追加に失敗したファイルがあります。")

    def _filter_files_by_tags(self):
        QMessageBox.information(self, "機能開発中", "この機能はまだ実装されていません。")

    def _open_file_on_double_click(self, index: QModelIndex):
        if index.isValid():
            row_data = self.model.get_row_data(index.row())
            if row_data and 'file_path' in row_data:
                file_path = row_data['file_path']
                if os.path.exists(file_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                else:
                    QMessageBox.warning(self, "ファイルが見つかりません", f"ファイルが見つかりません:\n{file_path}")

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
        if not self.db_manager: # メインスレッドのDBManagerが存在するか確認
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
                                # 既にDBにあるファイルはスキップ (メインスレッドのDBManagerを使用)
                                if not self.db_manager.get_file_metadata(full_path): 
                                    image_paths_to_add.append(full_path)
                        if not recursive:
                            break

            if image_paths_to_add:
                # 重複を排除し、ユニークなパスのみにする
                image_paths_to_add = list(set(image_paths_to_add))
                if not image_paths_to_add:
                    self.status_bar.showMessage("追加する新しい画像ファイルは見つかりませんでした。")
                    event.acceptProposedAction()
                    return

                # ImageAdderを起動 (db_pathを渡す)
                adder_emitter = ImageAddSignalEmitter()
                adder_emitter.progress_update.connect(self.status_bar.showMessage)
                adder_emitter.finished.connect(self._on_image_add_finished)
                adder_emitter.error.connect(lambda msg: QMessageBox.warning(self, "ファイル追加エラー", msg))

                adder = ImageAdder(self.db_path, image_paths_to_add, adder_emitter) # db_pathを渡す
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
        # データベースの総画像数を更新して表示 (メインスレッドのDBManagerを使用)
        total_count = self.db_manager.get_total_image_count() 
        self.status_bar.showMessage(f"データベース更新完了。全画像数: {total_count}")
        # テーブル表示を更新（必要であれば全件表示をリフレッシュ）
        self._display_all_images_from_db()


    def _display_all_images_from_db(self):
        """データベース内の全画像をテーブルに表示（類似度順はなし）"""
        if not self.db_manager:
            return
        try:
            all_db_data = self.db_manager.get_all_file_metadata() # メインスレッドのDBManagerを使用
            total_count = len(all_db_data)
            # スコアなしで表示する場合（または適当なデフォルトスコア）
            for item in all_db_data:
                item['score'] = None # スコアは検索時のみ設定される
            
            # ファイルパスでソートして表示
            display_data = sorted(all_db_data, key=lambda x: x.get('file_path', ''))
            
            self.model.set_data(display_data[:self.top_n_display_count], total_count)
            self.status_bar.showMessage(f"全画像を再表示しました。合計 {total_count} 件中、上位 {len(display_data[:self.top_n_display_count])} 件を表示中。")
        except Exception as e:
            self.status_bar.showMessage(f"全画像表示中にエラーが発生しました: {e}")
            print(f"全画像表示エラー: {e}", file=sys.stderr)


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