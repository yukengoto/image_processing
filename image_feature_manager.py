import sys
import os
import json
import sqlite3
import numpy as np
import hashlib
from PIL import Image # サムネイル生成用 (pip install Pillow)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableView, QLineEdit, QHeaderView,
    QStatusBar, QAbstractItemView, QMessageBox, QInputDialog
)
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QSize,
    QThreadPool, QRunnable, Signal, QObject, QUrl
)
from PySide6.QtGui import QPixmap, QImage, QDesktopServices # QDesktopServices for opening files

# --- 新しいモジュールのインポート ---
from db_manager import DBManager, blob_to_numpy, numpy_to_blob # db_managerから必要な関数とクラスをインポート
from clip_feature_extractor import CLIPFeatureExtractor # CLIP特徴量抽出器をインポート

# --- 1. サムネイル生成をバックグラウンドで行うためのQRunnableとシグナルエミッター ---
class ThumbnailSignalEmitter(QObject):
    """QRunnableからQAbstractTableModelにシグナルを送るためのヘルパークラス"""
    thumbnail_ready = Signal(QModelIndex, QPixmap)
    error = Signal(str)

class ThumbnailGenerator(QRunnable):
    """画像を読み込み、サムネイルを生成してキャッシュするタスク"""
    def __init__(self, image_path, size, cache_dir, index, signal_emitter):
        super().__init__()
        self.image_path = image_path
        self.size = size
        self.cache_dir = cache_dir
        self.index = index
        self.signal_emitter = signal_emitter
        self.cache_path = os.path.join(cache_dir, hashlib.md5(image_path.encode()).hexdigest() + ".png")

    def run(self):
        try:
            # キャッシュから読み込みを試みる
            if os.path.exists(self.cache_path):
                pixmap = QPixmap(self.cache_path)
                if not pixmap.isNull():
                    self.signal_emitter.thumbnail_ready.emit(self.index, pixmap)
                    return

            # 画像ファイルが存在するか最終確認
            if not os.path.exists(self.image_path):
                error_msg = f"ERROR: サムネイル生成を試みましたが、ファイルが見つかりません: {self.image_path}"
                print(error_msg, file=sys.stderr)
                self.signal_emitter.error.emit(f"ファイルが見つかりません: {os.path.basename(self.image_path)}")
                return

            # 画像が破損している可能性があるのでtry-exceptで囲む
            img = Image.open(self.image_path).convert("RGB")
            # 修正: QSizeオブジェクトから幅と高さを抽出し、タプルとして渡す
            img.thumbnail((self.size.width(), self.size.height()), Image.Resampling.LANCZOS) # より高品質なリサンプリング

            # PySide6/Qtで表示するためにQImageに変換
            data = img.tobytes("raw", "RGB")
            qimage = QImage(data, img.size[0], img.size[1], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # キャッシュに保存
            pixmap.save(self.cache_path)

            self.signal_emitter.thumbnail_ready.emit(self.index, pixmap)
        except Exception as e:
            # エラーメッセージをコンソールに詳細に出力
            error_msg = f"ERROR: サムネイル生成中に予期せぬエラーが発生しました ({self.image_path}): {e}"
            print(error_msg, file=sys.stderr)
            self.signal_emitter.error.emit(f"サムネイル生成エラー ({os.path.basename(self.image_path)}): {e}")

# --- 2. データベースのデータと連携するカスタムテーブルモデル ---
class ImageTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []
        self._total_image_count = 0 # フィルタリング前の全画像数
        self._headers = ["", "ファイル名", "パス", "類似度", "タグ"] # ヘッダーに「タグ」を追加
        self.thumbnail_cache = {}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(os.cpu_count() or 1) # スレッドプール数をCPUコア数に設定
        self.thumbnail_cache_dir = "thumbnail_cache"
        os.makedirs(self.thumbnail_cache_dir, exist_ok=True)
        
        self.thumbnail_signal_emitter = ThumbnailSignalEmitter()
        self.thumbnail_signal_emitter.thumbnail_ready.connect(self.update_thumbnail)
        self.thumbnail_signal_emitter.error.connect(self.parent().statusBar().showMessage) # エラーをステータスバーに表示

    def set_data(self, data, total_count):
        self.beginResetModel()
        self._data = data
        self._total_image_count = total_count
        self.thumbnail_cache.clear() # データ更新時にキャッシュをクリア
        self.endResetModel()

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
            if col == 0: # サムネイル列
                return "" # サムネイル自体はDecorationRoleで描画されるため空文字
            elif col == 1: # ファイル名
                return os.path.basename(item_data.get('file_path', ''))
            elif col == 2: # パス
                return item_data.get('file_path', '')
            elif col == 3: # 類似度
                score = item_data.get('score')
                return f"{score:.4f}" if score is not None else ""
            elif col == 4: # タグ列
                tags = item_data.get('tags', '') # タグデータを取得
                return tags if tags else "" # タグがあれば表示、なければ空文字列

        elif role == Qt.ItemDataRole.DecorationRole and col == 0: # サムネイル列
            file_path = item_data.get('file_path')
            if file_path in self.thumbnail_cache:
                return self.thumbnail_cache[file_path]
            else:
                # サムネイルがない場合、バックグラウンドで生成をリクエスト
                if file_path and os.path.exists(file_path):
                    generator = ThumbnailGenerator(
                        image_path=file_path,
                        size=QSize(100, 100), # ★ここはこのままで、行の高さで調整
                        cache_dir=self.thumbnail_cache_dir,
                        index=index,
                        signal_emitter=self.thumbnail_signal_emitter
                    )
                    self.thread_pool.start(generator)
                return QPixmap() # ロード中は空のPixmapを返す

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
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

# --- 3. メインアプリケーションウィンドウ ---
class ImageFeatureViewerApp(QMainWindow):
    # 設定ファイル名
    SETTINGS_FILE = "image_feature_manager.config.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("画像特徴量検索＆管理")
        self.setGeometry(100, 100, 1200, 800) # デフォルトの位置とサイズ

        self.db_path = None
        self.db_manager = None # DBManagerのインスタンス
        self.clip_feature_extractor = None # CLIPFeatureExtractorのインスタンス

        # デフォルトの設定値
        self.top_n_display_count = 1000 
        self.similarity_threshold = 0.25 # ★デフォルト閾値を0.25に変更
        self.recent_db_paths = [] # 最近開いたDBファイルのリスト
        self.window_x = 100 # デフォルトのウィンドウ位置
        self.window_y = 100 # デフォルトのウィンドウ位置

        self._load_settings() # ★設定を最初にロード

        self.move(self.window_x, self.window_y) # ★保存された位置に移動

        self._init_ui()
        self._init_menu() # メニューバーの初期化

        # DBパスの読み込みロジックは_load_settingsに統合されたため不要

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 検索入力とボタン
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("検索キーワードを入力...")
        self.search_button = QPushButton("検索")
        self.search_button.clicked.connect(self._perform_search)
        
        self.open_db_button = QPushButton("DBを開く")
        self.open_db_button.clicked.connect(self._open_db_file_dialog)

        # 新しいボタン：特徴量がないファイルを取得
        self.acquire_features_button = QPushButton("特徴量がないファイルを取得")
        self.acquire_features_button.clicked.connect(self._acquire_missing_features)
        self.acquire_features_button.setEnabled(False) # DBが開かれるまで無効

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.open_db_button)
        search_layout.addWidget(self.acquire_features_button)
        main_layout.addLayout(search_layout)

        # テーブルビュー
        self.model = ImageTableModel(self)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows) # 行全体を選択
        self.table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers) # 編集不可に
        self.table_view.doubleClicked.connect(self._open_file_on_double_click) # ダブルクリックでファイルを開く

        # ヘッダーのリサイズモード設定
        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed) # サムネイル列は固定
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch) # ファイル名は伸縮
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # パスは内容に合わせる
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # 類似度
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch) # タグは伸縮
        self.table_view.setColumnWidth(0, 100) # サムネイル列の幅を調整

        # ★行の高さをサムネイルサイズに合わせて調整
        self.table_view.verticalHeader().setDefaultSectionSize(105) # 100x100サムネイル + 5pxパディング

        main_layout.addWidget(self.table_view)

        # ステータスバー
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了。DBファイルを開いてください。")

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("ファイル")

        open_action = file_menu.addAction("DBを開く...")
        open_action.triggered.connect(self._open_db_file_dialog)

        # Recent Files メニュー
        self.recent_files_menu = menubar.addMenu("最近開いたファイル")
        self.recent_files_menu.aboutToShow.connect(self._populate_recent_files_menu) # メニュー表示時に更新

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
        self.add_tag_action.setEnabled(False) # DBが開かれるまで無効

        self.filter_by_tag_action = tag_menu.addAction("タグでフィルタリング...")
        self.filter_by_tag_action.triggered.connect(self._filter_files_by_tags)
        self.filter_by_tag_action.setEnabled(False) # DBが開かれるまで無効

    def _populate_recent_files_menu(self):
        self.recent_files_menu.clear()
        if not self.recent_db_paths:
            self.recent_files_menu.addAction("最近開いたファイルはありません").setEnabled(False)
            return

        for path in self.recent_db_paths:
            action = self.recent_files_menu.addAction(os.path.basename(path))
            action.setData(path) # パスをアクションに関連付ける
            action.triggered.connect(lambda checked, p=path: self._open_db(p))

    def _load_settings(self):
        """アプリケーション設定をファイルから読み込む"""
        if os.path.exists(self.SETTINGS_FILE):
            try:
                with open(self.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.top_n_display_count = settings.get('top_n_display_count', self.top_n_display_count)
                    self.similarity_threshold = settings.get('similarity_threshold', self.similarity_threshold)
                    self.recent_db_paths = settings.get('recent_db_paths', [])
                    # ウィンドウ位置をロード
                    self.window_x = settings.get('window_x', self.window_x)
                    self.window_y = settings.get('window_y', self.window_y)

                    # 最後に開いたDBパスを直接開く試み
                    last_path = settings.get('last_opened_db_path')
                    if last_path and os.path.exists(last_path):
                        # DBは後で_open_dbが呼ばれる際に開かれるので、ここではパスだけ設定
                        self.db_path = last_path 

            except (IOError, json.JSONDecodeError) as e:
                print(f"設定ファイルの読み込み中にエラーが発生しました: {e}", file=sys.stderr)
                # エラー時はデフォルト値を使用

    def _save_settings(self):
        """アプリケーション設定をファイルに保存する"""
        # 現在のウィンドウ位置を取得
        current_pos = self.pos()
        self.window_x = current_pos.x()
        self.window_y = current_pos.y()

        settings = {
            'top_n_display_count': self.top_n_display_count,
            'similarity_threshold': self.similarity_threshold,
            'recent_db_paths': self.recent_db_paths,
            'last_opened_db_path': self.db_path, # 現在開いているDBを最後に開いたDBとして保存
            'window_x': self.window_x,
            'window_y': self.window_y,
        }
        try:
            with open(self.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"設定ファイルの保存中にエラーが発生しました: {e}", file=sys.stderr)

    def _add_recent_db_path(self, path):
        """最近開いたDBファイルのリストにパスを追加する"""
        if path in self.recent_db_paths:
            self.recent_db_paths.remove(path) # 既存なら一旦削除して最新にする
        self.recent_db_paths.insert(0, path) # 先頭に追加
        # 最大数で制限
        max_recent = 10
        if len(self.recent_db_paths) > max_recent:
            self.recent_db_paths = self.recent_db_paths[:max_recent]

    def _open_db_file_dialog(self):
        options = QFileDialog.Option.DontUseNativeDialog
        db_path, _ = QFileDialog.getOpenFileName(self, "DBファイルを開く", "", "SQLite Databases (*.db);;All Files (*)", options=options)
        if db_path:
            self._open_db(db_path)

    def _open_db(self, db_path):
        if not os.path.exists(db_path):
            QMessageBox.warning(self, "エラー", f"指定されたDBファイルが見つかりません:\n{db_path}")
            return

        try:
            # 既存のDBManagerがあれば閉じる
            if self.db_manager:
                self.db_manager.close()
            
            self.db_path = db_path
            self.db_manager = DBManager(self.db_path)
            # CLIPFeatureExtractorはDBManagerとは独立してインスタンス化
            if self.clip_feature_extractor is None:
                self.clip_feature_extractor = CLIPFeatureExtractor()

            self.status_bar.showMessage(f"データベース '{os.path.basename(self.db_path)}' を開きました。")
            self._add_recent_db_path(self.db_path) # 最近開いたリストに追加
            
            # DBが開かれたら関連機能を有効化
            self.search_button.setEnabled(True)
            self.acquire_features_button.setEnabled(True)
            self.add_tag_action.setEnabled(True)
            self.filter_by_tag_action.setEnabled(True)

            # DBオープン時に全画像数を取得して表示
            total_count = self.db_manager.get_total_image_count()
            self.status_bar.showMessage(f"データベース '{os.path.basename(self.db_path)}' を開きました。全画像数: {total_count}")
            self.model.set_data([], total_count) # 初期表示は空で、総数のみセット
            
        except sqlite3.Error as e:
            QMessageBox.critical(self, "DB接続エラー", f"データベースへの接続に失敗しました:\n{e}")
            self.db_path = None
            self.db_manager = None
            self.status_bar.showMessage("DB未接続。")
            # DBが開かれていない場合は関連機能を無効化
            self.search_button.setEnabled(False)
            self.acquire_features_button.setEnabled(False)
            self.add_tag_action.setEnabled(False)
            self.filter_by_tag_action.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "初期化エラー", f"アプリケーションの初期化中にエラーが発生しました:\n{e}")

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
        """データベースにCLIP特徴量がないファイルを取得・追加する"""
        if not self.db_manager or not self.clip_feature_extractor:
            self.status_bar.showMessage("エラー: DBが選択されていないか、CLIPモデルが初期化されていません。")
            return
        
        self.status_bar.showMessage("特徴量がないファイルを検索し、取得中です...")
        QApplication.processEvents()
        
        try:
            files_without_features = self.db_manager.get_file_paths_without_clip_features()
            
            if not files_without_features:
                self.status_bar.showMessage("特徴量がないファイルは見つかりませんでした。")
                return

            self.status_bar.showMessage(f"{len(files_without_features)} 個のファイルの特徴量を抽出中...")
            QApplication.processEvents()

            extracted_features, processed_indices = self.clip_feature_extractor.extract_features_from_paths(files_without_features)
            
            if len(extracted_features) == 0:
                self.status_bar.showMessage("特徴量の抽出に成功したファイルはありませんでした。")
                return

            updated_count = 0
            for i, original_index in enumerate(processed_indices):
                file_path = files_without_features[original_index]
                feature_blob = numpy_to_blob(extracted_features[i])
                
                self.db_manager.update_file_metadata(file_path, {'clip_feature_blob': feature_blob})
                updated_count += 1
            
            self.status_bar.showMessage(f"{updated_count} 個のファイルにCLIP特徴量を追加しました。")
            QMessageBox.information(self, "完了", f"{updated_count} 個のファイルにCLIP特徴量を追加しました。")

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
                self._perform_search() # 現在の検索条件で再検索して表示を更新
            else:
                QMessageBox.warning(self, "警告", "タグの追加に失敗したファイルがあります。")

    def _filter_files_by_tags(self):
        QMessageBox.information(self, "機能開発中", "この機能はまだ実装されていません。")

    def _open_file_on_double_click(self, index: QModelIndex):
        """テーブルビューの行をダブルクリックしたときにファイルを開く"""
        if index.isValid():
            row_data = self.model.get_row_data(index.row())
            if row_data and 'file_path' in row_data:
                file_path = row_data['file_path']
                if os.path.exists(file_path):
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                else:
                    QMessageBox.warning(self, "ファイルが見つかりません", f"ファイルが見つかりません:\n{file_path}")

    def closeEvent(self, event):
        """アプリケーション終了時にDB接続を閉じ、設定を保存する"""
        self._save_settings() # ★設定を保存
        if self.db_manager:
            self.db_manager.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageFeatureViewerApp()
    window.show()
    sys.exit(app.exec())