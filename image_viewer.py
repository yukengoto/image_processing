import sys
import os
import sqlite3
import numpy as np
import hashlib
from PIL import Image # サムネイル生成用 (pip install Pillow)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableView, QLineEdit, QHeaderView, QLabel,
    QStatusBar, QAbstractItemView
)
from PySide6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, QSize,
    QThreadPool, QRunnable, Signal, QObject, QUrl
)
from PySide6.QtGui import QPixmap, QImage, QDesktopServices # QDesktopServices for opening files
from db_classifier import blob_to_numpy  # 画像特徴量の変換関数をインポート
import torch
from classify_image import ImageSorter # 画像分類用のモジュールをインポート


# --- 1. サムネイル生成をバックグラウンドで行うためのQRunnableとシグナルエミッター ---
# （前回のやり取りで提示したものに、エラーハンドリングなどを追加）
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
        self.setAutoDelete(True) # タスク完了後に自動削除

    def run(self):
        # キャッシュファイルのパスを生成
        cache_key = hashlib.md5(self.image_path.encode('utf-8')).hexdigest()
        cache_file_name = f"{cache_key}_{self.size.width()}x{self.size.height()}.png"
        cache_path = os.path.join(self.cache_dir, cache_file_name)

        pixmap = QPixmap()
        try:
            # キャッシュから読み込みを試みる
            if os.path.exists(cache_path):
                pixmap.load(cache_path)
                if not pixmap.isNull():
                    self.signal_emitter.thumbnail_ready.emit(self.index, pixmap)
                    return

            # キャッシュにない場合、または読み込み失敗の場合、画像を生成
            with Image.open(self.image_path) as img:
                img.thumbnail((self.size.width(), self.size.height()), Image.LANCZOS)
                
                # PIL ImageをQImageに変換
                # Pillowのmode 'RGBA' は QImage.Format_RGBA8888 に対応
                qimage = QImage(img.tobytes(), img.width, img.height, img.width * (4 if img.mode == 'RGBA' else 3), 
                                QImage.Format_RGBA8888 if img.mode == 'RGBA' else QImage.Format_RGB888)
                
                pixmap = QPixmap.fromImage(qimage)

                if not pixmap.isNull():
                    # サムネイルをキャッシュに保存
                    pixmap.save(cache_path, "PNG")
                    self.signal_emitter.thumbnail_ready.emit(self.index, pixmap)
                else:
                    self.signal_emitter.error.emit(f"サムネイル生成失敗 (QPixmap): {self.image_path}")
        except FileNotFoundError:
            self.signal_emitter.error.emit(f"ファイルが見つかりません: {self.image_path}")
        except Exception as e:
            self.signal_emitter.error.emit(f"サムネイル生成エラー ({self.image_path}): {e}")


# --- 2. データを保持し、ビューに提供するカスタムモデル ---
class ImageTableModel(QAbstractTableModel):
    def __init__(self, cache_dir, parent=None):
        super().__init__(parent)
        self._data = [] # 表示するデータ（上位N件）
        self._headers = ["サムネイル", "ファイル名", "スコア", "タグ", "ファイルパス"]
        self._total_image_count = 0 # DB内の全画像数
        self._thumbnails = {} # キャッシュされたサムネイル {index.row(): QPixmap}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(os.cpu_count() or 4) # スレッド数をCPUコア数に設定
        self.thumbnail_signal_emitter = ThumbnailSignalEmitter()
        self.thumbnail_signal_emitter.thumbnail_ready.connect(self._update_thumbnail)
        self.thumbnail_signal_emitter.error.connect(self._handle_thumbnail_error)
        self.thumbnail_size = QSize(100, 100) # サムネイルの表示サイズ

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        column = index.column()
        row_data = self._data[index.row()]

        if role == Qt.DisplayRole:
            if column == 1: # ファイル名
                return os.path.basename(row_data['file_path'])
            elif column == 2: # スコア
                return f"{row_data.get('score', 0.0):.4f}" # スコアを小数点以下4桁で表示
            elif column == 3: # タグ
                return ", ".join(row_data.get('tags', []))
            elif column == 4: # ファイルパス
                return row_data['file_path']
            return None # その他の列はDisplayRoleでは表示しない

        elif role == Qt.DecorationRole and column == 0: # サムネイル列
            if index.row() in self._thumbnails:
                return self._thumbnails[index.row()]
            else:
                # サムネイルがまだない場合、ロードタスクをキューに入れる
                file_path = row_data['file_path']
                if os.path.exists(file_path):
                    generator = ThumbnailGenerator(file_path, self.thumbnail_size, self.cache_dir, index, self.thumbnail_signal_emitter)
                    self.thread_pool.start(generator)
                
                # ロード中のプレースホルダー（空白のPixmap）を返す
                return QPixmap(self.thumbnail_size)

        elif role == Qt.TextAlignmentRole:
            if column in [1, 2]: # ファイル名、スコアは中央揃えなど
                 return int(Qt.AlignVCenter | Qt.AlignLeft) # 左寄せ
            return int(Qt.AlignVCenter | Qt.AlignLeft)

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None

    def set_data(self, data: list, total_count: int):
        """モデルのデータを更新する"""
        self.beginResetModel()
        self._data = data
        self._total_image_count = total_count
        self._thumbnails.clear() # データ更新時にキャッシュをクリア
        self.endResetModel()

    def _update_thumbnail(self, index, pixmap):
        """バックグラウンドでロードされたサムネイルをモデルに反映"""
        if index.isValid():
            self._thumbnails[index.row()] = pixmap
            # サムネイルが更新されたことをビューに通知
            self.dataChanged.emit(index, index, [Qt.DecorationRole])

    def _handle_thumbnail_error(self, message):
        """サムネイルロードエラーを処理（デバッグ用）"""
        print(f"サムネイルエラー: {message}")
        # ここでステータスバーなどにエラーを表示することも可能

    def get_row_data(self, row):
        """指定された行の全データを取得"""
        if 0 <= row < len(self._data):
            return self._data[row]
        return None

    def get_total_image_count(self):
        """データベース内の全画像数を返す"""
        return self._total_image_count


# --- 3. メインウィンドウの定義 ---
class ImageManagerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("画像マネージャー (PySide6)")
        self.setGeometry(100, 100, 1200, 800)

        self.conn = None # データベース接続
        self.sorter = None # CLIPモデルをロードするImageSorterオブジェクト
        self.db_path = None
        self.thumbnail_cache_dir = "thumbnail_cache"
        # デフォルトのトップN表示数
        self.top_n_display_count = 500

        self._create_ui()
        self._connect_signals()

        # CLIPモデルのロード（アプリケーション起動時に一度だけ実行）
        #self.status_bar = QStatusBar()
        #self.setStatusBar(self.status_bar) # ステータスバーをセットアップ
        self.status_bar.showMessage("CLIPモデルをロード中...", 0) # ロード開始メッセージ

        try:
            # feature_mode='full' で初期化するとCLIPモデルがロードされます
            self.sorter = ImageSorter(feature_mode='full') 
            self.status_bar.showMessage("CLIPモデルロード済み。", 5000) # 成功メッセージ
            # print("CLIPモデルが正常にロードされました。") # コンソール出力も追加可能
        except Exception as e:
            self.sorter = None # ロード失敗時はsorterをNoneに設定
            self.status_bar.showMessage(f"CLIPモデルのロードに失敗しました: {e}", 5000) # 失敗メッセージ
            # QMessageBox.warning(self, "エラー", f"CLIPモデルのロードに失敗しました。\n検索機能が利用できません。\n詳細: {e}")
            print(f"CLIPモデルのロードに失敗しました: {e}") # コンソール出力も追加可能



    def _create_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # ファイル選択とパス表示
        file_select_layout = QHBoxLayout()
        self.db_path_edit = QLineEdit("SQLiteファイルを選択してください...")
        self.db_path_edit.setReadOnly(True)
        self.select_db_button = QPushButton("DBファイルを開く")
        file_select_layout.addWidget(self.db_path_edit)
        file_select_layout.addWidget(self.select_db_button)
        self.main_layout.addLayout(file_select_layout)

        # 検索バーと閾値、上位N設定
        search_layout = QHBoxLayout()
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("キーワードを入力...")
        self.search_button = QPushButton("検索")
        
        self.threshold_input = QLineEdit("0.5") # デフォルト閾値
        self.threshold_input.setPlaceholderText("閾値 (0.0-1.0)")
        self.threshold_input.setFixedWidth(100) # 幅を固定

        self.top_n_input = QLineEdit(str(self.top_n_display_count)) # デフォルト表示件数
        self.top_n_input.setPlaceholderText("上位N件")
        self.top_n_input.setFixedWidth(100)

        search_layout.addWidget(self.keyword_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(QLabel("閾値:"))
        search_layout.addWidget(self.threshold_input)
        search_layout.addWidget(QLabel("上位N:"))
        search_layout.addWidget(self.top_n_input)
        self.main_layout.addLayout(search_layout)

        # リストビュー（QTableViewを使用）
        self.table_view = QTableView()
        self.model = ImageTableModel(self.thumbnail_cache_dir)
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows) # 行全体を選択
        self.table_view.setSelectionMode(QAbstractItemView.ExtendedSelection) # 複数行選択可能に

        # サムネイルの高さに合わせて行の高さを設定
        self.table_view.verticalHeader().setDefaultSectionSize(self.model.thumbnail_size.height())
        #self.table_view.verticalHeader().hide() # 必要であれば行番号を非表示にする

        # ヘッダー設定
        self.table_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed) # サムネイル列は固定幅
        self.table_view.setColumnWidth(0, self.model.thumbnail_size.width() + 10) # サムネイル列の幅
        self.table_view.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch) # ファイル名はストレッチ
        self.table_view.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch) # ファイルパスもストレッチ
        
        # ソート可能にする
        self.table_view.setSortingEnabled(True)

        self.main_layout.addWidget(self.table_view)

        # ステータスバー
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("データベースファイルを選択してください。")

        # 全画像数表示ラベル
        self.total_count_label = QLabel("全画像数: N/A")
        self.status_bar.addPermanentWidget(self.total_count_label)

    def _connect_signals(self):
        self.select_db_button.clicked.connect(self.open_sqlite_file)
        self.search_button.clicked.connect(self.perform_keyword_search)
        # テーブルビューのダブルクリックでファイルを開く
        self.table_view.doubleClicked.connect(self._open_file_on_double_click)
        self.top_n_input.textChanged.connect(self._update_top_n_limit)
        self.threshold_input.textChanged.connect(self._update_threshold)

    def _update_top_n_limit(self, text):
        try:
            self.top_n_display_count = int(text)
            if self.top_n_display_count <= 0:
                self.top_n_display_count = 100 # 最小値
            # ここで自動的に検索を再実行することも可能 (例: self.perform_keyword_search())
        except ValueError:
            self.top_n_display_count = 500 # 無効な入力の場合はデフォルトに戻す

    def _update_threshold(self, text):
        try:
            self.current_threshold = float(text)
            if not (0.0 <= self.current_threshold <= 1.0):
                self.current_threshold = 0.5 # 無効な場合はデフォルト
            # ここで自動的に検索を再実行することも可能
        except ValueError:
            self.current_threshold = 0.5 # 無効な入力の場合はデフォルトに戻す


    def open_sqlite_file(self):
        """ファイルダイアログを開き、SQLiteファイルを指定する"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "SQLiteデータベースファイルを開く", "", "SQLite Files (*.db *.sqlite *.sqlite3)"
        )
        if file_path:
            self.db_path = file_path
            self.db_path_edit.setText(self.db_path)
            self.status_bar.showMessage(f"データベース: {os.path.basename(self.db_path)} を読み込み中...")
            self.load_initial_data()

    def load_initial_data(self):
        """SQLiteファイルを読み込み、初期データをモデルにロードする"""
        if not self.db_path:
            self.status_bar.showMessage("SQLiteデータベースファイルが選択されていません。")
            return

        self.status_bar.showMessage("データを読み込み中...")
        QApplication.processEvents() # UI更新を強制

        # --- 非同期処理の概念 ---
        # データベースから全画像数を取得するタスクをバックグラウンドで実行
        # (ここでは簡略化のため同期的に記述)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 全画像数を取得
            cursor.execute("SELECT COUNT(*) FROM file_metadata")
            total_count = cursor.fetchone()[0]
            self.total_count_label.setText(f"全画像数: {total_count}")
            self.model.set_data([], total_count) # 初期状態では空で、全数だけ設定

            # 初期表示のためのデータ取得（例: 最初のN件、またはスコア0で全件取得しPythonでソート）
            # ここではシンプルにファイルパスだけを読み込む（スコアやタグは後で計算/取得）
            cursor.execute("""
                SELECT fm.file_path, '' AS score, '' AS tags
                FROM file_metadata fm
                LIMIT ?
            """, (self.top_n_display_count,)) # 初期ロードも上位N件
            
            initial_data = []
            for row in cursor.fetchall():
                initial_data.append({
                    'file_path': row[0],
                    'score': 0.0, # 初期スコアは0
                    'tags': [] # 初期タグは空
                })
            
            conn.close()

            # モデルを更新
            self.model.set_data(initial_data, total_count)
            self.status_bar.showMessage(f"データベース読み込み完了。上位 {len(initial_data)} 件を表示中。")

        except sqlite3.Error as e:
            self.status_bar.showMessage(f"データベースエラー: {e}")
            self.model.set_data([], 0) # エラー時はデータをクリア

    def perform_keyword_search(self):
        """キーワード検索を実行し、スコアに基づいて上位N件を表示する"""
        keyword = self.keyword_input.text().strip()
        if not keyword or not self.db_path:
            self.status_bar.showMessage("キーワードを入力し、DBファイルを選択してください。")
            return

        self.status_bar.showMessage(f"'{keyword}'で検索中...")
        QApplication.processEvents() # UI更新を強制

        # --- ここからがキーワード検索と特徴量計算の複雑な部分 ---
        # この部分はバックグラウンドスレッドで実行されるべき
        # 以下のコードは概念的なものです。実際にはImageSorterなどを利用します。
        
        # 擬似的な検索結果
        # データベースから全画像パスと特徴量を読み込む（この処理も非同期で！）
        # 例: features_data = self._load_features_from_db_async()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 全ファイルパスと特徴量を読み込む（ここが大規模データでボトルネックになりやすい）
            # 実際には、このデータ量に応じて最適化が必要
            cursor.execute("""
                SELECT fm.file_path, cf.clip_feature_blob, COALESCE(GROUP_CONCAT(ft.tag, ', '), '')
                FROM file_metadata fm
                JOIN clip_features cf ON fm.file_path = cf.file_path
                LEFT JOIN file_tags ft ON fm.file_path = ft.file_path
                GROUP BY fm.file_path, cf.clip_feature_blob
            """)

            all_image_data = cursor.fetchall()
            conn.close()

            if not all_image_data:
                self.status_bar.showMessage("データベースに画像特徴量が見つかりません。")
                self.model.set_data([], self.model.get_total_image_count())
                return

            # --- 概念的な特徴量計算とスコアリング ---
            # 実際には、ここにCLIPモデルを使ったキーワード埋め込みとコサイン類似度計算が入る
            # from your_clip_module import get_clip_embedding, calculate_similarity
            # keyword_embedding = get_clip_embedding(keyword)
            # キーワードのCLIP埋め込みを取得
            keyword_token = self.sorter.clip_tokenizer([keyword]).to(self.sorter.device)
            with torch.no_grad():
                keyword_embedding = self.sorter.model.encode_text(keyword_token).cpu().numpy()[0]
            keyword_embedding_norm = keyword_embedding / np.linalg.norm(keyword_embedding)
            # ... (この後に `keyword_embedding_norm` を使用した計算が続きます)            


            results = []
            for path, feature_vector_blob, tags_str in all_image_data:
                score = 0.0
                if feature_vector_blob:
                    # feature_vector_blob を numpy array に変換
                    #feature_vector = np.frombuffer(feature_vector_blob, dtype=np.float32)
                    
                    image_feature = blob_to_numpy(feature_vector_blob)
                    if image_feature is not None and image_feature.size > 0:
                        image_feature_norm = image_feature / np.linalg.norm(image_feature)
                        # ⭐ ここでキーワード特徴量と画像特徴量で類似度を計算しています ⭐
                        score = np.dot(keyword_embedding_norm, image_feature_norm)

                
                tags = tags_str.split(',') if tags_str else []
                results.append({
                    'file_path': path,
                    'score': score,
                    'tags': tags
                })

            # スコアと閾値でフィルタリング
            threshold = 0.0
            try:
                threshold = float(self.threshold_input.text())
            except ValueError:
                pass # 無効な入力は無視
            
            filtered_results = [r for r in results if r['score'] >= threshold]

            # スコアで降順にソート
            sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)

            # 上位N件に絞り込み
            display_data = sorted_results[:self.top_n_display_count]
            
            # モデルを更新
            self.model.set_data(display_data, self.model.get_total_image_count())
            self.status_bar.showMessage(f"検索完了。上位 {len(display_data)} 件を表示中。")

        except sqlite3.Error as e:
            self.status_bar.showMessage(f"検索中のデータベースエラー: {e}")
        except Exception as e:
            self.status_bar.showMessage(f"検索中にエラーが発生しました: {e}")


    def _open_file_on_double_click(self, index: QModelIndex):
        """テーブルビューの行をダブルクリックしたときにファイルを開く"""
        if index.isValid():
            row_data = self.model.get_row_data(index.row())
            if row_data and 'file_path' in row_data:
                file_path = row_data['file_path']
                if os.path.exists(file_path):
                    # OSのデフォルトビューアでファイルを開く
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
                    self.status_bar.showMessage(f"ファイルを開きました: {file_path}")
                else:
                    self.status_bar.showMessage(f"ファイルが見つかりません: {file_path}")

# --- アプリケーションの実行 ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageManagerApp()
    window.show()
    sys.exit(app.exec())
