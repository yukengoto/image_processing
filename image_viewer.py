import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeView, QListView, QPushButton, QLineEdit, QSplitter,
    QFileSystemModel # QFileDialog, 
)
from PySide6.QtCore import Qt, QDir, QFileInfo
import os
from PySide6.QtGui import QIcon # QAction, QIconはQtGuiのままでOKです


class FileBrowserWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PySide File Browser")
        self.setGeometry(100, 100, 1000, 700) # ウィンドウサイズを設定

        # --- モデルの作成 ---
        # QFileSystemModel は、ファイルシステムをツリー構造で表現するモデル
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.currentPath()) # 初期表示パスを設定 (例: 現在の作業ディレクトリ)

        # --- ビューの作成 ---
        # 1. ツリービュー (フォルダ階層)
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.model)
        self.tree_view.setRootIndex(self.model.index(QDir.currentPath())) # モデルのルートインデックスを設定
        
        # フォルダのみを表示するようにフィルタリング (オプション)
        # self.tree_view.setFilter(QDir.Dirs | QDir.NoDotAndDotDot) 
        # self.model.setFilter(QDir.Dirs | QDir.NoDotAndDotDot) # ツリービューのモデルにフィルタを設定
        self.tree_view.setHeaderHidden(True) # ヘッダーを非表示にする (フォルダツリーには不要な場合が多い)
        
        # 不要な列を隠す (例: Size, Type, Date Modified)
        # QFileSystemModelの列のインデックスは決まっている (0:名前, 1:サイズ, 2:タイプ, 3:日付)
        for i in range(1, self.model.columnCount()):
            self.tree_view.hideColumn(i)

        # 2. リストビュー (選択されたフォルダの中身)
        self.list_view = QListView()
        self.list_view.setModel(self.model) # 同じモデルを共有
        
        # ファイルとフォルダの両方を表示
        #self.list_view.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)

        # --- パス表示と移動用のウィジェット ---
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True) # 基本的に読み取り専用だが、将来的に編集可能にしても良い
        
        self.go_button = QPushButton("Go")
        self.go_button.setIcon(QIcon.fromTheme("go-jump")) # Goアイコンがあれば表示
        
        self.up_button = QPushButton("Up")
        self.up_button.setIcon(QIcon.fromTheme("go-up")) # Upアイコンがあれば表示

        # --- レイアウト ---
        # パス操作部分のレイアウト
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.go_button)
        path_layout.addWidget(self.up_button)

        # スプリッターでツリービューとリストビューを左右に配置 (サイズ変更可能)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree_view)
        splitter.addWidget(self.list_view)
        splitter.setStretchFactor(0, 1) # 左側を1の比率
        splitter.setStretchFactor(1, 3) # 右側を3の比率 (リストビューを広めに)

        # メインレイアウト
        main_layout = QVBoxLayout()
        main_layout.addLayout(path_layout)
        main_layout.addWidget(splitter)

        # 中央ウィジェットにレイアウトを設定
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # --- シグナルとスロットの接続 ---
        # ツリービューでフォルダが選択されたら、リストビューの内容を更新
        self.tree_view.selectionModel().currentChanged.connect(self.update_list_view)
        
        # リストビューでアイテムがダブルクリックされたら、そのフォルダを開くかファイルを開く
        self.list_view.doubleClicked.connect(self.handle_list_item_double_clicked)
        
        # Goボタンが押されたら、現在のパスに移動 (将来的にpath_editが編集可能になった場合)
        # self.go_button.clicked.connect(self.go_to_path)
        
        # Upボタンが押されたら、親ディレクトリに移動
        self.up_button.clicked.connect(self.go_up_directory)

        # 初期表示のパスをリストビューにも反映
        self.update_list_view(self.model.index(QDir.currentPath()))


    def update_list_view(self, index):
        """ツリービューで選択されたフォルダに基づいてリストビューを更新する"""
        if index.isValid():
            # リストビューのルートインデックスを、ツリービューで選択されたインデックスに設定
            self.list_view.setRootIndex(index)
            # パス表示を更新
            self.path_edit.setText(self.model.filePath(index))

    def handle_list_item_double_clicked(self, index):
        """リストビューでアイテムがダブルクリックされた際の処理"""
        if index.isValid():
            file_path = self.model.filePath(index)
            if self.model.isDir(index):
                # フォルダの場合、ツリービューとリストビューの両方をそのフォルダに更新
                self.tree_view.setCurrentIndex(index) # ツリービューの選択も更新
                self.list_view.setRootIndex(index)
                self.path_edit.setText(file_path)
            else:
                # ファイルの場合、ここで画像処理などのロジックを呼び出す
                print(f"ファイルがダブルクリックされました: {file_path}")
                # 例: self.process_image_file(file_path)

    def go_up_directory(self):
        """
        親ディレクトリに移動します。
        もし現在の場所がドライブのルートである場合、利用可能なドライブの一覧を表示します。
        """
        current_index = self.list_view.rootIndex()
        current_path = self.model.filePath(current_index)

        current_file_info = QFileInfo(current_path)
        # 親ディレクトリのパス文字列を取得
        parent_path = current_file_info.dir().absolutePath()

        # 親パスが現在のパスと異なる場合 (通常のディレクトリ移動)
        if parent_path != current_path:
            parent_index = self.model.index(parent_path)
            if parent_index.isValid():
                # tree_view と list_view の両方を更新
                self.tree_view.setCurrentIndex(parent_index)
                self.list_view.setRootIndex(parent_index)
                # パス表示も更新
                self.path_edit.setText(self.model.filePath(parent_index))
            else:
                print(f"警告: 親パス '{parent_path}' の有効なインデックスを取得できませんでした。")
        else:
            # 親パスが現在のパスと同じ場合 (つまり、すでにルートディレクトリにいる場合)
            if current_file_info.isRoot(): # 現在のパスが実際にルートディレクトリであるかを確認
                # Windowsの場合、QFileSystemModel のルートを空文字列に設定すると、
                # 自動的にすべての論理ドライブが列挙されます。
                self.model.setRootPath("") # ⭐ ここでドライブ列挙をトリガー ⭐
                
                # UIの表示を「コンピューター」などのドライブ一覧モードに切り替える
                root_index_for_drives = self.model.index("") # 空のパスはモデルのルート (ドライブ一覧) を指す
                self.tree_view.setCurrentIndex(root_index_for_drives)
                self.list_view.setRootIndex(root_index_for_drives)
                
                # パス表示を「コンピューター」や「マイコンピューター」など、
                # ドライブ一覧を表すテキストに変更
                self.path_edit.setText("コンピューター") # または "PC"、"ドライブ一覧" など
                print("ドライブ一覧を表示します。")
            else:
                # Windowsの非ルートパスで親パスが同じになる特殊なケース（通常は発生しにくい）
                # または、Linux/macOSでルートディレクトリ '/' にいる場合など
                print(f"これ以上上位に移動できません: {current_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileBrowserWindow()
    window.show()
    sys.exit(app.exec())