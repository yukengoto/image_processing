# image_processing_utils.py
# 画像処理関連の共通ユーティリティ関数とデータ

import sys
import os
from typing import List, Tuple, Optional

# --- 画像拡張子の定義 ---
# 様々な用途に対応した拡張子セット
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.svg'}
BASIC_IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')  # 基本的な画像形式
CLIP_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}  # CLIP処理対応
THUMBNAIL_SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico'}

# --- ファイル操作関数 ---
def get_image_paths(img_dirs: List[str], recursive: bool = False, max_count: Optional[int] = None) -> List[str]:
    """
    指定されたディレクトリから画像ファイルのパスを取得します。
    
    Args:
        img_dirs: 検索対象のディレクトリリスト
        recursive: サブディレクトリも再帰的に検索するか
        max_count: 取得する最大ファイル数
    
    Returns:
        画像ファイルパスのリスト
    """
    image_paths = []
    for img_dir in img_dirs:
        if not os.path.isdir(img_dir):
            print(f"警告: ディレクトリ '{img_dir}' が見つかりません。スキップします。", file=sys.stderr)
            continue
            
        if recursive:
            for root, _, files in os.walk(img_dir):
                for file in files:
                    if file.lower().endswith(BASIC_IMAGE_EXTENSIONS):
                        image_paths.append(os.path.join(root, file))
                        if max_count and len(image_paths) >= max_count:
                            return image_paths[:max_count]
        else:
            for file in os.listdir(img_dir):
                if file.lower().endswith(BASIC_IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(img_dir, file))
                    if max_count and len(image_paths) >= max_count:
                        return image_paths[:max_count]
    
    return image_paths

def remove_empty_dirs(root_dir: str) -> None:
    """
    指定されたディレクトリ以下の空のディレクトリを削除します。
    
    Args:
        root_dir: 削除対象の親ディレクトリ
    """
    for current_dir, dirs, files in os.walk(root_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(current_dir)
                print(f"空フォルダを削除: {current_dir}")
            except OSError:
                continue

def is_image_file(file_path: str) -> bool:
    """
    ファイルが画像ファイルかどうかを判定します。
    
    Args:
        file_path: ファイルパス
        
    Returns:
        画像ファイルの場合True
    """
    ext = os.path.splitext(file_path.lower())[1]
    return ext in SUPPORTED_IMAGE_EXTENSIONS

def is_clip_supported(file_path: str) -> bool:
    """
    ファイルがCLIP処理対応かどうかを判定します。
    
    Args:
        file_path: ファイルパス
        
    Returns:
        CLIP処理対応の場合True
    """
    ext = os.path.splitext(file_path.lower())[1]
    return ext in CLIP_SUPPORTED_EXTENSIONS

# --- 進捗表示関数 ---
class ProgressDisplay:
    """進捗表示のためのユーティリティクラス"""
    
    @staticmethod
    def show_progress(current: int, total: int, task_name: str = "", percentage: bool = False) -> None:
        """
        進捗を表示します。
        
        Args:
            current: 現在の進捗値
            total: 全体の値
            task_name: タスク名（表示用）
            percentage: パーセンテージも表示するか
        """
        if percentage and total > 0:
            percent = (current / total) * 100
            if task_name:
                sys.stdout.write(f"\r進捗 ({task_name}): {current}/{total} ({percent:.2f}%)")
            else:
                sys.stdout.write(f"\r進捗: {current}/{total} ({percent:.2f}%)")
        else:
            if task_name:
                sys.stdout.write(f"\r進捗 ({task_name}): {current}/{total}")
            else:
                sys.stdout.write(f"\r進捗: {current}/{total}")
        sys.stdout.flush()
    
    @staticmethod
    def finish_progress(task_name: str = "") -> None:
        """
        進捗表示を終了します。
        
        Args:
            task_name: 完了したタスク名
        """
        if task_name:
            print(f"\n{task_name}完了。")
        else:
            print("\n完了。")

# --- ファイルパス安全化関数 ---
def make_safe_filename(filename: str) -> str:
    """
    ファイルシステムで安全なファイル名に変換します。
    
    Args:
        filename: 元のファイル名
        
    Returns:
        安全なファイル名
    """
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ']
    safe_filename = filename
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')
    return safe_filename

# --- 定数定義クラス ---
class Constants:
    """アプリケーション全体で使用する定数"""
    
    # K-Meansパラメータ
    MIN_KMEANS_CLUSTERS = 8
    MAX_KMEANS_CLUSTERS = 30
    DEFAULT_KMEANS_INIT = 'auto'
    
    # 閾値
    KEYWORD_CLASSIFICATION_THRESHOLD = 0.25
    
    # ファイルサイズ閾値（バイト）
    THRESHOLD_AREA_SMALL = 250 * 250
    THRESHOLD_AREA_MEDIUM = 1100 * 900
    THRESHOLD_ASPECT_PORTRAIT = 0.95
    THRESHOLD_ASPECT_SQUARE = 1.05
    
    # デフォルト値
    DEFAULT_MAX_IMAGES = 100000
    DEFAULT_BATCH_SIZE = 50
    
    # ファイル名
    FEATURE_LOG_FILENAME = "image_features_log.npy"