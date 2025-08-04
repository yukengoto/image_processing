# clip_feature_extractor.py

import numpy as np
import sys
from PIL import Image
from external_keywords import EXTERNAL_KEYWORDS  # 外部キーワード定義をインポート
#import os

# --- 1. グローバル設定とキーワード定義 ---
# 将来的に外部ファイルから読み込むことを想定。
# 現状はここで直接定義し、修正の容易性を確保。
# キー: 最終的なフォルダ名 (カテゴリ名)
# 値: そのカテゴリを表すキーワードのリスト。最初のキーワードが代表として使われる。
# キーワードの定義
KEYWORDS_FOR_CLASSIFICATION = {
    # 自然・風景
    "Landscape": ["an outdoor landscape", "a natural scene", "a mountain view", "a forest", "a beach", "a sunset", "a river", "a field", "a sky"],
    "Nature_Elements": ["a flower", "a tree", "a plant", "clouds", "water", "a rock", "snow", "ice"],
    # 人物関連
    "Person_Portrait": ["a close up of a person", "a portrait", "a human face", "headshot of a person", "selfie"],
    "Person_FullBody": ["a person standing", "a person walking", "a person sitting", "a person from behind", "multiple people", "a crowd"],
    "Person_Action": ["a person running", "a person jumping", "a person dancing", "a person playing sports", "people interacting"],
    # 都市・建築
    "Building": ["a building", "a city view", "an urban landscape", "architecture", "a house", "a skyscraper", "a bridge", "a street", "an interior of a building"],
    "Vehicle": ["a car", "a truck", "a motorcycle", "an airplane", "a train", "a boat", "a bicycle"], # 乗り物を独立カテゴリに
    # 動物
    "Animal": ["an animal", "a pet", "wildlife", "a dog", "a cat", "a bird", "a fish", "an insect"],
    # アート・デザイン
    "Art_Illustration": ["a digital art illustration", "a cartoon character", "an abstract image", "a painting", "a drawing", "a sculpture", "a graphic design", "a sketch"],
    "Text_Document": ["a text document", "a screenshot", "a document with text", "a book page", "a sign with text", "a newspaper"],
    # 物体・その他
    "Food_Drink": ["a food item", "a dish", "fruit", "vegetables", "a drink", "a meal", "dessert"], # 食べ物を独立カテゴリに
    "Object_Household": ["furniture", "a chair", "a table", "a bed", "kitchenware", "home decor"], # 家庭用品を追加
    "Object_Misc": ["a scene with many objects", "a tool", "electronics", "a bag", "clothes", "a shoe", "a socks", "a hand", "money"],
    # その他視覚的特徴
    "Abstract_Pattern": ["an abstract pattern", "a texture", "a blurry image", "a background", "a design"], # 抽象的なものやパターン
    "Black_White": ["a black and white photo", "monochrome image"],
    "Vibrant_Color": ["a vibrant colorful image", "a colorful abstract"],
    "Blurred_Image": ["a blurred image", "out of focus background"],
    "Watermark_Overlay": ["an image with a watermark", "text overlay on image"], # ウォーターマークなども識別
    "Other_Visual": ["a strange image", "a distorted image", "a placeholder image", "an empty image"] # どうしても分類できないもの
}
KEYWORDS_FOR_CLASSIFICATION.update(EXTERNAL_KEYWORDS)

KEYWORDS_FOR_KMEANS_NAMING = [item for sublist in KEYWORDS_FOR_CLASSIFICATION.values() for item in sublist]

class CLIPFeatureExtractor:
    def __init__(self, clip_model_name='ViT-B-32', pretrained_dataset='laion2b_s34b_b79k'):

        self.model = None
        self.preprocess = None
        self.clip_model_name = clip_model_name
        self.pretrained_dataset = pretrained_dataset
        self.clip_tokenizer = None

        self.keyword_classification_map = {}
        self.keyword_categories = []
        self.keyword_features_np = None
        self.kmeans_naming_text_features_np = None

    def _load_clip_model(self):
        """CLIPモデルをロードします。"""
        if self.model is None:
            import torch # CLIPモデルロードに必要
            import open_clip # CLIPモデルロードに必要
            #
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"CLIPモデル ({self.clip_model_name}, {self.pretrained_dataset}) をロード中...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.pretrained_dataset
            )
            self.model.eval().to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
            print("CLIPモデルのロード完了。")

    def _prepare_keyword_features(self):
        """分類キーワードとK-Means命名用キーワードのCLIP特徴量を準備します。"""
        self._load_clip_model()

        if self.keyword_features_np is not None:
            return # 既に準備済みならスキップ
        
        import torch

        for category, kw_list in KEYWORDS_FOR_CLASSIFICATION.items():
            text = self.clip_tokenizer(kw_list[0]).to(self.device)
            with torch.no_grad():
                text_feat = self.model.encode_text(text).cpu().numpy()[0]
            self.keyword_classification_map[category] = text_feat / np.linalg.norm(text_feat)
        
        self.keyword_categories = list(self.keyword_classification_map.keys())
        self.keyword_features_np = np.array(list(self.keyword_classification_map.values()))
        print(f"分類キーワード {len(self.keyword_categories)}個の特徴量準備完了。")

        self.kmeans_naming_text_features = []
        with torch.no_grad():
            for kw in KEYWORDS_FOR_KMEANS_NAMING:
                text = self.clip_tokenizer(kw).to(self.device)
                text_feat = self.model.encode_text(text).cpu().numpy()[0]
                self.kmeans_naming_text_features.append(text_feat / np.linalg.norm(text_feat))
        self.kmeans_naming_text_features_np = np.array(self.kmeans_naming_text_features)
        print(f"K-Means命名用キーワード {len(KEYWORDS_FOR_KMEANS_NAMING)}個の特徴量準備完了。")

    def extract_features_from_paths(self, image_paths):
        """
        画像パスのリストからCLIP特徴量を抽出します。
        戻り値: (numpy配列の特徴量, 抽出に成功した画像の元のインデックス)
        """
        self._load_clip_model() # 確実にロードされていることを確認

        features = []
        processed_indices = []

        import torch
        
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                image = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = self.model.encode_image(image).cpu().numpy()[0]
                features.append(feat)
                processed_indices.append(i)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                    sys.stdout.write(f"\r進捗 (特徴量抽出): {i + 1}/{len(image_paths)}")
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("\n中断されました。")
                raise
            except Exception as e:
                print(f"\nエラー発生（{path}）：{e} - この画像をスキップします。", file=sys.stderr)
                continue

        if not features:
            print("エラー: featuresが空です。画像の読み込みに失敗した可能性があります。", file=sys.stderr)
            return np.array([]), []

        features_np = np.array(features)
        return features_np, processed_indices

    def extract_features_from_text(self, text_query: str) -> np.ndarray:
        """
        テキストクエリからCLIP特徴量を抽出します。
        戻り値: 正規化されたNumPy配列の特徴量
        """
        self._load_clip_model() # ここで_load_clip_model()を呼び出すことで、必要な場合にモデルがロードされる
        import torch # メソッド内でインポート

        text = self.clip_tokenizer(text_query).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        
        # 特徴量を正規化してNumPy配列として返す
        return text_features.cpu().numpy()[0] / np.linalg.norm(text_features.cpu().numpy()[0])

    def get_keyword_features(self):
        """分類キーワードの特徴量（NumPy配列）を返します。"""
        self._prepare_keyword_features()
        return self.keyword_features_np

    def get_keyword_categories(self):
        """分類キーワードのカテゴリ名リストを返します。"""
        self._prepare_keyword_features()
        return self.keyword_categories

    def get_kmeans_naming_features(self):
        """K-Means命名用キーワードの特徴量（NumPy配列）を返します。"""
        self._prepare_keyword_features()
        return self.kmeans_naming_text_features_np

# テスト用
# if __name__ == '__main__':
#     # ダミー画像ファイルを作成 (テスト用)
#     from PIL import Image
#     dummy_img_path = "temp_test_image.png"
#     Image.new('RGB', (60, 30), color = 'red').save(dummy_img_path)

#     extractor = CLIPFeatureExtractor()
#     features, indices = extractor.extract_features_from_paths([dummy_img_path])
#     print(f"\n抽出された特徴量の形状: {features.shape}")
#     print(f"キーワードカテゴリの数: {len(extractor.get_keyword_categories())}")

#     os.remove(dummy_img_path)
