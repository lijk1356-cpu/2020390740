import pandas as pd
import numpy as np
import re

# ========== 0) 데이터 로드 ==========
df = pd.read_csv('C:/고비어 실습/amazon.csv')

# ========== 1) 문자열 정리 함수 ==========
def to_float_keep_dot(x):
    """문자열에서 숫자/소수점만 남겨 float 변환. 변환불가 시 NaN"""
    if pd.isna(x):
        return np.nan
    s = re.sub(r'[^0-9.]', '', str(x))
    return float(s) if s != '' else np.nan

def to_int_digits(x):
    """문자열에서 숫자만 남겨 int 변환. 변환불가 시 NaN"""
    if pd.isna(x):
        return np.nan
    s = re.sub(r'\D', '', str(x))
    return int(s) if s != '' else np.nan

# ========== 2) 수치형 컬럼 정제 & 변환 ==========
# 가격/할인율
df['discounted_price']    = df['discounted_price'].apply(to_float_keep_dot)
df['actual_price']        = df['actual_price'].apply(to_float_keep_dot)
df['discount_percentage'] = df['discount_percentage'].apply(to_float_keep_dot)

# 평점/리뷰수
df['rating']       = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = df['rating_count'].apply(to_int_digits)

# 필수값 결측치 제거 (평점 포함)
df = df.dropna(subset=['actual_price', 'discounted_price', 'discount_percentage', 'rating'])

# ========== 3) 카테고리 분해 (최상위/최하위) ==========
def split_cat_first(x):
    if isinstance(x, str):
        return x.split('|')[0].strip()
    return np.nan

def split_cat_last(x):
    if isinstance(x, str):
        return x.split('|')[-1].strip()
    return np.nan

df['main_category']  = df['category'].apply(split_cat_first)
df['category_split'] = df['category'].apply(split_cat_last)

# ========== 4) 가격 이상치 제거 (상위 1% 컷) ==========
upper_limit = df['actual_price'].quantile(0.99)
df = df[df['actual_price'] <= upper_limit]

# ========== 5) 카테고리 축소 (Top5 + Other) ==========
top5 = df['main_category'].value_counts().nlargest(5).index
df['main_category'] = df['main_category'].apply(lambda x: x if x in top5 else 'Other')

# ========== 6) 타입 마무리 ==========
df['discounted_price']    = df['discounted_price'].round().astype(int)
df['actual_price']        = df['actual_price'].round().astype(int)
df['discount_percentage'] = df['discount_percentage'].round().astype(int)
df['rating_count']        = df['rating_count'].astype('Int64')  # nullable int

# ========== 7) CSV 저장 ==========
output_path_csv = 'C:/고비어 실습/amazon_cleaned.csv'
df.to_csv(output_path_csv, index=False, encoding='utf-8')

print('✅ Cleaning & Save done:', output_path_csv)
