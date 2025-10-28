import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1) 데이터 로드
# =========================
df = pd.read_csv('C:/고비어 실습/amazon_cleaned.csv')

# 안전 장치: 필수 컬럼 체크
required_cols = [
    "actual_price","discounted_price","discount_percentage",
    "rating","rating_count","main_category","category_split"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼 누락: {missing}")

# rating_count는 nullable 정수일 수 있으므로 분석용 보정
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

# =========================
# 2) 요약 통계 & 기본 점검
# =========================
report_lines = []

def add_report_line(line):
    report_lines.append(line)
    print(line)

add_report_line("=== BASIC INFO ===")
add_report_line(f"rows: {len(df)}, cols: {len(df.columns)}")

num_cols = ["actual_price","discounted_price","discount_percentage","rating","rating_count"]
desc = df[num_cols].describe().T
add_report_line("\n[Summary Stats]\n" + desc.to_string())

# 제약 체크
checks = {
    "discounted_le_actual_ratio": float((df["discounted_price"] <= df["actual_price"]).mean()),
    "rating_range_ok": int(((df["rating"] >= 1) & (df["rating"] <= 5)).all()),
    "discount_percentage_range_ok": int(((df["discount_percentage"] >= 0) & (df["discount_percentage"] <= 100)).all()),
    "zero_price_count": int(((df["actual_price"] == 0) | (df["discounted_price"] == 0)).sum()),
}
add_report_line("\n[Sanity Checks]\n" + "\n".join([f"- {k}: {v}" for k,v in checks.items()]))



# =========================
# 3) 유틸: 안전 로그 변환
# =========================
def safe_log_series(s: pd.Series, eps: float = 1.0):
    """0/음수 방지용 log1p 변환 권장하지만 여기선 직관적 로그 사용.
       히스토그램/산점도에서 스케일 왜곡 줄이기."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.clip(lower=0)  # 음수 방지
    return np.log1p(s)   # log1p 사용(0 처리에 안전)

# =========================
# 4) 분포(Distribution) 분석
# =========================

def save_show(fig, name):
    plt.show()
    plt.close(fig)

# (A) 가격 분포: 정가/할인가
for col in ["actual_price", "discounted_price"]:
    fig = plt.figure()
    plt.hist(df[col].dropna(), bins=50)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    save_show(fig, f"hist_{col}.png")

    # 로그 스케일(권장)
    fig = plt.figure()
    plt.hist(safe_log_series(df[col]).dropna(), bins=50)
    plt.title(f"Histogram of log(1+){col}")
    plt.xlabel(f"log(1+){col}")
    plt.ylabel("count")
    save_show(fig, f"hist_log_{col}.png")

# (B) 할인율 분포
fig = plt.figure()
plt.hist(df["discount_percentage"].dropna(), bins=50)
plt.title("Histogram of discount_percentage")
plt.xlabel("discount_percentage")
plt.ylabel("count")
save_show(fig, "hist_discount_percentage.png")

# (C) 평점 분포
fig = plt.figure()
plt.hist(df["rating"].dropna(), bins=40)
plt.title("Histogram of rating")
plt.xlabel("rating (1~5)")
plt.ylabel("count")
save_show(fig, "hist_rating.png")

# (D) 리뷰 수 분포 (강한 왜도 → 로그)
fig = plt.figure()
plt.hist(safe_log_series(df["rating_count"]).dropna(), bins=50)
plt.title("Histogram of log(1+) rating_count")
plt.xlabel("log(1+rating_count)")
plt.ylabel("count")
save_show(fig, "hist_log_rating_count.png")


# =========================
# 5) 상관관계(Correlation)
# =========================
corr_pearson = df[num_cols].corr(method="pearson")
corr_spearman = df[num_cols].corr(method="spearman")

add_report_line("\n[Pearson Correlation]\n" + corr_pearson.to_string())
add_report_line("\n[Spearman Correlation]\n" + corr_spearman.to_string())



# =========================
# 6) 산점도 & 간단 회귀(직선)
# =========================
def scatter_with_trend(x, y, x_label, y_label, fname, log_x=False, log_y=False):
    x_data = df[x].astype(float)
    y_data = df[y].astype(float)

    if log_x:
        x_plot = safe_log_series(x_data)
        x_lbl = f"log(1+){x_label}"
    else:
        x_plot = x_data
        x_lbl = x_label

    if log_y:
        y_plot = safe_log_series(y_data)
        y_lbl = f"log(1+){y_label}"
    else:
        y_plot = y_data
        y_lbl = y_label

    # 유효값 필터
    mask = x_plot.notna() & y_plot.notna()
    xp = x_plot[mask].values
    yp = y_plot[mask].values
    if len(xp) < 5:
        return

    # 회귀선 (1차)
    coeff = np.polyfit(xp, yp, 1)
    poly = np.poly1d(coeff)
    y_hat = poly(xp)
    # R^2
    ss_res = np.sum((yp - y_hat)**2)
    ss_tot = np.sum((yp - np.mean(yp))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    fig = plt.figure()
    plt.scatter(xp, yp, s=10, alpha=0.6)
    # 회귀선
    x_line = np.linspace(min(xp), max(xp), 200)
    plt.plot(x_line, poly(x_line))
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(f"{y_label} vs {x_label} (R^2={r2:.3f})")
    save_show(fig, fname)

# (1) 평점 vs 리뷰수 (리뷰수 로그)
scatter_with_trend(
    x="rating", y="rating_count",
    x_label="rating", y_label="rating_count",
    fname="scatter_rating_vs_ratingcount.png",
    log_x=False, log_y=True
)

# (2) 가격 vs 리뷰수 (리뷰수 로그)
scatter_with_trend(
    x="discounted_price", y="rating_count",
    x_label="discounted_price", y_label="rating_count",
    fname="scatter_price_vs_ratingcount.png",
    log_x=True, log_y=True  # 가격/리뷰수 모두 로그 추천
)

# (3) 할인율 vs 리뷰수 (리뷰수 로그)
scatter_with_trend(
    x="discount_percentage", y="rating_count",
    x_label="discount_percentage", y_label="rating_count",
    fname="scatter_discount_vs_ratingcount.png",
    log_x=False, log_y=True
)


# =========================
# 7) 카테고리별 비교 (집계표 + 그래프)
# =========================
group_cols = ["main_category"]
agg_map = {
    "actual_price": ["mean","median"],
    "discounted_price": ["mean","median"],
    "discount_percentage": ["mean","median"],
    "rating": ["mean","median"],
    "rating_count": ["mean","median","sum"]
}
cat_stats = df.groupby(group_cols).agg(agg_map)
# 컬럼 평탄화
cat_stats.columns = ["_".join(col).strip() for col in cat_stats.columns.values]
cat_stats = cat_stats.sort_values("rating_count_sum", ascending=False)

# 바 차트 그리기 함수
def bar_chart_from_series(series: pd.Series, title: str, xlabel: str, fname: str, rotate=30):
    fig = plt.figure()
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.xticks(rotation=rotate, ha="right")
    save_show(fig, fname)

# (A) 카테고리별 평균 할인가
bar_chart_from_series(
    cat_stats["discounted_price_mean"],
    "Avg Discounted Price by Main Category",
    "main_category",
    "bar_avg_discounted_price.png"
)

# (B) 카테고리별 평균 할인율
bar_chart_from_series(
    cat_stats["discount_percentage_mean"],
    "Avg Discount Percentage by Main Category",
    "main_category",
    "bar_avg_discount_percentage.png"
)

# (C) 카테고리별 평균 평점
bar_chart_from_series(
    cat_stats["rating_mean"],
    "Avg Rating by Main Category",
    "main_category",
    "bar_avg_rating.png"
)

# (D) 카테고리별 리뷰수 합계 (인기 지표)
bar_chart_from_series(
    cat_stats["rating_count_sum"],
    "Total Rating Count by Main Category",
    "main_category",
    "bar_total_rating_count.png"
)


# =========================
# 8) 상위/하위 제품 랭킹(옵션)
# =========================
# 리뷰 수 Top-N
top_n = 20
rank_popular = df.sort_values("rating_count", ascending=False).head(top_n)


# 할인율 Top-N
rank_discount = df.sort_values("discount_percentage", ascending=False).head(top_n)


# 가성비 지표(간단): 할인 후 가격이 낮고 평점 높은 순
df["_value_score"] = (5 - df["rating"]) * 0.0 + df["discounted_price"]  # 필요시 커스터마이즈
rank_value = df.sort_values(["rating","discounted_price"], ascending=[False, True]).head(top_n)
rank_value.drop(columns=["_value_score"], inplace=True)


