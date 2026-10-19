import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 페이지 설정
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

# 2. 데이터 불러오기 (캐싱 처리)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df_raw = load_data()

# 3. 데이터 전처리 및 파생 변수 생성
valid_mask = (
    df_raw[['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']].notnull().all(axis=1) &
    (df_raw['first_week_audi'] > 0) &
    (df_raw['first_scrn'] > 0) &
    (df_raw['total_audi'] > 0)
)

df = df_raw[valid_mask].copy()

# 파생 변수 및 전처리 속성 계산
df['log_first_scrn'] = np.log10(df['first_scrn'])
df['log_total_audi'] = np.log10(df['total_audi'])
df['days_in_top10_feat'] = df['days_in_top10']

# 롱런 지수 = total_audi / first_week_audi (20 초과는 20으로 Clip)
df['long_run_index'] = (df['total_audi'] / df['first_week_audi']).clip(upper=20)

# 속성 매핑 Dict
feature_map = {
    '스크린 수 (log10)': 'log_first_scrn',
    '누적 관객 수 (log10)': 'log_total_audi',
    '10위권 일수': 'days_in_top10_feat',
    '롱런 지수': 'long_run_index'
}

# 기호 데이터 (2~7개 대응)
CLUSTER_SYMBOLS = ['㉮', '㉯', '㉰', '㉱', '㉲', '㉳', '㉴']

# 4. 전체 편수 및 묶은 편수 한 줄 표시
st.write(f"**전체 영화 편수:** {len(df_raw):,}편 | **분석에 사용된 영화 편수:** {len(df):,}편")
st.markdown("---")

# 5. 군집화에 사용할 속성 및 묶음 수 선택
st.subheader("1. 군집화 설정")
col_feat, col_k = st.columns([2, 1])

with col_feat:
    selected_feature_names = st.multiselect(
        "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
        options=list(feature_map.keys()),
        default=list(feature_map.keys())
    )

with col_k:
    n_clusters = st.slider("묶음 수 (k) 선택:", min_value=2, max_value=7, value=3)

if len(selected_feature_names) < 2:
    st.warning("군집화를 위해 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_map[name] for name in selected_feature_names]

# 6. StandardScaler + KMeans
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df['raw_cluster'] = raw_clusters

# 7. 누적 관객(total_audi) 평균이 큰 순서대로 ㉮~㉴ 부여
cluster_audi_mean = df.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
current_symbols = CLUSTER_SYMBOLS[:n_clusters]

label_map = {old_cluster: new_label for old_cluster, new_label in zip(cluster_audi_mean.index, current_symbols)}
df['cluster'] = df['raw_cluster'].map(label_map)

category_orders = {'cluster': current_symbols}

st.markdown("---")
st.subheader("2. 산점도 시각화")

# 8. 2차원 산점도
col_x2d, col_y2d = st.columns(2)
with col_x2d:
    x_2d_name = st.selectbox("2D X축 속성 선택:", list(feature_map.keys()), index=0)
with col_y2d:
    y_2d_name = st.selectbox("2D Y축 속성 선택:", list(feature_map.keys()), index=1)

fig_2d = px.scatter(
    df,
    x=feature_map[x_2d_name],
    y=feature_map[y_2d_name],
    color='cluster',
    hover_name='movieNm',
    category_orders=category_orders,
    title=f"2D 산점도 ({x_2d_name} vs {y_2d_name})",
    labels={feature_map[x_2d_name]: x_2d_name, feature_map[y_2d_name]: y_2d_name, 'cluster': '묶음'}
)
st.plotly_chart(fig_2d, use_container_width=True)

# 9. 3차원 산점도
st.subheader("3D 산점도")
if len(selected_feature_names) < 3:
    st.info("💡 3차원 산점도를 그리려면 위에서 속성을 3개 이상 선택해 주세요.")
else:
    col_x3d, col_y3d, col_z3d = st.columns(3)
    with col_x3d:
        x_3d_name = st.selectbox("3D X축 속성 선택:", selected_feature_names, index=0)
    with col_y3d:
        y_3d_name = st.selectbox("3D Y축 속성 선택:", selected_feature_names, index=1)
    with col_z3d:
        z_3d_name = st.selectbox("3D Z축 속성 선택:", selected_feature_names, index=2)

    fig_3d = px.scatter_3d(
        df,
        x=feature_map[x_3d_name],
        y=feature_map[y_3d_name],
        z=feature_map[z_3d_name],
        color='cluster',
        hover_name='movieNm',
        category_orders=category_orders,
        title=f"3D 산점도 ({x_3d_name} / {y_3d_name} / {z_3d_name})",
        labels={
            feature_map[x_3d_name]: x_3d_name,
            feature_map[y_3d_name]: y_3d_name,
            feature_map[z_3d_name]: z_3d_name,
            'cluster': '묶음'
        }
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# 10. 묶음마다 편수와 네 속성의 원래 단위 평균 표 표시
st.subheader("3. 묶음별 요약 통계 (원래 단위 평균)")

df['orig_first_scrn'] = df['first_scrn']
df['orig_total_audi'] = df['total_audi']
df['orig_days_in_top10'] = df['days_in_top10']
df['orig_long_run_index'] = df['long_run_index']

summary_df = df.groupby('cluster').agg(
    movie_count=('movieCd', 'count'),
    scrn_mean=('orig_first_scrn', 'mean'),
    audi_mean=('orig_total_audi', 'mean'),
    top10_mean=('orig_days_in_top10', 'mean'),
    longrun_mean=('orig_long_run_index', 'mean')
).reindex(current_symbols)

summary_df.columns = ['영화 편수', '스크린 수 평균', '누적 관객 수 평균', '10위권 일수 평균', '롱런 지수 평균']

formatted_summary = summary_df.copy()
formatted_summary['영화 편수'] = formatted_summary['영화 편수'].map('{:,}편'.format)
formatted_summary['스크린 수 평균'] = formatted_summary['스크린 수 평균'].map('{:,.1f}개'.format)
formatted_summary['누적 관객 수 평균'] = formatted_summary['누적 관객 수 평균'].map('{:,.0f}명'.format)
formatted_summary['10위권 일수 평균'] = formatted_summary['10위권 일수 평균'].map('{:,.1f}일'.format)
formatted_summary['롱런 지수 평균'] = formatted_summary['롱런 지수 평균'].map('{:,.2f}'.format)

st.dataframe(formatted_summary, use_container_width=True)

st.markdown("---")

# 11. 묶음마다 누적 관객이 많은 순서로 상위 5개 영화 제목 표시
st.subheader("4. 묶음별 누적 관
