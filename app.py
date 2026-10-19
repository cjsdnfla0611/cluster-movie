import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
# 결측치 제거 대상 열 및 first_week_audi 0 제거
valid_mask = (
    df_raw[['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi']].notnull().all(axis=1) &
    (df_raw['first_week_audi'] > 0) &
    (df_raw['first_scrn'] > 0) &
    (df_raw['total_audi'] > 0)
)

df = df_raw[valid_mask].copy()

# 파생 변수 및 전처리 속성 계산 (원래 단위 보존)
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

# 5. 전체 편수 및 묶은 편수 한 줄 표시
st.write(f"**전체 영화 편수:** {len(df_raw):,}편 | **분석에 사용된 영화 편수:** {len(df):,}편")
st.markdown("---")

# 6. 군집화에 사용할 속성 선택 (다중 선택, 기본 4개)
st.subheader("1. 군집화 속성 선택")
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=list(feature_map.keys()),
    default=list(feature_map.keys())
)

if len(selected_feature_names) < 2:
    st.warning("군집화를 위해 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_map[name] for name in selected_feature_names]

# 7. StandardScaler + KMeans (n_clusters=3, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
raw_clusters = kmeans.fit_predict(X_scaled)
df['raw_cluster'] = raw_clusters

# 8. 누적 관객(total_audi) 평균이 큰 순서대로 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = df.groupby('raw_cluster')['total_audi'].mean().sort_values(ascending=False)
label_map = {old_cluster: new_label for old_cluster, new_label in zip(cluster_audi_mean.index, ['㉮', '㉯', '㉰'])}
df['cluster'] = df['raw_cluster'].map(label_map)

# 색상 지정 (㉮, ㉯, ㉰ 별 고정 색상)
category_orders = {'cluster': ['㉮', '㉯', '㉰']}

st.markdown("---")
st.subheader("2. 산점도 시각화")

# 9. 2차원 산점도
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

# 10. 3차원 산점도
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

# 11. 묶음마다 편수와 네 속성의 원래 단위 평균 표 표시
st.subheader("3. 묶음별 요약 통계 (원래 단위 평균)")

# 원래 단위 속성 계산
df['orig_first_scrn'] = df['first_scrn']
df['orig_total_audi'] = df['total_audi']
df['orig_days_in_top10'] = df['days_in_top10']
df['orig_long_run_index'] = df['long_run_index']

summary_df = df.groupby('cluster').agg(
    영화편수=('movieCd', 'count'),
    스크린수_평균=('orig_first_scrn', 'mean'),
    누적관객수_평균=('orig_total_audi', 'mean'),
    `10위권일수_평균`=('orig_days_in_top10', 'mean'),
    롱런지수_평균=('orig_long_run_index', 'mean')
).reindex(['㉮', '㉯', '㉰'])

summary_df.columns = ['영화 편수', '스크린 수 평균', '누적 관객 수 평균', '10위권 일수 평균', '롱런 지수 평균']

# 보기 좋은 형식으로 수치 포맷팅
formatted_summary = summary_df.copy()
formatted_summary['영화 편수'] = formatted_summary['영화 편수'].map('{:,}편'.format)
formatted_summary['스크린 수 평균'] = formatted_summary['스크린 수 평균'].map('{:,.1f}개'.format)
formatted_summary['누적 관객 수 평균'] = formatted_summary['누적 관객 수 평균'].map('{:,.0f}명'.format)
formatted_summary['10위권 일수 평균'] = formatted_summary['10위권 일수 평균'].map('{:,.1f}일'.format)
formatted_summary['롱런 지수 평균'] = formatted_summary['롱런 지수 평균'].map('{:,.2f}'.format)

st.dataframe(formatted_summary, use_container_width=True)

st.markdown("---")

# 12. 묶음마다 누적 관객이 많은 순서로 상위 5개 영화 제목 표시
st.subheader("4. 묶음별 누적 관객 수 상위 5개 영화")

cols = st.columns(3)
for idx, cluster_label in enumerate(['㉮', '㉯', '㉰']):
    with cols[idx]:
        st.markdown(f"### 묶음 {cluster_label}")
        top5_movies = df[df['cluster'] == cluster_label].sort_values(by='total_audi', ascending=False).head(5)
        
        for rank, (_, row) in enumerate(top5_movies.iterrows(), start=1):
            st.write(f"**{rank}. {row['movieNm']}** ({row['total_audi']:,}명)")
