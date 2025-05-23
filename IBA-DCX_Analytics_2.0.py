import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import os
import time
import re
import base64
import tempfile
import random
import itertools
import gc
import networkx as nx
import squarify
import urllib.request
import datetime
import pytz
import uuid
from collections import Counter, defaultdict
from wordcloud import WordCloud
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import altair as alt
import gspread
from random import choice
from google.oauth2.service_account import Credentials

        
# 강제 Light Mode
st.markdown("""
<style>
body, .stApp { background-color: white !important; color: black !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], .css-1d391kg, .css-1v0mbdj {
    background-color: white !important;
    color: black !important;
}
.markdown-text-container { color: black !important; }
</style>
""", unsafe_allow_html=True)

# 전역 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

FONT_PATH = "./NanumGothic-Regular.ttf"
font_prop = fm.FontProperties(fname=FONT_PATH)
fm.fontManager.addfont(FONT_PATH)
font_name = font_prop.get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

MAX_USERS = 2  # 최대 동시 사용자 수
TIMEOUT_MINUTES = 15  # 사용시간 (분)
TIMEZONE = pytz.timezone('Asia/Seoul')

# 데이터셋 매핑
KEYWORD_COLUMNS = ['맛', '서비스', '가격', '위치', '분위기', '위생']
KEYWORD_ENGLISH_MAP = {
    '맛': 'Taste',
    '서비스': 'Service',
    '가격': 'Price',
    '위치': 'Location',
    '분위기': 'Atmosphere',
    '위생': 'Hygiene'
}
DATASET_MAP = {
    '부산대': 'IBA-DCX_Analytics_2.0_PNU.csv',
    '경희대': 'IBA-DCX_Analytics_2.0_KHU.csv',
    '제주도': 'IBA-DCX_Analytics_2.0_Jeju.csv'
}

# Location English Mapping
LOCATION_ENGLISH_MAP = {
    '부산대': 'Pusan National University',
    '경희대': 'Kyung Hee University',
    '제주도': 'Jeju Island'
}


###############################################
# 리소스 관리

@st.cache_resource
def get_classifier():
    return pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")

@st.cache_data
def load_dataset(dataset_name: str) -> pd.DataFrame:
    import gdown
    file_ids = {
        'IBA-DCX_Analytics_2.0_PNU.csv': '1jfMMwnXi5zUOGE6F34B-KjQvfH5jjKmu',
        'IBA-DCX_Analytics_2.0_KHU.csv': '1pqbNRLg8SdsmnZgi9JnqkxjDp7VUPlb4',
        'IBA-DCX_Analytics_2.0_Jeju.csv': '1OeB_VE4bWYCLFAI85ozT7DwiL8V1W7yR'
    }
    file_id = file_ids.get(dataset_name)
    output = f".cache_{dataset_name}"
    if not os.path.exists(output):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=True)
    use_cols = ['Name', 'Content', 'Tokens', 'Image_Links', '맛', '서비스', '가격', '위치', '분위기', '위생', 'review_sentences', 'Date']
    return pd.read_csv(output, usecols=use_cols)

@st.cache_resource
def train_lda_model(corpus, _dictionary, num_topics=10):
    return LdaModel(corpus, num_topics=num_topics, id2word=_dictionary, passes=5)

@st.cache_resource
def get_lda_vis_data(_model, corpus, _dictionary):
    return gensimvis.prepare(_model, corpus, _dictionary)

###############################################

# 구글시트 설정
SERVICE_ACCOUNT_FILE = "dcx-tool-credentials.json"
SPREADSHEET_ID = "16ZU-AypnTli-BlXa2Tgvooe4YKsd0T1NqC3nZWsig_E"
SHEET_NAME = "DCX"
TIMEZONE = pytz.timezone('Asia/Seoul')

# 구글 시트 큐 설정
@st.cache_resource
def get_worksheet():
    creds = Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SPREADSHEET_ID)
    return sheet.worksheet(SHEET_NAME)

def load_queue(force_reload=False):
    if not force_reload:
        if st.session_state.get('queue_checked', False):
            return []
    ws = get_worksheet()
    return ws.get_all_records()

def save_queue(data):
    ws = get_worksheet()
    if not data:
        num_rows = len(ws.get_all_values())
        if num_rows > 1:
            ws.delete_rows(2, num_rows)
        ws.update('A1:B1', [["user_id", "start_time"]])
    else:
        update_data = [["user_id", "start_time"]] + [[row['user_id'], row['start_time']] for row in data]
        num_rows = len(ws.get_all_values())
        if num_rows > 1:
            ws.delete_rows(2, num_rows)
        end_row = len(update_data)
        ws.update(f"A1:B{end_row}", update_data)


def clean_expired_sessions():
    ws = get_worksheet()
    all_values = ws.get_all_values()
    now = datetime.datetime.now(tz=TIMEZONE)
    valid_rows = []
    for row in all_values[1:]:
        if not row or len(row) < 2:
            continue
            
        user_id, start_time_str = row[0], row[1]

        if not start_time_str.strip():
            continue

        try:
            start_time_naive = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            start_time = TIMEZONE.localize(start_time_naive)
            if (now - start_time).total_seconds() <= TIMEOUT_MINUTES * 60:
                valid_rows.append([user_id, start_time_str])
        except Exception:
            continue

    final_data = [["user_id", "start_time"]] + valid_rows

    ws.clear()
    time.sleep(1)
    ws.update('A1:B' + str(len(final_data)), final_data)
    time.sleep(1)

    st.cache_data.clear()

query_params = st.query_params
user_id = query_params.get('user_id')

if user_id is None:
    user_id = str(uuid.uuid4())[:8]
    st.query_params["user_id"] = user_id

if 'user_id' not in st.session_state:
    query_params = st.query_params
    if 'user_id' in query_params:
        st.session_state['user_id'] = query_params['user_id']
    else:
        new_user_id = str(uuid.uuid4())[:8]
        st.query_params["user_id"] = new_user_id
        st.session_state['user_id'] = new_user_id


if 'queue_checked' not in st.session_state:
    clean_expired_sessions()
    time.sleep(3)

    data = load_queue(force_reload=True)
    now = datetime.datetime.now(tz=TIMEZONE)
    user_ids = [row['user_id'] for row in data]

    if st.session_state['user_id'] not in user_ids:
        if len(user_ids) < MAX_USERS:
            ws = get_worksheet()
            ws.append_row([st.session_state['user_id'], now.strftime("%Y-%m-%d %H:%M:%S")])
            st.session_state['queue_checked'] = True
            st.session_state['start_time'] = now
            st.cache_data.clear()
        else:
            wait_sec = (len(user_ids) - MAX_USERS + 1) * TIMEOUT_MINUTES * 60
            m, s = divmod(wait_sec, 60)
            st.error(f"The maximum number of users has been reached. Estimated waiting time: {m}분")
            st.stop()
    else:
        st.session_state['queue_checked'] = True



# 사용 시간 만료 체크
expiration_time = st.session_state['start_time'] + datetime.timedelta(minutes=TIMEOUT_MINUTES)
now = datetime.datetime.now(tz=TIMEZONE)

if now >= expiration_time:
    data = load_queue(force_reload=True)
    new_data = [row for row in data if row['user_id'] != st.session_state.get('user_id')]
    save_queue(new_data)
    st.cache_data.clear()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.sidebar.warning("⏰ Your usage time has ended. Please reconnect.")
    st.stop()

else:
    expiration_str = expiration_time.strftime("%Y-%m-%d %H:%M:%S")
    st.sidebar.success(f"⏳ Your expiration time: {expiration_str}")

if st.sidebar.button("✅ To End Use"):
    user_id = st.session_state.get('user_id')

    ws = get_worksheet()
    all_values = ws.get_all_values()

    target_row = None
    for idx, row in enumerate(all_values[1:], start=2):
        if row and row[0] == user_id:
            target_row = idx
            break

    if target_row:
        ws.delete_rows(target_row)
        time.sleep(2)
        st.cache_data.clear()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.success("✅ The Usage has normally ended.")
    st.stop()


    
###############################################
# 기능
def compute_sentiment(text, classifier):
    if not isinstance(text, str):
        text = str(text)
    result = classifier(text)
    return result[0]['score'] if result[0]['label'] == 'LABEL_1' else 1 - result[0]['score']

def render_title(location, store):
    st.title(f"{location} - {store}")

def clean_memory(keys):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    plt.clf()
    plt.close('all')
    gc.collect()

def clean_tokens(text):
    text = re.sub(r"[^\w\s]", "", text)  # 콤마, 마침표 등 제거
    return text.split()

# 불용어 정의
stopwords = {
    # 조사 / 대명사 / 지시어
    '이', '그', '저', '것', '거', '곳', '수', '좀', '처럼', '까지', '에도', '에도요', '이나', '라도',

    # 접속사 / 연결어
    '그리고', '그래서', '그러나', '하지만', '또한', '즉', '결국', '때문에', '그래도',

    # 서술어 / 어미 / 보조 용언
    '합니다', '해요', '했어요', '하네요', '하시네요', '하시던데요', '같아요', '있어요', '없어요',
    '되네요', '되었어요', '보여요', '느껴져요', '하겠습니다', '되겠습니다', '있습니다', '없습니다',
    '합니다', '이에요', '이라', '해서',

    # 감탄사 / 리뷰 특유 표현
    'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', '^^', '^^;;', '~', '~~', '!!!', '??', '!?', '?!', '...', '!!', '~!!', '~^^!!',

    # 강조 표현
    '아주', '정말', '진짜', '엄청', '매우', '완전', '너무', '굉장히', '많이', '많아요', '적당히', '넘'

    # 기타
    '정도', '느낌', '같은', '니당', '네요', '있네요', '이네요', '이라서',
    '해서요', '보니까', '봤어요', '먹었어요', '마셨어요', '갔어요', '봤습니다', '하는', '하게', '드네', '또시',
    '이랑', '하고', '해도', '해도요', '때문에요', '이나요', '정도에요'
}



###############################################
# 모듈

# 사용법
def render_usage_tab():
    st.header("📊 IBA-DCX Tool")

    st.markdown("""
    <div style="background-color: #f5f8fa; padding: 20px; border-radius: 12px; border-left: 6px solid #0d6efd;">
        <p style="font-size:16px;">
        <strong>IBA DCX Tool</strong>은 <strong>온라인 리뷰 분석</strong>을 통해 <strong>고객 경험 기반 경영전략 수립</strong>을 지원하는 도구입니다.<br>
        본 도구를 이용하여 아래와 같은 기능을 실행할 수 있습니다.
        </p>
        <ul style="padding-left: 20px; font-size:15px; line-height: 1.6;">
            <li>Word Cloud Generation</li>
            <li>Treemap Chart Creation</li>
            <li>Frequency-Based Network Analysis</li>
            <li>LDA Topic Modeling</li>
            <li>Customer Satisfaction Analysis via Sentiment Analysis</li>
        </ul>
    </div>
    <br>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("### ✅ How to Use")

    st.markdown("""
    <div style="padding: 16px; background-color: #f9f9f9; border-radius: 10px; font-size: 15px; line-height: 1.7;">
        <ol>
            <li>In the <strong>sidebar</strong>, select a <span style="color:#0d6efd;">location</span> and <span style="color:#0d6efd;">store name</span>, then click the <strong>‘Confirm’</strong> button.</li>
            <li>Choose the desired analysis function from the <strong>function selection dropdown</strong>.</li>
            <li>To start a new analysis, <strong>refresh the page</strong> and begin again.</li>
            <li><strong>Sentiment analysis</strong> may take longer depending on the number of reviews.</li>
            <li>This tool is designed for <strong>Light Mode</strong>. You can change the theme via the menu (⋮) in the top-right corner.</li>
        </ol>
        <p style="font-size:14px; color:gray;">
        ⚠️ If you encounter issues, please contact the email address provided in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)


# 리뷰불러오기
def render_review_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: 리뷰 요약 및 이미지")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)
    image_links = df_store['Image_Links'].tolist()
    reviews = df_store['Content'].fillna('').astype(str).tolist()
    image_pattern = r'https?://[\S]+\.(?:jpg|jpeg|png|gif)'
    all_links, all_reviews = [], []
    for idx, link_str in enumerate(image_links):
        if isinstance(link_str, str):
            links = re.findall(image_pattern, link_str)
            all_links.extend(links)
            all_reviews.extend([reviews[idx]] * len(links))

    avg_length = np.mean([len(r) for r in reviews if isinstance(r, str)]) if reviews else 0
    st.markdown("### 📊 Review Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total number of Reviews", f"{len(df_store)} reviews")
    with col2:
        st.metric("Total number of Images", f"{len(all_links)} images")
    with col3:
        st.metric("Average Review Length", f"{avg_length:.1f}")
    highlight_keywords = ['맛', '서비스', '가격', '위치', '분의기', '위생']
    def highlight_keywords_in_text(text):
        for kw in highlight_keywords:
            text = re.sub(f"({kw})", r"<span style='color:#d9480f; font-weight:bold;'>\1</span>", text)
        return text

    st.markdown("### Top Review 🖼️ ")
    NUM_CARDS = 6
    if 'review_indices' not in st.session_state:
        st.session_state.review_indices = random.sample(range(len(all_links)), min(NUM_CARDS, len(all_links)))
    if st.button("🔄 Look at other reviews"):
        st.session_state.review_indices = random.sample(range(len(all_links)), min(NUM_CARDS, len(all_links)))
    for row_start in range(0, len(st.session_state.review_indices), 3):
        row_cols = st.columns(3)
        for i in range(3):
            if row_start + i >= len(st.session_state.review_indices):
                break
            idx = st.session_state.review_indices[row_start + i]
            with row_cols[i]:
                st.markdown(f"""
                <div style="height: 180px; overflow: hidden; border-radius: 8px;">
                    <img src="{all_links[idx]}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" />
                </div>
                """, unsafe_allow_html=True)

                highlighted = highlight_keywords_in_text(all_reviews[idx])
                st.markdown(f"""
                <div style="padding:12px; background-color:#f9f9f9; border-radius:10px;
                            box-shadow:0 2px 4px rgba(0,0,0,0.08); margin-top:8px;
                            height:150px; overflow:auto;">
                    <p style="font-size:14px; color:#333;">{highlighted}</p>
                </div>
                """, unsafe_allow_html=True)

# 워드클라우드
# 선명한 색상 리스트 정의
VIVID_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e31a1c", "#17becf"]

# 무작위 컬러 함수 정의
def vivid_color_func(*args, **kwargs):
    return choice(VIVID_COLORS)

# 워드클라우드 탭 렌더링 함수
def render_wordcloud_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: Wordcloud")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)

    columns_to_plot = ['Content'] + KEYWORD_COLUMNS

    container = st.container()
    cols = container.columns(3)

    for idx, column in enumerate(columns_to_plot):
        col = cols[idx % 3]
        text = ' '.join(df_store[column].dropna().map(str))
        tokens = text.split()
        filtered_tokens = [t for t in tokens if t not in stopwords]
        filtered_text = ' '.join(filtered_tokens)

        with col:
            st.markdown(
                f"<div style='text-align:center; font-weight:bold; font-size:16px; margin-bottom:5px;'>{column}</div>",
                unsafe_allow_html=True
            )

            if filtered_text.strip():
                wordcloud = WordCloud(
                    font_path=FONT_PATH,
                    width=800,
                    height=800,
                    contour_width=1.8,
                    contour_color='black',
                    background_color='white',
                    mode='RGB',
                    color_func=vivid_color_func,  # 💡 선명한 색상 지정
                    collocations=False
                ).generate(filtered_text)

                fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
                ax.imshow(wordcloud, interpolation='nearest')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.markdown(f"""
                <div style="padding:10px; text-align:center; background-color:#f9f9f9;
                            border-radius:10px; min-height:200px; height:260px;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="color:gray;">텍스트 없음</span>
                </div>
                """, unsafe_allow_html=True)

# 트리맵
def render_treemap_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: 트리맵")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)

    columns_to_plot = ['Content'] + KEYWORD_COLUMNS
    container = st.container()
    cols = container.columns(3)

    for idx, column in enumerate(columns_to_plot):
        col = cols[idx % 3]
        text = ' '.join(df_store[column].dropna().map(str))

        tokens = text.split()
        filtered_tokens = [t for t in tokens if t not in stopwords]
        word_count = Counter(filtered_tokens)

        with col:
            st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:16px; margin-bottom:5px;'>{column}</div>", unsafe_allow_html=True)

            if filtered_tokens and len(word_count) > 0:
                most_common = word_count.most_common(10)
                sizes = [count for _, count in most_common]
                labels = [f"{word} ({count})" for word, count in most_common]

                # 색상: 상위순서별 점진적 밝기
                cmap = plt.cm.get_cmap("Blues")
                normed_sizes = [s / max(sizes) for s in sizes]
                colors = [cmap(0.3 + 0.7 * s) for s in normed_sizes]

                fig, ax = plt.subplots(figsize=(4, 4))
                squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, ax=ax, text_kwargs={'fontsize':10})
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.markdown(f"""
                <div style="padding:20px; text-align:center; background-color:#f9f9f9;
                            border-radius:10px; min-height:260px;
                            display:flex; align-items:center; justify-content:center;
                            box-shadow:0px 1px 3px rgba(0,0,0,0.05);">
                    <span style="color:gray; font-size:16px;">{column}에 대한 텍스트 없음</span>
                </div>
                """, unsafe_allow_html=True)

    with st.expander("📘 Color Description"):
        st.markdown("""
        - 트리맵의 **색상은 해당 단어의 상대적 등장 빈도**를 의미합니다.  
        - **진한 파랑색**일수록 많이 언급된 단어입니다.  
        - **연한 색상**은 상대적으로 빈도수가 낮은 단어입니다.
        """)


#네트워크 분석
def render_network_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: Network Analysis")
    df_store = df[df['Name'] == store]

    if len(df_store) < 20:
        st.warning("리뷰 수가 부족하여 네트워크 분석을 실행할 수 없습니다.")
        return

    import re
    def clean_tokens(text):
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()
    
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)

    st.subheader("Setting the Word Filter Criteria")
    total_reviews = len(df_store)
    min_value = max(1, total_reviews // 20)
    max_value = max(2, total_reviews // 10)
    default_value = (min_value + max_value) // 2

    min_freq = st.slider(
        "단어 최소 등장 횟수",
        min_value=min_value,
        max_value=max_value,
        value=default_value
    )

    word_freq = Counter(itertools.chain(*df_store['Tokens']))
    filtered_words = {w for w, c in word_freq.items() if c >= min_freq}

    df_store['Filtered_Tokens'] = df_store['Tokens'].apply(
        lambda tokens: [w for w in tokens if w in filtered_words and w not in stopwords and len(w) > 1]
    )

    co_occurrence = defaultdict(int)
    for tokens in df_store['Filtered_Tokens']:
        for pair in itertools.combinations(set(tokens), 2):
            co_occurrence[tuple(sorted(pair))] += 1

    G = nx.Graph()
    for (w1, w2), freq in co_occurrence.items():
        G.add_edge(w1, w2, weight=freq)

    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        st.warning("In this condition, there is no matching network. Please, follow the filter's criteria.")
        return

    pos = nx.spring_layout(G, k=0.5, seed=42)
    degree_centrality = nx.degree_centrality(G)

    # 등빈 상/하위 30% 기준 색상 분류
    freq_dict = {node: word_freq.get(node, 0) for node in G.nodes()}
    freq_values = list(freq_dict.values())
    upper_thresh = np.percentile(freq_values, 70)  # 상위 30%
    lower_thresh = np.percentile(freq_values, 30)  # 하위 30%

    def get_color(freq):
        if freq >= upper_thresh:
            return 'green'
        elif freq <= lower_thresh:
            return 'crimson'
        else:
            return 'skyblue'

    node_colors = [get_color(freq_dict[n]) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(top=0.88, bottom=0.15)
    node_sizes = [1000 + len(n) * 250 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', ax=ax, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family=font_prop.get_name(), ax=ax)
    
    ax.set_title(f"{store} - 네트워크 분석", fontproperties=font_prop, fontsize=16, pad=12)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("🌈 Color Criteria"):
        st.markdown("""
        - 🟢 **Green**: High Frequency words 30%  
        - 🔴 **Red**: Low Frequency words 30%  
        - 🔵 **Blue**: Medium Frequency words
        """)


# 토픽모델링
def render_topic_tab(df, store):
    st.header(f"{st.session_state.get('selected_location', '')} - {store}: Topic Modeling")
    df_store = df[df['Name'] == store]
    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(clean_tokens)
    if len(df_store) < 50:
        st.warning("Not enough reviews to run topic modeling.")
        return

    df_store['Tokens'] = df_store['Tokens'].fillna('').map(str).map(str.split)
    if len(df_store) > 300:
        df_store = df_store.sample(300, random_state=42)

    dictionary = corpora.Dictionary(df_store['Tokens'])
    corpus = [dictionary.doc2bow(text) for text in df_store['Tokens']]

    if st.button("Execute Topic Modeling"):
        lda_model = train_lda_model(corpus, dictionary)
        vis_data = get_lda_vis_data(lda_model, corpus, dictionary)
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".html") as f:
            pyLDAvis.save_html(vis_data, f.name)
            html_path = f.name
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        b64 = base64.b64encode(html_content.encode()).decode()
        st.markdown(f'<a href="data:text/html;base64,{b64}" download="lda_result.html">📁 LDA 결과 HTML 다운로드</a>', unsafe_allow_html=True)
        del lda_model, vis_data, corpus, dictionary
        gc.collect()

# 감성분석
def render_sentiment_dashboard(df, store, classifier):
    region_avg_scores = {
        '부산대': {
            'total': 89.05,
            '맛': 90.12,
            '서비스': 87.86,
            '가격': 87.02,
            '위치': 81.43,
            '분위기': 88.63,
            '위생': 89.17
        },
        '경희대': {
            'total': 88.87,
            '맛': 91.05,
            '서비스': 87.88,
            '가격': 86.01,
            '위치': 78.23,
            '분위기': 85.76,
            '위생': 89.53
        },
        '제주도': {
            'total': 88.53,
            '맛': 88.92,
            '서비스': 88.00,
            '가격': 81.22,
            '위치': 81.47,
            '분위기': 85.09,
            '위생': 89.87
        }
    }
    st.header(f"{LOCATION_ENGLISH_MAP.get('selected_location', '')} - {store}: Customer Satisfaction Analysis")
    df_store = df[df['Name'] == store]

    if len(df_store) < 50:
        st.warning("Insufficient reviews to perform sentiment analysis.")
        return

    sentiment_key = f"sentiment_scores_{store}"

    if sentiment_key not in st.session_state:
        if st.button("🧠 Start Customer Satisfaction Analysis"):
            texts = df_store['review_sentences'].dropna().astype(str).tolist()
            keyword_inputs = {col: df_store[col].dropna().astype(str).tolist() for col in KEYWORD_COLUMNS}
            total_steps = len(texts) + sum(len(v) for v in keyword_inputs.values())
            completed_steps = 0
            progress_bar = st.progress(0)

            total_scores = []
            for text in texts:
                result = classifier(text)[0]
                score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
                total_scores.append(score)
                completed_steps += 1
                progress_bar.progress(completed_steps / total_steps)

            keyword_scores = {}
            for col, col_texts in keyword_inputs.items():
                if col_texts:
                    scores = []
                    for text in col_texts:
                        result = classifier(text)[0]
                        score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
                        scores.append(score)
                        completed_steps += 1
                        progress_bar.progress(completed_steps / total_steps)
                    keyword_scores[col] = np.mean(scores) * 100
                else:
                    keyword_scores[col] = None

            st.session_state[sentiment_key] = {
                'total': np.mean(total_scores) * 100,
                'keywords': keyword_scores
            }
        else:
            st.info("📌 Click the button above to start the analysis.")
            return

    # 결과 시각화
    region_name = st.session_state.get('selected_location', '')
    region_stats = region_avg_scores.get(region_name, {})
    sentiment_data = st.session_state[sentiment_key]
    
    # 종합 점수 비교
    st.subheader("🔎 Overall Sentiment Score Comparison")
    
    store_total = sentiment_data['total']
    region_total = region_stats.get('total', None)
    
    if region_total is not None:
        diff = store_total - region_total
        trend_icon = "▲" if diff > 0 else ("▼" if diff < 0 else "▶")
        trend_color = "green" if diff > 0 else ("crimson" if diff < 0 else "gray")
        trend_text = f"{trend_icon} {abs(diff):.2f} points difference"
    else:
        trend_text = "-"
        trend_color = "gray"
    
    col1, col2 = st.columns(2)
    
    box_style_total = """
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f5f5;
        text-align: center;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    """
    
    with col1:
        st.markdown(f"""
        <div style="{box_style_total}">
            <div style="font-size:18px; font-weight:bold;">현재 가게</div>
            <div style="font-size:36px; font-weight:bold; color:#2b8a3e;">{store_total:.2f}점</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="{box_style_total}">
            <div style="font-size:18px; font-weight:bold;">{region_name} 평균</div>
            <div style="font-size:36px; font-weight:bold; color:#1c7ed6;">{region_total:.2f}점</div>
            <div style="font-size:16px; color:{trend_color}; margin-top:5px;">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🔎 Keyword Sentiment Score Comparison")
    keyword_data = sentiment_data["keywords"]
    cols = st.columns(3)
    
    for idx, keyword in enumerate(KEYWORD_COLUMNS):
        with cols[idx % 3]:
            store_score = keyword_data.get(keyword)
            region_score = region_stats.get(keyword)
    
            box_style = """
                padding: 15px;
                border-radius: 10px;
                background-color: whitesmoke;
                text-align: center;
                box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
                min-height: 130px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            """
    
            if store_score is None:
                st.markdown(f"""
                    <div style="{box_style}">
                        <div style="font-size:18px; font-weight:bold">{KEYWORD_ENGLISH_MAP[keyword]}</div>
                        <div style="font-size:16px; color:gray; margin-top:12px;">Insufficient reviews for analysis</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                diff = store_score - region_score if region_score else 0
                trend = "▲" if diff > 0 else ("▼" if diff < 0 else "-")
                color = "green" if diff > 0 else ("crimson" if diff < 0 else "gray")
    
                st.markdown(f"""
                    <div style="{box_style}">
                        <div style="font-size:18px; font-weight:bold">{KEYWORD_ENGLISH_MAP[keyword]}</div>
                        <div style="font-size:28px; color:{color}">{store_score:.2f}Points {trend}</div>
                        <div style="font-size:14px; color:gray">Regional Average: {region_score:.2f}Points</div>
                    </div>
                """, unsafe_allow_html=True)



###############################################
# UI

# 사이드바
st.sidebar.image("DCX_Tool.png", use_container_width=True)
st.sidebar.title("Select Region and Store")

if 'location_locked' not in st.session_state:
    st.session_state['location_locked'] = False

if not st.session_state['location_locked']:
    location = st.sidebar.selectbox("Please select a region", [''] + list(DATASET_MAP.keys()), key="loc")
    if location:
        df = load_dataset(DATASET_MAP[location])
        stores = df['Name'].value_counts().index.tolist()
        store = st.sidebar.selectbox("Plase select a store", [''] + stores, key="store")
        if store and st.sidebar.button("✅Region/Store Selection Finalized"):
            st.session_state.update({
                'location_locked': True,
                'selected_location': location,
                'selected_store': store
            })
else:
    location = st.session_state.get('selected_location')
    store = st.session_state.get('selected_store')
    st.sidebar.markdown(f"🔒 Region: {location}\n\n🔒 Store: {store}")
    df = load_dataset(DATASET_MAP[location])

st.sidebar.markdown("""
## **This DCX analysis tool is only permitted for use in the following cases:**
* When used in educational settings such as universities for student education and research
* When used by small business owners for their own business purposes
* When used by university or graduate students as part of nonprofit community service activities to provide business strategies to local small business owners

<span style="color:red; font-weight:bold">
Except for the cases above, any commercial use of this analysis tool and reuse of the analysis data is strictly prohibited.
</span>
<br>
<br>
<br>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align:center; font-size:16px; font-weight:bold; margin-bottom:10px;">
📬 Inquiries & Information
</div>

<a href="mailto:peter@pusan.ac.kr">
    <button style="
        background-color:#f59f00;
        color:white;
        padding:8px 14px;
        border:none;
        border-radius:5px;
        font-size:14px;
        width:100%;
        margin-bottom:8px;
        cursor:pointer;">
        📧 Contact via Email
    </button>
</a>

<a href="https://ibalab.quv.kr/" target="_blank">
    <button style="
        background-color:#1c7ed6;
        color:white;
        padding:8px 14px;
        border:none;
        border-radius:5px;
        font-size:14px;
        width:100%;
        cursor:pointer;">
        🌐 IBA LAB Homepage
    </button>
</a>
""", unsafe_allow_html=True)


# 탭 설정
TABS = ["How to Use", "Photos & Reviews", "Word Cloud", "Treemap", "Network Analysis", "Topic Modeling", "Customer Satisfaction Analysis"]

if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = "How to Use"

# 색상 강제 적용: selectbox 라벨과 warning 텍스트
st.markdown("""
<style>
/* Fix the selectbox label text color */
label[for^=""] {
    color: black !important;
    font-weight: 600;
}

/* streamlit warning 박스 내부 텍스트 색상 강제 */
div[data-testid="stMarkdownContainer"] p {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.get("location_locked", False):
    selected_tab = st.selectbox("✅ Please select a feature", TABS)
    if st.session_state['current_tab'] != selected_tab:
        keys_to_clear = [
            key for key in st.session_state.keys()
            if key not in (
                'selected_location', 
                'selected_store', 
                'location_locked', 
                'user_id', 
                'queue_checked', 
                'start_time'
            )
        ]

        for k in keys_to_clear:
            del st.session_state[k]
        plt.clf()
        plt.close('all')
        gc.collect()
        st.session_state['current_tab'] = selected_tab
else:
    selected_tab = "How to Use"
    st.warning("⚠️ Please select the region and store first, then press 'Confirm' to activate the functions.")


# 탭별 기능 실행
if selected_tab == "How to Use":
    render_usage_tab()
elif selected_tab == "Photos & Reviews":
    render_review_tab(df, store)
elif selected_tab == "Word Cloud":
    render_wordcloud_tab(df, store)
elif selected_tab == "Treemap":
    render_treemap_tab(df, store)
elif selected_tab == "Network Analysis":
    render_network_tab(df, store)
elif selected_tab == "Topic Modeling":
    render_topic_tab(df, store)
elif selected_tab == "Customer Satisfaction Analysis":
    classifier = get_classifier()
    render_sentiment_dashboard(df, store, classifier)
