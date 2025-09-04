# steam_dashboard.py ───────────────────────────────────────────
# Streamlit dashboard for FronkonGames / Steam dataset
# Hanna 
# 1. Place this file next to steam_games.csv
# 2. make sure virtual enviornment is set up python -m venv venv
# 3. pip install -U streamlit pandas plotly numpy scikit-learn nltk wordcloud matplotlib gdown re os pathlin typing
# 4. to run write in terminal: streamlit run steamdb.py
# ──────────────────────────────────────────────────────────────
from pathlib import Path
import re, string
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import nltk
import re
import os
import gdown
from pathlib import Path
from typing import Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from typing import Union

# Hanna
# Cached helpers
@st.cache_resource(show_spinner="Training NLP model…")
def build_word_effects(df: pd.DataFrame, target_q=0.80):
    try:
        _ = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    stops = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation)

    mask = df["positive_ratio"].notna() & df["about_the_game"].notna()
    texts = df.loc[mask, "about_the_game"].str.lower().tolist()
    y = (df.loc[mask, "positive_ratio"] >= target_q).astype(int)

    tfidf = TfidfVectorizer(
        stop_words=list(stops), min_df=30, max_df=0.9, ngram_range=(1, 2)
    )
    X = tfidf.fit_transform(texts)
    model = LogisticRegression(max_iter=3000, n_jobs=-1).fit(X, y)
    coef = model.coef_[0]

    top = 15
    pos_idx = np.argsort(coef)[-top:][::-1]
    neg_idx = np.argsort(coef)[:top]

    terms = np.hstack(
        [tfidf.get_feature_names_out()[pos_idx], tfidf.get_feature_names_out()[neg_idx]]
    )
    weights = np.hstack([coef[pos_idx], coef[neg_idx]])
    return pd.DataFrame({"term": terms, "coef": weights})


@st.cache_resource(show_spinner="Building word‑cloud…")
def build_wordcloud(word_df: pd.DataFrame, top_n=120):
    pos_terms = word_df[word_df["coef"] > 0].nlargest(top_n, "coef")
    freqs = dict(zip(pos_terms["term"], pos_terms["coef"]))
    return WordCloud(
        width=800,
        height=400,
        background_color="#171a21",
        colormap="Blues",
        prefer_horizontal=0.9,
    ).generate_from_frequencies(freqs)


@st.cache_resource(show_spinner="Training success model…")
def build_success_model(df: pd.DataFrame, threshold=0.80):
    mask = df["positive_ratio"].notna()
    y = (df.loc[mask, "positive_ratio"] >= threshold).astype(int)

    num_cols = [
        c
        for c in [
            "price",
            "average_playtime_forever",
            "average_playtime_two_weeks",
            "dlc_count",
            "required_age",
        ]
        if c in df.columns
    ]
    X_num = df.loc[mask, num_cols].fillna(0)

    if "genres" in df.columns:
        mlb = MultiLabelBinarizer()
        hot = mlb.fit_transform(df.loc[mask, "genres"])
        genre_df = pd.DataFrame(
            hot,
            columns=[f"genre_{g}" for g in mlb.classes_],
            index=X_num.index,
        )
        X = pd.concat([X_num, genre_df], axis=1)
    else:
        X = X_num

    clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"
    ).fit(X, y)

    acc = accuracy_score(y, clf.predict(X))
    imp_df = (
        pd.Series(clf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
    )
    return imp_df, acc, clf, X.columns

# Hanna
# Recommendation tab with TF-IDF based content filtering
def run_recommender_tab(df):
    st.header("Game Recommender 🎯")

    if "about_the_game" not in df.columns or "positive_ratio" not in df.columns:
        st.warning("Missing required data for recommendations.")
        return

    # Only keep games with descriptions
    df = df[df["about_the_game"].notna()].copy()

    # Price filter
    if "price" in df.columns:
        price_min, price_max = float(df["price"].min()), float(df["price"].max())
        selected_price_range = st.slider("Filter by price", price_min, price_max, (price_min, price_max))
        df = df[df["price"].between(*selected_price_range)]

    # Limit to top 2000 games with highest review counts to save memory
    if {"positive", "negative"}.issubset(df.columns):
        df["total_reviews"] = df["positive"] + df["negative"]
        df = df.sort_values("total_reviews", ascending=False).head(2000)
    else:
        df = df.sample(n=2000, random_state=42)

    df["about_clean"] = df["about_the_game"].str.lower().fillna("")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["about_clean"])

    game_titles = df["name"].values
    title_to_idx = {title: i for i, title in enumerate(game_titles)}

    selected_title = st.selectbox("Choose a game you like", game_titles)
    if not selected_title:
        return

    idx = title_to_idx[selected_title]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    rec_indices = [i for i, _ in sim_scores]
    rec_games = df.iloc[rec_indices]

    # Sort options
    sort_option = st.radio("Sort recommendations by:", ["Similarity", "Success likelihood"])
    if sort_option == "Success likelihood" and "positive_ratio" in rec_games.columns:
        rec_games = rec_games.sort_values("positive_ratio", ascending=False)
    else:
        # rec_games already selected via iloc
        pass  # keep original similarity order

    st.subheader("You may also like:")
    for _, row in rec_games.iterrows():
        st.markdown(f"""
        **🎮 {row['name']}**
        - 💰 Price: ${row['price']:.2f}
        - 👍 Positive %: {row['positive_ratio']:.1%}
        - ⏱ Avg Playtime: {row['average_playtime_forever']:.1f} hrs
        - 📝 Description: {row['about_the_game'][:200]}...
        """)

    # Multi violin plot showing success distribution
    st.subheader("Success Distribution by Genre (Recommended Games)")

    if "genres" in rec_games.columns and rec_games["genres"].notna().any():
        rec_expanded = rec_games.explode("genres").dropna(subset=["genres", "positive_ratio"])
        top_genres = rec_expanded["genres"].value_counts().head(5).index  # Limit to top 5 for clarity
        filtered = rec_expanded[rec_expanded["genres"].isin(top_genres)]

        if not filtered.empty:
            fig_violin = px.violin(
                filtered,
                y="positive_ratio",
                x="genres",
                color="genres",
                box=True,
                points="all",
                template="plotly_dark",
                labels={"positive_ratio": "Positive Review %", "genres": "Genre"},
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            fig_violin.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis_tickformat=".0%",
                title="Distribution of Review Positivity by Genre (Top 5)",
                showlegend=False
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Not enough genre data to generate multi-violin plot.")
    else:
        st.info("Genre information is unavailable for the recommended games.")


# Hanna
# 1. Page setup
st.set_page_config("Steam Games Dashboard", "🎮", layout="wide")
STEAM_BG, STEAM_PANEL, STEAM_PRIMARY, STEAM_TEXT = (
    "#171a21",
    "#1B2838",
    "#66C0F4",
    "#C7D5E0",
)
st.markdown(
    f"""
<style>
.stApp {{background-color:{STEAM_BG}; color:{STEAM_TEXT};}}
.stTabs [role="tab"] {{background-color:{STEAM_PANEL};}}
.stTabs [aria-selected="true"] {{border-bottom:2px solid {STEAM_PRIMARY};}}
div[data-testid="stMetricValue"] {{color:{STEAM_PRIMARY};}}
div[data-testid="metric-container"] > label {{color:{STEAM_TEXT};}}
</style>
""",
    unsafe_allow_html=True,
)

# 2 Load & clean data
@st.cache_data(show_spinner="Loading CSV…", hash_funcs={Path: str})
def load_data(path: Union[str, Path] = "steam_games.csv") -> pd.DataFrame:
# https://drive.google.com/file/d/1ZHXAGndVyWHv5ldKk49J775c5pLac4a-/view?usp=drive_link
    file_id = "1ZHXAGndVyWHv5ldKk49J775c5pLac4a-"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download file if not already present
    if not os.path.exists(path):
        gdown.download(url, str(path), quiet=False, fuzzy=True)

    # Read CSV
    df = pd.read_csv(path)

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Convert release_date → datetime + extract year
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year

    # Convert numeric fields
    for col in [
        "price",
        "positive",
        "negative",
        "average_playtime_forever",
        "average_playtime_two_weeks",
        "dlc_count",
        "required_age",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate positive review ratio
    if {"positive", "negative"}.issubset(df.columns):
        denom = df["positive"] + df["negative"]
        df["positive_ratio"] = np.where(denom > 0, df["positive"] / denom, np.nan)

    # Parse genres
    if "genres" in df.columns:
        df["genres"] = (
            df["genres"]
            .fillna("")
            .apply(lambda s: tuple(g.strip() for g in re.split(r"[;,]", str(s)) if g.strip()))
        )

    return df

df = load_data()
st.write("✅ Dataset loaded:", df.shape)

# @st.cache_data(show_spinner="Loading CSV…")
# @st.cache_data(show_spinner="Loading CSV…")
# def load_data(path: Union[str, Path] = "steam_games.csv") -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df.columns = (
#         df.columns.str.strip()
#         .str.lower()
#         .str.replace(" ", "_")
#         .str.replace("-", "_")
#     )

#     if "release_date" in df.columns:
#         df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
#         df["year"] = df["release_date"].dt.year

#     for col in [
#         "price",
#         "positive",
#         "negative",
#         "average_playtime_forever",
#         "average_playtime_two_weeks",
#         "dlc_count",
#         "required_age",
#     ]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     if {"positive", "negative"}.issubset(df.columns):
#         denom = df["positive"] + df["negative"]
#         df["positive_ratio"] = np.where(denom > 0, df["positive"] / denom, np.nan)

#     if "genres" in df.columns:
#         df["genres"] = (
#             df["genres"]
#             .fillna("")
#             .apply(lambda s: tuple(g.strip() for g in re.split(r"[;,]", str(s)) if g.strip()))
#         )

#     return df


# df = load_data()

# Hanna
# 3. Sidebar filters
with st.sidebar:
    st.header("Filters")

    if "year" in df.columns and df["year"].notna().any():
        yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("Release year", yr_min, yr_max, (2015, yr_max))
    else:
        year_range = (None, None)

    platform_cols = [c for c in ["windows", "mac", "linux"] if c in df.columns]
    plat_sel = st.multiselect("Platforms", platform_cols, default=platform_cols)

    # genre_catalog = sorted({g for row in df.get("genres", []) for g in row})
    genre_catalog = sorted({g for row in df["genres"].dropna() if isinstance(row, list) for g in row})
    genre_sel = st.multiselect("Genres", genre_catalog)

# apply filters
view = df.copy()
if year_range != (None, None) and "year" in view.columns:
    view = view[view["year"].between(*year_range)]
if plat_sel:
    view = view[view[plat_sel].any(axis=1)]
if genre_sel and "genres" in view.columns:
    view = view[view["genres"].apply(lambda tags: any(g in tags for g in genre_sel))]

# 3. KPI
st.subheader("Key metrics")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Games", f"{len(view):,}")
if "price" in view.columns:
    k2.metric("Median price", f"${view['price'].median():.2f}")
if "average_playtime_forever" in view.columns:
    k3.metric("Avg play‑time", f"{view['average_playtime_forever'].mean():.1f} hrs")
if "positive_ratio" in view.columns:
    k4.metric("Avg positive %", f"{view['positive_ratio'].mean():.1%}")

# 4. Tabs
tabs = st.tabs(
    [
        "Data Analysis & Exploration 🔬",
        "Success Factors ⭐",
        "Game Recommendations 🤖",
        "Advanced Insights & Forecasting 🔮",
    ]
)

# Data Analysis & Exploration
with tabs[0]:
    st.header("Data Analysis & Exploration 🔬")

    st.subheader("Timeline 📈")
    if "year" not in view.columns:
        st.info("Dataset lacks release dates.")
    else:
        per_year = (
            view.groupby("year", dropna=True)
            .size()
            .reset_index(name="count")
            .sort_values("year")
        )
        fig = px.area(
            per_year,
            x="year",
            y="count",
            labels={"count": "Games released"},
            template="plotly_dark",
            color_discrete_sequence=[STEAM_PRIMARY],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Genres 🎭")
    if "genres" not in view.columns:
        st.info("No genre column present.")
    else:
        flat = view.explode("genres").dropna(subset=["genres"])
        top = (
            flat["genres"].value_counts().head(5).reset_index(name="count").rename(columns={"index": "genres"})
        )
        fig = px.bar(
            top.sort_values("count"),
            x="count",
            y="genres",
            orientation="h",
            template="plotly_dark",
            labels={"count": "Games", "genres": "Genres"},
            color_discrete_sequence=[STEAM_PRIMARY],
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Games: %{x}<extra></extra>")
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price ↔ Reviews 💲")
    if {"price", "positive_ratio"}.issubset(view.columns):
        st.subheader("Price vs. Positive reviews")
        c1, c2 = st.columns(2)
        pct_cap = c1.slider("Drop prices above percentile", 90, 100, 99)
        log_axis = c2.checkbox("Log‑scale X‑axis", value=False)

        price_cap = np.nanpercentile(view["price"], pct_cap)
        trimmed = view[view["price"] <= price_cap]

        scatter = px.scatter(
            trimmed,
            x="price",
            y="positive_ratio",
            labels={"positive_ratio": "Positive %"},
            template="plotly_dark",
            color_discrete_sequence=[STEAM_PRIMARY],
            hover_data=["name", "year"] if "year" in trimmed.columns else ["name"],
            log_x=log_axis,
        )
        scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(scatter, use_container_width=True)
        st.caption(
            f"Showing {len(trimmed):,} games priced ≤ ${price_cap:.2f} "
            f"({pct_cap}th percentile)."
        )
    else:
        st.info("Need price & review columns.")

    st.subheader("Platform Share 💻")
    if platform_cols:
        plat_counts = (
            view[platform_cols]
            .melt(var_name="platform", value_name="has")
            .groupby("platform")["has"]
            .sum()
            .reset_index(name="count")
        )
        fig = px.pie(
            plat_counts,
            names="platform",
            values="count",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No platform info.")

    st.subheader("NLP Insights 🔤")
    if "about_the_game" not in df.columns:
        st.info("No description text.")
    else:
        word_df = build_word_effects(df)
        st.caption("Top words & bigrams correlated with ≥80 % positive reviews")

        fig = px.bar(
            word_df.sort_values("coef"),
            x="coef",
            y="term",
            orientation="h",
            template="plotly_dark",
            labels={"coef": "Model weight", "term": "Terms"},
            color="coef",
            color_continuous_scale=["#D33636", STEAM_PRIMARY],
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Positive‑term word cloud"):
            wc = build_wordcloud(word_df)
            fig_wc, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)


    st.subheader("Exploration 🔍")
    st.caption("Quick multivariate explorations")

    if {"price", "average_playtime_forever", "positive_ratio"}.issubset(view.columns):
        fig3d = px.scatter_3d(
            view,
            x="price",
            y="average_playtime_forever",
            z="positive_ratio",
            color_discrete_sequence=[STEAM_PRIMARY],
            hover_data=["name"],
            opacity=0.6,
            template="plotly_dark",
        )
        fig3d.update_layout(margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(fig3d, use_container_width=True)

    numeric = [c for c in ["price", "positive_ratio", "average_playtime_forever", "dlc_count"] if c in view.columns]
    if len(numeric) >= 2:
        mat = px.scatter_matrix(
            view[numeric].dropna(),
            dimensions=numeric,
            template="plotly_dark",
            color_discrete_sequence=[STEAM_PRIMARY],
        )
        mat.update_layout(margin=dict(l=5, r=5, t=5, b=5))
        st.plotly_chart(mat, use_container_width=True)

# Success Factors
with tabs[1]:
    if "positive_ratio" not in df.columns:
        st.info("Need review data.")
    else:
        st.caption("Features that best predict landing in the top‑20 % for positive reviews")
        imp_df, acc, clf, feat_order = build_success_model(df)
        st.metric("In‑sample accuracy", f"{acc:.2%}")

        fig = px.bar(
            imp_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_dark",
            labels={"importance": "Gini importance", "feature": "Feature"},
            color_discrete_sequence=[STEAM_PRIMARY],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("What‑if simulator"):
            top6 = imp_df["feature"].head(6).tolist()
            user_vals = {}
            for feat in top6:
                if feat.startswith("genre_"):
                    user_vals[feat] = st.checkbox(feat.replace("genre_", ""), False)
                else:
                    lo, hi = df[feat].quantile([0.05, 0.95]).astype(float)
                    user_vals[feat] = st.slider(feat, float(lo), float(hi), float(lo))

            X_one = pd.DataFrame({c: [0.0] for c in feat_order})
            for k, v in user_vals.items():
                X_one.at[0, k] = float(v)
            prob = clf.predict_proba(X_one)[0, 1]
            st.metric("Predicted success %", f"{prob:.1%}")

# Hanna
# Recommender
with tabs[2]: 
    run_recommender_tab(df) 

with tabs[3]:
    st.header("Advanced Insights & Forecasting 🔮")

    st.subheader("Model Evaluation: Success Predictor")

    @st.cache_resource(show_spinner="Training models...")
    def evaluate_models(df: pd.DataFrame, threshold=0.80):
        from sklearn.metrics import roc_curve, roc_auc_score

        mask = df["positive_ratio"].notna()
        y = (df.loc[mask, "positive_ratio"] >= threshold).astype(int)

        num_cols = [
            "price", "average_playtime_forever", "average_playtime_two_weeks",
            "dlc_count", "required_age"
        ]
        num_cols = [c for c in num_cols if c in df.columns]
        X_num = df.loc[mask, num_cols].fillna(0)

        if "genres" in df.columns:
            mlb = MultiLabelBinarizer()
            hot = mlb.fit_transform(df.loc[mask, "genres"])
            genre_df = pd.DataFrame(
                hot,
                columns=[f"genre_{g}" for g in mlb.classes_],
                index=X_num.index,
            )
            X = pd.concat([X_num, genre_df], axis=1)
        else:
            X = X_num

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced"),
            "Logistic Regression": LogisticRegression(max_iter=3000, n_jobs=-1)
        }

        results = []
        for name, model in models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_proba)
            fpr, tpr, _ = roc_curve(y, y_proba)
            report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True, zero_division=0)).transpose()
            results.append((name, acc, auc, fpr, tpr, report_df))

        return results

    results = evaluate_models(df)
    model_names = [r[0] for r in results]
    selected_model = st.selectbox("Select a model to view details", model_names)

    selected_result = next(r for r in results if r[0] == selected_model)
    name, acc, auc, fpr, tpr, report_df = selected_result

    # Show selected model details
    st.markdown(f"### {name}")
    st.metric("Accuracy", f"{acc:.2%}")
    st.metric("AUC", f"{auc:.3f}")
    st.dataframe(report_df.round(2).style.highlight_max(axis=1), use_container_width=True)

    if st.checkbox("Show raw classification report"):
        st.text(classification_report(
            (df["positive_ratio"].fillna(0) >= 0.80).astype(int),
            df["positive_ratio"].fillna(0).apply(lambda x: 1 if x >= 0.80 else 0),
            target_names=["Not Successful", "Successful"],
            zero_division=0,
        ))

    # ROC Curve Comparison (all models)
    st.subheader("ROC Curve Comparison")
    fig = go.Figure()
    for name, _, auc, fpr, tpr, _ in results:
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash")))
    fig.update_layout(
        template="plotly_dark",
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend Forecasting of Genre Popularity 📈")
    if {"genres", "year"}.issubset(df.columns):
        forecast_df = df.explode("genres").dropna(subset=["year", "genres"])
        genre_counts = forecast_df.groupby(["year", "genres"]).size().reset_index(name="count")
        genre_selected = st.selectbox("Select genre to forecast", genre_counts["genres"].unique())

        genre_trend = genre_counts[genre_counts["genres"] == genre_selected]

        fig = px.line(
            genre_trend,
            x="year",
            y="count",
            title=f"Forecasting Trend for {genre_selected}",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Game Lifecycle Funnel ⏳")
    if "release_date" in df.columns and "average_playtime_forever" in df.columns:
        df_life = df.copy()
        df_life["year"] = df_life["release_date"].dt.year
        lifecycle = df_life.groupby("year")["average_playtime_forever"].mean().reset_index()

        fig = px.line(
                lifecycle,
                x="year",  
                y="average_playtime_forever",  
                title="Average Playtime by Release Year",
                markers=True,
                template="plotly_dark"
)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧭 Genre Innovation Map")
    if "genres" in df.columns:
        genre_map = df.explode("genres").dropna(subset=["genres"])
        summary = genre_map.groupby("genres").agg({
            "positive_ratio": "mean",
            "price": "mean",
            "average_playtime_forever": "mean"
        }).reset_index()

        fig = px.scatter(summary, 
                         x="price", y="positive_ratio", size="average_playtime_forever", 
                         hover_name="genres", template="plotly_dark",
                         labels={"positive_ratio": "Avg Positive %", "price": "Avg Price"},
                         title="Genre Innovation Map")
        st.plotly_chart(fig, use_container_width=True)



