import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config("Buzzword Dilution Dashboard", layout="wide")
st.title("🚀 Detecting Buzzword Dilution")
st.subheader("An AI|NLP-Based Model to Measure Semantic Shift of Technical Terms Over Time")



# PLATFORM_URLS = {
#     "Twitter": "https://twitter.com/search?q={query}&src=typed_query",
#     "X": "https://twitter.com/search?q={query}&src=typed_query",
#     "LinkedIn": "https://www.linkedin.com/search/results/content/?keywords={query}",
#     "Medium": "https://medium.com/search?q={query}",
#     "Reddit": "https://www.reddit.com/search/?q={query}",
#     "YouTube": "https://www.youtube.com/results?search_query={query}",
#     "Blog": "https://www.google.com/search?q={query}+blog",
#     "News": "https://www.google.com/search?q={query}+news",
# }


PLATFORM_CONFIG = {
    "Twitter": {
        "icon": "🐦",
        "url": "https://twitter.com/search?q={query}&src=typed_query"
    },
    "X": {
        "icon": "❌",
        "url": "https://twitter.com/search?q={query}&src=typed_query"
    },
    "LinkedIn": {
        "icon": "💼",
        "url": "https://www.linkedin.com/search/results/content/?keywords={query}"
    },
    "Medium": {
        "icon": "✍️",
        "url": "https://medium.com/search?q={query}"
    },
    "Reddit": {
        "icon": "👽",
        "url": "https://www.reddit.com/search/?q={query}"
    },
    # "YouTube": {
    #     "icon": "📺",
    #     "url": "https://www.youtube.com/results?search_query={query}"
    # },
    "Blog": {
        "icon": "📝",
        "url": "https://www.google.com/search?q={query}+blog"
    },
    # "News": {
    #     "icon": "🗞️",
    #     "url": "https://www.google.com/search?q={query}+news"
    # },
}



@st.cache_resource
def prepare_official_document(file_path):
    from docx import Document
    import re

    doc = Document(file_path)
    text = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
    text = re.sub(r"\s+", " ", text)

    # chunk document
    sentences = text.split(".")
    chunks = [
        ". ".join(sentences[i:i+3]).strip()
        for i in range(0, len(sentences), 3)
        if len(sentences[i].strip()) > 0
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    #np.save("official_embeddings.npy", embeddings)

    return chunks, embeddings


# RIGHT: Looks in your current project folder
official_chunks, official_embeddings = prepare_official_document(
    "Buzzwords official document.docx"
)





@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_sbert()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("buzzword_dilution_dataset.csv", encoding="latin1")

    dataset_similarities = []

    for text in df["text"]:
        text_emb = model.encode([text])

        similarities = cosine_similarity(
            text_emb, official_embeddings
        )[0]

        dataset_similarities.append(similarities.max())

    df["dataset similarity"] = dataset_similarities
    df["dilution"] = 1 - df["dataset similarity"]
    return df

@st.cache_resource
def load_embeddings():
    return np.load("Text_Embeddings.npy")

df = load_data()
embeddings = load_embeddings()

# Sidebar filters
st.sidebar.title("Controls")

year = st.sidebar.slider(
    "Select Year",
    int(df.year.min()),
    int(df.year.max()),
    2023
)

buzzword = st.sidebar.selectbox(
    "Select Buzzword",
    sorted(df.buzzword.unique())
)

# available_platforms = (
#     df[df["buzzword"] == buzzword]["platform"]
#     .dropna()
#     .unique()
#     .tolist()
# )


platform_counts = (
    df[df["buzzword"] == buzzword]
    .groupby("platform")
    .size()
    .to_dict()
)

platform_counts = (
    df[(df["buzzword"] == buzzword) & (df["year"] == year)]
    .groupby("platform")
    .size()
    .to_dict()
)



# st.sidebar.markdown("### 🔗 Explore Buzzword on Platforms")

# for platform in sorted(available_platforms):
#     if platform in PLATFORM_URLS:
#         url = PLATFORM_URLS[platform].format(
#             query=buzzword.replace(" ", "%20")
#         )

#         st.sidebar.link_button(
#             label=f"Open {platform}",
#             url=url,
#             use_container_width=True
#         )


st.sidebar.markdown("### 🔗 Explore Buzzword on Platforms")

for platform, count in sorted(
    platform_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    if platform in PLATFORM_CONFIG:
        config = PLATFORM_CONFIG[platform]

        url = config["url"].format(
            query=buzzword.replace(" ", "%20")
        )

        label = f"{config['icon']} {platform}"

        st.sidebar.link_button(
            label=label,
            url=url,
            use_container_width=True
        )




filtered_df = df[(df.year == year) & (df.buzzword == buzzword)]

def center_df(df):
    return df.style.set_properties(**{
        "text-align": "center"
    }).set_table_styles([
        dict(selector="th", props=[("text-align", "center")])
    ])



col1, col2, col3 = st.columns(3)

col1.metric(
    "Avg Dilution (Hype)",
    f"{df[df.label=='Hype']['dilution'].mean():.3f}"
)

col2.metric(
    "Avg Dilution (Technical)",
    f"{df[df.label=='Technical']['dilution'].mean():.3f}"
)

col3.metric(
    "Semantic Gap",
    f"{(df[df.label=='Hype']['dilution'].mean() - df[df.label=='Technical']['dilution'].mean()):.3f}",
    help="Difference between hype and technical dilution"
)

st.markdown("1️⃣ Avg Dilution (Hype) = 0.462 " \
"Hype-related texts are, on average, 46.2% semantically distant from the official definition. " \
"This indicates loose, metaphorical, or marketing-driven usage of technical terms. " \
"In practical terms: hype texts reuse the buzzword without fully preserving its core technical meaning. " \
"When people use buzzwords in a hype or marketing way, their meaning shifts significantly away from the official definition. " \
"This means the buzzword is often used loosely or vaguely, not in its original technical sense.")

st.markdown("2️⃣ Avg Dilution (Technical) = 0.442 " \
"Even technical texts are 44.2% distant from the official definition. " \
"This reflects domain specialization (not all technical contexts repeat the official wording) and partial abstraction or simplification. " \
"Importantly, technical usage is still closer to the official definition than hype usage. " \
"Even in technical writing, the wording is not exactly the same as the official definition, " \
"but it stays closer to the real meaning than hype usage.")

st.markdown("3️⃣ Semantic Gap = 0.020 " \
"This is the average excess dilution introduced by hype compared to technical usage. " \
"A 2% gap may look small numerically, but in embedding space, this is meaningful. " \
"SBERT similarities are already normalized, and even small percentage shifts correspond to measurable semantic drift. " \
"Hype usage pushes the meaning slightly farther away from the official definition compared to technical usage. " \
"This difference may look small, but it is consistent and significant.")
st.subheader("📉 Mean Dilution per Year (Hype vs Technical)")

year_label = (
    df.groupby(["year", "label"])[["dilution", "dataset similarity"]]
    .mean()
    .reset_index()
)

st.dataframe(center_df(year_label), use_container_width=True)








st.subheader("🧠 Buzzword Dilution by Label (Selected Year)")

buzzword_label_dilution = (
    df[df["year"] == year]
    .groupby(["buzzword", "label"])["dilution"]
    .mean()
    .reset_index()
)

fig = px.bar(
    buzzword_label_dilution,
    x="dilution",
    y="buzzword",
    color="label",
    orientation="h",
    barmode="group",
    title=f"Hype vs Technical Dilution in {year}",
)

st.plotly_chart(fig, use_container_width=True)








# st.subheader("📈 Temporal Trend: Mean Dilution per Year")

# yearly = (
#     df.groupby("year")["dilution"]
#     .mean()
#     .reset_index()
# )

# fig, ax = plt.subplots(figsize=(3.2, 2.6))
# ax.plot(yearly["year"], yearly["dilution"], marker="o", linewidth=1)
# ax.set_xlabel("Year", fontsize=8)
# ax.set_ylabel("Mean Dilution Score", fontsize=8)
# ax.set_title("Buzzword Dilution Over Time", fontsize=10)
# ax.tick_params(axis="both", labelsize=7)
# ax.grid(alpha=0.3)

# st.pyplot(fig, use_container_width=False)

# st.subheader("🏷️ Mean Dilution per Buzzword")

# buzzword_dilution = (
#     df.groupby("buzzword")["dilution"]
#     .mean()
#     .reset_index()
#     .sort_values("dilution", ascending=False)
# )

# fig2, ax2 = plt.subplots(figsize=(3.5, 3))
# ax2.barh(
#     buzzword_dilution["buzzword"],
#     buzzword_dilution["dilution"]
# )
# ax2.set_xlabel("Mean Dilution Score", fontsize=8)
# ax2.set_title("Buzzword-wise Semantic Dilution", fontsize=9)
# ax2.tick_params(axis="y", labelsize=7)
# ax2.invert_yaxis()

# st.pyplot(fig2, use_container_width=False)



# st.subheader("🧠 Interpretation & Insights")


st.subheader("📊 Buzzword Frequency for Selected Year")

# Aggregate buzzword counts for selected year
buzzword_year_count = (
    df[df["year"] == year]
    .groupby("buzzword")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)




buzzword_year_label = (
    df[df["year"] == year]
    .groupby(["buzzword", "label"])
    .size()
    .reset_index(name="count")
)

fig = px.bar(
    buzzword_year_label,
    x="buzzword",
    y="count",
    color="label",
    barmode="stack",
    title=f"Buzzword Usage by Label in {year}",
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("""
**Key Findings:**
- Hype-labeled texts consistently show higher dilution than technical texts.
- The semantic gap remains stable across years (~5–10%).
- Temporal trends are non-monotonic, indicating cyclical hype patterns.
- Classical correlation tests fail because dilution is semantic, not linear.

**Conclusion:**
Embedding-based similarity provides stronger evidence of buzzword dilution
than traditional statistical association tests.
""")



# Title
st.title("🚀 Buzzword Dilution Analysis (2020–2025)")


# Year animation (Plotly)
st.subheader("📈 Year-wise Buzzword Trend")

trend = (
    df.groupby(["year", "label"])
    .size()
    .reset_index(name="count")
)

fig = px.line(
    trend,
    x="year",
    y="count",
    color="label",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# Platform vs Label
st.subheader("🏷️ Platform vs Label")

platform_dist = (
    filtered_df.groupby(["platform", "label"])
    .size()
    .reset_index(name="count")
)

fig2 = px.bar(
    platform_dist,
    x="platform",
    y="count",
    color="label",
    barmode="stack",   # 🔹 stacked bars
)

fig2.update_traces(
    hovertemplate="Platform=%{x}<br>Label=%{legendgroup}<br>Count=%{y}<extra></extra>"
)

fig2.update_layout(
    yaxis_title="Count",
    xaxis_title="Platform",
    yaxis=dict(
        tickmode="linear",  # 🔹 forces integer stepping
        tick0=0,
        dtick=1
    )
)

st.plotly_chart(fig2, use_container_width=True)

fig2.update_traces(textposition="inside")
platform_order = (
    platform_dist.groupby("platform")["count"]
    .sum()
    .sort_values(ascending=False)
    .index
)

fig2.update_layout(xaxis=dict(categoryorder="array", categoryarray=platform_order))


# Semantic similarity section
st.subheader("🧠 Semantic Similarity (SBERT)")

user_similarity_score = None
best_match_text = None

st.subheader("🔍 User Text vs Official Documentation")
user_text = st.text_area(
    "✍️ Enter text to check similarity with official definition",height=150
)

if user_text.strip():
        user_emb = model.encode([user_text])

if st.button("Evaluate Semantic Similarity"):
    

    similaritys = cosine_similarity(user_emb, official_embeddings)[0]

    best_idx = similaritys.argmax()
    user_similarity_score = similaritys[best_idx] * 100
    best_match_text = official_chunks[best_idx]

st.caption(" Semantic Similarity | Hype vs Technical | SBERT")

c1,c2,c3 = st.columns(3)

if user_similarity_score is not None:
    c1.metric(
        "User Text → Official Similarity",
        f"{user_similarity_score:.2f} %",
        help="Semantic similarity between user-entered text and official definition"
    )
else:
    c1.metric("User Text → Official Similarity", "Enter text")

c2.metric("Hype %", round((filtered_df.label == "Hype").mean() * 100, 2))
c3.metric("Technical %", round((filtered_df.label == "Technical").mean() * 100, 2))

if best_match_text:
    st.subheader("📌 Closest Matching Official Paragraph")
    st.info(best_match_text)


if st.button("Find Similar"):
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # user_embedding = model.encode([user_text])

    user_similarities = cosine_similarity(user_emb, embeddings)[0]
    df["User similarity"] = user_similarities

    top_matches = df.sort_values("User similarity", ascending=False).head(5)
    
    st.write("### 🔍 Most Similar Texts")
    st.dataframe(
        top_matches[["text", "buzzword", "year", "platform", "label", "User similarity"]]
    )


st.subheader("📊 User Similarity Distribution")

if "User similarity" in df.columns:

    fig = px.histogram(
        df,
        x="User similarity",
        nbins=30,
        title="Distribution of User Text Similarity Across Dataset",
        opacity=0.75
    )

    fig.update_layout(
        xaxis_title="User Similarity Score",
        yaxis_title="Number of Texts",
        bargap=0.1
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter text and click **Find Similar** to see similarity distribution.")

st.markdown("🧠 What this similarity distribution shows")
st.markdown("Each bar shows how many texts in the dataset have a similarity score in that range.")
st.markdown("The similarity score tells how close the meaning of a text is to the user-entered text.")
st.markdown("Bars on the left mean texts that are less similar.")
st.markdown("Bars on the right mean texts that are more similar.")

# Data preview
st.subheader("📄 Filtered Data Preview")
st.dataframe(filtered_df.head(10))
