import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# title for web app
st.title('Tv Show Similarity Engine')

# read in data
tv_dataset = pd.read_csv("tv_dataset_final.csv")

# insert some graphics for visualization

st.header("Have you ever finished a great TV show and then wondered... what next?")
st.subheader("If you want a similar show to your past favorites, you could check any streaming service")
st.subheader("However, these services will only recommend their own shows")
st.subheader("This web app uses 1000+ of the most popular shows to recommend the most similar options using a universal sentence encoder and a number of other features including genres and ratings")
st.subheader("Simply select a show you like and look at the suggested options!")



# arr = tv_dataset["startYear"]
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)

# arr = tv_dataset["averageRating"]
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)

# fig, ax = plt.subplots()
# ax.scatter(tv_dataset["startYear"],tv_dataset["averageRating"])

# st.pyplot(fig)






# filtering to key shows
tv_dataset = tv_dataset[(tv_dataset["numVotes"] > 7500)  & (tv_dataset["startYear"] > 1970)]
tv_dataset.sort_values("primaryTitle", inplace=True)
tv_dataset.reset_index(drop=True, inplace=True)




# drop uneeded cols
tv_dataset.drop(["Unnamed: 0", "originalTitle", "endYear", "genres", "directors", "writers", "isOriginalTitle", "languages"], axis=1, inplace=True)
values = {'runtimeMinutes': np.mean(tv_dataset["runtimeMinutes"]),
          'storylines': "",
          'keywords': ""}
# fill na values in each col with appropriate vals
tv_dataset.fillna(value=values, inplace=True)

@st.cache
def load_module():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)
    return model

model = load_module()

@st.cache
def embed(input):
    return model(input)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_similarity(v1, v2):
    return 1 - abs(angle_between(v1,v2))


def similarity(index1, index2, df, verbose=False):
#     directors1 = set(test.iloc[index1, :])
#     directors2 = set(test.iloc[index2, :])
    if verbose:
        print(df.loc[index1, "tconst"], df.loc[index2, "tconst"])
#     print(directors1)
#     print(directors2)
    # -1 to sccount for the None overlsp
#     shared_directors = len(directors1 & directors2) - 1
    angle = angle_similarity(df.iloc[index1, 2:-2], df.iloc[index2, 2:-2])
    
    storyline1, storyline2 = df.loc[index1, "storylines"], df.loc[index2, "storylines"]
    keyword1, keyword2 = df.loc[index1, "keywords"], df.loc[index2, "keywords"]

#     storyline_sim = np.inner(embed([storyline1]), embed([storyline2]))
#     keyword_sim = np.inner(embed([keyword1]), embed([keyword2]))
    storyline_sim = angle_similarity(embed([storyline1])[0], embed([storyline2])[0])
    keyword_sim = angle_similarity(embed([keyword1])[0], embed([keyword2])[0])
#     similarity_score = angle + shared_directors

    # v1 has no keyword weighting
    similarity_score = angle + .5*storyline_sim + .5*keyword_sim

    if verbose:
        print("Storyline Similarity:", storyline_sim)
        print("Keywords Similarity:", keyword_sim)

        print("similarity =", similarity_score)
    return similarity_score

@st.cache
def most_similar(index, df, verbose=False, amount=5):
    similarities = [similarity(index, j, df) for j in list(df.index)]
    df_sims = pd.DataFrame({"indices":range(0, len(similarities)), "similarities":similarities})
    df_sims.sort_values(by="similarities", ascending=False, inplace=True)
        
    top = df_sims.iloc[:amount,:].copy()
    title = [df.loc[index, "primaryTitle"] for index in list(top["indices"])]
    tconst = [df.loc[index, "tconst"] for index in list(top["indices"])]
    top["title"] = title
    top["tconst"] = tconst
    return df["primaryTitle"][df_sims.iloc[1:16,0]]


def get_index(title):
    return tv_dataset[tv_dataset["primaryTitle"] == title].index[0]

num_shows_display = st.slider('How many similar shows would you like to see?', 1, 15, 3)


option = st.selectbox(
        'Select Show',
        tv_dataset["primaryTitle"])




top_results = most_similar(get_index(option), tv_dataset, True)


st.subheader("The most similar shows to " + option + str(" are:"))
for i in range(1, num_shows_display+1):
    st.text(top_results.iloc[i])





