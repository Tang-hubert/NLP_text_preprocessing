# Text mining

# Dataset loading
import nltk
import pandas

path = r"C:\\Users\\luke1\\Downloads\\textmining\\Amazon book reviews"

# Books_rating.csv
df_rating = pandas.read_csv(f"{path}\\Books_rating.csv")

print(df_rating)

# Select target data for dataframe
df_rating = (df_rating[df_rating['Price']>0])

len(df_rating.index)

df_rating = df_rating.reset_index()

for col in df_rating.columns:
    print(col)

df_rating_review_text = df_rating['review/text'][0]

df_rating_review_text

# Sentence Segmentation
sentences = nltk.sent_tokenize(df_rating_review_text)

# Word Segmentation
tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]
print(tokens)
for token in tokens:
    print(token)

# POS (Part-of-Speech Tagging)
pos = [nltk.pos_tag(token) for token in tokens]
for item in pos:
    print(item)

# Lemmatization (Word Form Restoration)
wordnet_pos = []
for p in pos:
    for word, tag in p:
        if tag.startswith('J'):
            wordnet_pos.append(nltk.corpus.wordnet.ADJ)
        elif tag.startswith('V'):
            wordnet_pos.append(nltk.corpus.wordnet.VERB)
        elif tag.startswith('N'):
            wordnet_pos.append(nltk.corpus.wordnet.NOUN)
        elif tag.startswith('R'):
            wordnet_pos.append(nltk.corpus.wordnet.ADV)
        else:
            wordnet_pos.append(nltk.corpus.wordnet.NOUN)

# Lemmatizer
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(p[n][0], pos=wordnet_pos[n]) for p in pos for n in range(len(p)) ]

for token in tokens:
    print(token)

# Stopword (Stop Word Removal)
nltk_stopwords = nltk.corpus.stopwords.words("english")
tokens = [token for token in tokens if token not in nltk_stopwords]
for token in tokens:
    print(token)

# NER (Named Entity Recognition)
ne_chunked_sents = [nltk.ne_chunk(tag) for tag in pos]
named_entities = []

for ne_tagged_sentence in ne_chunked_sents:
    for tagged_tree in ne_tagged_sentence:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
            entity_type = tagged_tree.label()
            named_entities.append((entity_name, entity_type))
            named_entities = list(set(named_entities))

for ner in named_entities:
    print(ner)

# Books_data.csv
df_data = pandas.read_csv(f"{path}\\Books_data.csv")
print(df_data)

for col in df_data.columns:
    print(col)