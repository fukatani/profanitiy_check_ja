from gensim.models import KeyedVectors
from pathlib import Path
import random
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy


def load_word2vec():
    model_path = '/home/ryo/Downloads/entity_vector/entity_vector.model.bin'
    return KeyedVectors.load_word2vec_format(model_path, binary=True)


def choose_word(num, inappropriates):
    inappropriates = set(inappropriates)
    vocab_path = '/home/ryo/Downloads/entity_vector/entity_vector.model.txt'
    vocabs = []
    found = 0
    with Path(vocab_path).open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            word = line.split()[0]

            if word in inappropriates:
                found += 1
                # print(word)
                vocabs.append(word)
            if random.random() > 0.99:
                vocabs.append(word)
            if i == num:
                break
    print(len(vocabs), found)
    return vocabs


def extract_inappropriate():
    vocab_path = '/home/ryo/work/github/profanity_check_ja/inappropriate.txt'
    vocabs = []
    with Path(vocab_path).open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            vocabs.append(line.split()[0])
    return vocabs


def make_labels(vocabs, model, inappropriates):
    xs = []
    ys = []

    for vocab in vocabs:
        xs.append(model.get_vector(vocab, norm=True))
        if vocab in inappropriates:
            ys.append(1)
        else:
            ys.append(0)
    return numpy.array(xs), numpy.array(ys)


def get_model():
    return RandomForestClassifier(class_weight='balanced')
    # return GradientBoostingClassifier()

def cross_validation(xs, ys):
    n_splits = 3
    model = get_model()
    cv = StratifiedKFold(n_splits=n_splits).split(xs, ys)
    print("Start cross validation")
    print(cross_val_score(model, xs, ys, cv=cv))


def test_model(xs, ys, word2vec):
    model = get_model()
    model.fit(xs, ys)

    for word in ["バカ", "天才", "チビ", "通常", "どじ", "まぬけ", "パソコン", "りんご", "ペニス", "ボート", "小便", "大便", "ホームベース"]:
        v = word2vec.get_vector(word, norm=True).reshape(1, -1)
        print(f"{word} {model.predict_proba(v)}")


if __name__ == '__main__':
    inappropriates = extract_inappropriate()
    vocabs = choose_word(10000000, inappropriates)
    word2vec = load_word2vec()
    xs, ys = make_labels(vocabs, word2vec, inappropriates)
    # cross_validation(xs, ys)
    test_model(xs, ys, word2vec)
