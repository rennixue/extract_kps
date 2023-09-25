import re

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import PyPDF2
from io import StringIO
import time
import spacy
# 没用到但是得装，不然nlp.add_pipe("textrank")会报错
import pytextrank
import pandas as pd


from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
wnl = WordNetLemmatizer()

# pdf2txt module
def pdf_to_text(pdfname):
    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    #codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(pdfname, 'rb')
    for page in PDFPage.get_pages(fp,check_extractable=False):
        time.sleep(0.001)
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()
    # with open('test.txt', 'w') as f:
    #     f.write(text)
    return text


def PyPDF22txt(FILE_PATH):
    txt = ''
    with open(FILE_PATH, mode='rb') as f:

        reader = PyPDF2.PdfReader(f)
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            txt = txt+page.extract_text()
    with open('text_pypdf2.txt', 'w') as f:
        f.write(txt)
    return txt


# information extractor module
stopwords = pd.read_csv("stopword_unit_guide.csv", encoding='utf-8')["stopwords"].tolist()


# 1、text_rank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")


def text_rank_extractor(text):
    # print(text)
    text = text.lower()
    text_list = [text[i: i + 90500] if i + 90500 < len(text) else text[i: len(text)] for i in range(0, len(text), 90000)]
    n = len(text) // 1000000 + 1
    words = []
    i = 0
    for _ in text_list:
        i += 1
        doc = nlp(_)
        t = [(phrase.text, phrase.rank) for phrase in doc._.phrases if phrase.text and phrase.rank]
        ranks = [j for i, j in t]
        average_rank = sum(ranks) / len(ranks)
        w = [(w,rank) for w, rank in t if rank >= average_rank]
        w.sort(key=lambda x: x[1], reverse=True)
        w = w[:100]
        # w = t.sort(key=lambda x: x[1], reverse=True)[:100]

        words.extend(w)
    print(i)
    return words


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def clean_repeat(kps):
    kpset = set()
    kps_ = []
    for w, _ in kps:
        kp_list = []
        for wd, pos in pos_tag(word_tokenize(w)):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            kp_list.append(wnl.lemmatize(wd, wordnet_pos))
        kp_string = ' '.join(kp_list)
        if kp_string.lower() in kpset:
            continue
        kpset.add(kp_string.lower())
        # 去除长度为1和长度大于4的
        if len(w.split()) <2 or len(w.split()) >4:
            continue
        # 去除包含特殊符号的关键词
        if re.findall(r'[^a-zA-Z\-\s]+', w):
            continue
        kps_.append((w, _))
    return kps_




def main(pdf_path):
    try:
        text = pdf_to_text(pdf_path)
        print(len(text))
        if len(text) < 50:
            text = PyPDF22txt(pdf_path)
    except Exception as e:
        text =PyPDF22txt(pdf_path)
    w = text_rank_extractor(text)
    w = clean_repeat(w)
    df = pd.DataFrame(w, columns=['kps', 'rank'])
    csv_name = '.'.join(pdf_path.split('.')[:-1]) + '.csv'
    df.to_csv(csv_name)
    return w


if __name__ == '__main__':
    pdf_path = './raw_data/economics/public economics/Public_economics.pdf'
    # pdf_path = '9912021v2.pdf'
    w = main(pdf_path)

    print(w)
    print(len(w))
    # word = [_ for _, r in w]
    #
    # df = pd.DataFrame(w, columns=['kps', 'rank'])
    # df.to_csv('text.csv')


    # w = 'g-cells'
    # for wd, pos in pos_tag(word_tokenize(w)):
    #     wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
    #     print(wd, pos)
    #     print(wnl.lemmatize(wd, wordnet_pos))