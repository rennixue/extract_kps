import os
import unicodedata

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
import time
import PyPDF2
import spacy
import pandas as pd
# 没用到但是得装，不然nlp.add_pipe("textrank")会报错
import pytextrank
import yake
from rake_nltk import Rake
import json
import string
import re
from tqdm import tqdm
from nltk import pos_tag, word_tokenize, ne_chunk
import nltk
from names_dataset import NameDataset


stopwords = pd.read_csv("stopword_unit_guide.csv", encoding='utf-8')["stopwords"].tolist()

m_v1 = NameDataset()

# pdf2txt

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
    with open('test.txt', 'w') as f:
        f.write(text)
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



# -------------------------------------------------------------------------------------------------------------
#textrank extractor
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
nlp.max_length = 10**8
def text_rank_extracter(sent):
    kps = []
    doc = nlp(sent)
    for phrase in doc._.phrases:
        # Phrase(text='economic conditions', chunks=[economic conditions], count=1, rank=0.17021645072531422)
        if phrase.text.lower() in stopwords: continue
        if phrase.rank < 0.00001:continue
        kps.append(phrase.text)
        #print(phrase.text,phrase.rank)
    # print(kps)
    return kps


# -------------------------------------------------------------------------------------------------------------
#yake extractor
kw_extractor = yake.KeywordExtractor()
language = "en"
max_ngram_size = 5
deduplication_threshold = 1#不允许有重复的词
numOfKeywords = 1
custom_kw_extractor = yake.KeywordExtractor(
                                lan=language, n=max_ngram_size,
                                dedupLim=deduplication_threshold,
                                top=numOfKeywords, features=None)
def yake_extracter(text):
    kps = []
    keywords = custom_kw_extractor.extract_keywords(text)
    for y,k in keywords:
        if k.lower() in stopwords: continue
        kps.append(k)
    return kps


# -------------------------------------------------------------------------------------------------------------
#rake extractor
r = Rake()
def rake_extracter(text):
    ss =r.extract_keywords_from_text(text)
    kps = [k for k in list(set(r.get_ranked_phrases())) if k.lower() not in stopwords]
    return kps


# -------------------------------------------------------------------------------------------------------------
# utils
def dump_json(obj, fp, encoding='utf-8', indent=4, ensure_ascii=False):
    with open(fp, 'w', encoding=encoding) as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii)


def load_json(fp, encoding='utf-8', json_lines=False):
    with open(fp, encoding=encoding) as fin:
        if not json_lines:
            return json.load(fin)
        else:
            ret = []
            for line in fin:
                ret.append(json.loads(line))
            return ret


def listAlphabet():
    dict_Alphabet ={}

    for i,a in enumerate(list(string.ascii_lowercase)):
        dict_Alphabet[a] = i
    return dict_Alphabet,list(string.ascii_lowercase)
alphabet_dict,alphabet_list =listAlphabet()


def has_same_words_kps(kp):
    ws = set()
    for k in kp.split():
        ws.add(k.strip().lower())
    if len(ws)<len(kp.split()):return True
    return False


def keep_first(kp_list):
    kpset =set()
    kp_set = set()
    for k in kp_list:
        if k.lower() in kpset:continue
        kpset.add(k.lower())
        kp_set.add(k)
    return list(kp_set)


def just_upper(k):
    if len(re.findall("[A-Z]",k.strip()))==len(k.strip()) and len(k.strip())>2 :return True
    else:return False
def couple_start(kp):
    if type(kp)!=str:return True
    set_letter = set()
    for k in kp.strip()[:2]:
        set_letter.add(k.lower())
    if len(set_letter)==1:return True
    return False
def remove_space_in_word(tt):
    tt =tt.replace("  ","#")
    tt =tt.replace(" ","")
    tt =tt.replace("#"," ")
    return tt
#count words frequency
#kps_sort = sorted(Counter(word_list).items(),key=lambda item:item[1],reverse=True)
def count_words(w):
    return len(str(w).strip().split(" "))

# -------------------------------------------------------------------------------------------------------------
# kps clean functions
def clean_kp_list_1(kps):
    kplist = []
    for kp in tqdm(kps):
        kp_ = re.sub("\s+", " ", re.sub("\W", " ", kp).strip()).strip()
        if (len(kp_.split()) < 2):
            if not just_upper(kp_): continue
        # if ("NN" not in pos_tag([kp_])[0][1]):continue
        kplist.append(kp_)
    return [k for k in kplist if len(k) > 1 and len(re.findall("\d+", k)) < 1]


def clean_kp_list_2(kps):
    kplist = []
    for kp in kps:
        if len(re.findall("^[a-z]\s+|[A-Z]\s+", kp)) > 0: continue
        if len(re.findall("\s+[A-Z]$|\s+[a-z]$", kp)) > 0: continue
        pt = pos_tag(word_tokenize(kp))
        for i, pog in enumerate(pt):
            if len(pt) == 2 and "DT" in pog[1]: break
            if "PRP$" in pog[1]: break
            # if  "IN" in pog[1]:break
            if "CD" in pog[1]: break
            if "NN" in pog[1]:
                kplist.append(kp.strip())
                break
    return list(set(kplist))


def clean_kp_1(kp):
    if type(kp) != str: return 0
    if len(str(kp)) > 45: return 0
    for s in kp.split(" "):
        if len(s.strip()) == 1:
            return 0
    klist = [k for k in str(kp).split() if k]
    if len(klist) > 5: return 0
    kset = list(set(klist))
    kset.sort(key=klist.index)
    kp = " ".join(kset)
    if len(re.findall(r'[\u4e00-\u9fa5]+', kp)) > 0: return 0
    w_p = pos_tag(word_tokenize(kp.strip()))
    k_lat = kp.split(" ")[-1]
    name = False
    if len(klist) > 1 and len(re.findall("[A-Z]", kp)) > 0:
        for k in klist:
            if m_v1.search_first_name(k):
                name = True
            else:
                name = False
    if name: return 0

    wl_p = pos_tag(word_tokenize(kp.strip()))
    entities = nltk.chunk.ne_chunk(wl_p)
    for ner in entities.pos():
        if ner[1] in ["GPE", "DATA", "LOCATION", "TIME", "MONEY", "PERCENT", "FACILITY"]:
            return 0

    nnp, nns = 0, 0
    for w, p in w_p:
        if p in ["WDT", "WP", "WP$", "RBR", "WRB", "PBR", "RBS", "RB", "EX", "MD", "JJR", "VBP", "VBN", "CC", "JJS",
                 "PDT", "POS", "PRP", "RBS", "RP", "UH", "VBD", "VBZ"]:
            return 0
        if p in ["NNP", "NNPS"]:
            nnp += 1
        if p == "NNS":
            nns += 1
    if nns > 1 and (len(re.findall("[A-Z]", kp)) == 0): return 0
    if nnp == len(w_p): return 0
    if w_p[0][1] in ["DT", "RB"]:
        return " ".join(kp.strip().split(" ")[1:])
    if "VB" in w_p[-1][1]: return 0
    if w_p[-1][1] in ["TO"]:
        return " ".join(kp.strip().split(" ")[:-1])
    return kp


def clean_kp_2(kp):
    if type(kp) != str: return 0
    if not just_upper(kp):
        if couple_start(kp): return 0
    # if count_words(kp)<2:return 0
    if count_words(kp) > 5: return 0
    if "niversity" in kp: return 0
    if "school" in kp: return 0
    klist = [k.strip().lower() for k in kp.split(" ")]
    pass_list = ["tion", "sion", "an", "date", "other", "outside", "issue", "terms", "notes", "new", "low", "main",
                 "money", "bank", "time", "income", "import", "costs"]
    for pl in pass_list:
        if pl in klist: return 0
    if len(re.findall("^[A-Z]", kp)) == 0:
        if len(re.findall("\s+[A-Z]", kp)) > 0:
            return 0
    for k in kp.split():
        if just_upper(k): continue
        if len(re.findall("[A-Z]", k)) > 1:
            return 0
        if (len(k) < 4) and (k not in ["of"]): return 0
    if has_same_words_kps(kp): return 0
    return kp.strip()


def clean_over_700(kp_list):
    if len(kp_list) < 500: return kp_list
    kplist, kpset = [], []
    for k in kp_list:
        kl = k.strip().split()
        if len(kl) == 1: kplist.append(k)
        kpt = pos_tag(word_tokenize(k.strip()))
        if kpt[-1][1] == "NNP": continue
        kplist.append(k)
    if len(kplist) > 700:
        for k in kplist:
            kl = k.strip().split()
            if len(kl) == 1: kpset.append(k)
            kpt = pos_tag(word_tokenize(k.strip()))
            pl = [p[1] for p in kpt]
            if "JJ" in pl and "NNS" in pl: continue
            if "JJ" in pl and "NN" in pl: continue
            kpset.append(k)
        return kpset
    else:
        return kplist
# -------------------------------------------------------------------------------------------------------------
##上面是工具，下面是运行步骤
# 步骤0 ，将pdf 转txt
# 步骤1，文件夹下的pdf转txt后文件提取词汇表部分的整体字符串
def extract_glossary_or_index_parts():
    kplist = []
    # for major in os.listdir("./text_book_22mar_1510"):
    for major in os.listdir("./raw_data"):
        if len(major.split(".")) > 1: continue

        for course in tqdm(os.listdir(f"./raw_data/{major}/")):
            if len(course.split(".")) == 1:
                for file in tqdm(os.listdir(f"./raw_data/{major}/{course}/")):
                    end = file.split(".")[-1]
                    if "txt" not in end: continue
                    glossary_last = []
                    content = []
                    with open(f"./raw_data/{major}/{course}/{file}", mode="r", encoding='utf-8') as f:
                        for i, line in enumerate(f.readlines()):
                            content.append(line)
                    content = [unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode().strip() for c in
                               content]
                    content = [c for c in content if c]
                    Stud_content = content.copy()
                    len_con = len(content)
                    kps = {}
                    label = "Stud"
                    if 100000 > len_con > 30000:
                        content = content[-int(len_con / 10):]
                    elif len_con > 50000:
                        content = content[-int(len_con / 20):]
                    else:
                        content = content[-5000:]
                    # print("len content: ", len(content))
                    # glossary first
                    for i, line in enumerate(content):
                        if ("glossary" in line.lower()) and (len(line.strip()) < 15) and (
                                len(re.findall("glossary$", line.strip().lower())) > 0):
                            if i == len(content) - 1: break
                            if "index" in content[i + 1].lower(): continue
                            glossary_last = content[i:]
                            label = "glossary"
                            break
                    # second windows count ^a num 6
                    if len(glossary_last) < 100:
                        for i, line in enumerate(content):
                            if i == len(content) - 1: break
                            windows = content[i:i + 10]
                            a_num = 0
                            for w in windows:
                                if len(re.findall("^a", w.lower().strip())) > 0:
                                    a_num += 1
                            if a_num > 6:
                                glossary_last = content[i - 50:]
                                label = "windows_a_6"
                                break
                    # third index
                    if len(glossary_last) < 100:
                        for i, line in enumerate(content):
                            if i == len(content) - 1: break
                            if ("index" in line.lower()) and (len(re.findall("^A+", content[i + 1].strip())) > 0) and (
                                    len(re.sub("^\s*\s*", "", line.strip()).strip()) < 6) and (
                                    len(re.findall("index$", line.strip().lower())) > 0):
                                glossary_last = content[i:]
                                label = "index"
                                # print("index",len(glossary_last))
                                break
                    # fourth windows count ^a num 4
                    if len(glossary_last) < 100:
                        for i, line in enumerate(content):
                            if i == len(content) - 1: break
                            windows = content[i:i + 10]
                            a_num = 0
                            for w in windows:
                                if len(re.findall("^a", w.lower().strip())) > 0:
                                    a_num += 1
                            if a_num > 4:
                                glossary_last = content[i - 10:]
                                label = "windows_a_4"
                                break
                    # second windows count ^b num
                    if len(glossary_last) < 100:
                        for i, line in enumerate(content):
                            if i == len(content) - 1: break
                            windows = content[i:i + 10]
                            a_num = 0
                            for w in windows:
                                if len(re.findall("^b", w.lower().strip())) > 0:
                                    a_num += 1
                            if a_num > 4:
                                glossary_last = content[i - 50:]
                                label = "windows_b"
                                break
                    # finanly Stud!!!!
                    if len(glossary_last) < 100: glossary_last = [s for s in Stud_content if
                                                                  len(re.findall("\d+", s)) > 0]
                    # print("*" * 30)
                    # print(label)
                    # print(major)
                    # print(course)
                    # print(file)
                    kps["major"] = major
                    kps["course"] = course
                    kps["file"] = file.split(".")[0] + ".pdf"
                    kps["label"] = label
                    kps["kps"] = glossary_last
                    kplist.append(kps)
    dump_json(kplist, "glossary_22Mar.json")

# extract_glossary_or_index_parts()

# 步骤2，词汇表里面提取知识点短语
# glossary_part = load_json("glossary_22Mar.json")
# kplist = []
# for g in glossary_part:
#     g["kps_rake"], g["kps_text_rank"] = [], []
#     # print(len(g["kps"]), "##", g["label"], "##", g["course"], "##", g["file"], "##")
#     # print("*" * 80)
#     for k in tqdm(g["kps"]):
#         # just text rank extracter
#         # g["kps_rake"].extend(rake_extracter(k))
#         g["kps_text_rank"].extend(text_rank_extracter(k))
#     kplist.append(g)
# dump_json(kplist, "glossary_kps_24Mar.json")

# 步骤3，清理知识点短语

glossary_kps = load_json("glossary_kps_24Mar.json")
kp_clean = []
for gkp in glossary_kps:
    text_rank_clean, rake_clean = [], []
    # just  clean text rank kps
    #     for k in tqdm(clean_kp_list_2(clean_kp_list_1(gkp["kps_rake"]))):
    #         k_ = clean_kp_2(clean_kp_1(k))
    #         if len(str(k_))>4 and len(str(k_).split(" "))>1:
    #             rake_clean.append(k_)
    #     gkp["kps_rake_clean"] = list(set(rake_clean))
    for k in tqdm(clean_kp_list_2(clean_kp_list_1(gkp["kps_text_rank"]))):
        k_ = clean_kp_2(clean_kp_1(k))
        if len(str(k_)) > 1:
            # print(k_)
            text_rank_clean.append(k_)
    # gkp["kps_text_rank_clean"] = list(set(text_rank_clean))
    # gkp["kps_concat_clean"] = list(set(text_rank_clean+rake_clean))
    gkp["kps_concat_clean"] = list(set(clean_over_700(text_rank_clean)))
    kp_clean.append(gkp)
dump_json(kp_clean, "glossary_kps_clean_24Mar.json")






# if __name__ == '__main__':
#     stopwords = pd.read_csv("stopword_unit_guide.csv", encoding='utf-8')["stopwords"].tolist()
#
#     filename = 'raw_data/Hal R. Varian - Microeconomic Analysis-W. W. Norton & Company (1992).pdf'
#     # pdf_to_text(filename)
#     # PyPDF22txt(filename)
#     sent = 'The size of the multiplier is bound to vary according to economic conditions. For an economy operating at full capacity, the fiscal multiplier should be zero. Since there are no spare resources, any increase in government demand would just replace spending else-where. But in a recession, when workers and factories lie idle, a fiscal boost can increase overall demand. And if the initial stimulus triggers a cascade of expenditure among con-sumers and businesses, the multiplier can be well above one. '
#     # text_rank_extracter(sent)
#     # print(yake_extracter(sent))
#     # print(rake_extracter(sent))
#     kps = ['hello world', 'economic conditions']
#     # print(clean_kp_list_2(kps))
#     print(clean_kp_1('hello world'))