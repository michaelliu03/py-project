import jieba


class Processor():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']

    def __init__(self):
        self.encoderFile = "./Q.txt"
        self.decoderFile = './A.txt'
        self.stopwordsFile = "./dict/stop_words.utf8"

    def wordToVocabulary(self, originFile, vocabFile, segementFile):
        # stopwords = [i.strip() for i in open(self.stopwordsFile).readlines()]
        # print(stopwords)
        # exit()
        vocabulary = []
        sege = open(segementFile, "w",encoding='utf-8')
        with open(originFile, 'r',encoding='utf-8') as en:
            for sent in en.readlines():
                # 去标点
                if "enc" in segementFile:
                    sentence = sent.strip()
                    words = jieba.lcut(sentence)
                    print(words)
                else:
                    words = jieba.lcut(sent.strip())
                vocabulary.extend(words)
                for word in words:
                    sege.write(word + " ")
                sege.write("\n")
        sege.close()

        # 去重并存入词典
        vocab_file = open(vocabFile, "w",encoding='utf-8')
        _vocabulary = list(set(vocabulary))
        _vocabulary.sort(key=vocabulary.index)
        _vocabulary = self.vocab + _vocabulary
        for index, word in enumerate(_vocabulary):
            vocab_file.write(word + "\n")
        vocab_file.close()

    def toVec(self, segementFile, vocabFile, doneFile):
        word_dicts = {}
        vec = []
        with open(vocabFile, "r",encoding='utf-8') as dict_f:
            for index, word in enumerate(dict_f.readlines()):
                word_dicts[word.strip()] = index

        f = open(doneFile, "w",encoding='utf-8')
        with open(segementFile, "r",encoding='utf-8') as sege_f:
            for sent in sege_f.readlines():
                sents = [i.strip() for i in sent.split(" ")[:-1]]
                vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word)) + " ")
                f.write("\n")
        f.close()

    def run(self):
        # 获得字典
        self.wordToVocabulary(
            self.encoderFile, './tfdata/enc.vocab', './tfdata/enc.segement')
        self.wordToVocabulary(
            self.decoderFile, './tfdata/dec.vocab', './tfdata/dec.segement')
        # 转向量
        self.toVec("./tfdata/enc.segement",
                   "./tfdata/enc.vocab",
                   "./tfdata/enc.vec")
        self.toVec("./tfdata/dec.segement",
                   "./tfdata/dec.vocab",
                   "./tfdata/dec.vec")


process = Processor()
process.run()
