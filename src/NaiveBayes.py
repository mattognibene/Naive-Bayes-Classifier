# Matt Ognibene, 2018
# MIT License
# Thanks to Prof. Freifeld
from enum import Enum
from math import log, e
from abc import ABC, abstractmethod
import string


class Category(Enum):
    Ham = 0,
    Spam = 0


class NaiveBayes:
    def __init__(self, extractor):
        self.hamWordMap = {}
        self.spamWordMap = {}
        self.numSpam = 0
        self.numHam = 0
        self.extractor = extractor



    #THINGS TODO IN THE FUTURE
    #Abstract extraction using function objects

    #String x Category
    #adds this training example
    def addTrainingExample(self, document, category):
        words = self.extractor.extract(document)

        if category == Category.Ham:
            self.numHam += 1
            for w in words:
                self.hamWordMap[w] = self.hamWordMap.get(w, 0) + 1
        else:
            self.numSpam += 1
            for w in words:
                self.spamWordMap[w] = self.spamWordMap.get(w, 0) + 1


    #classifys an unknown document
    def classify(self, document):
        words = self.extractor.extract(document)
        cumulativeLogProb = 0.0
        for w in words:
            spamCount = self.spamWordMap.get(w, 0)
            hamCount = self.hamWordMap.get(w, 0)
            if spamCount + hamCount == 0:
                continue
            else:
                spamProbability = spamCount / (hamCount + spamCount)
                hamProbability = 1 - spamProbability

                if(spamProbability == 0):
                    spamProbability = .001
                if(hamProbability == 0):
                    hamProbability = .001

                cumulativeLogProb += log(spamProbability / hamProbability, e)
        return log((self.numSpam / self.numHam), e) + cumulativeLogProb


class Extractor(ABC):
    @abstractmethod
    # returns a list of all words from this document with not punctuation or capitalization
    def extract(self, document):
        pass


#expects Strings
class StringExtractor(Extractor):
    def extract(self, document):
        table = str.maketrans("", "", string.punctuation)
        document = document.translate(table)
        return document.lower().split()