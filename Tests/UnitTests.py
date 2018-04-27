# Unit tests
from src.NaiveBayes import Category
from src.NaiveBayes import NaiveBayes
from src.NaiveBayes import StringExtractor


class Example:
    def __init__(self, ):
        spamExamples = open("TrainingData/TestTrainingSpam.txt", "r").readlines()
        hamExamples = open("TrainingData/TestTrainingHam.txt", "r").readlines()
        self.nb = NaiveBayes(StringExtractor())

        for sex in spamExamples:
            self.nb.addTrainingExample(sex, Category.Spam)
        for hex in hamExamples:
            self.nb.addTrainingExample(hex, Category.Ham)

    def test(self):
        falsePositives = 0
        falseNegatives = 0
        uncertainties = 0  # TODO tolerance
        spamConfidences = []
        hamConfidences = []
        hamTests = open("TestingData/HamTests.txt", "r").readlines()
        spamTests = open("TestingData/SpamTest.txt", "r").readlines()
        for ht in hamTests:
            hamConfidences.append(self.nb.classify(ht))
            if (self.nb.classify(ht) > 0):
                falsePositives += 1
            elif (self.nb.classify(ht) > -5):
                uncertainties += 1
        for st in spamTests:
            spamConfidences.append(self.nb.classify(st))
            if (self.nb.classify(st) < 0):
                falseNegatives += 1
            elif (self.nb.classify(st) < 5):
                uncertainties += 1

        hamSum = 0
        for i in hamConfidences:
            hamSum += i

        spamSum = 0
        for i in spamConfidences:
            spamSum += i

        hamSampleMean = hamSum / hamTests.__len__()
        spamSampleMean = spamSum / spamTests.__len__()

        spamDeviations = []
        for s in spamConfidences:
            spamDeviations.append((s - spamSampleMean)**2)

        spamDeviationSum = 0
        for sd in spamDeviations:
            spamDeviationSum += sd

        spamSampleDeviation = (spamDeviationSum / spamDeviations.__len__()) ** (1/2)

        hamDeviations = []
        for s in hamConfidences:
            hamDeviations.append((s - hamSampleMean) ** 2)

        hamDeviationSum = 0
        for sd in hamDeviations:
            hamDeviationSum += sd

        hamSampleDeviation = (hamDeviationSum / hamDeviations.__len__()) ** (1 / 2)

        alpha = .1 #90 percent certain of the classification
        print("TEST STATISTICS")
        #if(hamTests.__len__() + spamTests.__len__() > 30):
        #    print("n > 30 => z-table")
        #todo, contstruct a confidence interval using a normal distribution approximation for a  more meaningful uncertainty statistic

        print("Tested ", (hamTests.__len__() + spamTests.__len__()), " examples on a training population of"
              , self.nb.numHam + self.nb.numSpam)
        print("Total Ham tests: ", hamTests.__len__())
        print("Total Spam tests: ", spamTests.__len__())
        print("False negative : ", (falseNegatives / spamTests.__len__()) * 100, "%")
        print("False positives : ", (falsePositives / hamTests.__len__()) * 100, "%")
        print("Uncertain : ", (uncertainties / (hamTests.__len__() + spamTests.__len__())) * 100, "%")
        print("Spam sample confidence mean:, ", spamSampleMean)
        print("Spam sample deviation: ", spamSampleDeviation)
        print("Ham sample confidence mean:, ", hamSampleMean)
        print("Ham sample deviation: ", hamSampleDeviation)


ex = Example()
ex.test()
