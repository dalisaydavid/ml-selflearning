# K-Nearest Neighbors Algorithm - David Dalisay
# Performs the algorithm against a Flags data set. Each record contains attributes that describe a particular flag.
# Given a new flag (create one), the algorithm will classify the new flag to an existing flag. The existing flag will be its most similar.
# *Important note: My code will be done in the same steps as Jason Brownlee's "Tutorial to implement k-nearest neighbors in python from scratch".
import csv
import random
import math
import operator

# Load data from .data files
def loadData(fileName, fileLen):
    f = open(fileName, "r")
    trainData = []
    for line in f:
        splitList = line.split(",")
        row = [splitList[0]] + [int(splitList[i]) for i in range(len(splitList)) if i != 0]
        trainData.append(row)
    f.close()
    random.shuffle(trainData)
    return trainData
# Define distance metric: Euclidean distance
# @Return 0 if row1=row2
def euclDist(row1, row2):
    # print("row1:{0}, row1 size:{1}".format(row1,len(row1)))
    # print("row2:{0}, row2 size:{1}".format(row2,len(row2)))
    if len(row1) != len(row2):  # Cannot compare 2 rows/vectors with different lengths
        return -1
    row1 = row1[1:] # When calculating distance,
    row2 = row2[1:] # don't use the class data. (Class data is 1st element)
    euclDiff = 0
    for i in range(len(row1)):
        euclDiff += math.pow((row1[i] - row2[i]), 2)
    totalDist = math.sqrt(euclDiff)
    return totalDist


# Get k nearest neighbors around any data(test) point!
# "Nearest" is defined by Euclidean distance.
def getNeighbors(trainData, testPoint, k):
    distances = []
    for i in range(len(trainData)):
        currDist = euclDist(trainData[i],testPoint)
        distances.append((trainData[i],currDist))
    distances.sort(key=operator.itemgetter(1)) # Sort by 2nd element of each record in distances list.
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# Predict class of test point based on neighbors.
# Get a "response" representing the the majority vote class prediction.
def getResponse(n):
    classes = {}
    for i in range(len(n)):
        currClass = n[i][-1] # Get the class of the current neighbor.
        if currClass in classes:
            classes[currClass] += 1
        else:
            classes[currClass] = 1
    # Sort list of neighbors by vote weight.
    sortedClasses = sorted(list(classes.items()), key=operator.itemgetter(1), reverse=True)
    # print("sortedClasses={0}".format(sortedClasses))
    return sortedClasses[0][0] # Return the class that carries the most weight, or has the most votes.
                               # sortedClasses[highest voted][class name]

# Get the accuracy of the response. See how accurate the algorithm's predictions are.
# Accuracy = (sum of total correct predictions / total num of records in the test set) * 100
# the length of testData = length of predictions
def getAccuracy(testData, predictions):
    if len(testData) != len(predictions):
        return -1
    correctPredictions = 0
    for i in range(len(testData)):
        if testData[i][-1] == predictions[i]:
            correctPredictions += 1
    accuracy = float(correctPredictions/len(testData))*100.00
    return accuracy


# @TODO: Implement a way to find the optimal k value. For now, use k=3.
def main():
    trainSet = loadData('flag_data.data',194)
    # Given a new country, with a new flag "Lozania"
    testSet = [["Lozania",6,4,500,5,3,7,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,1,1,1,1,1,0,1,1]]
    k = 10
    predictions = []
    for currTestPoint in testSet:
        neighbors = getNeighbors(trainSet,currTestPoint,k)
        response = getResponse(neighbors)
        predictions.append(response)
    for i in neighbors:
        print(i)
    print("\n{0}'s top Classification: {1}".format(testSet[0][0],neighbors[0][0]))

main()