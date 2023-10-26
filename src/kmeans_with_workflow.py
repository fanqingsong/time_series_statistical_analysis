
from csv import reader
from sklearn.cluster import KMeans
import joblib
import dask


# Load a CSV file
def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    # discard head line
    next(lines)
    dataset = list(lines)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    # print("------")
    # print(column)
    for row in dataset:
        # print(f"row[column] = {row[column]}")
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def getRawIrisData():
    # Load iris dataset
    filename = '../iris.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
    # for one in dataset:
    #     print(one)
    # convert string columns to float
    for i in range(4):
        str_column_to_float(dataset, i)

    # convert class column to int
    lookup = str_column_to_int(dataset, 4)
    print(dataset[0])
    print(lookup)

    return dataset


@dask.delayed
def getTrainData():
    dataset = getRawIrisData()
    trainData = [ [one[0], one[1], one[2], one[3]] for one in dataset ]

    return trainData


@dask.delayed
def getNumClusters():
    return 3


@dask.delayed
def train(numClusters, trainData):
    print("numClusters=%d" % numClusters)

    model = KMeans(n_clusters=numClusters)

    model.fit(trainData)

    # save model for prediction
    joblib.dump(model, '../model.kmeans')

    return trainData


@dask.delayed
def predict(irisData):
    # test saved prediction
    model = joblib.load('../model.kmeans')

    # cluster result
    labels = model.predict(irisData)

    print("cluster result")
    print(labels)


def machine_learning_workflow_pipeline():
    trainData = getTrainData()
    numClusters = getNumClusters()
    trainData = train(numClusters, trainData)
    total = predict(trainData)

    total.compute()






