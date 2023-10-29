
from csv import reader
from sklearn.cluster import KMeans
import joblib
import dask
from .Ashare.Ashare import *
from .Ashare.MyTT import *
import matplotlib.pyplot as plt ;  from matplotlib.ticker import MultipleLocator


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


@dask.delayed
def get_data():
    df = get_price('sh000001', frequency='1d', count=100)
    # print(df)
    return df


@dask.delayed
def visualize(df):
    CLOSE = df.close.values
    print(CLOSE)

    MA5 = MA(CLOSE, 5)
    print(MA5)

    plt.figure(figsize=(15,8))
    plt.plot(CLOSE,label='SHZS');
    plt.legend();
    plt.grid(linewidth=0.5,alpha=0.7);
    plt.gcf().autofmt_xdate(rotation=45);
    plt.gca().xaxis.set_major_locator(MultipleLocator(len(CLOSE)/30))    #日期最多显示30个
    plt.title('SH-INDEX   &   BOLL SHOW',fontsize=20);
    plt.show()



def machine_learning_workflow_pipeline():
    # trainData = getTrainData()
    # numClusters = getNumClusters()
    # trainData = train(numClusters, trainData)
    # total = predict(trainData)
    #
    # total.compute()

    df = get_data()
    vr = visualize(df)
    vr.compute()
    print("-------lllllkkkkkooo")
    # print(df.compute())






