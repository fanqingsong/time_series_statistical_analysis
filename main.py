
import src
from watchfiles import watch
import importlib

if __name__ == "__main__":
    src.machine_learning_workflow_pipeline()

    for changes in watch('./src'):
        print(changes)
        importlib.reload(src.kmeans_with_workflow)
        importlib.reload(src)
        src.machine_learning_workflow_pipeline()


