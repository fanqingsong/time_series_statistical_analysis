
from src import machine_learning_workflow_pipeline
from watchfiles import watch


if __name__ == "__main__":
    machine_learning_workflow_pipeline()

    for changes in watch('./src'):
        print(changes)
        machine_learning_workflow_pipeline()


