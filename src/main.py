import os


os.environ['dist'] = '35'
os.environ['server'] = '0'
SERVER = bool(int(os.environ['server']))
DISTANCE = int(os.environ['dist'])



def main(test_class):
    """
    Loads the test class and executes it.
    """
    print("Loading %s..." % test_class.name)
    test = test_class()
    if SERVER:
        while True:
            test.Step(test.settings)
    test.run()



if __name__ == "__main__":
    from collision_processing import CollisionProcessing
    main(CollisionProcessing)