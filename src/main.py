import os


if "server" in os.environ:
    SERVER = bool(int(os.environ['server']))
else:
    SERVER = True
if 'dist' in os.environ:
    DISTANCE = int(os.environ['dist'])
else:
    DISTANCE = 25


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
    from agi_proto_framework import AgiProtoFramework
    main(AgiProtoFramework)
