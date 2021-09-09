import os
# from argparse import ArgumentParser
os.environ['dist'] = '35'
os.environ['server'] = '0'
SERVER = bool(int(os.environ['server']))
DISTANCE = int(os.environ['dist'])
print(DISTANCE)

# def parse_arguments():
#     parser = ArgumentParser(__doc__)
#     parser.add_argument("--server", "-s", help="run without pygame?", default=False)
#     parser.add_argument("--dist", "-d", help="distance for grab", default=25)
#     return parser.parse_args()
#
# args = parse_arguments()
# print(args)
# SERVER = args.server
# DISTANCE = args.dist


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