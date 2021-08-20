import os


def path_from_root(dirname: str) -> str:
    """
    Returns a project root path joined with :param dirname:
    The project root is considered the first directory containing 'requirements.txt'
    :param dirname: path to join with
    :return:
    """
    path = os.path.normpath(__file__)
    max_levels_up = 3
    counter = 0
    while path and counter < max_levels_up:
        parts = os.path.split(path)
        preceeding_part = parts[0]
        tried_filename = os.path.join(preceeding_part, 'requirements.txt')
        if os.path.exists(tried_filename):
            root_path = preceeding_part
            return os.path.join(preceeding_part, dirname)
        path = preceeding_part
        counter += 1
        if counter >= max_levels_up:
            root_path = preceeding_part
            return os.path.join(preceeding_part, dirname)
    return ''