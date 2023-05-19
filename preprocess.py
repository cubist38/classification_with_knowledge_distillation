import os

def map_class_to_index(data_root):
    class_names = os.listdir(data_root)
    class_to_index = dict((name, index) for index, name in enumerate(class_names))
    return class_to_index