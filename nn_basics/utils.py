def convert_number_to_class(class_number):
    number_mapping = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }
    return number_mapping[class_number]


def create_output_file(file_name):
    import os
    from pathlib import Path
    if not os.path.exists(file_name):
        Path(f'nn_data/{file_name}').touch()
