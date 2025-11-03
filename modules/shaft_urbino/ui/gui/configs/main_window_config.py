from config.config import OutputColorSpaces, OutputFormats

COLOR_SPACES = [
    {"id": idx, "name": color_space.value}
    for idx, color_space in enumerate(OutputColorSpaces)
]

FILE_FORMATS = [
    {"id": idx, "name": file_format.value}
    for idx, file_format in enumerate(OutputFormats)
]
