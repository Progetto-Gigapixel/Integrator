import configparser

def setRawTherapeeOptions(path='./modules/shaft/marsiglia/config.ini',value="C:\\Program Files\\RawTherapee\\5.11"):
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    config.read(path)
    print(config)
    config['directories']['rawtherapee_path'] = value
    # Write the changes back to the file
    with open(path, 'w') as configfile:
        config.write(configfile)

    return