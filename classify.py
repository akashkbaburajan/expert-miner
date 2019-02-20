if __name__=="__main__":
    filename = "irisdata.csv"
    with open(filename,"r") as fh:
        data = fh.read()
        lines = data.split('\n')

    for line in lines:
        attributes = line.split(',')
        sepal_length = attributes[0]
        sepal_width = attributes[1]
        petal_length = attributes[2]
        petal_width = attributes[3]
        class_name = attributes[4]