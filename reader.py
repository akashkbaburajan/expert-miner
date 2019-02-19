import csv
data = [];
with open('iris.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        print(row)
        data.append(row)

csvFile.close()
print ('\nprinting data\n'+str(data))