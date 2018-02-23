import csv
import sys

# with open('data/test_csv.csv') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         print row


# f = open(sys.argv[0], 'rb')
f = open('data/test.csv')
reader = csv.reader(f)


# printing CSV
rownum = 0
for row in reader:
    if rownum == 0:
        header = row
    else:
        colnum = 0
        for col in row:
            print '%-8s: %s' % (header[colnum], col)
        colnum += 1
    rownum += 1
f.close()

# Writing to CSV
f = open('data/test.csv', 'rb')
reader = csv.reader(f)
out = open('out.csv', 'wb')
writer = csv.writer(out, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)

for row in reader:
    writer.writerow(row)

f.close()
out.close()