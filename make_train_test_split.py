from random import shuffle
import csv
import glob

action_classes = ['notfall','fall']
print(len(set(action_classes)))
def create_csvs():
    train = []
    test = []

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('../dataset/videos/resized/{}/*.avi'.format(directory)):
            number = int(((filename.split('/')[-1]).split('.')[0]))

            if number%5==0:
                test.append([filename, myclass, directory])
            else:
                train.append([filename, myclass, directory])
                
    

    shuffle(train)
    shuffle(test)


    with open('train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'action'])
        mywriter.writerows(train)
        print('Training CSV file created successfully')

    with open('test.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'action'])
        mywriter.writerows(test)
        print('Testing CSV file created successfully')

    print('CSV files created successfully')


if __name__ == "__main__":
    create_csvs()
