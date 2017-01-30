import random


def gen_file():
    f = open('kaggle_submission.csv', 'w')
    header = 'id,PES1'
    f.write(header + '\n')
    for i in range(16000):
        guess = random.randint(1,2)
        f.write(str(i) + ',' + str(guess) + '\n')
    f.close()

gen_file()