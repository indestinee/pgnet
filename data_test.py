from my_folder import MyImageFolder

if __name__ == '__main__':
    d1 = MyImageFolder('data/train.zip')
    d2 = MyImageFolder('data/val.zip')

    for each in d2.classes:
        print(each)

