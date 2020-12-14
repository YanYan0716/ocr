'''
获取数据集的.txt文件
'''
import os


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


if __name__ == '__main__':
    root_path = 'D:\\algorithm\\ocr\\icdar\\train\\GT'
    path_list = []
    path_list = listdir(root_path, path_list)
    print(len(path_list))
    train_file = open('D:\\algorithm\\ocr\\icdar\\train\\GT.txt', 'w')
    for i in range(len(path_list)):
        # train_file.write(str(path_list[i])+' '+str(path_list[i]).split('\\')[-2]+'\n')
        train_file.write(str(path_list[i]) + '\n')
    train_file.close()