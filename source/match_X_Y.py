# import numpy as np
def apart(path):
    """
    获取时间和词向量
    """
    file=open(path,mode="r",encoding="UTF-8")
    lines=file.readlines()
    list_lines=[]
    for line in lines:
        line_new=line.replace("\n","")
        line_new=list(line_new.split("  "))
        list_lines.append(line_new)
    return list_lines


def compare_and_gain(path_5000,path_2,path_write):
    
    list_lines_5000=apart(path_5000)  #获取one_hot.txt中的时间和词向量
    list_lines_2=apart(path_2)  #获取train/test/validation.txt中的时间和向量

    file_write=open(path_write,mode="w",encoding="UTF-8")
    len_5000=len(list_lines_5000)
    len_2=len(list_lines_2)
    num_5000=0
    num_2=0
    count=0
    while 1:
        if num_2>=len_2 or num_5000>=len_5000:
            break
        if list_lines_2[num_2][0]==list_lines_5000[num_5000][0]:
            if len(list_lines_5000[num_5000][1])!=15000:
                num_5000+=1
                num_2+=1
                continue
               
            file_write.write(list_lines_5000[num_5000][1])
            
            file_write.write("  ")
            file_write.write(list_lines_2[num_2][1])
            count+=1
            file_write.write("\n")
            num_5000+=1
            num_2+=1
        else:
            num_5000+=1
    print(count)
    file_write.close()

if __name__=="__main__":
    path1="C:\\Users\\86185\\Desktop\\bit\\大二下\\知识工程\\作业一\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\"
    path2="C:\\Users\\86185\\Desktop\\bit\\大二下\\知识工程\\作业一\\8cf3c8484e73689514c0a39cb460bdaa_29eabca25474cbb525d8a9e3c00a8279_8\\课程数据\\"
    path_5000=path1+"one_hot.txt"

    path_2=path2+"test.txt"
    path_write=path2+"test_raw_ver1.txt"

    # path_2=path2+"validation.txt"
    # path_write=path2+"validation_raw_ver1.txt"

    # path_2=path2+"train.txt"
    # path_write=path2+"train_raw_ver1.txt"
    compare_and_gain(path_5000,path_2,path_write)