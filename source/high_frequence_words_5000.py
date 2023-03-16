
def high_frequence_word_4999(path):
    with open(path,"r", encoding='ANSI') as f:
        #统计所需高频词
        lines=f.readlines()
        sum=0
        dict={}
        for line in lines:#读取TXT文件的每一行

            #获取line中的每一个词
            line=list(line.split("  "))
            line=[i for i in line if(len(str(i))!=0)]#移除空元素
            line.remove('\n')#移除list中的换行符

            if line.__len__()==0:
                continue

            #判断每一个词
            for word in line:

                #将中文与词性标注分开
                word=list(word.split("/"))

                #删除中文词前的方括号
                if word[0][0]=='[':
                    len_word=word[0].__len__()
                    word[0]=word[0][1:len_word]
                    # print(word[0])

                #判断动词
                if word[1][0]=='v':
                    #判断该词是否出现过，若未出现，则添加进入字典；若已出现，则value值+1
                    if word[0] not in dict:
                        sum+=1
                        dict[word[0]]=1
                    else:
                        dict[word[0]]+=1
                
                #判断时间
                if word[1][0]=='t':
                    if word[0][-1]=='年':
                        continue
                    #判断该词是否出现过，若未出现，则添加进入字典；若已出现，则value值+1
                    if word[0] not in dict:
                        sum+=1
                        dict[word[0]]=1
                    else:
                        dict[word[0]]+=1
                
                #判断部分副词
                if word[1]=="ul" or word[1]=="uo" or word[1]=="d":
                    #判断该词是否出现过，若未出现，则添加进入字典；若已出现，则value值+1
                    if word[0] not in dict:
                        sum+=1
                        dict[word[0]]=1
                    else:
                        dict[word[0]]+=1
    print("find high frequence word finished\n")
    return dict

def descending_sort(dict):
    #对字典按照value值进行降序排序
    a_sort_list =sorted(dict.items(),key=lambda x:x[1], reverse=True)
    a_sort_dict = {}
    for n, s in a_sort_list:
        a_sort_dict[n] = s 
    print("descending sort finished\n")
    return a_sort_dict

def get_first_4999_words(a_sort_dict):
    #获取前4999高频词，存入列表
    list_high_frequence_old=list(a_sort_dict.keys())
    list_high_frequence_new=[]#前4999高频词
    for i in range(4999):
        list_high_frequence_new.append(list_high_frequence_old[i])
    print("get first 4999 words finished\n")
    return list_high_frequence_new

def change_word_to_embedding(path_raw,path_write,list_high_frequence_new):
    with open(path_raw,"r", encoding='ANSI') as f:
        lines=f.readlines()
        embedding_file=open(path_write,mode="w",encoding="ANSI")
        for line in lines:
            #获取line中的每一个词
            line=list(line.split("  "))
            line=[i for i in line if(len(str(i))!=0)]#移除空元素
            line.remove('\n')#移除list中的换行符

            if line.__len__()==0:
                continue
            #写入日期
            time_=line[0]
            for i in time_:
                if (i>="0" and i<='9') or i=='-':
                    embedding_file.write(i)
            embedding_file.write("  ")
            # time=list(time_.split("/"))
            # embedding_file.write(time[0]+"  ")
            
            #获取每行中的词，存入words list中
            line.remove(line[0])
            words=[]
            for word in line:
                #将中文与词性标注分开
                word=list(word.split("/"))

                #删除中文词前的方括号
                if word[0][0]=='[':
                    len_word=word[0].__len__()
                    word[0]=word[0][1:len_word]
                words.append(word[0])

            #查找每一行中是否有高频词
            word_embedding=[]
            cnt=0
            for i in range(4999):
                
                if list_high_frequence_new[i] in words:
                    num=words.count(list_high_frequence_new[i])
                    word_embedding.append(num)
                    cnt+=1
                else:
                    word_embedding.append(0)
                    cnt+=1

            #查找是否含有未登录词,若有，则把最后一维记为1
            for word in words:
                if word not in list_high_frequence_new:
                    word_embedding.append(1)
                    break
            
            embedding_file.write(str(word_embedding))
            embedding_file.write("\n")
            word_embedding.clear()
            words.clear()

        embedding_file.close()
        print("all finish")

if __name__=="__main__":
    path="C:\\Users\\86185\\Desktop\\bit\\大二下\\知识工程\\作业一\\20180712165812468713\\04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330\\"
    path_raw_file=path+"1998-01-2003版-带音.txt"
    path_embedding_file=path+"one_hot.txt"
    
    dicto=high_frequence_word_4999(path_raw_file)
    a_sort_dicto=descending_sort(dicto)
    list_high_frequence_new=get_first_4999_words(a_sort_dicto)
    change_word_to_embedding(path_raw_file,path_embedding_file,list_high_frequence_new)


 

