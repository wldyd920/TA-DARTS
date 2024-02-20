# path = "C:/Users/user/VSC/TD-DARTS/dragen0/alpha/"

def alpha_txt(path):
    f = open(path+"log.txt", 'r')
    # 첫줄 프린트
    # line = f.readline()
    # print(line)

    # # 모든 라인 프린트
    # while True:
    #     line = f.readline()
    #     if not line : break
    #     if line[0] == "E":
    #         E = line[7]+line[8]
    #     if line[-2] == ",":
    #         keep += line.strip()
    #     if line[0] == 't':
    #         keep += line.strip()
    #     if line[0] == " ":
    #         keep += line.strip()
    # print(keep)

    # 리스트로 뱉기
    lines = f.readlines()
    cnt=0
    linenumb = 0
    keep = ''
    keeps = []
    alpha_normals = []
    alpha_reduces = []

    # while True:
    #     line = f.readline()
    #     if not line : break

    for line in lines:

        if line[-2] == ',':
            keep += line.strip()

        if line[0] == ' ':
            if line[-2] == ')':
                keep += line.strip()
                keeps.append(keep[7:-37])
                
                if linenumb == 0 :
                    alpha_normals.append(keep[7:-37])
                    linenumb = 1
                elif linenumb == 1 :
                    alpha_reduces.append(keep[7:-37])
                    linenumb = 0

                keep = ''

    # 추출된 알파
    # print(len(keeps))
    # print(alpha_normals[0])
    # print(alpha_reduces[0])

    # 알파 기록(txt)
    with open(path+'alphas.txt','w') as fi :
        for i in keeps:
            fi.write(i + '\n')

    # # 알파 기록(csv)
    # with open('alphas.csv','w') as fi :
    #     for i in keeps:
    #         fi.write(i + '\n')

    f.close()

