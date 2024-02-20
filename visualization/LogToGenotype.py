path = "C:/Users/user/VSC/TD-DARTS/dragen0/alpha/"

def L2G(path):
    f = open(path+"log.txt", 'r')

    # 리스트로 뱉기
    lines = f.readlines()
    Genotypes = []


    for line in lines:
        if len(line) > 33:
            if line[34] == 'G':
                Genotypes.append(line[34:])


    # 알파 기록(txt)
    with open(path+'Log2Genotypes.txt','w') as fi :
        for i, x in enumerate(Genotypes):
            if i==0 or (i+1)%10==0:
                fi.write(f'TADARTS50_epoch{str(i)} = ' + x)

    # # 알파 기록(csv)
    # with open('alphas.csv','w') as fi :
    #     for i in keeps:
    #         fi.write(i + '\n')

    f.close()

L2G(path)