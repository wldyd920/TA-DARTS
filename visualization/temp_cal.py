temp_init = 10
temp_end = 0.1
epoch = 50

def temp_cal(temp_init, temp_end, epoch):
    interval = (temp_init - temp_end) / (epoch-1)
    print("Interval:", interval)
    for i in range(epoch):
        cur_temp = temp_init-(interval*i)
        print(i, cur_temp)

temp_cal(temp_init, temp_end, epoch)
