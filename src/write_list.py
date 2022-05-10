def main():
    str_beg = '/home/diva/John/mia/data/'
    str_end = '_T1.nii.gz'

    i = 0
    with open('list.txt', 'w') as f:
        for i in range(250):
            if i < 10:
                str_complete = str_beg + '00' + str(i) + str_end
            elif i < 100: 
                str_complete = str_beg + '0' + str(i) + str_end
            else:
                str_complete = str_beg + str(i) + str_end

            f.write(str_complete)
            f.write('\n')
    f.close()


if __name__ == '__main__':
    main()