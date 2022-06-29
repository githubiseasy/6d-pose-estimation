import os

test_result_path = '../experimental_results'
test_list = [_ for _ in os.listdir(test_result_path) if _.endswith('.txt')]
test_list.sort()

report = open('report.txt', 'w')
for i in test_list:
    with open(os.path.join(test_result_path, i)) as result:
        data = result.readline().split(' ')
        pixel_error = data[0]
        projection = data[3]
        iou = data[8]
    report.write('{}\t{}\t{}%\t{}%\n'.format(i[:6], pixel_error, projection, iou))
print('총 {}개 결과 종합 완료.'.format(len(test_list)))