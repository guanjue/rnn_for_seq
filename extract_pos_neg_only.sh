paste test.txt test_y1.txt | awk '{if ($101=='1') print $0}' > test_x_pos.txt

paste test.txt test_y1.txt | awk '{if ($101=='0') print $0}' > test_x_neg.txt





