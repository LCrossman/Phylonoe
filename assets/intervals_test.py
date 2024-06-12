#!/usr/bin/python

i = 0
keeps = []
intervals = [[1,3],[2,6],[8,10],[15,18]]


while i+1 < len(intervals):
     print(intervals[i])
     print(i)
     try:
        if intervals[i+1][1] >= intervals[i][1] and intervals[i+1][0] <= intervals[i][1]:
            print("in overlap try")
            newinterv = [intervals[i][0],intervals[i+1][1]]
            keeps.append(newinterv)
            i+=1
        else:
            print("in else")
            keeps.append(intervals[i])
        i+=1
     except:
       print("except", intervals[i])
       if intervals[-1][0] > intervals[-2][0] and intervals[-1][1] > intervals[-2][1]:
           keeps.append(intervals[-1])


print(keeps)
