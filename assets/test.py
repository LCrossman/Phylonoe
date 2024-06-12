#!/usr/bin/python

def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        for i, item in enumerate(nums1):
           print(i)
           print("minus", n - i)
           if n - i > 0:
               print("it is", nums2[i], nums1[i+1])
               if nums2[i] < nums1[i+1]:
                   print("nums2[i] is ", nums2[i])
                   print("nums1 i+1 ", nums1[i+1])
                   nums1.insert(nums2[i], nums1[i+1])
                   print("new nums1 ", nums1)
        return nums1


print(merge(nums1=[1,2,3,0,0,0], m=3, nums2=[2,5,6], n=3))
