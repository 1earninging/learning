#include<vector>
#include<iostream>
using namespace std;

class Solution {
    public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        std::vector<int> vec = nums1;
        int i = 0;
        int j = 0;
        while(i < m && j < n) {
            if (vec[i] < nums2[j]) {
                nums1[i + j] = vec[i];
                i++; 
            } else {
                nums1[i + j] = nums2[j];
                j++;
            }
        }
        if (i < m) {
            for (int k = i; k < m; k++) {
                nums1[k + j] = vec[k];
            }
        }
        if (j < n) {
            for (int k = j; k < n; k++) {
                nums1[i + k] = nums2[k];
            }
        }
    }
};

int main() {
    std::vector<int> nums1 = {1, 2, 3, 0, 0, 0};
    std::vector<int> nums2 = {2, 5, 6};
    Solution().merge(nums1, 3, nums2, 3);
    for (int i = 0; i < nums1.size(); i++) {
        std::cout << nums1[i] << " ";
    }
    return 0;
}