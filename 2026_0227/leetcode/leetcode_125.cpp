#include<iostream>
#include<string>
#include<cctype>
using namespace std;

class Solution {
    public:
        bool isPalindrome(string s) {
            std::string sgood;
            for(auto& c : s) {
                if(isalnum(c)) {
                    sgood += tolower(c);
                }
            }
            int n = sgood.size();
            int left = 0;
            int right = n - 1;
            while( left <= right) {
                if(sgood[left] != sgood[right]) {
                    return false;
                }
                ++left;
                --right;
            }
            return true;
        }
    };

int main() {
    Solution solution;
    std::string s = "A man, a plan, a canal: Panama";
    bool result = solution.isPalindrome(s);
    std::cout << result << std::endl;
    return 0;
}