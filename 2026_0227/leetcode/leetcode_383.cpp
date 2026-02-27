#include<iostream>
#include<vector>
using namespace std;
class Solution {
    public:
        bool canConstruct(string ransomNote, string magazine) {
            if (ransomNote.size() > magazine.size()) {
                return false;
            }
            std::vector<int> cnt(26);
            for(auto& c : magazine) {
                cnt[c - 'a']++;
            }
            for(auto& c: ransomNote) {
                cnt[c - 'a']--;
                if (cnt[c - 'a'] < 0) {
                    return false;
                }
            }
            return true;
        }
    };

int main() {
    Solution solution;
    std::string ransomNote = "aa";
    std::string magazine = "baa";
    bool result = solution.canConstruct(ransomNote, magazine);
    std::cout << result << std::endl;
    return 0;
}