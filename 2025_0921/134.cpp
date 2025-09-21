#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        int i = 0;
        cout << "\n=== Starting Gas Station Circuit Analysis ===" << endl;
        cout << "Total stations: " << n << endl;
        
        while(i < n) {
            cout << "\n--- Trying to start from station " << i << " ---" << endl;
            int sumOfGas = 0;
            int sumOfCost = 0;
            int cnt = 0;
            
            while(cnt < n) {
                int j = (i + cnt) % n;
                sumOfGas += gas[j];
                sumOfCost += cost[j];
                
                cout << "Step " << cnt + 1 << ": At station " << j 
                     << " -> Gas: +" << gas[j] << " (total: " << sumOfGas 
                     << "), Cost: +" << cost[j] << " (total: " << sumOfCost << ")";
                
                if (sumOfCost > sumOfGas) {
                    cout << " -> FAILED! Not enough gas." << endl;
                    break;
                } else {
                    cout << " -> OK, remaining gas: " << (sumOfGas - sumOfCost) << endl;
                }
                cnt++;
            }
            
            if (cnt == n) {
                cout << "SUCCESS! Completed full circuit from station " << i << endl;
                return i;
            } else {
                cout << "Failed at step " << cnt + 1 << ", skipping to station " << (i + cnt + 1) << endl;
                i = i + cnt + 1;
            }
        }
        cout << "\nNo valid starting station found." << endl;
        return -1;
    }
};

int main() {
    Solution s = Solution();
    std::vector<int> gas = {5,1,2,3,4};
    std::vector<int> cost = {4,4,1,5,1};
    
    cout << "Gas stations: ";
    for(int i = 0; i < gas.size(); i++) {
        cout << gas[i] << " ";
    }
    cout << endl;
    
    cout << "Costs: ";
    for(int i = 0; i < cost.size(); i++) {
        cout << cost[i] << " ";
    }
    cout << endl;
    
    int result = s.canCompleteCircuit(gas, cost);
    cout << "Starting station index: " << result << endl;
    
    if(result != -1) {
        cout << "Can complete circuit starting from station " << result << endl;
    } else {
        cout << "Cannot complete circuit from any starting station" << endl;
    }
    
    return 0;
}