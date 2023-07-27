#include <iostream>
#include <utility>
#include <vector>

std::pair<int, int> twoSum(const std::vector<int>& nums, int target) {
  int n = nums.size();
  int left = 0, right = n - 1;
  while (left < right) {
    int sum = nums[left] + nums[right];
    if (sum == target) {
      break;
    } else if (sum < target) {
      left++;
    } else {
      right--;
    }
  }
  if (left >= right) return std::make_pair(-1, -1);
  return std::make_pair(left, right);
}



int main() {
  auto res = twoSum({1, 4, 5, 7, 9}, 11);
  std::cout << res.first << " " << res.second << std::endl;
}
