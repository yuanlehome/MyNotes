#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

// 给定一个有序整数数组和一个目标数，在数组中找到总和等于目标数的两个元素，返回它们的索引
std::pair<int, int> twoSumV1(const std::vector<int>& nums, int target) {
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

// 给定一个无序整数数组和一个目标数，在数组中找到总和等于目标数的两个元素，返回它们的索引
std::pair<int, int> twoSumV2(const std::vector<int>& nums, int target) {
  int left = -1, right = -1, n = nums.size();
  std::unordered_map<int, int> value2index;
  for (int i = 0; i < n; ++i) {
    int value = target - nums[i];
    if (value2index.count(value)) {
      left = value2index.at(value);
      right = i;
      break;
    } else {
      value2index[nums[i]] = i;
    }
  }
  return std::make_pair(left, right);
}

// 给定一个整数数组和一个目标数，返回数组中其总和等于目标数的连续子数组的个数
// 暴力解法
int subarraySumV1(const std::vector<int>& nums, int target) {
  int count = 0, n = nums.size();
  for (int i = 0; i < n; ++i) {
    int sum = 0;
    for (int j = i; j < n; ++j) {
      sum += nums[j];
      count = sum == target ? count + 1 : count;
    }
  }
  return count;
}

// 给定一个整数数组和一个目标数，返回数组中其总和等于目标数的连续子数组的个数
// 前缀和
int subarraySumV2(const std::vector<int>& nums, int target) {
  int count = 0;
  std::unordered_map<int, int> sum2count;
  int sum = 0;
  sum2count[0] = 1;
  for (int num : nums) {
    sum += num;
    int prefix_sum = sum - target;
    if (sum2count.count(prefix_sum)) {
      count += sum2count.at(prefix_sum);
    }
    sum2count[sum]++;
  }
  return count;
}

// 给定固定长度的整数数组arr，请复制每次出现的零，将其余元素向右移动
void duplicateZeros(std::vector<int>& arr) {
  int zerosCount = std::count(arr.begin(), arr.end(), 0);
  if (zerosCount == 0) return;
  int n = arr.size();
  for (int i = n - 1; i >= 0; i--) {
    if (arr[i] == 0) {
      zerosCount--;
      if (zerosCount == 0) break;
    } else {
      int newPos = i + zerosCount;
      if (newPos < n) arr[newPos] = arr[i];
      arr[i] = 0;
    }
  }
}

int main() {
  // auto [i, j] = twoSumV1({1, 4, 5, 7, 9}, 11);
  // auto [i, j] = twoSumV2({1, 7, 5, 6, 9}, 9);
  // std::cout << i << " " << j << std::endl;
  // std::vector<int> arr = {1, 2, 0, 3, 0};
  // duplicateZeros(arr);
  // for (int num : arr) std::cout << num << " ";
  // std::cout << std::endl;
}
