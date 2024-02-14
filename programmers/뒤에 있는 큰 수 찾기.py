def solution(numbers):
    n = len(numbers)
    answer = [-1]*n
    stack = []
    for i,num in enumerate(numbers):
        while stack and numbers[stack[-1]] < num:
            answer[stack.pop()] = num
        stack.append(i)
    return answer


numbers = [2,3,3,5]
solution(numbers)

result = [3,5,5,-1]
