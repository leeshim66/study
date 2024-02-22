def solution(storey):
    answer = 0
    while storey > 0:
        storey, moves = divmod(storey, 10)
        if moves > 5 or (moves == 5 and storey % 10 >= 5):
            moves = 10 - moves
            storey += 1
        answer += moves
    return answer

storey = 16
answer = 6